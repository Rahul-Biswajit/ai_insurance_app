"""
backend/vector_store.py — Policy catalog + semantic retrieval (RAG).

Contains the 12-product mock catalog and two interchangeable retrievers:

* :class:`ChromaVectorStore`   — ChromaDB with its default embedding function.
* :class:`InMemoryVectorStore` — a dependency-free TF-IDF cosine index with
  insurance-domain synonym expansion. Deterministic and always available.

``get_vector_store()`` picks one according to ``settings.vector_backend``
(``auto`` | ``chroma`` | ``memory``).

Each policy carries the four static radar dimensions used by the dashboard
(affordability, coverage breadth, flexibility, perks); the fifth axis — risk
match — is computed per applicant in ``agents/recommendation_agent.py``.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Protocol

from pydantic import BaseModel, ConfigDict, Field

from config import settings

logger = logging.getLogger(__name__)

__all__ = [
    "Policy",
    "CoverageAllocation",
    "POLICY_CATALOG",
    "get_policy",
    "get_vector_store",
    "InMemoryVectorStore",
    "ChromaVectorStore",
    "RetrievalHit",
]


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
class CoverageAllocation(BaseModel):
    """Share of total coverage value by component (percentages sum to 100)."""

    model_config = ConfigDict(frozen=True)

    liability: int
    collision: int
    comprehensive: int
    medical: int
    roadside: int

    def as_pairs(self) -> list[tuple[str, int]]:
        return [
            ("Liability", self.liability),
            ("Collision", self.collision),
            ("Comprehensive", self.comprehensive),
            ("Medical", self.medical),
            ("Roadside", self.roadside),
        ]


class Policy(BaseModel):
    """A single catalog product."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    insurer: str
    category: str
    tagline: str

    # pricing
    base_monthly: float = Field(gt=0)
    default_deductible: int
    coverage_limit: int

    # radar dimensions (0-100, static product attributes)
    affordability: int = Field(ge=0, le=100)
    coverage_breadth: int = Field(ge=0, le=100)
    flexibility: int = Field(ge=0, le=100)
    perks: int = Field(ge=0, le=100)

    # underwriting fit
    ideal_tiers: tuple[str, ...]
    min_age: int = 18
    max_age: int = 100
    max_annual_mileage: int = 100_000
    max_vehicle_age: int = 40

    allocation: CoverageAllocation
    features: tuple[str, ...]
    exclusions: tuple[str, ...]
    add_ons: tuple[str, ...] = ()

    @property
    def document(self) -> str:
        """The text embedded into the vector index and used as RAG evidence."""
        return (
            f"{self.name} by {self.insurer}. Category: {self.category}. {self.tagline}\n"
            f"Base premium: ${self.base_monthly:.2f} per month "
            f"(${self.base_monthly * 12:,.2f} per year). "
            f"Standard deductible: ${self.default_deductible:,}. "
            f"Total coverage limit: ${self.coverage_limit:,}.\n"
            f"Coverage allocation — liability {self.allocation.liability}%, "
            f"collision {self.allocation.collision}%, "
            f"comprehensive {self.allocation.comprehensive}%, "
            f"medical {self.allocation.medical}%, "
            f"roadside {self.allocation.roadside}%.\n"
            f"Eligibility: ages {self.min_age} to {self.max_age}, "
            f"up to {self.max_annual_mileage:,} miles per year, "
            f"vehicles up to {self.max_vehicle_age} years old. "
            f"Best suited to {', '.join(self.ideal_tiers)} risk profiles.\n"
            f"Included features: {'; '.join(self.features)}.\n"
            f"Optional add-ons: {'; '.join(self.add_ons) if self.add_ons else 'none'}.\n"
            f"Exclusions: {'; '.join(self.exclusions)}.\n"
            f"Scores — affordability {self.affordability}/100, "
            f"coverage breadth {self.coverage_breadth}/100, "
            f"flexibility {self.flexibility}/100, perks {self.perks}/100."
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "insurer": self.insurer,
            "category": self.category,
            "base_monthly": self.base_monthly,
            "coverage_limit": self.coverage_limit,
            "ideal_tiers": ",".join(self.ideal_tiers),
        }


@dataclass
class RetrievalHit:
    """A retrieved policy with its similarity score."""

    policy: Policy
    score: float
    rank: int = 0


# --------------------------------------------------------------------------- #
# The 12-product catalog
# --------------------------------------------------------------------------- #
def _alloc(l: int, c: int, comp: int, m: int, r: int) -> CoverageAllocation:
    return CoverageAllocation(liability=l, collision=c, comprehensive=comp, medical=m, roadside=r)


POLICY_CATALOG: tuple[Policy, ...] = (
    Policy(
        id="AEG-ESS-01",
        name="Essential Drive",
        insurer="Aegis Mutual",
        category="Budget",
        tagline="State-minimum liability protection at the lowest defensible price.",
        base_monthly=58.0,
        default_deductible=2000,
        coverage_limit=50_000,
        affordability=98,
        coverage_breadth=32,
        flexibility=45,
        perks=15,
        ideal_tiers=("Low Risk", "Preferred", "Standard"),
        max_annual_mileage=12_000,
        max_vehicle_age=25,
        allocation=_alloc(70, 10, 8, 10, 2),
        features=(
            "Bodily injury and property damage liability",
            "Uninsured motorist bodily injury",
            "Paperless billing discount of 3%",
        ),
        exclusions=(
            "No collision cover for at-fault accidents",
            "No comprehensive cover for theft, fire or flood",
            "No rental reimbursement",
        ),
        add_ons=("Roadside assistance rider (+$6/mo)",),
    ),
    Policy(
        id="AEG-BAL-02",
        name="Balanced Shield",
        insurer="Aegis Mutual",
        category="Standard",
        tagline="The everyday all-rounder: full liability, collision and comprehensive.",
        base_monthly=112.0,
        default_deductible=1000,
        coverage_limit=250_000,
        affordability=74,
        coverage_breadth=72,
        flexibility=68,
        perks=52,
        ideal_tiers=("Low Risk", "Preferred", "Standard"),
        max_annual_mileage=20_000,
        allocation=_alloc(40, 24, 20, 12, 4),
        features=(
            "Collision and comprehensive cover",
            "24/7 roadside assistance",
            "Rental car reimbursement up to $40/day for 30 days",
            "Accident forgiveness after 3 claim-free years",
        ),
        exclusions=(
            "Racing and track-day use",
            "Commercial ride-hailing use",
            "Wear-and-tear or mechanical breakdown",
        ),
        add_ons=("Gap cover (+$9/mo)", "New-car replacement (+$14/mo)"),
    ),
    Policy(
        id="AEG-PRM-03",
        name="Guardian Shield Premier",
        insurer="Aegis Assurance",
        category="Premium",
        tagline="Maximum protection with concierge claims and OEM parts guarantee.",
        base_monthly=198.0,
        default_deductible=500,
        coverage_limit=1_000_000,
        affordability=38,
        coverage_breadth=97,
        flexibility=82,
        perks=95,
        ideal_tiers=("Low Risk", "Preferred"),
        max_annual_mileage=25_000,
        max_vehicle_age=15,
        allocation=_alloc(32, 24, 22, 16, 6),
        features=(
            "$1,000,000 combined single limit",
            "OEM replacement parts guarantee",
            "Diminishing deductible: $100 off per claim-free year",
            "Concierge claims handling with a named adjuster",
            "Worldwide rental cover including international trips",
            "Key replacement and lock-out service",
        ),
        exclusions=(
            "Intentional damage",
            "Use in commission of a felony",
            "Undeclared modifications above $5,000 in value",
        ),
        add_ons=("Agreed-value endorsement (+$18/mo)", "Track-day rider (+$45/mo)"),
    ),
    Policy(
        id="AEG-YNG-04",
        name="FirstMile Young Driver",
        insurer="Aegis Mutual",
        category="Young Driver",
        tagline="Built for drivers under 25, with telematics rewards that cut premiums fast.",
        base_monthly=148.0,
        default_deductible=1000,
        coverage_limit=150_000,
        affordability=62,
        coverage_breadth=58,
        flexibility=88,
        perks=64,
        ideal_tiers=("Standard", "Elevated Risk", "High Risk"),
        min_age=18,
        max_age=27,
        max_annual_mileage=18_000,
        allocation=_alloc(45, 22, 15, 15, 3),
        features=(
            "Telematics safe-driving discount up to 30%",
            "Good-student discount of 12%",
            "Driver-training course credit of 8%",
            "No young-driver surcharge after 24 claim-free months",
        ),
        exclusions=(
            "Drivers with a DUI conviction in the last 5 years",
            "Vehicles with a performance rating above 400 hp",
        ),
        add_ons=("Parental co-pilot monitoring (free)",),
    ),
    Policy(
        id="AEG-SEN-05",
        name="Silver Journey",
        insurer="Aegis Assurance",
        category="Senior",
        tagline="Low-mileage value for experienced drivers aged 55 and over.",
        base_monthly=86.0,
        default_deductible=500,
        coverage_limit=300_000,
        affordability=84,
        coverage_breadth=76,
        flexibility=58,
        perks=72,
        ideal_tiers=("Low Risk", "Preferred"),
        min_age=55,
        max_annual_mileage=9_000,
        allocation=_alloc(38, 20, 22, 17, 3),
        features=(
            "Mature-driver discount of 15%",
            "Low-mileage credit below 8,000 miles per year",
            "Enhanced medical payments cover of $25,000",
            "Defensive-driving refresher credit",
            "Vanishing deductible after 2 claim-free years",
        ),
        exclusions=("Drivers under 55", "Annual mileage above 9,000"),
        add_ons=("Medical transport rider (+$7/mo)",),
    ),
    Policy(
        id="AEG-COM-06",
        name="Commuter Pro",
        insurer="Aegis Mutual",
        category="High Mileage",
        tagline="Unlimited-mileage cover for long-distance and highway commuters.",
        base_monthly=164.0,
        default_deductible=1000,
        coverage_limit=400_000,
        affordability=56,
        coverage_breadth=80,
        flexibility=74,
        perks=58,
        ideal_tiers=("Preferred", "Standard", "Elevated Risk"),
        max_annual_mileage=100_000,
        allocation=_alloc(38, 26, 18, 14, 4),
        features=(
            "No mileage cap or high-mileage surcharge",
            "Highway breakdown priority response within 45 minutes",
            "Tyre and windscreen cover with zero deductible",
            "Fatigue-related incident cover",
            "Loaner vehicle within 4 hours",
        ),
        exclusions=("Ride-hailing or delivery use", "Vehicles over 15 years old"),
        add_ons=("Cross-border cover (+$12/mo)",),
    ),
    Policy(
        id="AEG-EV-07",
        name="Voltguard EV",
        insurer="Aegis Innovate",
        category="Electric Vehicle",
        tagline="Purpose-built for EVs: battery, charger and charging-cable protection.",
        base_monthly=134.0,
        default_deductible=750,
        coverage_limit=500_000,
        affordability=66,
        coverage_breadth=86,
        flexibility=78,
        perks=84,
        ideal_tiers=("Low Risk", "Preferred", "Standard"),
        max_annual_mileage=22_000,
        max_vehicle_age=10,
        allocation=_alloc(34, 24, 24, 14, 4),
        features=(
            "Traction battery cover including degradation from a covered event",
            "Home wall-charger and cable cover up to $3,000",
            "Out-of-charge recovery tow to the nearest charger",
            "Specialist EV repair network with certified technicians",
            "Green parts credit of 5%",
        ),
        exclusions=(
            "Battery degradation from normal use",
            "Non-approved third-party charger damage",
            "Internal combustion vehicles",
        ),
        add_ons=("Public charging liability (+$5/mo)",),
    ),
    Policy(
        id="AEG-FAM-08",
        name="Family Fleet Guard",
        insurer="Aegis Assurance",
        category="Family",
        tagline="Multi-vehicle, multi-driver household cover under one renewal date.",
        base_monthly=176.0,
        default_deductible=750,
        coverage_limit=600_000,
        affordability=58,
        coverage_breadth=90,
        flexibility=86,
        perks=80,
        ideal_tiers=("Low Risk", "Preferred", "Standard"),
        max_annual_mileage=30_000,
        allocation=_alloc(36, 22, 20, 18, 4),
        features=(
            "Covers up to 4 vehicles and 6 named drivers",
            "Multi-car discount of 22%",
            "Child car-seat replacement after any covered accident",
            "Newly licensed household driver added free for 90 days",
            "Single household deductible per incident",
        ),
        exclusions=("Drivers not listed on the schedule", "Commercial use of any listed vehicle"),
        add_ons=("Teen driver telematics (free)", "Second-home garage cover (+$11/mo)"),
    ),
    Policy(
        id="AEG-RID-09",
        name="Rideshare Flex",
        insurer="Aegis Innovate",
        category="Commercial Hybrid",
        tagline="Seamless personal-to-commercial cover for rideshare and delivery drivers.",
        base_monthly=212.0,
        default_deductible=1000,
        coverage_limit=750_000,
        affordability=42,
        coverage_breadth=88,
        flexibility=94,
        perks=62,
        ideal_tiers=("Standard", "Elevated Risk"),
        max_annual_mileage=100_000,
        allocation=_alloc(44, 22, 16, 14, 4),
        features=(
            "Period 1/2/3 rideshare gap cover",
            "Food and parcel delivery endorsement included",
            "Passenger medical cover of $50,000 per person",
            "Downtime income protection of $60/day for up to 21 days",
            "Deductible waived when the platform's cover applies",
        ),
        exclusions=("Long-haul freight", "Vehicles over 10 years old", "Unlisted commercial drivers"),
        add_ons=("Airport queue cover (+$8/mo)",),
    ),
    Policy(
        id="AEG-CLS-10",
        name="Heritage Classic",
        insurer="Aegis Assurance",
        category="Classic",
        tagline="Agreed-value protection for collector and vintage vehicles.",
        base_monthly=94.0,
        default_deductible=1000,
        coverage_limit=350_000,
        affordability=78,
        coverage_breadth=64,
        flexibility=52,
        perks=88,
        ideal_tiers=("Low Risk", "Preferred"),
        min_age=25,
        max_annual_mileage=5_000,
        max_vehicle_age=80,
        allocation=_alloc(34, 18, 34, 10, 4),
        features=(
            "Agreed-value settlement with no depreciation",
            "Spare-parts cover up to $15,000",
            "Show and exhibition transit cover",
            "Specialist restoration workshop network",
            "Inflation-guard valuation review each year",
        ),
        exclusions=(
            "Daily commuting use",
            "Annual mileage above 5,000",
            "Drivers under 25",
            "Unsecured overnight parking",
        ),
        add_ons=("Rally and tour cover (+$22/mo)",),
    ),
    Policy(
        id="AEG-USG-11",
        name="PayPerMile Telematics",
        insurer="Aegis Innovate",
        category="Usage Based",
        tagline="A low base rate plus a per-mile charge — ideal below 7,000 miles a year.",
        base_monthly=44.0,
        default_deductible=1000,
        coverage_limit=200_000,
        affordability=92,
        coverage_breadth=66,
        flexibility=96,
        perks=48,
        ideal_tiers=("Low Risk", "Preferred", "Standard"),
        max_annual_mileage=8_000,
        allocation=_alloc(42, 22, 18, 14, 4),
        features=(
            "Base rate of $44/month plus $0.061 per mile driven",
            "Real-time trip scoring in the mobile app",
            "Safe-driving cashback of up to 25% each quarter",
            "Cover pauses automatically while the vehicle is parked long term",
            "No charge for miles driven as a passenger",
        ),
        exclusions=(
            "Annual mileage above 8,000 without re-rating",
            "Vehicles without a compatible OBD-II port",
            "Telematics device removal voids the discount",
        ),
        add_ons=("Family trip-sharing (free)",),
    ),
    Policy(
        id="AEG-ELT-12",
        name="Aegis Elite Total Care",
        insurer="Aegis Assurance",
        category="Elite",
        tagline="Our highest limits, zero-deductible glass, and a dedicated risk manager.",
        base_monthly=289.0,
        default_deductible=250,
        coverage_limit=2_000_000,
        affordability=22,
        coverage_breadth=100,
        flexibility=90,
        perks=100,
        ideal_tiers=("Low Risk", "Preferred"),
        min_age=30,
        max_annual_mileage=30_000,
        max_vehicle_age=8,
        allocation=_alloc(30, 24, 22, 18, 6),
        features=(
            "$2,000,000 combined single limit",
            "Zero-deductible glass and windscreen cover",
            "Dedicated personal risk manager",
            "Worldwide cover including hired vehicles abroad",
            "Guaranteed OEM parts and manufacturer-approved bodyshops",
            "Identity-theft restoration service",
            "Chauffeur service while a claim is open",
        ),
        exclusions=(
            "Drivers under 30",
            "Vehicles over 8 years old",
            "Undisclosed performance modifications",
        ),
        add_ons=("Fine-art in transit (+$26/mo)", "Executive kidnap and ransom (+$60/mo)"),
    ),
)

_BY_ID: dict[str, Policy] = {p.id: p for p in POLICY_CATALOG}


def get_policy(policy_id: str) -> Policy | None:
    return _BY_ID.get(policy_id)


# --------------------------------------------------------------------------- #
# Text processing for the in-memory index
# --------------------------------------------------------------------------- #
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have i in is it its of on or that the to was were
    will with you your my me we our they their this these those there here what which who whom
    how when where why do does did but if then than so such can could should would may might must
    not no nor own same too very just also about into over under again further once""".split()
)

_SYNONYMS: dict[str, tuple[str, ...]] = {
    "cheap": ("affordable", "budget", "low", "inexpensive", "economical", "value"),
    "affordable": ("cheap", "budget", "low", "value"),
    "expensive": ("premium", "elite", "high"),
    "best": ("premium", "elite", "comprehensive", "top"),
    "young": ("under", "student", "teen", "firstmile", "telematics"),
    "old": ("senior", "mature", "silver", "classic", "vintage"),
    "senior": ("mature", "silver", "older"),
    "electric": ("ev", "battery", "charger", "charging", "voltguard"),
    "ev": ("electric", "battery", "charger", "voltguard"),
    "family": ("household", "multi", "children", "kids", "fleet"),
    "uber": ("rideshare", "delivery", "commercial", "lyft"),
    "lyft": ("rideshare", "delivery", "commercial", "uber"),
    "rideshare": ("commercial", "delivery", "uber", "lyft"),
    "commute": ("commuter", "highway", "mileage", "long"),
    "commuting": ("commuter", "highway", "mileage", "long"),
    "vintage": ("classic", "collector", "heritage", "agreed"),
    "classic": ("vintage", "collector", "heritage"),
    "theft": ("comprehensive", "stolen"),
    "flood": ("comprehensive", "weather", "catastrophe"),
    "accident": ("collision", "crash", "claim"),
    "crash": ("collision", "accident"),
    "towing": ("roadside", "breakdown", "recovery"),
    "tow": ("roadside", "breakdown", "recovery"),
    "medical": ("injury", "bodily", "health"),
    "coverage": ("cover", "protection", "limit"),
    "cover": ("coverage", "protection"),
    "protection": ("coverage", "cover", "shield"),
    "low": ("minimum", "budget", "affordable"),
    "mileage": ("miles", "usage", "distance"),
    "miles": ("mileage", "usage", "distance"),
    "flexible": ("flexibility", "customisable", "adjustable", "usage"),
    "perks": ("benefits", "rewards", "extras", "concierge"),
}


def _tokenize(text: str) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", (text or "").lower())
    tokens: list[str] = []
    for token in raw:
        if token in _STOPWORDS or len(token) < 2:
            continue
        # Crude singularisation keeps "policies"/"policy" and "miles"/"mile" aligned.
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


def _expand(tokens: Iterable[str]) -> list[str]:
    """Query-time synonym expansion (half weight via single duplication)."""
    out = list(tokens)
    for token in list(tokens):
        out.extend(_SYNONYMS.get(token, ()))
    return out


# --------------------------------------------------------------------------- #
# Retriever protocol
# --------------------------------------------------------------------------- #
class VectorStore(Protocol):
    backend: str

    def query(self, text: str, k: int = 5) -> list[RetrievalHit]: ...
    def documents_for(self, policy_ids: Iterable[str]) -> str: ...


class _BaseStore:
    backend = "base"

    def documents_for(self, policy_ids: Iterable[str]) -> str:
        """Concatenated evidence block for grounding checks and Q&A prompts."""
        chunks = []
        for pid in policy_ids:
            policy = _BY_ID.get(pid)
            if policy:
                chunks.append(f"### {policy.name} ({policy.id})\n{policy.document}")
        return "\n\n".join(chunks)


class InMemoryVectorStore(_BaseStore):
    """
    Deterministic TF-IDF cosine index over the catalog.

    Chosen over a random-projection stand-in because TF-IDF gives genuinely
    useful lexical retrieval for a 12-document corpus, needs no model download,
    and returns identical results on every machine.
    """

    backend = "in-memory-tfidf"

    def __init__(self, policies: tuple[Policy, ...] = POLICY_CATALOG) -> None:
        self.policies = policies
        self._doc_tokens = [_tokenize(p.document) for p in policies]

        doc_count = len(policies)
        doc_freq: Counter[str] = Counter()
        for tokens in self._doc_tokens:
            doc_freq.update(set(tokens))

        # Smoothed IDF keeps every term informative in a small corpus.
        self._idf = {
            term: math.log((doc_count + 1) / (freq + 1)) + 1.0 for term, freq in doc_freq.items()
        }
        self._vectors = [self._vectorize(tokens) for tokens in self._doc_tokens]

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = sum(counts.values())
        vector = {
            term: (count / total) * self._idf.get(term, 1.0) for term, count in counts.items()
        }
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {term: value / norm for term, value in vector.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(value * b.get(term, 0.0) for term, value in a.items())

    def query(self, text: str, k: int = 5) -> list[RetrievalHit]:
        query_vector = self._vectorize(_expand(_tokenize(text)))
        if not query_vector:
            return [RetrievalHit(policy=p, score=0.0, rank=i) for i, p in enumerate(self.policies[:k])]

        scored = [
            (self._cosine(query_vector, vector), policy)
            for vector, policy in zip(self._vectors, self.policies)
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [
            RetrievalHit(policy=policy, score=round(score, 4), rank=rank)
            for rank, (score, policy) in enumerate(scored[:k])
        ]


class ChromaVectorStore(_BaseStore):
    """ChromaDB-backed retriever using Chroma's default embedding function."""

    backend = "chromadb"

    def __init__(self, collection: Any) -> None:
        self.collection = collection

    @classmethod
    def try_build(cls) -> "ChromaVectorStore | None":
        try:
            import chromadb  # type: ignore
            from chromadb.config import Settings as ChromaSettings  # type: ignore
        except Exception as exc:
            logger.info("chromadb unavailable (%s) — using in-memory index.", exc)
            return None

        try:
            settings.vector_persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(settings.vector_persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            collection = client.get_or_create_collection(
                name=settings.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            if collection.count() < len(POLICY_CATALOG):
                collection.upsert(
                    ids=[p.id for p in POLICY_CATALOG],
                    documents=[p.document for p in POLICY_CATALOG],
                    metadatas=[p.metadata() for p in POLICY_CATALOG],
                )
            logger.info("ChromaDB collection ready (%d docs).", collection.count())
            return cls(collection)
        except Exception as exc:
            logger.warning("ChromaDB initialisation failed: %s", exc)
            return None

    def query(self, text: str, k: int = 5) -> list[RetrievalHit]:
        try:
            result = self.collection.query(query_texts=[text or ""], n_results=k)
        except Exception as exc:
            logger.warning("Chroma query failed (%s) — falling back.", exc)
            return _fallback_store().query(text, k)

        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0] or [0.0] * len(ids)

        hits: list[RetrievalHit] = []
        for rank, (pid, distance) in enumerate(zip(ids, distances)):
            policy = _BY_ID.get(pid)
            if policy:
                # Cosine distance -> similarity.
                hits.append(RetrievalHit(policy=policy, score=round(1.0 - float(distance), 4), rank=rank))
        return hits


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _fallback_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Return the process-wide retriever, honouring ``settings.vector_backend``."""
    global _store
    if _store is not None:
        return _store

    backend = (settings.vector_backend or "auto").lower()

    if backend in {"auto", "chroma"}:
        chroma = ChromaVectorStore.try_build()
        if chroma is not None:
            _store = chroma
            return _store
        if backend == "chroma":
            raise RuntimeError(
                "VECTOR_BACKEND=chroma but ChromaDB could not be initialised. "
                "Install it with `pip install chromadb`, or set VECTOR_BACKEND=memory."
            )

    _store = _fallback_store()
    return _store


def reset_vector_store() -> None:
    """Drop the cached retriever (used when the backend setting changes)."""
    global _store
    _store = None
    _fallback_store.cache_clear()
