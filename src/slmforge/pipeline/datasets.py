"""Stage 2 — Dataset Curation.

Turns a reviewed `MLSpec` into real, trustworthy train/validation/test files:
discovery first, then a *materialisation* step that only runs with approval.

Two modes, one stage
--------------------
The stage is run twice, and which mode it runs in is decided entirely by
`ctx.params`, so the same code path serves the API, the CLI and the worker:

- **plan** (default) — derive search terms from the spec, score candidates with a
  transparent rubric, profile the recommended one *remotely* with DuckDB (no
  download), and return a `DatasetPlan` for review. It persists on the project as
  `data_plan`, reports `needs_approval=True`, and therefore does **not** advance
  the project to `data_ready`. Nothing has been committed yet.
- **build** (`params={"build": true}`) — materialise the selected sources: one
  remote read that writes only the rows we keep, then normalise, map labels,
  deduplicate, split with the configured seed, and write JSONL + a `DatasetCard`.

Approval is a parameter, not a flag on the class, because the expensive half of
this stage is I/O rather than compute. The API refuses to run a build
synchronously (see `api/routes/stages.py`); it goes through the `dataset_build`
job so a slow download cannot occupy the request thread.

Parameters (all optional, documented because they are the stage's public API)
---------------------------------------------------------------------------

``build``
    Run the materialisation step instead of the discovery step.
``selected``
    Candidate ids to build from, e.g. ``["hf:Tobi-Bueck/..."]``. Defaults to the
    plan's recommended candidates. ``upload:<path>`` ids are accepted and are the
    user-upload route (JSONL/CSV, treated as untrusted input).
``plan``
    A user-edited `DatasetPlan` (as a dict). Reviewed plans win over re-discovery,
    so an approval is never silently replaced by a fresh opinion.
``columns``
    ``{"text": ["subject", "body"], "label": "queue"}`` — required for a candidate
    that is not in the curated catalog, because only the catalog knows a dataset's
    column names. Without it the candidate is reported as unbuildable rather than
    guessed at.
``target_examples``
    The number of examples the build will deliver **at most** (default 4000),
    allocated proportionally across labels. It is an approval boundary rather than a
    suggestion: the fetch is capped above it so the class balance is not decided by
    parquet's row order, and the build is trimmed down to it, because "how much
    training was approved" is a number the user is entitled to have honoured.
``offline``
    Skip Hub search and remote profiling; the catalog is still offered from its
    recorded facts, clearly marked as unverified.

What the stage refuses to do
----------------------------
- It never downloads a dataset that has not been selected, and never more columns
  than it needs (`allow_patterns` equivalent: a projection, not a full read).
- It never maps a raw label onto a category it was not told about, and never
  invents examples to fill a class. Unmapped rows are dropped *with a reason and a
  count*, and an empty class is reported as an error-severity quality issue rather
  than hidden by an unrelated label.
- It never counts synthetic rows as ground truth: the synthetic fraction is
  recorded on the card, per source.

Exit criteria: split files plus a `DatasetCard` whose sources, licences,
synthetic fraction and quality issues are all recorded, and a sufficiency verdict
that says "ask for data" when the sources cannot support the task.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..integrations import hf_cli, parquet
from ..log import get_logger
from ..models import StageName
from ..schemas import (
    DataQualityIssue,
    DatasetCandidate,
    DatasetCard,
    DatasetPlan,
    LabelMapping,
    License,
    MLSpec,
    OutputFormat,
    Provenance,
    SplitSizes,
    TaskType,
)
from .base import Stage, StageContext, StageResult

logger = get_logger("pipeline.datasets")

#: Tasks this stage can build data for. Anything else is refused, not guessed.
CURATABLE_TASKS = {TaskType.classification, TaskType.structured_generation, TaskType.extraction}

DEFAULT_TARGET_EXAMPLES = 4000
#: Fetch this much more than we need, so stratified sampling is not at the mercy
#: of whatever order parquet happened to store. A second of remote filtering is
#: cheaper than a skewed dataset.
FETCH_MULTIPLIER = 4
#: Below this the text carries no signal worth training on.
MIN_TEXT_CHARS = 8
#: Class-count ratio above which the split is called out as imbalanced.
IMBALANCE_RATIO = 5.0
#: Near-duplicate detection: Jaccard over word shingles.
SHINGLE_SIZE = 4
NEAR_DUP_THRESHOLD = 0.85
#: Leakage checking is quadratic-ish; sample deterministically beyond this.
MAX_LEAKAGE_CHECK = 20_000
#: Uploads are untrusted: cap what we are willing to parse in-process.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024

PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("phone", re.compile(r"\b(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]?\d{3}[ -]?\d{4}\b")),
    ("card_number", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
    ("national_id", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
)

#: The curated catalog. Entries are datasets a human verified by hand — including
#: the exact facts quoted here — and are the only sources buildable without an
#: explicit column mapping, because only the catalog knows a dataset's columns.
#:
#: Every entry MUST record what was actually observed, plus the date-sensitive
#: facts (row counts, licence) as verified values, so an offline plan is still
#: honest about where its numbers came from.
CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "hf:Tobi-Bueck/customer-support-tickets",
        "repo_id": "Tobi-Bueck/customer-support-tickets",
        "name": "IT & customer support tickets (synthetic generator)",
        "rows": 61765,
        "columns": [
            "subject",
            "body",
            "answer",
            "type",
            "queue",
            "priority",
            "language",
            "version",
            *(f"tag_{n}" for n in range(1, 9)),
        ],
        "text_columns": ["subject", "body"],
        "label_column": "queue",
        "license": "cc-by-nc-4.0",
        "commercial_ok": False,
        "license_note": (
            "cc-by-nc-4.0 — non-commercial. Fine for this demo, but a model trained "
            "on it cannot be offered commercially without a licence or a different source."
        ),
        "synthetic": True,
        "keywords": [
            "ticket",
            "tickets",
            "support",
            "customer",
            "helpdesk",
            "service desk",
            "routing",
            "classification",
            "it",
        ],
        "task_types": ["classification"],
        # Verified live (2026-10): raw `queue` values -> the spec's label set.
        # Anything not listed is dropped, deliberately and visibly.
        "label_map": {
            "Billing": ["billing and payments"],
            "Technical": [
                "technical support",
                "it support",
                "product support",
                "service outages and maintenance",
                "it & technology/security operations",
            ],
            "Account Access": [],
            "General": [
                "general inquiry",
                "customer service",
                "sales and pre-sales",
                "returns and exchanges",
                "human resources",
            ],
        },
        # Verified live (2026-10): counts after applying label_map above. Recorded
        # so a plan still reports the real class balance when the Hub is unreachable.
        "verified_mapped_counts": {
            "Billing": 4874,
            "Technical": 31148,
            "Account Access": 0,
            "General": 12930,
        },
        "verified_dropped": 12813,
        "notes": (
            "Synthetic IT ticket generator (per the dataset card: no PII). "
            "Verified live: 61,765 rows, one parquet file, columns subject/body/"
            "answer/type/queue/priority/language/version/tag_1..tag_8. The `queue` "
            "column holds 52 values, most of them generic content categories "
            "(News, Sports, Real Estate) that are not support-routing labels."
        ),
    },
)

#: Dataset card blurb for a source we could not profile; never guess a number.
_UNVERIFIED = "Not profiled — open the dataset on the Hub before trusting it."


# --- small helpers -----------------------------------------------------


def normalise_key(value: str | None) -> str:
    """Lower-case, punctuation-free form used for label and duplicate matching."""
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def normalise_text(value: str) -> str:
    """Collapse whitespace so trivial formatting differences are not unique rows."""
    return re.sub(r"\s+", " ", value).strip()


def text_hash(value: str) -> str:
    """Stable hash of normalised text, used for exact-duplicate detection."""
    return hashlib.sha256(normalise_text(value).lower().encode("utf-8")).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    """Content hash of a written artifact, recorded on the card and the Artifact row."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def suspected_pii(text: str) -> str | None:
    """Name of the first PII pattern that matches, or None.

    We report the *kind* and a count, never the matched value: the point is to warn
    the user, not to copy personal data into an artifact.
    """
    for name, pattern in PII_PATTERNS:
        if pattern.search(text):
            return name
    return None


def shingles(text: str) -> frozenset[str]:
    """Word-level shingles for near-duplicate comparison."""
    words = normalise_key(text).split()
    if len(words) < SHINGLE_SIZE:
        return frozenset(words)
    return frozenset(
        " ".join(words[i : i + SHINGLE_SIZE]) for i in range(len(words) - SHINGLE_SIZE + 1)
    )


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def build_system_prompt(spec: MLSpec) -> str:
    """The instruction every training example carries.

    Kept in one place so the same string appears in the data, the baseline prompt
    and the training script — a mismatch there quietly invalidates comparisons.
    """
    labels = ", ".join(spec.allowed_labels) or "the requested categories"
    verb = "Classify" if spec.task_type is TaskType.classification else "Process"
    if spec.output.format is OutputFormat.json_object:
        return (
            f"{verb} the input below. Reply with a JSON object containing exactly "
            f'the key "label". Allowed values: {labels}.'
        )
    return f"{verb} the input below. Reply with exactly one of: {labels}."


def _license_for(entry: dict[str, Any]) -> License:
    licence_id = entry.get("license") or "unknown"
    return License(
        id=licence_id,
        commercial_ok=entry.get("commercial_ok"),
        unknown=licence_id == "unknown",
        notes=entry.get("license_note"),
    )


def _provenance(entry: dict[str, Any], *, transformations: list[str]) -> Provenance:
    return Provenance(
        source="hf_dataset" if entry.get("repo_id") else str(entry.get("source", "catalog")),
        uri=entry.get("uri")
        or (
            f"https://huggingface.co/datasets/{entry['repo_id']}" if entry.get("repo_id") else None
        ),
        license=_license_for(entry),
        transformations=transformations,
        synthetic=bool(entry.get("synthetic")),
        synthetic_generator="dataset publisher" if entry.get("synthetic") else None,
        notes=entry.get("notes"),
    )


# --- discovery ---------------------------------------------------------


def discovery_queries(spec: MLSpec, *, limit: int = 3) -> list[str]:
    """Search terms derived from the spec, most specific first."""
    terms: list[str] = []
    domain = normalise_key(spec.domain)
    title = normalise_key(spec.title)
    if domain:
        terms.append(f"{domain} {spec.task_type.value.replace('_', ' ')}")
    if title:
        terms.append(title)
    # The label names themselves are often the best query ("support tickets").
    for label in spec.allowed_labels:
        cleaned = normalise_key(label)
        if cleaned and cleaned not in terms:
            terms.append(cleaned)
    if not terms:
        terms.append(spec.task_type.value.replace("_", " "))
    deduped: list[str] = []
    for term in terms:
        if term not in deduped:
            deduped.append(term)
    return deduped[:limit]


def score_candidate(
    entry: dict[str, Any],
    spec: MLSpec,
    *,
    label_coverage: dict[str, int] | None = None,
) -> tuple[float, str, dict[str, int]]:
    """Score a catalog entry against the spec and explain the score.

    Four weighted terms, each reported, so the number can be argued with instead of
    trusted: keyword relevance, task family, label coverage, and licence clearance.
    """
    described = " ".join([spec.title, spec.summary, spec.domain or "", spec.input_description])
    prompt_terms = set(normalise_key(described).split())
    keywords = {normalise_key(k) for k in entry.get("keywords", [])}
    hits = len(prompt_terms & keywords)
    keyword_score = min(1.0, hits / 3.0) if hits else 0.0

    task_types = set(entry.get("task_types", []))
    task_score = 1.0 if spec.task_type.value in task_types else 0.0

    coverage = label_coverage or {}
    if spec.allowed_labels:
        covered = sum(1 for label in spec.allowed_labels if coverage.get(label, 0) > 0)
        coverage_score = covered / len(spec.allowed_labels)
    elif coverage:
        coverage_score = 1.0
    else:
        coverage_score = 0.5  # unknown, not good: neutral rather than flattering

    commercial = entry.get("commercial_ok")
    licence_score = 1.0 if commercial else (0.3 if commercial is None else 0.0)

    total = 0.4 * keyword_score + 0.2 * task_score + 0.3 * coverage_score + 0.1 * licence_score
    missing = [
        label for label in spec.allowed_labels if spec.allowed_labels and not coverage.get(label, 0)
    ]
    rationale = "; ".join(
        [
            f"keyword match {hits}/{len(keywords)} ({keyword_score:.2f})",
            f"task family {'match' if task_score else 'no match'}",
            (
                f"labels covered {len(spec.allowed_labels) - len(missing)}/{len(spec.allowed_labels)}"
                if spec.allowed_labels
                else "labels unknown"
            ),
            f"licence {entry.get('license') or 'unknown'}",
        ]
    )
    if missing:
        rationale += f"; NO SOURCE ROWS for {', '.join(missing)}"
    return round(total, 3), rationale, coverage


def catalog_candidates(spec: MLSpec) -> list[DatasetCandidate]:
    """Curated entries, scored. Offline-safe: every fact is recorded on the entry."""
    out: list[DatasetCandidate] = []
    for entry in CATALOG:
        coverage = dict(entry.get("verified_mapped_counts") or {})
        score, rationale, coverage = score_candidate(entry, spec, label_coverage=coverage)
        warnings: list[str] = []
        if entry.get("commercial_ok") is False:
            warnings.append(
                f"Licence {entry.get('license')} does not permit commercial use — "
                "the resulting model inherits that restriction."
            )
        if entry.get("synthetic"):
            warnings.append("Synthetic source: it is labelled as such and is not ground truth.")
        for label, count in coverage.items():
            if count == 0:
                warnings.append(
                    f"No source rows map onto '{label}' in this dataset — the class "
                    "cannot be learned from it."
                )
        out.append(
            DatasetCandidate(
                id=str(entry["id"]),
                name=str(entry["name"]),
                source="hf_dataset" if entry.get("repo_id") else "catalog",
                uri=entry.get("uri")
                or (
                    f"https://huggingface.co/datasets/{entry['repo_id']}"
                    if entry.get("repo_id")
                    else None
                ),
                rows=entry.get("rows"),
                columns=list(entry.get("columns") or []),
                fit_score=score,
                fit_rationale=rationale,
                label_coverage=coverage,
                license_note=entry.get("license_note") or (entry.get("license") or _UNVERIFIED),
                commercial_ok=entry.get("commercial_ok"),
                warnings=warnings,
                approved=True,
                provenance=_provenance(entry, transformations=["catalog entry (verified by hand)"]),
            )
        )
    out.sort(key=lambda candidate: candidate.fit_score or 0.0, reverse=True)
    return out


def hub_candidates(spec: MLSpec, *, limit: int = 8) -> tuple[list[DatasetCandidate], str | None]:
    """Search the Hub for extra candidates. Returns (candidates, warning).

    These are *not* buildable on their own: the Hub tells us a dataset exists, not
    which of its columns holds the text or the label. They are offered as leads with
    that limitation stated, and a column mapping turns one into a real option.
    """
    known = {str(entry["id"]) for entry in CATALOG}
    found: dict[str, DatasetCandidate] = {}
    try:
        for query in discovery_queries(spec, limit=2):
            for row in hf_cli.list_datasets(query, limit=limit):
                repo_id = row.get("id") or row.get("name")
                if not repo_id or f"hf:{repo_id}" in known or repo_id in found:
                    continue
                found[repo_id] = DatasetCandidate(
                    id=f"hf:{repo_id}",
                    name=repo_id,
                    source="hf_dataset",
                    uri=f"https://huggingface.co/datasets/{repo_id}",
                    rows=row.get("downloads") and None,
                    fit_score=0.5,
                    fit_rationale=(
                        f"matched the Hub query '{query}' by name; columns, labels and "
                        "licence are unverified until profiled"
                    ),
                    license_note="unknown until profiled",
                    commercial_ok=None,
                    warnings=[
                        "Not profiled: supply a column mapping "
                        '(`params={"columns": {"text": [...], "label": "..."}}`) to build from it.'
                    ],
                    approved=False,
                )
    except Exception as exc:
        logger.info("Hub dataset search unavailable: %s", exc)
        return list(found.values()), f"Hub search unavailable ({type(exc).__name__}); catalog only."

    return sorted(found.values(), key=lambda c: c.name), None


def profile_candidate(
    candidate: DatasetCandidate,
    spec: MLSpec,
    *,
    label_column: str | None,
    threads: int | None = None,
) -> tuple[dict[str, int], list[str]]:
    """Profile a Hub dataset remotely and return (label coverage, warnings)."""
    warnings: list[str] = []
    repo_id = candidate.id.removeprefix("hf:")
    try:
        urls = hf_cli.dataset_parquet_urls(repo_id)
    except Exception as exc:
        return {}, [f"Could not resolve parquet files ({type(exc).__name__}). {_UNVERIFIED}"]

    if not urls:
        return {}, [
            "No parquet conversion available, so the dataset cannot be filtered remotely. "
            "It would have to be downloaded in full."
        ]

    try:
        profile = parquet.profile(urls, label_column=label_column, threads=threads)
    except Exception as exc:
        return {}, [f"Remote profile failed ({type(exc).__name__}). {_UNVERIFIED}"]

    candidate.rows = profile.get("rows") or candidate.rows
    candidate.columns = profile.get("column_names") or candidate.columns
    distribution = profile.get("label_distribution") or {}
    if not label_column or not distribution:
        warnings.append("No label column could be profiled, so label coverage is unknown.")
        return {}, warnings
    coverage, _ = map_labels(
        distribution, spec, label_map=_catalog_label_map(candidate, label_column)
    )
    return coverage, warnings


def catalog_entry(candidate_id: str) -> dict[str, Any] | None:
    """The raw catalog entry for an id, or None when it is not curated."""
    for entry in CATALOG:
        if str(entry["id"]) == candidate_id:
            return dict(entry)
    return None


def _catalog_label_map(candidate: DatasetCandidate, label_column: str) -> dict[str, list[str]]:
    """The catalog's verified mapping for this candidate, if any."""
    for entry in CATALOG:
        if str(entry["id"]) == candidate.id and entry.get("label_column") == label_column:
            return dict(entry.get("label_map") or {})
    return {}


def _plan_label_mapping(candidate: DatasetCandidate | None, spec: MLSpec) -> list[LabelMapping]:
    """The mapping the plan proposes, as an audit trail the user can review."""
    if candidate is None:
        return []
    for entry in CATALOG:
        if str(entry["id"]) != candidate.id:
            continue
        return [
            LabelMapping(
                raw=raw,
                mapped=target,
                action="map" if target in spec.allowed_labels else "drop",
                reason=(
                    f"catalog mapping: '{raw}' -> '{target}'"
                    if target in spec.allowed_labels
                    else f"catalog mapping maps '{raw}' to '{target}', which the spec omits"
                ),
            )
            for target, aliases in (entry.get("label_map") or {}).items()
            for raw in aliases
        ]
    return []


# --- label mapping -----------------------------------------------------


def map_labels(
    distribution: dict[str, int],
    spec: MLSpec,
    *,
    label_map: dict[str, list[str]] | None = None,
) -> tuple[dict[str, int], list[LabelMapping]]:
    """Map raw label values onto the spec's `allowed_labels`.

    Returns (coverage by allowed label, the audit trail). Every raw value gets an
    explicit decision: `map` (with the reason), `keep` (already a spec label), or
    `drop` (with the reason). Nothing is dropped silently, and a raw label is never
    forced into a category that does not claim it.
    """
    allowed = list(spec.allowed_labels)
    allowed_keys = {normalise_key(label): label for label in allowed}
    inverse: dict[str, str] = {}
    for target, aliases in (label_map or {}).items():
        for alias in aliases:
            inverse[normalise_key(alias)] = target

    coverage: dict[str, int] = dict.fromkeys(allowed, 0)
    mappings: list[LabelMapping] = []

    for raw, count in sorted(distribution.items(), key=lambda item: -item[1]):
        key = normalise_key(raw)
        if not allowed:
            # No declared label set: keep the dataset's own labels verbatim.
            coverage[raw] = coverage.get(raw, 0) + count
            mappings.append(
                LabelMapping(raw=raw, mapped=raw, action="keep", reason="no declared label set")
            )
            continue
        if key in allowed_keys:
            target = allowed_keys[key]
            coverage[target] = coverage.get(target, 0) + count
            mappings.append(
                LabelMapping(raw=raw, mapped=target, action="keep", reason="already a spec label")
            )
            continue
        mapped_target = inverse.get(key)
        if mapped_target is None:
            mappings.append(
                LabelMapping(
                    raw=raw,
                    mapped=None,
                    action="drop",
                    reason="not a routing label for this task (no mapping claims it)",
                )
            )
            continue
        if mapped_target not in coverage:
            mappings.append(
                LabelMapping(
                    raw=raw,
                    mapped=None,
                    action="drop",
                    reason=f"maps to '{mapped_target}', which the spec does not list",
                )
            )
            continue
        coverage[mapped_target] = coverage.get(mapped_target, 0) + count
        mappings.append(
            LabelMapping(
                raw=raw, mapped=mapped_target, action="map", reason=f"mapped to '{mapped_target}'"
            )
        )

    for label in allowed:
        coverage.setdefault(label, 0)
    return coverage, mappings


def sufficiency_verdict(
    coverage: dict[str, int],
    *,
    required: int,
    allowed_labels: list[str],
) -> tuple[str, str]:
    """`sufficient` | `marginal` | `insufficient`, plus the reason in plain English."""
    usable = sum(coverage.values())
    if usable < required:
        return (
            "insufficient",
            f"Only {usable} usable examples after label mapping, below the target of "
            f"{required}. Add a source or upload examples rather than training on this.",
        )
    missing = [label for label in allowed_labels if coverage.get(label, 0) == 0]
    counts = [count for count in coverage.values() if count > 0]
    ratio = (max(counts) / min(counts)) if counts else 0.0
    if missing:
        return (
            "marginal",
            f"{usable} usable examples, but no source rows map onto: {', '.join(missing)}. "
            "Either re-specify without those labels or supply examples for them.",
        )
    if ratio > IMBALANCE_RATIO:
        return (
            "marginal",
            f"{usable} usable examples, but the classes are imbalanced "
            f"(largest/smallest = {ratio:.1f}x); report per-class metrics and expect the "
            "rare class to lag.",
        )
    return "sufficient", f"{usable} usable examples across {len(counts)} balanced classes."


# --- records, dedupe, splits -------------------------------------------


def make_record(
    *,
    text: str,
    label: str,
    spec: MLSpec,
    system_prompt: str,
    source_id: str,
    synthetic: bool,
) -> dict[str, Any]:
    """One canonical training example, in chat JSONL form."""
    if spec.output.format is OutputFormat.json_object:
        answer = json.dumps({"label": label}, ensure_ascii=False)
    else:
        answer = label
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
            {"role": "assistant", "content": answer},
        ],
        "label": label,
        "source": source_id,
        "synthetic": synthetic,
    }


def clean_records(
    rows: list[dict[str, Any]],
    spec: MLSpec,
    *,
    label_map: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], list[DataQualityIssue], dict[str, int]]:
    """Normalise, filter, map and deduplicate raw rows into training records.

    Pure and deterministic: same rows in, same records out — which is what makes
    "re-running the plan produces identical splits" testable.
    """
    system_prompt = build_system_prompt(spec)
    issues: list[DataQualityIssue] = []
    counts: Counter[str] = Counter()

    short_examples: list[str] = []
    pii_kinds: Counter[str] = Counter()
    missing_label = 0
    unknown_labels: Counter[str] = Counter()

    by_text: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    duplicate_examples: list[str] = []

    for row in rows:
        raw_text = str(row.get("text") or "")
        raw_label = str(row.get("raw_label") or "").strip()
        source_id = str(row.get("source") or "")
        synthetic = bool(row.get("synthetic"))

        if not raw_label:
            missing_label += 1
            continue

        mapped, mapping = _map_one(raw_label, spec, label_map)
        if mapping is not None and mapping.action == "drop":
            unknown_labels[raw_label] += 1
            continue

        text = normalise_text(raw_text)
        if len(text) < MIN_TEXT_CHARS:
            counts["too_short"] += 1
            if len(short_examples) < 3:
                short_examples.append(f"{text[:40]!r} (label={mapped})")
            continue

        pii = suspected_pii(text)
        if pii:
            pii_kinds[pii] += 1
            counts["pii_suspected"] += 1

        key = text_hash(text)
        if key in by_text:
            duplicate_count += 1
            if len(duplicate_examples) < 3:
                duplicate_examples.append(text[:60])
            continue
        by_text[key] = make_record(
            text=text,
            label=mapped or raw_label,
            spec=spec,
            system_prompt=system_prompt,
            source_id=source_id,
            synthetic=synthetic,
        )

    records = list(by_text.values())
    if missing_label:
        issues.append(
            DataQualityIssue(
                kind="missing_label",
                severity="warning",
                count=missing_label,
                detail="Rows dropped because their label column was empty.",
            )
        )
    if short_examples:
        issues.append(
            DataQualityIssue(
                kind="too_short",
                severity="warning",
                count=counts["too_short"],
                detail=f"Rows dropped as too short (<{MIN_TEXT_CHARS} characters).",
                examples=short_examples,
            )
        )
    if duplicate_count:
        issues.append(
            DataQualityIssue(
                kind="duplicate",
                severity="info",
                count=duplicate_count,
                detail="Exact duplicates removed after whitespace/case normalisation.",
                examples=duplicate_examples,
            )
        )
    if pii_kinds:
        issues.append(
            DataQualityIssue(
                kind="pii_suspected",
                severity="error" if spec.is_high_stakes else "warning",
                count=sum(pii_kinds.values()),
                # Deliberately no examples: reporting them would copy the data we
                # are warning about.
                detail=(
                    "Rows whose text matched a personal-data pattern "
                    f"({', '.join(sorted(pii_kinds))}). They were kept, but a human "
                    "should review them before this dataset is used."
                ),
            )
        )
    if unknown_labels:
        issues.append(
            DataQualityIssue(
                kind="unmapped_label",
                severity="info",
                count=int(sum(unknown_labels.values())),
                detail=(
                    "Rows dropped because their label maps onto no allowed label: "
                    + ", ".join(f"{name} ({n})" for name, n in unknown_labels.most_common(5))
                ),
            )
        )
    return records, issues, dict(counts)


def _map_one(
    raw_label: str, spec: MLSpec, label_map: dict[str, list[str]] | None
) -> tuple[str | None, LabelMapping | None]:
    """Map a single raw label, mirroring `map_labels` for row-level filtering."""
    allowed_keys = {normalise_key(label): label for label in spec.allowed_labels}
    if not spec.allowed_labels:
        return raw_label, None
    key = normalise_key(raw_label)
    if key in allowed_keys:
        return allowed_keys[key], None
    for target, aliases in (label_map or {}).items():
        if key in {normalise_key(alias) for alias in aliases}:
            if target in allowed_keys.values():
                return target, LabelMapping(raw=raw_label, mapped=target, action="map")
            return None, LabelMapping(
                raw=raw_label,
                mapped=None,
                action="drop",
                reason=f"maps to '{target}', which the spec does not list",
            )
    return None, LabelMapping(raw=raw_label, mapped=None, action="drop")


def stratified_split(
    records: list[dict[str, Any]],
    *,
    seed: int,
    test_fraction: float,
    validation_fraction: float,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministic, label-stratified split.

    Rows are sorted before shuffling so the result depends on the *content* of the
    dataset and the seed, never on the order a download happened to arrive in. Each
    label is split independently, which keeps a rare class present in every split
    instead of letting a global shuffle swallow it.
    """
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[str(record.get("label"))].append(record)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda r: text_hash(_record_text(r)))
        random.Random(f"{seed}:{label}").shuffle(group)
        total = len(group)
        n_test = round(total * test_fraction)
        n_validation = round(total * validation_fraction)
        # Never take everything: a one-row class belongs in train, not in test.
        if total > 1:
            n_test = min(n_test, total - 1)
            n_validation = min(n_validation, total - 1 - n_test)
        test.extend(group[:n_test])
        validation.extend(group[n_test : n_test + n_validation])
        train.extend(group[n_test + n_validation :])

    return {"train": train, "validation": validation, "test": test}


def _record_text(record: dict[str, Any]) -> str:
    for message in record.get("messages") or []:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def find_leakage(
    split: dict[str, list[dict[str, Any]]],
    *,
    limit: int = MAX_LEAKAGE_CHECK,
) -> tuple[dict[str, list[dict[str, Any]]], int, bool]:
    """Drop near-duplicates that appear in both train and an evaluation split.

    Leakage is the one data defect that silently inflates every later number, so it
    is checked here rather than trusted to the dedupe pass (which only removes exact
    matches within the whole corpus). Returns (clean split, leaked count, sampled).
    """
    train = split.get("train", [])
    sampled = len(train) > limit
    sample = train[:limit] if sampled else train
    train_signatures = [(record, shingles(_record_text(record))) for record in sample]

    leaked = 0
    cleaned: dict[str, list[dict[str, Any]]] = {"train": list(train)}
    for name in ("validation", "test"):
        kept: list[dict[str, Any]] = []
        for record in split.get(name, []):
            signature = shingles(_record_text(record))
            if any(
                jaccard(signature, other) >= NEAR_DUP_THRESHOLD
                for _, other in train_signatures
                if other
            ):
                leaked += 1
                continue
            kept.append(record)
        cleaned[name] = kept
    return cleaned, leaked, sampled


def write_jsonl(records: list[dict[str, Any]], path: Path) -> Path:
    """Write records as newline-delimited JSON, sorted for byte-stable output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return path


# --- fetching rows -----------------------------------------------------


def _quote_ident(name: str) -> str:
    """Quote a column name, refusing anything that is not a plain identifier."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_ ]*", name):
        raise ValueError(f"Unsafe column name: {name!r}")
    return '"' + name.replace('"', "") + '"'


def fetch_sql(columns: dict[str, Any], *, fetch_cap: int) -> str:
    """The single query that turns a wide parquet table into (text, raw_label)."""
    text_cols = list(columns.get("text") or [])
    label_col = str(columns.get("label") or "")
    if not text_cols or not label_col:
        raise ValueError("A column mapping needs both 'text' and 'label'.")
    text_expr = " || ' ' || ".join(
        f"COALESCE(CAST({_quote_ident(col)} AS VARCHAR), '')" for col in text_cols
    )
    return (
        f"SELECT NULLIF(TRIM({text_expr}), '') AS text, "
        f"CAST({_quote_ident(label_col)} AS VARCHAR) AS raw_label "
        f"FROM data LIMIT {int(fetch_cap)}"
    )


def read_local_rows(
    path: Path,
    *,
    columns: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    """Read an uploaded JSONL/CSV file into raw rows.

    Uploads are untrusted input: the extension is checked against a whitelist, the
    file size is capped, and decoding uses `errors="replace"` so a malformed byte
    cannot abort the run.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Upload not found: {path}")
    if path.suffix.lower() not in {".jsonl", ".csv"}:
        raise ValueError(f"Unsupported upload type: {path.suffix or '(none)'}")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError(f"Upload is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")

    text_cols = list(columns.get("text") or [])
    label_col = str(columns.get("label") or "")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        if path.suffix.lower() == ".csv":
            for record in csv.DictReader(handle):
                rows.append(_row_from(record, text_cols, label_col))
                if len(rows) >= limit:
                    break
        else:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    continue
                rows.append(_row_from(payload, text_cols, label_col))
                if len(rows) >= limit:
                    break
    return rows


def _resolve_upload_path(raw_path: str, settings: Any) -> Path:
    """Resolve an `upload:<path>` candidate id, refusing anything outside `upload_root`.

    `raw_path` comes straight from request params, so without this check a caller
    could pass `upload:C:/anything.csv` and have it ingested as training data —
    an arbitrary-file-read, not just an untrusted-content problem.
    """
    upload_root = Path(settings.upload_root).resolve()
    path = Path(raw_path)
    path = (upload_root / path).resolve() if not path.is_absolute() else path.resolve()
    if path != upload_root and upload_root not in path.parents:
        raise ValueError(f"Upload path must be inside the upload root ({upload_root}).")
    return path


def _row_from(record: dict[str, Any], text_cols: list[str], label_col: str) -> dict[str, Any]:
    parts = [str(record.get(col) or "") for col in text_cols]
    return {
        "text": " ".join(part for part in parts if part),
        "raw_label": record.get(label_col),
        "source": "upload",
    }


def fetch_remote_rows(
    repo_id: str,
    *,
    columns: dict[str, Any],
    work_dir: Path,
    fetch_cap: int,
    threads: int | None = None,
) -> list[dict[str, Any]]:
    """Download only the columns we need, once, into the job's work directory.

    This is the *only* remote read in a build: read the parquet URL, project the two
    columns we care about, write a local JSONL, then work entirely from that file.
    Re-querying the Hub per step is what makes these datasets slow and flaky.
    """
    urls = hf_cli.dataset_parquet_urls(repo_id)
    if not urls:
        raise RuntimeError(
            f"{repo_id} has no parquet conversion, so it cannot be filtered remotely. "
            "Download it through the `hf` CLI and build from the local file instead."
        )
    out = work_dir / "source_rows.jsonl"
    parquet.materialize(
        urls,
        fetch_sql(columns, fetch_cap=fetch_cap),
        out,
        fmt="json",
        threads=threads,
        description=f"dataset materialization ({repo_id})",
    )
    rows: list[dict[str, Any]] = []
    with out.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                payload["source"] = repo_id
                rows.append(payload)
    return rows


def select_to_target(
    records: list[dict[str, Any]], *, target: int, seed: int
) -> list[dict[str, Any]]:
    """Trim cleaned records to the approved number of examples.

    The quota per label is proportional to what the sources yielded, so the build
    keeps the dataset's own class balance instead of flattening it; the rounding
    remainder goes to the largest groups so the result lands on the approved number
    rather than a quirky one below it. `target_examples` is an approval boundary —
    it decides how much training the user agreed to pay for — so delivering four
    times it (which an uncapped fetch does) is not a harmless overshoot.
    """
    if len(records) <= target:
        return records

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[str(record.get("label"))].append(record)
    total = len(records)

    quota = {
        label: max(1, int(target * len(group) / total)) for label, group in sorted(by_label.items())
    }
    remainder = target - sum(quota.values())
    for label in sorted(quota, key=lambda key: -len(by_label[key])):
        if remainder <= 0:
            break
        if quota[label] < len(by_label[label]):
            quota[label] += 1
            remainder -= 1

    selected: list[dict[str, Any]] = []
    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda record: text_hash(_record_text(record)))
        random.Random(f"{seed}:keep:{label}").shuffle(group)
        selected.extend(group[: quota[label]])
    return selected


def stratified_sample(
    rows: list[dict[str, Any]],
    spec: MLSpec,
    *,
    label_map: dict[str, list[str]] | None,
    target: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Cap the row count, keeping the class balance of what was fetched.

    Applied *before* cleaning so the fetch cap does not decide the class balance.
    Deterministic: seeded, and ordered by the row's text before sampling.
    """
    if len(rows) <= target:
        return rows
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        mapped, mapping = _map_one(str(row.get("raw_label") or ""), spec, label_map)
        key = mapped or (f"__drop__{mapping.action if mapping else 'keep'}")
        buckets[key].append(row)

    per_class = max(1, target // max(1, len(buckets)))
    sampled: list[dict[str, Any]] = []
    for key in sorted(buckets):
        group = sorted(buckets[key], key=lambda r: text_hash(str(r.get("text") or "")))
        random.Random(f"{seed}:{key}").shuffle(group)
        sampled.extend(group[:per_class])
    return sampled


# --- the stage ---------------------------------------------------------


class DatasetStage(Stage):
    name = StageName.curate_data
    label = "Curate dataset"
    implemented = True  # Stage 2
    #: Approval lives in `params["build"]` (and the API routes builds to the job
    #: queue), so the class-level gate stays off — the plan itself is free to run.
    requires_approval = False

    def run(self, ctx: StageContext) -> StageResult:
        spec = _spec_from_project(ctx)
        if spec is None:
            return StageResult(
                status="failed",
                error="No ML specification on the project. Run the analyze stage first.",
            )
        if spec.task_type not in CURATABLE_TASKS:
            return StageResult(
                status="failed",
                error=(
                    f"Dataset curation is not implemented for task type "
                    f"'{spec.task_type.value}'. Supported: "
                    + ", ".join(sorted(t.value for t in CURATABLE_TASKS))
                ),
            )

        plan = self._plan(ctx, spec)
        if not ctx.params.get("build"):
            return self._plan_result(plan)

        return self._build(ctx, spec, plan)

    # -- planning -------------------------------------------------------
    def _plan(self, ctx: StageContext, spec: MLSpec) -> DatasetPlan:
        """Reuse a user-reviewed plan, or discover candidates fresh."""
        if ctx.params.get("plan"):
            reviewed = DatasetPlan.model_validate(ctx.params["plan"])
            reviewed.requires_user_input = False
            reviewed.sufficiency_notes = (
                reviewed.sufficiency_notes or "Plan supplied by the user; not re-discovered."
            )
            return reviewed

        offline = bool(ctx.params.get("offline"))
        candidates = catalog_candidates(spec)
        warnings: list[str] = []
        if not offline:
            extra, warning = hub_candidates(spec)
            candidates = [*candidates, *extra]
            if warning:
                warnings.append(warning)

        recommended = candidates[0] if candidates else None
        profile_warnings: list[str] = []
        if recommended is not None and not offline:
            coverage, profile_warnings = profile_candidate(
                recommended,
                spec,
                label_column=_label_column_for(recommended, ctx.params),
                threads=ctx.settings.duckdb_threads,
            )
            if coverage:
                self._apply_coverage(recommended, spec, coverage)
            warnings.extend(profile_warnings)

        target = int(ctx.params.get("target_examples") or DEFAULT_TARGET_EXAMPLES)
        coverage = recommended.label_coverage if recommended else {}
        verdict, verdict_note = sufficiency_verdict(
            coverage, required=target, allowed_labels=sorted(set(spec.allowed_labels))
        )
        if recommended is None:
            verdict = "insufficient"
            verdict_note = (
                "No candidate dataset was found for this task. Upload examples "
                "(an upload: candidate) to continue."
            )

        plan = DatasetPlan(
            task_type=spec.task_type.value,
            target_examples=target,
            candidates=candidates,
            selected=[recommended.id] if recommended else [],
            augmentation=["none"],
            label_mapping=_plan_label_mapping(recommended, spec),
            dedupe=True,
            normalize=True,
            splits=SplitSizes(
                train=round(
                    target
                    * (1 - spec.evaluation.test_fraction - spec.evaluation.validation_fraction)
                ),
                validation=round(target * spec.evaluation.validation_fraction),
                test=round(target * spec.evaluation.test_fraction),
            ),
            leakage_checks=[
                "exact duplicate removal (whitespace/case normalised)",
                f"near-duplicate check across splits (Jaccard >= {NEAR_DUP_THRESHOLD})",
            ],
            sufficiency=verdict,
            sufficiency_notes=" ".join([verdict_note, *warnings]) or verdict_note,
            requires_user_input=verdict == "insufficient",
        )
        return plan

    def _apply_coverage(
        self, candidate: DatasetCandidate, spec: MLSpec, coverage: dict[str, int]
    ) -> None:
        """Refresh a candidate's facts after a live profile, and re-score it.

        The score is recomputed because label coverage is one of the weighted terms:
        a dataset that loses a class to a live profile must rank lower, not keep the
        flattering number a catalog entry earned before it was checked.
        """
        candidate.label_coverage = coverage
        entry = catalog_entry(candidate.id) or {
            "keywords": [],
            "task_types": [spec.task_type.value],
            "license": candidate.license_note,
            "commercial_ok": candidate.commercial_ok,
        }
        score, rationale, _ = score_candidate(entry, spec, label_coverage=coverage)
        candidate.fit_score = score
        candidate.fit_rationale = rationale
        for label in spec.allowed_labels:
            if coverage.get(label, 0) == 0:
                note = (
                    f"No source rows map onto '{label}' in this dataset — the class "
                    "cannot be learned from it."
                )
                if note not in candidate.warnings:
                    candidate.warnings.append(note)

    def _plan_result(self, plan: DatasetPlan) -> StageResult:
        recommended = plan.candidates[0] if plan.candidates else None
        summary = (
            f"Reviewed {len(plan.candidates)} candidate(s); recommended "
            f"'{recommended.name}' (fit {recommended.fit_score}). Sufficiency: {plan.sufficiency}."
            if recommended
            else "No candidate dataset found; uploads are required."
        )
        empty = [
            label
            for label, count in (recommended.label_coverage if recommended else {}).items()
            if count == 0
        ]
        if empty:
            summary += f" Warning: no source rows for {', '.join(empty)}."
        return StageResult(
            output=plan.model_dump(mode="json"),
            summary=summary,
            metrics={"candidates": float(len(plan.candidates))},
            artifacts=[
                {
                    "kind": "dataset_plan",
                    "name": "dataset_plan.json",
                    "path": "dataset/dataset_plan.json",
                    "meta": {"sufficiency": plan.sufficiency, "selected": plan.selected},
                }
            ],
            # Nothing has been downloaded yet, so the project must not advance.
            needs_approval=True,
            blocked_on=["approve_dataset_build"],
        )

    # -- building -------------------------------------------------------
    def _build(self, ctx: StageContext, spec: MLSpec, plan: DatasetPlan) -> StageResult:
        selected_ids = list(ctx.params.get("selected") or plan.selected)
        if not selected_ids:
            return StageResult(
                status="failed",
                error="No dataset selected. Review the plan and pass `selected` (or a plan).",
            )

        target = int(
            ctx.params.get("target_examples") or plan.target_examples or DEFAULT_TARGET_EXAMPLES
        )
        fetch_cap = max(target * FETCH_MULTIPLIER, target)
        issues: list[DataQualityIssue] = []
        rows: list[dict[str, Any]] = []
        sources: list[Provenance] = []
        synthetic_rows = 0

        for candidate_id in selected_ids:
            try:
                built = self._rows_for(ctx, spec, candidate_id, fetch_cap=fetch_cap)
            except Exception as exc:
                return StageResult(
                    status="failed",
                    error=f"Could not read source '{candidate_id}': {type(exc).__name__}: {exc}",
                )
            rows.extend(built["rows"])
            sources.append(built["provenance"])
            synthetic_rows += sum(1 for row in built["rows"] if row.get("synthetic"))

        if not rows:
            return StageResult(
                status="failed",
                error=(
                    "Every selected source produced zero usable rows. Nothing was materialised; "
                    "add a dataset or upload examples and run the plan again."
                ),
            )

        label_map = self._label_map_for(ctx, spec, selected_ids)
        rows = stratified_sample(
            rows, spec, label_map=label_map, target=fetch_cap, seed=ctx.settings.seed
        )

        pool, quality, _counts = clean_records(rows, spec, label_map=label_map)
        issues.extend(quality)
        if not pool:
            return StageResult(
                status="failed",
                error=(
                    "All rows were removed by validation (labels, length or duplicates). "
                    "See the quality issues for counts and examples."
                ),
            )

        # The verdict is about whether the *sources* can support the task, so it is
        # judged on the usable pool. Whether the build then kept exactly the approved
        # number is reported by the card's split sizes, which is a different fact.
        pool_distribution = Counter(str(record["label"]) for record in pool)
        verdict, verdict_note = sufficiency_verdict(
            dict(pool_distribution),
            required=target,
            allowed_labels=sorted(set(spec.allowed_labels)),
        )

        selected_records = select_to_target(pool, target=target, seed=ctx.settings.seed)
        split = stratified_split(
            selected_records,
            seed=ctx.settings.seed,
            test_fraction=spec.evaluation.test_fraction,
            validation_fraction=spec.evaluation.validation_fraction,
        )
        split, leaked, sampled = find_leakage(split)
        if leaked:
            issues.append(
                DataQualityIssue(
                    kind="leakage",
                    severity="error",
                    count=leaked,
                    detail=(
                        "Near-duplicate rows were found in both train and an evaluation split and "
                        "were removed from the evaluation splits."
                        + (" (train was sampled for this check.)" if sampled else "")
                    ),
                )
            )

        # Every number below describes the files that were actually written, so the
        # card adds up: totals equal the split sizes, and the counts match the JSONL.
        delivered = [*split["train"], *split["validation"], *split["test"]]
        distribution = Counter(str(record["label"]) for record in delivered)
        missing_classes = [
            label for label in spec.allowed_labels if distribution.get(label, 0) == 0
        ]
        if missing_classes:
            issues.append(
                DataQualityIssue(
                    kind="missing_class",
                    severity="error",
                    count=len(missing_classes),
                    detail=(
                        "No usable examples for: "
                        + ", ".join(missing_classes)
                        + ". The model cannot learn these labels from this data; "
                        "drop them from the spec or supply examples."
                    ),
                )
            )

        dataset_dir = ctx.artifact_dir / "dataset"
        files: dict[str, str] = {}
        hashes: dict[str, str] = {}
        artifacts: list[dict[str, Any]] = []
        for name in ("train", "validation", "test"):
            path = write_jsonl(split[name], dataset_dir / f"{name}.jsonl")
            relative = f"dataset/{name}.jsonl"
            files[name] = relative
            hashes[name] = sha256_file(path)
            artifacts.append(
                {
                    "kind": "dataset",
                    "name": f"{name}.jsonl",
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": hashes[name],
                    "meta": {"rows": len(split[name]), "format": "chat_jsonl"},
                }
            )

        card = DatasetCard(
            id=f"ds_{ctx.project_id[:8]}",
            project_id=ctx.project_id,
            total_examples=len(delivered),
            splits=SplitSizes(
                train=len(split["train"]),
                validation=len(split["validation"]),
                test=len(split["test"]),
            ),
            label_distribution=dict(distribution),
            sources=sources,
            synthetic_fraction=round(synthetic_rows / max(1, len(rows)), 4),
            # `synthetic_fraction` is over the rows the sources produced; this is the
            # pool that survived validation, before the build trimmed to the target.
            issues=issues,
            files=files,
            sha256=hashes,
            format="chat_jsonl",
            notes=(
                "Chat-format JSONL: messages[system,user,assistant] plus the label, source "
                "and synthetic flag. The system message is the exact instruction used by the "
                "baseline and training stages."
            ),
            stats={
                "system_prompt": build_system_prompt(spec),
                "seed": ctx.settings.seed,
                "test_fraction": spec.evaluation.test_fraction,
                "validation_fraction": spec.evaluation.validation_fraction,
                "leaked_rows_removed": leaked,
                "usable_pool": len(pool),
                "target_examples": target,
                "sources_selected": selected_ids,
            },
        )
        card_path = dataset_dir / "dataset_card.json"
        card_path.write_text(
            json.dumps(card.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
        )
        artifacts.append(
            {
                "kind": "dataset_card",
                "name": "dataset_card.json",
                "path": "dataset/dataset_card.json",
                "size_bytes": card_path.stat().st_size,
                "sha256": sha256_file(card_path),
                "meta": {"sufficiency": verdict},
            }
        )

        plan.splits = card.splits
        plan.sufficiency = verdict
        plan.sufficiency_notes = verdict_note
        plan.requires_user_input = verdict == "insufficient"

        summary = (
            f"Built {card.total_examples} examples "
            f"(train {card.splits.train} / val {card.splits.validation} / test {card.splits.test}); "
            f"sufficiency={verdict}."
        )
        if missing_classes:
            summary += f" Missing classes: {', '.join(missing_classes)}."

        return StageResult(
            output={**plan.model_dump(mode="json"), "card": card.model_dump(mode="json")},
            summary=summary,
            metrics={
                "examples": float(card.total_examples),
                "train": float(card.splits.train),
                "validation": float(card.splits.validation),
                "test": float(card.splits.test),
                "labels": float(len(distribution)),
                "issues": float(len(issues)),
                "synthetic_fraction": card.synthetic_fraction,
            },
            artifacts=artifacts,
            # `insufficient` must not advance the project: the user has to fix the data.
            needs_approval=verdict == "insufficient",
            blocked_on=["insufficient_data"] if verdict == "insufficient" else [],
            error=None,
        )

    # -- per-candidate row loading --------------------------------------
    def _rows_for(
        self, ctx: StageContext, spec: MLSpec, candidate_id: str, *, fetch_cap: int
    ) -> dict[str, Any]:
        """Load raw rows for one selected candidate, plus its provenance."""
        columns, _label_map, entry = self._columns_for(ctx, spec, candidate_id)

        if candidate_id.startswith("upload:"):
            path = _resolve_upload_path(candidate_id.removeprefix("upload:"), ctx.settings)
            rows = read_local_rows(path, columns=columns, limit=fetch_cap)
            provenance = Provenance(
                source="user_upload",
                uri=path.as_posix(),
                license=License(id="unknown", unknown=True, notes="User-supplied upload."),
                transformations=["read as untrusted upload"],
                notes="Uploaded by the user; not independently verified.",
            )
            return {"rows": rows, "provenance": provenance}

        repo_id = candidate_id.removeprefix("hf:")
        rows = fetch_remote_rows(
            repo_id,
            columns=columns,
            work_dir=ctx.work_dir,
            fetch_cap=fetch_cap,
            threads=ctx.settings.duckdb_threads,
        )
        if entry:
            for row in rows:
                row["synthetic"] = bool(entry.get("synthetic"))
        provenance = _provenance(
            entry or {"repo_id": repo_id},
            transformations=[
                "remote projection of the text and label columns (single read)",
                "labelling mapped onto the spec's allowed labels",
                "exact-duplicate removal (whitespace/case normalised)",
                f"seeded stratified split (seed={ctx.settings.seed})",
                "near-duplicate leakage check across splits",
            ],
        )
        return {"rows": rows, "provenance": provenance}

    def _columns_for(
        self, ctx: StageContext, spec: MLSpec, candidate_id: str
    ) -> tuple[dict[str, Any], dict[str, list[str]] | None, dict[str, Any] | None]:
        """Resolve the column mapping and label map for a candidate.

        Only the catalog knows a dataset's columns, so an unmapped non-catalog
        candidate is a clear error rather than a guess.
        """
        supplied = ctx.params.get("columns") or {}
        for entry in CATALOG:
            if str(entry["id"]) == candidate_id:
                columns = {
                    "text": supplied.get("text") or entry["text_columns"],
                    "label": supplied.get("label") or entry["label_column"],
                }
                return columns, dict(entry.get("label_map") or {}), dict(entry)
        if supplied.get("text") and supplied.get("label"):
            return {"text": list(supplied["text"]), "label": str(supplied["label"])}, None, None
        raise ValueError(
            f"'{candidate_id}' is not in the curated catalog, so its columns are unknown. "
            'Pass a mapping, e.g. params={"columns": {"text": ["body"], "label": "type"}}.'
        )

    def _label_map_for(
        self, ctx: StageContext, spec: MLSpec, selected: list[str]
    ) -> dict[str, list[str]]:
        """The effective raw-label -> allowed-label map for this build.

        A reviewed plan's own `label_mapping` is applied first: the user edited it, so
        it outranks the catalog's default. Catalog mappings fill in the rest.
        """
        merged: dict[str, list[str]] = _reviewed_label_map(ctx)
        for candidate_id in selected:
            try:
                _columns, label_map, _entry = self._columns_for(ctx, spec, candidate_id)
            except ValueError:
                continue
            for target, aliases in (label_map or {}).items():
                merged.setdefault(target, [])
                merged[target].extend(alias for alias in aliases if alias not in merged[target])
        return merged


def _reviewed_label_map(ctx: StageContext) -> dict[str, list[str]]:
    """The label map carried by a user-reviewed plan, ignoring dropped rows."""
    plan = ctx.params.get("plan") or {}
    reviewed: dict[str, list[str]] = {}
    for item in plan.get("label_mapping") or []:
        if not isinstance(item, dict):
            continue
        raw, mapped = item.get("raw"), item.get("mapped")
        if not raw or not mapped or item.get("action", "keep") == "drop":
            continue
        reviewed.setdefault(str(mapped), [])
        if str(raw) not in reviewed[str(mapped)]:
            reviewed[str(mapped)].append(str(raw))
    return reviewed


def _label_column_for(candidate: DatasetCandidate, params: dict[str, Any]) -> str | None:
    supplied = params.get("columns") or {}
    if supplied.get("label"):
        return str(supplied["label"])
    for entry in CATALOG:
        if str(entry["id"]) == candidate.id:
            return str(entry["label_column"])
    return None


def _spec_from_project(ctx: StageContext) -> MLSpec | None:
    """Parse the project's stored spec. Returns None when the analyze stage has not run."""
    raw = getattr(ctx.project, "spec", None)
    if not raw:
        return None
    try:
        return MLSpec.model_validate(raw)
    except Exception as exc:
        logger.warning("Project %s has an unreadable spec: %s", ctx.project_id[:8], exc)
        return None


__all__ = [
    "CATALOG",
    "CURATABLE_TASKS",
    "DEFAULT_TARGET_EXAMPLES",
    "DatasetStage",
    "build_system_prompt",
    "catalog_candidates",
    "catalog_entry",
    "clean_records",
    "discovery_queries",
    "fetch_remote_rows",
    "fetch_sql",
    "find_leakage",
    "hub_candidates",
    "map_labels",
    "normalise_key",
    "profile_candidate",
    "read_local_rows",
    "score_candidate",
    "select_to_target",
    "sha256_file",
    "shingles",
    "stratified_sample",
    "stratified_split",
    "sufficiency_verdict",
    "write_jsonl",
]
