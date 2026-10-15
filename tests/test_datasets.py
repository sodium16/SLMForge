"""Stage 2 — dataset curation contracts.

Every test runs on a handful of hand-written rows and never touches the network, so
the suite stays fast and honest: what is asserted is the *contract* (labels mapped or
dropped with a reason, splits reproducible, cards truthful about their sources) rather
than the internals of the algorithm.

The invariant that matters most here is reproducibility: the same inputs and the same
seed must produce byte-identical splits, or no later comparison between runs means
anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slmforge.config import settings
from slmforge.db import SessionLocal
from slmforge.models import JobKind, Project, ProjectStatus
from slmforge.pipeline import datasets as ds
from slmforge.pipeline.base import StageContext
from slmforge.pipeline.datasets import DatasetStage
from slmforge.schemas import (
    DatasetPlan,
    MLSpec,
    OutputFormat,
    OutputSpec,
    TaskType,
)

LABELS = ["Billing", "Technical", "Account Access", "General"]


def make_spec(*, labels: list[str] | None = None) -> MLSpec:
    return MLSpec(
        title="Support ticket router",
        summary="Route customer support tickets to the right queue.",
        task_type=TaskType.classification,
        input_description="ticket text",
        allowed_labels=list(LABELS if labels is None else labels),
        output=OutputSpec(format=OutputFormat.json_object),
        domain="customer support",
    )


def _ctx(
    tmp_path: Path,
    *,
    spec: MLSpec | None = None,
    params: dict | None = None,
) -> StageContext:
    project = Project(
        id="proj-data", name="p", prompt="route tickets", status=ProjectStatus.specified
    )
    project.spec = (spec or make_spec()).model_dump(mode="json", by_alias=True)
    return StageContext(
        project_id=project.id,
        prompt=project.prompt,
        project=project,
        artifact_dir=tmp_path / "artifacts",
        work_dir=tmp_path / "work",
        params=params or {},
    )


def _upload(tmp_path: Path, rows: list[dict]) -> str:
    # Must live under settings.upload_root: the stage refuses upload: paths outside it.
    upload_dir = Path(settings.upload_root)
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{tmp_path.name}-upload.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return f"upload:{path.as_posix()}"


# --- label mapping ---------------------------------------------------------


def test_declared_labels_are_kept_verbatim() -> None:
    spec = make_spec()
    coverage, mappings = ds.map_labels({"Billing": 10, "Technical": 4}, spec)
    assert coverage["Billing"] == 10
    assert coverage["Technical"] == 4
    assert coverage["General"] == 0
    assert {m.action for m in mappings} == {"keep"}


def test_aliases_map_and_everything_else_is_dropped_with_a_reason() -> None:
    """A row must never be silently relabelled or silently thrown away."""
    spec = make_spec()
    label_map = {"Billing": ["Billing and Payments"], "Technical": ["IT Support"]}
    coverage, mappings = ds.map_labels(
        {"Billing and Payments": 7, "IT Support": 3, "News": 99}, spec, label_map=label_map
    )
    assert coverage["Billing"] == 7
    assert coverage["Technical"] == 3
    assert coverage["Account Access"] == 0

    dropped = {m.raw: m for m in mappings if m.action == "drop"}
    assert set(dropped) == {"News"}
    assert dropped["News"].mapped is None
    assert dropped["News"].reason, "a drop without a reason is indistinguishable from data loss"


def test_a_mapping_to_an_unspecified_label_is_dropped_not_forced() -> None:
    spec = make_spec(labels=["Billing", "General"])
    coverage, mappings = ds.map_labels(
        {"Refunds": 5}, spec, label_map={"Account Access": ["Refunds"]}
    )
    assert coverage == {"Billing": 0, "General": 0}
    assert mappings[0].action == "drop"
    assert "Account Access" in (mappings[0].reason or "")


def test_no_declared_labels_means_the_datasets_own_labels_survive() -> None:
    spec = make_spec(labels=[])
    coverage, _ = ds.map_labels({"Pricing": 2, "Bug": 1}, spec)
    assert coverage == {"Pricing": 2, "Bug": 1}


# --- cleaning --------------------------------------------------------------


def test_duplicates_short_rows_and_missing_labels_are_removed_and_counted() -> None:
    spec = make_spec()
    rows = [
        {"text": "I was charged twice for invoice 4471", "raw_label": "Billing"},
        {"text": "I was charged  twice for invoice 4471  ", "raw_label": "Billing"},  # dupe
        {"text": "hi", "raw_label": "General"},  # too short
        {"text": "The app crashes on upload", "raw_label": ""},  # missing label
        {"text": "The app crashes whenever I upload a file", "raw_label": "Technical"},
    ]
    records, issues, _counts = ds.clean_records(rows, spec)
    assert len(records) == 2
    kinds = {issue.kind: issue.count for issue in issues}
    assert kinds["duplicate"] == 1
    assert kinds["too_short"] == 1
    assert kinds["missing_label"] == 1


def test_records_are_chat_jsonl_with_the_spec_s_output_shape() -> None:
    spec = make_spec()
    label_map = {"Billing": ["Billing and Payments"]}
    records, _issues, _counts = ds.clean_records(
        [{"text": "My card was declined again", "raw_label": "Billing and Payments"}],
        spec,
        label_map=label_map,
    )
    record = records[0]
    assert record["label"] == "Billing"
    assert [m["role"] for m in record["messages"]] == ["system", "user", "assistant"]
    assert record["messages"][0]["content"] == ds.build_system_prompt(spec)
    assert json.loads(record["messages"][2]["content"]) == {"label": "Billing"}


def test_pii_is_counted_but_never_copied_into_the_report() -> None:
    """The warning must not become the leak it warns about."""
    spec = make_spec()
    rows = [{"text": "Email me at jane.doe@example.com about invoice 12", "raw_label": "Billing"}]
    _records, issues, _counts = ds.clean_records(rows, spec)
    pii = next(issue for issue in issues if issue.kind == "pii_suspected")
    assert pii.count == 1
    assert pii.examples == []
    assert "jane.doe@example.com" not in json.dumps(pii.model_dump())


def test_high_stakes_raises_pii_severity() -> None:
    spec = make_spec().model_copy(update={"is_high_stakes": True})
    rows = [{"text": "Reach the patient on 555-123-4567", "raw_label": "General"}]
    _records, issues, _counts = ds.clean_records(rows, spec)
    assert next(i for i in issues if i.kind == "pii_suspected").severity == "error"


# --- splits ----------------------------------------------------------------


def _toy_records() -> list[dict]:
    records = []
    for label in LABELS:
        for n in range(10):
            records.append(
                {
                    "messages": [
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": f"{label} ticket number {n}"},
                        {"role": "assistant", "content": json.dumps({"label": label})},
                    ],
                    "label": label,
                    "source": "toy",
                    "synthetic": False,
                }
            )
    return records


def test_splits_are_deterministic_and_cover_every_class() -> None:
    records = _toy_records()
    first = ds.stratified_split(records, seed=42, test_fraction=0.2, validation_fraction=0.1)
    second = ds.stratified_split(
        list(reversed(records)), seed=42, test_fraction=0.2, validation_fraction=0.1
    )
    assert first == second, "input order must not change the split"

    for name in ("train", "validation", "test"):
        labels = {record["label"] for record in first[name]}
        assert labels == set(LABELS), f"{name} lost a class"
    assert sum(len(first[name]) for name in first) == len(records)


def test_a_one_row_class_stays_in_train_instead_of_vanishing() -> None:
    records = [
        *_toy_records(),
        {
            "messages": [{"role": "user", "content": "only one of these"}],
            "label": "Rare",
            "source": "toy",
            "synthetic": False,
        },
    ]
    split = ds.stratified_split(records, seed=7, test_fraction=0.5, validation_fraction=0.5)
    assert sum(1 for r in split["train"] if r["label"] == "Rare") == 1


def test_written_splits_are_byte_identical_on_a_rerun(tmp_path: Path) -> None:
    """The reproducibility promise, checked on the actual files."""
    records = _toy_records()
    split = ds.stratified_split(records, seed=42, test_fraction=0.2, validation_fraction=0.1)

    first = ds.write_jsonl(split["train"], tmp_path / "a" / "train.jsonl")
    second = ds.write_jsonl(split["train"], tmp_path / "b" / "train.jsonl")
    assert first.read_bytes() == second.read_bytes()
    assert ds.sha256_file(first) == ds.sha256_file(second)


def test_near_duplicates_across_train_and_test_are_removed() -> None:
    train_text = "The dashboard will not load for our whole team since this morning"
    leaking = {
        "messages": [{"role": "user", "content": train_text + " again"}],
        "label": "Technical",
        "source": "toy",
        "synthetic": False,
    }
    split = {
        "train": [
            {
                "messages": [{"role": "user", "content": train_text}],
                "label": "Technical",
                "source": "toy",
                "synthetic": False,
            }
        ],
        "validation": [],
        "test": [leaking, dict(leaking, messages=[{"role": "user", "content": "unrelated text"}])],
    }
    cleaned, leaked, sampled = ds.find_leakage(split)
    assert leaked == 1
    assert not sampled
    assert len(cleaned["test"]) == 1
    assert len(cleaned["train"]) == 1


# --- the fetch query -------------------------------------------------------


def test_fetch_sql_projects_only_the_needed_columns() -> None:
    sql = ds.fetch_sql({"text": ["subject", "body"], "label": "queue"}, fetch_cap=100)
    assert '"subject"' in sql and '"body"' in sql and '"queue"' in sql
    assert "LIMIT 100" in sql
    assert sql.count("FROM data") == 1


def test_fetch_sql_refuses_an_injected_column_name() -> None:
    with pytest.raises(ValueError):
        ds.fetch_sql({"text": ["body"]}, fetch_cap=1)
    with pytest.raises(ValueError):
        ds.fetch_sql({"text": ["body; DROP TABLE x"], "label": "queue"}, fetch_cap=1)


# --- uploads ---------------------------------------------------------------


def test_csv_upload_is_read_through_a_column_mapper(tmp_path: Path) -> None:
    path = tmp_path / "tickets.csv"
    path.write_text(
        "subject,body,queue\n"
        "Card declined,My card was declined, Billing\n"
        "App crash,The app crashes on upload,Technical\n",
        encoding="utf-8",
    )
    rows = ds.read_local_rows(
        path, columns={"text": ["subject", "body"], "label": "queue"}, limit=10
    )
    assert rows[0]["text"] == "Card declined My card was declined"
    assert rows[0]["raw_label"].strip() == "Billing"


def test_upload_type_and_size_are_guarded(tmp_path: Path) -> None:
    bad = tmp_path / "notes.txt"
    bad.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported upload type"):
        ds.read_local_rows(bad, columns={"text": ["a"], "label": "b"}, limit=1)

    missing = tmp_path / "nope.jsonl"
    with pytest.raises(FileNotFoundError):
        ds.read_local_rows(missing, columns={"text": ["a"], "label": "b"}, limit=1)


def test_upload_path_outside_the_upload_root_is_refused(tmp_path: Path) -> None:
    """An `upload:` candidate id is attacker-controlled; it must not read any file."""
    outside = tmp_path / "secret.jsonl"
    outside.write_text(json.dumps({"text": "x", "label": "y"}), encoding="utf-8")
    with pytest.raises(ValueError, match="upload root"):
        ds._resolve_upload_path(outside.as_posix(), settings)


# --- scoring and the catalog ----------------------------------------------


def test_the_catalog_is_honest_about_licence_and_missing_classes() -> None:
    candidates = ds.catalog_candidates(make_spec())
    entry = candidates[0]
    assert entry.commercial_ok is False
    assert entry.approved is True
    joined = " ".join(entry.warnings)
    assert "commercial" in joined.lower()
    assert "Account Access" in joined, "a class with no source rows must be called out"
    assert entry.fit_rationale and "licence" in entry.fit_rationale


def test_a_matching_entry_scores_above_an_unrelated_one() -> None:
    spec = make_spec()
    matching = ds.CATALOG[0]
    unrelated = {
        "id": "hf:example/unrelated",
        "name": "unrelated",
        "keywords": ["weather", "forecast"],
        "task_types": ["summarization"],
        "license": "mit",
        "commercial_ok": True,
    }
    good, _rationale, _coverage = ds.score_candidate(
        matching, spec, label_coverage={"Billing": 10, "Technical": 10, "General": 5}
    )
    poor, _r2, _c2 = ds.score_candidate(unrelated, spec, label_coverage={})
    assert good > poor


def test_discovery_queries_are_derived_from_the_spec() -> None:
    queries = ds.discovery_queries(make_spec())
    assert queries, "a spec with a domain and labels must yield at least one query"
    assert all(query == query.lower() for query in queries)


# --- sufficiency -----------------------------------------------------------


@pytest.mark.parametrize(
    ("coverage", "expected"),
    [
        ({"Billing": 10, "Technical": 10}, "insufficient"),
        ({"Billing": 5000, "Technical": 0, "General": 100}, "marginal"),
        ({"Billing": 20000, "Technical": 1000}, "marginal"),
        ({"Billing": 2000, "Technical": 1800, "General": 2200}, "sufficient"),
    ],
)
def test_sufficiency_verdicts(coverage: dict[str, int], expected: str) -> None:
    verdict, note = ds.sufficiency_verdict(coverage, required=4000, allowed_labels=sorted(coverage))
    assert verdict == expected
    assert note, "a verdict without a reason is not reviewable"


# --- the stage -------------------------------------------------------------


def test_missing_spec_is_a_clear_failure(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ctx.project.spec = None
    result = DatasetStage().run(ctx)
    assert result.status == "failed"
    assert "analyze" in (result.error or "")


def test_unsupported_task_type_is_refused(tmp_path: Path) -> None:
    spec = make_spec().model_copy(update={"task_type": TaskType.summarization})
    result = DatasetStage().run(_ctx(tmp_path, spec=spec))
    assert result.status == "failed"
    assert "summarization" in (result.error or "")


def test_plan_mode_produces_a_reviewable_plan_and_does_not_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Planning is free, so it may run — but nothing is downloaded until approved."""
    monkeypatch.setattr(ds, "hub_candidates", lambda spec, limit=8: ([], None))
    result = DatasetStage().run(_ctx(tmp_path, params={"offline": True}))
    _assert_no_remote_reads(result)
    assert result.status == "succeeded"
    assert result.needs_approval is True
    assert result.blocked_on == ["approve_dataset_build"]

    plan = DatasetPlan.model_validate(result.output)
    assert plan.candidates and plan.selected == [plan.candidates[0].id]
    assert plan.sufficiency in {"marginal", "insufficient"}
    assert plan.sufficiency_notes


def _assert_no_remote_reads(result) -> None:
    """A plan must not have downloaded anything: it reports, it does not fetch."""
    assert result.artifacts and all(
        artifact["kind"] == "dataset_plan" for artifact in result.artifacts
    )


def test_offline_planning_still_reports_the_verified_class_balance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ds, "hub_candidates", lambda spec, limit=8: ([], None))
    result = DatasetStage().run(_ctx(tmp_path, params={"offline": True}))
    candidate = DatasetPlan.model_validate(result.output).candidates[0]
    assert candidate.label_coverage.get("Account Access") == 0
    assert candidate.label_coverage.get("Technical", 0) > 0


def test_build_requires_an_explicit_selection(tmp_path: Path) -> None:
    plan = DatasetPlan(task_type="classification", target_examples=100, selected=[])
    ctx = _ctx(tmp_path, params={"build": True, "plan": plan.model_dump(mode="json")})
    result = DatasetStage().run(ctx)
    assert result.status == "failed"
    assert "selected" in (result.error or "")


def test_build_from_an_upload_writes_splits_a_card_and_artifacts(tmp_path: Path) -> None:
    """The whole contract in one run: files, hashes, provenance and verdict."""
    rows = []
    for label in ("Billing", "Technical"):
        for n in range(12):
            rows.append({"text": f"{label} problem number {n} needs attention", "label": label})
    rows.append({"text": "   ", "label": "Billing"})  # dropped: too short

    ctx = _ctx(
        tmp_path,
        params={
            "build": True,
            "offline": True,
            "selected": [_upload(tmp_path, rows)],
            "columns": {"text": ["text"], "label": "label"},
            "target_examples": 24,
        },
    )
    result = DatasetStage().run(ctx)
    assert result.status == "succeeded", result.error

    card = result.output["card"]
    assert card["total_examples"] == 24
    assert card["total_examples"] == sum(card["splits"].values())
    assert sum(card["label_distribution"].values()) == card["total_examples"]
    assert card["splits"]["train"] + card["splits"]["test"] + card["splits"]["validation"] == 24
    assert card["label_distribution"] == {"Billing": 12, "Technical": 12}
    assert card["synthetic_fraction"] == 0.0

    source = card["sources"][0]
    assert source["source"] == "user_upload"
    assert source["license"]["unknown"] is True

    for name in ("train", "validation", "test"):
        path = ctx.artifact_dir / "dataset" / f"{name}.jsonl"
        assert path.is_file()
        assert card["sha256"][name] == ds.sha256_file(path)
        payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert payload["messages"][0]["role"] == "system"

    kinds = {artifact["kind"] for artifact in result.artifacts}
    assert kinds == {"dataset", "dataset_card"}


def test_a_build_with_an_empty_class_reports_it_and_asks_for_approval(tmp_path: Path) -> None:
    """A missing class is a finding, not something to paper over with another label."""
    rows = [
        {"text": f"billing question number {n} about an invoice", "label": "Billing"}
        for n in range(30)
    ]
    ctx = _ctx(
        tmp_path,
        params={
            "build": True,
            "offline": True,
            "selected": [_upload(tmp_path, rows)],
            "columns": {"text": ["text"], "label": "label"},
            "target_examples": 10,
        },
    )
    result = DatasetStage().run(ctx)
    assert result.status == "succeeded"
    issues = {issue["kind"]: issue for issue in result.output["card"]["issues"]}
    assert "missing_class" in issues
    assert "Technical" in issues["missing_class"]["detail"]
    assert result.output["sufficiency"] == "marginal"


def test_a_build_below_the_target_is_flagged_insufficient_and_blocks(tmp_path: Path) -> None:
    rows = [{"text": f"small row number {n} about a problem", "label": "Billing"} for n in range(3)]
    ctx = _ctx(
        tmp_path,
        params={
            "build": True,
            "offline": True,
            "selected": [_upload(tmp_path, rows)],
            "columns": {"text": ["text"], "label": "label"},
            "target_examples": 1000,
        },
    )
    result = DatasetStage().run(ctx)
    assert result.status == "succeeded"
    assert result.needs_approval is True
    assert result.blocked_on == ["insufficient_data"]
    assert "insufficient" in (result.summary or "").lower()


def test_a_non_catalog_source_without_a_column_mapping_fails_with_advice(
    tmp_path: Path,
) -> None:
    ctx = _ctx(
        tmp_path,
        params={"build": True, "offline": True, "selected": ["hf:someone/unknown-dataset"]},
    )
    result = DatasetStage().run(ctx)
    assert result.status == "failed"
    assert "columns" in (result.error or "")


def test_a_source_that_yields_nothing_fails_instead_of_writing_an_empty_dataset(
    tmp_path: Path,
) -> None:
    ctx = _ctx(
        tmp_path,
        params={
            "build": True,
            "offline": True,
            "selected": [_upload(tmp_path, [])],
            "columns": {"text": ["text"], "label": "label"},
        },
    )
    result = DatasetStage().run(ctx)
    assert result.status == "failed"
    assert "zero usable rows" in (result.error or "")


def test_an_unreadable_source_is_reported_not_raised(tmp_path: Path) -> None:
    ctx = _ctx(
        tmp_path,
        params={
            "build": True,
            "offline": True,
            "selected": [f"upload:{(tmp_path / 'missing.jsonl').as_posix()}"],
            "columns": {"text": ["text"], "label": "label"},
        },
    )
    result = DatasetStage().run(ctx)
    assert result.status == "failed"
    assert "Could not read source" in (result.error or "")


def test_a_reviewed_plan_overrides_the_catalog_mapping(tmp_path: Path) -> None:
    """User edits win: the plan is the approval, so it must be the thing that runs."""
    rows = [
        {"text": "my invoice shows the wrong total again", "label": "Billing and Payments"},
        {"text": "the login page rejects my password", "label": "IT Support"},
    ]
    upload = _upload(tmp_path, rows)
    plan = DatasetPlan(
        task_type="classification",
        target_examples=4,
        selected=[upload],
        label_mapping=[
            {"raw": "Billing and Payments", "mapped": "Billing", "action": "map"},
            {"raw": "IT Support", "mapped": "Account Access", "action": "map"},
        ],
    )
    ctx = _ctx(
        tmp_path,
        params={
            "build": True,
            "plan": plan.model_dump(mode="json"),
            "columns": {"text": ["text"], "label": "label"},
        },
    )
    result = DatasetStage().run(ctx)
    assert result.status == "succeeded", result.error
    assert result.output["card"]["label_distribution"] == {"Billing": 1, "Account Access": 1}


def test_the_stage_is_implemented_and_registered() -> None:
    assert DatasetStage().implemented is True
    assert DatasetStage().requires_approval is False
    assert settings.seed is not None


# --- the remote profile (mocked, never the real Hub) -----------------------


def test_a_remote_profile_fills_in_the_real_label_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ds.hf_cli, "dataset_parquet_urls", lambda repo_id, **kwargs: ["https://x/p.parquet"]
    )
    monkeypatch.setattr(
        ds.parquet,
        "profile",
        lambda urls, **kwargs: {
            "rows": 100,
            "column_names": ["subject", "body", "queue"],
            "label_distribution": {"Billing and Payments": 60, "News": 40},
        },
    )
    candidate = ds.catalog_candidates(make_spec())[0]
    coverage, warnings = ds.profile_candidate(candidate, make_spec(), label_column="queue")
    assert coverage["Billing"] == 60
    assert coverage["General"] == 0, "an unclaimed label must stay empty, not be guessed"
    assert candidate.rows == 100
    assert warnings == []


def test_a_failed_profile_warns_with_the_error_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> list[str]:
        raise RuntimeError("TLS connection reset by peer")

    monkeypatch.setattr(ds.hf_cli, "dataset_parquet_urls", boom)
    coverage, warnings = ds.profile_candidate(
        ds.catalog_candidates(make_spec())[0], make_spec(), label_column="queue"
    )
    assert coverage == {}
    assert warnings and "RuntimeError" in warnings[0]


# --- the approved size is a boundary, not a suggestion ---------------------


def test_the_build_delivers_the_approved_number_and_the_card_adds_up(
    tmp_path: Path,
) -> None:
    """Four times the approved example count is a cost the user did not agree to."""
    rows = [
        {"text": f"billing question number {n} about an invoice", "label": "Billing"}
        for n in range(40)
    ] + [
        {"text": f"technical problem number {n} with the app crash", "label": "Technical"}
        for n in range(40)
    ]
    ctx = _ctx(
        tmp_path,
        params={
            "build": True,
            "offline": True,
            "selected": [_upload(tmp_path, rows)],
            "columns": {"text": ["text"], "label": "label"},
            "target_examples": 30,
        },
    )
    result = DatasetStage().run(ctx)
    assert result.status == "succeeded", result.error
    card = result.output["card"]

    assert card["total_examples"] == 30, "the approved number is the delivered maximum"
    assert card["total_examples"] == sum(card["splits"].values())
    assert sum(card["label_distribution"].values()) == card["total_examples"]
    assert set(card["label_distribution"]) == {"Billing", "Technical"}
    assert card["stats"]["usable_pool"] == 80, "the pool was larger than the approval"
    assert card["stats"]["target_examples"] == 30


def test_select_to_target_keeps_the_class_balance_and_is_deterministic() -> None:
    records = [
        {
            "messages": [{"role": "user", "content": f"{label} row {n}"}],
            "label": label,
            "source": "toy",
            "synthetic": False,
        }
        for label, count in (("Big", 90), ("Small", 10))
        for n in range(count)
    ]
    first = ds.select_to_target(records, target=50, seed=42)
    second = ds.select_to_target(list(reversed(records)), target=50, seed=42)
    assert len(first) == 50
    assert [r["label"] for r in first] == [r["label"] for r in second]
    counts = {"Big": 0, "Small": 0}
    for record in first:
        counts[record["label"]] += 1
    assert counts["Big"] > counts["Small"], "a proportional quota keeps the balance"
    assert counts["Small"] >= 1, "the rare class is not dropped entirely"

    # Nothing to trim: the pool is returned unchanged.
    assert ds.select_to_target(records, target=500, seed=42) == records


# --- the worker path -------------------------------------------------------


def test_the_worker_runs_an_approved_build_from_the_queued_params(client, tmp_path: Path) -> None:
    """The approved parameters must survive the queue, or the gate is decorative.

    This drives the real path: enqueue over HTTP, claim with a worker session, run the
    handler, then read the project back through the API.
    """
    from slmforge.db import SessionLocal
    from slmforge.jobs.handlers import handle_dataset_build
    from slmforge.jobs.queue import JobQueue
    from slmforge.models import JobKind, RunStatus

    created = client.post(
        "/api/projects", json={"prompt": "Route support tickets into four queues."}
    ).json()
    project_id = created["id"]
    assert (
        client.put(
            f"/api/projects/{project_id}/spec",
            json={"spec": make_spec().model_dump(mode="json", by_alias=True)},
        ).status_code
        == 200
    )

    rows = [
        {"text": f"invoice {n} shows the wrong amount please fix", "label": "Billing"}
        for n in range(10)
    ]
    job = client.post(
        f"/api/projects/{project_id}/jobs",
        json={
            "kind": "dataset_build",
            "params": {
                "build": True,
                "offline": True,
                "selected": [_upload(tmp_path, rows)],
                "columns": {"text": ["text"], "label": "label"},
                # Matches the pool size, so this test is about the queue plumbing
                # rather than about trimming.
                "target_examples": 10,
            },
        },
    )
    assert job.status_code == 201, job.text
    job_id = job.json()["id"]

    db = SessionLocal()
    try:
        queue = JobQueue(db, worker_id="test-worker")
        claimed = None
        for _ in range(10):
            candidate = queue.claim_next(kinds=[JobKind.dataset_build])
            if candidate is None:
                break
            if candidate.id == job_id:
                claimed = candidate
                break
            queue.complete(candidate, status=RunStatus.cancelled, error="other test's job")
        assert claimed is not None, "the queued job was never claimable"
        assert handle_dataset_build(claimed, queue) == RunStatus.succeeded
    finally:
        db.close()

    detail = client.get(f"/api/projects/{project_id}").json()
    assert detail["status"] == "data_ready"
    card = detail["dataset_plan"]["card"]
    assert card["total_examples"] == 10
    assert card["splits"]["train"] > 0
    assert {artifact["name"] for artifact in detail["artifacts"]} == {
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "dataset_card.json",
    }
    assert all(artifact["sha256"] for artifact in detail["artifacts"])


def test_a_failed_build_leaves_the_job_failed_in_the_database(client, tmp_path: Path) -> None:
    """A finished job must never keep reading as `running`.

    The handler and the queue must share one session: completing the job through a
    second session silently writes nothing, and a dead job that still says "running"
    is unfalsifiable from the UI.
    """
    from slmforge.jobs.handlers import handle_dataset_build
    from slmforge.jobs.queue import JobQueue
    from slmforge.jobs.worker import Worker
    from slmforge.models import Job, RunStatus

    created = client.post(
        "/api/projects", json={"prompt": "Route support tickets into four queues."}
    ).json()
    project_id = created["id"]
    client.put(
        f"/api/projects/{project_id}/spec",
        json={"spec": make_spec().model_dump(mode="json", by_alias=True)},
    )
    # An unmapped, uncurated source fails before any network call.
    queued = client.post(
        f"/api/projects/{project_id}/jobs",
        json={"kind": "dataset_build", "params": {"build": True, "selected": ["hf:nobody/x"]}},
    ).json()

    worker = Worker(worker_id="test-worker")
    worker.register(JobKind.dataset_build, handle_dataset_build)
    assert worker.run_once() is True

    with SessionLocal() as fresh:
        job = fresh.get(Job, queued["id"])
        assert job is not None
        assert job.status is RunStatus.failed
        assert job.error and "Could not read source" in job.error
        assert job.heartbeat_at is not None, "the lease was claimed"

    detail = client.get(f"/api/projects/{project_id}").json()
    assert any(
        job["status"] == "failed" for job in client.get(f"/api/projects/{project_id}/jobs").json()
    )
    assert detail["status"] == "specified", "a failed build must not advance the project"

    # The queue still works afterwards: a failed job is not a wedged queue.
    assert JobQueue(SessionLocal()).claim_next(kinds=[JobKind.dataset_build]) is None


def test_a_dataset_without_a_parquet_conversion_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ds.hf_cli, "dataset_parquet_urls", lambda repo_id, **kwargs: [])
    coverage, warnings = ds.profile_candidate(
        ds.catalog_candidates(make_spec())[0], make_spec(), label_column="queue"
    )
    assert coverage == {}
    assert "download" in warnings[0]
