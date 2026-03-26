from __future__ import annotations

from pathlib import Path

from knowledge_guided_symbolic_alpha.runtime import ensure_run_output_dirs, run_preflight, write_run_manifest


def test_core_preflight_passes_with_required_modules() -> None:
    report = run_preflight("core")

    assert report.ok
    assert {check.name for check in report.checks} == {"numpy", "pandas", "yaml"}


def test_wrds_preflight_reports_missing_env_vars(monkeypatch) -> None:
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    monkeypatch.delenv("WRDS_PASSWORD", raising=False)

    report = run_preflight("wrds")

    statuses = {check.name: check.status for check in report.checks}
    assert statuses["WRDS_USERNAME"] == "missing"
    assert statuses["WRDS_PASSWORD"] == "missing"


def test_train_preflight_allows_torch_timeout_when_module_exists(monkeypatch) -> None:
    monkeypatch.setattr(
        "knowledge_guided_symbolic_alpha.runtime.preflight.probe_torch_health",
        lambda: {"ok": False, "status": "timeout", "message": "probe timed out"},
    )

    report = run_preflight("train")

    statuses = {check.name: check.status for check in report.checks}
    required = {check.name: check.required for check in report.checks}
    assert report.ok
    assert statuses["torch"] == "ok"
    assert statuses["torch_health"] == "timeout"
    assert required["torch"] is True
    assert required["torch_health"] is False


def test_run_manifest_is_written_under_run_scoped_output_tree(tmp_path: Path) -> None:
    output_dirs = ensure_run_output_dirs(tmp_path, "smoke run")

    manifest_path = write_run_manifest(
        output_dirs,
        script_name="tests/smoke.py",
        profile="core",
        preflight={"profile": "core", "ok": True, "checks": []},
        config_paths={"data_config": "configs/us_equities_smoke.yaml"},
        dataset_name="us_equities",
        subset="liquid150_2010_2025",
        seed=7,
    )

    assert output_dirs["root"] == tmp_path / "smoke-run"
    assert manifest_path == output_dirs["root"] / "run_manifest.json"
    assert manifest_path.exists()
