from gas_power.utils.pipeline import run_legacy_scripts


def test_optimization_pipeline_can_be_planned(tmp_path) -> None:
    records = run_legacy_scripts(
        ["38_optimization_constraint_audit.py"],
        "test_dry_run",
        dry_run=True,
        artifact_root=tmp_path,
    )
    assert records == [{"script": "38_optimization_constraint_audit.py", "status": "planned"}]
    assert (tmp_path / "pipeline_manifests" / "test_dry_run.json").is_file()
