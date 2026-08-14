from gas_power.settings import PATHS
from gas_power.utils.pipeline import run_legacy_scripts


def test_optimization_pipeline_can_be_planned() -> None:
    records = run_legacy_scripts(["38_optimization_constraint_audit.py"], "test_dry_run", dry_run=True)
    assert records == [{"script": "38_optimization_constraint_audit.py", "status": "planned"}]
    assert (PATHS.results / "pipeline_manifests" / "test_dry_run.json").is_file()

