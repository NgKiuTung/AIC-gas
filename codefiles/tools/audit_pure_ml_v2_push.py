"""Inspect staged release content without publishing raw competition data."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = "results/experiments/pure_ml_candidate_v2/20260905T071738Z/"


def main():
    names = subprocess.check_output(["git", "diff", "--cached", "--name-only", "-z"], cwd=ROOT).decode().split("\0")
    archive = json.loads((ROOT / RUN / "artifact_hashes.json").read_text(encoding="utf-8"))
    records = []
    models = 0
    for name in filter(None, names):
        assert not name.startswith("dataset/"), name
        assert not name.endswith((".zip", ".gz")), name
        assert Path(name).name not in {"input.csv", "result.csv", ".env"}, name
        data = subprocess.check_output(["git", "show", ":" + name], cwd=ROOT)
        assert len(data) < 50 * 1024**2, name
        secret = re.search(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)", data)
        assert secret is None, f"Possible credential in {name}; inspect privately"
        digest = hashlib.sha256(data).hexdigest()
        if name.startswith(RUN) and Path(name).name.startswith("target_"):
            assert digest == archive[name], name
            models += 1
        records.append(dict(path=name, bytes=len(data), staged_sha256=digest))
    assert models == 64, models
    result = dict(passed=True, model_count=models, staged_file_count=len(records),
                  staged_bytes=sum(r["bytes"] for r in records),
                  raw_data_and_zip_excluded=True, credential_pattern_scan_passed=True,
                  model_archive_hashes_match=True, files=records)
    output = ROOT / "results/repository_validation/pure_ml_v2_push_audit.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}))


if __name__ == "__main__":
    main()
