from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.final_test_evaluator import evaluate_locked_final_candidate  # noqa: E402
from elc_rl.physics_test_dataset import (  # noqa: E402
    load_final_test_spec,
    verify_physics_test_dependencies,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consume the sealed test set once for one locked final candidate."
    )
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--candidate-lock", type=Path, required=True)
    arguments = parser.parse_args()

    candidate = arguments.candidate.resolve()
    candidate_lock_path = arguments.candidate_lock.resolve()
    lock = json.loads(candidate_lock_path.read_text(encoding="utf-8"))
    spec = load_final_test_spec(PROJECT_ROOT)
    test_manifest = verify_physics_test_dependencies(PROJECT_ROOT)
    if lock.get("status") != "training_complete_candidate_locked":
        raise ValueError("candidate lock status is invalid")
    if lock.get("test_suite_id") != spec["test_suite_id"]:
        raise ValueError("candidate lock targets a different test suite")
    if lock.get("candidate_file_sha256") != _sha256(candidate):
        raise ValueError("candidate changed after it was locked")
    if lock.get("test_ensemble_sha256") != test_manifest["test_ensemble_sha256"]:
        raise ValueError("candidate lock targets a different test ensemble")
    if not all(bool(value) for value in lock["declarations"].values()):
        raise ValueError("candidate lock declarations are incomplete")

    output_policy = spec["output_policy"]
    report_path = PROJECT_ROOT / Path(output_policy["final_report"])
    marker_path = PROJECT_ROOT / Path(output_policy["consumption_marker"])
    if report_path.parent != marker_path.parent:
        raise ValueError("final report and consumption marker must share a directory")
    output_dir = report_path.parent
    if report_path.exists() or marker_path.exists():
        raise PermissionError(
            "final-test suite has already been consumed; repeat evaluation is forbidden"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    marker = {
        "schema_version": 2,
        "status": "started_test_suite_consumed",
        "test_suite_id": spec["test_suite_id"],
        "candidate_file_sha256": _sha256(candidate),
        "candidate_lock_sha256": _sha256(candidate_lock_path),
        "test_ensemble_sha256": test_manifest["test_ensemble_sha256"],
        "performance_targets_sha256": test_manifest[
            "performance_targets_sha256"
        ],
        "started_at_utc": started_at,
    }
    marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with np.load(candidate, allow_pickle=False) as archive:
        parameters = np.asarray(archive["parameters"], dtype=np.float64)
    report = evaluate_locked_final_candidate(PROJECT_ROOT, parameters)
    report["provenance"] = {
        "candidate_path": str(candidate),
        "candidate_file_sha256": marker["candidate_file_sha256"],
        "candidate_lock_path": str(candidate_lock_path),
        "candidate_lock_sha256": marker["candidate_lock_sha256"],
        "test_ensemble_sha256": marker["test_ensemble_sha256"],
        "performance_targets_sha256": marker[
            "performance_targets_sha256"
        ],
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    marker.update(
        {
            "status": "completed_test_suite_consumed",
            "completed_at_utc": report["provenance"]["completed_at_utc"],
            "report_sha256": _sha256(report_path),
            "overall_pass": bool(report["overall_pass"]),
        }
    )
    marker_path.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "overall_pass": report["overall_pass"],
                "summary": report["summary"],
                "report": str(report_path),
                "policy": "test result is final and must not be used for retraining",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
