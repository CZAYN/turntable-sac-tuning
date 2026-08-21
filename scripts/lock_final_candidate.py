from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.controller_parameters import (  # noqa: E402
    load_physics_controller_parameter_space,
)
from elc_rl.physics_test_dataset import (  # noqa: E402
    FINAL_TEST_MANIFEST_RELATIVE_PATH,
    load_final_test_spec,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lock one completed-training candidate before final testing."
    )
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--declare-training-complete",
        action="store_true",
        help="required declaration that training, validation and selection are finished",
    )
    arguments = parser.parse_args()
    if not arguments.declare_training_complete:
        raise SystemExit("refusing to lock: --declare-training-complete is required")

    candidate = arguments.candidate.resolve()
    output = (
        arguments.output.resolve()
        if arguments.output is not None
        else candidate.with_name(f"{candidate.stem}_training_lock.json")
    )
    if output.exists():
        raise FileExistsError(f"candidate lock already exists: {output}")
    spec = load_final_test_spec(PROJECT_ROOT)
    final_report = PROJECT_ROOT / spec["output_policy"]["final_report"]
    if final_report.exists():
        raise PermissionError("final test has already been consumed")

    with np.load(candidate, allow_pickle=False) as archive:
        parameters = np.asarray(archive["parameters"], dtype=np.float64)
    space = load_physics_controller_parameter_space(PROJECT_ROOT)
    space.normalize(parameters)
    test_manifest = json.loads(
        (PROJECT_ROOT / FINAL_TEST_MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    lock = {
        "schema_version": 2,
        "status": "training_complete_candidate_locked",
        "backend": "physics",
        "test_suite_id": spec["test_suite_id"],
        "candidate_path": str(candidate),
        "candidate_file_sha256": _sha256(candidate),
        "parameter_names": list(space.names),
        "parameters_sha256": hashlib.sha256(
            np.ascontiguousarray(parameters, dtype=np.float64).tobytes()
        ).hexdigest(),
        "test_ensemble_sha256": test_manifest["test_ensemble_sha256"],
        "declarations": {
            "training_complete": True,
            "validation_complete": True,
            "candidate_selection_complete": True,
            "test_results_will_not_be_used_to_select_or_retrain": True,
        },
    }
    output.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(lock, ensure_ascii=False, indent=2))
