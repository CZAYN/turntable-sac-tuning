from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from elc_rl.frozen_policy import inspect_frozen_policy_run
from elc_rl.sac_training import TRAINING_PROTOCOL_SCHEMA_VERSION
from elc_rl.tuning_env import STAGE_ORDER


def _make_policy_run(root: Path) -> Path:
    run = root / "seed_7"
    models = run / "models"
    models.mkdir(parents=True)
    completed = []
    for index, stage in enumerate(STAGE_ORDER, 1):
        relative = f"models/stage_{index:02d}_{stage}_final.zip"
        (run / relative).write_bytes(f"model-{stage}".encode())
        completed.append(
            {
                "stage": stage,
                "model": relative,
                "selected_parameters": np.arange(11, dtype=float).tolist(),
            }
        )
    fingerprint = "f" * 64
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": TRAINING_PROTOCOL_SCHEMA_VERSION,
                "run_kind": "formal_training",
                "seed": 7,
                "configuration_path": "/server/config/ablation_difficulty.json",
                "configuration": {},
                "initial_parameters": np.arange(11, dtype=float).tolist(),
                "training_inputs": {"fingerprint": fingerprint},
            }
        ),
        encoding="utf-8",
    )
    (run / "seed_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "seed": 7,
                "input_fingerprint": fingerprint,
                "completed_stages": completed,
            }
        ),
        encoding="utf-8",
    )
    return run


def test_inspect_frozen_policy_run_checks_provenance_and_hashes(
    tmp_path: Path,
) -> None:
    run = _make_policy_run(tmp_path)

    result = inspect_frozen_policy_run(run)

    assert result["seed"] == 7
    assert tuple(item["stage"] for item in result["stages"]) == STAGE_ORDER
    first = run / "models/stage_01_current_final.zip"
    assert result["stages"][0]["model_sha256"] == hashlib.sha256(
        first.read_bytes()
    ).hexdigest()


def test_inspect_frozen_policy_run_rejects_model_path_escape(
    tmp_path: Path,
) -> None:
    run = _make_policy_run(tmp_path)
    summary_path = run / "seed_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    escaped = run.parent / "escaped.zip"
    escaped.write_bytes(b"outside")
    summary["completed_stages"][0]["model"] = "../escaped.zip"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="outside the run models directory"):
        inspect_frozen_policy_run(run)
