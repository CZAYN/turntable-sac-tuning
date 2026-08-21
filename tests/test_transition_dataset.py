from pathlib import Path

import numpy as np

from elc_rl.transition_dataset import (
    TRANSITION_SCHEMA_VERSION,
    generate_transition_dataset,
    validate_transition_archive,
)
from elc_rl.tuning_env import OBSERVATION_KEYS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_small_transition_archive_is_aligned_and_reproducible(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    first_manifest = generate_transition_dataset(
        PROJECT_ROOT,
        first,
        transitions_per_stage=2,
        seed=731,
        stages=("current",),
    )
    generate_transition_dataset(
        PROJECT_ROOT,
        second,
        transitions_per_stage=2,
        seed=731,
        stages=("current",),
    )
    assert first_manifest["transition_count"] == 2
    assert first.read_bytes() == second.read_bytes()
    validation = validate_transition_archive(first)
    assert validation["observation_keys"] == sorted(OBSERVATION_KEYS)
    assert validation["stage_counts"]["current"] == 2
    with np.load(first, allow_pickle=False) as data:
        assert int(data["schema_version"]) == TRANSITION_SCHEMA_VERSION
        assert "observation__frf_context" not in data.files
        assert data["observation__friction_context"].shape == (2, 6)
        assert data["observation__performance_metrics"].shape == (2, 18)
        assert data["performance_metric_names"].shape == (18,)
        assert "observation__metrics" not in data.files
        assert "observation__time_metrics" not in data.files
        assert "frequency_stage_cost" in data.files
        assert "time_stage_cost" in data.files
        assert "total_stage_cost" in data.files
        assert np.all(data["action"][:, 3:8] == 0.0)
