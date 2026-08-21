from pathlib import Path

import numpy as np

from elc_rl.performance_targets import (
    frequency_normalized_errors,
    load_controller_performance_targets,
    time_normalized_errors,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_literal_controller_performance_target_table_is_loaded_with_units():
    targets = load_controller_performance_targets(PROJECT_ROOT)
    expected = {
        "current": (1500.0, 5.0, 65.0, 0.20, 0.05, 0.1),
        "speed": (100.0, 5.0, 65.0, 0.50, 0.5, 1.0),
        "position": (20.0, 2.0, 70.0, 0.20, 0.5, 1.0),
    }
    for loop, values in expected.items():
        target = targets.loop(loop)
        actual = (
            target.bandwidth_hz,
            target.minimum_gain_margin_db,
            target.minimum_phase_margin_deg,
            target.maximum_overshoot_ratio,
            target.maximum_rise_time_s,
            target.maximum_settling_time_s,
        )
        assert actual == values


def test_six_metric_error_directions_match_target_and_limit_semantics():
    targets = load_controller_performance_targets(PROJECT_ROOT)
    target = targets.loop("speed")
    frequency = frequency_normalized_errors(
        bandwidth_hz=target.bandwidth_hz,
        gain_margin_db=target.minimum_gain_margin_db + 1.0,
        phase_margin_deg=target.minimum_phase_margin_deg + 1.0,
        target=target,
        bandwidth_relative_scale=targets.cost.bandwidth_relative_scale,
        invalid_error=targets.cost.invalid_normalized_error,
    )
    assert frequency == {"bandwidth": 0.0, "gain_margin": 0.0, "phase_margin": 0.0}
    assert frequency_normalized_errors(
        bandwidth_hz=80.0,
        gain_margin_db=4.0,
        phase_margin_deg=64.0,
        target=target,
        bandwidth_relative_scale=targets.cost.bandwidth_relative_scale,
        invalid_error=targets.cost.invalid_normalized_error,
    )["bandwidth"] < 0.0

    within = time_normalized_errors(
        overshoot_ratio=0.1,
        rise_time_s=0.1,
        settling_time_s=0.2,
        target=target,
        invalid_error=targets.cost.invalid_normalized_error,
    )
    assert within == {"overshoot": 0.0, "rise_time": 0.0, "settling_time": 0.0}
    exceeded = time_normalized_errors(
        overshoot_ratio=1.0,
        rise_time_s=1.0,
        settling_time_s=2.0,
        target=target,
        invalid_error=targets.cost.invalid_normalized_error,
    )
    assert np.allclose(list(exceeded.values()), [1.0, 1.0, 1.0])
