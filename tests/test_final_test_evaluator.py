from pathlib import Path
from types import SimpleNamespace

import numpy as np

from elc_rl import final_test_evaluator
from elc_rl.final_test_evaluator import _evaluate_six_metrics
from elc_rl.performance_targets import load_controller_performance_targets
from elc_rl.physics_motor_model import load_physics_motor_config
from elc_rl.physics_test_dataset import OFFICIAL_METRICS_PER_LOOP


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _frequency(**overrides):
    values = {
        "closed_loop_bandwidth_hz": 1500.0,
        "gain_margin_db": 5.0,
        "phase_margin_deg": 65.0,
        "valid": True,
    }
    values.update(overrides)
    return values


def _time(**overrides):
    values = {
        "overshoot_ratio": 0.2,
        "rise_time_s": 0.05,
        "settling_time_s": 0.1,
        "valid": True,
    }
    values.update(overrides)
    return values


def test_exact_current_table_targets_pass_and_report_exactly_six_actuals():
    target = load_controller_performance_targets(PROJECT_ROOT).loop("current")
    report = _evaluate_six_metrics(
        _frequency(),
        _time(),
        target,
        bandwidth_tolerance_fraction=0.1,
    )
    assert report["pass"]
    assert report["failures"] == []
    assert tuple(report["actual"]) == OFFICIAL_METRICS_PER_LOOP


def test_each_six_metric_boundary_contributes_its_own_failure():
    target = load_controller_performance_targets(PROJECT_ROOT).loop("current")
    report = _evaluate_six_metrics(
        _frequency(
            closed_loop_bandwidth_hz=1665.1,
            gain_margin_db=4.9,
            phase_margin_deg=64.9,
        ),
        _time(
            overshoot_ratio=0.201,
            rise_time_s=0.051,
            settling_time_s=0.101,
        ),
        target,
        bandwidth_tolerance_fraction=0.1,
    )
    assert not report["pass"]
    assert set(report["failures"]) == set(OFFICIAL_METRICS_PER_LOOP)


def test_invalid_measurements_fail_without_adding_a_legacy_metric():
    target = load_controller_performance_targets(PROJECT_ROOT).loop("current")
    report = _evaluate_six_metrics(
        _frequency(valid=False),
        _time(valid=False),
        target,
        bandwidth_tolerance_fraction=0.1,
    )
    assert not report["pass"]
    assert report["failures"] == [
        "invalid_frequency_measurement",
        "invalid_time_measurement",
    ]


def test_final_evaluator_has_no_legacy_objective_path():
    text = (PROJECT_ROOT / "src" / "elc_rl" / "final_test_evaluator.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "_disturbance_metrics",
        "_control_metrics",
        "sensitivity_peak",
        "crossover_ratio",
        '"iae"',
        "steady_state_error",
    ):
        assert forbidden not in text


def test_unsettled_step_response_is_an_invalid_final_time_measurement(monkeypatch):
    monkeypatch.setattr(
        final_test_evaluator,
        "simulate_scenario",
        lambda *args, **kwargs: SimpleNamespace(terminated=False),
    )
    monkeypatch.setattr(
        final_test_evaluator,
        "_reference_metrics",
        lambda trace: {
            "overshoot_ratio": 0.0,
            "rise_time_s": 0.01,
            "settling_time_s": 0.1,
            "reference_metric_valid": True,
            "reached_10_percent": True,
            "reached_90_percent": True,
            "settled": False,
        },
    )
    result = final_test_evaluator._time_metrics(
        PROJECT_ROOT,
        load_physics_motor_config(PROJECT_ROOT).nominal,
        np.zeros(11, dtype=np.float64),
        seed=0,
    )
    assert all(not loop["valid"] for loop in result.values())
