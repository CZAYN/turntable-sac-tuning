from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from elc_rl.controller_parameters import load_physics_controller_parameter_space
from elc_rl.physics_motor_model import load_physics_motor_config
from scripts.scan_multirate_current_feasibility import (
    _candidate_rank,
    _frequency_metrics,
    config_for_sample_rate,
    current_corner_motors,
    current_frequency_grid,
    current_uncertainty_grid,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_sample_rate_override_is_in_memory_and_preserves_fixed_delays():
    base = load_physics_motor_config(PROJECT_ROOT)
    original_payload = json.dumps(base.payload, sort_keys=True)
    original_outer_periods = {
        loop: base.sample_period_s_for(loop) for loop in ("speed", "position")
    }
    for sample_rate_hz in (5000.0, 10000.0, 15000.0, 20000.0, 40000.0):
        temporary = config_for_sample_rate(base, sample_rate_hz)
        assert temporary.sample_period_s_for("current") == 1.0 / sample_rate_hz
        assert {
            loop: temporary.sample_period_s_for(loop)
            for loop in ("speed", "position")
        } == original_outer_periods
        assert temporary.nominal.current_delay_s == 200e-6
        assert temporary.nominal.speed_measurement_delay_s == 200e-6
        assert temporary.nominal.position_measurement_delay_s == 200e-6
        assert temporary.derivative_filter_s["current"] == 200e-6
    assert json.dumps(base.payload, sort_keys=True) == original_payload
    assert base.sample_period_s_for("current") == 25e-6


def test_frequency_grid_tracks_nyquist_strictly():
    base = load_physics_motor_config(PROJECT_ROOT)
    expected_upper = (2250.0, 4500.0, 6750.0, 9000.0, 18000.0)
    for sample_rate_hz, upper_hz in zip(
        (5000.0, 10000.0, 15000.0, 20000.0, 40000.0), expected_upper
    ):
        temporary = config_for_sample_rate(base, sample_rate_hz)
        grid = current_frequency_grid(temporary, 128)
        assert np.isclose(grid[-1], upper_hz)
        assert np.all(grid > 0.0)
        assert np.all(grid < 0.5 / temporary.sample_period_s_for("current"))


def test_current_corner_and_dense_grids_change_only_l_r_and_delay():
    config = load_physics_motor_config(PROJECT_ROOT)
    corners = current_corner_motors(config)
    dense = current_uncertainty_grid(config, levels=5)
    assert len(corners) == 8
    assert len({name for name, _ in corners}) == 8
    assert len(dense) == 125
    assert len({motor.as_array().tobytes() for _, motor in dense}) == 125

    unchanged = set(config.nominal.__dataclass_fields__) - {
        "inductance_h",
        "resistance_ohm",
        "current_delay_s",
    }
    for _, motor in corners:
        for name in unchanged:
            assert getattr(motor, name) == getattr(config.nominal, name)
        assert motor.inductance_h in {
            config.nominal.inductance_h * 0.95,
            config.nominal.inductance_h * 1.05,
        }
        assert motor.resistance_ohm in {
            config.nominal.resistance_ohm * 0.95,
            config.nominal.resistance_ohm * 1.05,
        }
        assert motor.current_delay_s in {
            config.nominal.current_delay_s * 0.90,
            config.nominal.current_delay_s * 1.10,
        }


def test_initial_pidf_is_numerically_evaluable_at_all_sample_rates():
    base = load_physics_motor_config(PROJECT_ROOT)
    parameters = load_physics_controller_parameter_space(PROJECT_ROOT).initial
    for sample_rate_hz in (5000.0, 10000.0, 15000.0, 20000.0, 40000.0):
        temporary = config_for_sample_rate(base, sample_rate_hz)
        metrics = _frequency_metrics(
            temporary,
            temporary.nominal,
            parameters,
            points=128,
        )
        assert np.isfinite(float(metrics["bandwidth_hz"]))
        assert np.isfinite(float(metrics["gain_margin_db"]))
        assert np.isfinite(float(metrics["phase_margin_deg"]))


def test_candidate_rank_never_uses_validation_diagnostics():
    group = {
        "model_count": 1,
        "all_six_pass_count": 1,
        "maximum_constraint_violation": 0.0,
        "mean_smooth_cost": 1.0,
    }
    audit = {
        "groups": {
            "nominal": dict(group),
            "corners_8": dict(group),
            "dense_grid_125": dict(group),
            "training_40": dict(group),
            "validation_16": {
                "model_count": 16,
                "all_six_pass_count": 0,
                "maximum_constraint_violation": 1e6,
                "mean_smooth_cost": 1e6,
            },
        }
    }
    assert _candidate_rank(audit) == (0.0, 0.0, 4.0)
