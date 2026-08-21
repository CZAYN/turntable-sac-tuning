import json
from pathlib import Path

import numpy as np
import pytest

import elc_rl.discrete_loop_model as discrete
from elc_rl.physics_motor_model import load_physics_motor_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    config = load_physics_motor_config(PROJECT_ROOT)
    payload = json.loads(
        (PROJECT_ROOT / "data/processed/controller_parameter_space.json").read_text(
            encoding="utf-8"
        )
    )
    parameters = np.asarray(
        [spec["initial"] for spec in payload["parameters"]], dtype=np.float64
    )
    return config, config.nominal, parameters


def test_multirate_models_use_declared_loop_clocks_and_nyquist_limits():
    config, motor, parameters = _fixture()
    models = discrete.build_discrete_loop_models(config, motor, parameters)

    assert config.controller_update_ratios == {
        "current": 1,
        "speed": 8,
        "position": 8,
    }
    assert models["current"].sample_period_s == pytest.approx(25e-6)
    assert models["speed"].sample_period_s == pytest.approx(200e-6)
    assert models["position"].sample_period_s == pytest.approx(200e-6)

    assert np.isfinite(
        models["current"].open_loop_response(np.asarray([19_999.0]))
    ).all()
    assert np.isfinite(
        models["speed"].open_loop_response(np.asarray([2_499.0]))
    ).all()
    with pytest.raises(ValueError, match="strictly below Nyquist"):
        models["current"].open_loop_response(np.asarray([20_000.0]))
    with pytest.raises(ValueError, match="strictly below Nyquist"):
        models["speed"].open_loop_response(np.asarray([2_500.0]))


def test_speed_and_position_lifts_execute_eight_current_motor_substeps(monkeypatch):
    config, motor, parameters = _fixture()
    calls = {"current_pid": 0, "speed_pid": 0, "position_pid": 0, "friction": 0}
    original_pid = discrete._pid_step
    original_friction = discrete._friction_step

    pid_values = {
        loop: discrete._pid_values(parameters, loop)
        for loop in ("current", "speed", "position")
    }

    def counting_pid(error, pid, *args):
        for loop, expected in pid_values.items():
            if pid == expected:
                calls[f"{loop}_pid"] += 1
                break
        return original_pid(error, pid, *args)

    def counting_friction(*args):
        calls["friction"] += 1
        return original_friction(*args)

    monkeypatch.setattr(discrete, "_pid_step", counting_pid)
    monkeypatch.setattr(discrete, "_friction_step", counting_friction)

    discrete._speed_step(
        config,
        motor,
        parameters,
        np.zeros(len(discrete.LOOP_STATE_NAMES["speed"])),
        1e-7,
    )
    assert calls == {
        "current_pid": 8,
        "speed_pid": 1,
        "position_pid": 0,
        "friction": 8,
    }

    calls.update({name: 0 for name in calls})
    discrete._position_step(
        config,
        motor,
        parameters,
        np.zeros(len(discrete.LOOP_STATE_NAMES["position"])),
        1e-7,
    )
    assert calls == {
        "current_pid": 8,
        "speed_pid": 1,
        "position_pid": 1,
        "friction": 8,
    }


def test_lifted_models_embed_dobc_only_in_speed_and_position():
    config, motor, parameters = _fixture()
    changed = parameters.copy()
    changed[6] = 0.0
    changed[7] *= 2.0
    baseline = discrete.build_discrete_loop_models(config, motor, parameters)
    without_dobc = discrete.build_discrete_loop_models(config, motor, changed)
    frequency_hz = np.geomspace(0.2, 500.0, 128)

    assert np.array_equal(
        baseline["current"].state_matrix,
        without_dobc["current"].state_matrix,
    )
    for loop in ("speed", "position"):
        difference = np.max(
            np.abs(
                baseline[loop].open_loop_response(frequency_hz)
                - without_dobc[loop].open_loop_response(frequency_hz)
            )
        )
        assert difference > 1e-6


def test_multirate_lift_rejects_non_integer_clock_relationship():
    with pytest.raises(ValueError, match="integer update ratios"):
        discrete._integer_ratio(200e-6, 30e-6)
