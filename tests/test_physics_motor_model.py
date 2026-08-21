from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from elc_rl.controller_parameters import (
    derive_physics_controller_initials,
    load_physics_controller_parameter_space,
)
from elc_rl.physics_motor_model import (
    MODEL_PARAMETER_NAMES,
    PhysicsMotorConfig,
    build_physics_motor_ensemble,
    clear_physics_motor_config_cache,
    load_physics_motor_config,
    load_physics_motor_ensemble,
    simulate_scenario,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_mentor_model_values_and_units_are_preserved():
    config = load_physics_motor_config(PROJECT_ROOT)
    motor = config.nominal
    assert motor.inductance_h == 0.002707
    assert motor.resistance_ohm == 4.993
    assert motor.inertia_kg_m2 == 0.00221
    assert motor.torque_constant_nm_per_a == 0.1633
    assert config.sample_periods_s == {
        "current": 25e-6,
        "speed": 200e-6,
        "position": 200e-6,
    }
    assert config.sample_period_s == 25e-6
    assert config.base_sample_period_s == 25e-6
    assert config.controller_update_ratios == {
        "current": 1,
        "speed": 8,
        "position": 8,
    }
    assert motor.current_delay_s == 200e-6
    assert motor.speed_measurement_delay_s == 200e-6
    assert motor.position_measurement_delay_s == 200e-6
    assert config.derivative_filter_s["current"] == 200e-6
    assert np.isclose(motor.electrical_time_constant_s, 0.002707 / 4.993)
    assert np.isclose(motor.mechanical_time_constant_s, 0.00221 / 0.0237)


def test_single_rate_schema_is_rejected_with_migration_message():
    base = load_physics_motor_config(PROJECT_ROOT)
    payload = json.loads(json.dumps(base.payload))
    payload["schema_version"] = 2
    payload["sample_period_s"] = 200e-6
    payload.pop("sample_periods_s")
    stale = replace(base, payload=payload)

    with pytest.raises(ValueError, match="single-rate.*schema 3"):
        stale.validate()


def test_single_rate_ensemble_state_is_rejected_before_use(tmp_path):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    np.savez_compressed(
        processed / "physics_motor_ensemble.npz",
        schema_version=np.asarray(2, dtype=np.int16),
    )

    with pytest.raises(ValueError, match="stale.*schema 3"):
        load_physics_motor_ensemble(tmp_path)


def test_outer_periods_must_be_integer_base_step_multiples():
    base = load_physics_motor_config(PROJECT_ROOT)
    payload = json.loads(json.dumps(base.payload))
    payload["sample_periods_s"]["position"] = 190e-6
    invalid = replace(base, payload=payload)

    with pytest.raises(ValueError, match="integer multiple"):
        invalid.validate()


def test_multirate_kernel_holds_outer_command_for_eight_current_steps():
    base = load_physics_motor_config(PROJECT_ROOT)
    payload = json.loads(json.dumps(base.payload))
    payload["scenarios"]["speed_duration_s"] = 0.001
    payload["scenarios"]["position_duration_s"] = 0.001
    config = replace(base, payload=payload)
    config.validate()
    parameters = np.asarray(
        [
            1.0,
            0.0,
            0.0,
            0.05,
            0.1,
            0.0,
            0.0,
            0.002,
            0.1,
            1.0,
            0.0,
        ],
        dtype=np.float64,
    )

    trace = simulate_scenario(config, config.nominal, parameters, "speed")

    ratio = config.controller_update_ratios["speed"]
    assert ratio == 8
    for start in range(0, trace.current_reference_a.size, ratio):
        held = trace.current_reference_a[start : start + ratio]
        assert np.all(held == held[0])
    assert np.any(np.diff(trace.current_reference_a[::ratio]) != 0.0)
    assert np.any(np.diff(trace.voltage_v[:ratio]) != 0.0)

    position_trace = simulate_scenario(
        config, config.nominal, parameters, "position"
    )
    for start in range(0, position_trace.speed_reference_rad_s.size, ratio):
        held_speed = position_trace.speed_reference_rad_s[start : start + ratio]
        held_current = position_trace.current_reference_a[start : start + ratio]
        assert np.all(held_speed == held_speed[0])
        assert np.all(held_current == held_current[0])
    assert position_trace.speed_reference_rad_s[0] > 0.0
    assert position_trace.current_reference_a[0] > 0.0


def test_provisional_lugre_configuration_is_explicit_and_bounded():
    config = load_physics_motor_config(PROJECT_ROOT)
    motor = config.nominal
    assert len(MODEL_PARAMETER_NAMES) == 13
    assert MODEL_PARAMETER_NAMES[-2:] == (
        "coulomb_friction_nm",
        "static_friction_nm",
    )
    assert config.payload["schema_version"] == 3
    assert config.active_friction_model == "lugre"
    assert config.friction_model["stribeck_exponent"] == 2.0
    assert config.friction_model["initial_bristle_state_rad"] == 0.0
    assert motor.coulomb_friction_nm == 0.015
    assert motor.static_friction_nm == 0.020
    assert config.uncertainty_fraction["coulomb_friction_nm"] == 0.1
    assert config.uncertainty_fraction["static_friction_nm"] == 0.1
    assert "provisional" in config.friction_model["parameter_status"][
        "coulomb_friction_nm"
    ]
    assert "provisional" in config.friction_model["parameter_status"][
        "static_friction_nm"
    ]
    assert (
        motor.static_friction_nm
        * (1.0 - config.uncertainty_fraction["static_friction_nm"])
        > motor.coulomb_friction_nm
        * (1.0 + config.uncertainty_fraction["coulomb_friction_nm"])
    )
    motor.validate(friction_model="lugre")


def test_lugre_friction_level_constraints_are_enforced():
    nominal = load_physics_motor_config(PROJECT_ROOT).nominal
    identified = replace(
        nominal,
        coulomb_friction_nm=0.02,
        static_friction_nm=0.03,
    )
    identified.validate(friction_model="lugre")
    with pytest.raises(ValueError, match="static_friction_nm must be"):
        replace(
            identified,
            coulomb_friction_nm=0.04,
            static_friction_nm=0.03,
        ).validate(friction_model="lugre")


def test_identified_lugre_configuration_runs_through_full_scenario_wrapper():
    base = load_physics_motor_config(PROJECT_ROOT)
    motor = replace(
        base.nominal,
        coulomb_friction_nm=0.04,
        static_friction_nm=0.07,
    )
    payload = json.loads(json.dumps(base.payload))
    payload["friction_model"]["active"] = "lugre"
    payload["friction_model"]["parameter_status"]["coulomb_friction_nm"] = (
        "synthetic_test_fixture"
    )
    payload["friction_model"]["parameter_status"]["static_friction_nm"] = (
        "synthetic_test_fixture"
    )
    payload["nominal_parameters"]["coulomb_friction_nm"] = (
        motor.coulomb_friction_nm
    )
    payload["nominal_parameters"]["static_friction_nm"] = motor.static_friction_nm
    active = PhysicsMotorConfig(
        project_root=base.project_root,
        path=base.path,
        payload=payload,
        nominal=motor,
    )
    active.validate()
    space = load_physics_controller_parameter_space(PROJECT_ROOT)

    trace = simulate_scenario(
        active,
        motor,
        space.initial,
        "speed",
        scenario_overrides={"speed_reference_rad_s": 0.25},
    )

    assert not trace.terminated
    assert np.isfinite(trace.friction_torque_nm).all()
    assert np.isfinite(trace.bristle_state_rad).all()
    assert np.isfinite(trace.bristle_rate_rad_s).all()
    assert np.max(np.abs(trace.bristle_state_rad)) > 0.0
    viscous_only = motor.viscous_friction_nm_s_per_rad * trace.speed_rad_s
    assert not np.allclose(trace.friction_torque_nm, viscous_only)


def test_active_lugre_rejects_initial_bristle_state_above_static_limit():
    base = load_physics_motor_config(PROJECT_ROOT)
    motor = replace(
        base.nominal,
        coulomb_friction_nm=0.04,
        static_friction_nm=0.07,
    )
    payload = json.loads(json.dumps(base.payload))
    payload["friction_model"]["active"] = "lugre"
    payload["friction_model"]["parameter_status"]["coulomb_friction_nm"] = (
        "synthetic_test_fixture"
    )
    payload["friction_model"]["parameter_status"]["static_friction_nm"] = (
        "synthetic_test_fixture"
    )
    payload["friction_model"]["initial_bristle_state_rad"] = (
        1.01 * motor.static_friction_nm / motor.friction_stiffness_nm_per_rad
    )
    active = PhysicsMotorConfig(
        project_root=base.project_root,
        path=base.path,
        payload=payload,
        nominal=motor,
    )

    with pytest.raises(ValueError, match="exceeds the static-friction bound"):
        active.validate()


def test_lugre_uncertainty_extremes_must_preserve_level_order(tmp_path):
    payload = json.loads(
        (PROJECT_ROOT / "config" / "motor_physics.json").read_text(encoding="utf-8")
    )
    payload["friction_model"]["active"] = "lugre"
    payload["friction_model"]["parameter_status"]["coulomb_friction_nm"] = (
        "identified"
    )
    payload["friction_model"]["parameter_status"]["static_friction_nm"] = (
        "identified"
    )
    payload["nominal_parameters"]["coulomb_friction_nm"] = 0.02
    payload["nominal_parameters"]["static_friction_nm"] = 0.025
    payload["uncertainty_fraction"]["coulomb_friction_nm"] = 0.1
    payload["uncertainty_fraction"]["static_friction_nm"] = 0.2
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "motor_physics.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    clear_physics_motor_config_cache()
    with pytest.raises(ValueError, match="uncertainty ranges can violate"):
        load_physics_motor_config(tmp_path)


def test_physics_ensemble_is_coherent_and_inside_declared_uncertainty(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    shutil.copy2(PROJECT_ROOT / "config" / "motor_physics.json", config_dir)
    clear_physics_motor_config_cache()
    build_physics_motor_ensemble(tmp_path)
    config = load_physics_motor_config(tmp_path)
    ensemble = load_physics_motor_ensemble(tmp_path)
    assert ensemble["parameters"].shape == (56, len(MODEL_PARAMETER_NAMES))
    assert int(ensemble["schema_version"]) == 3
    assert ensemble["sample_period_names"].tolist() == [
        "current",
        "speed",
        "position",
    ]
    assert np.array_equal(
        ensemble["sample_periods_s"], np.asarray([25e-6, 200e-6, 200e-6])
    )
    assert np.count_nonzero(ensemble["role"] == "train") == 40
    assert np.count_nonzero(ensemble["role"] == "validation") == 16
    nominal_index = int(np.flatnonzero(ensemble["is_nominal"] == 1)[0])
    assert np.array_equal(ensemble["parameters"][nominal_index], config.nominal.as_array())
    nominal = config.nominal.as_array()
    nonzero = nominal != 0.0
    relative = np.zeros_like(ensemble["parameters"])
    relative[:, nonzero] = np.abs(
        ensemble["parameters"][:, nonzero] / nominal[nonzero] - 1.0
    )
    limits = np.asarray(
        [config.uncertainty_fraction[name] for name in MODEL_PARAMETER_NAMES]
    )
    assert np.all(relative <= limits[None, :] + 1e-12)
    assert np.all(ensemble["parameters"][:, ~nonzero] == 0.0)


def test_all_11_physics_initials_are_active_and_match_design():
    space = load_physics_controller_parameter_space(PROJECT_ROOT)
    expected = derive_physics_controller_initials(PROJECT_ROOT)
    evidence = space.metadata.get("current_pidf_feasibility_evidence")
    if evidence is not None:
        for name, value in evidence["selected_current_parameters"].items():
            expected[space.names.index(name)] = value
    assert np.allclose(space.initial, expected, rtol=1e-12, atol=0.0)
    assert np.all(space.initial > 0.0)
    expected_period_s = {
        "current": 25e-6,
        "speed": 200e-6,
        "position": 200e-6,
        "DOBC": 200e-6,
    }
    assert all(
        spec.sample_period_s == expected_period_s[spec.module]
        for spec in space.specs
    )
    assert space.metadata["physics_model"]["primary_training_plant"]


def test_nominal_discrete_scenarios_respect_simulation_envelope():
    config = load_physics_motor_config(PROJECT_ROOT)
    space = load_physics_controller_parameter_space(PROJECT_ROOT)
    for scenario in ("current", "speed", "position", "disturbance"):
        trace = simulate_scenario(config, config.nominal, space.initial, scenario)
        assert not trace.terminated
        assert np.isfinite(trace.output).all()
        assert np.max(np.abs(trace.voltage_v)) <= config.limits["voltage_v"] + 1e-12
        assert np.max(np.abs(trace.current_a)) <= config.limits["hard_current_a"] + 1e-6
        assert np.max(np.abs(trace.speed_rad_s)) < config.limits["hard_speed_rad_s"]


def test_approved_dobc_sign_reduces_resisting_load_disturbance():
    config = load_physics_motor_config(PROJECT_ROOT)
    space = load_physics_controller_parameter_space(PROJECT_ROOT)
    disabled = space.initial.copy()
    disabled[6] = 0.0
    without_dobc = simulate_scenario(
        config, config.nominal, disabled, "disturbance"
    )
    with_dobc = simulate_scenario(
        config, config.nominal, space.initial, "disturbance"
    )
    start = int(
        config.scenarios["disturbance_start_s"] / config.base_sample_period_s
    )
    target = config.scenarios["speed_reference_rad_s"]
    peak_without = np.max(np.abs(target - without_dobc.output[start:]))
    peak_with = np.max(np.abs(target - with_dobc.output[start:]))
    assert peak_with < 0.5 * peak_without


def test_encoder_stress_and_scenario_override_are_seed_reproducible():
    config = load_physics_motor_config(PROJECT_ROOT)
    space = load_physics_controller_parameter_space(PROJECT_ROOT)
    override = {"load_torque_step_nm": 0.08165}
    first = simulate_scenario(
        config,
        config.nominal,
        space.initial,
        "disturbance",
        encoder_effects=True,
        seed=20260724,
        scenario_overrides=override,
    )
    second = simulate_scenario(
        config,
        config.nominal,
        space.initial,
        "disturbance",
        encoder_effects=True,
        seed=20260724,
        scenario_overrides=override,
    )
    assert np.array_equal(first.output, second.output)
    assert np.array_equal(first.current_reference_a, second.current_reference_a)
    assert np.max(first.load_torque_nm) == 0.08165
    assert not first.terminated
    assert np.isfinite(first.output).all()
