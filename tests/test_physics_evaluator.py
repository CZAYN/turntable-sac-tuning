from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from elc_rl.discrete_loop_model import (
    build_discrete_loop_model,
    build_discrete_loop_models,
)
from elc_rl.physics_evaluator import (
    PHYSICS_FREQUENCY_POINTS,
    PHYSICS_FRICTION_CONTEXT_PARAMETER_NAMES,
    PhysicsControllerEvaluator,
    get_physics_controller_evaluator,
    get_physics_time_domain_evaluator,
)
from elc_rl.physics_motor_model import MODEL_PARAMETER_NAMES, simulate_scenario
from elc_rl.tuning_env import PIDTuningEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LEGACY_PUBLIC_KEY_FRAGMENTS = (
    "iae",
    "steady_state_error",
    "sensitivity_peak",
    "complementary_peak",
    "disturbance_",
    "control_peak",
    "control_rms",
    "control_slew",
    "crossover_ratio",
)


def _assert_no_legacy_public_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            assert not any(fragment in key for fragment in LEGACY_PUBLIC_KEY_FRAGMENTS)
            assert key != "splits"
            _assert_no_legacy_public_keys(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_legacy_public_keys(child)


def test_physics_frequency_report_uses_only_the_three_frequency_table_metrics():
    evaluator = get_physics_controller_evaluator(PROJECT_ROOT)
    report = evaluator.audit(evaluator.space.initial)
    assert report["backend"] == "physics"
    assert report["evaluated_model_count"] == 56
    assert report["safety"]["safe"]
    assert set(report["cost"]) == {"loops", "frequency_total", "total"}
    assert set(report["cost"]["loops"]) == {"current", "speed", "position"}
    expected_bandwidths = {"current": 1500.0, "speed": 100.0, "position": 20.0}
    for loop, bandwidth_hz in expected_bandwidths.items():
        metrics = report["performance_metrics"][loop]
        assert metrics["target"]["bandwidth_hz"] == bandwidth_hz
        assert set(metrics["normalized_errors"]) == {
            "bandwidth",
            "gain_margin",
            "phase_margin",
        }
        assert np.isfinite(metrics["frequency_cost"])
    assert "crossover" not in report["cost"]
    assert "sensitivity" not in report["cost"]
    assert "bandwidth_hierarchy" not in report["cost"]
    assert "dobc_idealized" not in report["cost"]
    _assert_no_legacy_public_keys(report)


def test_physics_time_report_uses_only_the_three_time_table_metrics():
    frequency = get_physics_controller_evaluator(PROJECT_ROOT)
    evaluator = get_physics_time_domain_evaluator(PROJECT_ROOT)
    sampled = frequency.sample_training_indices(np.random.default_rng(20260722))
    report = evaluator.train(evaluator.space.initial, sampled)
    assert report["backend"] == "physics"
    assert report["safety"]["safe"]
    assert set(report["cost"]) == {"loops", "time_total", "total"}
    for loop in ("current", "speed", "position"):
        metrics = report["performance_metrics"][loop]
        assert set(metrics["normalized_errors"]) == {
            "overshoot",
            "rise_time",
            "settling_time",
        }
        assert np.isfinite(metrics["time_cost"])
    _assert_no_legacy_public_keys(report)


def test_dobc_changes_speed_and_position_frequency_models_but_not_current():
    evaluator = PhysicsControllerEvaluator(PROJECT_ROOT)
    motor = evaluator.motor(int(evaluator.training_indices[0]))
    baseline = evaluator.space.initial.copy()
    changed = baseline.copy()
    changed[6] = 0.0
    changed[7] = evaluator.space.specs[7].upper
    original_systems = build_discrete_loop_models(
        evaluator.config, motor, baseline
    )
    changed_systems = build_discrete_loop_models(evaluator.config, motor, changed)

    frequencies = np.geomspace(0.1, 500.0, 256)
    assert np.allclose(
        original_systems["current"].open_loop_response(frequencies),
        changed_systems["current"].open_loop_response(frequencies),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.allclose(
        original_systems["current"].closed_actual_response(frequencies),
        changed_systems["current"].closed_actual_response(frequencies),
        rtol=0.0,
        atol=1e-12,
    )
    for name in ("speed", "position"):
        difference = np.max(
            np.abs(
                original_systems[name].open_loop_response(frequencies)
                - changed_systems[name].open_loop_response(frequencies)
            )
        )
        assert difference > 1e-6


def test_discrete_current_loop_matches_the_sampled_kernel_equations():
    evaluator = PhysicsControllerEvaluator(PROJECT_ROOT)
    motor = evaluator.config.nominal
    parameters = evaluator.space.initial
    model = build_discrete_loop_model(
        evaluator.config, motor, parameters, "current"
    )
    frequency_hz = np.geomspace(0.2, 2250.0, 512)
    dt = evaluator.config.sample_period_s
    q = np.exp(-1j * 2.0 * np.pi * frequency_hz * dt)
    derivative_alpha = dt / (
        evaluator.config.derivative_filter_s["current"] + dt
    )
    derivative = (
        derivative_alpha
        / dt
        * (1.0 - q)
        / (1.0 - (1.0 - derivative_alpha) * q)
    )
    controller = (
        parameters[8]
        + parameters[9] * dt / (1.0 - q)
        + parameters[10] * derivative
    )
    delay_alpha = dt / (motor.current_delay_s + dt)
    actuator = delay_alpha / (1.0 - (1.0 - delay_alpha) * q)
    electrical_pole = 1.0 - dt * motor.resistance_ohm / motor.inductance_h
    electrical = (
        dt
        / motor.inductance_h
        * q
        / (1.0 - electrical_pole * q)
    )
    expected = controller * actuator * electrical
    assert np.allclose(
        model.open_loop_response(frequency_hz), expected, rtol=1e-10, atol=1e-9
    )


def test_discrete_closed_steps_match_the_nonlinear_kernel_at_small_signal():
    evaluator = PhysicsControllerEvaluator(PROJECT_ROOT)
    motor = evaluator.config.nominal
    parameters = evaluator.space.initial
    models = build_discrete_loop_models(evaluator.config, motor, parameters)
    references = {"current": 1e-6, "speed": 1e-6, "position": 1e-7}
    override_names = {
        "current": "current_reference_a",
        "speed": "speed_reference_rad_s",
        "position": "position_reference_rad",
    }
    for loop, reference in references.items():
        trace = simulate_scenario(
            evaluator.config,
            motor,
            parameters,
            loop,
            scenario_overrides={override_names[loop]: reference},
        )
        # A lifted speed/position sample contains eight 25 us kernel steps.
        # Compare at each completed loop interval, not at the held 40 kHz trace
        # samples between outer-controller updates.
        update_ratio = evaluator.config.controller_update_ratios[loop]
        sampled_output = trace.output[update_ratio - 1 :: update_ratio]
        predicted = models[loop].closed_step_response(
            reference, sampled_output.size
        )
        assert np.allclose(sampled_output, predicted, rtol=1e-3, atol=1e-12)


def test_validation_models_are_reported_but_excluded_from_frequency_cost():
    evaluator = PhysicsControllerEvaluator(PROJECT_ROOT)
    audit = evaluator.audit(evaluator.space.initial)
    training_only = evaluator._evaluate(
        evaluator.space.initial,
        evaluator.training_indices,
        frequency_points=PHYSICS_FREQUENCY_POINTS,
        mode="test_training_only",
        include_models=False,
    )
    assert audit["cost"] == training_only["cost"]
    assert audit["safety"]["safe"] == training_only["safety"]["safe"]
    assert "validation_diagnostics" in audit
    assert audit["validation_diagnostics"]["cost"] is not audit["cost"]


def test_validation_models_are_reported_but_excluded_from_time_cost():
    frequency = PhysicsControllerEvaluator(PROJECT_ROOT)
    evaluator = get_physics_time_domain_evaluator(PROJECT_ROOT)
    nominal = int(np.flatnonzero(frequency.ensemble["is_nominal"] == 1)[0])
    validation = int(np.flatnonzero(frequency.ensemble["role"] == "validation")[0])
    training_only = evaluator.evaluate(
        frequency.space.initial,
        np.asarray([nominal], dtype=np.int64),
        mode="test_training_only",
    )
    with_validation = evaluator.evaluate(
        frequency.space.initial,
        np.asarray([nominal, validation], dtype=np.int64),
        mode="test_with_validation",
    )
    assert with_validation["cost"] == training_only["cost"]
    assert with_validation["safety"]["safe"] == training_only["safety"]["safe"]
    assert "validation_diagnostics" in with_validation


def test_default_environment_uses_physics_and_one_coherent_motor():
    environment = PIDTuningEnv(
        PROJECT_ROOT,
        stage="joint",
        max_episode_steps=2,
        audit_interval=8,
        initial_perturbation=0.0,
    )
    observation, info = environment.reset(seed=20260722, options={"perturb": False})
    assert environment.backend == "physics"
    assert info["backend"] == "physics"
    assert len(info["sampled_model_ids"]) == 1
    assert info["fast_safe"] and info["audit_safe"]
    assert environment.observation_space.contains(observation)
    assert observation["sampled_frf"].shape == (96,)
    transition = environment.step(np.zeros(11, dtype=np.float32))
    assert environment.observation_space.contains(transition[0])
    assert not transition[2]


def test_sampled_physics_frf_is_finite_deterministic_and_task_independent():
    evaluator = get_physics_controller_evaluator(PROJECT_ROOT)
    sampled = evaluator.sample_training_indices(np.random.default_rng(20260730))
    first = evaluator.sampled_frf_vector(sampled)
    second = evaluator.sampled_frf_vector(sampled)
    assert first.shape == (96,)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    friction = evaluator.friction_context_vector(sampled)
    repeated_friction = evaluator.friction_context_vector(sampled)
    assert friction.shape == (6,)
    assert np.isfinite(friction).all()
    assert np.array_equal(friction, repeated_friction)
    assert np.max(np.abs(friction)) <= 1.0 + 1e-12
    assert np.any(np.abs(friction) > 0.0)


def test_active_lugre_context_exposes_the_sampled_model_uncertainty():
    evaluator = PhysicsControllerEvaluator(PROJECT_ROOT)
    payload = json.loads(json.dumps(evaluator.config.payload))
    payload["friction_model"]["active"] = "lugre"
    payload["friction_model"]["parameter_status"]["coulomb_friction_nm"] = (
        "synthetic_test_fixture"
    )
    payload["friction_model"]["parameter_status"]["static_friction_nm"] = (
        "synthetic_test_fixture"
    )
    payload["nominal_parameters"]["coulomb_friction_nm"] = 0.04
    payload["nominal_parameters"]["static_friction_nm"] = 0.07
    payload["uncertainty_fraction"]["coulomb_friction_nm"] = 0.1
    payload["uncertainty_fraction"]["static_friction_nm"] = 0.1
    nominal = replace(
        evaluator.config.nominal,
        coulomb_friction_nm=0.04,
        static_friction_nm=0.07,
    )
    evaluator.config = replace(
        evaluator.config,
        payload=payload,
        nominal=nominal,
    )
    evaluator.config.validate()

    sampled = np.asarray([int(evaluator.training_indices[1])], dtype=np.int64)
    deviations = np.asarray([0.5, -0.5, 0.25, 0.75, -0.75, 1.0])
    row = nominal.as_array().copy()
    for name, deviation in zip(
        PHYSICS_FRICTION_CONTEXT_PARAMETER_NAMES, deviations
    ):
        index = MODEL_PARAMETER_NAMES.index(name)
        uncertainty = evaluator.config.uncertainty_fraction[name]
        row[index] *= 1.0 + uncertainty * deviation
    evaluator.ensemble = {
        key: np.asarray(value).copy() for key, value in evaluator.ensemble.items()
    }
    evaluator.ensemble["parameters"][sampled[0]] = row

    context = evaluator.friction_context_vector(sampled)

    assert np.allclose(context, deviations, rtol=0.0, atol=1e-12)
