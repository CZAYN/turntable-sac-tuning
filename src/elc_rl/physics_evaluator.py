"""Frequency- and time-domain evaluators for the physics motor backend."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .evaluation_utils import (
    GAIN_MARGIN_CAP_DB,
    interpolate_pair,
    zero_crossing_locations,
)
from .controller_parameters import load_physics_controller_parameter_space
from .discrete_loop_model import (
    DiscreteLoopModel,
    build_discrete_loop_models,
)
from .performance_targets import (
    FREQUENCY_ERROR_ORDER,
    LOOP_ORDER,
    TIME_ERROR_ORDER,
    aggregate_loop_costs,
    aggregate_model_costs,
    frequency_normalized_errors,
    frequency_target_violations,
    load_controller_performance_targets,
    metric_cost,
    time_normalized_errors,
)
from .physics_motor_model import (
    MotorParameters,
    PhysicsMotorConfig,
    SimulationTrace,
    load_physics_motor_config,
    load_physics_motor_ensemble,
    simulate_scenario,
)


PHYSICS_FREQUENCY_POINTS = 1024
PHYSICS_TRAIN_FREQUENCY_POINTS = 320
PHYSICS_OBSERVATION_FREQUENCY_POINTS_PER_LOOP = 16
PHYSICS_OBSERVATION_CROSSOVER_RATIOS = (0.1, 10.0)
PHYSICS_FRICTION_CONTEXT_PARAMETER_NAMES = (
    "friction_stiffness_nm_per_rad",
    "friction_damping_nm_s_per_rad",
    "viscous_friction_nm_s_per_rad",
    "coulomb_friction_nm",
    "static_friction_nm",
    "stribeck_velocity_rad_s",
)
PHYSICS_FREQUENCY_LIMITS_HZ = {
    "current": (0.2, 2250.0),
    "speed": (0.02, 800.0),
    "position": (0.01, 250.0),
}


def _trim(values: np.ndarray) -> np.ndarray:
    result = np.trim_zeros(np.asarray(values, dtype=np.float64), trim="f")
    return result if result.size else np.asarray([0.0])


def _tf(
    numerator: np.ndarray | list[float], denominator: np.ndarray | list[float]
) -> tuple[np.ndarray, np.ndarray]:
    num = _trim(np.asarray(numerator, dtype=np.float64))
    den = _trim(np.asarray(denominator, dtype=np.float64))
    if den[0] == 0.0 or not np.isfinite(np.concatenate([num, den])).all():
        raise ValueError("invalid transfer function")
    return num / den[0], den / den[0]


def _series(
    *systems: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    numerator = np.asarray([1.0])
    denominator = np.asarray([1.0])
    for num, den in systems:
        numerator = np.convolve(numerator, num)
        denominator = np.convolve(denominator, den)
    return _tf(numerator, denominator)


def _scale(
    system: tuple[np.ndarray, np.ndarray], gain: float
) -> tuple[np.ndarray, np.ndarray]:
    numerator, denominator = system
    return _tf(float(gain) * numerator, denominator)


def _parallel(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
    *,
    right_gain: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Add two rational systems, optionally negating/scaling the right one."""

    left_num, left_den = left
    right_num, right_den = right
    return _tf(
        np.polyadd(
            np.convolve(left_num, right_den),
            float(right_gain) * np.convolve(right_num, left_den),
        ),
        np.convolve(left_den, right_den),
    )


def _divide(
    numerator_system: tuple[np.ndarray, np.ndarray],
    denominator_system: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    numerator_num, numerator_den = numerator_system
    denominator_num, denominator_den = denominator_system
    return _tf(
        np.convolve(numerator_num, denominator_den),
        np.convolve(numerator_den, denominator_num),
    )


def _feedback(
    forward: tuple[np.ndarray, np.ndarray],
    feedback: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    forward_num, forward_den = forward
    feedback_num, feedback_den = (
        _tf([1.0], [1.0]) if feedback is None else feedback
    )
    numerator = np.convolve(forward_num, feedback_den)
    denominator = np.polyadd(
        np.convolve(forward_den, feedback_den),
        np.convolve(forward_num, feedback_num),
    )
    return _tf(numerator, denominator)


def _response(
    system: tuple[np.ndarray, np.ndarray], frequency_hz: np.ndarray
) -> np.ndarray:
    numerator, denominator = system
    s = 1j * 2.0 * np.pi * np.asarray(frequency_hz, dtype=np.float64)
    return np.polyval(numerator, s) / np.polyval(denominator, s)


def _pid_tf(
    pid: tuple[float, float, float], filter_time_s: float
) -> tuple[np.ndarray, np.ndarray]:
    kp, ki, kd = pid
    return _tf(
        [kp * filter_time_s + kd, kp + ki * filter_time_s, ki],
        [filter_time_s, 1.0, 0.0],
    )


def _pid_values(parameters: np.ndarray, loop: str) -> tuple[float, float, float]:
    indices = {"position": (0, 1, 2), "speed": (3, 4, 5), "current": (8, 9, 10)}[
        loop
    ]
    return tuple(float(parameters[index]) for index in indices)  # type: ignore[return-value]


def physics_loop_transfers(
    config: PhysicsMotorConfig,
    motor: MotorParameters,
    controller_parameters: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Construct legacy continuous transfers for plant-context diagnostics.

    Reward, candidate audit and final evaluation use
    :func:`build_discrete_loop_models`; this continuous representation remains
    only for the 96-value plant context and held-out measured-FRF comparison.
    """

    filters = config.derivative_filter_s
    electrical = _tf(
        [1.0],
        np.convolve(
            [motor.current_delay_s, 1.0],
            [motor.inductance_h, motor.resistance_ohm],
        ),
    )
    current_controller = _pid_tf(
        _pid_values(controller_parameters, "current"), filters["current"]
    )
    current_open = _series(current_controller, electrical)
    current_closed = _feedback(current_open)

    mechanical = _tf(
        [motor.torque_constant_nm_per_a],
        [motor.inertia_kg_m2, motor.viscous_friction_nm_s_per_rad],
    )
    speed_sensor = _tf([1.0], [motor.speed_measurement_delay_s, 1.0])
    speed_controller = _pid_tf(
        _pid_values(controller_parameters, "speed"), filters["speed"]
    )
    current_to_speed = _series(current_closed, mechanical)

    # The DOBC is not a fourth loop.  It is an internal positive compensation
    # path inside the speed controller.  Linearizing the implemented equations
    # gives, from speed-loop current command to compensation current,
    #
    # D(s)=KG/Ktn * Q(s) * [Ktn*Ti(s)
    #                       -(Jn*s+Bn)*Ti(s)*Gm(s)*Hspeed(s)].
    #
    # Hence the effective speed forward path is Cspeed*Ti*Gm/(1-D).  This makes
    # both DOBC parameters affect the speed frequency metrics and, naturally,
    # the outer position loop while leaving the standalone current loop intact.
    nominal = config.nominal
    dobc_gain = float(controller_parameters[6])
    dobc_time_s = float(controller_parameters[7])
    q_filter = _tf([1.0], [dobc_time_s, 1.0])
    speed_measurement_from_iq_ref = _series(
        current_closed, mechanical, speed_sensor
    )
    nominal_mechanical_operator = _tf(
        [nominal.inertia_kg_m2, nominal.viscous_friction_nm_s_per_rad], [1.0]
    )
    estimator_from_iq_ref = _series(
        q_filter,
        _parallel(
            _scale(current_closed, nominal.torque_constant_nm_per_a),
            _series(nominal_mechanical_operator, speed_measurement_from_iq_ref),
            right_gain=-1.0,
        ),
    )
    dobc_return = _scale(
        estimator_from_iq_ref,
        dobc_gain / nominal.torque_constant_nm_per_a,
    )
    one_minus_dobc = _parallel(
        _tf([1.0], [1.0]), dobc_return, right_gain=-1.0
    )
    speed_forward_without_dobc = _series(speed_controller, current_to_speed)
    speed_effective_forward = _divide(
        speed_forward_without_dobc, one_minus_dobc
    )
    speed_open = _series(speed_effective_forward, speed_sensor)
    speed_actual_closed = _feedback(speed_effective_forward, speed_sensor)

    position_sensor = _tf([1.0], [motor.position_measurement_delay_s, 1.0])
    position_controller = _pid_tf(
        _pid_values(controller_parameters, "position"), filters["position"]
    )
    speed_to_position_feedback = _series(
        speed_actual_closed, _tf([1.0], [1.0, 0.0]), position_sensor
    )
    position_open = _series(position_controller, speed_to_position_feedback)
    return {
        "current_open": current_open,
        "current_closed": current_closed,
        "speed_open": speed_open,
        "speed_actual_closed": speed_actual_closed,
        "dobc_return": dobc_return,
        "position_open": position_open,
        "electrical_plant": electrical,
        "speed_measurement_plant": _series(mechanical, speed_sensor),
        "position_measurement_plant": _series(
            _tf([1.0], [1.0, 0.0]), position_sensor
        ),
    }


@lru_cache(maxsize=32)
def _frequency_grid(loop: str, points: int) -> np.ndarray:
    lower, upper = PHYSICS_FREQUENCY_LIMITS_HZ[loop]
    grid = np.geomspace(lower, upper, points)
    grid.setflags(write=False)
    return grid


def _classical_phase_margin_deg(phase_at_gain_crossing_deg: float) -> float:
    """Return phase margin on the conventional (-180, 180] degree branch.

    ``np.unwrap`` is useful for locating phase crossings, but its result can
    differ by an integer number of turns.  A position loop can therefore have
    a gain-crossover phase near +257 degrees instead of the equivalent
    -103 degrees.  Normalize only the gain-crossover phase used for phase
    margin; the unwrapped trace remains available for gain-margin crossings.
    """

    phase_deg = float(np.mod(phase_at_gain_crossing_deg, 360.0))
    if phase_deg > 0.0:
        phase_deg -= 360.0
    return float(180.0 + phase_deg)


def _evaluate_open_loop(
    loop: str,
    model: DiscreteLoopModel,
    *,
    frequency_points: int,
) -> dict[str, float | bool]:
    """Evaluate a sampled-data return ratio and its actual-output bandwidth."""

    if model.loop != loop:
        raise ValueError(f"discrete loop model mismatch: {model.loop} != {loop}")
    frequency_hz = _frequency_grid(loop, frequency_points)
    nyquist_hz = 0.5 / model.sample_period_s
    if float(frequency_hz[-1]) >= nyquist_hz:
        raise ValueError(
            f"{loop} frequency grid reaches {frequency_hz[-1]:g} Hz but "
            f"its sampled model Nyquist frequency is {nyquist_hz:g} Hz"
        )
    open_loop = model.open_loop_response(frequency_hz)
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(open_loop), 1e-300))
    phase_deg = np.rad2deg(np.unwrap(np.angle(open_loop)))
    log_frequency = np.log10(frequency_hz)

    gain_crossings = zero_crossing_locations(log_frequency, magnitude_db)
    if gain_crossings:
        phase_margin_deg = float(
            min(
                _classical_phase_margin_deg(
                    interpolate_pair(phase_deg, index, fraction)
                )
                for _, index, fraction in gain_crossings
            )
        )
    else:
        phase_margin_deg = -180.0

    gain_margin_candidates: list[float] = []
    target = -180.0
    while target >= float(np.min(phase_deg)):
        for _, index, fraction in zero_crossing_locations(
            log_frequency, phase_deg - target
        ):
            gain_margin_candidates.append(
            -interpolate_pair(magnitude_db, index, fraction)
            )
        target -= 360.0
    gain_margin_db = float(
        min(gain_margin_candidates)
        if gain_margin_candidates
        else GAIN_MARGIN_CAP_DB
    )

    closed_actual = model.closed_actual_response(frequency_hz)
    low_frequency_gain = float(abs(closed_actual[0]))
    relative_magnitude_db = 20.0 * np.log10(
        np.maximum(
            np.abs(closed_actual) / max(low_frequency_gain, 1e-300), 1e-300
        )
    )
    bandwidth_crossings = [
        crossing
        for crossing in zero_crossing_locations(
            log_frequency, relative_magnitude_db + 10.0 * np.log10(2.0)
        )
        if relative_magnitude_db[crossing[1]] >= -10.0 * np.log10(2.0)
        and relative_magnitude_db[crossing[1] + 1] < -10.0 * np.log10(2.0)
    ]
    bandwidth_found = bool(bandwidth_crossings)
    bandwidth_hz = float(
        bandwidth_crossings[0][0] if bandwidth_found else frequency_hz[-1]
    )
    poles = model.closed_loop_io_poles
    finite = bool(
        np.isfinite(open_loop.real).all()
        and np.isfinite(open_loop.imag).all()
        and np.isfinite(closed_actual.real).all()
        and np.isfinite(closed_actual.imag).all()
        and np.isfinite(poles.real).all()
        and np.isfinite(poles.imag).all()
    )
    maximum_pole_magnitude = float(np.max(np.abs(poles)))
    pole_stable = bool(maximum_pole_magnitude < 1.0 - 1e-10)
    # Missing gain or -3 dB crossings are finite but poor controllers, not a
    # numerical failure.  Their sentinel margins/bandwidth create a large Cost
    # and target_pass remains false, while SAC is allowed to recover.  Only a
    # non-finite response or an unstable discrete closed-loop I/O mode is invalid.
    metric_valid = bool(finite and pole_stable)
    return {
        "metric_valid": metric_valid,
        "phase_margin_deg": phase_margin_deg,
        "gain_margin_db": gain_margin_db,
        "bandwidth_hz": bandwidth_hz,
        "bandwidth_found": bandwidth_found,
    }


def _physics_split(loop: str, role: str) -> str:
    if role == "validation":
        return f"{loop}_validation"
    return {
        "current": "current_reference",
        "speed": "speed_train",
        "position": "position_surrogate",
    }[loop]


def _validated_loops(loops: Iterable[str] | None) -> tuple[str, ...]:
    selected = LOOP_ORDER if loops is None else tuple(str(loop) for loop in loops)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("evaluation loops must be a non-empty unique sequence")
    if any(loop not in LOOP_ORDER for loop in selected):
        raise ValueError(f"evaluation loops must be drawn from {LOOP_ORDER}")
    return tuple(loop for loop in LOOP_ORDER if loop in selected)


def _aggregate_selected_loop_costs(
    loop_costs: dict[str, float], settings: Any
) -> float:
    if tuple(loop_costs) == LOOP_ORDER:
        return aggregate_loop_costs(loop_costs, settings)
    values = tuple(float(value) for value in loop_costs.values())
    return float(
        np.mean(values) + settings.joint_worst_loop_weight * np.max(values)
    )


def _frequency_performance(
    rows: list[dict[str, Any]],
    targets: Any,
    loops: Iterable[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the frequency half of the literal six-metric objective."""

    settings = targets.cost
    performance: dict[str, Any] = {}
    loop_costs: dict[str, float] = {}
    loop_validity: dict[str, bool] = {}
    selected_loops = _validated_loops(loops)
    for loop in selected_loops:
        loop_rows = [row for row in rows if row["loop"] == loop]
        if not loop_rows:
            raise ValueError(f"frequency evaluation has no {loop} rows")
        target = targets.loop(loop)
        row_costs: list[float] = []
        row_errors: list[dict[str, float]] = []
        row_violations: list[dict[str, float]] = []
        for row in loop_rows:
            errors = frequency_normalized_errors(
                bandwidth_hz=float(row["bandwidth_hz"]),
                gain_margin_db=float(row["gain_margin_db"]),
                phase_margin_deg=float(row["phase_margin_deg"]),
                target=target,
                bandwidth_relative_scale=settings.bandwidth_relative_scale,
                invalid_error=settings.invalid_normalized_error,
            )
            cost = metric_cost(errors, FREQUENCY_ERROR_ORDER, delta=settings.huber_delta)
            violations = frequency_target_violations(
                bandwidth_hz=float(row["bandwidth_hz"]),
                gain_margin_db=float(row["gain_margin_db"]),
                phase_margin_deg=float(row["phase_margin_deg"]),
                target=target,
                invalid_error=settings.invalid_normalized_error,
            )
            if not bool(row["metric_valid"]):
                violations = {
                    name: float(settings.invalid_normalized_error)
                    for name in FREQUENCY_ERROR_ORDER
                }
            row["normalized_errors"] = errors
            row["target_violations"] = violations
            row["frequency_cost"] = cost
            row_errors.append(errors)
            row_violations.append(violations)
            row_costs.append(cost)
        bandwidth_errors = np.asarray(
            [errors["bandwidth"] for errors in row_errors], dtype=np.float64
        )
        representative_bandwidth_error = float(
            bandwidth_errors[int(np.argmax(np.abs(bandwidth_errors)))]
        )
        normalized_errors = {
            "bandwidth": representative_bandwidth_error,
            "gain_margin": float(
                max(errors["gain_margin"] for errors in row_errors)
            ),
            "phase_margin": float(
                max(errors["phase_margin"] for errors in row_errors)
            ),
        }
        target_violations = {
            name: float(max(violations[name] for violations in row_violations))
            for name in FREQUENCY_ERROR_ORDER
        }
        target_violation_model_count = int(
            sum(
                any(float(value) > 0.0 for value in violations.values())
                for violations in row_violations
            )
        )
        valid = bool(all(bool(row["metric_valid"]) for row in loop_rows))
        target_pass = bool(valid and target_violation_model_count == 0)
        loop_cost = aggregate_model_costs(row_costs, settings)
        loop_costs[loop] = loop_cost
        loop_validity[loop] = valid
        performance[loop] = {
            "target": target.as_dict(),
            "actual": {
                "bandwidth_hz_median": float(
                    np.median([float(row["bandwidth_hz"]) for row in loop_rows])
                ),
                "gain_margin_db_worst": float(
                    min(float(row["gain_margin_db"]) for row in loop_rows)
                ),
                "phase_margin_deg_worst": float(
                    min(float(row["phase_margin_deg"]) for row in loop_rows)
                ),
            },
            "normalized_errors": normalized_errors,
            "target_violations": target_violations,
            "target_violation_model_count": target_violation_model_count,
            "frequency_cost": loop_cost,
            "valid": valid,
            "target_pass": target_pass,
        }
    frequency_total = _aggregate_selected_loop_costs(loop_costs, settings)
    cost = {
        "loops": dict(loop_costs),
        "frequency_total": frequency_total,
        "total": frequency_total,
    }
    safety = {
        "safe": bool(all(loop_validity.values())),
        "frequency_metrics_valid": bool(all(loop_validity.values())),
        "loop_validity": loop_validity,
        "all_frequency_targets_met": bool(
            all(performance[loop]["target_pass"] for loop in selected_loops)
        ),
    }
    return performance, cost, safety


class PhysicsControllerEvaluator:
    """Frequency evaluator over coherent randomized physical motor instances."""

    backend = "physics"

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.config = load_physics_motor_config(self.project_root)
        self.space = load_physics_controller_parameter_space(self.project_root)
        self.performance_targets = load_controller_performance_targets(
            self.project_root
        )
        self.ensemble = load_physics_motor_ensemble(self.project_root)
        self.training_indices = np.flatnonzero(
            self.ensemble["active_for_training"] == 1
        ).astype(np.int64)
        self.audit_indices = np.flatnonzero(
            self.ensemble["active_for_audit"] == 1
        ).astype(np.int64)
        if self.training_indices.size != 40 or self.audit_indices.size != 56:
            raise ValueError("physics ensemble split sizes are invalid")

    def sample_training_indices(self, rng: np.random.Generator) -> np.ndarray:
        return np.asarray([int(rng.choice(self.training_indices))], dtype=np.int64)

    def validate_sampled_indices(self, indices: np.ndarray) -> np.ndarray:
        values = np.asarray(indices, dtype=np.int64)
        if values.shape != (1,):
            raise ValueError("physics requires one coherent model index per episode")
        if int(values[0]) not in set(self.training_indices.tolist()):
            raise ValueError("physics sampled model is not in the training split")
        return values

    def model_ids(self, indices: np.ndarray) -> tuple[str, ...]:
        return tuple(
            str(self.ensemble["model_id"][int(index)])
            for index in np.asarray(indices, dtype=np.int64)
        )

    def motor(self, index: int) -> MotorParameters:
        return MotorParameters.from_array(self.ensemble["parameters"][int(index)])

    def sampled_frf_vector(self, indices: np.ndarray) -> np.ndarray:
        """Encode one sampled physics plant on a model-defined frequency grid."""

        values = self.validate_sampled_indices(indices)
        motor = self.motor(int(values[0]))
        systems = physics_loop_transfers(self.config, motor, self.space.initial)
        bandwidth_targets = {
            loop: self.performance_targets.loop(loop).bandwidth_hz
            for loop in LOOP_ORDER
        }
        lower_ratio, upper_ratio = PHYSICS_OBSERVATION_CROSSOVER_RATIOS
        frequencies = {}
        for loop, target_hz in bandwidth_targets.items():
            physical_lower_hz, physical_upper_hz = PHYSICS_FREQUENCY_LIMITS_HZ[loop]
            lower_hz = max(physical_lower_hz, lower_ratio * target_hz)
            upper_hz = min(physical_upper_hz, upper_ratio * target_hz)
            if not 0.0 < lower_hz < upper_hz:
                raise ValueError(f"invalid physics observation frequency grid for {loop}")
            frequencies[loop] = np.geomspace(
                lower_hz,
                upper_hz,
                PHYSICS_OBSERVATION_FREQUENCY_POINTS_PER_LOOP,
            )
        plant_names = {
            "current": "electrical_plant",
            "speed": "speed_measurement_plant",
            "position": "position_measurement_plant",
        }
        parts: list[np.ndarray] = []
        for loop in ("current", "speed", "position"):
            response = _response(systems[plant_names[loop]], frequencies[loop])
            features = np.column_stack(
                [
                    20.0 * np.log10(np.maximum(np.abs(response), 1e-300)) / 40.0,
                    np.rad2deg(np.unwrap(np.angle(response))) / 180.0,
                ]
            )
            parts.append(features.reshape(-1))
        vector = np.concatenate(parts).astype(np.float64)
        if vector.shape != (96,) or not np.isfinite(vector).all():
            raise ValueError("physics sampled FRF vector is invalid")
        return vector

    def friction_context_vector(self, indices: np.ndarray) -> np.ndarray:
        """Encode the sampled model's LuGre uncertainty for the SAC policy.

        The active model is represented as a centered fraction of each declared
        uncertainty interval, so every randomized component normally lies in
        ``[-1, 1]``.  A deliberately selected viscous-only compatibility
        configuration returns zeros because its stored LuGre values do not
        participate in episode dynamics.
        """

        values = self.validate_sampled_indices(indices)
        if self.config.active_friction_model != "lugre":
            return np.zeros(
                len(PHYSICS_FRICTION_CONTEXT_PARAMETER_NAMES), dtype=np.float64
            )

        motor = self.motor(int(values[0]))
        context: list[float] = []
        for name in PHYSICS_FRICTION_CONTEXT_PARAMETER_NAMES:
            nominal = float(getattr(self.config.nominal, name))
            value = float(getattr(motor, name))
            uncertainty = float(self.config.uncertainty_fraction[name])
            if nominal <= 0.0:
                raise ValueError(f"active LuGre context has invalid nominal {name}")
            normalized = 0.0
            if uncertainty > 0.0:
                normalized = (value / nominal - 1.0) / uncertainty
            context.append(normalized)
        vector = np.asarray(context, dtype=np.float64)
        if (
            vector.shape != (len(PHYSICS_FRICTION_CONTEXT_PARAMETER_NAMES),)
            or not np.isfinite(vector).all()
        ):
            raise ValueError("physics friction context is invalid")
        return np.clip(vector, -5.0, 5.0)

    def _evaluate(
        self,
        parameters: np.ndarray,
        indices: np.ndarray,
        *,
        frequency_points: int,
        mode: str,
        include_models: bool,
        loops: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        selected_loops = _validated_loops(loops)
        values = np.asarray(parameters, dtype=np.float64)
        self.space.normalize(values)
        rows: list[dict[str, Any]] = []
        for raw_index in np.asarray(indices, dtype=np.int64):
            index = int(raw_index)
            motor = self.motor(index)
            models = build_discrete_loop_models(self.config, motor, values)
            role = str(self.ensemble["role"][index])
            for loop in selected_loops:
                split = _physics_split(loop, role)
                row: dict[str, Any] = {
                    "model_id": str(self.ensemble["model_id"][index]),
                    "loop": loop,
                    "role": role,
                    "split": split,
                    **_evaluate_open_loop(
                        loop,
                        models[loop],
                        frequency_points=frequency_points,
                    ),
                }
                rows.append(row)
        training_rows = [row for row in rows if row["role"] != "validation"]
        validation_rows = [row for row in rows if row["role"] == "validation"]
        performance, cost, safety = _frequency_performance(
            training_rows, self.performance_targets, selected_loops
        )
        validation_diagnostics = None
        if validation_rows:
            (
                validation_performance,
                validation_cost,
                validation_safety,
            ) = _frequency_performance(
                validation_rows, self.performance_targets, selected_loops
            )
            validation_diagnostics = {
                "performance_metrics": validation_performance,
                "cost": validation_cost,
                "safety": validation_safety,
            }
            safety["validation_metrics_valid"] = bool(
                validation_safety["frequency_metrics_valid"]
            )
        else:
            safety["validation_metrics_valid"] = None
        report: dict[str, Any] = {
            "schema_version": 3,
            "backend": self.backend,
            "task_id": self.space.task_id,
            "evaluation_mode": mode,
            "parameter_names": list(self.space.names),
            "parameters": values.tolist(),
            "evaluated_model_count": int(len(indices)),
            "evaluated_model_ids": list(self.model_ids(indices)),
            "evaluated_loops": list(selected_loops),
            "performance_targets": self.performance_targets.as_dict(),
            "performance_metrics": performance,
            "cost": cost,
            "safety": safety,
            "sample_periods_s": {
                loop: self.config.sample_period_s_for(loop)
                for loop in LOOP_ORDER
            },
            "controller_update_ratios": dict(
                self.config.controller_update_ratios
            ),
            "semantics": {
                "motor": "mentor physics model with coherent per-episode uncertainty",
                "frequency_model": "periodically lifted sampled-data state space: one outer-loop transition contains its integer number of 40 kHz current and motor substeps",
                "current": "40 kHz sampled PIDF, discrete current actuator lag and explicit-Euler electrical state",
                "speed": "5 kHz sampled PIDF and DOBC with a zero-order-held current reference over eight closed 40 kHz current, motor and LuGre substeps",
                "position": "5 kHz sampled PIDF with the closed 5 kHz speed-plus-DOBC loop and eight closed 40 kHz current, motor and LuGre substeps",
                "dobc": "embedded in the discrete speed and position models; evaluated only through their six metrics",
                "objective": "closed-loop bandwidth, gain margin and phase margin only",
                "measured_frf": "validation context only; not fitted into these model parameters",
                "linearization_boundary": "small-signal zero-operating-point model; saturation, encoder quantization, hard termination and cross-rate alias images are excluded",
            },
        }
        if include_models:
            report["models"] = [
                {
                    "model_id": row["model_id"],
                    "loop": row["loop"],
                    "role": row["role"],
                    "split": row["split"],
                    "bandwidth_hz": row["bandwidth_hz"],
                    "gain_margin_db": row["gain_margin_db"],
                    "phase_margin_deg": row["phase_margin_deg"],
                    "bandwidth_found": row["bandwidth_found"],
                    "metric_valid": row["metric_valid"],
                    "normalized_errors": row["normalized_errors"],
                    "target_violations": row["target_violations"],
                    "frequency_cost": row["frequency_cost"],
                }
                for row in rows
            ]
        if validation_diagnostics is not None:
            report["validation_diagnostics"] = validation_diagnostics
        return report

    def train(
        self,
        parameters: np.ndarray,
        sampled_indices: np.ndarray,
        *,
        include_models: bool = False,
        loops: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        indices = self.validate_sampled_indices(sampled_indices)
        return self._evaluate(
            parameters,
            indices,
            frequency_points=PHYSICS_TRAIN_FREQUENCY_POINTS,
            mode="train",
            include_models=include_models,
            loops=loops,
        )

    def audit(
        self,
        parameters: np.ndarray,
        *,
        include_models: bool = False,
        loops: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        return self._evaluate(
            parameters,
            self.audit_indices,
            frequency_points=PHYSICS_FREQUENCY_POINTS,
            mode="audit",
            include_models=include_models,
            loops=loops,
        )

    def training_audit(
        self,
        parameters: np.ndarray,
        *,
        include_models: bool = False,
        loops: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Audit only the 40 training-role models at production resolution."""

        return self._evaluate(
            parameters,
            self.training_indices,
            frequency_points=PHYSICS_FREQUENCY_POINTS,
            mode="training_audit",
            include_models=include_models,
            loops=loops,
        )


def _reference_metrics(trace: SimulationTrace) -> dict[str, float | bool]:
    target = float(trace.reference[-1])
    if abs(target) <= 1e-12:
        raise ValueError("reference scenario has a zero target")
    normalized = trace.output / target
    error = 1.0 - normalized
    ten = np.flatnonzero(normalized >= 0.1)
    ninety = np.flatnonzero(normalized >= 0.9)
    rise_time = (
        float(trace.time_s[ninety[0]] - trace.time_s[ten[0]])
        if ten.size and ninety.size and ninety[0] >= ten[0]
        else float(trace.time_s[-1])
    )
    outside = np.flatnonzero(np.abs(error) > 0.02)
    settled = bool(not outside.size or outside[-1] < len(error) - 1)
    settling_time = (
        0.0
        if not outside.size
        else float(trace.time_s[min(int(outside[-1]) + 1, len(error) - 1)])
    )
    return {
        "rise_time_s": rise_time,
        "settling_time_s": settling_time,
        "settled": settled,
        "reached_10_percent": bool(ten.size),
        "reached_90_percent": bool(ninety.size),
        "reference_metric_valid": bool(
            np.isfinite(trace.time_s).all()
            and np.isfinite(trace.reference).all()
            and np.isfinite(trace.output).all()
            and np.isfinite(trace.primary_control).all()
            and np.isfinite(trace.voltage_v).all()
            and np.isfinite(trace.current_a).all()
            and np.isfinite(normalized).all()
        ),
        "overshoot_ratio": max(0.0, float(np.max(normalized) - 1.0)),
    }


def _time_performance(
    rows: list[dict[str, Any]],
    targets: Any,
    loops: Iterable[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the time half of the literal six-metric objective."""

    settings = targets.cost
    performance: dict[str, Any] = {}
    loop_costs: dict[str, float] = {}
    loop_validity: dict[str, bool] = {}
    selected_loops = _validated_loops(loops)
    for loop in selected_loops:
        loop_rows = [row for row in rows if row["loop"] == loop]
        if not loop_rows:
            raise ValueError(f"time-domain evaluation has no {loop} rows")
        target = targets.loop(loop)
        row_costs: list[float] = []
        row_errors: list[dict[str, float]] = []
        row_violations: list[dict[str, float]] = []
        for row in loop_rows:
            errors = time_normalized_errors(
                overshoot_ratio=float(row["overshoot_ratio"]),
                rise_time_s=float(row["rise_time_s"]),
                settling_time_s=float(row["settling_time_s"]),
                target=target,
                invalid_error=settings.invalid_normalized_error,
            )
            cost = metric_cost(errors, TIME_ERROR_ORDER, delta=settings.huber_delta)
            violations = dict(errors)
            if not bool(row["time_domain_stable"]) or not bool(
                row["reference_metric_valid"]
            ):
                violations = {
                    name: float(settings.invalid_normalized_error)
                    for name in TIME_ERROR_ORDER
                }
            else:
                if not bool(row["reached_90_percent"]):
                    violations["rise_time"] = float(
                        settings.invalid_normalized_error
                    )
                if not bool(row["settled"]):
                    violations["settling_time"] = float(
                        settings.invalid_normalized_error
                    )
            row["normalized_errors"] = errors
            row["target_violations"] = violations
            row["time_cost"] = cost
            row_errors.append(errors)
            row_violations.append(violations)
            row_costs.append(cost)
        normalized_errors = {
            name: float(max(errors[name] for errors in row_errors))
            for name in TIME_ERROR_ORDER
        }
        target_violations = {
            name: float(max(violations[name] for violations in row_violations))
            for name in TIME_ERROR_ORDER
        }
        target_violation_model_count = int(
            sum(
                any(float(value) > 0.0 for value in violations.values())
                for violations in row_violations
            )
        )
        valid = bool(
            all(
                bool(row["time_domain_stable"])
                and bool(row["reference_metric_valid"])
                for row in loop_rows
            )
        )
        target_pass = bool(valid and target_violation_model_count == 0)
        loop_cost = aggregate_model_costs(row_costs, settings)
        loop_costs[loop] = loop_cost
        loop_validity[loop] = valid
        performance[loop] = {
            "target": target.as_dict(),
            "actual": {
                "overshoot_ratio_worst": float(
                    max(float(row["overshoot_ratio"]) for row in loop_rows)
                ),
                "rise_time_s_worst": float(
                    max(float(row["rise_time_s"]) for row in loop_rows)
                ),
                "settling_time_s_worst": float(
                    max(float(row["settling_time_s"]) for row in loop_rows)
                ),
            },
            "normalized_errors": normalized_errors,
            "target_violations": target_violations,
            "target_violation_model_count": target_violation_model_count,
            "time_cost": loop_cost,
            "valid": valid,
            "target_pass": target_pass,
        }
    time_total = _aggregate_selected_loop_costs(loop_costs, settings)
    cost = {
        "loops": dict(loop_costs),
        "time_total": time_total,
        "total": time_total,
    }
    safety = {
        "time_metrics_valid": bool(all(loop_validity.values())),
        "loop_validity": loop_validity,
        "all_time_targets_met": bool(
            all(performance[loop]["target_pass"] for loop in selected_loops)
        ),
    }
    return performance, cost, safety


class PhysicsTimeDomainEvaluator:
    """Nonlinear discrete-time evaluator with limits, anti-windup and DOBC."""

    backend = "physics"

    def __init__(self, frequency_evaluator: PhysicsControllerEvaluator) -> None:
        self.frequency_evaluator = frequency_evaluator
        self.project_root = frequency_evaluator.project_root
        self.config = frequency_evaluator.config
        self.space = frequency_evaluator.space
        self.performance_targets = frequency_evaluator.performance_targets
        self.ensemble = frequency_evaluator.ensemble
        validation = np.flatnonzero(self.ensemble["role"] == "validation").astype(
            np.int64
        )
        nominal = int(np.flatnonzero(self.ensemble["is_nominal"] == 1)[0])
        validation_probe = validation[
            np.linspace(0, validation.size - 1, 3, dtype=np.int64)
        ]
        self.runtime_audit_indices = np.concatenate(
            [np.asarray([nominal], dtype=np.int64), validation_probe]
        )

    def _raw_scenario(
        self, index: int, scenario: str, parameters: np.ndarray
    ) -> dict[str, Any]:
        trace = simulate_scenario(
            self.config,
            self.frequency_evaluator.motor(index),
            parameters,
            scenario,
        )
        metrics: dict[str, Any] = {
            "time_domain_stable": bool(not trace.terminated),
        }
        metrics.update(_reference_metrics(trace))
        return metrics

    def _model_metrics(
        self, index: int, loop: str, parameters: np.ndarray
    ) -> dict[str, Any]:
        return self._raw_scenario(index, loop, parameters)

    def evaluate(
        self,
        parameters: np.ndarray,
        indices: np.ndarray,
        *,
        mode: str,
        include_models: bool = False,
    ) -> dict[str, Any]:
        values = np.asarray(parameters, dtype=np.float64)
        self.space.normalize(values)
        rows: list[dict[str, Any]] = []
        for raw_index in np.asarray(indices, dtype=np.int64):
            index = int(raw_index)
            role = str(self.ensemble["role"][index])
            for loop in ("current", "speed", "position"):
                split = _physics_split(loop, role)
                row = {
                    "model_id": str(self.ensemble["model_id"][index]),
                    "loop": loop,
                    "role": role,
                    "split": split,
                    **self._model_metrics(index, loop, values),
                }
                rows.append(row)
        training_rows = [row for row in rows if row["role"] != "validation"]
        validation_rows = [row for row in rows if row["role"] == "validation"]
        performance, cost, performance_safety = _time_performance(
            training_rows, self.performance_targets
        )
        validation_diagnostics = None
        if validation_rows:
            (
                validation_performance,
                validation_cost,
                validation_performance_safety,
            ) = _time_performance(validation_rows, self.performance_targets)
            validation_diagnostics = {
                "performance_metrics": validation_performance,
                "cost": validation_cost,
                "safety": validation_performance_safety,
            }
        metrics_valid = bool(performance_safety["time_metrics_valid"])
        validation_metrics_valid = bool(
            True
            if validation_diagnostics is None
            else validation_diagnostics["safety"]["time_metrics_valid"]
        )
        report: dict[str, Any] = {
            "schema_version": 3,
            "backend": self.backend,
            "task_id": self.space.task_id,
            "evaluation_mode": mode,
            "evaluated_model_count": int(len(indices)),
            "evaluated_model_ids": list(self.frequency_evaluator.model_ids(indices)),
            "performance_targets": self.performance_targets.as_dict(),
            "performance_metrics": performance,
            "cost": cost,
            "safety": {
                "safe": metrics_valid,
                "core_metrics_valid": metrics_valid,
                "validation_metrics_valid": (
                    None
                    if validation_diagnostics is None
                    else validation_metrics_valid
                ),
                **performance_safety,
            },
            "assumptions": {
                "integration_step_s": self.config.sample_period_s,
                "controller": "three filtered PID controllers with conditional anti-windup",
                "dobc": self.config.payload["controller_design"]["dobc"]["structure"],
                "objective": "overshoot, 10-to-90-percent rise time and plus-or-minus-2-percent settling time only",
                "friction": self.config.payload["friction_model"],
                "encoder_effects_during_reward": False,
            },
        }
        if include_models:
            report["models"] = [
                {
                    "model_id": row["model_id"],
                    "loop": row["loop"],
                    "role": row["role"],
                    "split": row["split"],
                    "overshoot_ratio": row["overshoot_ratio"],
                    "rise_time_s": row["rise_time_s"],
                    "settling_time_s": row["settling_time_s"],
                    "reached_10_percent": row["reached_10_percent"],
                    "reached_90_percent": row["reached_90_percent"],
                    "settled": row["settled"],
                    "reference_metric_valid": row["reference_metric_valid"],
                    "time_domain_stable": row["time_domain_stable"],
                    "normalized_errors": row["normalized_errors"],
                    "target_violations": row["target_violations"],
                    "time_cost": row["time_cost"],
                }
                for row in rows
            ]
        if validation_diagnostics is not None:
            report["validation_diagnostics"] = validation_diagnostics
        return report

    def train(
        self,
        parameters: np.ndarray,
        sampled_indices: np.ndarray,
        *,
        include_models: bool = False,
    ) -> dict[str, Any]:
        indices = self.frequency_evaluator.validate_sampled_indices(sampled_indices)
        return self.evaluate(
            parameters, indices, mode="train", include_models=include_models
        )

    def audit(
        self, parameters: np.ndarray, *, include_models: bool = False
    ) -> dict[str, Any]:
        return self.evaluate(
            parameters,
            self.runtime_audit_indices,
            mode="runtime_audit",
            include_models=include_models,
        )

    def full_audit(
        self, parameters: np.ndarray, *, include_models: bool = False
    ) -> dict[str, Any]:
        """Run the expensive nonlinear audit over all 56 fixed motor models."""

        return self.evaluate(
            parameters,
            self.frequency_evaluator.audit_indices,
            mode="full_audit",
            include_models=include_models,
        )


@lru_cache(maxsize=4)
def _cached_physics_controller_evaluator(project_root: str) -> PhysicsControllerEvaluator:
    return PhysicsControllerEvaluator(Path(project_root))


def get_physics_controller_evaluator(project_root: Path) -> PhysicsControllerEvaluator:
    return _cached_physics_controller_evaluator(str(Path(project_root).resolve()))


@lru_cache(maxsize=4)
def _cached_physics_time_evaluator(project_root: str) -> PhysicsTimeDomainEvaluator:
    return PhysicsTimeDomainEvaluator(
        get_physics_controller_evaluator(Path(project_root))
    )


def get_physics_time_domain_evaluator(project_root: Path) -> PhysicsTimeDomainEvaluator:
    return _cached_physics_time_evaluator(str(Path(project_root).resolve()))


def clear_physics_evaluator_caches() -> None:
    _cached_physics_time_evaluator.cache_clear()
    _cached_physics_controller_evaluator.cache_clear()


def build_physics_baseline_evaluation(project_root: Path) -> dict[str, Any]:
    evaluator = get_physics_controller_evaluator(project_root)
    frequency = evaluator.audit(evaluator.space.initial, include_models=True)
    time_domain = get_physics_time_domain_evaluator(project_root).full_audit(
        evaluator.space.initial, include_models=True
    )
    report = {
        "schema_version": 1,
        "backend": "physics",
        "frequency": frequency,
        "time_domain": time_domain,
    }
    output = Path(project_root) / "outputs" / "physics_baseline_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
