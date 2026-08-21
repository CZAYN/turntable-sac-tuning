"""Implementation-aligned multirate small-signal models for the three loops.

The nonlinear simulator uses the current-loop period as its base integration
step.  Speed and position control execute at their declared slower periods and
hold their commands between updates.  Frequency-domain evaluation therefore
uses a lifted transition: one speed/position model step contains the exact
integer number of current-loop and motor integration substeps.  This module
linearizes those unsaturated lifted equations at the zero operating point and
exposes the return ratio at each loop summing junction.

The model deliberately excludes saturation, encoder quantization and hard
termination because gain/phase margins are small-signal quantities.  LuGre is
linearized from the exact state update used by :mod:`elc_rl.simulation_kernel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .physics_motor_model import MotorParameters, PhysicsMotorConfig
from .simulation_kernel import lugre_friction_step


LOOP_STATE_NAMES: dict[str, tuple[str, ...]] = {
    "current": (
        "current_integral",
        "current_derivative",
        "current_previous_error",
        "applied_voltage",
        "current",
    ),
    "speed": (
        "speed_integral",
        "speed_derivative",
        "speed_previous_error",
        "current_integral",
        "current_derivative",
        "current_previous_error",
        "current",
        "speed",
        "applied_voltage",
        "speed_feedback",
        "previous_speed_feedback",
        "estimated_load",
        "bristle_state",
    ),
    "position": (
        "position_integral",
        "position_derivative",
        "position_previous_error",
        "speed_integral",
        "speed_derivative",
        "speed_previous_error",
        "current_integral",
        "current_derivative",
        "current_previous_error",
        "current",
        "speed",
        "position",
        "applied_voltage",
        "speed_feedback",
        "position_feedback",
        "previous_speed_feedback",
        "estimated_load",
        "bristle_state",
    ),
}


def _pid_step(
    error: float,
    pid: tuple[float, float, float],
    filter_time_s: float,
    sample_period_s: float,
    integral: float,
    derivative: float,
    previous_error: float,
) -> tuple[float, float, float, float]:
    """Unsaturated local form of ``simulation_kernel._pid_update``."""

    kp, ki, kd = pid
    raw_derivative = (error - previous_error) / sample_period_s
    alpha = sample_period_s / (filter_time_s + sample_period_s)
    next_derivative = derivative + alpha * (raw_derivative - derivative)
    next_integral = integral + ki * error * sample_period_s
    command = kp * error + next_integral + kd * next_derivative
    return command, next_integral, next_derivative, error


def _pid_values(
    parameters: np.ndarray, loop: str
) -> tuple[float, float, float]:
    indices = {
        "position": (0, 1, 2),
        "speed": (3, 4, 5),
        "current": (8, 9, 10),
    }[loop]
    return tuple(float(parameters[index]) for index in indices)  # type: ignore[return-value]


def _friction_step(
    config: PhysicsMotorConfig,
    motor: MotorParameters,
    speed_rad_s: float,
    bristle_state: float,
    sample_period_s: float,
) -> tuple[float, float]:
    mode = 1 if config.active_friction_model == "lugre" else 0
    torque, next_bristle, _ = lugre_friction_step(
        mode,
        speed_rad_s,
        bristle_state,
        sample_period_s,
        motor.friction_stiffness_nm_per_rad,
        motor.friction_damping_nm_s_per_rad,
        motor.viscous_friction_nm_s_per_rad,
        motor.stribeck_velocity_rad_s,
        motor.coulomb_friction_nm,
        motor.static_friction_nm,
        float(config.friction_model["stribeck_exponent"]),
    )
    return float(torque), float(next_bristle)


def _sample_period(config: PhysicsMotorConfig, loop: str) -> float:
    """Read one loop period through the schema-3 multirate API."""

    period = float(config.sample_period_s_for(loop))
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError(f"{loop} sample period must be positive and finite")
    return period


def _integer_ratio(slower_period_s: float, faster_period_s: float) -> int:
    """Return an exact integer multirate ratio or reject an ambiguous schedule."""

    ratio_float = slower_period_s / faster_period_s
    ratio = int(round(ratio_float))
    if ratio <= 0 or not np.isclose(ratio_float, ratio, rtol=0.0, atol=1e-12):
        raise ValueError(
            "multirate periods must form positive integer update ratios"
        )
    return ratio


def _current_step(
    config: PhysicsMotorConfig,
    motor: MotorParameters,
    parameters: np.ndarray,
    state: np.ndarray,
    external_error: float,
) -> tuple[np.ndarray, float]:
    dt = _sample_period(config, "current")
    integral, derivative, previous_error, applied_voltage, current = state
    command, integral, derivative, previous_error = _pid_step(
        external_error,
        _pid_values(parameters, "current"),
        config.derivative_filter_s["current"],
        dt,
        integral,
        derivative,
        previous_error,
    )
    delay_alpha = dt / (motor.current_delay_s + dt)
    applied_voltage += delay_alpha * (command - applied_voltage)
    current += dt * (
        applied_voltage - motor.resistance_ohm * current
    ) / motor.inductance_h
    return (
        np.asarray(
            [integral, derivative, previous_error, applied_voltage, current],
            dtype=np.float64,
        ),
        float(state[4]),
    )


def _speed_step(
    config: PhysicsMotorConfig,
    motor: MotorParameters,
    parameters: np.ndarray,
    state: np.ndarray,
    external_error: float,
) -> tuple[np.ndarray, float]:
    current_dt = _sample_period(config, "current")
    speed_dt = _sample_period(config, "speed")
    current_steps = _integer_ratio(speed_dt, current_dt)
    (
        speed_integral,
        speed_derivative,
        speed_previous_error,
        current_integral,
        current_derivative,
        current_previous_error,
        current,
        speed,
        applied_voltage,
        speed_feedback,
        previous_speed_feedback,
        estimated_load,
        bristle_state,
    ) = state

    # The speed measurement, speed PIDF and DOBC update on the 5 kHz speed
    # clock.  ``current_reference`` is a zero-order-held command over all eight
    # 40 kHz current/motor substeps in the configured 40 kHz / 5 kHz schedule.
    speed_alpha = speed_dt / (motor.speed_measurement_delay_s + speed_dt)
    speed_feedback += speed_alpha * (speed - speed_feedback)
    feedback_output = float(speed_feedback)
    iq_pid, speed_integral, speed_derivative, speed_previous_error = _pid_step(
        external_error,
        _pid_values(parameters, "speed"),
        config.derivative_filter_s["speed"],
        speed_dt,
        speed_integral,
        speed_derivative,
        speed_previous_error,
    )

    nominal = config.nominal
    measured_acceleration = (
        speed_feedback - previous_speed_feedback
    ) / speed_dt
    raw_load_estimate = nominal.torque_constant_nm_per_a * current - (
        nominal.inertia_kg_m2 * measured_acceleration
        + nominal.viscous_friction_nm_s_per_rad * speed_feedback
    )
    dobc_alpha = speed_dt / (float(parameters[7]) + speed_dt)
    estimated_load += dobc_alpha * (raw_load_estimate - estimated_load)
    current_reference = iq_pid + (
        float(parameters[6]) / nominal.torque_constant_nm_per_a * estimated_load
    )

    delay_alpha = current_dt / (motor.current_delay_s + current_dt)
    for _ in range(current_steps):
        current_error = current_reference - current
        (
            voltage_command,
            current_integral,
            current_derivative,
            current_previous_error,
        ) = _pid_step(
            current_error,
            _pid_values(parameters, "current"),
            config.derivative_filter_s["current"],
            current_dt,
            current_integral,
            current_derivative,
            current_previous_error,
        )
        applied_voltage += delay_alpha * (voltage_command - applied_voltage)
        current += current_dt * (
            applied_voltage - motor.resistance_ohm * current
        ) / motor.inductance_h

        friction_torque, bristle_state = _friction_step(
            config, motor, speed, bristle_state, current_dt
        )
        speed += current_dt * (
            motor.torque_constant_nm_per_a * current - friction_torque
        ) / motor.inertia_kg_m2
    return (
        np.asarray(
            [
                speed_integral,
                speed_derivative,
                speed_previous_error,
                current_integral,
                current_derivative,
                current_previous_error,
                current,
                speed,
                applied_voltage,
                speed_feedback,
                speed_feedback,
                estimated_load,
                bristle_state,
            ],
            dtype=np.float64,
        ),
        feedback_output,
    )


def _position_step(
    config: PhysicsMotorConfig,
    motor: MotorParameters,
    parameters: np.ndarray,
    state: np.ndarray,
    external_error: float,
) -> tuple[np.ndarray, float]:
    current_dt = _sample_period(config, "current")
    speed_dt = _sample_period(config, "speed")
    position_dt = _sample_period(config, "position")
    current_steps = _integer_ratio(speed_dt, current_dt)
    speed_steps = _integer_ratio(position_dt, speed_dt)
    (
        position_integral,
        position_derivative,
        position_previous_error,
        speed_integral,
        speed_derivative,
        speed_previous_error,
        current_integral,
        current_derivative,
        current_previous_error,
        current,
        speed,
        position,
        applied_voltage,
        speed_feedback,
        position_feedback,
        previous_speed_feedback,
        estimated_load,
        bristle_state,
    ) = state

    # The position PIDF and position measurement update once per position
    # sample.  Its speed command is held while the embedded speed/DOBC loop
    # executes ``speed_steps`` times, and every speed command is in turn held
    # for ``current_steps`` current/motor updates.
    position_alpha = position_dt / (
        motor.position_measurement_delay_s + position_dt
    )
    position_feedback += position_alpha * (position - position_feedback)
    feedback_output = float(position_feedback)

    (
        speed_command,
        position_integral,
        position_derivative,
        position_previous_error,
    ) = _pid_step(
        external_error,
        _pid_values(parameters, "position"),
        config.derivative_filter_s["position"],
        position_dt,
        position_integral,
        position_derivative,
        position_previous_error,
    )
    nominal = config.nominal
    speed_alpha = speed_dt / (motor.speed_measurement_delay_s + speed_dt)
    dobc_alpha = speed_dt / (float(parameters[7]) + speed_dt)
    delay_alpha = current_dt / (motor.current_delay_s + current_dt)
    for _ in range(speed_steps):
        speed_feedback += speed_alpha * (speed - speed_feedback)
        speed_error = speed_command - speed_feedback
        (
            iq_pid,
            speed_integral,
            speed_derivative,
            speed_previous_error,
        ) = _pid_step(
            speed_error,
            _pid_values(parameters, "speed"),
            config.derivative_filter_s["speed"],
            speed_dt,
            speed_integral,
            speed_derivative,
            speed_previous_error,
        )

        measured_acceleration = (
            speed_feedback - previous_speed_feedback
        ) / speed_dt
        raw_load_estimate = nominal.torque_constant_nm_per_a * current - (
            nominal.inertia_kg_m2 * measured_acceleration
            + nominal.viscous_friction_nm_s_per_rad * speed_feedback
        )
        estimated_load += dobc_alpha * (raw_load_estimate - estimated_load)
        current_reference = iq_pid + (
            float(parameters[6])
            / nominal.torque_constant_nm_per_a
            * estimated_load
        )
        previous_speed_feedback = speed_feedback

        for _ in range(current_steps):
            current_error = current_reference - current
            (
                voltage_command,
                current_integral,
                current_derivative,
                current_previous_error,
            ) = _pid_step(
                current_error,
                _pid_values(parameters, "current"),
                config.derivative_filter_s["current"],
                current_dt,
                current_integral,
                current_derivative,
                current_previous_error,
            )
            applied_voltage += delay_alpha * (
                voltage_command - applied_voltage
            )
            current += current_dt * (
                applied_voltage - motor.resistance_ohm * current
            ) / motor.inductance_h

            friction_torque, bristle_state = _friction_step(
                config, motor, speed, bristle_state, current_dt
            )
            speed += current_dt * (
                motor.torque_constant_nm_per_a * current - friction_torque
            ) / motor.inertia_kg_m2
            position += current_dt * speed
    return (
        np.asarray(
            [
                position_integral,
                position_derivative,
                position_previous_error,
                speed_integral,
                speed_derivative,
                speed_previous_error,
                current_integral,
                current_derivative,
                current_previous_error,
                current,
                speed,
                position,
                applied_voltage,
                speed_feedback,
                position_feedback,
                previous_speed_feedback,
                estimated_load,
                bristle_state,
            ],
            dtype=np.float64,
        ),
        feedback_output,
    )


@dataclass(frozen=True)
class DiscreteLoopModel:
    """Discrete return-ratio model at one loop's error summing junction."""

    loop: str
    sample_period_s: float
    state_names: tuple[str, ...]
    state_matrix: np.ndarray
    error_input: np.ndarray
    feedback_output: np.ndarray
    actual_output: np.ndarray

    @property
    def closed_loop_state_matrix(self) -> np.ndarray:
        return self.state_matrix - np.outer(self.error_input, self.feedback_output)

    @property
    def closed_loop_poles(self) -> np.ndarray:
        return np.linalg.eigvals(self.closed_loop_state_matrix)

    @property
    def closed_loop_io_poles(self) -> np.ndarray:
        """Return poles visible from the reference-to-output transfer.

        The LuGre state and controller integral can create a continuum of zero-
        speed equilibria.  Its exact unit pole has zero reference-to-current,
        speed and position residue and must not be mistaken for an unstable I/O
        mode.  Other modes are retained using both actual and feedback outputs.
        """

        poles, eigenvectors = np.linalg.eig(self.closed_loop_state_matrix)
        coefficients = np.linalg.solve(eigenvectors, self.error_input)
        actual_residues = (self.actual_output @ eigenvectors) * coefficients
        feedback_residues = (self.feedback_output @ eigenvectors) * coefficients
        residue_magnitude = np.maximum(
            np.abs(actual_residues), np.abs(feedback_residues)
        )
        tolerance = 1e-10 * max(1.0, float(np.max(residue_magnitude)))
        visible = poles[residue_magnitude > tolerance]
        if visible.size == 0:
            raise ValueError("closed loop has no observable reference response")
        return visible

    def _response(
        self,
        state_matrix: np.ndarray,
        output: np.ndarray,
        frequency_hz: np.ndarray,
    ) -> np.ndarray:
        frequencies = np.asarray(frequency_hz, dtype=np.float64)
        if frequencies.ndim != 1 or not np.isfinite(frequencies).all():
            raise ValueError("frequency_hz must be one finite vector")
        nyquist_hz = 0.5 / self.sample_period_s
        if np.any(frequencies <= 0.0) or np.any(frequencies >= nyquist_hz):
            raise ValueError("discrete frequency grid must lie strictly below Nyquist")
        z = np.exp(1j * 2.0 * np.pi * frequencies * self.sample_period_s)
        identity = np.eye(state_matrix.shape[0], dtype=np.complex128)
        matrices = z[:, None, None] * identity[None, :, :] - state_matrix
        inputs = np.broadcast_to(
            self.error_input[None, :, None],
            (frequencies.size, state_matrix.shape[0], 1),
        )
        states = np.linalg.solve(matrices, inputs)[..., 0]
        return np.einsum("j,ij->i", output, states)

    def open_loop_response(self, frequency_hz: np.ndarray) -> np.ndarray:
        return self._response(
            self.state_matrix, self.feedback_output, frequency_hz
        )

    def closed_actual_response(self, frequency_hz: np.ndarray) -> np.ndarray:
        return self._response(
            self.closed_loop_state_matrix, self.actual_output, frequency_hz
        )

    def closed_step_response(self, reference: float, points: int) -> np.ndarray:
        if points <= 0:
            raise ValueError("points must be positive")
        state = np.zeros(len(self.state_names), dtype=np.float64)
        outputs = np.empty(points, dtype=np.float64)
        closed = self.closed_loop_state_matrix
        for index in range(points):
            state = closed @ state + self.error_input * float(reference)
            outputs[index] = float(self.actual_output @ state)
        return outputs


def _actual_output(loop: str, state_names: tuple[str, ...]) -> np.ndarray:
    name = {"current": "current", "speed": "speed", "position": "position"}[
        loop
    ]
    output = np.zeros(len(state_names), dtype=np.float64)
    output[state_names.index(name)] = 1.0
    return output


def build_discrete_loop_model(
    config: PhysicsMotorConfig,
    motor: MotorParameters,
    controller_parameters: np.ndarray,
    loop: str,
    *,
    linearization_step: float = 1e-7,
) -> DiscreteLoopModel:
    """Linearize one unsaturated simulator loop at the zero operating point."""

    if loop not in LOOP_STATE_NAMES:
        raise ValueError(f"unknown control loop: {loop}")
    parameters = np.asarray(controller_parameters, dtype=np.float64)
    if parameters.shape != (11,) or not np.isfinite(parameters).all():
        raise ValueError("controller_parameters must be finite with shape (11,)")
    if not np.isfinite(linearization_step) or linearization_step <= 0.0:
        raise ValueError("linearization_step must be positive and finite")
    loop_period_s = _sample_period(config, loop)
    current_period_s = _sample_period(config, "current")
    speed_period_s = _sample_period(config, "speed")
    _integer_ratio(speed_period_s, current_period_s)
    _integer_ratio(_sample_period(config, "position"), speed_period_s)
    nyquist_hz = 0.5 / loop_period_s
    if nyquist_hz <= 0.0:
        raise ValueError("sample period must define a positive Nyquist frequency")

    state_names = LOOP_STATE_NAMES[loop]
    transition: Callable[[np.ndarray, float], tuple[np.ndarray, float]] = {
        "current": lambda state, error: _current_step(
            config, motor, parameters, state, error
        ),
        "speed": lambda state, error: _speed_step(
            config, motor, parameters, state, error
        ),
        "position": lambda state, error: _position_step(
            config, motor, parameters, state, error
        ),
    }[loop]
    state_count = len(state_names)
    origin = np.zeros(state_count, dtype=np.float64)
    state_matrix = np.empty((state_count, state_count), dtype=np.float64)
    feedback_output = np.empty(state_count, dtype=np.float64)
    for index in range(state_count):
        perturbation = np.zeros(state_count, dtype=np.float64)
        perturbation[index] = linearization_step
        positive_state, positive_feedback = transition(perturbation, 0.0)
        negative_state, negative_feedback = transition(-perturbation, 0.0)
        state_matrix[:, index] = (
            positive_state - negative_state
        ) / (2.0 * linearization_step)
        feedback_output[index] = (
            positive_feedback - negative_feedback
        ) / (2.0 * linearization_step)

    positive_state, positive_feedback = transition(origin, linearization_step)
    negative_state, negative_feedback = transition(origin, -linearization_step)
    error_input = (positive_state - negative_state) / (2.0 * linearization_step)
    direct_feedback = (positive_feedback - negative_feedback) / (
        2.0 * linearization_step
    )
    if abs(direct_feedback) > 1e-10:
        raise ValueError("loop linearization unexpectedly contains direct feedback")
    values = np.concatenate(
        [
            state_matrix.reshape(-1),
            error_input,
            feedback_output,
        ]
    )
    if not np.isfinite(values).all():
        raise ValueError("discrete loop linearization is non-finite")
    return DiscreteLoopModel(
        loop=loop,
        sample_period_s=loop_period_s,
        state_names=state_names,
        state_matrix=state_matrix,
        error_input=error_input,
        feedback_output=feedback_output,
        actual_output=_actual_output(loop, state_names),
    )


def build_discrete_loop_models(
    config: PhysicsMotorConfig,
    motor: MotorParameters,
    controller_parameters: np.ndarray,
) -> dict[str, DiscreteLoopModel]:
    return {
        loop: build_discrete_loop_model(
            config, motor, controller_parameters, loop
        )
        for loop in ("current", "speed", "position")
    }
