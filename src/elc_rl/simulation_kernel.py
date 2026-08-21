"""Compiled numerical kernel for the cascaded motor simulation."""

from __future__ import annotations

import math

from numba import njit
import numpy as np


SCENARIO_CURRENT = 0
SCENARIO_SPEED = 1
SCENARIO_POSITION = 2
SCENARIO_DISTURBANCE = 3


@njit(cache=True, nogil=True)
def _clip_scalar(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


@njit(cache=True, nogil=True)
def _isclose_scalar(left: float, right: float) -> bool:
    if math.isnan(left) or math.isnan(right):
        return False
    if left == right:
        return True
    return abs(left - right) <= 1e-8 + 1e-5 * abs(right)


@njit(cache=True, nogil=True)
def lugre_friction_step(
    mode_code: int,
    omega_rad_s: float,
    bristle_state: float,
    sample_period_s: float,
    friction_stiffness_nm_per_rad: float,
    friction_damping_nm_s_per_rad: float,
    viscous_friction_nm_s_per_rad: float,
    stribeck_velocity_rad_s: float,
    coulomb_friction_nm: float,
    static_friction_nm: float,
    stribeck_exponent: float,
) -> tuple[float, float, float]:
    """Advance one friction interval and return torque, state, and state rate.

    ``mode_code == 0`` is the legacy viscous-only model.  ``mode_code == 1``
    is the symmetric LuGre model.  For LuGre, velocity is held constant over
    the interval so the bristle state can be integrated exactly.  The returned
    rate and torque are evaluated at the updated state; this preserves the
    standard LuGre output equation without an explicit-Euler state update.
    """

    if mode_code == 0:
        return viscous_friction_nm_s_per_rad * omega_rad_s, 0.0, 0.0
    if mode_code != 1:
        raise ValueError("friction mode must be 0 (viscous) or 1 (LuGre)")
    if sample_period_s <= 0.0:
        raise ValueError("sample period must be positive")
    if friction_stiffness_nm_per_rad <= 0.0:
        raise ValueError("LuGre friction stiffness must be positive")
    if friction_damping_nm_s_per_rad < 0.0:
        raise ValueError("LuGre friction damping must be non-negative")
    if viscous_friction_nm_s_per_rad < 0.0:
        raise ValueError("viscous friction must be non-negative")
    if stribeck_velocity_rad_s <= 0.0:
        raise ValueError("Stribeck velocity must be positive")
    if coulomb_friction_nm <= 0.0:
        raise ValueError("Coulomb friction must be positive")
    if static_friction_nm < coulomb_friction_nm:
        raise ValueError("static friction must not be below Coulomb friction")
    if stribeck_exponent <= 0.0:
        raise ValueError("Stribeck exponent must be positive")

    absolute_speed = abs(omega_rad_s)
    if absolute_speed == 0.0:
        friction_torque = (
            friction_stiffness_nm_per_rad * bristle_state
        )
        return friction_torque, bristle_state, 0.0

    velocity_ratio = absolute_speed / stribeck_velocity_rad_s
    stribeck_torque = coulomb_friction_nm + (
        static_friction_nm - coulomb_friction_nm
    ) * math.exp(-(velocity_ratio**stribeck_exponent))
    state_decay_rate = (
        friction_stiffness_nm_per_rad * absolute_speed / stribeck_torque
    )
    speed_sign = 1.0 if omega_rad_s > 0.0 else -1.0
    steady_bristle_state = (
        speed_sign * stribeck_torque / friction_stiffness_nm_per_rad
    )
    next_bristle_state = steady_bristle_state + (
        bristle_state - steady_bristle_state
    ) * math.exp(-state_decay_rate * sample_period_s)
    bristle_rate = (
        omega_rad_s - state_decay_rate * next_bristle_state
    )
    friction_torque = (
        friction_stiffness_nm_per_rad * next_bristle_state
        + friction_damping_nm_s_per_rad * bristle_rate
        + viscous_friction_nm_s_per_rad * omega_rad_s
    )
    return friction_torque, next_bristle_state, bristle_rate


@njit(cache=True, nogil=True)
def _pid_update(
    error: float,
    kp: float,
    ki: float,
    kd: float,
    filter_time_s: float,
    sample_period_s: float,
    integral: float,
    derivative: float,
    previous_error: float,
    lower: float,
    upper: float,
) -> tuple[float, float, float, float, float]:
    raw_derivative = (error - previous_error) / sample_period_s
    alpha = sample_period_s / (filter_time_s + sample_period_s)
    derivative += alpha * (raw_derivative - derivative)
    candidate_integral = integral + ki * error * sample_period_s
    unsaturated = kp * error + candidate_integral + kd * derivative
    saturated = _clip_scalar(unsaturated, lower, upper)
    drives_further_into_saturation = (
        (unsaturated > upper and error > 0.0)
        or (unsaturated < lower and error < 0.0)
    )
    if drives_further_into_saturation:
        unsaturated = kp * error + integral + kd * derivative
        saturated = _clip_scalar(unsaturated, lower, upper)
    else:
        integral = candidate_integral
    return saturated, unsaturated, integral, derivative, error


@njit(cache=True, nogil=True)
def simulate_scenario_kernel(
    scenario_code: int,
    sample_periods_s: np.ndarray,
    point_count: int,
    controller_parameters: np.ndarray,
    derivative_filters_s: np.ndarray,
    motor_parameters: np.ndarray,
    friction_settings: np.ndarray,
    nominal_inverse_parameters: np.ndarray,
    limits: np.ndarray,
    scenario_values: np.ndarray,
    encoder_lsb: float,
    encoder_noise: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    bool,
]:
    """Run one deterministic multi-rate scenario using primitive inputs.

    ``sample_periods_s`` is ordered as current, speed, position. The current
    period is the plant/LuGre integration base step. Speed, position and DOBC
    execute on integer base-step boundaries and their outputs are held between
    updates.
    """

    if sample_periods_s.ndim != 1 or sample_periods_s.size != 3:
        raise ValueError("sample periods must have current, speed, position")
    current_dt = sample_periods_s[0]
    speed_dt = sample_periods_s[1]
    position_dt = sample_periods_s[2]
    if current_dt <= 0.0 or speed_dt <= 0.0 or position_dt <= 0.0:
        raise ValueError("sample periods must be positive")
    speed_ratio_float = speed_dt / current_dt
    position_ratio_float = position_dt / current_dt
    speed_update_ratio = int(round(speed_ratio_float))
    position_update_ratio = int(round(position_ratio_float))
    if (
        speed_update_ratio < 1
        or position_update_ratio < 1
        or abs(speed_ratio_float - speed_update_ratio) > 1e-12
        or abs(position_ratio_float - position_update_ratio) > 1e-12
    ):
        raise ValueError("outer-loop periods must be integer current-step multiples")
    dt = current_dt
    time_s = np.arange(point_count, dtype=np.float64) * dt
    output = np.zeros(point_count, dtype=np.float64)
    reference = np.zeros(point_count, dtype=np.float64)
    primary_control = np.zeros(point_count, dtype=np.float64)
    voltage = np.zeros(point_count, dtype=np.float64)
    current = np.zeros(point_count, dtype=np.float64)
    speed = np.zeros(point_count, dtype=np.float64)
    position = np.zeros(point_count, dtype=np.float64)
    current_reference = np.zeros(point_count, dtype=np.float64)
    speed_reference = np.zeros(point_count, dtype=np.float64)
    load_torque = np.zeros(point_count, dtype=np.float64)
    friction_torque = np.zeros(point_count, dtype=np.float64)
    bristle_state_trace = np.zeros(point_count, dtype=np.float64)
    bristle_rate = np.zeros(point_count, dtype=np.float64)

    position_kp = controller_parameters[0]
    position_ki = controller_parameters[1]
    position_kd = controller_parameters[2]
    speed_kp = controller_parameters[3]
    speed_ki = controller_parameters[4]
    speed_kd = controller_parameters[5]
    dobc_gain = controller_parameters[6]
    dobc_time = controller_parameters[7]
    current_kp = controller_parameters[8]
    current_ki = controller_parameters[9]
    current_kd = controller_parameters[10]

    inductance_h = motor_parameters[0]
    resistance_ohm = motor_parameters[1]
    current_delay_s = motor_parameters[2]
    viscous_friction_nm_s_per_rad = motor_parameters[5]
    speed_measurement_delay_s = motor_parameters[7]
    inertia_kg_m2 = motor_parameters[8]
    torque_constant_nm_per_a = motor_parameters[9]
    position_measurement_delay_s = motor_parameters[10]

    friction_mode = int(friction_settings[0])
    stribeck_exponent = friction_settings[1]
    initial_bristle_state = friction_settings[2]
    friction_stiffness_nm_per_rad = 0.0
    friction_damping_nm_s_per_rad = 0.0
    stribeck_velocity_rad_s = 1.0
    coulomb_friction_nm = 1.0
    static_friction_nm = 1.0
    if friction_mode == 1:
        friction_stiffness_nm_per_rad = motor_parameters[3]
        friction_damping_nm_s_per_rad = motor_parameters[4]
        stribeck_velocity_rad_s = motor_parameters[6]
        coulomb_friction_nm = motor_parameters[11]
        static_friction_nm = motor_parameters[12]

    nominal_viscous_friction = nominal_inverse_parameters[5]
    nominal_inertia = nominal_inverse_parameters[8]
    nominal_torque_constant = nominal_inverse_parameters[9]

    current_limit = limits[0]
    speed_command_limit = limits[1]
    voltage_limit = limits[2]
    termination_speed = limits[3]

    requested_current = scenario_values[0]
    requested_speed = scenario_values[1]
    requested_position = scenario_values[2]
    load_torque_step = scenario_values[3]
    disturbance_start = scenario_values[4]

    position_integral = 0.0
    position_derivative = 0.0
    position_previous_error = 0.0
    speed_integral = 0.0
    speed_derivative = 0.0
    speed_previous_error = 0.0
    current_integral = 0.0
    current_derivative = 0.0
    current_previous_error = 0.0

    i_a = 0.0
    omega = 0.0
    theta = 0.0
    applied_voltage = 0.0
    omega_feedback = 0.0
    theta_feedback = 0.0
    previous_theta_observed = 0.0
    previous_omega_feedback = 0.0
    estimated_load = 0.0
    speed_command = 0.0
    iq_reference = 0.0
    theta_observed = 0.0
    bristle_state = initial_bristle_state if friction_mode == 1 else 0.0
    saturation_count = 0
    terminated = False

    current_delay_alpha = dt / (current_delay_s + dt)
    speed_delay_alpha = speed_dt / (speed_measurement_delay_s + speed_dt)
    position_delay_alpha = position_dt / (
        position_measurement_delay_s + position_dt
    )
    dobc_alpha = speed_dt / (dobc_time + speed_dt)

    for index in range(point_count):
        now = time_s[index]
        position_update = index % position_update_ratio == 0
        speed_update = index % speed_update_ratio == 0

        if position_update:
            theta_feedback += position_delay_alpha * (theta - theta_feedback)
            if encoder_noise.size:
                theta_observed = theta_feedback + encoder_noise[index] * encoder_lsb
                theta_observed = np.round(theta_observed / encoder_lsb) * encoder_lsb
            else:
                theta_observed = theta_feedback

        if speed_update:
            if encoder_noise.size:
                encoder_speed = (
                    theta_observed - previous_theta_observed
                ) / speed_dt
                omega_feedback += speed_delay_alpha * (
                    encoder_speed - omega_feedback
                )
                previous_theta_observed = theta_observed
            else:
                omega_feedback += speed_delay_alpha * (omega - omega_feedback)

        if scenario_code == SCENARIO_CURRENT:
            iq_reference = requested_current
            speed_command = 0.0
            reference[index] = requested_current
        else:
            if scenario_code == SCENARIO_POSITION:
                if position_update:
                    position_error = (
                        (requested_position - theta_observed + np.pi)
                        % (2.0 * np.pi)
                        - np.pi
                    )
                    (
                        speed_command,
                        speed_unsaturated,
                        position_integral,
                        position_derivative,
                        position_previous_error,
                    ) = _pid_update(
                        position_error,
                        position_kp,
                        position_ki,
                        position_kd,
                        derivative_filters_s[0],
                        position_dt,
                        position_integral,
                        position_derivative,
                        position_previous_error,
                        -speed_command_limit,
                        speed_command_limit,
                    )
                    saturation_count += int(
                        not _isclose_scalar(speed_command, speed_unsaturated)
                    )
                reference[index] = requested_position
            else:
                speed_command = requested_speed
                reference[index] = requested_speed

            if speed_update:
                speed_error = speed_command - omega_feedback
                (
                    iq_pid,
                    iq_unsaturated,
                    speed_integral,
                    speed_derivative,
                    speed_previous_error,
                ) = _pid_update(
                    speed_error,
                    speed_kp,
                    speed_ki,
                    speed_kd,
                    derivative_filters_s[1],
                    speed_dt,
                    speed_integral,
                    speed_derivative,
                    speed_previous_error,
                    -current_limit,
                    current_limit,
                )
                saturation_count += int(
                    not _isclose_scalar(iq_pid, iq_unsaturated)
                )
                measured_acceleration = (
                    omega_feedback - previous_omega_feedback
                ) / speed_dt
                raw_load_estimate = nominal_torque_constant * i_a - (
                    nominal_inertia * measured_acceleration
                    + nominal_viscous_friction * omega_feedback
                )
                estimated_load += dobc_alpha * (
                    raw_load_estimate - estimated_load
                )
                iq_reference_unsaturated = (
                    iq_pid + dobc_gain / nominal_torque_constant * estimated_load
                )
                iq_reference = _clip_scalar(
                    iq_reference_unsaturated, -current_limit, current_limit
                )
                saturation_count += int(
                    not _isclose_scalar(iq_reference, iq_reference_unsaturated)
                )
                previous_omega_feedback = omega_feedback

        current_error = iq_reference - i_a
        (
            voltage_command,
            voltage_unsaturated,
            current_integral,
            current_derivative,
            current_previous_error,
        ) = _pid_update(
            current_error,
            current_kp,
            current_ki,
            current_kd,
            derivative_filters_s[2],
            dt,
            current_integral,
            current_derivative,
            current_previous_error,
            -voltage_limit,
            voltage_limit,
        )
        saturation_count += int(
            not _isclose_scalar(voltage_command, voltage_unsaturated)
        )
        applied_voltage += current_delay_alpha * (
            voltage_command - applied_voltage
        )

        di_dt = (
            applied_voltage - resistance_ohm * i_a
        ) / inductance_h
        i_a += dt * di_dt
        i_a = _clip_scalar(i_a, -1.25 * current_limit, 1.25 * current_limit)

        active_load = 0.0
        if (
            scenario_code == SCENARIO_DISTURBANCE
            and now >= disturbance_start
        ):
            active_load = load_torque_step
        (
            active_friction_torque,
            bristle_state,
            active_bristle_rate,
        ) = lugre_friction_step(
            friction_mode,
            omega,
            bristle_state,
            dt,
            friction_stiffness_nm_per_rad,
            friction_damping_nm_s_per_rad,
            viscous_friction_nm_s_per_rad,
            stribeck_velocity_rad_s,
            coulomb_friction_nm,
            static_friction_nm,
            stribeck_exponent,
        )
        domega_dt = (
            torque_constant_nm_per_a * i_a
            - active_friction_torque
            - active_load
        ) / inertia_kg_m2
        omega += dt * domega_dt
        theta += dt * omega

        current[index] = i_a
        speed[index] = omega
        position[index] = theta
        current_reference[index] = iq_reference
        speed_reference[index] = speed_command
        load_torque[index] = active_load
        friction_torque[index] = active_friction_torque
        bristle_state_trace[index] = bristle_state
        bristle_rate[index] = active_bristle_rate
        voltage[index] = voltage_command

        if scenario_code == SCENARIO_CURRENT:
            primary_control[index] = voltage_command
            output[index] = i_a
        elif scenario_code == SCENARIO_POSITION:
            primary_control[index] = speed_command
            output[index] = theta
        else:
            primary_control[index] = iq_reference
            output[index] = omega

        if (
            abs(omega) > termination_speed
            or not math.isfinite(i_a)
            or not math.isfinite(omega)
            or not math.isfinite(theta)
            or not math.isfinite(applied_voltage)
            or not math.isfinite(active_friction_torque)
            or not math.isfinite(bristle_state)
            or not math.isfinite(active_bristle_rate)
        ):
            terminated = True
            if index + 1 < point_count:
                for fill_index in range(index + 1, point_count):
                    output[fill_index] = output[index]
                    reference[fill_index] = reference[index]
                    primary_control[fill_index] = primary_control[index]
                    voltage[fill_index] = voltage[index]
                    current[fill_index] = current[index]
                    speed[fill_index] = speed[index]
                    position[fill_index] = position[index]
                    current_reference[fill_index] = current_reference[index]
                    speed_reference[fill_index] = speed_reference[index]
                    load_torque[fill_index] = load_torque[index]
                    friction_torque[fill_index] = friction_torque[index]
                    bristle_state_trace[fill_index] = (
                        bristle_state_trace[index]
                    )
                    bristle_rate[fill_index] = bristle_rate[index]
            break

    return (
        time_s,
        output,
        reference,
        primary_control,
        voltage,
        current,
        speed,
        position,
        current_reference,
        speed_reference,
        load_torque,
        friction_torque,
        bristle_state_trace,
        bristle_rate,
        saturation_count,
        terminated,
    )
