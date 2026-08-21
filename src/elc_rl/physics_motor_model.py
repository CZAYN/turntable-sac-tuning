"""Mentor-supplied physics motor model and deterministic simulation utilities.

The model is a single-axis, FOC-decoupled servo approximation.  It keeps the
electrical, mechanical, measurement-delay and saturation semantics explicit so
that the reinforcement-learning environment does not need to infer them from
measured frequency responses.  Measured FRFs remain independent validation
data.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .simulation_kernel import (
    SCENARIO_CURRENT,
    SCENARIO_DISTURBANCE,
    SCENARIO_POSITION,
    SCENARIO_SPEED,
    simulate_scenario_kernel,
)


MODEL_PARAMETER_NAMES = (
    "inductance_h",
    "resistance_ohm",
    "current_delay_s",
    "friction_stiffness_nm_per_rad",
    "friction_damping_nm_s_per_rad",
    "viscous_friction_nm_s_per_rad",
    "stribeck_velocity_rad_s",
    "speed_measurement_delay_s",
    "inertia_kg_m2",
    "torque_constant_nm_per_a",
    "position_measurement_delay_s",
    "coulomb_friction_nm",
    "static_friction_nm",
)

LUGRE_FRICTION_LEVEL_NAMES = (
    "coulomb_friction_nm",
    "static_friction_nm",
)
SUPPORTED_FRICTION_MODELS = {"viscous_B_only", "lugre"}
UNIDENTIFIED_FRICTION_STATUS = "unidentified_disabled_zero_sentinel"
FRICTION_TORQUE_SIGN_CONVENTION = (
    "tau_f_has_velocity_sign_and_is_subtracted_in_"
    "J_omega_dot=Kt_iq-tau_f-tau_load"
)

PHYSICS_CONFIG_RELATIVE_PATH = Path("config") / "motor_physics.json"
PHYSICS_ENSEMBLE_RELATIVE_PATH = (
    Path("data") / "processed" / "physics_motor_ensemble.npz"
)
PHYSICS_ENSEMBLE_MANIFEST_RELATIVE_PATH = (
    Path("data") / "processed" / "physics_motor_ensemble_manifest.json"
)
PHYSICS_CONFIG_SCHEMA_VERSION = 3
PHYSICS_ENSEMBLE_SCHEMA_VERSION = 3
SAMPLE_PERIOD_LOOP_NAMES = ("current", "speed", "position")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MotorParameters:
    """One physical motor instance, expressed entirely in SI units."""

    inductance_h: float
    resistance_ohm: float
    current_delay_s: float
    friction_stiffness_nm_per_rad: float
    friction_damping_nm_s_per_rad: float
    viscous_friction_nm_s_per_rad: float
    stribeck_velocity_rad_s: float
    speed_measurement_delay_s: float
    inertia_kg_m2: float
    torque_constant_nm_per_a: float
    position_measurement_delay_s: float
    coulomb_friction_nm: float
    static_friction_nm: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "MotorParameters":
        return cls(**{name: float(values[name]) for name in MODEL_PARAMETER_NAMES})

    @classmethod
    def from_array(cls, values: np.ndarray) -> "MotorParameters":
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(MODEL_PARAMETER_NAMES),):
            raise ValueError("motor parameter vector has an invalid shape")
        return cls(**dict(zip(MODEL_PARAMETER_NAMES, array.tolist())))

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [getattr(self, name) for name in MODEL_PARAMETER_NAMES], dtype=np.float64
        )

    def validate(self, *, friction_model: str = "viscous_B_only") -> None:
        values = self.as_array()
        if friction_model not in SUPPORTED_FRICTION_MODELS:
            raise ValueError(f"unsupported friction model: {friction_model}")
        if not np.isfinite(values).all():
            raise ValueError("all physics motor parameters must be finite")
        if np.any(values[: -len(LUGRE_FRICTION_LEVEL_NAMES)] <= 0.0):
            raise ValueError("non-LuGre-level physics motor parameters must be positive")
        if self.coulomb_friction_nm < 0.0 or self.static_friction_nm < 0.0:
            raise ValueError("LuGre friction levels must be non-negative")
        if self.static_friction_nm < self.coulomb_friction_nm:
            raise ValueError(
                "static_friction_nm must be greater than or equal to "
                "coulomb_friction_nm"
            )
        if friction_model == "lugre" and self.coulomb_friction_nm <= 0.0:
            raise ValueError(
                "active LuGre friction requires static_friction_nm >= "
                "coulomb_friction_nm > 0"
            )

    @property
    def electrical_time_constant_s(self) -> float:
        return self.inductance_h / self.resistance_ohm

    @property
    def mechanical_time_constant_s(self) -> float:
        return self.inertia_kg_m2 / self.viscous_friction_nm_s_per_rad


@dataclass(frozen=True)
class PhysicsMotorConfig:
    """Validated configuration plus its nominal motor and derived sections."""

    project_root: Path
    path: Path
    payload: dict[str, Any]
    nominal: MotorParameters

    @property
    def sample_periods_s(self) -> dict[str, float]:
        """Return the independent digital-controller periods in seconds."""

        values = self.payload["sample_periods_s"]
        return {loop: float(values[loop]) for loop in SAMPLE_PERIOD_LOOP_NAMES}

    def sample_period_s_for(self, loop: str) -> float:
        """Return one loop's digital update period."""

        if loop not in SAMPLE_PERIOD_LOOP_NAMES:
            raise ValueError(f"unknown controller loop: {loop}")
        return self.sample_periods_s[loop]

    @property
    def base_sample_period_s(self) -> float:
        """Return the plant integration/current-controller base period."""

        return self.sample_period_s_for("current")

    @property
    def sample_period_s(self) -> float:
        """Compatibility alias for the current-loop/base period.

        New code must call :meth:`sample_period_s_for` whenever the loop is
        known. This alias no longer means that all loops share one clock.
        """

        return self.base_sample_period_s

    @property
    def controller_update_ratios(self) -> dict[str, int]:
        """Return integer loop update periods measured in base steps."""

        base = self.base_sample_period_s
        return {
            loop: int(round(self.sample_period_s_for(loop) / base))
            for loop in SAMPLE_PERIOD_LOOP_NAMES
        }

    @property
    def uncertainty_fraction(self) -> dict[str, float]:
        return {
            name: float(self.payload["uncertainty_fraction"][name])
            for name in MODEL_PARAMETER_NAMES
        }

    @property
    def friction_model(self) -> dict[str, Any]:
        return dict(self.payload["friction_model"])

    @property
    def active_friction_model(self) -> str:
        return str(self.payload["friction_model"]["active"])

    @property
    def limits(self) -> dict[str, Any]:
        return dict(self.payload["simulation_limits"])

    @property
    def scenarios(self) -> dict[str, float]:
        return {
            key: float(value) for key, value in self.payload["scenarios"].items()
        }

    @property
    def derivative_filter_s(self) -> dict[str, float]:
        return {
            loop: float(value)
            for loop, value in self.payload["controller_design"][
                "derivative_filter_s"
            ].items()
        }

    def validate(self) -> None:
        schema_version = int(self.payload.get("schema_version", -1))
        if schema_version != PHYSICS_CONFIG_SCHEMA_VERSION:
            if schema_version == 2 or "sample_period_s" in self.payload:
                raise ValueError(
                    "single-rate physics configuration/state is unsupported; "
                    "migrate to schema 3 sample_periods_s and rebuild all "
                    "derived physics data"
                )
            raise ValueError(
                "unsupported physics motor configuration schema; expected schema 3"
            )
        if self.payload["model_id"] != "mentor_motor_physics":
            raise ValueError("unexpected physics motor model_id")
        raw_periods = self.payload.get("sample_periods_s")
        if not isinstance(raw_periods, Mapping) or set(raw_periods) != set(
            SAMPLE_PERIOD_LOOP_NAMES
        ):
            raise ValueError(
                "sample_periods_s must contain exactly current, speed, and position"
            )
        periods = self.sample_periods_s
        if not all(np.isfinite(value) and value > 0.0 for value in periods.values()):
            raise ValueError("all controller sample periods must be finite and positive")
        base = periods["current"]
        for loop in SAMPLE_PERIOD_LOOP_NAMES:
            ratio = periods[loop] / base
            rounded = round(ratio)
            if rounded < 1 or not np.isclose(ratio, rounded, rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"{loop} sample period must be an integer multiple of the "
                    "current/base sample period"
                )
        friction = self.friction_model
        active_friction_model = self.active_friction_model
        if active_friction_model not in SUPPORTED_FRICTION_MODELS:
            raise ValueError(f"unsupported friction model: {active_friction_model}")
        exponent = float(friction["stribeck_exponent"])
        if not np.isfinite(exponent) or exponent <= 0.0:
            raise ValueError("LuGre Stribeck exponent must be finite and positive")
        initial_bristle_state = float(friction["initial_bristle_state_rad"])
        if not np.isfinite(initial_bristle_state):
            raise ValueError("initial LuGre bristle state must be finite")
        if friction["torque_sign_convention"] != FRICTION_TORQUE_SIGN_CONVENTION:
            raise ValueError("unexpected LuGre friction torque sign convention")
        parameter_status = friction["parameter_status"]
        for name in LUGRE_FRICTION_LEVEL_NAMES:
            if name not in parameter_status:
                raise ValueError(f"missing friction parameter status for {name}")
        self.nominal.validate(friction_model=active_friction_model)
        if (
            active_friction_model == "lugre"
            and abs(initial_bristle_state)
            > self.nominal.static_friction_nm
            / self.nominal.friction_stiffness_nm_per_rad
        ):
            raise ValueError(
                "initial LuGre bristle state exceeds the static-friction bound"
            )
        for name, fraction in self.uncertainty_fraction.items():
            if not 0.0 <= fraction <= 0.5:
                raise ValueError(f"invalid uncertainty fraction for {name}")
        for name in LUGRE_FRICTION_LEVEL_NAMES:
            value = float(getattr(self.nominal, name))
            uncertainty = self.uncertainty_fraction[name]
            if value == 0.0 and uncertainty != 0.0:
                raise ValueError(
                    f"zero sentinel {name} must have zero uncertainty"
                )
            if (
                active_friction_model == "lugre"
                and parameter_status[name] == UNIDENTIFIED_FRICTION_STATUS
            ):
                raise ValueError(f"active LuGre parameter is still unidentified: {name}")
        minimum_static = self.nominal.static_friction_nm * (
            1.0 - self.uncertainty_fraction["static_friction_nm"]
        )
        maximum_coulomb = self.nominal.coulomb_friction_nm * (
            1.0 + self.uncertainty_fraction["coulomb_friction_nm"]
        )
        if minimum_static < maximum_coulomb:
            raise ValueError(
                "LuGre uncertainty ranges can violate static_friction_nm >= "
                "coulomb_friction_nm"
            )
        limits = self.limits
        if not (
            float(limits["hard_current_a"])
            >= float(limits["training_current_a"])
            > 0.0
        ):
            raise ValueError("current simulation limits are inconsistent")
        if not (
            float(limits["termination_speed_rad_s"])
            > float(limits["hard_speed_rad_s"])
            >= float(limits["training_speed_rad_s"])
            > 0.0
        ):
            raise ValueError("speed simulation limits are inconsistent")


@lru_cache(maxsize=4)
def _cached_config(project_root: str) -> PhysicsMotorConfig:
    root = Path(project_root)
    path = root / PHYSICS_CONFIG_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = PhysicsMotorConfig(
        project_root=root,
        path=path,
        payload=payload,
        nominal=MotorParameters.from_mapping(payload["nominal_parameters"]),
    )
    config.validate()
    return config


def load_physics_motor_config(project_root: Path) -> PhysicsMotorConfig:
    """Load the immutable physics configuration for one project root."""

    return _cached_config(str(Path(project_root).resolve()))


def clear_physics_motor_config_cache() -> None:
    _cached_config.cache_clear()


def _validate_motor_parameter_matrix(
    parameters: np.ndarray, friction_model: str
) -> None:
    """Validate every model row, including LuGre friction-level ordering."""

    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(MODEL_PARAMETER_NAMES):
        raise ValueError("physics ensemble matrix has an invalid shape")
    if not np.isfinite(values).all():
        raise ValueError("physics ensemble contains non-finite values")
    for row in values:
        MotorParameters.from_array(row).validate(friction_model=friction_model)


def build_physics_motor_ensemble(project_root: Path) -> dict[str, np.ndarray]:
    """Build a deterministic 40-train/16-validation physical model ensemble."""

    root = Path(project_root).resolve()
    config = load_physics_motor_config(root)
    ensemble_spec = config.payload["ensemble"]
    training_count = int(ensemble_spec["training_models"])
    validation_count = int(ensemble_spec["validation_models"])
    model_count = training_count + validation_count
    if training_count <= 0 or validation_count <= 0:
        raise ValueError("physics ensemble needs training and validation models")

    rng = np.random.default_rng(int(ensemble_spec["seed"]))
    nominal = config.nominal.as_array()
    uncertainty = np.asarray(
        [config.uncertainty_fraction[name] for name in MODEL_PARAMETER_NAMES],
        dtype=np.float64,
    )
    unit = np.empty((model_count, nominal.size), dtype=np.float64)
    for column in range(nominal.size):
        strata = (rng.permutation(model_count) + rng.random(model_count)) / model_count
        unit[:, column] = 2.0 * strata - 1.0
    parameters = nominal[None, :] * (1.0 + unit * uncertainty[None, :])
    parameters[0] = nominal
    _validate_motor_parameter_matrix(parameters, config.active_friction_model)

    role = np.asarray(
        ["train"] * training_count + ["validation"] * validation_count
    )
    model_id = np.asarray(
        [
            "physics_nominal"
            if index == 0
            else f"physics_{role[index]}_{index:03d}"
            for index in range(model_count)
        ]
    )
    result = {
        "schema_version": np.asarray(
            PHYSICS_ENSEMBLE_SCHEMA_VERSION, dtype=np.int16
        ),
        "sample_period_names": np.asarray(SAMPLE_PERIOD_LOOP_NAMES),
        "sample_periods_s": np.asarray(
            [config.sample_period_s_for(loop) for loop in SAMPLE_PERIOD_LOOP_NAMES],
            dtype=np.float64,
        ),
        "parameter_names": np.asarray(MODEL_PARAMETER_NAMES),
        "parameters": parameters,
        "model_id": model_id,
        "role": role,
        "is_nominal": (np.arange(model_count) == 0).astype(np.int8),
        "active_for_training": (role == "train").astype(np.int8),
        "active_for_audit": np.ones(model_count, dtype=np.int8),
    }
    output = root / PHYSICS_ENSEMBLE_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    manifest = {
        "schema_version": PHYSICS_ENSEMBLE_SCHEMA_VERSION,
        "model_id": config.payload["model_id"],
        "config_path": str(PHYSICS_CONFIG_RELATIVE_PATH).replace("\\", "/"),
        "config_sha256": _sha256(config.path),
        "ensemble_path": str(PHYSICS_ENSEMBLE_RELATIVE_PATH).replace("\\", "/"),
        "model_count": model_count,
        "training_models": training_count,
        "validation_models": validation_count,
        "seed": int(ensemble_spec["seed"]),
        "parameter_names": list(MODEL_PARAMETER_NAMES),
        "sample_periods_s": config.sample_periods_s,
        "controller_update_ratios": config.controller_update_ratios,
        "uncertainty_fraction": config.uncertainty_fraction,
        "nominal_model_included": True,
        "measured_frf_used_for_fitting": False,
        "measured_frf_role": config.payload["provenance"]["measured_frf_role"],
    }
    manifest_path = root / PHYSICS_ENSEMBLE_MANIFEST_RELATIVE_PATH
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def load_physics_motor_ensemble(project_root: Path) -> dict[str, np.ndarray]:
    """Load and validate the generated physical-model ensemble."""

    root = Path(project_root).resolve()
    path = root / PHYSICS_ENSEMBLE_RELATIVE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run scripts/build_physics_model.py first"
        )
    with np.load(path, allow_pickle=False) as archive:
        ensemble = {name: archive[name] for name in archive.files}
    schema_version = int(np.asarray(ensemble["schema_version"]).item())
    if schema_version != PHYSICS_ENSEMBLE_SCHEMA_VERSION:
        raise ValueError(
            "single-rate or stale physics ensemble state is unsupported; rebuild "
            "the schema 3 multi-rate 13-parameter ensemble"
        )
    if tuple(ensemble["parameter_names"].tolist()) != MODEL_PARAMETER_NAMES:
        raise ValueError("physics ensemble parameter order is invalid")
    parameters = np.asarray(ensemble["parameters"], dtype=np.float64)
    if parameters.ndim != 2 or parameters.shape[1] != len(MODEL_PARAMETER_NAMES):
        raise ValueError("physics ensemble matrix has an invalid shape")
    config = load_physics_motor_config(root)
    if tuple(ensemble["sample_period_names"].tolist()) != SAMPLE_PERIOD_LOOP_NAMES:
        raise ValueError("physics ensemble sample-period order is invalid")
    expected_periods = np.asarray(
        [config.sample_period_s_for(loop) for loop in SAMPLE_PERIOD_LOOP_NAMES],
        dtype=np.float64,
    )
    if not np.array_equal(
        np.asarray(ensemble["sample_periods_s"], dtype=np.float64),
        expected_periods,
    ):
        raise ValueError(
            "physics ensemble sample periods do not match the active configuration; "
            "rebuild the ensemble"
        )
    _validate_motor_parameter_matrix(parameters, config.active_friction_model)
    if np.count_nonzero(ensemble["is_nominal"]) != 1:
        raise ValueError("physics ensemble must contain exactly one nominal model")
    return ensemble


@dataclass(frozen=True)
class SimulationTrace:
    scenario: str
    time_s: np.ndarray
    output: np.ndarray
    reference: np.ndarray
    primary_control: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    speed_rad_s: np.ndarray
    position_rad: np.ndarray
    current_reference_a: np.ndarray
    speed_reference_rad_s: np.ndarray
    load_torque_nm: np.ndarray
    friction_torque_nm: np.ndarray
    bristle_state_rad: np.ndarray
    bristle_rate_rad_s: np.ndarray
    saturation_count: int
    terminated: bool


def wrap_angle(angle_rad: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(angle_rad) + np.pi) % (2.0 * np.pi) - np.pi


def _controller_vector(parameters: np.ndarray) -> dict[str, tuple[float, float, float]]:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (11,) or not np.isfinite(values).all():
        raise ValueError("controller parameter vector must be finite with shape (11,)")
    return {
        "position": tuple(values[0:3]),
        "speed": tuple(values[3:6]),
        "current": tuple(values[8:11]),
    }


def simulate_scenario(
    config: PhysicsMotorConfig,
    motor: MotorParameters,
    controller_parameters: np.ndarray,
    scenario: str,
    *,
    encoder_effects: bool = False,
    seed: int = 0,
    scenario_overrides: Mapping[str, float] | None = None,
) -> SimulationTrace:
    """Run one deterministic cascaded PIDF/DOBC multi-rate scenario."""

    if scenario not in {"current", "speed", "position", "disturbance"}:
        raise ValueError(f"unknown simulation scenario: {scenario}")
    motor.validate(friction_model=config.active_friction_model)
    values = np.asarray(controller_parameters, dtype=np.float64)
    _controller_vector(values)
    dt = config.base_sample_period_s
    scenario_spec = config.scenarios
    if scenario_overrides is not None:
        allowed_overrides = {
            "current_reference_a",
            "speed_reference_rad_s",
            "position_reference_rad",
            "load_torque_step_nm",
            "disturbance_start_s",
        }
        unknown = set(scenario_overrides).difference(allowed_overrides)
        if unknown:
            raise ValueError(f"unsupported scenario overrides: {sorted(unknown)}")
        for name, raw_value in scenario_overrides.items():
            value = float(raw_value)
            if not np.isfinite(value):
                raise ValueError(f"non-finite scenario override: {name}")
            if name in {"load_torque_step_nm", "disturbance_start_s"} and value < 0.0:
                raise ValueError(f"scenario override must be non-negative: {name}")
            scenario_spec[name] = value
    duration_key = {
        "current": "current_duration_s",
        "speed": "speed_duration_s",
        "position": "position_duration_s",
        "disturbance": "disturbance_duration_s",
    }[scenario]
    points = int(round(scenario_spec[duration_key] / dt)) + 1
    filters = config.derivative_filter_s
    limits = config.limits
    encoder_lsb = float(config.payload["encoder"]["position_lsb_rad"])
    encoder_noise = (
        np.random.default_rng(seed).uniform(-0.5, 0.5, size=points)
        if encoder_effects
        else np.empty(0, dtype=np.float64)
    )
    scenario_codes = {
        "current": SCENARIO_CURRENT,
        "speed": SCENARIO_SPEED,
        "position": SCENARIO_POSITION,
        "disturbance": SCENARIO_DISTURBANCE,
    }
    (
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
        bristle_state,
        bristle_rate,
        saturation_count,
        terminated,
    ) = simulate_scenario_kernel(
        scenario_codes[scenario],
        np.asarray(
            [
                config.sample_period_s_for("current"),
                config.sample_period_s_for("speed"),
                config.sample_period_s_for("position"),
            ],
            dtype=np.float64,
        ),
        points,
        values,
        np.asarray(
            [filters["position"], filters["speed"], filters["current"]],
            dtype=np.float64,
        ),
        motor.as_array(),
        np.asarray(
            [
                1.0 if config.active_friction_model == "lugre" else 0.0,
                float(config.friction_model["stribeck_exponent"]),
                float(config.friction_model["initial_bristle_state_rad"]),
            ],
            dtype=np.float64,
        ),
        config.nominal.as_array(),
        np.asarray(
            [
                float(limits["hard_current_a"]),
                float(limits["training_speed_rad_s"]),
                float(limits["voltage_v"]),
                float(limits["termination_speed_rad_s"]),
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                scenario_spec["current_reference_a"],
                scenario_spec["speed_reference_rad_s"],
                scenario_spec["position_reference_rad"],
                scenario_spec["load_torque_step_nm"],
                scenario_spec["disturbance_start_s"],
            ],
            dtype=np.float64,
        ),
        encoder_lsb,
        encoder_noise,
    )

    return SimulationTrace(
        scenario=scenario,
        time_s=time_s,
        output=output,
        reference=reference,
        primary_control=primary_control,
        voltage_v=voltage,
        current_a=current,
        speed_rad_s=speed,
        position_rad=position,
        current_reference_a=current_reference,
        speed_reference_rad_s=speed_reference,
        load_torque_nm=load_torque,
        friction_torque_nm=friction_torque,
        bristle_state_rad=bristle_state,
        bristle_rate_rad_s=bristle_rate,
        saturation_count=saturation_count,
        terminated=terminated,
    )
