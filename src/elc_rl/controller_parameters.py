"""Auditable 11-dimensional physics controller parameter space."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .performance_targets import (
    PERFORMANCE_TARGETS_RELATIVE_PATH,
    load_controller_performance_targets,
)
from .physics_motor_model import load_physics_motor_config


PHYSICS_TASK_ID = "cgs_turntable_001"
PARAMETER_ORDER = (
    "kppos",
    "kipos",
    "kdpos",
    "kpspeed",
    "kispeed",
    "kdspeed",
    "kgspeed",
    "tauspeed",
    "kpcurr",
    "kicurr",
    "kdcurr",
)

PHYSICS_PARAMETER_SPACE_JSON = "controller_parameter_space.json"
PHYSICS_PARAMETER_SPACE_NPZ = "controller_parameter_space.npz"
PHYSICS_TRAINING_ANCHOR_RELATIVE_PATH = Path(
    "data/processed/training_anchor.json"
)
_BOUNDARY_EPS_FACTOR = 128.0
_FEASIBILITY_BOUND_MARGIN_FRACTION = 0.05
_CURRENT_EVIDENCE_SELECTION_GROUPS = (
    "nominal",
    "corners_8",
    "dense_grid_125",
    "training_40",
)


def _physical_boundary_tolerance(lower: float, upper: float) -> float:
    scale = max(1.0, abs(lower), abs(upper), abs(upper - lower))
    return _BOUNDARY_EPS_FACTOR * np.finfo(np.float64).eps * scale


@dataclass(frozen=True)
class ParameterSpec:
    """One physical parameter and its bounded RL representation."""

    name: str
    module: str
    initial: float
    lower: float
    upper: float
    transform: str
    action_step_fraction: float
    source_kind: str
    source: str
    original_value: float | None
    unit: str
    sample_period_s: float | None
    digital_initial: float | None
    training_stage: int
    hardware_status: str


@dataclass(frozen=True)
class ControllerParameterSpace:
    """Validated physical/normalized mapping for the 11 controller outputs."""

    task_id: str
    specs: tuple[ParameterSpec, ...]
    metadata: dict[str, Any]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    @property
    def initial(self) -> np.ndarray:
        return np.asarray([spec.initial for spec in self.specs], dtype=np.float64)

    @property
    def lower(self) -> np.ndarray:
        return np.asarray([spec.lower for spec in self.specs], dtype=np.float64)

    @property
    def upper(self) -> np.ndarray:
        return np.asarray([spec.upper for spec in self.specs], dtype=np.float64)

    @property
    def action_step_fraction(self) -> np.ndarray:
        return np.asarray(
            [spec.action_step_fraction for spec in self.specs], dtype=np.float64
        )

    def validate(self) -> None:
        if self.task_id != PHYSICS_TASK_ID:
            raise ValueError(f"unexpected task_id: {self.task_id}")
        if self.names != PARAMETER_ORDER:
            raise ValueError(f"unexpected parameter order: {self.names}")

        lower = self.lower
        initial = self.initial
        upper = self.upper
        if not np.isfinite(np.concatenate([lower, initial, upper])).all():
            raise ValueError("parameter space contains non-finite values")
        if not np.all(lower < upper):
            raise ValueError("every parameter must have a non-empty interval")
        if not np.all((lower <= initial) & (initial <= upper)):
            raise ValueError("initial parameter values must be inside their bounds")

        for spec in self.specs:
            if spec.transform not in {"linear", "log"}:
                raise ValueError(f"unsupported transform for {spec.name}: {spec.transform}")
            if spec.transform == "log" and spec.lower <= 0:
                raise ValueError(f"log-transformed lower bound is not positive: {spec.name}")
            if not 0 < spec.action_step_fraction <= 1:
                raise ValueError(f"invalid action step fraction: {spec.name}")
            if spec.source_kind != "excel_baseline" and spec.original_value is not None:
                raise ValueError(f"inferred parameter cannot claim an original value: {spec.name}")
            if spec.sample_period_s is not None and spec.sample_period_s <= 0:
                raise ValueError(f"invalid sample period: {spec.name}")

    def normalize(self, physical: np.ndarray) -> np.ndarray:
        """Map physical values to [-1, 1] using each declared transform."""

        values = np.asarray(physical, dtype=np.float64)
        if values.shape[-1:] != (len(self.specs),):
            raise ValueError(f"expected final dimension {len(self.specs)}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("physical values must be finite")
        normalized = np.empty_like(values, dtype=np.float64)
        for index, spec in enumerate(self.specs):
            value = values[..., index]
            tolerance = _physical_boundary_tolerance(spec.lower, spec.upper)
            if np.any(
                (value < spec.lower - tolerance) | (value > spec.upper + tolerance)
            ):
                raise ValueError(f"physical value outside bounds: {spec.name}")
            value = np.clip(value, spec.lower, spec.upper)
            if spec.transform == "log":
                fraction = (np.log(value) - np.log(spec.lower)) / (
                    np.log(spec.upper) - np.log(spec.lower)
                )
            else:
                fraction = (value - spec.lower) / (spec.upper - spec.lower)
            normalized[..., index] = np.clip(2.0 * fraction - 1.0, -1.0, 1.0)
        return normalized

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """Map normalized values in [-1, 1] back to physical parameters."""

        values = np.asarray(normalized, dtype=np.float64)
        if values.shape[-1:] != (len(self.specs),):
            raise ValueError(f"expected final dimension {len(self.specs)}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("normalized values must be finite")
        tolerance = _BOUNDARY_EPS_FACTOR * np.finfo(np.float64).eps
        if np.any((values < -1.0 - tolerance) | (values > 1.0 + tolerance)):
            raise ValueError("normalized values must stay inside [-1, 1]")
        values = np.clip(values, -1.0, 1.0)
        physical = np.empty_like(values, dtype=np.float64)
        for index, spec in enumerate(self.specs):
            fraction = (values[..., index] + 1.0) / 2.0
            if spec.transform == "log":
                transformed = np.exp(
                    np.log(spec.lower)
                    + fraction * (np.log(spec.upper) - np.log(spec.lower))
                )
            else:
                transformed = spec.lower + fraction * (
                    spec.upper - spec.lower
                )
            physical[..., index] = np.where(
                fraction <= 0.0,
                spec.lower,
                np.where(fraction >= 1.0, spec.upper, transformed),
            )
        return np.clip(physical, self.lower, self.upper)

    def apply_action(self, physical: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Apply a bounded normalized delta action to one physical parameter vector."""

        current = self.normalize(np.asarray(physical, dtype=np.float64))
        delta = np.asarray(action, dtype=np.float64)
        if delta.shape != (len(self.specs),):
            raise ValueError(f"expected action shape {(len(self.specs),)}, got {delta.shape}")
        if not np.isfinite(delta).all():
            raise ValueError("action values must be finite")
        if np.any((delta < -1.0) | (delta > 1.0)):
            raise ValueError("action values must stay inside [-1, 1]")
        updated = np.clip(current + delta * self.action_step_fraction, -1.0, 1.0)
        return self.denormalize(updated)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_vector_sha256(parameters: np.ndarray) -> str:
    values = np.asarray(parameters, dtype="<f8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def load_physics_training_anchor(project_root: Path) -> dict[str, Any]:
    """Load and validate the optional formal-training initialization anchor.

    The anchor is deliberately separate from the controller parameter-space
    artifact.  Training packages include it, while the sealed final-test
    package does not depend on it.
    """

    root = Path(project_root).resolve()
    path = root / PHYSICS_TRAINING_ANCHOR_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("training anchor has an unsupported schema version")
    if payload.get("backend") != "physics":
        raise ValueError("training anchor is not for the physics backend")
    if payload.get("task_id") != PHYSICS_TASK_ID:
        raise ValueError("training anchor task_id does not match the controller task")
    if payload.get("purpose") != "formal_training_initialization_only":
        raise ValueError("training anchor has an invalid purpose")
    if tuple(payload.get("parameter_order", ())) != PARAMETER_ORDER:
        raise ValueError("training anchor parameter order does not match the controller")
    if payload.get("parameter_vector_encoding") != "little_endian_float64_c_order":
        raise ValueError("training anchor has an unsupported vector encoding")

    parameters = np.asarray(payload.get("parameters", ()), dtype=np.float64)
    if parameters.shape != (len(PARAMETER_ORDER),):
        raise ValueError("training anchor must contain exactly 11 parameters")
    if not np.isfinite(parameters).all():
        raise ValueError("training anchor contains non-finite parameters")
    actual_vector_sha256 = _parameter_vector_sha256(parameters)
    if payload.get("parameter_vector_sha256") != actual_vector_sha256:
        raise ValueError("training anchor parameter-vector SHA256 does not match")

    provenance = payload.get("provenance", {})
    selection_models = provenance.get("selection_models", {})
    if selection_models != {"training": 40, "validation": 0, "sealed_test": 0}:
        raise ValueError("training anchor provenance violates model isolation")
    if provenance.get("full_three_loop_six_metric_audit_executed") is not False:
        raise ValueError("training anchor must not claim a completed formal audit")
    for key in ("selected_report", "upstream_dobc_report"):
        source = provenance.get(key, {})
        digest = str(source.get("sha256", ""))
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"training anchor {key} lacks a lowercase SHA256")
        if source.get("included_in_server_package") is not False:
            raise ValueError(f"training anchor {key} must remain provenance-only")

    acceptance = payload.get("acceptance", {})
    if acceptance.get("status") != "engineering_initialization_only":
        raise ValueError("training anchor acceptance status is invalid")
    if acceptance.get("passed_all_three_loops_six_metrics") is not False:
        raise ValueError("training anchor must not claim final metric acceptance")
    if acceptance.get("eligible_as_final_candidate") is not False:
        raise ValueError("training anchor must not be eligible as a final candidate")
    if payload.get("hardware_use_allowed") is not False:
        raise ValueError("training anchor must remain simulation-only")

    return {
        **payload,
        "path": str(PHYSICS_TRAINING_ANCHOR_RELATIVE_PATH).replace("\\", "/"),
        "file_sha256": _sha256(path),
        "parameters_array": parameters,
    }


def _with_optional_training_anchor(
    space: ControllerParameterSpace,
    project_root: Path,
) -> ControllerParameterSpace:
    root = Path(project_root).resolve()
    path = root / PHYSICS_TRAINING_ANCHOR_RELATIVE_PATH
    if not path.is_file():
        return space

    anchor = load_physics_training_anchor(root)
    parameters = np.asarray(anchor["parameters_array"], dtype=np.float64)
    tolerance = np.asarray(
        [
            _physical_boundary_tolerance(spec.lower, spec.upper)
            for spec in space.specs
        ],
        dtype=np.float64,
    )
    if np.any(parameters < space.lower - tolerance) or np.any(
        parameters > space.upper + tolerance
    ):
        raise ValueError("training anchor contains a parameter outside formal bounds")
    parameters = np.clip(parameters, space.lower, space.upper)

    source = (
        f"formal training initialization anchor {anchor['path']} "
        f"(vector SHA256 {anchor['parameter_vector_sha256']}); not an acceptance result"
    )
    specs = []
    for spec, initial in zip(space.specs, parameters, strict=True):
        digital_initial = None
        if spec.sample_period_s is not None and spec.module in {
            "current",
            "speed",
            "position",
        }:
            digital_initial = _digital_value(spec.name, float(initial), spec.sample_period_s)
        specs.append(
            replace(
                spec,
                initial=float(initial),
                digital_initial=digital_initial,
                source_kind="formal_training_anchor",
                source=source,
            )
        )
    metadata = {
        **space.metadata,
        "training_anchor": {
            "path": anchor["path"],
            "file_sha256": anchor["file_sha256"],
            "parameter_vector_sha256": anchor["parameter_vector_sha256"],
            "purpose": anchor["purpose"],
            "acceptance_status": anchor["acceptance"]["status"],
            "eligible_as_final_candidate": False,
        },
    }
    anchored = ControllerParameterSpace(
        task_id=space.task_id,
        specs=tuple(specs),
        metadata=metadata,
    )
    anchored.validate()
    return anchored


def _sample_period_for_module(config: Any, module: str) -> float | None:
    """Return the implementation period belonging to one controller module."""

    if module == "DOBC":
        module = "speed"
    if module not in {"current", "speed", "position"}:
        return None
    return float(config.sample_period_s_for(module))


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        value = path.resolve().relative_to(root.resolve())
    except ValueError:
        value = path.resolve()
    return str(value).replace("\\", "/")


def load_current_pidf_feasibility_evidence(
    project_root: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Validate one non-training 40 kHz scan before it can inform bounds.

    Validation diagnostics may be present in the report, but the selected
    candidate must have been ranked using only nominal, uncertainty-grid and
    training-model groups.  This function deliberately does not inspect the
    validation group's pass/fail result.
    """

    root = Path(project_root).resolve()
    path = Path(report_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_kind") != "non_training_multirate_current_pidf_feasibility_scan":
        raise ValueError("current PIDF evidence is not a non-training scan report")
    if payload.get("formal_configuration_modified") is not False:
        raise ValueError("current PIDF evidence modified the formal configuration")
    if payload.get("formal_parameter_space_modified") is not False:
        raise ValueError("current PIDF evidence modified the formal parameter space")
    if payload.get("sac_training_executed") is not False:
        raise ValueError("current PIDF evidence must not execute SAC training")

    config = load_physics_motor_config(root)
    required_rate_hz = 1.0 / config.sample_period_s_for("current")
    matching = [
        row
        for row in payload.get("sample_rates", [])
        if np.isclose(
            float(row.get("sample_rate_hz", np.nan)),
            required_rate_hz,
            rtol=1e-12,
            atol=1e-9,
        )
    ]
    if len(matching) != 1:
        raise ValueError(
            "current PIDF evidence must contain exactly one result matching "
            f"the configured {required_rate_hz:g} Hz current loop"
        )
    row = matching[0]
    search = row.get("search", {})
    selection_groups = tuple(str(name) for name in search.get("candidate_selection_groups", []))
    if search.get("validation_used_for_candidate_selection") is not False:
        raise ValueError("validation models must not participate in PIDF selection")
    if selection_groups != _CURRENT_EVIDENCE_SELECTION_GROUPS:
        raise ValueError(
            "candidate selection groups must be nominal, corners_8, "
            "dense_grid_125 and training_40, in that order"
        )

    selected = row.get("selected", {})
    groups = selected.get("groups", {})
    if any(name not in groups for name in selection_groups):
        raise ValueError("current PIDF evidence is missing a selection audit group")
    if not all(bool(groups[name].get("all_six_pass")) for name in selection_groups):
        raise ValueError("current PIDF evidence candidate failed a selection audit group")
    current_parameters = selected.get("current_parameters", {})
    candidate = {
        name: float(current_parameters[name]) for name in ("kpcurr", "kicurr", "kdcurr")
    }
    if not np.isfinite(np.asarray(list(candidate.values()), dtype=np.float64)).all():
        raise ValueError("current PIDF evidence contains non-finite parameters")
    if any(value <= 0.0 for value in candidate.values()):
        raise ValueError("current PIDF evidence parameters must be positive")

    continuous_path = path.with_name("continuous_worst_case_report.json")
    continuous = json.loads(continuous_path.read_text(encoding="utf-8"))
    if continuous.get("run_kind") != "non_training_continuous_current_uncertainty_refinement":
        raise ValueError("current PIDF evidence lacks the continuous-box refinement")
    if continuous.get("formal_configuration_modified") is not False:
        raise ValueError("continuous refinement modified the formal configuration")
    if continuous.get("formal_parameter_space_modified") is not False:
        raise ValueError("continuous refinement modified the formal parameter space")
    if continuous.get("sac_training_executed") is not False:
        raise ValueError("continuous refinement must not execute SAC training")
    refinements = [
        item
        for item in continuous.get("refinements", [])
        if np.isclose(
            float(item.get("sample_rate_hz", np.nan)),
            required_rate_hz,
            rtol=1e-12,
            atol=1e-9,
        )
    ]
    if len(refinements) != 1:
        raise ValueError("continuous refinement must contain the configured current rate")
    refinement = refinements[0]
    refined_candidate = refinement.get("current_parameters", {})
    if any(
        not np.isclose(
            float(refined_candidate.get(name, np.nan)),
            value,
            rtol=1e-12,
            atol=1e-15,
        )
        for name, value in candidate.items()
    ):
        raise ValueError("continuous refinement candidate does not match the scan")
    if refinement.get("continuous_box_all_six_pass") is not True:
        raise ValueError("current PIDF candidate failed continuous-box refinement")
    return {
        "report_path": _relative_or_absolute(path, root),
        "report_sha256": _sha256(path),
        "continuous_report_path": _relative_or_absolute(continuous_path, root),
        "continuous_report_sha256": _sha256(continuous_path),
        "sample_rate_hz": required_rate_hz,
        "sample_period_s": config.sample_period_s_for("current"),
        "candidate_selection_groups": list(selection_groups),
        "validation_used_for_candidate_selection": False,
        "selected_current_parameters": candidate,
        "bound_margin_fraction": _FEASIBILITY_BOUND_MARGIN_FRACTION,
        "role": "bounds_only_not_acceptance_or_hardware_evidence",
    }


def _digital_value(name: str, analog_value: float, sample_period_s: float) -> float:
    if name.startswith("kp"):
        return analog_value
    if name.startswith("ki"):
        return analog_value * sample_period_s
    if name.startswith("kd"):
        return analog_value / sample_period_s
    raise KeyError(name)


def _parameter(
    *,
    name: str,
    module: str,
    initial: float,
    lower: float,
    upper: float,
    transform: str,
    action_step_fraction: float,
    source_kind: str,
    source: str,
    original_value: float | None,
    unit: str,
    sample_period_s: float | None,
    training_stage: int,
    hardware_status: str,
) -> dict[str, Any]:
    digital_initial = None
    if sample_period_s is not None and module in {"current", "speed", "position"}:
        digital_initial = _digital_value(name, initial, sample_period_s)
    return {
        "name": name,
        "module": module,
        "initial": float(initial),
        "lower": float(lower),
        "upper": float(upper),
        "transform": transform,
        "action_step_fraction": float(action_step_fraction),
        "source_kind": source_kind,
        "source": source,
        "original_value": None if original_value is None else float(original_value),
        "unit": unit,
        "sample_period_s": sample_period_s,
        "digital_initial": digital_initial,
        "training_stage": training_stage,
        "hardware_status": hardware_status,
    }



def _space_from_payload(payload: dict[str, Any]) -> ControllerParameterSpace:
    specs = tuple(ParameterSpec(**values) for values in payload["parameters"])
    metadata = {key: value for key, value in payload.items() if key != "parameters"}
    return ControllerParameterSpace(
        task_id=str(payload["task_id"]), specs=specs, metadata=metadata
    )



def _controller_seed_from_bandwidths(
    project_root: Path, bandwidths_hz: dict[str, float]
) -> np.ndarray:
    config = load_physics_motor_config(project_root)
    motor = config.nominal
    design = config.payload["controller_design"]
    current_omega = 2.0 * np.pi * float(bandwidths_hz["current"])
    speed_omega = 2.0 * np.pi * float(bandwidths_hz["speed"])
    position_omega = 2.0 * np.pi * float(bandwidths_hz["position"])
    derivative_ratio = float(design["derivative_ratio_at_crossover"])
    position_integral_ratio = float(
        design["position_integral_ratio_at_crossover"]
    )

    current_delay_correction = np.sqrt(
        1.0 + (current_omega * motor.current_delay_s) ** 2
    )
    current_kp = motor.inductance_h * current_omega * current_delay_correction
    current_ki = motor.resistance_ohm * current_omega * current_delay_correction
    current_kd = derivative_ratio * current_kp / current_omega

    speed_delay_correction = np.sqrt(
        1.0 + (speed_omega * motor.speed_measurement_delay_s) ** 2
    )
    speed_kp = (
        motor.inertia_kg_m2
        * speed_omega
        * speed_delay_correction
        / motor.torque_constant_nm_per_a
    )
    speed_ki = speed_kp * (
        motor.viscous_friction_nm_s_per_rad / motor.inertia_kg_m2
    )
    speed_kd = derivative_ratio * speed_kp / speed_omega

    position_derivative_ratio = derivative_ratio
    position_kp = position_omega / np.sqrt(
        1.0
        + (position_integral_ratio - position_derivative_ratio) ** 2
    )
    position_ki = position_integral_ratio * position_omega * position_kp
    position_kd = position_derivative_ratio * position_kp / position_omega

    dobc = design["dobc"]
    return np.asarray(
        [
            position_kp,
            position_ki,
            position_kd,
            speed_kp,
            speed_ki,
            speed_kd,
            float(dobc["gain_initial"]),
            float(dobc["filter_time_initial_s"]),
            current_kp,
            current_ki,
            current_kd,
        ],
        dtype=np.float64,
    )


def _initialization_bandwidths_hz(project_root: Path) -> dict[str, float]:
    """Return conservative loop-shaping frequencies for a valid reset seed.

    These are numerical initialization heuristics only.  They are deliberately
    kept separate from the real closed-loop performance targets used by the
    evaluator and Reward.
    """

    config = load_physics_motor_config(project_root)
    targets = load_controller_performance_targets(project_root)
    current = min(
        targets.loop("current").bandwidth_hz,
        0.08 / config.sample_period_s_for("current"),
        0.08 / config.nominal.current_delay_s,
    )
    speed = min(
        targets.loop("speed").bandwidth_hz,
        0.008 / config.sample_period_s_for("speed"),
        current / 10.0,
    )
    position = min(
        targets.loop("position").bandwidth_hz,
        0.002 / config.sample_period_s_for("position"),
        speed / 4.0,
    )
    return {"current": current, "speed": speed, "position": position}


def derive_physics_controller_initials(project_root: Path) -> np.ndarray:
    """Derive a conservative, numerically valid initial search seed."""

    return _controller_seed_from_bandwidths(
        project_root, _initialization_bandwidths_hz(project_root)
    )


def build_physics_controller_parameter_space(
    project_root: Path,
    *,
    current_feasibility_report: Path | None = None,
) -> dict[str, Any]:
    """Build a separate 11-D space for the physics training backend."""

    root = Path(project_root).resolve()
    config = load_physics_motor_config(root)
    targets = load_controller_performance_targets(root)
    initial = derive_physics_controller_initials(root)
    by_name = dict(zip(PARAMETER_ORDER, initial.tolist()))
    target_seed = _controller_seed_from_bandwidths(
        root,
        {
            loop: targets.loop(loop).bandwidth_hz
            for loop in ("current", "speed", "position")
        },
    )
    target_seed_by_name = dict(zip(PARAMETER_ORDER, target_seed.tolist()))
    sample_periods_s = {
        loop: config.sample_period_s_for(loop)
        for loop in ("current", "speed", "position")
    }
    feasibility_evidence = (
        None
        if current_feasibility_report is None
        else load_current_pidf_feasibility_evidence(
            root, Path(current_feasibility_report)
        )
    )
    evidence_parameters = (
        {}
        if feasibility_evidence is None
        else feasibility_evidence["selected_current_parameters"]
    )
    design = config.payload["controller_design"]
    dobc = design["dobc"]

    def gain_parameter(
        name: str,
        module: str,
        *,
        stage: int,
        derivative: bool = False,
    ) -> dict[str, Any]:
        seed_value = by_name[name]
        target_value = target_seed_by_name[name]
        if derivative:
            lower, upper, transform = (
                0.0,
                max(seed_value, target_value) * 4.0,
                "linear",
            )
            step = 0.05
        else:
            lower, upper, transform = (
                seed_value / 4.0,
                max(seed_value, target_value) * 4.0,
                "log",
            )
            step = 0.06
        evidence_value = evidence_parameters.get(name)
        initial_value = (
            seed_value if evidence_value is None else float(evidence_value)
        )
        if evidence_value is not None:
            margin = 1.0 + _FEASIBILITY_BOUND_MARGIN_FRACTION
            if not derivative:
                lower = min(lower, float(evidence_value) / margin)
            upper = max(upper, float(evidence_value) * margin)
        return _parameter(
            name=name,
            module=module,
            initial=initial_value,
            lower=lower,
            upper=upper,
            transform=transform,
            action_step_fraction=step,
            source_kind=(
                "performance_target_seed_derived"
                if evidence_value is None
                else "non_training_feasibility_candidate_derived"
            ),
            source=(
                "numerical loop-shaping seed from the motor model and the real "
                "closed-loop bandwidth table; not an acceptance measurement"
                + (
                    "; current-loop bounds minimally cover the validated "
                    "non-training feasibility candidate with 5 percent margin"
                    if evidence_value is not None
                    else ""
                )
            ),
            original_value=None,
            unit=("native_analog_gain_s" if derivative else (
                "native_analog_gain_per_s" if name.startswith("ki") else "native_analog_gain"
            )),
            sample_period_s=_sample_period_for_module(config, module),
            training_stage=stage,
            hardware_status="simulation_only_requires_hil_and_hardware_validation",
        )

    parameters = [
        gain_parameter("kppos", "position", stage=3),
        gain_parameter("kipos", "position", stage=3),
        gain_parameter("kdpos", "position", stage=3, derivative=True),
        gain_parameter("kpspeed", "speed", stage=2),
        gain_parameter("kispeed", "speed", stage=2),
        gain_parameter("kdspeed", "speed", stage=2, derivative=True),
        _parameter(
            name="kgspeed",
            module="DOBC",
            initial=float(dobc["gain_initial"]),
            lower=float(dobc["gain_bounds"][0]),
            upper=float(dobc["gain_bounds"][1]),
            transform="linear",
            action_step_fraction=0.05,
            source_kind="approved_dobc_structure",
            source=str(dobc["structure"]),
            original_value=None,
            unit="dimensionless",
            sample_period_s=_sample_period_for_module(config, "DOBC"),
            training_stage=2,
            hardware_status="simulation_only_speed_loop_parameter",
        ),
        _parameter(
            name="tauspeed",
            module="DOBC",
            initial=float(dobc["filter_time_initial_s"]),
            lower=float(dobc["filter_time_bounds_s"][0]),
            upper=float(dobc["filter_time_bounds_s"][1]),
            transform="log",
            action_step_fraction=0.06,
            source_kind="approved_dobc_structure",
            source=str(dobc["structure"]),
            original_value=None,
            unit="s",
            sample_period_s=_sample_period_for_module(config, "DOBC"),
            training_stage=2,
            hardware_status="simulation_only_speed_loop_parameter",
        ),
        gain_parameter("kpcurr", "current", stage=1),
        gain_parameter("kicurr", "current", stage=1),
        gain_parameter("kdcurr", "current", stage=1, derivative=True),
    ]

    payload: dict[str, Any] = {
        "schema_version": 3,
        "profile": "physics",
        "task_id": PHYSICS_TASK_ID,
        "parameter_order": list(PARAMETER_ORDER),
        "controller_convention": (
            "continuous filtered PIDF: C(s)=Kp+Ki/s+Kd*s/(Tf*s+1); "
            "implemented discretely with per-loop sample periods and anti-windup"
        ),
        "digital_conversion": {
            "Kp_d": "Kp",
            "Ki_d": "Ki*Ts",
            "Kd_d": "Kd/Ts",
            "sample_periods_s": sample_periods_s,
            "implementation_note": "runtime uses physical continuous gains, not these display conversions",
        },
        "physics_model": {
            "model_id": config.payload["model_id"],
            "config_relative_path": str(config.path.relative_to(root)).replace("\\", "/"),
            "config_sha256": _sha256(config.path),
            "primary_training_plant": True,
            "measured_frf_used_as_training_plant": False,
        },
        "controller_design": {
            "derivative_filter_s": config.derivative_filter_s,
            "derivative_ratio_at_crossover": float(
                design["derivative_ratio_at_crossover"]
            ),
            "position_integral_ratio_at_crossover": float(
                design["position_integral_ratio_at_crossover"]
            ),
        },
        "performance_targets": {
            "config_relative_path": str(PERFORMANCE_TARGETS_RELATIVE_PATH).replace(
                "\\", "/"
            ),
            "config_sha256": _sha256(root / PERFORMANCE_TARGETS_RELATIVE_PATH),
            "closed_loop_bandwidth_hz": {
                loop: targets.loop(loop).bandwidth_hz
                for loop in ("current", "speed", "position")
            },
            "use": "initial_search_seed_and_direct_six_metric_evaluation",
            "bandwidth_is_not_relabelled_as_open_loop_crossover": True,
        },
        "initialization_heuristic": {
            "loop_shaping_bandwidth_hz": _initialization_bandwidths_hz(root),
            "sample_period_fraction_for_current": 0.08,
            "physical_delay_fraction_for_current": 0.08,
            "sample_period_fraction_for_speed": 0.008,
            "sample_period_fraction_for_position": 0.002,
            "current_to_speed_seed_ratio": 10.0,
            "speed_to_position_seed_ratio": 4.0,
            "evaluation_role": "none",
        },
        "current_pidf_feasibility_evidence": feasibility_evidence,
        "dobc_design": {
            "status": "approved_simulation_structure",
            "structure": str(dobc["structure"]),
            "nominal_inverse_excludes_measurement_delay": bool(
                dobc["nominal_inverse_excludes_measurement_delay"]
            ),
        },
        "parameters": parameters,
        "safety_policy": {
            "direct_hardware_use_allowed": False,
            "simulation_limits_are_hardware_ratings": False,
            "required_before_hardware": [
                "confirm voltage/current/speed limits against drive and motor ratings",
                "confirm encoder resolution and feedback filtering",
                "validate candidates in HIL and bounded low-energy tests",
            ],
        },
    }
    space = _space_from_payload(payload)
    space.validate()
    output_dir = root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / PHYSICS_PARAMETER_SPACE_JSON
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / PHYSICS_PARAMETER_SPACE_NPZ,
        schema_version=np.asarray(3, dtype=np.int16),
        profile=np.asarray("physics"),
        task_id=np.asarray(PHYSICS_TASK_ID),
        parameter_names=np.asarray(space.names),
        initial=space.initial,
        lower=space.lower,
        upper=space.upper,
        transform=np.asarray([spec.transform for spec in space.specs]),
        source_kind=np.asarray([spec.source_kind for spec in space.specs]),
        training_stage=np.asarray(
            [spec.training_stage for spec in space.specs], dtype=np.int16
        ),
        action_step_fraction=space.action_step_fraction,
    )
    return payload


def load_physics_controller_parameter_space(
    project_root: Path,
) -> ControllerParameterSpace:
    path = (
        Path(project_root).resolve()
        / "data"
        / "processed"
        / PHYSICS_PARAMETER_SPACE_JSON
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("profile") != "physics":
        raise ValueError("controller parameter file is not the physics profile")
    if int(payload.get("schema_version", 0)) != 3:
        raise ValueError(
            "controller parameter space is obsolete; rebuild schema version 3"
        )
    space = _space_from_payload(payload)
    space.validate()
    return _with_optional_training_anchor(space, project_root)
