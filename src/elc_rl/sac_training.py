"""Resumable formal SAC training for the physics tuning environment.

This module intentionally depends on the production environment's public API.
It does not import the small pipeline-check training entry point or consume any
artifacts produced by that entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import tempfile
import time
from typing import Any, Iterable, Mapping

import cloudpickle
import gymnasium
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import torch

from .parallel_env import configure_thread_limits, create_training_vec_env
from .plant_sampling import PlantSamplingConfig
from .sampling_vec_env import PlantSamplingVecEnv
from .tuning_env import PIDTuningEnv, STAGE_ORDER, combined_stage_cost


TRAINING_INPUT_RELATIVE_PATHS = (
    "config/controller_acceptance_tolerances.json",
    "config/controller_performance_targets.json",
    "config/motor_physics.json",
    "data/processed/controller_parameter_space.json",
    "data/processed/physics_motor_ensemble.npz",
    "data/processed/physics_motor_ensemble_manifest.json",
    "src/elc_rl/__init__.py",
    "src/elc_rl/controller_parameters.py",
    "src/elc_rl/discrete_loop_model.py",
    "src/elc_rl/evaluation_utils.py",
    "src/elc_rl/parallel_env.py",
    "src/elc_rl/plant_sampling.py",
    "src/elc_rl/sampling_vec_env.py",
    "src/elc_rl/performance_targets.py",
    "src/elc_rl/physics_evaluator.py",
    "src/elc_rl/physics_motor_model.py",
    "src/elc_rl/simulation_kernel.py",
    "src/elc_rl/sac_training.py",
    "src/elc_rl/tuning_env.py",
    "scripts/train_sac.py",
)

PROGRESS_REPORT_INTERVAL_TIMESTEPS = 1000
TRAINING_PROTOCOL_SCHEMA_VERSION = 6


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _required_keys(payload: Mapping[str, Any], keys: Iterable[str], context: str) -> None:
    missing = sorted(set(keys) - set(payload))
    if missing:
        raise ValueError(f"{context} is missing keys: {missing}")


@dataclass(frozen=True)
class StageTrainingSpec:
    name: str
    total_timesteps: int


@dataclass(frozen=True)
class FormalTrainingConfig:
    path: Path
    payload: dict[str, Any]
    stages: tuple[StageTrainingSpec, ...]
    seeds: tuple[int, ...]

    @property
    def sha256(self) -> str:
        return sha256_file(self.path)

    @property
    def run_name(self) -> str:
        return str(self.payload["run_name"])

    @property
    def default_device(self) -> str:
        return str(self.payload["default_device"])


def load_formal_training_config(
    project_root: Path,
    config_path: Path | None = None,
) -> FormalTrainingConfig:
    root = Path(project_root).resolve()
    path = (
        root / "config" / "sac_training.json"
        if config_path is None
        else Path(config_path).resolve()
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    _required_keys(
        payload,
        (
            "schema_version",
            "backend",
            "run_name",
            "default_device",
            "seeds",
            "stages",
            "environment",
            "sac",
            "parallelism",
            "checkpoint",
            "validation",
            "tensorboard",
            "runtime",
            "isolation",
        ),
        "formal training configuration",
    )
    if payload["schema_version"] != 1 or payload["backend"] != "physics":
        raise ValueError("formal training configuration is not physics schema 1")

    raw_stages = payload["stages"]
    if not isinstance(raw_stages, list):
        raise ValueError("stages must be a list")
    stages = tuple(
        StageTrainingSpec(
            name=str(item["name"]),
            total_timesteps=int(item["total_timesteps"]),
        )
        for item in raw_stages
    )
    if tuple(stage.name for stage in stages) != STAGE_ORDER:
        raise ValueError(f"formal stages must exactly match {STAGE_ORDER}")
    if any(stage.total_timesteps <= 0 for stage in stages):
        raise ValueError("every formal stage needs positive total_timesteps")

    seeds = tuple(int(value) for value in payload["seeds"])
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("formal training requires at least three unique seeds")
    if any(value < 0 for value in seeds):
        raise ValueError("training seeds must be non-negative")

    environment = payload["environment"]
    _required_keys(
        environment,
        ("max_episode_steps", "audit_interval", "initial_perturbation"),
        "environment configuration",
    )
    if int(environment["max_episode_steps"]) <= 0:
        raise ValueError("max_episode_steps must be positive")
    if int(environment["audit_interval"]) <= 0:
        raise ValueError("audit_interval must be positive")
    if not 0.0 <= float(environment["initial_perturbation"]) <= 0.5:
        raise ValueError("initial_perturbation must be between 0 and 0.5")

    sampling = PlantSamplingConfig.from_mapping(payload.get("plant_sampling"))
    if sampling.maximum_probability < 1.0 / 40:
        raise ValueError("plant maximum_probability is below the 40-model uniform probability")

    sac = payload["sac"]
    _required_keys(
        sac,
        (
            "policy",
            "learning_rate",
            "buffer_size",
            "learning_starts",
            "batch_size",
            "tau",
            "gamma",
            "train_frequency",
            "gradient_steps_per_transition",
            "network_architecture",
        ),
        "SAC configuration",
    )
    if str(sac["policy"]) != "MultiInputPolicy":
        raise ValueError("physics formal training requires MultiInputPolicy")
    positive_integer_fields = (
        "buffer_size",
        "learning_starts",
        "batch_size",
        "train_frequency",
        "gradient_steps_per_transition",
    )
    if any(int(sac[name]) <= 0 for name in positive_integer_fields):
        raise ValueError("SAC integer hyperparameters must be positive")
    if int(sac["buffer_size"]) < int(sac["batch_size"]):
        raise ValueError("buffer_size must not be smaller than batch_size")
    architecture = tuple(int(value) for value in sac["network_architecture"])
    if not architecture or any(value <= 0 for value in architecture):
        raise ValueError("network_architecture must contain positive widths")

    parallelism = payload["parallelism"]
    _required_keys(
        parallelism,
        (
            "environments_per_seed",
            "concurrent_seeds",
            "start_method",
            "numerical_threads_per_process",
        ),
        "parallelism configuration",
    )
    environment_count = int(parallelism["environments_per_seed"])
    if environment_count <= 0:
        raise ValueError("environments_per_seed must be positive")
    if int(parallelism["concurrent_seeds"]) <= 0:
        raise ValueError("concurrent_seeds must be positive")
    if int(parallelism["numerical_threads_per_process"]) <= 0:
        raise ValueError("numerical_threads_per_process must be positive")
    if str(parallelism["start_method"]) not in {"auto", "spawn", "forkserver"}:
        raise ValueError("parallel start_method is invalid")
    if any(stage.total_timesteps % environment_count for stage in stages):
        raise ValueError("stage timesteps must be divisible by environments_per_seed")

    checkpoint = payload["checkpoint"]
    _required_keys(
        checkpoint,
        ("interval_timesteps", "save_replay_buffer", "keep_last"),
        "checkpoint configuration",
    )
    if int(checkpoint["interval_timesteps"]) <= 0:
        raise ValueError("checkpoint interval must be positive")
    if int(checkpoint["keep_last"]) <= 0:
        raise ValueError("checkpoint keep_last must be positive")
    if int(checkpoint["interval_timesteps"]) % environment_count:
        raise ValueError(
            "checkpoint interval must be divisible by environments_per_seed"
        )

    validation = payload["validation"]
    _required_keys(
        validation,
        (
            "interval_timesteps",
            "candidate_pool_size",
            "periodic_candidate_limit",
            "stage_finalist_limit",
            "minimum_cost_improvement",
        ),
        "validation configuration",
    )
    if any(
        int(validation[name]) <= 0
        for name in (
            "interval_timesteps",
            "candidate_pool_size",
            "periodic_candidate_limit",
            "stage_finalist_limit",
        )
    ):
        raise ValueError("validation counts and intervals must be positive")
    if float(validation["minimum_cost_improvement"]) < 0.0:
        raise ValueError("minimum_cost_improvement must be non-negative")
    if int(validation["interval_timesteps"]) % environment_count:
        raise ValueError(
            "validation interval must be divisible by environments_per_seed"
        )

    runtime = payload["runtime"]
    if int(runtime.get("progress_interval_timesteps", 0)) <= 0:
        raise ValueError("progress_interval_timesteps must be positive")

    isolation = payload["isolation"]
    if (
        int(isolation.get("training_models", -1)) != 40
        or int(isolation.get("validation_models", -1)) != 16
        or bool(isolation.get("sealed_evaluation_available_during_training", True))
    ):
        raise ValueError("formal training data-isolation declaration is invalid")

    return FormalTrainingConfig(
        path=path,
        payload=payload,
        stages=stages,
        seeds=seeds,
    )


def build_training_input_manifest(
    project_root: Path,
    config: FormalTrainingConfig,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    relative_config = None
    try:
        relative_config = config.path.relative_to(root).as_posix()
    except ValueError:
        relative_config = str(config.path)
    paths = list(TRAINING_INPUT_RELATIVE_PATHS)
    anchor_path = "data/processed/training_anchor.json"
    if (root / anchor_path).is_file():
        paths.append(anchor_path)
    if relative_config not in paths:
        paths.append(relative_config)

    files: list[dict[str, Any]] = []
    for name in paths:
        candidate = Path(name)
        path = candidate if candidate.is_absolute() else root / candidate
        if not path.is_file():
            raise FileNotFoundError(f"missing formal-training input: {path}")
        content = path.read_bytes()
        if path.suffix in {".py", ".json"}:
            content = content.replace(b"\r\n", b"\n")
        files.append(
            {
                "path": name.replace("\\", "/"),
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    fingerprint = hashlib.sha256(_canonical_json(files)).hexdigest()
    return {
        "schema_version": TRAINING_PROTOCOL_SCHEMA_VERSION,
        "backend": "physics",
        "files": files,
        "fingerprint": fingerprint,
        "fingerprint_encoding": "UTF-8 source/JSON with LF newlines; binary inputs unchanged",
    }


@dataclass
class CandidateRecord:
    stage: str
    fast_cost: float
    parameters: np.ndarray
    global_timestep: int
    fast_target_pass: bool = False
    fast_maximum_target_violation: float = float("inf")
    fast_target_violation_count: int = 2**31 - 1

    def to_payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "fast_cost": float(self.fast_cost),
            "parameters": np.asarray(self.parameters, dtype=np.float64).tolist(),
            "global_timestep": int(self.global_timestep),
            "fast_target_pass": bool(self.fast_target_pass),
            "fast_maximum_target_violation": float(
                self.fast_maximum_target_violation
            ),
            "fast_target_violation_count": int(
                self.fast_target_violation_count
            ),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CandidateRecord":
        parameters = np.asarray(payload["parameters"], dtype=np.float64)
        if parameters.shape != (11,) or not np.isfinite(parameters).all():
            raise ValueError("serialized candidate parameters are invalid")
        return cls(
            stage=str(payload["stage"]),
            fast_cost=float(payload["fast_cost"]),
            parameters=parameters,
            global_timestep=int(payload["global_timestep"]),
            fast_target_pass=bool(payload["fast_target_pass"]),
            fast_maximum_target_violation=float(
                payload["fast_maximum_target_violation"]
            ),
            fast_target_violation_count=int(
                payload["fast_target_violation_count"]
            ),
        )


def _candidate_record_rank(record: CandidateRecord) -> tuple[Any, ...]:
    return (
        not bool(record.fast_target_pass),
        float(record.fast_maximum_target_violation),
        int(record.fast_target_violation_count),
        float(record.fast_cost),
        int(record.global_timestep),
    )


class CandidatePool:
    def __init__(
        self,
        stage: str,
        maximum_size: int,
        records: Iterable[CandidateRecord] = (),
    ) -> None:
        if stage not in STAGE_ORDER:
            raise ValueError(f"invalid candidate-pool stage: {stage}")
        if maximum_size <= 0:
            raise ValueError("candidate-pool size must be positive")
        self.stage = stage
        self.maximum_size = int(maximum_size)
        self.records: list[CandidateRecord] = []
        for record in records:
            self.add(record)

    def add(self, record: CandidateRecord) -> None:
        if record.stage != self.stage:
            raise ValueError("candidate stage does not match pool stage")
        values = np.asarray(record.parameters, dtype=np.float64)
        if values.shape != (11,) or not np.isfinite(values).all():
            raise ValueError("candidate parameters must be finite shape-(11,) values")
        duplicate = next(
            (
                existing
                for existing in self.records
                if np.allclose(
                    values,
                    existing.parameters,
                    rtol=1e-12,
                    atol=1e-14,
                )
            ),
            None,
        )
        if duplicate is not None:
            if _candidate_record_rank(record) < _candidate_record_rank(duplicate):
                duplicate.fast_cost = float(record.fast_cost)
                duplicate.global_timestep = int(record.global_timestep)
                duplicate.fast_target_pass = bool(record.fast_target_pass)
                duplicate.fast_maximum_target_violation = float(
                    record.fast_maximum_target_violation
                )
                duplicate.fast_target_violation_count = int(
                    record.fast_target_violation_count
                )
                self.records.sort(key=_candidate_record_rank)
            return
        self.records.append(
            CandidateRecord(
                stage=record.stage,
                fast_cost=float(record.fast_cost),
                parameters=values.copy(),
                global_timestep=int(record.global_timestep),
                fast_target_pass=bool(record.fast_target_pass),
                fast_maximum_target_violation=float(
                    record.fast_maximum_target_violation
                ),
                fast_target_violation_count=int(
                    record.fast_target_violation_count
                ),
            )
        )
        self.records.sort(key=_candidate_record_rank)
        del self.records[self.maximum_size :]

    def to_payload(self) -> list[dict[str, Any]]:
        return [record.to_payload() for record in self.records]


class StopController:
    """Mutable stop flag set by the command-line signal handlers."""

    def __init__(self) -> None:
        self.requested = False
        self.signal_name: str | None = None

    def request(self, signal_name: str) -> None:
        self.requested = True
        self.signal_name = signal_name


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingProgressReporter:
    """Emit flush-safe progress and ETA lines for local or redirected logs."""

    def __init__(
        self,
        *,
        seed: int,
        stage: str,
        stage_start_global_steps: int,
        stage_total_steps: int,
        run_total_steps: int,
        previous_wall_time_s: float,
        session_started: float,
        initial_global_steps: int,
        interval_timesteps: int = PROGRESS_REPORT_INTERVAL_TIMESTEPS,
    ) -> None:
        self.seed = int(seed)
        self.stage = stage
        self.stage_start_global_steps = int(stage_start_global_steps)
        self.stage_total_steps = int(stage_total_steps)
        self.run_total_steps = int(run_total_steps)
        self.previous_wall_time_s = float(previous_wall_time_s)
        self.session_started = float(session_started)
        self.interval_timesteps = int(interval_timesteps)
        if self.interval_timesteps <= 0:
            raise ValueError("progress interval must be positive")
        self.next_report_steps = (
            (int(initial_global_steps) // self.interval_timesteps) + 1
        ) * self.interval_timesteps

    def report(self, global_steps: int, *, force: bool = False) -> str | None:
        completed = int(global_steps)
        if not force and completed < self.next_report_steps:
            return None
        while self.next_report_steps <= completed:
            self.next_report_steps += self.interval_timesteps

        elapsed = (
            self.previous_wall_time_s + time.perf_counter() - self.session_started
        )
        rate = completed / elapsed if elapsed > 0.0 else 0.0
        remaining_steps = max(0, self.run_total_steps - completed)
        eta = remaining_steps / rate if rate > 0.0 else float("inf")
        stage_completed = min(
            self.stage_total_steps,
            max(0, completed - self.stage_start_global_steps),
        )
        eta_text = _format_duration(eta) if np.isfinite(eta) else "unknown"
        message = (
            f"[progress] seed={self.seed} stage={self.stage} "
            f"stage_steps={stage_completed}/{self.stage_total_steps} "
            f"total_steps={completed}/{self.run_total_steps} "
            f"rate={rate:.2f} steps/s elapsed={_format_duration(elapsed)} "
            f"eta={eta_text}"
        )
        print(message, flush=True)
        return message


class CandidateCollectorCallback(BaseCallback):
    def __init__(
        self,
        pool: CandidatePool,
        stop_controller: StopController,
        progress_reporter: TrainingProgressReporter | None = None,
    ) -> None:
        super().__init__(verbose=0)
        self.pool = pool
        self.stop_controller = stop_controller
        self.progress_reporter = progress_reporter

    def _on_step(self) -> bool:
        infos = list(self.locals.get("infos", []))
        first_timestep = int(self.model.num_timesteps) - len(infos) + 1
        for rank, info in enumerate(infos):
            if not bool(info.get("fast_safe", False)):
                continue
            self.pool.add(
                CandidateRecord(
                    stage=self.pool.stage,
                    fast_cost=float(info["stage_cost"]),
                    parameters=np.asarray(info["parameters"], dtype=np.float64),
                    global_timestep=first_timestep + rank,
                    fast_target_pass=bool(info["stage_target_pass"]),
                    fast_maximum_target_violation=float(
                        info["stage_maximum_target_violation"]
                    ),
                    fast_target_violation_count=int(
                        info["stage_target_violation_count"]
                    ),
                )
            )
        if self.progress_reporter is not None:
            self.progress_reporter.report(int(self.model.num_timesteps))
        return not self.stop_controller.requested


def audit_parameters(
    environment: PIDTuningEnv,
    parameters: np.ndarray,
    stage: str,
    *,
    full_time_domain: bool,
) -> dict[str, Any]:
    values = np.asarray(parameters, dtype=np.float64)
    environment.parameter_space.normalize(values)
    if environment.stage != stage:
        raise ValueError(
            f"audit stage {stage!r} does not match environment stage "
            f"{environment.stage!r}"
        )
    result = environment.audit_parameters(
        values,
        full_time_domain=full_time_domain,
    )
    result["parameters"] = values.tolist()
    return result


def _unique_parameter_sets(values: Iterable[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for raw in values:
        candidate = np.asarray(raw, dtype=np.float64)
        if not any(
            np.allclose(candidate, existing, rtol=1e-12, atol=1e-14)
            for existing in unique
        ):
            unique.append(candidate.copy())
    return unique


def _audit_rank(report: Mapping[str, Any]) -> tuple[Any, ...]:
    """Rank validity and six-metric feasibility before the original Cost."""

    return (
        not bool(report["safe"]),
        not bool(report["target_pass"]),
        float(report["maximum_target_violation"]),
        int(report["target_violation_count"]),
        float(report["cost"]),
    )


def _audit_improves(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    minimum_cost_improvement: float,
) -> bool:
    """Accept a strict feasibility improvement or a material Cost improvement."""

    if not bool(candidate["safe"]):
        return False
    if not bool(baseline["safe"]):
        return True
    candidate_pass = bool(candidate["target_pass"])
    baseline_pass = bool(baseline["target_pass"])
    if candidate_pass != baseline_pass:
        return candidate_pass
    tolerance = 1e-12
    candidate_maximum = float(candidate["maximum_target_violation"])
    baseline_maximum = float(baseline["maximum_target_violation"])
    if candidate_maximum < baseline_maximum - tolerance:
        return True
    if candidate_maximum > baseline_maximum + tolerance:
        return False
    candidate_count = int(candidate["target_violation_count"])
    baseline_count = int(baseline["target_violation_count"])
    if candidate_count != baseline_count:
        return candidate_count < baseline_count
    return bool(
        float(candidate["cost"])
        < float(baseline["cost"]) - float(minimum_cost_improvement)
    )


def validate_candidate_pool(
    environment: PIDTuningEnv,
    stage: str,
    stage_base_parameters: np.ndarray,
    pool: CandidatePool,
    candidate_limit: int,
) -> dict[str, Any]:
    candidates = _unique_parameter_sets(
        [
            np.asarray(stage_base_parameters, dtype=np.float64),
            *(record.parameters for record in pool.records[:candidate_limit]),
        ]
    )
    audits = [
        audit_parameters(
            environment,
            parameters,
            stage,
            full_time_domain=False,
        )
        for parameters in candidates
    ]
    safe = [report for report in audits if report["safe"]]
    selected = min(safe, key=_audit_rank) if safe else None
    return {
        "schema_version": 1,
        "backend": "physics",
        "stage": stage,
        "validation_kind": "runtime_training_validation",
        "candidate_count": len(audits),
        "safe_candidate_count": len(safe),
        "all_target_pass_candidate_count": sum(
            bool(report["safe"] and report["target_pass"])
            for report in audits
        ),
        "selection_policy": (
            "numerical validity, all active-loop targets, maximum normalized "
            "target violation, violation count, then original six-metric Cost"
        ),
        "selected": selected,
        "candidates": audits,
    }


def select_stage_curriculum_parameters(
    environment: PIDTuningEnv,
    stage: str,
    stage_base_parameters: np.ndarray,
    pool: CandidatePool,
    periodic_candidate_limit: int,
    finalist_limit: int,
    minimum_improvement: float,
    *,
    full_time_domain: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    runtime = validate_candidate_pool(
        environment,
        stage,
        stage_base_parameters,
        pool,
        periodic_candidate_limit,
    )
    runtime_candidates = sorted(
        (
            item
            for item in runtime["candidates"]
            if bool(item["safe"])
        ),
        key=_audit_rank,
    )
    finalist_values = _unique_parameter_sets(
        [
            np.asarray(stage_base_parameters, dtype=np.float64),
            *(
                np.asarray(item["parameters"], dtype=np.float64)
                for item in runtime_candidates[:finalist_limit]
            ),
        ]
    )
    full_audits = [
        audit_parameters(
            environment,
            parameters,
            stage,
            full_time_domain=full_time_domain,
        )
        for parameters in finalist_values
    ]
    baseline_audit = next(
        item
        for item in full_audits
        if np.allclose(
            np.asarray(item["parameters"], dtype=np.float64),
            np.asarray(stage_base_parameters, dtype=np.float64),
            rtol=1e-12,
            atol=1e-14,
        )
    )
    safe_finalists = [item for item in full_audits if bool(item["safe"])]
    best = (
        min(safe_finalists, key=_audit_rank)
        if safe_finalists
        else baseline_audit
    )
    accepted = _audit_improves(
        best,
        baseline_audit,
        minimum_improvement,
    )
    selected = (
        np.asarray(best["parameters"], dtype=np.float64)
        if accepted
        else np.asarray(stage_base_parameters, dtype=np.float64)
    )
    report = {
        "schema_version": 1,
        "backend": "physics",
        "stage": stage,
        "runtime_validation": runtime,
        "finalist_audit_scope": (
            "all_56_models" if full_time_domain else "runtime_validation_probe"
        ),
        "full_audit_finalists": full_audits,
        "baseline_full_audit": baseline_audit,
        "best_full_audit": best,
        "selection_policy": (
            "numerical validity and all active-loop targets precede original "
            "six-metric Cost across the training and validation models present"
        ),
        "accepted": accepted,
        "selected_parameters": selected.tolist(),
    }
    return selected.copy(), report


def _system_manifest(device: str) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "numba": __import__("numba").__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": __import__("stable_baselines3").__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": device,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def _configure_randomness(seed: int, deterministic_algorithms: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(
        bool(deterministic_algorithms),
        warn_only=True,
    )


def _save_rng_state(path: Path) -> None:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    torch.save(state, path)


def _restore_rng_state(path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _save_model_atomic(model: SAC, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.zip")
    model.save(temporary)
    os.replace(temporary, path)


def _checkpoint_name(
    stage_index: int,
    stage: str,
    stage_steps: int,
    global_steps: int,
) -> str:
    return (
        f"stage_{stage_index + 1:02d}_{stage}"
        f"_s{stage_steps:09d}_g{global_steps:09d}"
    )


def _save_resume_checkpoint(
    model: SAC,
    run_dir: Path,
    state: dict[str, Any],
    config: FormalTrainingConfig,
) -> Path:
    checkpoint_root = run_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    name = _checkpoint_name(
        int(state["stage_index"]),
        str(state["stage"]),
        int(state["stage_timesteps_completed"]),
        int(state["global_timesteps_completed"]),
    )
    target = checkpoint_root / name
    if target.exists():
        suffix = 1
        while (checkpoint_root / f"{name}_r{suffix:02d}").exists():
            suffix += 1
        target = checkpoint_root / f"{name}_r{suffix:02d}"
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=checkpoint_root)
    )
    try:
        model.save(temporary / "model.zip")
        if bool(config.payload["checkpoint"]["save_replay_buffer"]):
            model.save_replay_buffer(temporary / "replay_buffer.pkl")
        _save_rng_state(temporary / "rng_state.pt")
        model_environment = (
            model.get_env() if hasattr(model, "get_env") else None
        )
        if model_environment is not None:
            environment_states = model_environment.env_method("export_state")
            with (temporary / "environment_states.pkl").open("wb") as stream:
                cloudpickle.dump(environment_states, stream)
        sampling_saved = isinstance(model_environment, PlantSamplingVecEnv)
        if sampling_saved:
            _atomic_write_json(
                temporary / "plant_sampler_state.json",
                model_environment.export_sampling_state(),
            )
        metadata = {
            "schema_version": TRAINING_PROTOCOL_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "stage_order": list(STAGE_ORDER),
            "stage": state["stage"],
            "stage_index": state["stage_index"],
            "stage_timesteps_completed": state["stage_timesteps_completed"],
            "global_timesteps_completed": state["global_timesteps_completed"],
            "config_sha256": state["config_sha256"],
            "input_fingerprint": state["input_fingerprint"],
            "n_envs": int(
                model_environment.num_envs
                if model_environment is not None
                else getattr(model, "n_envs", 1)
            ),
            "environment_state_saved": model_environment is not None,
            "plant_sampler_state_saved": sampling_saved,
        }
        _atomic_write_json(temporary / "checkpoint.json", metadata)
        (temporary / "COMPLETE").write_text("complete\n", encoding="utf-8")
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    state["latest_checkpoint"] = target.relative_to(run_dir).as_posix()
    state["updated_at_utc"] = utc_now()
    _atomic_write_json(run_dir / "trainer_state.json", state)
    if isinstance(model_environment, PlantSamplingVecEnv):
        _atomic_write_json(
            run_dir / "sampling" / f"{state['stage']}_summary.json",
            model_environment.sampling_summary(),
        )
    _prune_resume_checkpoints(
        checkpoint_root,
        keep_last=int(config.payload["checkpoint"]["keep_last"]),
        protected=target,
    )
    return target


def _prune_resume_checkpoints(
    checkpoint_root: Path,
    *,
    keep_last: int,
    protected: Path,
) -> None:
    completed = sorted(
        (
            path
            for path in checkpoint_root.iterdir()
            if path.is_dir() and (path / "COMPLETE").is_file()
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    keep = set(completed[:keep_last]) | {protected}
    root = checkpoint_root.resolve()
    for path in completed:
        if path in keep:
            continue
        resolved = path.resolve()
        if resolved.parent != root:
            raise RuntimeError(f"refusing to prune unexpected checkpoint path: {resolved}")
        shutil.rmtree(resolved)


def _load_checkpoint(
    run_dir: Path,
    state: Mapping[str, Any],
    environment: VecEnv,
    device: str,
    tensorboard_log: str | None,
    expect_replay_buffer: bool,
) -> SAC:
    relative = state.get("latest_checkpoint")
    if not relative:
        raise ValueError("resume state does not reference a checkpoint")
    checkpoint = (run_dir / str(relative)).resolve()
    if checkpoint.parent != (run_dir / "checkpoints").resolve():
        raise ValueError("resume checkpoint is outside the run checkpoint directory")
    if not (checkpoint / "COMPLETE").is_file():
        raise ValueError("resume checkpoint is incomplete")
    checkpoint_metadata = json.loads(
        (checkpoint / "checkpoint.json").read_text(encoding="utf-8")
    )
    if (
        checkpoint_metadata.get("schema_version")
        != TRAINING_PROTOCOL_SCHEMA_VERSION
    ):
        raise ValueError(
            "checkpoint schema does not match the current four-stage "
            "training protocol"
        )
    if tuple(checkpoint_metadata.get("stage_order", ())) != STAGE_ORDER:
        raise ValueError("checkpoint stage order does not match current protocol")
    if checkpoint_metadata.get("config_sha256") != state.get("config_sha256"):
        raise ValueError("checkpoint configuration does not match trainer state")
    if checkpoint_metadata.get("input_fingerprint") != state.get(
        "input_fingerprint"
    ):
        raise ValueError("checkpoint input fingerprint does not match trainer state")
    checkpoint_stage_index = int(checkpoint_metadata["stage_index"])
    state_stage_index = int(state["stage_index"])
    same_stage = (
        checkpoint_stage_index == state_stage_index
        and checkpoint_metadata["stage"] == state["stage"]
    )
    crossed_stage_boundary = (
        state_stage_index == checkpoint_stage_index + 1
        and int(state["stage_timesteps_completed"]) == 0
        and int(checkpoint_metadata["global_timesteps_completed"])
        == int(state["global_timesteps_completed"])
    )
    if not same_stage and not crossed_stage_boundary:
        raise ValueError(
            "checkpoint stage is incompatible with trainer state: "
            f"checkpoint=({checkpoint_stage_index}, "
            f"{checkpoint_metadata['stage']}), "
            f"state=({state_stage_index}, {state['stage']})"
        )
    expected_n_envs = int(checkpoint_metadata.get("n_envs", 1))
    if environment.num_envs != expected_n_envs:
        raise ValueError(
            "checkpoint environment-count mismatch: "
            f"checkpoint={expected_n_envs}, current={environment.num_envs}"
        )
    environment_state_path = checkpoint / "environment_states.pkl"
    if same_stage and bool(
        checkpoint_metadata.get("environment_state_saved", False)
    ):
        if not environment_state_path.is_file():
            raise FileNotFoundError("checkpoint has no vector environment state")
        with environment_state_path.open("rb") as stream:
            environment_states = cloudpickle.load(stream)
        if len(environment_states) != environment.num_envs:
            raise ValueError("checkpoint vector environment state count mismatch")
        for rank, environment_state in enumerate(environment_states):
            environment.env_method(
                "restore_state",
                environment_state,
                indices=rank,
            )
    if same_stage and isinstance(environment, PlantSamplingVecEnv):
        sampler_path = checkpoint / "plant_sampler_state.json"
        if not checkpoint_metadata.get("plant_sampler_state_saved") or not sampler_path.is_file():
            raise ValueError("checkpoint is missing the required plant sampler state")
        environment.restore_sampling_state(json.loads(sampler_path.read_text(encoding="utf-8")))
    model = SAC.load(
        checkpoint / "model.zip",
        env=environment,
        device=device,
        tensorboard_log=tensorboard_log,
        force_reset=not same_stage,
    )
    replay_path = checkpoint / "replay_buffer.pkl"
    if expect_replay_buffer:
        if not replay_path.is_file():
            raise FileNotFoundError("resume checkpoint has no Replay Buffer")
        model.load_replay_buffer(replay_path)
    _restore_rng_state(checkpoint / "rng_state.pt")
    expected_steps = int(state["global_timesteps_completed"])
    if int(model.num_timesteps) != expected_steps:
        raise ValueError(
            "checkpoint timestep mismatch: "
            f"model={model.num_timesteps}, state={expected_steps}"
        )
    return model


def _new_model(
    environment: VecEnv,
    seed: int,
    device: str,
    tensorboard_log: str | None,
    effective_sac: Mapping[str, Any],
) -> SAC:
    sac = effective_sac
    return SAC(
        str(sac["policy"]),
        environment,
        learning_rate=float(sac["learning_rate"]),
        buffer_size=int(sac["buffer_size"]),
        learning_starts=int(sac["learning_starts"]),
        batch_size=int(sac["batch_size"]),
        tau=float(sac["tau"]),
        gamma=float(sac["gamma"]),
        train_freq=(int(sac["train_frequency"]), "step"),
        gradient_steps=int(sac["gradient_steps"]),
        policy_kwargs={
            "net_arch": [int(value) for value in sac["network_architecture"]]
        },
        tensorboard_log=tensorboard_log,
        seed=seed,
        device=device,
        verbose=0,
    )


def _effective_sac_parameters(
    config: FormalTrainingConfig,
    run_kind: str,
    n_envs: int,
) -> dict[str, Any]:
    parameters = dict(config.payload["sac"])
    parameters["network_architecture"] = list(
        config.payload["sac"]["network_architecture"]
    )
    updates_per_transition = int(
        parameters.pop("gradient_steps_per_transition")
    )
    parameters["gradient_steps"] = (
        int(parameters["train_frequency"])
        * int(n_envs)
        * updates_per_transition
    )
    if run_kind == "engineering_check":
        parameters["buffer_size"] = min(int(parameters["buffer_size"]), 512)
        parameters["learning_starts"] = min(
            int(parameters["learning_starts"]),
            64,
        )
        parameters["batch_size"] = min(int(parameters["batch_size"]), 64)
    return parameters


def _effective_stage_steps(
    config: FormalTrainingConfig,
    engineering_steps_per_stage: int | None,
) -> dict[str, int]:
    if engineering_steps_per_stage is None:
        return {
            stage.name: int(stage.total_timesteps)
            for stage in config.stages
        }
    if engineering_steps_per_stage <= 0:
        raise ValueError("engineering-check steps must be positive")
    return {
        stage.name: int(engineering_steps_per_stage)
        for stage in config.stages
    }


def _reconcile_stage_progress(
    state: dict[str, Any],
    effective_steps: Mapping[str, int],
    global_steps: int,
) -> bool:
    stage_index = int(state["stage_index"])
    if not 0 <= stage_index <= len(STAGE_ORDER):
        raise ValueError("cannot reconcile an invalid stage index")
    prior_steps = sum(
        int(effective_steps[stage]) for stage in STAGE_ORDER[:stage_index]
    )
    completed_global = int(global_steps)
    if stage_index == len(STAGE_ORDER):
        expected_total = prior_steps
        if completed_global != expected_total:
            raise ValueError(
                "post-stage global timestep mismatch: "
                f"model={completed_global}, expected={expected_total}"
            )
        state["global_timesteps_completed"] = completed_global
        return False

    stage_total = int(effective_steps[STAGE_ORDER[stage_index]])
    inferred_stage_steps = completed_global - prior_steps
    if not 0 <= inferred_stage_steps <= stage_total:
        raise ValueError(
            "model timesteps cannot be assigned to the active stage: "
            f"stage={STAGE_ORDER[stage_index]}, inferred={inferred_stage_steps}, "
            f"budget={stage_total}"
        )
    recorded_stage_steps = int(state["stage_timesteps_completed"])
    if recorded_stage_steps > inferred_stage_steps:
        raise ValueError(
            "trainer state is ahead of the model checkpoint: "
            f"state={recorded_stage_steps}, model={inferred_stage_steps}"
        )
    changed = (
        recorded_stage_steps != inferred_stage_steps
        or int(state["global_timesteps_completed"]) != completed_global
    )
    state["stage_timesteps_completed"] = inferred_stage_steps
    state["global_timesteps_completed"] = completed_global
    if changed:
        state["updated_at_utc"] = utc_now()
    return changed


def _initial_state(
    config: FormalTrainingConfig,
    input_manifest: Mapping[str, Any],
    seed: int,
    run_kind: str,
    effective_steps: Mapping[str, int],
    n_envs: int,
    initial_parameters: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_PROTOCOL_SCHEMA_VERSION,
        "backend": "physics",
        "status": "running",
        "run_kind": run_kind,
        "seed": int(seed),
        "config_sha256": config.sha256,
        "input_fingerprint": input_manifest["fingerprint"],
        "effective_stage_timesteps": dict(effective_steps),
        "n_envs": int(n_envs),
        "stage_order": list(STAGE_ORDER),
        "stage_index": 0,
        "stage": STAGE_ORDER[0],
        "stage_timesteps_completed": 0,
        "global_timesteps_completed": 0,
        "curriculum_parameters": np.asarray(
            initial_parameters, dtype=np.float64
        ).tolist(),
        "stage_base_parameters": np.asarray(
            initial_parameters, dtype=np.float64
        ).tolist(),
        "candidate_pool": [],
        "completed_stages": [],
        "latest_checkpoint": None,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "interruption": None,
        "accumulated_wall_time_s": 0.0,
    }


def _verify_resume_state(
    state: Mapping[str, Any],
    config: FormalTrainingConfig,
    input_manifest: Mapping[str, Any],
    seed: int,
    run_kind: str,
    effective_steps: Mapping[str, int],
    n_envs: int,
) -> None:
    if state.get("schema_version") != TRAINING_PROTOCOL_SCHEMA_VERSION:
        raise ValueError(
            "resume state schema does not match the current four-stage "
            "training protocol"
        )
    expected = {
        "backend": "physics",
        "seed": int(seed),
        "config_sha256": config.sha256,
        "run_kind": run_kind,
        "effective_stage_timesteps": dict(effective_steps),
        "n_envs": int(n_envs),
        "stage_order": list(STAGE_ORDER),
    }
    mismatches = {
        key: (state.get(key), value)
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise ValueError(f"resume state is incompatible: {mismatches}")
    if state.get("input_fingerprint") != input_manifest["fingerprint"]:
        raise ValueError("resume input fingerprint changed")
    if state.get("status") == "completed":
        return
    stage_index = int(state["stage_index"])
    if not 0 <= stage_index <= len(STAGE_ORDER):
        raise ValueError("resume stage index is invalid")
    if stage_index == len(STAGE_ORDER):
        if state["stage"] not in {STAGE_ORDER[-1], "completed"}:
            raise ValueError("post-stage resume state has an invalid stage name")
    elif state["stage"] != STAGE_ORDER[stage_index]:
        raise ValueError("resume stage name and index disagree")


def _write_periodic_validation(
    run_dir: Path,
    stage_index: int,
    stage: str,
    global_steps: int,
    report: Mapping[str, Any],
) -> Path:
    directory = run_dir / "validation_reports"
    path = directory / (
        f"stage_{stage_index + 1:02d}_{stage}_g{global_steps:09d}.json"
    )
    _atomic_write_json(path, report)
    return path


def _close_model_environment(model: SAC | None) -> None:
    if model is None:
        return
    environment = model.get_env()
    if environment is not None:
        environment.close()


def run_formal_training(
    project_root: Path,
    *,
    config_path: Path | None,
    seed: int,
    run_dir: Path,
    device: str | None = None,
    resume: bool = False,
    engineering_steps_per_stage: int | None = None,
    n_envs: int | None = None,
    stop_controller: StopController | None = None,
) -> dict[str, Any]:
    """Run or resume one independent formal-training seed."""

    root = Path(project_root).resolve()
    output = Path(run_dir).resolve()
    config = load_formal_training_config(root, config_path)
    if seed not in config.seeds and engineering_steps_per_stage is None:
        raise ValueError(
            f"formal seed {seed} is not declared in the training configuration"
        )
    selected_device = config.default_device if device is None else str(device)
    if selected_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    run_kind = (
        "formal_training"
        if engineering_steps_per_stage is None
        else "engineering_check"
    )
    effective_steps = _effective_stage_steps(config, engineering_steps_per_stage)
    configured_n_envs = int(
        config.payload["parallelism"]["environments_per_seed"]
    )
    selected_n_envs = configured_n_envs if n_envs is None else int(n_envs)
    if selected_n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if any(int(value) % selected_n_envs for value in effective_steps.values()):
        raise ValueError("effective stage timesteps must be divisible by n_envs")
    for interval_name, interval in (
        (
            "checkpoint",
            int(config.payload["checkpoint"]["interval_timesteps"]),
        ),
        (
            "validation",
            int(config.payload["validation"]["interval_timesteps"]),
        ),
    ):
        if interval % selected_n_envs:
            raise ValueError(f"{interval_name} interval must be divisible by n_envs")
    numerical_threads = int(
        config.payload["parallelism"]["numerical_threads_per_process"]
    )
    configure_thread_limits(numerical_threads)
    torch.set_num_threads(numerical_threads)
    effective_sac = _effective_sac_parameters(
        config,
        run_kind,
        selected_n_envs,
    )
    input_manifest = build_training_input_manifest(root, config)
    env_config = config.payload["environment"]
    bootstrap_environment = PIDTuningEnv(
        root,
        stage=STAGE_ORDER[0],
        max_episode_steps=int(env_config["max_episode_steps"]),
        audit_interval=int(env_config["audit_interval"]),
        initial_perturbation=0.0,
    )
    try:
        initial_parameters = bootstrap_environment.parameter_space.initial.copy()
    finally:
        bootstrap_environment.close()
    controller = stop_controller if stop_controller is not None else StopController()

    if resume:
        state_path = output / "trainer_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"missing resume state: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        _verify_resume_state(
            state,
            config,
            input_manifest,
            seed,
            run_kind,
            effective_steps,
            selected_n_envs,
        )
        if state["status"] == "completed":
            summary_path = output / "seed_summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(
                    "completed trainer state has no seed_summary.json"
                )
            return json.loads(summary_path.read_text(encoding="utf-8"))
        state["status"] = "running"
        state["interruption"] = None
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"new formal-training output directory is not empty: {output}"
            )
        output.mkdir(parents=True, exist_ok=True)
        state = _initial_state(
            config,
            input_manifest,
            seed,
            run_kind,
            effective_steps,
            selected_n_envs,
            initial_parameters,
        )
        manifest = {
            "schema_version": TRAINING_PROTOCOL_SCHEMA_VERSION,
            "backend": "physics",
            "run_kind": run_kind,
            "run_name": config.run_name,
            "seed": int(seed),
            "created_at_utc": utc_now(),
            "project_root_at_launch": str(root),
            "output_directory": str(output),
            "configuration_path": str(config.path),
            "configuration": config.payload,
            "initial_parameters": initial_parameters.tolist(),
            "effective_stage_timesteps": effective_steps,
            "effective_sac": effective_sac,
            "effective_parallelism": {
                "n_envs": selected_n_envs,
                "start_method": config.payload["parallelism"]["start_method"],
                "numerical_threads_per_process": numerical_threads,
            },
            "training_inputs": input_manifest,
            "system": _system_manifest(selected_device),
            "data_policy": "training and validation ensemble only",
            "hardware_use_allowed": False,
        }
        _atomic_write_json(output / "run_manifest.json", manifest)
        _atomic_write_json(output / "trainer_state.json", state)

    deterministic = bool(
        config.payload["runtime"]["torch_deterministic_algorithms"]
    )
    _configure_randomness(seed, deterministic)
    tensorboard_log = (
        str(output / str(config.payload["tensorboard"]["subdirectory"]))
        if bool(config.payload["tensorboard"]["enabled"])
        else None
    )
    checkpoint_interval = int(
        config.payload["checkpoint"]["interval_timesteps"]
    )
    validation_interval = int(
        config.payload["validation"]["interval_timesteps"]
    )
    model: SAC | None = None
    environment: PIDTuningEnv | None = None
    training_environment: VecEnv | None = None
    session_started = time.perf_counter()
    previous_wall_time = float(state.get("accumulated_wall_time_s", 0.0))
    run_total_steps = sum(int(effective_steps[stage]) for stage in STAGE_ORDER)

    try:
        while int(state["stage_index"]) < len(STAGE_ORDER):
            stage_index = int(state["stage_index"])
            stage = STAGE_ORDER[stage_index]
            total_stage_steps = int(effective_steps[stage])
            stage_base = np.asarray(
                state["stage_base_parameters"], dtype=np.float64
            )
            if environment is not None:
                environment.close()
            environment = PIDTuningEnv(
                root,
                stage=stage,
                max_episode_steps=int(env_config["max_episode_steps"]),
                audit_interval=int(env_config["audit_interval"]),
                initial_perturbation=float(env_config["initial_perturbation"]),
                base_parameters=stage_base,
            )
            stage_seed = seed + 1009 * stage_index
            environment.action_space.seed(stage_seed)
            environment.reset(
                seed=stage_seed + 999_983,
                options={"perturb": False},
            )
            training_environment = create_training_vec_env(
                root,
                stage=stage,
                max_episode_steps=int(env_config["max_episode_steps"]),
                audit_interval=int(env_config["audit_interval"]),
                initial_perturbation=float(env_config["initial_perturbation"]),
                base_parameters=stage_base,
                n_envs=selected_n_envs,
                stage_seed=stage_seed,
                start_method=str(
                    config.payload["parallelism"]["start_method"]
                ),
                plant_sampling=PlantSamplingConfig.from_mapping(config.payload.get("plant_sampling")).as_dict(),
                sampling_log_path=output / "sampling" / f"{stage}_episodes.jsonl",
            )

            if model is None:
                if resume and int(state["global_timesteps_completed"]) > 0:
                    model = _load_checkpoint(
                        output,
                        state,
                        training_environment,
                        selected_device,
                        tensorboard_log,
                        expect_replay_buffer=bool(
                            config.payload["checkpoint"]["save_replay_buffer"]
                        ),
                    )
                    progress_reconciled = _reconcile_stage_progress(
                        state,
                        effective_steps,
                        int(model.num_timesteps),
                    )
                    if progress_reconciled:
                        _save_resume_checkpoint(model, output, state, config)
                        print(
                            "[resume] reconciled checkpoint progress to "
                            f"stage_steps={state['stage_timesteps_completed']} "
                            f"global_steps={state['global_timesteps_completed']}",
                            flush=True,
                        )
                    resume = False
                else:
                    model = _new_model(
                        training_environment,
                        seed,
                        selected_device,
                        tensorboard_log,
                        effective_sac,
                    )
            else:
                previous = model.get_env()
                model.set_env(training_environment, force_reset=True)
                if previous is not None:
                    previous.close()

            pool = CandidatePool(
                stage,
                int(config.payload["validation"]["candidate_pool_size"]),
                (
                    CandidateRecord.from_payload(item)
                    for item in state.get("candidate_pool", [])
                ),
            )
            completed = int(state["stage_timesteps_completed"])
            stage_start_global_steps = sum(
                int(effective_steps[name]) for name in STAGE_ORDER[:stage_index]
            )
            progress_reporter = TrainingProgressReporter(
                seed=seed,
                stage=stage,
                stage_start_global_steps=stage_start_global_steps,
                stage_total_steps=total_stage_steps,
                run_total_steps=run_total_steps,
                previous_wall_time_s=previous_wall_time,
                session_started=session_started,
                initial_global_steps=int(model.num_timesteps),
                interval_timesteps=int(
                    config.payload["runtime"]["progress_interval_timesteps"]
                ),
            )
            progress_reporter.report(int(model.num_timesteps), force=True)
            next_checkpoint = (
                (completed // checkpoint_interval) + 1
            ) * checkpoint_interval
            next_validation = (
                (completed // validation_interval) + 1
            ) * validation_interval

            while completed < total_stage_steps:
                event_step = min(
                    total_stage_steps,
                    next_checkpoint,
                    next_validation,
                )
                requested_steps = event_step - completed
                callback = CandidateCollectorCallback(
                    pool,
                    controller,
                    progress_reporter,
                )
                before = int(model.num_timesteps)
                model.learn(
                    total_timesteps=requested_steps,
                    callback=callback,
                    reset_num_timesteps=False,
                    progress_bar=bool(config.payload["runtime"]["progress_bar"]),
                    tb_log_name=f"seed_{seed}",
                )
                learned = int(model.num_timesteps) - before
                if learned <= 0:
                    raise RuntimeError("SAC made no progress during a training chunk")
                completed += learned
                state["stage_timesteps_completed"] = completed
                state["global_timesteps_completed"] = int(model.num_timesteps)
                state["candidate_pool"] = pool.to_payload()
                state["updated_at_utc"] = utc_now()

                if completed >= next_validation and not controller.requested:
                    validation_report = validate_candidate_pool(
                        environment,
                        stage,
                        stage_base,
                        pool,
                        int(
                            config.payload["validation"][
                                "periodic_candidate_limit"
                            ]
                        ),
                    )
                    validation_report["global_timesteps"] = int(
                        model.num_timesteps
                    )
                    validation_report["stage_timesteps"] = completed
                    _write_periodic_validation(
                        output,
                        stage_index,
                        stage,
                        int(model.num_timesteps),
                        validation_report,
                    )
                    while next_validation <= completed:
                        next_validation += validation_interval

                checkpoint_due = (
                    completed >= next_checkpoint
                    or completed >= total_stage_steps
                    or controller.requested
                )
                if checkpoint_due:
                    _save_resume_checkpoint(model, output, state, config)
                    while next_checkpoint <= completed:
                        next_checkpoint += checkpoint_interval

                if controller.requested:
                    state["accumulated_wall_time_s"] = (
                        previous_wall_time + time.perf_counter() - session_started
                    )
                    state["status"] = "interrupted"
                    state["interruption"] = {
                        "signal": controller.signal_name,
                        "at_utc": utc_now(),
                    }
                    state["updated_at_utc"] = utc_now()
                    _atomic_write_json(output / "trainer_state.json", state)
                    return {
                        "schema_version": 1,
                        "status": "interrupted",
                        "run_kind": run_kind,
                        "seed": seed,
                        "n_envs": selected_n_envs,
                        "stage": stage,
                        "stage_timesteps_completed": completed,
                        "global_timesteps_completed": int(model.num_timesteps),
                        "accumulated_wall_time_s": state[
                            "accumulated_wall_time_s"
                        ],
                        "run_dir": str(output),
                        "resume_command_required": True,
                    }

            selected, stage_report = select_stage_curriculum_parameters(
                environment,
                stage,
                stage_base,
                pool,
                int(config.payload["validation"]["periodic_candidate_limit"]),
                int(config.payload["validation"]["stage_finalist_limit"]),
                float(
                    config.payload["validation"]["minimum_cost_improvement"]
                ),
                full_time_domain=bool(
                    run_kind == "formal_training" or stage == "joint"
                ),
            )
            stage_report.update(
                {
                    "stage_index": stage_index,
                    "seed": seed,
                    "stage_seed": stage_seed,
                    "stage_timesteps": completed,
                    "global_timesteps": int(model.num_timesteps),
                    "candidate_pool_size": len(pool.records),
                    "plant_sampling": training_environment.sampling_summary(),
                }
            )
            stage_report_path = output / (
                f"stage_{stage_index + 1:02d}_{stage}_report.json"
            )
            _atomic_write_json(stage_report_path, stage_report)
            final_model_path = (
                output
                / "models"
                / f"stage_{stage_index + 1:02d}_{stage}_final.zip"
            )
            _save_model_atomic(model, final_model_path)

            state["completed_stages"].append(
                {
                    "stage": stage,
                    "stage_timesteps": completed,
                    "global_timesteps": int(model.num_timesteps),
                    "selected_parameters": selected.tolist(),
                    "accepted": bool(stage_report["accepted"]),
                    "report": stage_report_path.relative_to(output).as_posix(),
                    "model": final_model_path.relative_to(output).as_posix(),
                    "plant_sampling": stage_report["plant_sampling"],
                }
            )
            state["curriculum_parameters"] = selected.tolist()
            state["stage_index"] = stage_index + 1
            state["stage_timesteps_completed"] = 0
            state["candidate_pool"] = []
            state["updated_at_utc"] = utc_now()
            if int(state["stage_index"]) < len(STAGE_ORDER):
                state["stage"] = STAGE_ORDER[int(state["stage_index"])]
                state["stage_base_parameters"] = selected.tolist()
            _atomic_write_json(output / "trainer_state.json", state)

        if model is None:
            final_parameters_for_resume = np.asarray(
                state["curriculum_parameters"], dtype=np.float64
            )
            env_config = config.payload["environment"]
            environment = PIDTuningEnv(
                root,
                stage="joint",
                max_episode_steps=int(env_config["max_episode_steps"]),
                audit_interval=int(env_config["audit_interval"]),
                initial_perturbation=0.0,
                base_parameters=final_parameters_for_resume,
            )
            final_stage_seed = seed + 1009 * (len(STAGE_ORDER) - 1)
            environment.reset(
                seed=final_stage_seed + 999_983,
                options={"perturb": False},
            )
            training_environment = create_training_vec_env(
                root,
                stage="joint",
                max_episode_steps=int(env_config["max_episode_steps"]),
                audit_interval=int(env_config["audit_interval"]),
                initial_perturbation=0.0,
                base_parameters=final_parameters_for_resume,
                n_envs=selected_n_envs,
                stage_seed=final_stage_seed,
                start_method=str(
                    config.payload["parallelism"]["start_method"]
                ),
                plant_sampling=PlantSamplingConfig.from_mapping(config.payload.get("plant_sampling")).as_dict(),
                sampling_log_path=output / "sampling" / "joint_episodes.jsonl",
            )
            if int(state["global_timesteps_completed"]) <= 0:
                raise RuntimeError(
                    "completed stage state has no model checkpoint to finalize"
                )
            model = _load_checkpoint(
                output,
                state,
                training_environment,
                selected_device,
                tensorboard_log,
                expect_replay_buffer=bool(
                    config.payload["checkpoint"]["save_replay_buffer"]
                ),
            )

        final_parameters = np.asarray(
            state["curriculum_parameters"], dtype=np.float64
        )
        if environment is None or model is None:
            raise RuntimeError("formal training completed without a model")
        final_audit = None
        if state["completed_stages"]:
            last_stage = state["completed_stages"][-1]
            if last_stage["stage"] == "joint":
                last_report = json.loads(
                    (output / last_stage["report"]).read_text(encoding="utf-8")
                )
                if last_report.get("finalist_audit_scope") == "all_56_models":
                    selected_report = (
                        last_report["best_full_audit"]
                        if bool(last_report["accepted"])
                        else last_report["baseline_full_audit"]
                    )
                    if np.allclose(
                        np.asarray(
                            selected_report["parameters"], dtype=np.float64
                        ),
                        final_parameters,
                        rtol=1e-12,
                        atol=1e-14,
                    ):
                        final_audit = selected_report
        if final_audit is None:
            final_audit = audit_parameters(
                environment,
                final_parameters,
                "joint",
                full_time_domain=True,
            )
        eligible = bool(run_kind == "formal_training" and final_audit["safe"])
        accumulated_wall_time = (
            previous_wall_time + time.perf_counter() - session_started
        )
        candidate_path = output / "seed_candidate.npz"
        _atomic_save_npz(
            candidate_path,
            parameter_names=np.asarray(environment.parameter_space.names),
            parameters=final_parameters,
            normalized_parameters=environment.parameter_space.normalize(
                final_parameters
            ),
            seed=np.asarray(seed, dtype=np.int64),
            training_complete=np.asarray(True),
            eligible_for_selection=np.asarray(eligible),
            input_fingerprint=np.asarray(input_manifest["fingerprint"]),
            training_protocol_schema_version=np.asarray(
                TRAINING_PROTOCOL_SCHEMA_VERSION, dtype=np.int64
            ),
        )
        _atomic_write_json(output / "seed_candidate_audit.json", final_audit)
        summary = {
            "schema_version": TRAINING_PROTOCOL_SCHEMA_VERSION,
            "backend": "physics",
            "status": "completed",
            "run_kind": run_kind,
            "seed": seed,
            "n_envs": selected_n_envs,
            "device": str(model.device),
            "total_timesteps": int(model.num_timesteps),
            "effective_stage_timesteps": effective_steps,
            "plant_sampling": PlantSamplingConfig.from_mapping(config.payload.get("plant_sampling")).as_dict(),
            "candidate": candidate_path.relative_to(output).as_posix(),
            "candidate_valid_over_training_models": bool(final_audit["safe"]),
            "candidate_joint_cost": float(final_audit["cost"]),
            "eligible_for_multi_seed_selection": eligible,
            "completed_stages": state["completed_stages"],
            "input_fingerprint": input_manifest["fingerprint"],
            "accumulated_wall_time_s": accumulated_wall_time,
            "mean_environment_steps_per_second": (
                float(model.num_timesteps) / accumulated_wall_time
                if accumulated_wall_time > 0.0
                else 0.0
            ),
            "completed_at_utc": utc_now(),
            "run_dir": str(output),
            "hardware_use_allowed": False,
        }
        _atomic_write_json(output / "seed_summary.json", summary)
        state["accumulated_wall_time_s"] = accumulated_wall_time
        state["status"] = "completed"
        state["stage"] = "completed"
        state["candidate_pool"] = []
        state["updated_at_utc"] = utc_now()
        _atomic_write_json(output / "trainer_state.json", state)
        return summary
    except BaseException as error:
        state["accumulated_wall_time_s"] = (
            previous_wall_time + time.perf_counter() - session_started
        )
        state["status"] = "failed"
        state["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "at_utc": utc_now(),
        }
        state["updated_at_utc"] = utc_now()
        if model is not None and int(state["stage_index"]) < len(STAGE_ORDER):
            try:
                _reconcile_stage_progress(
                    state,
                    effective_steps,
                    int(model.num_timesteps),
                )
                _save_resume_checkpoint(model, output, state, config)
            except BaseException as checkpoint_error:
                state["failure"]["checkpoint_error"] = str(checkpoint_error)
        _atomic_write_json(output / "trainer_state.json", state)
        raise
    finally:
        attached_environment = model.get_env() if model is not None else None
        _close_model_environment(model)
        if (
            training_environment is not None
            and training_environment is not attached_environment
        ):
            training_environment.close()
        if environment is not None:
            environment.close()


def discover_seed_candidates(runs_root: Path) -> list[Path]:
    root = Path(runs_root).resolve()
    return sorted(root.glob("seed_*/seed_candidate.npz"))


def select_multi_seed_candidate(
    project_root: Path,
    candidate_paths: Iterable[Path],
    output_dir: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"selection output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    environment = PIDTuningEnv(
        root,
        stage="joint",
        initial_perturbation=0.0,
    )
    space = environment.parameter_space
    expected_input_fingerprint = build_training_input_manifest(
        root,
        load_formal_training_config(root, config_path),
    )["fingerprint"]
    rows: list[dict[str, Any]] = []
    for raw_path in candidate_paths:
        path = Path(raw_path).resolve()
        with np.load(path, allow_pickle=False) as archive:
            parameter_names = tuple(
                str(value) for value in archive["parameter_names"]
            )
            parameters = np.asarray(archive["parameters"], dtype=np.float64)
            seed = int(np.asarray(archive["seed"]).item())
            complete = bool(np.asarray(archive["training_complete"]).item())
            eligible = bool(
                np.asarray(archive["eligible_for_selection"]).item()
            )
            fingerprint = str(np.asarray(archive["input_fingerprint"]).item())
            protocol_schema = int(
                np.asarray(
                    archive.get("training_protocol_schema_version", -1)
                ).item()
            )
        if parameter_names != space.names:
            raise ValueError(f"candidate parameter order is invalid: {path}")
        space.normalize(parameters)
        if not complete or not eligible:
            raise ValueError(
                f"candidate is not eligible for formal selection: {path}"
            )
        if protocol_schema != TRAINING_PROTOCOL_SCHEMA_VERSION:
            raise ValueError(
                f"candidate does not match current protocol: {path}"
            )
        if fingerprint != expected_input_fingerprint:
            raise ValueError(
                f"candidate training input fingerprint does not match current protocol: {path}"
            )
        result = environment.audit_parameters(
            parameters,
            full_time_domain=True,
        )
        safe = bool(result["safe"])
        cost = float(result["cost"])
        validation_cost = combined_stage_cost(
            result["frequency"]["validation_diagnostics"],
            result["time_domain"]["validation_diagnostics"],
            "joint",
        )
        audit = {
            "schema_version": 1,
            "backend": "physics",
            "candidate": str(path),
            "seed": seed,
            "input_fingerprint": fingerprint,
            "safe": safe,
            "target_pass": bool(result["target_pass"]),
            "maximum_target_violation": float(
                result["maximum_target_violation"]
            ),
            "target_violation_count": int(result["target_violation_count"]),
            "joint_cost": cost,
            "validation_joint_cost": float(validation_cost),
            "parameter_names": list(space.names),
            "parameters": parameters.tolist(),
            "frequency": result["frequency"],
            "time_domain": result["time_domain"],
            "hardware_use_allowed": False,
        }
        audit_path = output / "audits" / f"seed_{seed}_audit.json"
        _atomic_write_json(audit_path, audit)
        rows.append(
            {
                "seed": seed,
                "candidate": str(path),
                "candidate_sha256": sha256_file(path),
                "input_fingerprint": fingerprint,
                "safe": safe,
                "target_pass": bool(result["target_pass"]),
                "maximum_target_violation": float(
                    result["maximum_target_violation"]
                ),
                "target_violation_count": int(
                    result["target_violation_count"]
                ),
                "joint_cost": float(cost),
                "validation_joint_cost": float(validation_cost),
                "parameters": parameters.tolist(),
                "audit": audit_path.relative_to(output).as_posix(),
            }
        )

    if len(rows) < 3:
        raise ValueError("multi-seed selection requires at least three candidates")
    seeds = [int(row["seed"]) for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("multi-seed selection received duplicate seeds")
    fingerprints = {row["input_fingerprint"] for row in rows}
    if len(fingerprints) != 1:
        raise ValueError("candidate runs used different training inputs")
    safe_rows = [row for row in rows if row["safe"]]
    if not safe_rows:
        raise RuntimeError(
            "no candidate is numerically valid over the training and validation models"
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            not bool(row["safe"]),
            not bool(row["target_pass"]),
            float(row["maximum_target_violation"]),
            int(row["target_violation_count"]),
            float(row["validation_joint_cost"]),
            float(row["joint_cost"]),
            int(row["seed"]),
        ),
    )
    selected = ranked[0]
    selected_parameters = np.asarray(selected["parameters"], dtype=np.float64)
    final_candidate = output / "final_candidate.npz"
    _atomic_save_npz(
        final_candidate,
        parameter_names=np.asarray(space.names),
        parameters=selected_parameters,
        normalized_parameters=space.normalize(selected_parameters),
        selected_seed=np.asarray(selected["seed"], dtype=np.int64),
        source_candidate_sha256=np.asarray(selected["candidate_sha256"]),
        input_fingerprint=np.asarray(selected["input_fingerprint"]),
        training_protocol_schema_version=np.asarray(
            TRAINING_PROTOCOL_SCHEMA_VERSION, dtype=np.int64
        ),
        valid_over_training_models=np.asarray(True),
        valid_over_validation_models=np.asarray(True),
        all_training_validation_targets_met=np.asarray(
            selected["target_pass"]
        ),
        maximum_target_violation=np.asarray(
            selected["maximum_target_violation"], dtype=np.float64
        ),
        target_violation_count=np.asarray(
            selected["target_violation_count"], dtype=np.int64
        ),
    )
    leaderboard = {
        "schema_version": 1,
        "backend": "physics",
        "selection_policy": (
            "numerical validity and all six targets over the 40 training and "
            "16 validation models, then minimum maximum normalized target "
            "violation, violation count, validation joint Cost and training "
            "joint Cost"
        ),
        "candidate_count": len(rows),
        "safe_candidate_count": len(safe_rows),
        "all_target_pass_candidate_count": sum(
            bool(row["safe"] and row["target_pass"]) for row in rows
        ),
        "ranking": ranked,
        "selected_seed": selected["seed"],
        "selected_source_candidate": selected["candidate"],
        "selected_joint_cost": selected["joint_cost"],
        "selected_validation_joint_cost": selected["validation_joint_cost"],
        "selected_target_pass": selected["target_pass"],
        "selected_maximum_target_violation": selected[
            "maximum_target_violation"
        ],
        "selected_target_violation_count": selected[
            "target_violation_count"
        ],
        "final_candidate": final_candidate.name,
        "created_at_utc": utc_now(),
        "hardware_use_allowed": False,
    }
    _atomic_write_json(output / "candidate_leaderboard.json", leaderboard)
    selected_audit = json.loads(
        (output / selected["audit"]).read_text(encoding="utf-8")
    )
    _atomic_write_json(output / "final_candidate_audit.json", selected_audit)
    environment.close()
    return leaderboard
