"""Factories for deterministic parallel PID tuning environments."""

from __future__ import annotations

from functools import partial
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from .tuning_env import PIDTuningEnv
from .plant_sampling import PlantSamplingConfig, StagePlantSampler
from .sampling_vec_env import PlantSamplingVecEnv
from .physics_evaluator import get_physics_controller_evaluator


def configure_thread_limits(thread_count: int = 1) -> None:
    """Limit nested numerical runtimes before spawning environment workers."""

    value = str(int(thread_count))
    if int(thread_count) <= 0:
        raise ValueError("thread_count must be positive")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value


def resolve_start_method(configured: str) -> str:
    if configured == "auto":
        return "spawn" if os.name == "nt" else "forkserver"
    if configured not in {"spawn", "forkserver"}:
        raise ValueError("parallel start_method must be auto, spawn or forkserver")
    if os.name == "nt" and configured == "forkserver":
        raise ValueError("forkserver is unavailable on Windows")
    return configured


def _build_environment(
    project_root: str,
    stage: str,
    max_episode_steps: int,
    audit_interval: int,
    initial_perturbation: float,
    base_parameters: np.ndarray,
    worker_rank: int,
) -> PIDTuningEnv:
    return PIDTuningEnv(
        Path(project_root),
        stage=stage,
        max_episode_steps=max_episode_steps,
        audit_interval=audit_interval,
        initial_perturbation=initial_perturbation,
        base_parameters=np.asarray(base_parameters, dtype=np.float64),
        worker_rank=worker_rank,
    )


def create_training_vec_env(
    project_root: Path,
    *,
    stage: str,
    max_episode_steps: int,
    audit_interval: int,
    initial_perturbation: float,
    base_parameters: np.ndarray,
    n_envs: int,
    stage_seed: int,
    start_method: str,
    plant_sampling: Mapping[str, Any] | None = None,
    sampling_log_path: Path | None = None,
) -> VecEnv:
    """Create seeded CPU workers for one SAC learner."""

    worker_count = int(n_envs)
    if worker_count <= 0:
        raise ValueError("n_envs must be positive")
    root = str(Path(project_root).resolve())
    parameters = np.asarray(base_parameters, dtype=np.float64)
    factories = [
        partial(
            _build_environment,
            root,
            stage,
            int(max_episode_steps),
            int(audit_interval),
            float(initial_perturbation),
            parameters.copy(),
            rank,
        )
        for rank in range(worker_count)
    ]
    if worker_count == 1:
        environment: VecEnv = DummyVecEnv(factories)
    else:
        environment = SubprocVecEnv(
            factories,
            start_method=resolve_start_method(start_method),
        )
    if plant_sampling is not None:
        evaluator = get_physics_controller_evaluator(Path(root))
        sampler = StagePlantSampler(
            evaluator.model_ids(evaluator.training_indices), stage,
            PlantSamplingConfig.from_mapping(plant_sampling),
        )
        environment = PlantSamplingVecEnv(environment, sampler, sampling_log_path)
    environment.seed(int(stage_seed))
    return environment
