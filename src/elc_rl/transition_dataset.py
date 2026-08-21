"""Reproducible pre-training transitions for the staged tuning environment.

These records are reinforcement-learning experience tuples, not supervised
labels for optimal controller gains.  Production SAC training continues to
collect fresh tuples online from the same environment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .tuning_env import (
    OBSERVATION_KEYS,
    PERFORMANCE_METRIC_NAMES,
    PIDTuningEnv,
    STAGE_ORDER,
)


TRANSITION_SCHEMA_VERSION = 4
DEFAULT_TRANSITIONS_PER_STAGE = 128
DEFAULT_SEED = 20260722


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stack(rows: list[np.ndarray], dtype: np.dtype[Any]) -> np.ndarray:
    return np.asarray(rows, dtype=dtype)


def validate_transition_archive(path: Path) -> dict[str, Any]:
    """Validate schema-4 tuple alignment, shapes, bounds, and finiteness."""

    with np.load(path, allow_pickle=False) as data:
        required = {
            "schema_version",
            "action",
            "reward",
            "terminated",
            "truncated",
            "stage_index",
            "episode_id",
            "episode_step",
            "parameters_physical",
            "next_parameters_physical",
            "frequency_stage_cost",
            "time_stage_cost",
            "total_stage_cost",
            "performance_metric_names",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"transition archive is missing arrays: {sorted(missing)}")
        if int(data["schema_version"]) != TRANSITION_SCHEMA_VERSION:
            raise ValueError("unexpected transition schema version")
        count = int(data["action"].shape[0])
        if count <= 0 or data["action"].shape != (count, 11):
            raise ValueError("action array must have shape (N, 11) with N > 0")
        for name in data.files:
            if name in {
                "schema_version",
                "observation_keys",
                "performance_metric_names",
            }:
                continue
            array = data[name]
            if array.shape[0] != count:
                raise ValueError(f"unaligned transition array: {name}")
            if array.dtype.kind in "fc" and not np.isfinite(array).all():
                raise ValueError(f"non-finite transition values: {name}")
        if np.any(np.abs(data["action"]) > 1.0):
            raise ValueError("transition action lies outside [-1, 1]")
        stage_index = data["stage_index"]
        if np.any((stage_index < 0) | (stage_index >= len(STAGE_ORDER))):
            raise ValueError("invalid stage index")
        observation_keys = tuple(str(value) for value in data["observation_keys"])
        if (
            len(observation_keys) != len(OBSERVATION_KEYS)
            or set(observation_keys) != set(OBSERVATION_KEYS)
        ):
            raise ValueError(
                "transition observation keys do not match the pure-physics schema"
            )
        for key in observation_keys:
            observation_name = f"observation__{key}"
            next_name = f"next_observation__{key}"
            if observation_name not in data or next_name not in data:
                raise ValueError(f"missing observation pair for {key}")
            if data[observation_name].shape != data[next_name].shape:
                raise ValueError(f"observation shape mismatch for {key}")
        if "performance_metrics" not in observation_keys:
            raise ValueError("schema 4 requires the six-metric performance observation")
        if data["observation__performance_metrics"].shape != (count, 18):
            raise ValueError("performance metric observation must have shape (N, 18)")
        metric_names = tuple(str(value) for value in data["performance_metric_names"])
        if metric_names != PERFORMANCE_METRIC_NAMES:
            raise ValueError("schema-4 performance metric order is invalid")
        legacy_keys = {"metrics", "time_metrics", "frf_context"}
        if legacy_keys.intersection(observation_keys):
            raise ValueError("schema 4 cannot contain legacy objective observations")
        return {
            "transition_count": count,
            "observation_keys": list(observation_keys),
            "terminated_count": int(np.sum(data["terminated"])),
            "truncated_count": int(np.sum(data["truncated"])),
            "stage_counts": {
                stage: int(np.sum(stage_index == index))
                for index, stage in enumerate(STAGE_ORDER)
            },
        }


def generate_transition_dataset(
    project_root: Path,
    output_path: Path,
    *,
    transitions_per_stage: int = DEFAULT_TRANSITIONS_PER_STAGE,
    seed: int = DEFAULT_SEED,
    stages: Iterable[str] = STAGE_ORDER,
) -> dict[str, Any]:
    """Generate deterministic local-policy tuples for pipeline verification."""

    if transitions_per_stage <= 0:
        raise ValueError("transitions_per_stage must be positive")
    selected_stages = tuple(stages)
    if not selected_stages or any(stage not in STAGE_ORDER for stage in selected_stages):
        raise ValueError(f"stages must be a non-empty subset of {STAGE_ORDER}")
    if len(set(selected_stages)) != len(selected_stages):
        raise ValueError("stages must not contain duplicates")

    observations: dict[str, list[np.ndarray]] = {}
    next_observations: dict[str, list[np.ndarray]] = {}
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated_values: list[bool] = []
    truncated_values: list[bool] = []
    stage_indices: list[int] = []
    episode_ids: list[int] = []
    episode_steps: list[int] = []
    frequency_costs: list[float] = []
    time_costs: list[float] = []
    total_costs: list[float] = []
    parameters: list[np.ndarray] = []
    next_parameters: list[np.ndarray] = []
    sampled_model_ids: list[tuple[str, ...]] = []

    episode_id = 0
    for stage_offset, stage in enumerate(selected_stages):
        stage_seed = int(seed + 1009 * stage_offset)
        rng = np.random.default_rng(stage_seed)
        environment = PIDTuningEnv(
            project_root,
            stage=stage,
            max_episode_steps=32,
            audit_interval=16,
            initial_perturbation=0.03,
        )
        observation, info = environment.reset(seed=stage_seed)
        step_in_episode = 0
        for _ in range(transitions_per_stage):
            before_parameters = environment.parameters
            action = np.clip(rng.normal(0.0, 0.25, size=11), -1.0, 1.0)
            if rng.random() < 0.10:
                action.fill(0.0)
            action *= environment.action_mask.astype(np.float64)
            next_observation, reward, terminated, truncated, next_info = (
                environment.step(action.astype(np.float32))
            )
            if not observations:
                observations = {key: [] for key in observation}
                next_observations = {key: [] for key in observation}
            for key in observations:
                observations[key].append(observation[key].copy())
                next_observations[key].append(next_observation[key].copy())
            actions.append(action.astype(np.float32))
            rewards.append(float(reward))
            terminated_values.append(bool(terminated))
            truncated_values.append(bool(truncated))
            stage_indices.append(STAGE_ORDER.index(stage))
            episode_ids.append(episode_id)
            episode_steps.append(step_in_episode)
            frequency_costs.append(float(next_info["frequency_stage_cost"]))
            time_costs.append(float(next_info["time_stage_cost"]))
            total_costs.append(float(next_info["stage_cost"]))
            parameters.append(before_parameters)
            next_parameters.append(environment.parameters)
            sampled_model_ids.append(tuple(next_info["sampled_model_ids"]))

            observation = next_observation
            info = next_info
            step_in_episode += 1
            if terminated or truncated:
                episode_id += 1
                step_in_episode = 0
                observation, info = environment.reset()
        environment.close()
        episode_id += 1

    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(TRANSITION_SCHEMA_VERSION, dtype=np.int16),
        "observation_keys": np.asarray(sorted(observations)),
        "performance_metric_names": np.asarray(PERFORMANCE_METRIC_NAMES),
        "action": _stack(actions, np.dtype(np.float32)),
        "reward": _stack(rewards, np.dtype(np.float32)),
        "terminated": _stack(terminated_values, np.dtype(np.bool_)),
        "truncated": _stack(truncated_values, np.dtype(np.bool_)),
        "stage_index": _stack(stage_indices, np.dtype(np.int8)),
        "episode_id": _stack(episode_ids, np.dtype(np.int32)),
        "episode_step": _stack(episode_steps, np.dtype(np.int16)),
        "frequency_stage_cost": _stack(frequency_costs, np.dtype(np.float32)),
        "time_stage_cost": _stack(time_costs, np.dtype(np.float32)),
        "total_stage_cost": _stack(total_costs, np.dtype(np.float32)),
        "parameters_physical": _stack(parameters, np.dtype(np.float64)),
        "next_parameters_physical": _stack(next_parameters, np.dtype(np.float64)),
        "sampled_model_ids": np.asarray(sampled_model_ids),
    }
    for key in sorted(observations):
        arrays[f"observation__{key}"] = _stack(
            observations[key], np.dtype(np.float32)
        )
        arrays[f"next_observation__{key}"] = _stack(
            next_observations[key], np.dtype(np.float32)
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    validation = validate_transition_archive(output_path)
    manifest = {
        "schema_version": TRANSITION_SCHEMA_VERSION,
        "task_id": "cgs_turntable_001",
        "environment_backend": "physics",
        "archive": output_path.name,
        "sha256": _sha256(output_path),
        "seed": int(seed),
        "behavior_policy": (
            "stage-masked clipped Gaussian delta action, sigma=0.25, "
            "10 percent zero actions"
        ),
        "data_role": (
            "pre-training RL experience tuples for reproducibility and pipeline validation; "
            "not optimal-parameter labels"
        ),
        "objective": (
            "three loops times six literal performance metrics: closed-loop "
            "bandwidth, gain margin, phase margin, overshoot, rise time and "
            "settling time"
        ),
        "performance_metric_order": list(PERFORMANCE_METRIC_NAMES),
        "tuple_semantics": "(observation, action, reward, next_observation, terminated, truncated)",
        "stages": list(selected_stages),
        "transitions_per_stage": int(transitions_per_stage),
        **validation,
        "observation_schema": {
            key: {
                "shape": list(arrays[f"observation__{key}"].shape[1:]),
                "dtype": str(arrays[f"observation__{key}"].dtype),
            }
            for key in sorted(observations)
        },
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
