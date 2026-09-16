from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from stable_baselines3 import SAC
from .crossq import StageCrossQ

from .sac_training import (
    TRAINING_PROTOCOL_SCHEMA_VERSION,
    CandidatePool,
    CandidateRecord,
    build_training_input_manifest,
    load_formal_training_config,
    select_stage_curriculum_parameters,
)
from .tuning_env import PIDTuningEnv, STAGE_ORDER


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_frozen_policy_run(run_dir: Path) -> dict[str, Any]:
    """Validate one completed four-stage run without loading pickle data."""

    run = Path(run_dir).resolve()
    manifest_path = run / "run_manifest.json"
    summary_path = run / "seed_summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("policy run is missing its manifest or seed summary")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != TRAINING_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("policy run does not match the current training protocol")
    if manifest.get("run_kind") != "formal_training":
        raise ValueError("policy weights must come from formal training")
    if summary.get("status") != "completed":
        raise ValueError("policy run is not complete")
    if int(summary.get("seed", -1)) != int(manifest.get("seed", -2)):
        raise ValueError("policy run seed provenance is inconsistent")

    manifest_fingerprint = str(manifest["training_inputs"]["fingerprint"])
    if str(summary.get("input_fingerprint")) != manifest_fingerprint:
        raise ValueError("policy run input fingerprint is inconsistent")

    completed = list(summary.get("completed_stages", ()))
    if tuple(item.get("stage") for item in completed) != STAGE_ORDER:
        raise ValueError("policy run does not contain the required four stages")

    models_root = (run / "models").resolve()
    stages: list[dict[str, Any]] = []
    for item in completed:
        model_path = (run / str(item["model"])).resolve()
        if model_path.parent != models_root or model_path.suffix != ".zip":
            raise ValueError("stage model path is outside the run models directory")
        if not model_path.is_file():
            raise FileNotFoundError(f"stage model is missing: {model_path}")
        selected = np.asarray(item["selected_parameters"], dtype=np.float64)
        if selected.shape != (11,) or not np.isfinite(selected).all():
            raise ValueError("stage selected parameters are invalid")
        stages.append(
            {
                "stage": str(item["stage"]),
                "model_path": model_path,
                "model_sha256": _sha256(model_path),
                "training_selected_parameters": selected,
            }
        )

    initial = np.asarray(manifest["initial_parameters"], dtype=np.float64)
    if initial.shape != (11,) or not np.isfinite(initial).all():
        raise ValueError("policy run initial parameters are invalid")
    return {
        "run_dir": run,
        "seed": int(summary["seed"]),
        "input_fingerprint": manifest_fingerprint,
        "configuration_filename": Path(
            str(manifest["configuration_path"])
        ).name,
        "configuration": manifest["configuration"],
        "initial_parameters": initial,
        "stages": stages,
    }


def _validate_policy_action(action: np.ndarray) -> np.ndarray:
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (11,) or not np.isfinite(values).all():
        raise RuntimeError("frozen policy produced an invalid 11-D action")
    if np.any(values < -1.0000001) or np.any(values > 1.0000001):
        raise RuntimeError("frozen policy action is outside [-1, 1]")
    return values


def run_frozen_policy_curriculum(
    project_root: Path,
    run_dir: Path,
    *,
    episodes_per_stage: int = 4,
    steps_per_episode: int | None = None,
    rollout_seed: int | None = None,
    device: str = "cpu",
    initial_parameters: np.ndarray | None = None,
    verify_training_inputs: bool = True,
    training_project_root: Path | None = None,
) -> dict[str, Any]:
    """Run four frozen CrossQ or SAC snapshots and select a curriculum candidate.

    This is the internal inference core. It uses the project's existing plant,
    targets, audits, and parameter space; public input and result file schemas
    are deliberately outside this function.
    """

    if episodes_per_stage <= 0:
        raise ValueError("episodes_per_stage must be positive")
    source = inspect_frozen_policy_run(run_dir)
    root = Path(project_root).resolve()
    if verify_training_inputs:
        training_root = (
            root
            if training_project_root is None
            else Path(training_project_root).resolve()
        )
        local_config_path = (
            training_root / "config" / source["configuration_filename"]
        )
        local_config = load_formal_training_config(
            training_root, local_config_path
        )
        local_fingerprint = build_training_input_manifest(
            training_root,
            local_config,
        )["fingerprint"]
        if local_fingerprint != source["input_fingerprint"]:
            raise ValueError(
                "local training inputs do not match the frozen policy source"
            )
    configuration: Mapping[str, Any] = source["configuration"]
    algorithm = configuration.get("algorithm", "sac")
    if algorithm not in {"sac", "crossq"}:
        raise ValueError(f"unsupported frozen policy algorithm: {algorithm}")
    model_class = StageCrossQ if algorithm == "crossq" else SAC
    environment_config = configuration["environment"]
    validation_config = configuration["validation"]
    episode_steps = (
        int(environment_config["max_episode_steps"])
        if steps_per_episode is None
        else int(steps_per_episode)
    )
    if episode_steps <= 0:
        raise ValueError("steps_per_episode must be positive")

    parameters = (
        source["initial_parameters"].copy()
        if initial_parameters is None
        else np.asarray(initial_parameters, dtype=np.float64).copy()
    )
    if parameters.shape != (11,) or not np.isfinite(parameters).all():
        raise ValueError("initial_parameters must be finite shape-(11,) values")
    base_seed = source["seed"] if rollout_seed is None else int(rollout_seed)
    stage_results: list[dict[str, Any]] = []
    global_step = 0

    for stage_index, stage_source in enumerate(source["stages"]):
        stage = str(stage_source["stage"])
        stage_base = parameters.copy()
        environment = PIDTuningEnv(
            root,
            stage=stage,
            max_episode_steps=episode_steps,
            audit_interval=int(environment_config["audit_interval"]),
            initial_perturbation=0.0,
            base_parameters=stage_base,
        )
        pool = CandidatePool(
            stage,
            int(validation_config["candidate_pool_size"]),
        )
        model: SAC | None = None
        terminated_episodes = 0
        try:
            model = model_class.load(
                stage_source["model_path"],
                env=environment,
                device=device,
            )
            model.policy.set_training_mode(False)
            for tensor in model.policy.parameters():
                tensor.requires_grad_(False)
            if tuple(model.action_space.shape) != (11,):
                raise RuntimeError("frozen policy action space is not 11-D")

            for episode in range(episodes_per_stage):
                observation, _ = environment.reset(
                    seed=base_seed + 1009 * stage_index + episode,
                    options={"perturb": False},
                )
                if not environment.observation_space.contains(observation):
                    raise RuntimeError("frozen policy received an invalid observation")
                for _ in range(episode_steps):
                    action, _ = model.predict(observation, deterministic=True)
                    values = _validate_policy_action(action)
                    observation, _, terminated, truncated, info = environment.step(
                        values
                    )
                    global_step += 1
                    if bool(info["fast_safe"]):
                        pool.add(
                            CandidateRecord(
                                stage=stage,
                                fast_cost=float(info["stage_cost"]),
                                parameters=np.asarray(
                                    info["parameters"], dtype=np.float64
                                ),
                                global_timestep=global_step,
                                fast_target_pass=bool(
                                    info["stage_target_pass"]
                                ),
                                fast_maximum_target_violation=float(
                                    info["stage_maximum_target_violation"]
                                ),
                                fast_target_violation_count=int(
                                    info["stage_target_violation_count"]
                                ),
                            )
                        )
                    if terminated or truncated:
                        terminated_episodes += int(terminated)
                        break

            parameters, selection = select_stage_curriculum_parameters(
                environment,
                stage,
                stage_base,
                pool,
                int(validation_config["periodic_candidate_limit"]),
                int(validation_config["stage_finalist_limit"]),
                float(validation_config["minimum_cost_improvement"]),
                full_time_domain=True,
            )
        finally:
            model_environment = None if model is None else model.get_env()
            if model_environment is not None:
                model_environment.close()
            else:
                environment.close()

        stage_results.append(
            {
                "stage": stage,
                "model_sha256": stage_source["model_sha256"],
                "weights_frozen": True,
                "episodes": episodes_per_stage,
                "steps_per_episode": episode_steps,
                "candidate_count": len(pool.records),
                "terminated_episodes": terminated_episodes,
                "accepted": bool(selection["accepted"]),
                "selected_parameters": parameters.copy(),
                "selection": selection,
            }
        )

    return {
        "training_seed": source["seed"],
        "input_fingerprint": source["input_fingerprint"],
        "weights_frozen": True,
        "deterministic": True,
        "episodes_per_stage": episodes_per_stage,
        "steps_per_episode": episode_steps,
        "stages": stage_results,
        "parameters": parameters.copy(),
    }
