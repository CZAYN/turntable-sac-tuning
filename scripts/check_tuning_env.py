from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import gymnasium
import numpy as np
import stable_baselines3
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gymnasium.utils.env_checker import check_env as gymnasium_check_env  # noqa: E402
from stable_baselines3.common.env_checker import check_env as sb3_check_env  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

from elc_rl.tuning_env import PIDTuningEnv, STAGE_ORDER  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the PID tuning environment.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="device for the SAC policy construction check",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON report path (defaults under the selected project root)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run check_env and a short joint-stage probe instead of all stage probes",
    )
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    if arguments.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested but CUDA is unavailable")
    policy_device = (
        "cuda" if arguments.device == "auto" and torch.cuda.is_available()
        else "cpu" if arguments.device == "auto"
        else arguments.device
    )
    selected_stages = ("joint",) if arguments.quick else STAGE_ORDER
    transition_budget = 2 if arguments.quick else 16
    summaries = []
    for stage_index, stage in enumerate(selected_stages):
        environment = PIDTuningEnv(
            root,
            stage=stage,
            max_episode_steps=8,
            audit_interval=8,
            initial_perturbation=0.02,
        )
        if stage_index == 0:
            gymnasium_check_env(environment, skip_render_check=True)
            sb3_check_env(environment, warn=True)
        observation, info = environment.reset(seed=20260715)
        environment.action_space.seed(20260715 + stage_index)
        reward_sum = 0.0
        transitions = 0
        for _ in range(transition_budget):
            action = environment.action_space.sample()
            observation, reward, terminated, truncated, info = environment.step(action)
            reward_sum += reward
            transitions += 1
            if terminated or truncated:
                observation, info = environment.reset()
        summaries.append(
            {
                "stage": stage,
                "transitions": transitions,
                "reward_sum": reward_sum,
                "observation_keys": sorted(observation),
                "last_fast_safe": info["fast_safe"],
            }
        )
        environment.close()

    policy_environment = PIDTuningEnv(
        root,
        stage="joint",
        max_episode_steps=2,
        audit_interval=2,
        initial_perturbation=0.0,
    )
    policy_model = SAC(
        "MultiInputPolicy",
        policy_environment,
        policy_kwargs={"net_arch": [64, 64]},
        device=policy_device,
        seed=20260715,
        verbose=0,
    )
    policy_observation, _ = policy_environment.reset(
        seed=20260715, options={"perturb": False}
    )
    policy_action, _ = policy_model.predict(policy_observation, deterministic=True)
    if policy_action.shape != (11,) or not np.isfinite(policy_action).all():
        raise RuntimeError("SAC MultiInputPolicy prediction is invalid")
    effective_policy_device = str(policy_model.device)
    policy_model.get_env().close()

    benchmark = PIDTuningEnv(
        root,
        stage="joint",
        max_episode_steps=32,
        audit_interval=16,
        initial_perturbation=0.0,
    )
    benchmark.reset(seed=20260715, options={"perturb": False})
    zero_action = np.zeros(benchmark.action_space.shape, dtype=np.float32)
    benchmark_steps = 4 if arguments.quick else 128
    start = time.perf_counter()
    for _ in range(benchmark_steps):
        _, _, terminated, truncated, _ = benchmark.step(zero_action)
        if terminated or truncated:
            benchmark.reset(options={"perturb": False})
    elapsed = time.perf_counter() - start
    benchmark.close()

    report = {
        "schema_version": 1,
        "backend": "physics",
        "reward": "six_target_metrics",
        "gymnasium_version": gymnasium.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "gymnasium_check_env": "passed",
        "stable_baselines3_check_env": "passed",
        "sac_multi_input_policy": "passed",
        "sac_policy_device": effective_policy_device,
        "stages": summaries,
        "benchmark": {
            "steps": benchmark_steps,
            "elapsed_s": elapsed,
            "steps_per_second": benchmark_steps / elapsed,
            "audit_interval": benchmark.audit_interval,
        },
    }
    output = (
        root / "outputs" / "environment_validation_physics.json"
        if arguments.output is None
        else arguments.output.resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
