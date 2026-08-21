from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.parallel_env import (  # noqa: E402
    configure_thread_limits,
    create_training_vec_env,
)
from elc_rl.sac_training import load_formal_training_config  # noqa: E402
from elc_rl.tuning_env import PIDTuningEnv, STAGE_ORDER  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark vectorized physics-environment transition throughput."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--steps-per-env", type=int, default=64)
    parser.add_argument("--stage", choices=STAGE_ORDER, default="joint")
    arguments = parser.parse_args()

    root = arguments.project_root.resolve()
    config = load_formal_training_config(root, arguments.config)
    n_envs = (
        int(config.payload["parallelism"]["environments_per_seed"])
        if arguments.n_envs is None
        else int(arguments.n_envs)
    )
    if n_envs <= 0:
        parser.error("--n-envs must be positive")
    if arguments.steps_per_env <= 0:
        parser.error("--steps-per-env must be positive")
    thread_count = int(
        config.payload["parallelism"]["numerical_threads_per_process"]
    )
    configure_thread_limits(thread_count)
    environment_config = config.payload["environment"]
    bootstrap = PIDTuningEnv(
        root,
        stage=str(arguments.stage),
        max_episode_steps=int(environment_config["max_episode_steps"]),
        audit_interval=int(environment_config["audit_interval"]),
        initial_perturbation=0.0,
    )
    try:
        base_parameters = bootstrap.parameter_space.initial.copy()
    finally:
        bootstrap.close()
    environment = create_training_vec_env(
        root,
        stage=str(arguments.stage),
        max_episode_steps=int(environment_config["max_episode_steps"]),
        audit_interval=int(environment_config["audit_interval"]),
        initial_perturbation=0.0,
        base_parameters=base_parameters,
        n_envs=n_envs,
        stage_seed=20260728,
        start_method=str(config.payload["parallelism"]["start_method"]),
    )
    actions = np.zeros((n_envs, 11), dtype=np.float32)
    try:
        environment.reset()
        start = time.perf_counter()
        for _ in range(arguments.steps_per_env):
            environment.step(actions)
        elapsed = time.perf_counter() - start
    finally:
        environment.close()

    transitions = n_envs * int(arguments.steps_per_env)
    print(
        json.dumps(
            {
                "stage": str(arguments.stage),
                "n_envs": n_envs,
                "steps_per_env": int(arguments.steps_per_env),
                "transitions": transitions,
                "elapsed_s": elapsed,
                "transitions_per_second": transitions / elapsed,
                "scope": "environment_only_without_SAC_gradient_updates",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
