from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.sac_training import load_formal_training_config  # noqa: E402


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all declared formal CrossQ seeds with bounded concurrency."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--parallel-seeds", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--engineering-check-steps-per-stage", type=int, default=None)
    arguments = parser.parse_args()

    root = arguments.project_root.resolve()
    config = load_formal_training_config(root, arguments.config)
    parallel_seeds = (
        int(config.payload["parallelism"]["concurrent_seeds"])
        if arguments.parallel_seeds is None
        else int(arguments.parallel_seeds)
    )
    if parallel_seeds <= 0:
        parser.error("--parallel-seeds must be positive")
    if arguments.n_envs is not None and arguments.n_envs <= 0:
        parser.error("--n-envs must be positive")

    engineering_check = arguments.engineering_check_steps_per_stage is not None
    default_run_directory = (
        f"{config.run_name}_engineering_check"
        if engineering_check
        else config.run_name
    )
    output_root = (
        root / "outputs" / default_run_directory
        if arguments.output_root is None
        else arguments.output_root.resolve()
    )
    log_root = (
        root / "logs" / default_run_directory
        if arguments.log_root is None
        else arguments.log_root.resolve()
    )
    log_root.mkdir(parents=True, exist_ok=True)
    pending = list(config.seeds)
    running: dict[subprocess.Popen[bytes], tuple[int, TextIO, Path]] = {}
    completed: dict[int, int] = {}
    stop_requested = False

    def stop_children(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"[launcher] received signal {signum}; stopping children", flush=True)
        for process in list(running):
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, stop_children)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_children)

    try:
        while pending or running:
            while pending and len(running) < parallel_seeds and not stop_requested:
                seed = pending.pop(0)
                run_dir = output_root / f"seed_{seed}"
                log_path = log_root / f"seed_{seed}.log"
                if not arguments.resume and run_dir.exists() and any(run_dir.iterdir()):
                    raise FileExistsError(
                        f"seed output directory is not empty: {run_dir}"
                    )
                command = [
                    sys.executable,
                    "-u",
                    str(root / "scripts" / "train_sac.py"),
                    "--project-root",
                    str(root),
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(run_dir),
                ]
                command.extend(["--config", str(config.path)])
                if arguments.device is not None:
                    command.extend(["--device", str(arguments.device)])
                if arguments.n_envs is not None:
                    command.extend(["--n-envs", str(arguments.n_envs)])
                resume_seed = arguments.resume and (run_dir / "trainer_state.json").is_file()
                if arguments.resume and run_dir.exists() and any(run_dir.iterdir()) and not resume_seed:
                    raise FileExistsError(f"cannot resume a nonempty seed directory without trainer_state.json: {run_dir}")
                if resume_seed:
                    command.append("--resume")
                if arguments.engineering_check_steps_per_stage is not None:
                    command.extend(
                        [
                            "--engineering-check-steps-per-stage",
                            str(arguments.engineering_check_steps_per_stage),
                        ]
                    )
                log_handle = log_path.open(
                    "a" if arguments.resume else "w",
                    encoding="utf-8",
                )
                process = subprocess.Popen(
                    command,
                    cwd=root,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                running[process] = (seed, log_handle, log_path)
                print(
                    f"[launcher] seed={seed} pid={process.pid} log={log_path}",
                    flush=True,
                )

            registry = {
                "schema_version": 1,
                "launcher_pid": os.getpid(),
                "run_name": config.run_name,
                "run_kind": (
                    "engineering_check" if engineering_check else "formal_training"
                ),
                "output_root": str(output_root),
                "resume": bool(arguments.resume),
                "running": [
                    {
                        "seed": seed,
                        "pid": process.pid,
                        "log": str(log_path),
                    }
                    for process, (seed, _, log_path) in running.items()
                    if process.poll() is None
                ],
                "completed": completed,
            }
            _atomic_write_json(log_root / "launcher_state.json", registry)

            for process, (seed, log_handle, log_path) in list(running.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                log_handle.close()
                completed[seed] = int(return_code)
                del running[process]
                print(
                    f"[launcher] seed={seed} exited code={return_code} "
                    f"log={log_path}",
                    flush=True,
                )
                if return_code != 0:
                    stop_requested = True
                    for sibling in running:
                        if sibling.poll() is None:
                            sibling.terminate()

            registry["running"] = [
                {
                    "seed": seed,
                    "pid": process.pid,
                    "log": str(log_path),
                }
                for process, (seed, _, log_path) in running.items()
                if process.poll() is None
            ]
            registry["completed"] = completed
            _atomic_write_json(log_root / "launcher_state.json", registry)

            if running:
                time.sleep(1.0)
            if stop_requested and not running:
                break
    finally:
        for process, (_, log_handle, _) in list(running.items()):
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log_handle.close()

    failed = {seed: code for seed, code in completed.items() if code != 0}
    return 1 if stop_requested or failed or pending else 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
