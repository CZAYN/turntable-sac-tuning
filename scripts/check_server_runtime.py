from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.sac_training import (  # noqa: E402
    build_training_input_manifest,
    load_formal_training_config,
)
from elc_rl.tuning_env import PIDTuningEnv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Python runtime before launching formal SAC training."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--skip-environment-step", action="store_true")
    arguments = parser.parse_args()

    root = arguments.project_root.resolve()
    config = load_formal_training_config(root, arguments.config)
    device = config.default_device if arguments.device is None else arguments.device
    output_root = (
        root / "outputs"
        if arguments.output_root is None
        else arguments.output_root.resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    checks: dict[str, object] = {}

    checks["cuda_available"] = torch.cuda.is_available()
    checks["requested_device"] = device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but PyTorch CUDA is unavailable")
    checks["gpu_name"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    )
    checks["torch_version"] = torch.__version__
    checks["torch_cuda_version"] = torch.version.cuda

    manifest = build_training_input_manifest(root, config)
    checks["training_input_file_count"] = len(manifest["files"])
    checks["training_input_fingerprint"] = manifest["fingerprint"]

    usage = shutil.disk_usage(output_root)
    free_gb = usage.free / (1024**3)
    minimum_gb = float(config.payload["runtime"]["minimum_free_disk_gb"])
    checks["free_disk_gb"] = free_gb
    checks["minimum_free_disk_gb"] = minimum_gb
    if free_gb < minimum_gb:
        raise RuntimeError(
            f"free disk {free_gb:.2f} GiB is below required {minimum_gb:.2f} GiB"
        )

    with tempfile.TemporaryDirectory(dir=output_root) as temporary:
        probe = Path(temporary) / "write_probe.txt"
        probe.write_text("ok\n", encoding="utf-8")
        checks["output_write_probe"] = probe.read_text(encoding="utf-8").strip() == "ok"

    environment = PIDTuningEnv(
        root,
        stage="joint",
        max_episode_steps=2,
        audit_interval=2,
        initial_perturbation=0.0,
    )
    observation, reset_info = environment.reset(
        seed=config.seeds[0],
        options={"perturb": False},
    )
    checks["observation_valid"] = environment.observation_space.contains(observation)
    checks["reset_safe"] = bool(
        reset_info["fast_safe"] and reset_info["audit_safe"]
    )
    if not arguments.skip_environment_step:
        transition = environment.step(np.zeros(11, dtype=np.float32))
        checks["environment_step_valid"] = bool(
            environment.observation_space.contains(transition[0])
            and np.isfinite(transition[1])
        )
    environment.close()

    checks["passed"] = bool(
        checks["observation_valid"]
        and checks["reset_safe"]
        and checks["output_write_probe"]
        and checks.get("environment_step_valid", True)
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if checks["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
