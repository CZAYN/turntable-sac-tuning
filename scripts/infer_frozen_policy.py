"""Run the existing frozen-policy curriculum with source fingerprint checks."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from elc_rl.frozen_policy import run_frozen_policy_curriculum
import math
from collections.abc import Mapping
import numpy as np


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes-per-stage", type=int, default=4)
    parser.add_argument("--steps-per-episode", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    result = run_frozen_policy_curriculum(
        ROOT, args.run_dir, device=args.device,
        episodes_per_stage=args.episodes_per_stage,
        steps_per_episode=args.steps_per_episode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())

if __name__ == "__main__":
    main()
