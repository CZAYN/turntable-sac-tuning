from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.transition_dataset import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_TRANSITIONS_PER_STAGE,
    generate_transition_dataset,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic schema-4 pre-training transitions for the "
            "four-stage, six-metric environment."
        )
    )
    parser.add_argument(
        "--transitions-per-stage",
        type=int,
        default=DEFAULT_TRANSITIONS_PER_STAGE,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    arguments = parser.parse_args()
    output = (
        arguments.output
        if arguments.output is not None
        else PROJECT_ROOT
        / "data"
        / "processed"
        / f"rl_transitions_physics_seed{arguments.seed}.npz"
    )
    manifest = generate_transition_dataset(
        PROJECT_ROOT,
        output,
        transitions_per_stage=arguments.transitions_per_stage,
        seed=arguments.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
