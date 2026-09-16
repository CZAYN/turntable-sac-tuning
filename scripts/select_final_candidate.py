from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.sac_training import (  # noqa: E402
    discover_seed_candidates,
    select_multi_seed_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select one final 11-D candidate from completed independent "
            "formal-training seeds using training/validation audits only."
        )
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("candidates", nargs="*", type=Path)
    arguments = parser.parse_args()

    project_root = arguments.project_root.resolve()
    runs_root = (
        project_root / "outputs" / "crossq_training"
        if arguments.runs_root is None
        else arguments.runs_root.resolve()
    )
    candidates = (
        [path.resolve() for path in arguments.candidates]
        if arguments.candidates
        else discover_seed_candidates(runs_root)
    )
    if not candidates:
        raise SystemExit(f"no seed candidates found under {runs_root}")
    output_dir = (
        runs_root / "selection"
        if arguments.output_dir is None
        else arguments.output_dir.resolve()
    )
    result = select_multi_seed_candidate(
        project_root,
        candidates,
        output_dir,
        arguments.config,
    )
    print(
        json.dumps(
            {
                "candidate_count": result["candidate_count"],
                "safe_candidate_count": result["safe_candidate_count"],
                "selected_seed": result["selected_seed"],
                "selected_joint_cost": result["selected_joint_cost"],
                "selected_validation_joint_cost": result[
                    "selected_validation_joint_cost"
                ],
                "selected_target_pass": result["selected_target_pass"],
                "final_candidate": str(output_dir / result["final_candidate"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
