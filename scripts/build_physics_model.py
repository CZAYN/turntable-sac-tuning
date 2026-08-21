from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.controller_parameters import (  # noqa: E402
    build_physics_controller_parameter_space,
    load_physics_controller_parameter_space,
)
from elc_rl.physics_motor_model import (  # noqa: E402
    build_physics_motor_ensemble,
    load_physics_motor_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the physics ensemble and 11-D controller parameter space."
    )
    parser.add_argument(
        "--current-feasibility-report",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "current_pidf_feasibility_40khz"
            / "multirate_feasibility_report.json"
        ),
        help=(
            "Validated non-training 40 kHz current-loop feasibility report used "
            "to seed and minimally expand the current PIDF bounds. Defaults to "
            "the tracked canonical evidence under data/processed."
        ),
    )
    arguments = parser.parse_args()
    config = load_physics_motor_config(PROJECT_ROOT)
    ensemble = build_physics_motor_ensemble(PROJECT_ROOT)
    build_physics_controller_parameter_space(
        PROJECT_ROOT,
        current_feasibility_report=arguments.current_feasibility_report,
    )
    space = load_physics_controller_parameter_space(PROJECT_ROOT)
    print(
        json.dumps(
            {
                "model_id": config.payload["model_id"],
                "sample_periods_s": config.sample_periods_s,
                "sample_frequencies_hz": {
                    loop: 1.0 / period
                    for loop, period in config.sample_periods_s.items()
                },
                "controller_update_ratios": config.controller_update_ratios,
                "ensemble_models": int(ensemble["parameters"].shape[0]),
                "training_models": int(
                    (ensemble["active_for_training"] == 1).sum()
                ),
                "validation_models": int(
                    (ensemble["role"] == "validation").sum()
                ),
                "parameter_names": list(space.names),
                "initial_parameters": {
                    name: float(value)
                    for name, value in zip(space.names, space.initial)
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
