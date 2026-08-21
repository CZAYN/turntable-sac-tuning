from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elc_rl.controller_parameters import (  # noqa: E402
    PHYSICS_PARAMETER_SPACE_JSON,
    PHYSICS_PARAMETER_SPACE_NPZ,
    build_physics_controller_parameter_space,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the auditable 11-D physics controller parameter space."
    )
    parser.add_argument(
        "--current-feasibility-report",
        type=Path,
        help=(
            "Optional non-training 40 kHz scan report. When supplied, only a "
            "validated selected current-PIDF candidate may minimally expand bounds."
        ),
    )
    arguments = parser.parse_args()
    payload = build_physics_controller_parameter_space(
        PROJECT_ROOT,
        current_feasibility_report=arguments.current_feasibility_report,
    )
    json_name = PHYSICS_PARAMETER_SPACE_JSON
    npz_name = PHYSICS_PARAMETER_SPACE_NPZ
    json_path = PROJECT_ROOT / "data" / "processed" / json_name
    npz_path = PROJECT_ROOT / "data" / "processed" / npz_name
    summary = {
        "profile": "physics",
        "parameter_count": len(payload["parameters"]),
        "source_baselines": sum(
            parameter["source_kind"] == "excel_baseline"
            for parameter in payload["parameters"]
        ),
        "simulation_only_initials": sum(
            parameter["source_kind"] != "excel_baseline"
            for parameter in payload["parameters"]
        ),
        "sample_periods_s": payload["digital_conversion"]["sample_periods_s"],
        "current_pidf_feasibility_evidence": payload[
            "current_pidf_feasibility_evidence"
        ],
        "json_sha256": _sha256(json_path),
        "npz_sha256": _sha256(npz_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
