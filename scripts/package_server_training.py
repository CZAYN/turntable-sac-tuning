from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SERVER_PACKAGE_FILES = (
    "pyproject.toml",
    "environment-server.yml",
    "SERVER_TRAINING.md",
    "config/controller_performance_targets.json",
    "config/motor_physics.json",
    "config/sac_training.json",
    "data/processed/controller_parameter_space.json",
    "data/processed/physics_motor_ensemble.npz",
    "data/processed/physics_motor_ensemble_manifest.json",
    "scripts/benchmark_parallel_env.py",
    "scripts/check_server_runtime.py",
    "scripts/select_final_candidate.py",
    "scripts/train_all_seeds.py",
    "scripts/train_sac.py",
    "src/elc_rl/__init__.py",
    "src/elc_rl/controller_parameters.py",
    "src/elc_rl/discrete_loop_model.py",
    "src/elc_rl/evaluation_utils.py",
    "src/elc_rl/parallel_env.py",
    "src/elc_rl/performance_targets.py",
    "src/elc_rl/physics_evaluator.py",
    "src/elc_rl/physics_motor_model.py",
    "src/elc_rl/sac_training.py",
    "src/elc_rl/simulation_kernel.py",
    "src/elc_rl/tuning_env.py",
)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def build_package(project_root: Path, output: Path, overwrite: bool) -> dict[str, object]:
    root = project_root.resolve()
    destination = output.resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"server package already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    contents: list[tuple[str, bytes]] = []
    for relative in SERVER_PACKAGE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing server-package file: {path}")
        content = path.read_bytes()
        normalized = relative.replace("\\", "/")
        entries.append(
            {
                "path": normalized,
                "size_bytes": len(content),
                "sha256": _sha256_bytes(content),
            }
        )
        contents.append((normalized, content))

    internal_manifest = {
        "schema_version": 1,
        "package_kind": "physics_formal_training_only",
        "data_policy": "contains only training and validation runtime inputs",
        "entry_count": len(entries),
        "entries": entries,
    }
    temporary = destination.with_name(f".{destination.name}.tmp")
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name, content in contents:
            archive.writestr(_zip_info(name), content)
        archive.writestr(
            _zip_info("SERVER_PACKAGE_MANIFEST.json"),
            json.dumps(
                internal_manifest,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")
            + b"\n",
        )
    temporary.replace(destination)
    external_manifest = {
        **internal_manifest,
        "archive": destination.name,
        "archive_size_bytes": destination.stat().st_size,
        "archive_sha256": _sha256_file(destination),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = destination.with_name(f"{destination.stem}_manifest.json")
    manifest_path.write_text(
        json.dumps(external_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return external_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic server bundle for formal SAC training."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    output = (
        root / "dist" / "elc_rl_server_training.zip"
        if arguments.output is None
        else arguments.output.resolve()
    )
    result = build_package(root, output, arguments.overwrite)
    print(
        json.dumps(
            {
                "archive": str(output),
                "archive_size_bytes": result["archive_size_bytes"],
                "archive_sha256": result["archive_sha256"],
                "entry_count": result["entry_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
