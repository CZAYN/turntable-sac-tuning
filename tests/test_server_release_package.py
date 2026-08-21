import importlib.util
import io
import json
from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_TEST_PACKAGE_SCRIPT = PROJECT_ROOT / "scripts" / "package_server_final_test.py"
RELEASE_PACKAGE_SCRIPT = PROJECT_ROOT / "scripts" / "package_server_release.py"


def _load_packager(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_final_test_package_is_reproducible_and_excludes_training_runtime(tmp_path):
    packager = _load_packager("package_server_final_test", FINAL_TEST_PACKAGE_SCRIPT)
    first_path = tmp_path / "first_final.zip"
    second_path = tmp_path / "second_final.zip"
    first = packager.build_package(PROJECT_ROOT, first_path, overwrite=False)
    second = packager.build_package(PROJECT_ROOT, second_path, overwrite=False)
    assert first["archive_sha256"] == second["archive_sha256"]

    with zipfile.ZipFile(first_path) as archive:
        names = set(archive.namelist())
        embedded = json.loads(
            archive.read("FINAL_TEST_PACKAGE_MANIFEST.json").decode("utf-8")
        )
    assert "scripts/lock_final_candidate.py" in names
    assert "scripts/run_final_test.py" in names
    assert "data/processed/physics_motor_six_metric_test_v2.npz" in names
    assert "src/elc_rl/simulation_kernel.py" in names
    assert "src/elc_rl/discrete_loop_model.py" in names
    assert embedded["entry_count"] == len(names) - 1
    assert "scripts/train_sac.py" not in names
    assert "scripts/select_final_candidate.py" not in names
    assert "src/elc_rl/sac_training.py" not in names
    assert not any("frf_tasks" in name for name in names)
    assert "src/elc_rl/task_dataset.py" not in names


def test_single_upload_release_contains_separate_inner_archives(tmp_path):
    packager = _load_packager("package_server_release", RELEASE_PACKAGE_SCRIPT)
    first_path = tmp_path / "first_release.zip"
    second_path = tmp_path / "second_release.zip"
    first = packager.build_release(PROJECT_ROOT, first_path, overwrite=False)
    second = packager.build_release(PROJECT_ROOT, second_path, overwrite=False)
    assert first["archive_sha256"] == second["archive_sha256"]

    with zipfile.ZipFile(first_path) as release:
        names = set(release.namelist())
        manifest = json.loads(
            release.read("SERVER_RELEASE_MANIFEST.json").decode("utf-8")
        )
        training_bytes = release.read("training/elc_rl_server_training.zip")
        final_test_bytes = release.read("final_test/elc_rl_server_final_test.zip")

    assert names == {
        "training/elc_rl_server_training.zip",
        "final_test/elc_rl_server_final_test.zip",
        "SERVER_RELEASE.md",
        "SERVER_RELEASE_MANIFEST.json",
    }
    assert manifest["entry_count"] == 3

    with zipfile.ZipFile(io.BytesIO(training_bytes)) as training:
        training_names = set(training.namelist())
    with zipfile.ZipFile(io.BytesIO(final_test_bytes)) as final_test:
        final_test_names = set(final_test.namelist())

    assert "scripts/train_sac.py" in training_names
    assert "scripts/train_all_seeds.py" in training_names
    assert "src/elc_rl/parallel_env.py" in training_names
    assert "src/elc_rl/simulation_kernel.py" in training_names
    assert "src/elc_rl/discrete_loop_model.py" in training_names
    assert not any("physics_motor_test" in name for name in training_names)
    assert not any("frf_tasks" in name for name in training_names)
    assert "src/elc_rl/task_dataset.py" not in training_names
    assert "data/processed/physics_motor_six_metric_test_v2.npz" in final_test_names
    assert "src/elc_rl/simulation_kernel.py" in final_test_names
    assert "src/elc_rl/discrete_loop_model.py" in final_test_names
    assert "scripts/train_sac.py" not in final_test_names
    assert not any("frf_tasks" in name for name in final_test_names)
    assert "src/elc_rl/task_dataset.py" not in final_test_names
