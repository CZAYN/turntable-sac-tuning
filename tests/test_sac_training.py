import json
from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import numpy as np
import pytest

from elc_rl.sac_training import (
    CandidateCollectorCallback,
    CandidatePool,
    CandidateRecord,
    StopController,
    TRAINING_PROTOCOL_SCHEMA_VERSION,
    TrainingProgressReporter,
    _effective_sac_parameters,
    _load_checkpoint,
    _reconcile_stage_progress,
    _save_resume_checkpoint,
    build_training_input_manifest,
    load_formal_training_config,
    select_multi_seed_candidate,
)
from elc_rl.controller_parameters import load_physics_controller_parameter_space
from elc_rl.tuning_env import STAGE_ORDER


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_formal_training_configuration_is_complete_and_multi_seed():
    config = load_formal_training_config(PROJECT_ROOT)
    assert tuple(stage.name for stage in config.stages) == STAGE_ORDER
    assert STAGE_ORDER == ("current", "speed", "position", "joint")
    assert "dobc" not in STAGE_ORDER
    assert all(stage.total_timesteps > 0 for stage in config.stages)
    assert len(config.seeds) >= 3
    assert len(set(config.seeds)) == len(config.seeds)
    assert config.payload["default_device"] == "cuda"
    assert config.payload["checkpoint"]["save_replay_buffer"]
    assert config.payload["tensorboard"]["enabled"]


def test_training_input_manifest_is_deterministic_and_training_only():
    config = load_formal_training_config(PROJECT_ROOT)
    first = build_training_input_manifest(PROJECT_ROOT, config)
    second = build_training_input_manifest(PROJECT_ROOT, config)
    assert first == second
    assert first["schema_version"] == TRAINING_PROTOCOL_SCHEMA_VERSION
    assert len(first["fingerprint"]) == 64
    names = {item["path"] for item in first["files"]}
    assert "data/processed/physics_motor_ensemble.npz" in names
    assert "scripts/train_sac.py" in names
    assert "config/controller_performance_targets.json" in names
    assert "src/elc_rl/performance_targets.py" in names
    assert not any("physics_motor_test" in name for name in names)
    assert not any("final_test" in name for name in names)
    assert not any("frf_tasks" in name for name in names)
    assert "src/elc_rl/task_dataset.py" not in names


def test_candidate_pool_deduplicates_and_keeps_the_lowest_costs():
    pool = CandidatePool("joint", maximum_size=2)
    first = np.arange(11, dtype=np.float64)
    second = first + 1.0
    third = first + 2.0
    pool.add(CandidateRecord("joint", 3.0, first, 1))
    pool.add(CandidateRecord("joint", 2.0, second, 2))
    pool.add(CandidateRecord("joint", 1.0, third, 3))
    assert [record.fast_cost for record in pool.records] == [1.0, 2.0]

    pool.add(CandidateRecord("joint", 0.5, second.copy(), 4))
    assert len(pool.records) == 2
    assert [record.fast_cost for record in pool.records] == [0.5, 1.0]
    assert pool.records[0].global_timestep == 4


class _FakeVecEnv:
    num_envs = 2

    def env_method(self, method_name: str):
        assert method_name == "export_state"
        return [{"worker_rank": 0}, {"worker_rank": 1}]


class _FakeSAC:
    def __init__(self, timesteps: int, environment=None) -> None:
        self.num_timesteps = timesteps
        self.environment = environment
        self.n_envs = 1 if environment is None else environment.num_envs

    def save(self, path: Path) -> None:
        Path(path).write_bytes(b"model")

    def save_replay_buffer(self, path: Path) -> None:
        Path(path).write_bytes(b"replay")

    def get_env(self):
        return self.environment


def test_resume_checkpoint_is_complete_and_rotated(tmp_path):
    config = load_formal_training_config(PROJECT_ROOT)
    state = {
        "schema_version": TRAINING_PROTOCOL_SCHEMA_VERSION,
        "stage_order": list(STAGE_ORDER),
        "stage_index": 0,
        "stage": "current",
        "stage_timesteps_completed": 0,
        "global_timesteps_completed": 0,
        "config_sha256": config.sha256,
        "input_fingerprint": "a" * 64,
        "latest_checkpoint": None,
    }
    for steps in (10, 20, 30):
        state["stage_timesteps_completed"] = steps
        state["global_timesteps_completed"] = steps
        checkpoint = _save_resume_checkpoint(
            _FakeSAC(steps),
            tmp_path,
            state,
            config,
        )
        assert (checkpoint / "COMPLETE").is_file()
        assert (checkpoint / "model.zip").read_bytes() == b"model"
        assert (checkpoint / "replay_buffer.pkl").read_bytes() == b"replay"
        assert (checkpoint / "rng_state.pt").is_file()

    completed = [
        path
        for path in (tmp_path / "checkpoints").iterdir()
        if path.is_dir() and (path / "COMPLETE").is_file()
    ]
    assert len(completed) == config.payload["checkpoint"]["keep_last"]
    persisted = json.loads(
        (tmp_path / "trainer_state.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (checkpoint / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert metadata["schema_version"] == TRAINING_PROTOCOL_SCHEMA_VERSION
    assert tuple(metadata["stage_order"]) == STAGE_ORDER
    assert persisted["global_timesteps_completed"] == 30
    assert (tmp_path / persisted["latest_checkpoint"]).is_dir()


def test_resume_checkpoint_contains_every_vector_environment_state(tmp_path):
    config = load_formal_training_config(PROJECT_ROOT)
    state = {
        "schema_version": TRAINING_PROTOCOL_SCHEMA_VERSION,
        "stage_order": list(STAGE_ORDER),
        "stage_index": 0,
        "stage": "current",
        "stage_timesteps_completed": 8,
        "global_timesteps_completed": 8,
        "config_sha256": config.sha256,
        "input_fingerprint": "b" * 64,
        "latest_checkpoint": None,
    }
    checkpoint = _save_resume_checkpoint(
        _FakeSAC(8, _FakeVecEnv()),
        tmp_path,
        state,
        config,
    )
    metadata = json.loads(
        (checkpoint / "checkpoint.json").read_text(encoding="utf-8")
    )
    with (checkpoint / "environment_states.pkl").open("rb") as stream:
        environment_states = cloudpickle.load(stream)
    assert metadata["n_envs"] == 2
    assert metadata["environment_state_saved"]
    assert [item["worker_rank"] for item in environment_states] == [0, 1]


class _ResumeVecEnv:
    num_envs = 2

    def __init__(self) -> None:
        self.restored: list[tuple[int, dict[str, object]]] = []

    def env_method(
        self,
        method_name: str,
        state: dict[str, object],
        *,
        indices: int,
    ):
        assert method_name == "restore_state"
        self.restored.append((indices, state))


def test_resume_at_stage_boundary_starts_fresh_environment(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoints" / "current_complete"
    checkpoint.mkdir(parents=True)
    (checkpoint / "COMPLETE").write_text("complete\n", encoding="utf-8")
    (checkpoint / "checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": TRAINING_PROTOCOL_SCHEMA_VERSION,
                "stage_order": list(STAGE_ORDER),
                "stage": "current",
                "stage_index": 0,
                "stage_timesteps_completed": 80,
                "global_timesteps_completed": 80,
                "n_envs": 2,
                "environment_state_saved": True,
                "config_sha256": "config",
                "input_fingerprint": "fingerprint",
            }
        ),
        encoding="utf-8",
    )
    state = {
        "latest_checkpoint": "checkpoints/current_complete",
        "stage": "speed",
        "stage_index": 1,
        "stage_timesteps_completed": 0,
        "global_timesteps_completed": 80,
        "config_sha256": "config",
        "input_fingerprint": "fingerprint",
    }
    load_arguments = {}
    loaded_model = SimpleNamespace(num_timesteps=80)

    def fake_load(*_args, **kwargs):
        load_arguments.update(kwargs)
        return loaded_model

    monkeypatch.setattr(
        "elc_rl.sac_training.SAC",
        SimpleNamespace(load=fake_load),
    )
    monkeypatch.setattr(
        "elc_rl.sac_training._restore_rng_state",
        lambda _path: None,
    )
    environment = _ResumeVecEnv()
    result = _load_checkpoint(
        tmp_path,
        state,
        environment,
        "cpu",
        None,
        expect_replay_buffer=False,
    )
    assert result is loaded_model
    assert load_arguments["force_reset"]
    assert environment.restored == []


def test_old_five_stage_checkpoint_schema_is_rejected(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "old_protocol"
    checkpoint.mkdir(parents=True)
    (checkpoint / "COMPLETE").write_text("complete\n", encoding="utf-8")
    (checkpoint / "checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "dobc",
                "stage_index": 3,
                "stage_timesteps_completed": 1,
                "global_timesteps_completed": 1,
            }
        ),
        encoding="utf-8",
    )
    state = {
        "latest_checkpoint": "checkpoints/old_protocol",
        "stage": "position",
        "stage_index": 2,
        "stage_timesteps_completed": 1,
        "global_timesteps_completed": 1,
    }
    with pytest.raises(ValueError, match="checkpoint schema"):
        _load_checkpoint(
            tmp_path,
            state,
            _ResumeVecEnv(),
            "cpu",
            None,
            expect_replay_buffer=False,
        )


def test_parallel_sampling_preserves_gradient_updates_per_transition():
    config = load_formal_training_config(PROJECT_ROOT)
    single = _effective_sac_parameters(config, "formal_training", 1)
    parallel = _effective_sac_parameters(config, "formal_training", 4)
    assert single["gradient_steps"] == 1
    assert parallel["gradient_steps"] == 4
    assert "gradient_steps_per_transition" not in parallel


def test_candidate_callback_assigns_distinct_vector_transition_timesteps():
    pool = CandidatePool("joint", maximum_size=8)
    callback = CandidateCollectorCallback(pool, StopController())
    callback.model = SimpleNamespace(num_timesteps=104)
    callback.locals = {
        "infos": [
            {
                "fast_safe": True,
                "stage_cost": float(cost),
                "parameters": np.full(11, cost, dtype=np.float64),
            }
            for cost in (1, 2, 3, 4)
        ]
    }
    assert callback._on_step()
    by_cost = {record.fast_cost: record.global_timestep for record in pool.records}
    assert by_cost == {1.0: 101, 2.0: 102, 3.0: 103, 4.0: 104}


def test_failed_checkpoint_progress_is_reconciled_from_model_timesteps():
    config = load_formal_training_config(PROJECT_ROOT)
    effective_steps = {
        stage.name: stage.total_timesteps for stage in config.stages
    }
    state = {
        "stage_index": 1,
        "stage": "speed",
        "stage_timesteps_completed": 0,
        "global_timesteps_completed": 30395,
    }

    assert _reconcile_stage_progress(state, effective_steps, 30395)
    assert state["stage_timesteps_completed"] == 395
    assert state["global_timesteps_completed"] == 30395

    state["stage_timesteps_completed"] = 500
    with pytest.raises(ValueError, match="ahead of the model"):
        _reconcile_stage_progress(state, effective_steps, 30395)


def test_progress_reporter_prints_redirect_safe_eta(capsys):
    reporter = TrainingProgressReporter(
        seed=20260801,
        stage="speed",
        stage_start_global_steps=30000,
        stage_total_steps=40000,
        run_total_steps=260000,
        previous_wall_time_s=100.0,
        session_started=0.0,
        initial_global_steps=30395,
    )
    message = reporter.report(31000, force=True)
    captured = capsys.readouterr().out
    assert message is not None
    assert "seed=20260801" in captured
    assert "stage=speed" in captured
    assert "stage_steps=1000/40000" in captured
    assert "total_steps=31000/260000" in captured
    assert "eta=" in captured


def test_engineering_candidate_is_rejected_by_formal_selection(tmp_path):
    space = load_physics_controller_parameter_space(PROJECT_ROOT)
    candidate = tmp_path / "engineering_candidate.npz"
    np.savez_compressed(
        candidate,
        parameter_names=np.asarray(space.names),
        parameters=space.initial,
        seed=np.asarray(1, dtype=np.int64),
        training_complete=np.asarray(True),
        eligible_for_selection=np.asarray(False),
        input_fingerprint=np.asarray("a" * 64),
        training_protocol_schema_version=np.asarray(
            TRAINING_PROTOCOL_SCHEMA_VERSION, dtype=np.int64
        ),
    )
    with pytest.raises(ValueError, match="not eligible"):
        select_multi_seed_candidate(
            PROJECT_ROOT,
            [candidate],
            tmp_path / "selection",
        )


def test_candidate_from_old_training_protocol_is_rejected(tmp_path):
    space = load_physics_controller_parameter_space(PROJECT_ROOT)
    candidate = tmp_path / "old_protocol_candidate.npz"
    np.savez_compressed(
        candidate,
        parameter_names=np.asarray(space.names),
        parameters=space.initial,
        seed=np.asarray(1, dtype=np.int64),
        training_complete=np.asarray(True),
        eligible_for_selection=np.asarray(True),
        input_fingerprint=np.asarray("a" * 64),
        training_protocol_schema_version=np.asarray(1, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="does not match current protocol"):
        select_multi_seed_candidate(
            PROJECT_ROOT,
            [candidate],
            tmp_path / "selection",
        )
