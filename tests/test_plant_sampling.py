import copy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from elc_rl.plant_sampling import PlantSamplingConfig, StagePlantSampler, capped_mixture
from elc_rl.sampling_vec_env import PlantSamplingVecEnv
from elc_rl.sac_training import (
    TRAINING_PROTOCOL_SCHEMA_VERSION, _load_checkpoint, _save_resume_checkpoint,
    load_formal_training_config,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = tuple(f"training_{index:02d}" for index in range(40))


def _sampler(mode="difficulty", **kwargs):
    return StagePlantSampler(MODEL_IDS, "speed", PlantSamplingConfig.from_mapping({"mode": mode, **kwargs}))


def _observe(sampler, index, *, cost=0.0, margin=0.0, valid=True):
    cost_score, margin_score, invalid = sampler.step_scores(cost, margin, valid)
    sampler.visits[index] += 1
    sampler.transitions[index] += 1
    return sampler.finish_episode(MODEL_IDS[index], {
        "steps": 1, "cost_sum": cost_score, "cost_max": cost_score,
        "margin_max": margin_score, "invalid": invalid,
    })


def test_uniform_control_is_unaffected_by_recorded_difficulty():
    sampler = _sampler("uniform", warmup_episodes_per_model=0)
    for index in range(40):
        _observe(sampler, index, cost=index, valid=index > 3)
    assert np.array_equal(sampler.probabilities(), np.full(40, 1 / 40))


def test_uniform_warmup_then_harder_plants_receive_more_probability():
    sampler = _sampler(warmup_episodes_per_model=1)
    for index in range(39):
        _observe(sampler, index)
    assert not sampler.adaptive
    assert np.array_equal(sampler.probabilities(), np.full(40, 1 / 40))
    _observe(sampler, 39, cost=100, valid=False)
    probabilities = sampler.probabilities()
    assert sampler.adaptive
    assert probabilities[39] > probabilities[0]
    assert probabilities.max() <= 0.1 + 1e-12
    assert probabilities.min() >= 0.3 / 40 - 1e-12
    assert np.isclose(probabilities.sum(), 1.0)


def test_probability_floor_and_cap_hold_for_extreme_weights():
    probabilities = capped_mixture(np.array([1e100] + [1e-100] * 39), 0.3, 0.1)
    assert np.isfinite(probabilities).all()
    assert probabilities[0] == pytest.approx(0.1)
    assert probabilities.min() >= 0.3 / 40
    assert probabilities.sum() == pytest.approx(1.0)


def test_nonfinite_episode_is_bounded_and_foreign_models_are_rejected():
    sampler = _sampler(warmup_episodes_per_model=0)
    event = _observe(sampler, 0, cost=np.nan, margin=np.inf)
    assert event["episode_difficulty"] == 1.0
    assert sampler.invalid_episodes[0] == 1
    assert np.isfinite(sampler.probabilities()).all()
    with pytest.raises(ValueError, match="training model IDs only"):
        sampler.index("validation_00")
    with pytest.raises(ValueError, match="training model IDs only"):
        sampler.index("sealed_test_00")


def test_stage_and_seed_statistics_are_independent_and_restore_exactly():
    first = _sampler(warmup_episodes_per_model=0)
    independent_seed = _sampler(warmup_episodes_per_model=0)
    for index in range(40):
        _observe(first, index, cost=index, valid=index != 3)
    assert independent_seed.completed_episodes == 0
    state = json.loads(json.dumps(first.export_state()))
    restored = _sampler(warmup_episodes_per_model=0)
    restored.restore_state(state)
    assert np.array_equal(restored.probabilities(), first.probabilities())
    assert _observe(first, 3, cost=2) == _observe(restored, 3, cost=2)
    wrong_stage = StagePlantSampler(MODEL_IDS, "joint", first.config)
    with pytest.raises(ValueError, match="stage mismatch"):
        wrong_stage.restore_state(state)


@pytest.mark.parametrize("payload", [
    {"mode": "per"}, {"uniform_fraction": 0}, {"ema_rate": float("nan")},
    {"warmup_episodes_per_model": 0.5}, {"maximum_probability": 0}, {"typo": 1},
])
def test_invalid_sampler_config_is_rejected(payload):
    with pytest.raises(ValueError):
        PlantSamplingConfig.from_mapping(payload)


class _ToyPlantEnv(gym.Env):
    """Cheap episodes for testing SB3 auto-reset, ordering, and serialization."""

    def __init__(self):
        self.observation_space = gym.spaces.Box(0, 100, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.probabilities = None
        self.model = 0
        self.steps = 0
        self.probability = 1 / 40

    def set_plant_sampling_probabilities(self, probabilities):
        self.probabilities = None if probabilities is None else np.asarray(probabilities).copy()

    def _observation(self):
        return np.array([self.model, self.steps], dtype=np.float32)

    def _info(self):
        return {
            "stage": "speed", "sampled_model_ids": (MODEL_IDS[self.model],),
            "sampled_model_probability": self.probability,
            "stage_cost": float(self.model), "stage_margin_violation": 0.0,
            "fast_valid": True,
            # Deliberately hostile validation/audit fields must be ignored.
            "audit_valid": False, "validation_cost": 1e100,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.model = int(self.np_random.choice(40, p=self.probabilities))
        self.probability = 1 / 40 if self.probabilities is None else float(self.probabilities[self.model])
        self.steps = 0
        return self._observation(), self._info()

    def step(self, action):
        self.steps += 1
        return self._observation(), -float(self.model), False, self.steps >= 3, self._info()

    def export_state(self):
        return {
            "model": self.model, "steps": self.steps, "probability": self.probability,
            "probabilities": copy.deepcopy(self.probabilities),
            "rng": copy.deepcopy(self.np_random.bit_generator.state),
        }

    def restore_state(self, state):
        self.model, self.steps = state["model"], state["steps"]
        self.probability = state["probability"]
        self.probabilities = copy.deepcopy(state["probabilities"])
        self.np_random.bit_generator.state = copy.deepcopy(state["rng"])


def _vector(mode="difficulty", *, subprocess=False, log_path=None):
    factories = [_ToyPlantEnv, _ToyPlantEnv]
    base = SubprocVecEnv(factories, start_method="spawn") if subprocess else DummyVecEnv(factories)
    result = PlantSamplingVecEnv(base, _sampler(mode, warmup_episodes_per_model=0), log_path)
    result.seed(9876)
    return result


def test_uniform_wrapper_preserves_the_legacy_trajectory():
    legacy = DummyVecEnv([_ToyPlantEnv, _ToyPlantEnv])
    legacy.seed(9876)
    wrapped = _vector("uniform")
    try:
        assert np.array_equal(legacy.reset(), wrapped.reset())
        for _ in range(60):
            expected = legacy.step(np.zeros((2, 1)))
            actual = wrapped.step(np.zeros((2, 1)))
            for left, right in zip(expected[:3], actual[:3], strict=True):
                assert np.array_equal(left, right)
        assert wrapped.sampler.completed_episodes == 40
        assert not np.any(wrapped.sampler.invalid_episodes)
    finally:
        legacy.close()
        wrapped.close()


def test_vector_restore_preserves_partial_episodes_rng_and_trace(tmp_path):
    log_path = tmp_path / "episodes.jsonl"
    original = _vector(log_path=log_path)
    restored = _vector(log_path=log_path)
    try:
        original.reset()
        for _ in range(7):
            original.step(np.zeros((2, 1)))
        worker_states = original.env_method("export_state")
        sampling_state = original.export_sampling_state()
        expected = [original.step(np.zeros((2, 1))) for _ in range(8)]
        expected_sampler = original.sampler.export_state()
        assert len(log_path.read_text().splitlines()) == 10
        for rank, state in enumerate(worker_states):
            restored.env_method("restore_state", state, indices=rank)
        restored.restore_sampling_state(sampling_state)
        assert len(log_path.read_text().splitlines()) == 4
        for transition in expected:
            actual = restored.step(np.zeros((2, 1)))
            for left, right in zip(transition[:3], actual[:3], strict=True):
                assert np.array_equal(left, right)
        assert restored.sampler.export_state() == expected_sampler
        events = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert [event["update"] for event in events] == list(range(1, 11))
    finally:
        original.close()
        restored.close()


def test_spawned_workers_share_the_same_rank_ordered_statistics():
    local = _vector()
    parallel = _vector(subprocess=True)
    try:
        assert np.array_equal(local.reset(), parallel.reset())
        for _ in range(12):
            expected = local.step(np.zeros((2, 1)))
            actual = parallel.step(np.zeros((2, 1)))
            for left, right in zip(expected[:3], actual[:3], strict=True):
                assert np.array_equal(left, right)
        assert local.sampler.export_state() == parallel.sampler.export_state()
        broadcast = parallel.get_attr("probabilities")
        assert np.array_equal(broadcast[0], broadcast[1])
        assert not np.any(parallel.sampler.invalid_episodes)
    finally:
        local.close()
        parallel.close()


def test_sac_checkpoint_includes_sampler_and_resumes_without_reset(tmp_path):
    environment = _vector()
    resumed = _vector()
    try:
        model = SAC("MlpPolicy", environment, seed=123, learning_starts=100, buffer_size=64, device="cpu")
        model.learn(8)
        config = load_formal_training_config(ROOT)
        # This fixture exercises SAC compatibility explicitly; the project
        # default is CrossQ and must not label these SAC weights as CrossQ.
        payload = copy.deepcopy(config.payload)
        payload["algorithm"] = "sac"
        payload.pop("crossq")
        fixture_config = tmp_path / "sac_fixture.json"
        fixture_config.write_text(json.dumps(payload), encoding="utf-8")
        config = load_formal_training_config(ROOT, fixture_config)
        state = {
            "schema_version": TRAINING_PROTOCOL_SCHEMA_VERSION,
            "stage": "speed", "stage_index": 1, "stage_timesteps_completed": 8,
            "global_timesteps_completed": 8, "config_sha256": config.sha256,
            "input_fingerprint": "a" * 64, "latest_checkpoint": None,
        }
        checkpoint = _save_resume_checkpoint(model, tmp_path, state, config)
        assert (checkpoint / "plant_sampler_state.json").is_file()
        expected = [environment.step(np.zeros((2, 1))) for _ in range(5)]
        loaded = _load_checkpoint(tmp_path, state, resumed, "cpu", None, expect_replay_buffer=True)
        assert loaded.num_timesteps == 8
        for transition in expected:
            actual = resumed.step(np.zeros((2, 1)))
            for left, right in zip(transition[:3], actual[:3], strict=True):
                assert np.array_equal(left, right)
        assert resumed.sampler.export_state() == environment.sampler.export_state()
    finally:
        environment.close()
        resumed.close()
