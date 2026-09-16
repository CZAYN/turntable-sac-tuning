import copy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.logger import configure

from elc_rl.crossq import StageBatchRenorm, StageCrossQ, stage_ids
from elc_rl.sac_training import write_training_diagnostics
from elc_rl.sac_training import load_formal_training_config, _new_model, _effective_sac_parameters


class StageEnv(gym.Env):
    observation_space = gym.spaces.Dict({
        "x": gym.spaces.Box(-10, 10, (6,), np.float32),
        "stage": gym.spaces.Box(0, 1, (4,), np.float32),
    })
    action_space = gym.spaces.Box(-1, 1, (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self.obs(), {}

    def obs(self):
        stage = self.steps % 4
        return {"x": (self.np_random.normal(size=6) + stage).astype(np.float32),
                "stage": np.eye(4, dtype=np.float32)[stage]}

    def step(self, action):
        self.steps += 1
        return self.obs(), -float(np.square(action).sum()), self.steps == 19, False, {}


def small_model():
    torch.set_num_threads(1)
    model = StageCrossQ("MultiInputPolicy", StageEnv(), learning_starts=16,
                        buffer_size=256, batch_size=32, seed=31,
                        policy_kwargs={"net_arch": [16, 16], "brn_kwargs": {
                            "min_batch_size": 2, "warmup_steps": 1}})
    model.learn(48)
    return model


def test_brn_isolates_banks_preserves_order_and_uses_shared_affine():
    layer = StageBatchRenorm(3, momentum=1, warmup_steps=1, min_batch_size=2)
    values = torch.tensor([[1., 2., 3.], [101., 102., 103.], [3., 4., 5.], [103., 104., 105.]])
    ids = torch.tensor([0, 2, 0, 2])
    output = layer(values, ids)
    torch.testing.assert_close(layer.running_mean[0], torch.tensor([2., 3., 4.]))
    torch.testing.assert_close(layer.running_mean[2], torch.tensor([102., 103., 104.]))
    assert layer.num_batches_tracked.tolist() == [1, 0, 1, 0]
    torch.testing.assert_close(output[0], output[1])
    assert layer.weight.shape == (3,) and layer.bias.shape == (3,)
    before = copy.deepcopy(layer.state_dict())
    layer.eval()
    permutation = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(layer(values, ids)[permutation], layer(values[permutation], ids[permutation]))
    for key in before:
        torch.testing.assert_close(before[key], layer.state_dict()[key])


def test_brn_singleton_does_not_update_stats_and_has_finite_gradients():
    layer = StageBatchRenorm(2, warmup_steps=0, min_batch_size=2)
    values = torch.randn(5, 2, requires_grad=True)
    ids = torch.tensor([0, 0, 1, 2, 2])
    layer(values, ids).square().sum().backward()
    assert torch.isfinite(values.grad).all()
    assert layer.num_batches_tracked.tolist() == [1, 0, 1, 0]
    torch.testing.assert_close(layer.running_mean[1], torch.zeros(2))


def test_brn_correction_is_clipped_and_detached():
    layer = StageBatchRenorm(1, momentum=0.01, warmup_steps=0, min_batch_size=2)
    values = torch.tensor([[100.], [120.]], requires_grad=True)
    output = layer(values, torch.tensor([0, 0]))
    std = torch.sqrt(torch.tensor(100. + layer.eps))
    expected = (values - values.mean()) / std * 3 + 5
    torch.testing.assert_close(output, expected)
    # With detached r/d, the sum is invariant to a common input translation.
    output.sum().backward()
    torch.testing.assert_close(values.grad, torch.zeros_like(values))


def test_crossq_joint_forward_counts_actor_gradients_and_roundtrip(tmp_path):
    model = small_model()
    assert not hasattr(model, "critic_target") and not hasattr(model.policy, "critic_target")
    before_actor = copy.deepcopy(model.actor.state_dict())
    calls = []
    original_forward = model.critic.forward

    def record(obs, actions):
        calls.append((len(actions), model.critic.training))
        return original_forward(obs, actions)

    model.critic.forward = record
    model._n_updates = 0
    before_counts = model.critic.q_networks[0].layers[0][1].num_batches_tracked.clone()
    model.train(3, 32)
    model.critic.forward = original_forward
    assert calls == [(64, True), (64, True), (64, True), (32, False)]
    assert torch.all(model.critic.q_networks[0].layers[0][1].num_batches_tracked - before_counts == 3)
    assert not torch.equal(before_actor["mu.weight"], model.actor.mu.weight)
    assert all(p.requires_grad for p in model.critic.parameters())
    model.save(tmp_path / "model.zip")
    model.save_replay_buffer(tmp_path / "replay.pkl")
    loaded = StageCrossQ.load(tmp_path / "model.zip", env=StageEnv())
    loaded.load_replay_buffer(tmp_path / "replay.pkl")
    loaded.set_logger(configure(str(tmp_path), []))
    assert loaded.policy_delay == 3
    for key, value in model.policy.state_dict().items():
        torch.testing.assert_close(value, loaded.policy.state_dict()[key])
    obs, _ = StageEnv().reset(seed=4)
    np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], loaded.predict(obs, deterministic=True)[0])
    # Same replay and RNG -> same next optimizer updates, including all BRN banks.
    for learner in (model, loaded):
        np.random.seed(4)
        torch.manual_seed(4)
        learner.train(3, 32)
    for key, value in model.policy.state_dict().items():
        torch.testing.assert_close(value, loaded.policy.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(model.log_ent_coef, loaded.log_ent_coef)


def test_diagnostics_do_not_change_rng_statistics_or_parameters(tmp_path):
    model = small_model()
    before = copy.deepcopy(model.policy.state_dict())
    np.random.seed(12)
    torch.manual_seed(12)
    numpy_state, torch_state = np.random.get_state(), torch.get_rng_state()
    write_training_diagnostics(model, "joint", tmp_path)
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    torch.testing.assert_close(torch.get_rng_state(), torch_state)
    for key, value in before.items():
        torch.testing.assert_close(value, model.policy.state_dict()[key])
    report = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert len(report["replay_by_stage"]) == 4


def test_stage_validation_rejects_ambiguous_route():
    with pytest.raises(ValueError, match="one-hot"):
        stage_ids({"stage": torch.zeros(2, 4)})


def test_default_configuration_builds_current_crossq_method(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = load_formal_training_config(root)
    assert config.payload["algorithm"] == "crossq"
    assert config.payload["plant_sampling"]["mode"] == "difficulty"
    assert sum(stage.total_timesteps for stage in config.stages) == 260000
    model = _new_model(StageEnv(), 31, "cpu", None,
                       _effective_sac_parameters(config, "engineering_check", 1))
    assert isinstance(model, StageCrossQ)
    assert not hasattr(model, "critic_target")
    assert model.policy_delay == 3
    model.get_env().close()
    payload = copy.deepcopy(config.payload)
    payload["crossq"]["normalization"] = "batch_norm"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="stage_brn"):
        load_formal_training_config(root, path)
