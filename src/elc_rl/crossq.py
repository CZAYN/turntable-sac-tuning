"""Compact CrossQ for dictionary observations with stage-conditioned BRN.

Algorithm reference: https://aditya.bhatts.org/CrossQ/ (ICLR 2024).
This project adaptation keeps the 256 x 256 networks and SAC hyperparameters;
it is not a reproduction of the paper's wide-network benchmark configuration.
SB3 supplies rollout, replay, entropy distribution and checkpoint infrastructure.
"""
from __future__ import annotations

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.sac.policies import Actor, SACPolicy, LOG_STD_MIN, LOG_STD_MAX


class StageBatchRenorm(nn.Module):
    """Shared affine parameters; four independent running-statistic banks.

    A bank is updated only by a sufficiently large group from that stage.
    Warm-up counts qualifying updates per bank, not global training steps.
    Small groups and inference use running statistics without changing them.
    """

    def __init__(self, features: int, momentum: float = 0.01,
                 warmup_steps: int = 1000, min_batch_size: int = 8,
                 eps: float = 1e-5, rmax: float = 3.0, dmax: float = 5.0):
        super().__init__()
        if not 0 < momentum <= 1 or warmup_steps < 0 or min_batch_size < 2:
            raise ValueError("invalid stage BRN settings")
        self.momentum, self.warmup_steps = momentum, warmup_steps
        self.min_batch_size, self.eps = min_batch_size, eps
        self.rmax, self.dmax = rmax, dmax
        self.weight = nn.Parameter(th.ones(features))
        self.bias = nn.Parameter(th.zeros(features))
        self.register_buffer("running_mean", th.zeros(4, features))
        self.register_buffer("running_var", th.ones(4, features))
        self.register_buffer("num_batches_tracked", th.zeros(4, dtype=th.long))

    def forward(self, x: th.Tensor, stage_ids: th.Tensor) -> th.Tensor:
        result = th.empty_like(x)
        for stage in range(4):
            indices = th.where(stage_ids == stage)[0]
            if indices.numel() == 0:
                continue
            values = x[indices]
            # Clone: updating buffers below must not mutate autograd's inputs.
            mean_r = self.running_mean[stage].detach().clone()
            var_r = self.running_var[stage].detach().clone()
            std_r = th.sqrt(var_r + self.eps)
            if self.training and indices.numel() >= self.min_batch_size:
                var_b, mean_b = th.var_mean(values, dim=0, unbiased=False)
                std_b = th.sqrt(var_b + self.eps)
                normalized = (values - mean_b) / std_b
                if self.num_batches_tracked[stage] >= self.warmup_steps:
                    r = (std_b / std_r).detach().clamp(1 / self.rmax, self.rmax)
                    d = ((mean_b - mean_r) / std_r).detach().clamp(-self.dmax, self.dmax)
                    normalized = normalized * r + d
                with th.no_grad():
                    self.running_mean[stage].lerp_(mean_b.detach(), self.momentum)
                    # Store unbiased variance, as in BatchNorm running statistics.
                    unbiased_var = var_b.detach() * (indices.numel() / (indices.numel() - 1))
                    self.running_var[stage].lerp_(unbiased_var, self.momentum)
                    self.num_batches_tracked[stage] += 1
            else:
                normalized = (values - mean_r) / std_r
            result[indices] = normalized * self.weight + self.bias
        return result


def stage_ids(observations: dict[str, th.Tensor]) -> th.Tensor:
    stage = observations["stage"]
    if stage.ndim != 2 or stage.shape[1] != 4:
        raise ValueError("CrossQ requires a four-dimensional stage one-hot observation")
    if not th.all((stage == 0) | (stage == 1)) or not th.all(stage.sum(dim=1) == 1):
        raise ValueError("invalid stage one-hot observation")
    return stage.argmax(dim=1)


class StageMLP(nn.Module):
    def __init__(self, input_dim, widths, activation_fn, brn_kwargs, output_dim=None):
        super().__init__()
        self.layers = nn.ModuleList()
        for width in widths:
            self.layers.append(nn.ModuleList([
                nn.Linear(input_dim, width), StageBatchRenorm(width, **brn_kwargs), activation_fn(),
            ]))
            input_dim = width
        self.output = nn.Identity() if output_dim is None else nn.Linear(input_dim, output_dim)

    def forward(self, values, ids):
        for linear, norm, activation in self.layers:
            values = activation(norm(linear(values), ids))
        return self.output(values)


class StageActor(Actor):
    def __init__(self, *args, brn_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_sde:
            raise ValueError("StageCrossQ supports the existing squashed Gaussian actor only")
        self.latent_pi = StageMLP(self.features_dim, self.net_arch, self.activation_fn, brn_kwargs or {})

    def get_action_dist_params(self, obs):
        latent = self.latent_pi(self.extract_features(obs, self.features_extractor), stage_ids(obs))
        return self.mu(latent), self.log_std(latent).clamp(LOG_STD_MIN, LOG_STD_MAX), {}


class StageCritic(ContinuousCritic):
    def __init__(self, *args, brn_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the standard Q MLPs, retaining independent parameters for Q1/Q2.
        self.q_networks = []
        input_dim = kwargs["features_dim"] + int(np.prod(self.action_space.shape))
        for index in range(self.n_critics):
            network = StageMLP(input_dim, kwargs["net_arch"], kwargs["activation_fn"], brn_kwargs or {}, 1)
            self.add_module(f"qf{index}", network)
            self.q_networks.append(network)

    def forward(self, obs, actions):
        features = self.extract_features(obs, self.features_extractor)
        inputs = th.cat((features, actions), dim=1)
        ids = stage_ids(obs)
        return tuple(network(inputs, ids) for network in self.q_networks)

    def q1_forward(self, obs, actions):
        features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](th.cat((features, actions), dim=1), stage_ids(obs))


class StageCrossQPolicy(SACPolicy):
    def __init__(self, *args, brn_kwargs=None, **kwargs):
        self.brn_kwargs = dict(brn_kwargs or {})
        kwargs.setdefault("features_extractor_class", CombinedExtractor)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        self.actor = self.make_actor()
        self.critic = self.make_critic()
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        # Deliberately no target critic.

    def make_actor(self, features_extractor=None):
        kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return StageActor(**kwargs, brn_kwargs=self.brn_kwargs).to(self.device)

    def make_critic(self, features_extractor=None):
        kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return StageCritic(**kwargs, brn_kwargs=self.brn_kwargs).to(self.device)

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), "brn_kwargs": self.brn_kwargs}


class StageCrossQ(SAC):
    """SAC-compatible infrastructure with a target-free CrossQ update rule."""
    policy_aliases = {"MultiInputPolicy": StageCrossQPolicy}

    def __init__(self, *args, policy_delay: int = 3, **kwargs):
        if policy_delay < 1:
            raise ValueError("policy_delay must be positive")
        self.policy_delay = policy_delay
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        # Skip SAC's setup because it assumes a target network exists.
        OffPolicyAlgorithm._setup_model(self)
        self.actor, self.critic = self.policy.actor, self.policy.critic
        self.target_entropy = (-float(np.prod(self.action_space.shape))
                               if self.target_entropy == "auto" else float(self.target_entropy))
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            initial = float(self.ent_coef.split("_", 1)[1]) if "_" in self.ent_coef else 1.0
            if initial <= 0:
                raise ValueError("initial entropy coefficient must be positive")
            self.log_ent_coef = th.tensor(np.log(initial), device=self.device, dtype=th.float32).reshape(1).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def train(self, gradient_steps: int, batch_size: int = 64):
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)
        losses, actor_losses, td_errors, q_means, ent_losses = [], [], [], [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            alpha = self.log_ent_coef.detach().exp() if self.log_ent_coef is not None else self.ent_coef_tensor
            self.actor.set_training_mode(False)
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay.next_observations)
            joined_obs = {key: th.cat((replay.observations[key], replay.next_observations[key]), dim=0)
                          for key in replay.observations}
            joined_actions = th.cat((replay.actions, next_actions), dim=0)
            self.critic.set_training_mode(True)
            joined_q = th.cat(self.critic(joined_obs, joined_actions), dim=1)
            self.critic.set_training_mode(False)
            current_q, next_q = joined_q.split(batch_size, dim=0)
            discounts = replay.discounts if replay.discounts is not None else self.gamma
            with th.no_grad():
                target = replay.rewards + (1 - replay.dones) * discounts * (
                    next_q.min(dim=1, keepdim=True).values - alpha * next_log_prob.reshape(-1, 1))
            loss = 0.5 * sum(F.mse_loss(q, target) for q in current_q.split(1, dim=1))
            if not th.isfinite(loss):
                raise FloatingPointError("non-finite CrossQ critic loss")
            self.critic.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.critic.optimizer.step()
            losses.append(loss.item())
            td_errors.append((current_q.detach() - target).abs().mean().item())
            q_means.append(current_q.detach().mean().item())

            if self._n_updates % self.policy_delay == 0:
                self.actor.set_training_mode(True)
                actions, log_prob = self.actor.action_log_prob(replay.observations)
                self.actor.set_training_mode(False)
                log_prob = log_prob.reshape(-1, 1)
                # Keep gradients through actions but not critic parameters/stats.
                self.critic.requires_grad_(False)
                try:
                    q_pi = th.cat(self.critic(replay.observations, actions), dim=1).min(dim=1, keepdim=True).values
                    actor_loss = (alpha * log_prob - q_pi).mean()
                    if not th.isfinite(actor_loss):
                        raise FloatingPointError("non-finite CrossQ actor loss")
                    self.actor.optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    self.actor.optimizer.step()
                finally:
                    self.critic.requires_grad_(True)
                actor_losses.append(actor_loss.item())
                if self.ent_coef_optimizer is not None:
                    ent_loss = -(self.log_ent_coef * (log_prob.detach() + self.target_entropy)).mean()
                    self.ent_coef_optimizer.zero_grad(set_to_none=True)
                    ent_loss.backward()
                    self.ent_coef_optimizer.step()
                    ent_losses.append(ent_loss.item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for key, values in (("critic_loss", losses), ("actor_loss", actor_losses),
                            ("td_error_abs", td_errors), ("q_mean", q_means), ("ent_coef_loss", ent_losses)):
            if values:
                self.logger.record(f"train/{key}", float(np.mean(values)))
        if gradient_steps:
            self.logger.record("train/ent_coef", alpha.item())
