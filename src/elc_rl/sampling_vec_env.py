"""Deterministic learner-side sampling updates for SB3 vector environments."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from .plant_sampling import StagePlantSampler


class PlantSamplingVecEnv(VecEnvWrapper):
    """Share difficulty across workers without shared-memory update races.

    SB3 resets a finished worker before returning step_wait(). That reset uses
    the preceding broadcast. New statistics are published after the full vector
    step, in worker-rank order, and affect subsequent resets. There is no extra
    reset, simulator call, or validation feedback in this wrapper.
    """

    def __init__(self, venv: VecEnv, sampler: StagePlantSampler, log_path: Path | None = None):
        super().__init__(venv)
        self.sampler = sampler
        self.log_path = None if log_path is None else Path(log_path)
        self._episodes: list[dict[str, Any] | None] = [None] * self.num_envs
        self._publish()

    def _publish(self) -> None:
        # None preserves np.random.Generator.choice's original uniform path,
        # including its RNG consumption, for A and for B's uniform warm-up.
        probabilities = self.sampler.probabilities() if self.sampler.adaptive else None
        self.venv.env_method("set_plant_sampling_probabilities", probabilities)

    def _start_episode(self, rank: int, info: Mapping[str, Any]) -> None:
        model_ids = info["sampled_model_ids"]
        if len(model_ids) != 1 or info["stage"] != self.sampler.stage:
            raise ValueError("sampling reset must identify one model in the current stage")
        model_id = str(model_ids[0])
        self.sampler.visits[self.sampler.index(model_id)] += 1
        self._episodes[rank] = {
            "model_id": model_id,
            "sampling_probability": float(info["sampled_model_probability"]),
            "steps": 0,
            "cost_sum": 0.0,
            "cost_max": 0.0,
            "margin_max": 0.0,
            "invalid": False,
        }

    def reset(self):
        observations = self.venv.reset()
        self.reset_infos = self.venv.reset_infos
        for rank, info in enumerate(self.reset_infos):
            self._start_episode(rank, info)
        return observations

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        self.reset_infos = self.venv.reset_infos
        events = []
        for rank, info in enumerate(infos):
            episode = self._episodes[rank]
            if episode is None:
                raise RuntimeError("sampling vector environment must be reset or restored")
            if tuple(info["sampled_model_ids"]) != (episode["model_id"],) or info["stage"] != self.sampler.stage:
                raise ValueError("sampling transition model or stage mismatch")
            cost, margin, invalid = self.sampler.step_scores(
                float(info["stage_cost"]),
                float(info["stage_margin_violation"]),
                bool(info["fast_valid"]),
            )
            episode["steps"] += 1
            episode["cost_sum"] += cost
            episode["cost_max"] = max(episode["cost_max"], cost)
            episode["margin_max"] = max(episode["margin_max"], margin)
            episode["invalid"] = bool(episode["invalid"] or invalid)
            self.sampler.transitions[self.sampler.index(episode["model_id"])] += 1
            if dones[rank]:
                event = self.sampler.finish_episode(episode["model_id"], episode)
                event["worker_rank"] = rank
                event["sampling_probability"] = episode["sampling_probability"]
                events.append(event)
                self._start_episode(rank, self.reset_infos[rank])
        if events:
            self._publish()
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as stream:
                    for event in events:
                        stream.write(json.dumps(event, allow_nan=False, sort_keys=True) + "\n")
        return observations, rewards, dones, infos

    def export_sampling_state(self) -> dict[str, Any]:
        if any(episode is None for episode in self._episodes):
            raise RuntimeError("cannot checkpoint an uninitialized sampling wrapper")
        return {
            "schema_version": 1,
            "n_envs": self.num_envs,
            "sampler": self.sampler.export_state(),
            "worker_episodes": copy.deepcopy(self._episodes),
        }

    def restore_sampling_state(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 1 or state.get("n_envs") != self.num_envs:
            raise ValueError("sampling wrapper schema or environment count mismatch")
        episodes = copy.deepcopy(state["worker_episodes"])
        if len(episodes) != self.num_envs:
            raise ValueError("sampling wrapper worker state count mismatch")
        for episode in episodes:
            self.sampler.index(episode["model_id"])
            if int(episode["steps"]) < 0 or not 0.0 < float(episode["sampling_probability"]) <= 1.0:
                raise ValueError("invalid sampling wrapper episode state")
            if not all(np.isfinite(episode[name]) and episode[name] >= 0 for name in ("cost_sum", "cost_max", "margin_max")):
                raise ValueError("invalid sampling wrapper difficulty state")
        self.sampler.restore_state(state["sampler"])
        self._episodes = episodes
        self._publish()
        # Remove events newer than the restored checkpoint, so a resumed trace
        # has one record per completed episode rather than duplicated updates.
        if self.log_path is not None and self.log_path.is_file():
            retained = []
            for line in self.log_path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A crash can leave an incomplete trailing line.
                if int(event["update"]) <= self.sampler.completed_episodes:
                    retained.append(json.dumps(event, allow_nan=False, sort_keys=True))
            temporary = self.log_path.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(line + "\n" for line in retained), encoding="utf-8")
            temporary.replace(self.log_path)

    def sampling_summary(self) -> dict[str, Any]:
        return {
            **self.sampler.summary(),
            "n_envs": self.num_envs,
            "worker_update_order": "ascending_rank_after_vector_step",
            "probability_update_timing": "after_auto_reset_affects_subsequent_resets",
            "worker_episodes": copy.deepcopy(self._episodes),
        }
