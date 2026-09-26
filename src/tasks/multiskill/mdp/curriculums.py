"""Curricula for the multi-skill task.

The only curriculum here is the transition schedule: skills start out fixed per
episode and progressively become allowed to change mid-episode. Training
transitions from the beginning produces policies that hold each skill but fall
apart when the skill changes; leaving them out entirely produces policies that
never learn to switch at all.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

from src.tasks.multiskill.mdp.utils import skill_term


def skill_switch_probability(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: tuple[tuple[int, float], ...] = (),
) -> dict[str, torch.Tensor]:
  """Ramp the per-step probability of switching skill mid-episode.

  ``stages`` are ``(common_step_counter, probability)`` pairs; the probability
  of the last stage whose step threshold has been passed is applied. Stages
  count environment steps, and PPO advances ``common_step_counter`` by
  ``num_steps_per_env`` (24 by default) per iteration.

  The probability is per environment step, not per episode: at 100 Hz a value of
  0.0015 already means a switch roughly every 6.7 s. Anything near 0.05 switches
  faster than any skill can settle.
  """
  del env_ids  # Applied globally.
  term = skill_term(env, command_name)
  probability = 0.0
  for step, value in stages:
    if env.common_step_counter >= step:
      probability = value
  term.set_switch_prob(probability)
  return {"skill_switch_prob": torch.tensor(probability, device=env.device)}
