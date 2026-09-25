"""Terminations for the multi-skill task.

Two properties matter:

- body contact must end a walking / handstand / jump episode, but it is the
  whole point of get-up, so that termination is masked off for ``getup``
- the jump must not be terminated by leg contact during takeoff or landing,
  which is what killed the Go2 backflip run

Both are expressed by masking the single inherited contact termination.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp.terminations import illegal_contact

from src.tasks.skills.mdp.utils import skill_mask


def contact_termination(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
  allowed_skills: tuple[str, ...] = ("walk", "handstand", "jump"),
  command_name: str = "skill",
) -> torch.Tensor:
  """Illegal body contact, enabled only for the skills that must stay upright."""
  fired = illegal_contact(env, sensor_name=sensor_name, force_threshold=force_threshold)
  allowed = torch.zeros_like(fired)
  for skill in allowed_skills:
    allowed = allowed | skill_mask(env, skill, command_name).bool()
  # Must stay boolean: the termination manager ORs terms into a bool buffer.
  return fired & allowed
