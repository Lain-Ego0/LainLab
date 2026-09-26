"""Jump-task observation terms."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

from src.tasks.jump.mdp.command import JumpCommand


def jump_twist_command(
  env: ManagerBasedRlEnv,
  command_name: str = "jump",
) -> torch.Tensor:
  """The commanded ``[vx, vy, wz]`` the jump has to carry through the air.

  Kept as its own 3-wide block rather than folded into the ``command`` block,
  because that block is already full with the phase clock and the takeoff flag
  and is what keeps the unified observation's first 48 fields identical to this
  task's native observation.
  """
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, JumpCommand), f"{command_name!r} is not a JumpCommand"
  return term.twist
