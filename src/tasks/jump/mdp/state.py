"""Jump bookkeeping shared by the jump task and the unified skills task.

The jump needs per-environment state: a phase clock, horizontal twist command,
peak-height latch and four-feet-together takeoff detection. That state used to be
duplicated between ``JumpCommand`` (the jump task) and ``SkillCommandTerm`` (the
unified task), and the two silently drifted once -- the unified term put the
phase in the skill block while the jump task put it in the command block, which
stopped the jump expert taking off at all. It lives here now so there is one
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import sample_uniform


@dataclass(frozen=True)
class JumpStateCfg:
  """Parameters of the jump cycle, shared by both command terms."""

  period_s: float = 2.5
  standing_height: float = 0.151
  takeoff_margin: float = 0.015
  """Base height above standing that counts as a real takeoff, not a wobble."""
  twist_ranges: tuple[tuple[float, float], ...] = (
    (-0.3, 0.3),
    (-0.2, 0.2),
    (-0.4, 0.4),
  )
  """Commanded ``[vx, vy, wz]`` while jumping.

  Deliberately narrower than walking: leaving the ground while translating is
  strictly harder, and a small robot has little time in the air to correct.
  """
  spread_tolerance: float = 2.0
  """Takeoff spread (in control steps) at which the simultaneity reward halves.

  The whole push-off is only ~2.6 control steps, so all four feet leaving within
  two steps is already a well synchronised jump.
  """


class JumpState:
  """Per-environment jump cycle state: phase, twist, and takeoff bookkeeping."""

  def __init__(self, num_envs: int, device: str, cfg: JumpStateCfg) -> None:
    self.cfg = cfg
    self.num_envs = num_envs
    self.device = device
    self.phase = torch.zeros(num_envs, device=device)
    self.twist = torch.zeros(num_envs, 3, device=device)
    self.peak_height = torch.full((num_envs,), cfg.standing_height, device=device)
    self.left_ground = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.grounded_once = torch.zeros(num_envs, dtype=torch.bool, device=device)
    # Four-feet-together takeoff tracking.
    self._was_contact = torch.zeros(num_envs, 4, dtype=torch.bool, device=device)
    self._liftoff_step = torch.full((num_envs, 4), -1, dtype=torch.long, device=device)
    self.takeoff_spread = torch.zeros(num_envs, device=device)
    self.takeoff_complete = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self._takeoff_edge = torch.zeros(num_envs, dtype=torch.bool, device=device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.phase[env_ids] = 0.0
    self.twist[env_ids] = 0.0
    self.peak_height[env_ids] = self.cfg.standing_height
    self.left_ground[env_ids] = False
    self.grounded_once[env_ids] = False
    self._was_contact[env_ids] = False
    self._liftoff_step[env_ids] = -1
    self.takeoff_spread[env_ids] = 0.0
    self.takeoff_complete[env_ids] = False
    self._takeoff_edge[env_ids] = False

  def sample_twist(self, env_ids: torch.Tensor) -> None:
    """Sample a fresh horizontal twist command for these environments."""
    for axis, (low, high) in enumerate(self.cfg.twist_ranges):
      self.twist[env_ids, axis] = sample_uniform(
        low, high, (len(env_ids),), self.device
      )

  def update(
    self,
    env_ids: torch.Tensor | None,
    *,
    dt: float,
    base_height: torch.Tensor,
    airborne: torch.Tensor,
    feet_contact: torch.Tensor,
    step_index: torch.Tensor,
  ) -> None:
    """Advance the phase clock and all takeoff bookkeeping.

    ``feet_contact`` is ``[B, 4]`` and ``step_index`` is the per-environment step
    counter used to time the liftoff spread.
    """
    ids = (
      torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    )
    self.phase[ids] = (self.phase[ids] + dt / self.cfg.period_s) % 1.0
    self.peak_height[ids] = torch.maximum(self.peak_height[ids], base_height[ids])
    took_off = (
      self.grounded_once[ids]
      & airborne[ids]
      & (base_height[ids] > self.cfg.standing_height + self.cfg.takeoff_margin)
    )
    self.left_ground[ids] |= took_off
    self.grounded_once[ids] |= ~airborne[ids]

    self._update_takeoff(ids, feet_contact=feet_contact, step_index=step_index)

  def _update_takeoff(
    self,
    ids: torch.Tensor,
    *,
    feet_contact: torch.Tensor,
    step_index: torch.Tensor,
  ) -> None:
    """Record when each foot leaves the ground and score how together it was."""
    contact = feet_contact[ids]
    lifted = self._was_contact[ids] & ~contact
    self._liftoff_step[ids] = torch.where(
      lifted,
      step_index[ids].unsqueeze(-1).expand_as(self._liftoff_step[ids]),
      self._liftoff_step[ids],
    )
    self._was_contact[ids] = contact

    all_lifted = (self._liftoff_step[ids] >= 0).all(dim=-1)
    spread = (
      self._liftoff_step[ids].max(dim=-1).values
      - self._liftoff_step[ids].min(dim=-1).values
    )
    # Finalise once per takeoff: the first step at which all four are airborne.
    fresh = all_lifted & ~self.takeoff_complete[ids]
    self.takeoff_spread[ids] = torch.where(
      fresh, spread.to(self.takeoff_spread.dtype), self.takeoff_spread[ids]
    )
    self.takeoff_complete[ids] |= fresh
    self._takeoff_edge[ids] = fresh

    # Landing clears the record so the next jump is measured on its own.
    landed = contact.any(dim=-1)
    reset_here = landed & self.takeoff_complete[ids]
    self.takeoff_complete[ids] = torch.where(
      reset_here, torch.zeros_like(reset_here), self.takeoff_complete[ids]
    )
    self._liftoff_step[ids] = torch.where(
      reset_here.unsqueeze(-1),
      torch.full_like(self._liftoff_step[ids], -1),
      self._liftoff_step[ids],
    )

  @property
  def takeoff_edge(self) -> torch.Tensor:
    """True on the step a four-foot takeoff was just finalised."""
    return self._takeoff_edge

  def simultaneity(self) -> torch.Tensor:
    """``exp(-spread / tolerance)`` while a completed takeoff is in effect."""
    return torch.exp(-self.takeoff_spread / self.cfg.spread_tolerance)
