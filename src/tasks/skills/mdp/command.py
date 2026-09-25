"""Multi-skill command term: one command block, one skill identity per env.

This is what turns four separate tasks into one policy. Each environment is
assigned a skill; the term then

- exposes a ``skill`` index plus a phase clock as an observation block, and
- fills the 3-wide ``command`` block with whatever that skill means by a command
  (velocity for walking, ``[left_ground]`` for jumping, zeros for the stationary
  skills),

so the observation layout stays exactly the 48 fields the single-skill tasks use.

It also carries the jump bookkeeping (``airborne``, ``peak_height``,
``left_ground``) so the jump rewards can be reused verbatim, and it can switch an
environment's skill mid-episode for transition training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import sample_uniform

SKILL_NAMES = ("walk", "handstand", "getup", "jump")


@dataclass(kw_only=True)
class SkillCommandCfg(CommandTermCfg):
  """Configuration for the multi-skill command term."""

  entity_name: str = "robot"
  contact_sensor_name: str = "feet_ground_contact"
  skill_names: tuple[str, ...] = SKILL_NAMES
  walk_command_ranges: tuple[tuple[float, float], ...] = (
    (-0.8, 0.8),
    (-0.5, 0.5),
    (-0.8, 0.8),
  )
  """Velocity command ranges for the ``walk`` skill, from the flat profile."""
  walk_resample_range: tuple[float, float] = (3.0, 8.0)
  standing_fraction: float = 0.1
  """Fraction of walking envs given a zero command, matching the velocity task."""
  jump_period_s: float = 2.5
  standing_height: float = 0.151
  takeoff_margin: float = 0.015
  switch_prob: float = 0.0
  """Per-step probability of switching skill mid-episode (transition training)."""
  skill_weights: tuple[float, ...] = field(default_factory=tuple)
  """Optional sampling weights per skill; empty means uniform."""

  def build(self, env: ManagerBasedRlEnv) -> "SkillCommandTerm":
    return SkillCommandTerm(self, env)


class SkillCommandTerm(CommandTerm):
  """Assigns a skill per environment and produces its command block."""

  cfg: SkillCommandCfg

  def __init__(self, cfg: SkillCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.skill = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._command = torch.zeros(self.num_envs, 3, device=self.device)
    self.phase = torch.zeros(self.num_envs, device=self.device)
    self.peak_height = torch.zeros(self.num_envs, device=self.device)
    self.left_ground = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.grounded_once = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._walk_time_left = torch.zeros(self.num_envs, device=self.device)
    # The reset event runs *before* the command manager (see
    # `ManagerBasedRlEnv._reset_idx`), and it needs the new skill to pick the
    # reset distribution. It therefore samples first and marks the ids here so
    # the command manager does not sample a second, different skill.
    self._sampled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._dt = float(env.step_dt)
    weights = cfg.skill_weights or tuple(1.0 for _ in cfg.skill_names)
    if len(weights) != len(cfg.skill_names):
      raise ValueError("skill_weights must match skill_names in length")
    self._weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
    self._weights = self._weights / self._weights.sum()

  # -- public helpers -------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    return self._command

  @property
  def skill_names(self) -> tuple[str, ...]:
    return self.cfg.skill_names

  def skill_index(self, name: str) -> int:
    return self.cfg.skill_names.index(name)

  def skill_is(self, name: str) -> torch.Tensor:
    """Boolean mask of the environments currently running ``name``."""
    return self.skill == self.skill_index(name)

  def skill_phase(self) -> torch.Tensor:
    """``[sin, cos]`` of the phase clock, shape ``[B, 2]``."""
    angle = 2.0 * math.pi * self.phase
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)

  def force_skill(self, name: str, env_ids: torch.Tensor | None = None) -> None:
    """Force ``env_ids`` (all by default) onto one skill immediately.

    Used by evaluation to script a transition sequence without going through the
    random sampler.
    """
    ids = (
      torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    )
    self.skill[ids] = self.skill_index(name)
    self._reset_skill_state(ids)
    self._refresh_commands(ids)

  def set_switch_prob(self, value: float) -> None:
    self.cfg.switch_prob = float(value)

  def set_skill_weights(self, weights: tuple[float, ...]) -> None:
    tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
    self._weights = tensor / tensor.sum()

  # -- jump state, matching `JumpCommand` for reward reuse -------------------

  def airborne(self) -> torch.Tensor:
    sensor: ContactSensor = self._env.scene[self.cfg.contact_sensor_name]
    found = sensor.data.found
    assert found is not None, "Skills need a contact sensor that tracks 'found'."
    flags = (found > 0).to(torch.float32)
    if flags.dim() == 3:
      flags = flags.amax(dim=-1)
    return flags.amax(dim=-1) < 0.5

  def base_height(self) -> torch.Tensor:
    return self.robot.data.root_link_pos_w[:, 2] - self._env.scene.env_origins[:, 2]

  # -- CommandTerm hooks ----------------------------------------------------

  def compute(
    self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    # `_update_command` needs dt to advance timers; the base signature drops it,
    # and the reset path calls compute with dt=0, so stash the real value.
    self._dt = float(dt) if not isinstance(dt, torch.Tensor) else float(dt.mean())
    super().compute(dt, env_ids)

  def sample_skills(self, env_ids: torch.Tensor) -> None:
    """Sample a skill for ``env_ids`` and reset its per-episode state.

    Called by the reset event so the reset distribution can follow the skill.
    """
    self._assign_skills(env_ids)
    self._reset_skill_state(env_ids)
    self._refresh_commands(env_ids)
    self._sampled[env_ids] = True

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    pending = env_ids[~self._sampled[env_ids]]
    if len(pending) > 0:
      self._assign_skills(pending)
      self._reset_skill_state(pending)
      self._refresh_commands(pending)
    self._sampled[env_ids] = False

  def _assign_skills(self, env_ids: torch.Tensor) -> None:
    weights = self._weights.expand(len(env_ids), -1)
    self.skill[env_ids] = torch.multinomial(weights, 1).squeeze(-1)

  def _reset_skill_state(self, env_ids: torch.Tensor) -> None:
    self.phase[env_ids] = 0.0
    self.peak_height[env_ids] = self.cfg.standing_height
    self.left_ground[env_ids] = False
    self.grounded_once[env_ids] = False
    self._command[env_ids] = 0.0

  def _refresh_commands(self, env_ids: torch.Tensor) -> None:
    """Write the command block implied by each environment's skill.

    Only ``walk`` gets a velocity command and only ``jump`` carries the takeoff
    flag; the stationary skills must see exactly zeros. Writing a velocity
    command for every environment -- as an earlier version did -- silently
    corrupted the handstand and get-up observations, because their experts were
    trained with a zero command block.
    """
    self._command[env_ids] = 0.0
    walk_ids = env_ids[self.skill_is("walk")[env_ids]]
    if len(walk_ids) > 0:
      self._sample_walk_velocity(walk_ids)
    jump_ids = env_ids[self.skill_is("jump")[env_ids]]
    if len(jump_ids) > 0:
      # The jump task's command block *is* the phase clock, so reproduce it here
      # rather than only in the skill block.
      phase = self.skill_phase()[jump_ids]
      self._command[jump_ids, 0] = phase[:, 0]
      self._command[jump_ids, 1] = phase[:, 1]
      self._command[jump_ids, 2] = self.left_ground[jump_ids].to(torch.float32)

  def _sample_walk_velocity(self, env_ids: torch.Tensor) -> None:
    """Sample a velocity command for walking environments only."""
    self._walk_time_left[env_ids] = sample_uniform(
      *self.cfg.walk_resample_range, (len(env_ids),), self.device
    )
    lin_x, lin_y, ang_z = self.cfg.walk_command_ranges
    self._command[env_ids, 0] = sample_uniform(*lin_x, (len(env_ids),), self.device)
    self._command[env_ids, 1] = sample_uniform(*lin_y, (len(env_ids),), self.device)
    self._command[env_ids, 2] = sample_uniform(*ang_z, (len(env_ids),), self.device)
    standing = torch.rand(len(env_ids), device=self.device) < self.cfg.standing_fraction
    self._command[env_ids[standing]] = 0.0

  def _update_metrics(self) -> None:
    """Per-skill population is logged from the observations side."""

  def _maybe_switch_skill(self, env_ids: torch.Tensor) -> None:
    if self.cfg.switch_prob <= 0.0 or len(env_ids) == 0:
      return
    draw = torch.rand(len(env_ids), device=self.device)
    switch = env_ids[draw < self.cfg.switch_prob]
    if len(switch) == 0:
      return
    self._assign_skills(switch)
    self._reset_skill_state(switch)
    self._refresh_commands(switch)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    ids = (
      torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    )
    self._maybe_switch_skill(ids)

    # Jump phase clock and takeoff bookkeeping.
    jump_mask = self.skill_is("jump")
    jump_ids = ids[jump_mask[ids]]
    if len(jump_ids) > 0:
      self.phase[jump_ids] = (
        self.phase[jump_ids] + self._dt / self.cfg.jump_period_s
      ) % 1.0
      height = self.base_height()
      airborne = self.airborne()
      self.peak_height[jump_ids] = torch.maximum(
        self.peak_height[jump_ids], height[jump_ids]
      )
      took_off = (
        self.grounded_once[jump_ids]
        & airborne[jump_ids]
        & (height[jump_ids] > self.cfg.standing_height + self.cfg.takeoff_margin)
      )
      self.left_ground[jump_ids] |= took_off
      self.grounded_once[jump_ids] |= ~airborne[jump_ids]
      phase = self.skill_phase()[jump_ids]
      self._command[jump_ids, 0] = phase[:, 0]
      self._command[jump_ids, 1] = phase[:, 1]
      self._command[jump_ids, 2] = self.left_ground[jump_ids].to(torch.float32)

    # Walking velocity command resampling, matching the velocity task cadence.
    walk_ids = ids[self.skill_is("walk")[ids]]
    if len(walk_ids) > 0:
      self._walk_time_left[walk_ids] -= self._dt
      due = walk_ids[self._walk_time_left[walk_ids] <= 0.0]
      if len(due) > 0:
        self._sample_walk_velocity(due)
