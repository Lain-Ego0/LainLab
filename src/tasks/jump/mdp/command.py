"""Jump command: a phase clock that also tracks per-episode jump state.

The calibration in ``docs/opendoge_multiskill_targets.md`` shows the whole
push-off stroke is 34 mm and lasts about 26 ms, which is 2.6 control steps at
100 Hz. A policy cannot close a loop inside that window, so the takeoff has to
be pre-programmed: this term supplies the phase clock the policy schedules
against, plus the ``left_ground`` flag that lets rewards tell a real jump from
standing still.

The command block is 3 wide -- ``[phase_sin, phase_cos, left_ground]`` -- which
keeps the observation at the same 48 fields as the walking, handstand and
get-up tasks; only the *meaning* of the command block differs per skill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor


@dataclass(kw_only=True)
class JumpCommandCfg(CommandTermCfg):
  """Configuration for the jump phase clock."""

  entity_name: str = "robot"
  contact_sensor_name: str = "feet_ground_contact"
  period_s: float = 2.5
  """Length of one jump cycle in seconds."""
  standing_height: float = 0.151
  """Settled standing base height, used for the flight/recovery bookkeeping."""
  takeoff_margin: float = 0.015
  """Base height above standing that counts as a real takeoff rather than a wobble."""

  def build(self, env: ManagerBasedRlEnv) -> "JumpCommand":
    return JumpCommand(self, env)


class JumpCommand(CommandTerm):
  """Phase clock plus per-episode jump bookkeeping."""

  cfg: JumpCommandCfg

  def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self._command = torch.zeros(self.num_envs, 3, device=self.device)
    self.phase = torch.zeros(self.num_envs, device=self.device)
    self.peak_height = torch.zeros(self.num_envs, device=self.device)
    self.left_ground = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    # The robot is dropped onto the ground at reset, so "all feet in the air" is
    # briefly true before the first landing. Requiring one prior contact stops
    # that reset transient from being scored as a jump.
    self.grounded_once = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def airborne(self) -> torch.Tensor:
    """True where no foot is in contact, i.e. a real flight phase."""
    sensor: ContactSensor = self._env.scene[self.cfg.contact_sensor_name]
    found = sensor.data.found
    assert found is not None, "Jump needs a contact sensor that tracks 'found'."
    flags = (found > 0).to(torch.float32)
    if flags.dim() == 3:
      flags = flags.amax(dim=-1)
    return flags.amax(dim=-1) < 0.5

  def base_height(self) -> torch.Tensor:
    return self.robot.data.root_link_pos_w[:, 2] - self._env.scene.env_origins[:, 2]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.phase[env_ids] = 0.0
    self.peak_height[env_ids] = self.cfg.standing_height
    self.left_ground[env_ids] = False
    self.grounded_once[env_ids] = False

  def _update_metrics(self) -> None:
    """No scalar metrics: the rewards already log peak height and air time."""

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.phase[ids] = (self.phase[ids] + self._env.step_dt / self.cfg.period_s) % 1.0
    height = self.base_height()
    airborne = self.airborne()
    self.peak_height[ids] = torch.maximum(self.peak_height[ids], height[ids])
    took_off = (
      self.grounded_once[ids]
      & airborne[ids]
      & (height[ids] > self.cfg.standing_height + self.cfg.takeoff_margin)
    )
    self.left_ground[ids] |= took_off
    self.grounded_once[ids] |= ~airborne[ids]
    angle = 2.0 * math.pi * self.phase[ids]
    self._command[ids, 0] = torch.sin(angle)
    self._command[ids, 1] = torch.cos(angle)
    self._command[ids, 2] = self.left_ground[ids].to(torch.float32)
