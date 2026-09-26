"""Jump command: a phase clock, a twist command, and per-episode jump state.

The calibration in ``docs/opendoge_multiskill_targets.md`` shows the whole
push-off stroke is 34 mm and lasts about 26 ms, which is 2.6 control steps at
100 Hz. A policy cannot close a loop inside that window, so the takeoff has to
be pre-programmed: this term supplies the phase clock the policy schedules
against.

Two observation blocks come out of it:

- the ``command`` block stays ``[phase_sin, phase_cos, left_ground]``, which is
  what makes the unified task's first 48 fields identical to this task's own
  observation
- a ``jump_twist`` block carries the commanded ``[vx, vy, wz]`` the jump has to
  carry through the air

The per-environment jump state itself lives in
:class:`src.tasks.jump.mdp.state.JumpState`, shared with the unified multi-skill
task so the two cannot drift apart -- they already did once, and it silently
stopped the jump expert taking off inside the unified environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor

from src.tasks.jump.mdp.state import JumpState, JumpStateCfg


@dataclass(kw_only=True)
class JumpCommandCfg(CommandTermCfg):
  """Configuration for the jump phase clock and twist command."""

  entity_name: str = "robot"
  contact_sensor_name: str = "feet_ground_contact"
  period_s: float = 2.5
  """Length of one jump cycle in seconds."""
  standing_height: float = 0.151
  """Settled standing base height, used for the flight/recovery bookkeeping."""
  takeoff_margin: float = 0.015
  """Base height above standing that counts as a real takeoff rather than a wobble."""
  twist_ranges: tuple[tuple[float, float], ...] = field(
    default_factory=lambda: ((-0.3, 0.3), (-0.2, 0.2), (-0.4, 0.4))
  )
  """Commanded ``[vx, vy, wz]`` carried through the jump."""
  spread_tolerance: float = 2.0
  """Takeoff spread (control steps) at which the simultaneity reward halves."""
  flight_window: tuple[float, float] = (0.10, 0.22)
  """Phase slice that counts as the jump; must match the reward window."""

  def build(self, env: ManagerBasedRlEnv) -> "JumpCommand":
    return JumpCommand(self, env)


class JumpCommand(CommandTerm):
  """Phase clock plus per-episode jump bookkeeping."""

  cfg: JumpCommandCfg

  def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.state = JumpState(
      self.num_envs,
      self.device,
      JumpStateCfg(
        period_s=cfg.period_s,
        standing_height=cfg.standing_height,
        takeoff_margin=cfg.takeoff_margin,
        twist_ranges=cfg.twist_ranges,
        spread_tolerance=cfg.spread_tolerance,
        flight_window=cfg.flight_window,
      ),
    )
    self._command = torch.zeros(self.num_envs, 3, device=self.device)

  # -- jump state, proxied so reward terms can read it uniformly ------------

  @property
  def phase(self) -> torch.Tensor:
    return self.state.phase

  @property
  def peak_height(self) -> torch.Tensor:
    return self.state.peak_height

  @property
  def left_ground(self) -> torch.Tensor:
    return self.state.left_ground

  @property
  def twist(self) -> torch.Tensor:
    return self.state.twist

  @property
  def takeoff_spread(self) -> torch.Tensor:
    return self.state.takeoff_spread

  @property
  def takeoff_complete(self) -> torch.Tensor:
    return self.state.takeoff_complete

  def simultaneity(self) -> torch.Tensor:
    return self.state.simultaneity()

  def first_flight_gate(self) -> torch.Tensor:
    return self.state.first_flight_gate()

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def feet_contact(self) -> torch.Tensor:
    """Per-foot contact flags, shape ``[B, 4]``."""
    sensor: ContactSensor = self._env.scene[self.cfg.contact_sensor_name]
    found = sensor.data.found
    assert found is not None, "Jump needs a contact sensor that tracks 'found'."
    flags = (found > 0).to(torch.float32)
    if flags.dim() == 3:
      flags = flags.amax(dim=-1)
    return flags > 0.5

  def airborne(self) -> torch.Tensor:
    """True where no foot is in contact, i.e. a real flight phase."""
    return ~self.feet_contact().any(dim=-1)

  def base_height(self) -> torch.Tensor:
    return self.robot.data.root_link_pos_w[:, 2] - self._env.scene.env_origins[:, 2]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.state.reset(env_ids)
    self.state.sample_twist(env_ids)

  def _update_metrics(self) -> None:
    """No scalar metrics: the rewards already log peak height and air time."""

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    self.state.update(
      env_ids,
      dt=self._env.step_dt,
      base_height=self.base_height(),
      airborne=self.airborne(),
      feet_contact=self.feet_contact(),
      step_index=self._env.episode_length_buf,
    )
    ids = (
      torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    )
    angle = 2.0 * math.pi * self.state.phase[ids]
    self._command[ids, 0] = torch.sin(angle)
    self._command[ids, 1] = torch.cos(angle)
    self._command[ids, 2] = self.state.left_ground[ids].to(torch.float32)
