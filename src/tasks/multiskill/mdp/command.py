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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import sample_uniform

from src.tasks.jump.mdp.state import JumpState, JumpStateCfg

if TYPE_CHECKING:
  import viser

SKILL_NAMES = ("walk", "handstand", "getup", "jump")
VELOCITY_SKILLS = ("walk", "jump")
"""Skills with a viewer-driveable velocity.

Walking keeps it in the ``command`` block; the jump keeps it in the separate
``jump_twist`` block, because its command block is already full with the phase
clock and the takeoff flag.
"""


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
  jump_twist_ranges: tuple[tuple[float, float], ...] = (
    (-0.3, 0.3),
    (-0.2, 0.2),
    (-0.4, 0.4),
  )
  """Commanded ``[vx, vy, wz]`` carried through the jump; harvested from the jump task."""
  jump_spread_tolerance: float = 2.0
  """Takeoff spread (control steps) at which the jump simultaneity reward halves."""
  jump_flight_window: tuple[float, float] = (0.10, 0.22)
  """Phase slice that counts as the jump; must match the reward window."""
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
    # Jump bookkeeping lives in the shared JumpState so this term and the jump
    # task's JumpCommand cannot diverge (they did once, and the jump expert
    # silently stopped taking off inside this environment).
    self.jump = JumpState(
      self.num_envs,
      self.device,
      JumpStateCfg(
        period_s=cfg.jump_period_s,
        standing_height=cfg.standing_height,
        takeoff_margin=cfg.takeoff_margin,
        twist_ranges=cfg.jump_twist_ranges,
        spread_tolerance=cfg.jump_spread_tolerance,
        flight_window=cfg.jump_flight_window,
      ),
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
    # Viewer-only per-environment skill override (-1 = none). Set from the Viser
    # GUI so a demo can pick a skill and have it stick across resets; training
    # never touches it, so the sampled distribution is unchanged.
    self._skill_override = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    # Viewer-only joystick: (enable_checkbox, sliders, get_env_idx). Populated by
    # create_gui, None everywhere else, so training never reads it.
    self._joystick: tuple[Any, list[Any], Callable[[], int]] | None = None

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

  @property
  def phase(self) -> torch.Tensor:
    return self.jump.phase

  @property
  def peak_height(self) -> torch.Tensor:
    return self.jump.peak_height

  @property
  def left_ground(self) -> torch.Tensor:
    return self.jump.left_ground

  @property
  def twist(self) -> torch.Tensor:
    return self.jump.twist

  @property
  def takeoff_spread(self) -> torch.Tensor:
    return self.jump.takeoff_spread

  @property
  def takeoff_complete(self) -> torch.Tensor:
    return self.jump.takeoff_complete

  def simultaneity(self) -> torch.Tensor:
    return self.jump.simultaneity()

  def first_flight_gate(self) -> torch.Tensor:
    return self.jump.first_flight_gate()

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

  def set_skill_override(
    self, name: str | None, env_ids: torch.Tensor | None = None
  ) -> None:
    """Pin ``env_ids`` (all by default) to ``name``; ``None`` releases them.

    Unlike :meth:`force_skill`, an override survives environment resets, which is
    what makes the viewer's Reset button usable after picking a skill.
    """
    ids = (
      torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    )
    if name is None:
      self._skill_override[ids] = -1
      return
    self._skill_override[ids] = self.skill_index(name)
    self.force_skill(name, ids)

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, object], None] | None = None,
  ) -> None:
    """Viser controls: pick the skill for one environment, or for all of them.

    Applying a skill also requests a reset of those environments, so the chosen
    skill starts from its own reset distribution (standing for walk/handstand/
    jump, a random fall for get-up) instead of inheriting whatever pose the
    previous skill left behind. Because the override is sticky, the viewer's own
    Reset button keeps the selection.
    """
    from viser import Icon

    def _apply(all_envs: bool) -> None:
      ids = (
        None
        if all_envs
        else torch.tensor([get_env_idx()], dtype=torch.long, device=self.device)
      )
      self.set_skill_override(str(dropdown.value), ids)
      if request_action is not None:
        request_action("CUSTOM", {"type": "gui_reset", "all_envs": all_envs})
      if on_change is not None:
        on_change()

    with server.gui.add_folder(f"{name} (multi-skill)"):
      dropdown = server.gui.add_dropdown(
        "Skill",
        options=list(self.cfg.skill_names),
        initial_value=self.cfg.skill_names[0],
      )
      one_btn = server.gui.add_button("Apply to selected env")
      all_btn = server.gui.add_button("Apply to all envs")
      release_btn = server.gui.add_button("Release overrides", icon=Icon.SQUARE_X)

      @one_btn.on_click
      def _(_) -> None:
        _apply(all_envs=False)

      @all_btn.on_click
      def _(_) -> None:
        _apply(all_envs=True)

      @release_btn.on_click
      def _(_) -> None:
        self.set_skill_override(None)
        if on_change is not None:
          on_change()

    # Velocity joystick for the skills whose command block is a twist. Without
    # it the walk velocity is only ever sampled internally, so the viewer has no
    # way to drive the robot by hand.
    lin_x, lin_y, ang_z = self.cfg.walk_command_ranges
    with server.gui.add_folder(f"{name} velocity"):
      drive = server.gui.add_checkbox("Drive selected env", initial_value=False)
      sliders = [
        server.gui.add_slider(
          label,
          min=min(0.0, low),
          max=max(0.0, high),
          step=0.05,
          initial_value=0.0,
        )
        for label, (low, high) in (
          ("vx", lin_x),
          ("vy", lin_y),
          ("vw", ang_z),
        )
      ]
      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for slider in sliders:
          slider.value = 0.0

    self._joystick = (drive, sliders, get_env_idx)

  def set_switch_prob(self, value: float) -> None:
    self.cfg.switch_prob = float(value)

  def set_skill_weights(self, weights: tuple[float, ...]) -> None:
    tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
    self._weights = tensor / tensor.sum()

  # -- jump state, matching `JumpCommand` for reward reuse -------------------

  def feet_contact(self) -> torch.Tensor:
    """Per-foot contact flags, shape ``[B, 4]``."""
    sensor: ContactSensor = self._env.scene[self.cfg.contact_sensor_name]
    found = sensor.data.found
    assert found is not None, "Skills need a contact sensor that tracks 'found'."
    flags = (found > 0).to(torch.float32)
    if flags.dim() == 3:
      flags = flags.amax(dim=-1)
    return flags > 0.5

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
    sampled = torch.multinomial(weights, 1).squeeze(-1)
    override = self._skill_override[env_ids]
    self.skill[env_ids] = torch.where(override >= 0, override, sampled)

  def _reset_skill_state(self, env_ids: torch.Tensor) -> None:
    self.jump.reset(env_ids)
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
      # Resampled only here (reset / skill change), matching the jump task.
      self.jump.sample_twist(jump_ids)
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

    # Jump phase clock, twist and takeoff bookkeeping.
    jump_mask = self.skill_is("jump")
    jump_ids = ids[jump_mask[ids]]
    if len(jump_ids) > 0:
      self.jump.update(
        jump_ids,
        dt=self._dt,
        base_height=self.base_height(),
        airborne=self.airborne(),
        feet_contact=self.feet_contact(),
        step_index=self._env.episode_length_buf,
      )
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

    # The joystick writes last so it wins over both the sampled command and the
    # resampling timer; it only applies to the selected environment, and only
    # when that environment runs a skill whose command block is a twist.
    if env_ids is None and self._joystick is not None:
      drive, sliders, get_env_idx = self._joystick
      index = get_env_idx()
      if bool(drive.value) and 0 <= index < self.num_envs:
        skill = self.cfg.skill_names[int(self.skill[index])]
        values = torch.tensor(
          [float(slider.value) for slider in sliders],
          dtype=self._command.dtype,
          device=self.device,
        )
        if skill == "walk":
          self._command[index, :3] = values
        elif skill == "jump":
          self.jump.twist[index] = values
