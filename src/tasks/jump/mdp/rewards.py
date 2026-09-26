"""Reward terms for the in-place vertical jump task.

The shaping is built so that standing still is unprofitable:

- ``flight`` pays only while *all four* feet are off the ground, so tiptoeing or
  a handstand-style two-foot pose earns nothing
- ``apex_height`` peaks at ``standing + target_rise``, which is above what leg
  extension alone can reach (34 mm of stroke), so the policy must break contact
- ``twist_tracking`` pays for carrying the commanded horizontal velocity through
  the air and after landing, so the jump translates instead of hopping in place
- ``takeoff_simultaneity`` pays when all four feet leave the ground together
- ``failure`` punishes an episode that times out without ever leaving the ground

State (peak height, ``left_ground``) lives in ``JumpCommand``.
"""

from __future__ import annotations

from typing import cast

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from src.tasks.jump.mdp.command import JumpCommand

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_GRAVITY_UP = (0.0, 0.0, -1.0)


def _jump_term(env: ManagerBasedRlEnv, command_name: str) -> JumpCommand:
  """Fetch the jump state holder.

  Duck-typed rather than ``isinstance``-checked so the multi-skill command term
  -- which exposes the same jump bookkeeping for its ``jump`` environments -- can
  host these rewards without the skills package importing this module's type.
  """
  term = env.command_manager.get_term(command_name)
  for attribute in ("airborne", "base_height", "peak_height", "left_ground"):
    assert hasattr(term, attribute), (
      f"Command term {command_name!r} does not expose jump state ({attribute})"
    )
  return cast(JumpCommand, term)


def _contact_flags(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None, f"Sensor {sensor_name!r} does not track 'found'."
  flags = (found > 0).to(torch.float32)
  if flags.dim() == 3:
    flags = flags.amax(dim=-1)
  return flags


FLIGHT_WINDOW = (0.10, 0.22)
"""Slice of the jump cycle during which flight is rewarded.

Deliberately only 0.30 s. The window gates *when* flight pays, not *how often*,
so a long window is filled with as many short bounces as fit in it -- a 0.75 s
window produced 5.4 bounces per cycle (2.2 per second, each ~0.14 s) instead of
one jump. One flight needs 0.14-0.23 s, so a 0.30 s window admits exactly one.
"""


def ground_travel(
  env: ManagerBasedRlEnv,
  command_name: str = "jump",
  airborne_window: tuple[float, float] = (0.05, 0.30),
) -> torch.Tensor:
  """Reward staying on the ground and tracking the twist outside the jump.

  Without this the only thing separating "jump" from "travel" is the reward
  window, and bounce-farming is re-discovered whenever the window is loosened.
  Compatible with a moving jump: it does not require all four feet down, only
  that the robot is not airborne, so any gait satisfies it.
  """
  term = _jump_term(env, command_name)
  on_ground = ~term.airborne()
  in_window = phase_window(term.phase, *airborne_window)
  return on_ground.float() * (1.0 - in_window) * twist_tracking(env, command_name)


def phase_window(phase: torch.Tensor, start: float, end: float) -> torch.Tensor:
  """Float mask for the slice of the jump cycle inside ``[start, end)``."""
  return ((phase >= start) & (phase < end)).float()


def flight(
  env: ManagerBasedRlEnv,
  command_name: str = "jump",
  window: tuple[float, float] = FLIGHT_WINDOW,
) -> torch.Tensor:
  """Pay while every foot is off the ground, but only inside the flight window.

  The window is what makes this a *jump* rather than a bouncer: air time is
  capped at the window length per cycle, so a policy that hops continuously
  earns no more than one that jumps once and then recovers.
  """
  term = _jump_term(env, command_name)
  airborne = term.airborne().float()
  env.extras.setdefault("log", {})["Metrics/jump_airborne_frac"] = airborne.mean()
  return airborne * phase_window(term.phase, *window) * term.first_flight_gate()


def apex_height(
  env: ManagerBasedRlEnv,
  command_name: str = "jump",
  standing_height: float = 0.151,
  target_rise: float = 0.05,
  tolerance: float = 0.035,
  window: tuple[float, float] = FLIGHT_WINDOW,
) -> torch.Tensor:
  """Reward the base being at ``standing_height + target_rise``.

  Uses the *current* height rather than the running peak. A peak-based reward
  would keep paying for the rest of the episode after one lucky hop, and the
  tolerance is kept below ``target_rise`` so standing still (which sits ~7 mm
  above the settled height at reset) scores exactly zero. It is also confined to
  the flight window, otherwise rearing back on the hind legs would collect it.
  """
  term = _jump_term(env, command_name)
  error = (term.base_height() - (standing_height + target_rise)) / tolerance
  env.extras.setdefault("log", {})["Metrics/jump_peak_rise"] = (
    term.peak_height - standing_height
  ).mean()
  height_reward = torch.clamp(1.0 - torch.square(error), min=0.0)
  return height_reward * phase_window(term.phase, *window) * term.first_flight_gate()


def upright(
  env: ManagerBasedRlEnv,
  target: tuple[float, float, float] = _DEFAULT_GRAVITY_UP,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward keeping the body level: a vertical jump must not pitch or roll."""
  asset: Entity = env.scene[asset_cfg.name]
  target_tensor = torch.tensor(target, device=env.device)
  return torch.clamp(asset.data.projected_gravity_b @ target_tensor, -1.0, 1.0)


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize slamming down: downward speed weighted by ground contact."""
  asset: Entity = env.scene[asset_cfg.name]
  downward = torch.clamp(-asset.data.root_link_lin_vel_w[:, 2], min=0.0)
  any_contact = (_contact_flags(env, sensor_name).amax(dim=1) > 0.5).float()
  return downward * any_contact


def twist_tracking(
  env: ManagerBasedRlEnv,
  command_name: str = "jump",
  std_linear: float = 0.35,
  std_angular: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward carrying the commanded horizontal velocity through the jump.

  Active over the whole cycle on purpose: the point is that the robot keeps
  tracking while it leaves the ground and after it lands, not that it stops
  still. This is what replaced the old in-place planar-velocity penalty, which
  actively fought a moving jump.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = _jump_term(env, command_name).twist
  linear_error = torch.sum(
    torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1
  )
  angular_error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-linear_error / std_linear**2) * torch.exp(
    -angular_error / std_angular**2
  )


def takeoff_simultaneity(
  env: ManagerBasedRlEnv,
  command_name: str = "jump",
  window: tuple[float, float] = FLIGHT_WINDOW,
) -> torch.Tensor:
  """Reward all four feet leaving the ground together.

  Paid continuously through the flight window (rather than once) so it carries
  weight against the flight reward it is meant to shape, and scaled by how
  tightly the four liftoffs clustered. A jump that pushes off one leg at a time
  scores ~0 here even though it still earns flight reward.
  """
  term = _jump_term(env, command_name)
  simultaneity = term.simultaneity()
  env.extras.setdefault("log", {})["Metrics/jump_takeoff_spread"] = (
    term.takeoff_spread[term.takeoff_complete].mean()
    if bool(term.takeoff_complete.any())
    else torch.zeros((), device=env.device)
  )
  return (
    simultaneity * term.takeoff_complete.float() * phase_window(term.phase, *window)
  )


def jump_failure(
  env: ManagerBasedRlEnv,
  command_name: str = "jump",
) -> torch.Tensor:
  """Return 1 when an episode times out without ever leaving the ground."""
  term = _jump_term(env, command_name)
  timed_out = env.episode_length_buf >= env.max_episode_length
  return (timed_out & ~term.left_ground).float()
