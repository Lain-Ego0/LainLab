"""Reward terms for the front-support handstand task.

Targets come from the statics calibration in
``docs/opendoge_multiskill_targets.md``: the body pitches to vertical
(``projected_gravity_b -> [1, 0, 0]``), the front feet stay planted and the rear
feet are held clear of the ground.

Tracking terms are bounded quadratic wells rather than Gaussians on purpose. A
Gaussian with a small std is numerically zero over most of the state space, so
from a standing reset the policy would receive no useful gradient; these wells
stay informative until the error reaches their tolerance.
"""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_GRAVITY_TARGET = (1.0, 0.0, 0.0)


def _contact_flags(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-primary boolean contact flags as float, shape ``[B, P]``."""
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None, f"Sensor {sensor_name!r} does not track 'found'."
  flags = (found > 0).to(torch.float32)
  if flags.dim() == 3:  # [B, P, num_slots] -> [B, P]
    flags = flags.amax(dim=-1)
  return flags


def _alignment(
  env: ManagerBasedRlEnv,
  target: tuple[float, float, float],
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Dot product between projected gravity and the handstand target.

  Runs from 0 while standing (``projected_gravity_b == [0, 0, -1]``) to 1 at the
  target orientation. Using the raw dot product keeps the signal dense across
  the whole pitch-up manoeuvre, which is what the Gaussian form fails at.
  """
  asset: Entity = env.scene[asset_cfg.name]
  target_tensor = torch.tensor(target, device=env.device)
  return torch.clamp(asset.data.projected_gravity_b @ target_tensor, -1.0, 1.0)


def gravity_alignment(
  env: ManagerBasedRlEnv,
  target: tuple[float, float, float] = _DEFAULT_GRAVITY_TARGET,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward body orientation matching the handstand target (0 standing, 1 held)."""
  return _alignment(env, target, asset_cfg)


def base_height_tracking(
  env: ManagerBasedRlEnv,
  target_height: float,
  tolerance: float = 0.08,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward holding the base at the calibrated handstand height."""
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  error = (height - target_height) / tolerance
  return torch.clamp(1.0 - torch.square(error), min=0.0)


def rear_foot_clearance(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  rear_ids: tuple[int, ...],
  target_height: float,
  tolerance: float = 0.15,
) -> torch.Tensor:
  """Reward the raised pair sitting at the target clearance above the ground."""
  sensor: TerrainHeightSensor = env.scene[sensor_name]
  heights = sensor.data.heights  # [B, F]
  selected = heights[:, list(rear_ids)]
  error = torch.sum(torch.square((selected - target_height) / tolerance), dim=1)
  return torch.clamp(1.0 - error / len(rear_ids), min=0.0)


def support_feet_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  support_ids: tuple[int, ...],
) -> torch.Tensor:
  """Reward keeping the support pair planted."""
  flags = _contact_flags(env, sensor_name)[:, list(support_ids)]
  return torch.sum(flags, dim=1)


def rear_feet_airborne(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  rear_ids: tuple[int, ...],
) -> torch.Tensor:
  """Reward keeping the raised pair off the ground."""
  flags = _contact_flags(env, sensor_name)[:, list(rear_ids)]
  return torch.sum(1.0 - flags, dim=1)


def gated_ang_vel_xy(
  env: ManagerBasedRlEnv,
  target: tuple[float, float, float] = _DEFAULT_GRAVITY_TARGET,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize roll/pitch wobble only once the body is near the handstand pose.

  Gating matters: while the robot is still pitching up, large body rates are how
  it gets there, and penalizing them unconditionally fights the rise.
  """
  asset: Entity = env.scene[asset_cfg.name]
  gate = torch.clamp(_alignment(env, target, asset_cfg), 0.0, 1.0)
  return gate * torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def planar_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize drifting; a handstand should stay over its support feet."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_lin_vel_w[:, :2]), dim=1)


def handstand_failure(
  env: ManagerBasedRlEnv,
  target: tuple[float, float, float] = _DEFAULT_GRAVITY_TARGET,
  threshold: float = 0.9,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return 1 when an episode times out while still not in the handstand.

  ``threshold=0.9`` corresponds to roughly 25 degrees off vertical. This closes
  the "survive without attempting the task" loophole that the Go2 handstand
  migration ran into.
  """
  timed_out = env.episode_length_buf >= env.max_episode_length
  not_held = _alignment(env, target, asset_cfg) < threshold
  return (timed_out & not_held).float()
