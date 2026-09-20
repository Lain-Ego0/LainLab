"""Reward terms for fall-recovery training."""

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def upright_linear(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return -1 when upside-down and +1 when upright."""
  asset: Entity = env.scene[asset_cfg.name]
  return -asset.data.projected_gravity_b[:, 2]


def base_height_tracking(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track root height relative to the local terrain origin."""
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return -torch.abs(height - target_height)


def joint_vel_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_power_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  effort = torch.abs(asset.data.actuator_force[:, asset_cfg.actuator_ids])
  velocity = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
  return torch.sum(effort * velocity, dim=1)


def default_pos_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_pos = asset.data.default_joint_pos
  assert default_pos is not None
  return torch.sum(
    torch.abs(
      asset.data.joint_pos[:, asset_cfg.joint_ids] - default_pos[:, asset_cfg.joint_ids]
    ),
    dim=1,
  )


def gated_ang_vel_xy(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize roll/pitch wobble only once the robot is near upright."""
  asset: Entity = env.scene[asset_cfg.name]
  uprightness = -asset.data.projected_gravity_b[:, 2]
  gate = torch.clamp((uprightness - 0.3) / 0.7, 0.0, 1.0)
  return gate * torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def impact(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize downward root velocity near the standing height."""
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  near_target = torch.exp(-torch.square((height - target_height) / 0.06))
  downward_velocity = torch.clamp(-asset.data.root_link_lin_vel_w[:, 2], min=0.0)
  return downward_velocity * near_target


def getup_failure(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return 1 when an episode times out while the robot is still down."""
  asset: Entity = env.scene[asset_cfg.name]
  uprightness = -asset.data.projected_gravity_b[:, 2]
  timed_out = env.episode_length_buf >= env.max_episode_length
  still_down = uprightness < 0.7
  return (timed_out & still_down).float()
