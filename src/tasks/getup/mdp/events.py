"""Reset events for fall-recovery training."""

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_mul, sample_uniform

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_fallen_root(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  height_range: tuple[float, float] = (0.04, 0.14),
  xy_range: tuple[float, float] = (-0.25, 0.25),
  lin_vel_range: tuple[float, float] = (-0.3, 0.3),
  ang_vel_range: tuple[float, float] = (-2.0, 2.0),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset the root into a random non-upright fallen pose."""
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()
  num_envs = len(env_ids)
  device = env.device

  positions = root_states[:, :3] + env.scene.env_origins[env_ids]
  positions[:, 0] += sample_uniform(*xy_range, (num_envs,), device)
  positions[:, 1] += sample_uniform(*xy_range, (num_envs,), device)
  positions[:, 2] = env.scene.env_origins[env_ids, 2] + sample_uniform(
    *height_range, (num_envs,), device
  )

  angle_xy = sample_uniform(0.0, 2.0 * torch.pi, (num_envs,), device)
  tilt = sample_uniform(torch.pi / 3.0, torch.pi, (num_envs,), device)
  half = tilt / 2.0
  sin_half = torch.sin(half)
  random_quat = torch.stack(
    (
      torch.cos(angle_xy) * sin_half,
      torch.sin(angle_xy) * sin_half,
      torch.zeros_like(sin_half),
      torch.cos(half),
    ),
    dim=-1,
  )
  orientations = quat_mul(root_states[:, 3:7], random_quat)

  velocities = torch.cat(
    (
      sample_uniform(*lin_vel_range, (num_envs, 3), device),
      sample_uniform(*ang_vel_range, (num_envs, 3), device),
    ),
    dim=-1,
  )

  asset.write_root_link_pose_to_sim(
    torch.cat((positions, orientations), dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def reset_fallen_joints(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  scale_range: tuple[float, float] = (0.3, 1.7),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset joints around a wider, scaled default pose for fallen starts."""
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  soft_limits = asset.data.soft_joint_pos_limits
  assert default_joint_pos is not None
  assert soft_limits is not None

  default = default_joint_pos[env_ids]
  scales = sample_uniform(*scale_range, default.shape, env.device)
  joint_pos = default * scales
  limits = soft_limits[env_ids]
  joint_pos = torch.maximum(torch.minimum(joint_pos, limits[..., 1]), limits[..., 0])
  joint_vel = torch.zeros_like(joint_pos)
  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
