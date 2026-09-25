"""Reset events for the handstand task."""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_standing_joints(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  scale_range: tuple[float, float] = (0.85, 1.15),
  velocity_range: tuple[float, float] = (0.0, 0.0),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset joints around the standing pose with a mild multiplicative spread.

  The handstand must build itself up from standing, so unlike get-up this reset
  stays near the nominal pose instead of sampling fallen configurations.
  """
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
  joint_vel = sample_uniform(*velocity_range, default.shape, env.device)
  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
