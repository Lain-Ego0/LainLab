"""Recovery-specific MDP terms for the recover-to-walk task."""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from src.tasks.getup.mdp.events import reset_fallen_joints, reset_fallen_root

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _split_fallen(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor, probability: float
) -> tuple[torch.Tensor, torch.Tensor]:
  """Split ``env_ids`` into (fallen, standing) by ``probability``."""
  draw = torch.rand(len(env_ids), device=env.device)
  fallen_mask = draw < probability
  return env_ids[fallen_mask], env_ids[~fallen_mask]


def reset_mixed_root(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  fallen_probability: float = 0.5,
  standing_pose_range: dict[str, tuple[float, float]] | None = None,
  fallen_height_range: tuple[float, float] = (0.04, 0.14),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Start part of the batch standing and the rest from a random fall.

  A pure standing start never shows the policy how to *get* to standing, and a
  pure fallen start never shows it a clean nominal stance. Recovery-to-walking
  needs both, so the batch is split.
  """
  env_ids = resolve_env_ids(env, env_ids)
  pose_range = standing_pose_range or {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (0.004, 0.018),
    "yaw": (-3.14, 3.14),
  }
  fallen, standing = _split_fallen(env, env_ids, fallen_probability)
  if len(standing) > 0:
    envs_mdp.reset_root_state_uniform(
      env, standing, pose_range=pose_range, velocity_range={}, asset_cfg=asset_cfg
    )
  if len(fallen) > 0:
    reset_fallen_root(
      env, fallen, height_range=fallen_height_range, asset_cfg=asset_cfg
    )


def reset_mixed_joints(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  fallen_probability: float = 0.5,
  standing_scale_range: tuple[float, float] = (0.97, 1.03),
  fallen_scale_range: tuple[float, float] = (0.3, 1.7),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Joint-side counterpart of :func:`reset_mixed_root`."""
  env_ids = resolve_env_ids(env, env_ids)
  fallen, standing = _split_fallen(env, env_ids, fallen_probability)
  if len(standing) > 0:
    envs_mdp.reset_joints_by_offset(
      env,
      standing,
      position_range=(
        (standing_scale_range[0] - 1.0),
        (standing_scale_range[1] - 1.0),
      ),
      velocity_range=(0.0, 0.0),
      asset_cfg=asset_cfg,
    )
  if len(fallen) > 0:
    reset_fallen_joints(
      env, fallen, scale_range=fallen_scale_range, asset_cfg=asset_cfg
    )


def nonfoot_contact_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str = "nonfoot_ground_touch",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize body/knee contact with the ground.

  The walk task expresses this as an ``illegal_contact`` *termination*, which
  cannot be kept here: a fallen start would terminate immediately. A dense
  penalty replaces it, so dragging itself along on its belly is discouraged
  without forbidding the contact that recovery requires.
  """
  del asset_cfg
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None, f"Sensor {sensor_name!r} does not track 'force'."
  return torch.norm(force, dim=-1).sum(dim=-1)


def uprightness(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Diagnostic term: -1 upside-down, +1 upright (same shape as get-up's)."""
  asset: Entity = env.scene[asset_cfg.name]
  return -asset.data.projected_gravity_b[:, 2]
