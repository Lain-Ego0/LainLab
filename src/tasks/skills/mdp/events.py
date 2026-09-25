"""Per-skill reset dispatch.

The reset distribution is skill-specific and that is not optional: walking,
handstand and jump all start from standing, while get-up must start from a
random fallen pose. Mixing those distributions would teach the policy that a
fallen pose sometimes means "walk", which is exactly the negative transfer the
single-policy design has to avoid.

Each sub-task's own standing-reset parameters are harvested from its validated
configuration rather than retyped here, so a skill cannot start from a different
distribution in the unified environment than it did on its own. That matters:
the jump is phase-timed, and giving it a joint perturbation its expert never saw
at reset cut its takeoff rate from 32/32 to 4/32.

The command manager resets *after* the event manager (see
``ManagerBasedRlEnv._reset_idx``), so the freshly sampled skill is established
here and the command manager is told not to sample a second one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

from src.tasks.getup.mdp.events import reset_fallen_joints, reset_fallen_root
from src.tasks.skills.mdp.utils import skill_term

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_VELOCITY_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


@dataclass(frozen=True)
class StandingReset:
  """How one skill starts an episode from standing."""

  height_offset_range: tuple[float, float] = (0.0, 0.0)
  """Offset from the default root height, matching ``pose_range['z']``."""
  joint_scale_range: tuple[float, float] | None = None
  """Multiplicative joint spread; ``None`` means an additive offset is used."""
  joint_offset_range: tuple[float, float] | None = None
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)


def _sample_velocity(
  ranges: dict[str, tuple[float, float]], count: int, device: str
) -> torch.Tensor:
  """Sample ``[x, y, z, roll, pitch, yaw]`` from per-axis ranges."""
  columns = []
  for axis in _VELOCITY_AXES:
    low, high = ranges.get(axis, (0.0, 0.0))
    columns.append(sample_uniform(low, high, (count,), device))
  return torch.stack(columns, dim=-1)


def reset_skill_episode(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str = "skill",
  standing: dict[str, StandingReset] | None = None,
  fallen_height_range: tuple[float, float] = (0.04, 0.14),
  fallen_joint_scale_range: tuple[float, float] = (0.3, 1.7),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset every environment according to the skill it was just assigned.

  All environments get a standing reset first, then get-up environments are
  overwritten with a fallen reset. Overwriting is cheaper than partitioning the
  ids, and the two resets write disjoint values.
  """
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  default_root_state = asset.data.default_root_state
  default_joint_pos = asset.data.default_joint_pos
  soft_limits = asset.data.soft_joint_pos_limits
  assert default_root_state is not None
  assert default_joint_pos is not None
  assert soft_limits is not None

  # This event runs before the command manager resets, so it has to establish
  # the skill itself; otherwise the reset distribution would follow the previous
  # episode's skill.
  term = skill_term(env, command_name)
  term.sample_skills(env_ids)

  num_envs = len(env_ids)
  device = str(env.device)
  specs = standing or {}
  fallback = StandingReset()

  offset_low = torch.zeros(num_envs, device=device)
  offset_high = torch.zeros(num_envs, device=device)
  scale_low = torch.ones(num_envs, device=device)
  scale_high = torch.ones(num_envs, device=device)
  shift_low = torch.zeros(num_envs, device=device)
  shift_high = torch.zeros(num_envs, device=device)
  for name in term.skill_names:
    mask = term.skill_is(name)[env_ids]
    if not bool(mask.any()):
      continue
    spec = specs.get(name, fallback)
    offset_low[mask], offset_high[mask] = spec.height_offset_range
    if spec.joint_scale_range is not None:
      scale_low[mask], scale_high[mask] = spec.joint_scale_range
    if spec.joint_offset_range is not None:
      shift_low[mask], shift_high[mask] = spec.joint_offset_range

  root_states = default_root_state[env_ids].clone()
  positions = root_states[:, :3] + env.scene.env_origins[env_ids]
  positions[:, 2] += (
    sample_uniform(0.0, 1.0, (num_envs,), device) * (offset_high - offset_low)
    + offset_low
  )

  velocities = torch.zeros(num_envs, 6, device=device)
  for name in term.skill_names:
    mask = term.skill_is(name)[env_ids]
    if not bool(mask.any()):
      continue
    ranges = specs.get(name, fallback).velocity_range
    if ranges:
      velocities[mask] = _sample_velocity(ranges, int(mask.sum()), str(device))

  asset.write_root_link_pose_to_sim(
    torch.cat((positions, root_states[:, 3:7]), dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)

  draw = sample_uniform(0.0, 1.0, default_joint_pos[env_ids].shape, device)
  joint_pos = default_joint_pos[env_ids] * (
    scale_low[:, None] + draw * (scale_high - scale_low)[:, None]
  ) + (shift_low[:, None] + draw * (shift_high - shift_low)[:, None])
  limits = soft_limits[env_ids]
  joint_pos = torch.maximum(torch.minimum(joint_pos, limits[..., 1]), limits[..., 0])
  asset.write_joint_state_to_sim(
    joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids
  )

  getup_ids = env_ids[term.skill_is("getup")[env_ids]]
  if len(getup_ids) == 0:
    return
  reset_fallen_root(
    env, getup_ids, height_range=fallen_height_range, asset_cfg=asset_cfg
  )
  reset_fallen_joints(
    env, getup_ids, scale_range=fallen_joint_scale_range, asset_cfg=asset_cfg
  )
