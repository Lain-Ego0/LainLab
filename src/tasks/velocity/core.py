"""Velocity-task types, environment builders, registration, and facade."""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  PinholeCameraPatternCfg,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from src.tasks.rl import make_ppo_runner_cfg

RobotKind = Literal["quadruped", "humanoid"]
TerrainName = Literal["Flat", "Rough"]
ActionScale = float | dict[str, float]
CommandRange = tuple[float, float]
CommandRanges = tuple[CommandRange, CommandRange, CommandRange]
SensorPatternCfg = GridPatternCfg | PinholeCameraPatternCfg | RingPatternCfg

TASK_GROUP_UNITREE = "Unitree"
TASK_GROUP_LAINLAB = "LainLab"

_TASK_GROUP_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_TERRAINS: frozenset[TerrainName] = frozenset({"Flat", "Rough"})


@dataclass(frozen=True)
class VelocityScaling:
  """Explicit environment/action scaling for one robot profile.

  Keeping these values in a separate object avoids silently inheriting
  Go2-sized defaults when a new robot is added.
  """

  action_scale: ActionScale
  command_z_offset: float
  max_iterations: int
  viewer_distance: float
  foot_radius: float | None
  foot_sample_count: int | None
  clearance_target_height: float
  reset_height_range: tuple[float, float]
  command_ranges: CommandRanges
  play_command_ranges: CommandRanges
  command_velocity_stages: tuple[dict[str, object], ...] | None
  base_com_scale: float
  decimation: int


@dataclass(frozen=True)
class SubTerrainOverrideCfg:
  """Partial field overrides for one named terrain sub-generator."""

  step_height_range: tuple[float, float] | None = None
  step_width: float | None = None
  noise_range: tuple[float, float] | None = None
  scale_with_difficulty: bool | None = None
  slope_range: tuple[float, float] | None = None
  amplitude_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class RoughTerrainOverrides:
  """Declarative overrides for a rough terrain generator."""

  max_init_terrain_level: int | None = None
  generator_size: tuple[float, float] | None = None
  generator_border_width: float | None = None
  generator_difficulty_range: tuple[float, float] | None = None
  max_level_stages: tuple[tuple[int, int], ...] = ()
  promotion_distance_scale: float | None = None
  sub_terrains: Mapping[str, SubTerrainOverrideCfg] = field(default_factory=dict)


@dataclass(frozen=True)
class SensorOverrideCfg:
  """Pattern override for a named raycast sensor."""

  name: str
  pattern: SensorPatternCfg


@dataclass(frozen=True)
class SimOverrides:
  """Explicit simulation buffer overrides for one variant."""

  nconmax: int | None = None
  njmax: int | None = None
  contact_sensor_maxmatch: int | None = None
  mujoco_ccd_iterations: int | None = None


@dataclass(frozen=True)
class RoughVariantCfg:
  """Optional rough-terrain configuration for a robot profile."""

  scaling: VelocityScaling | None = None
  terrain: RoughTerrainOverrides | None = None
  sensors: tuple[SensorOverrideCfg, ...] = ()
  sim: SimOverrides | None = None


def quadruped_velocity_scaling() -> VelocityScaling:
  """Baseline scaling for the existing small/medium Unitree quadrupeds."""
  return VelocityScaling(
    action_scale=0.25,
    command_z_offset=0.5,
    max_iterations=10_000,
    viewer_distance=1.5,
    foot_radius=None,
    foot_sample_count=None,
    clearance_target_height=0.1,
    reset_height_range=(0.01, 0.05),
    command_ranges=((-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5)),
    play_command_ranges=((-1.0, 1.5), (-0.5, 0.5), (-0.7, 0.7)),
    command_velocity_stages=None,
    base_com_scale=1.0,
    decimation=4,
  )


def humanoid_velocity_scaling(
  action_scale: ActionScale,
  command_z_offset: float,
  max_iterations: int = 30_000,
) -> VelocityScaling:
  """Baseline scaling for the existing Unitree humanoids."""
  return VelocityScaling(
    action_scale=action_scale,
    command_z_offset=command_z_offset,
    max_iterations=max_iterations,
    viewer_distance=1.5,
    foot_radius=None,
    foot_sample_count=None,
    clearance_target_height=0.1,
    reset_height_range=(0.01, 0.05),
    command_ranges=((-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5)),
    play_command_ranges=((-1.0, 1.5), (-0.5, 0.5), (-0.7, 0.7)),
    command_velocity_stages=None,
    base_com_scale=1.0,
    decimation=4,
  )


@dataclass(frozen=True)
class VelocityRobotProfile:
  """Robot-specific names and tuning layered on mjlab's shared velocity task.

  ``rough`` declares typed rough-only scaling, terrain, sensor, and simulation
  overrides. It cannot mutate observations, rewards, or actions.
  """

  task_name: str
  robot_cfg: Callable[[], EntityCfg]
  kind: RobotKind
  root_body: str
  viewer_body: str
  foot_sites: tuple[str, ...]
  foot_geoms: tuple[str, ...]
  foot_contact_pattern: str | tuple[str, ...]
  scaling: VelocityScaling
  task_group: str = ""
  terrains: tuple[TerrainName, ...] = ("Flat", "Rough")
  rough: RoughVariantCfg | None = None

  def __post_init__(self) -> None:
    if not self.task_group:
      raise ValueError(f"Velocity profile {self.task_name!r} must declare a task_group")
    if not _TASK_GROUP_RE.fullmatch(self.task_group):
      raise ValueError(
        f"Invalid task_group {self.task_group!r}; expected letters, digits, "
        "underscores, or hyphens"
      )
    if not self.terrains:
      raise ValueError(f"Velocity profile {self.task_name!r} must declare terrains")
    if len(set(self.terrains)) != len(self.terrains):
      raise ValueError(f"Velocity profile {self.task_name!r} has duplicate terrains")
    invalid = set(self.terrains) - _TERRAINS
    if invalid:
      raise ValueError(
        f"Velocity profile {self.task_name!r} has unknown terrains: {sorted(invalid)}"
      )

  @property
  def action_scale(self) -> ActionScale:
    return self.scaling.action_scale

  @property
  def command_z_offset(self) -> float:
    return self.scaling.command_z_offset

  @property
  def max_iterations(self) -> int:
    return self.scaling.max_iterations

  @property
  def viewer_distance(self) -> float:
    return self.scaling.viewer_distance

  @property
  def foot_radius(self) -> float | None:
    return self.scaling.foot_radius

  @property
  def foot_sample_count(self) -> int | None:
    return self.scaling.foot_sample_count

  @property
  def clearance_target_height(self) -> float:
    return self.scaling.clearance_target_height

  @property
  def reset_height_range(self) -> tuple[float, float]:
    return self.scaling.reset_height_range

  @property
  def command_ranges(self) -> CommandRanges:
    return self.scaling.command_ranges

  @property
  def play_command_ranges(self) -> CommandRanges:
    return self.scaling.play_command_ranges

  @property
  def command_velocity_stages(self) -> tuple[dict[str, object], ...] | None:
    return self.scaling.command_velocity_stages

  @property
  def base_com_scale(self) -> float:
    return self.scaling.base_com_scale

  @property
  def decimation(self) -> int:
    return self.scaling.decimation


def _configure_height_sensors(
  cfg: ManagerBasedRlEnvCfg, profile: VelocityRobotProfile
) -> None:
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = profile.root_body
    elif sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=name, entity="robot") for name in profile.foot_sites
      )
      radius = (
        profile.foot_radius
        if profile.foot_radius is not None
        else (0.04 if profile.kind == "quadruped" else 0.03)
      )
      samples = (
        profile.foot_sample_count
        if profile.foot_sample_count is not None
        else (4 if profile.kind == "quadruped" else 6)
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=radius, num_samples=samples)


def _contact_sensors(
  profile: VelocityRobotProfile,
) -> tuple[ContactSensorCfg, ContactSensorCfg]:
  primary_mode = "geom" if profile.kind == "quadruped" else "subtree"
  feet = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode=primary_mode,
      pattern=profile.foot_contact_pattern,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  if profile.kind == "humanoid":
    other = ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    )
  else:
    other = ContactSensorCfg(
      name="nonfoot_ground_touch",
      primary=ContactMatch(
        mode="geom",
        pattern=r".*_collision\d*$",
        exclude=profile.foot_geoms,
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    )
  return feet, other


def _configure_posture(
  cfg: ManagerBasedRlEnvCfg, profile: VelocityRobotProfile
) -> None:
  if profile.kind == "quadruped":
    cfg.rewards["pose"].params["std_standing"] = {
      r".*_(hip|thigh)_joint.*": 0.05,
      r".*_calf_joint.*": 0.1,
    }
    moving = {
      r".*_(hip|thigh)_joint.*": 0.3,
      r".*_calf_joint.*": 0.6,
    }
  else:
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    moving = {
      r".*hip_pitch.*": 0.3,
      r".*hip_(roll|yaw).*": 0.15,
      r".*knee.*": 0.35,
      r".*ankle_pitch.*": 0.25,
      r".*ankle_roll.*": 0.1,
      r".*(waist|torso).*": 0.2,
      r".*shoulder.*": 0.15,
      r".*elbow.*": 0.15,
      r".*wrist.*": 0.3,
    }
  cfg.rewards["pose"].params["std_walking"] = moving
  cfg.rewards["pose"].params["std_running"] = moving


def _make_base_env_cfg(profile: VelocityRobotProfile) -> ManagerBasedRlEnvCfg:
  """Build the terrain-agnostic velocity environment for a robot profile."""
  cfg = make_velocity_env_cfg()
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70 if profile.kind == "humanoid" else None
  cfg.scene.entities = {"robot": profile.robot_cfg()}

  _configure_height_sensors(cfg, profile)
  feet_sensor, other_sensor = _contact_sensors(profile)
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_sensor, other_sensor)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = profile.action_scale
  cfg.decimation = profile.decimation

  cfg.viewer.body_name = profile.viewer_body
  cfg.viewer.distance = profile.viewer_distance
  cfg.viewer.elevation = -10.0
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.viz.z_offset = profile.command_z_offset
  command.ranges.lin_vel_x = profile.command_ranges[0]
  command.ranges.lin_vel_y = profile.command_ranges[1]
  command.ranges.ang_vel_z = profile.command_ranges[2]
  if profile.command_velocity_stages is not None:
    cfg.curriculum["command_vel"].params["velocity_stages"] = list(
      profile.command_velocity_stages
    )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = profile.foot_geoms
  cfg.events["base_com"].params["asset_cfg"].body_names = (profile.viewer_body,)
  cfg.events["reset_base"].params["pose_range"]["z"] = profile.reset_height_range
  if profile.base_com_scale != 1.0:
    cfg.events["base_com"].params["ranges"] = {
      axis: (low * profile.base_com_scale, high * profile.base_com_scale)
      for axis, (low, high) in cfg.events["base_com"].params["ranges"].items()
    }
  _configure_posture(cfg, profile)
  cfg.rewards["upright"].params["asset_cfg"].body_names = (profile.viewer_body,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (profile.viewer_body,)
  for name in ("foot_clearance", "foot_slip"):
    cfg.rewards[name].params["asset_cfg"].site_names = profile.foot_sites
  cfg.rewards["foot_clearance"].params["target_height"] = (
    profile.clearance_target_height
  )
  cfg.rewards["foot_swing_height"].params["target_height"] = (
    profile.clearance_target_height
  )

  if profile.kind == "humanoid":
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["self_collisions"] = RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": other_sensor.name, "force_threshold": 10.0},
    )
  else:
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": other_sensor.name, "force_threshold": 10.0},
    )
  return cfg


def _apply_play_overrides(
  cfg: ManagerBasedRlEnvCfg,
  play: bool,
  *,
  command_ranges: CommandRanges | None = None,
  fixed_terrain_level: int | None = None,
  fixed_terrain_type: int | None = None,
) -> None:
  if not play:
    return
  cfg.episode_length_s = int(1e9)
  cfg.observations["actor"].enable_corruption = False
  cfg.events.pop("push_robot", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum = {}
  if fixed_terrain_level is None:
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
  else:
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=set_terrain_level,
      mode="reset",
      params={
        "fixed_level": fixed_terrain_level,
        "fixed_type": fixed_terrain_type,
      },
    )
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    terrain = cfg.scene.terrain.terrain_generator
    terrain.curriculum = fixed_terrain_level is not None
    if fixed_terrain_level is None:
      terrain.num_cols = 5
      terrain.num_rows = 5
    terrain.border_width = 10.0
  if command_ranges is not None:
    command = cfg.commands["twist"]
    assert isinstance(command, UniformVelocityCommandCfg)
    command.ranges.lin_vel_x = command_ranges[0]
    command.ranges.lin_vel_y = command_ranges[1]
    command.ranges.ang_vel_z = command_ranges[2]


def _sub_terrain_patch(patch: SubTerrainOverrideCfg) -> dict[str, object]:
  values: dict[str, object] = {}
  for name in (
    "step_height_range",
    "step_width",
    "noise_range",
    "scale_with_difficulty",
    "slope_range",
    "amplitude_range",
  ):
    value = getattr(patch, name)
    if value is not None:
      values[name] = value
  return values


def set_terrain_level(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  fixed_level: int,
  fixed_type: int | None = None,
) -> None:
  """Place reset environments on one curriculum row for deterministic playback."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    return
  level = min(max(fixed_level, 0), terrain.terrain_origins.shape[0] - 1)
  if fixed_type is not None:
    fixed_type = min(max(fixed_type, 0), terrain.terrain_origins.shape[1] - 1)
    terrain.terrain_types[env_ids] = fixed_type
  terrain.terrain_levels[env_ids] = level
  terrain.env_origins[env_ids] = terrain.terrain_origins[
    terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
  ]


def _max_terrain_level_for_step(
  common_step_counter: int,
  max_level_stages: tuple[tuple[int, int], ...],
) -> int | None:
  max_level = None
  for step, level in max_level_stages:
    if common_step_counter >= step:
      max_level = level
  return max_level


def terrain_levels_vel_limited(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  *,
  max_level_stages: tuple[tuple[int, int], ...] = (),
  promotion_distance_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
  """Advance terrain levels while optionally limiting the maximum level."""
  asset = env.scene["robot"]
  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None
  command = env.command_manager.get_command(command_name)
  assert command is not None

  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )
  move_up = distance > (terrain_generator.size[0] / 2.0 * promotion_distance_scale)
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # On the initial reset the robot has not walked anywhere yet.
  if env.common_step_counter == 0:
    move_up = torch.zeros_like(move_up)
    move_down = torch.zeros_like(move_down)

  max_level = _max_terrain_level_for_step(env.common_step_counter, max_level_stages)
  if max_level is not None:
    move_up &= terrain.terrain_levels[env_ids] < max_level

  terrain.update_env_origins(env_ids, move_up, move_down)

  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def _apply_rough_terrain_overrides(
  cfg: ManagerBasedRlEnvCfg, overrides: RoughTerrainOverrides
) -> None:
  terrain = cfg.scene.terrain
  if terrain is None or terrain.terrain_generator is None:
    raise ValueError("Rough terrain overrides require a terrain generator")
  if overrides.max_init_terrain_level is not None:
    terrain.max_init_terrain_level = overrides.max_init_terrain_level

  generator = terrain.terrain_generator
  if overrides.generator_size is not None:
    generator.size = overrides.generator_size
  if overrides.generator_border_width is not None:
    generator.border_width = overrides.generator_border_width
  if overrides.generator_difficulty_range is not None:
    generator.difficulty_range = overrides.generator_difficulty_range

  if overrides.max_level_stages or overrides.promotion_distance_scale is not None:
    term = cfg.curriculum.get("terrain_levels")
    if term is None:
      raise ValueError("Terrain level curriculum is not configured")
    term.func = terrain_levels_vel_limited
    term.params = dict(term.params or {})
    term.params["max_level_stages"] = overrides.max_level_stages
    if overrides.promotion_distance_scale is not None:
      term.params["promotion_distance_scale"] = overrides.promotion_distance_scale

  if overrides.sub_terrains:
    # ``make_velocity_env_cfg`` builds its generator with a shallow
    # ``replace(ROUGH_TERRAINS_CFG)``, so the sub-terrain dict is the *same*
    # object as mjlab's module-level default and is shared by every rough task
    # built in this process. Patch a private copy, or one robot's overrides leak
    # into all the others (including already-built configs).
    generator.sub_terrains = dict(generator.sub_terrains)

  for name, patch in overrides.sub_terrains.items():
    if name not in generator.sub_terrains:
      raise ValueError(f"Unknown sub-terrain override {name!r}")
    patch_values = _sub_terrain_patch(patch)
    if patch_values:
      generator.sub_terrains[name] = replace(
        generator.sub_terrains[name],
        **patch_values,
      )


def _apply_sensor_overrides(
  cfg: ManagerBasedRlEnvCfg, overrides: tuple[SensorOverrideCfg, ...]
) -> None:
  sensors = {sensor.name: sensor for sensor in (cfg.scene.sensors or ())}
  for override in overrides:
    sensor = sensors.get(override.name)
    if sensor is None:
      raise ValueError(f"Unknown sensor override {override.name!r}")
    if not isinstance(sensor, RayCastSensorCfg):
      raise TypeError(f"Sensor {override.name!r} is not a raycast sensor")
    sensor.pattern = override.pattern


def _apply_sim_overrides(cfg: ManagerBasedRlEnvCfg, overrides: SimOverrides) -> None:
  if overrides.nconmax is not None:
    cfg.sim.nconmax = overrides.nconmax
  if overrides.njmax is not None:
    cfg.sim.njmax = overrides.njmax
  if overrides.contact_sensor_maxmatch is not None:
    cfg.sim.contact_sensor_maxmatch = overrides.contact_sensor_maxmatch
  if overrides.mujoco_ccd_iterations is not None:
    cfg.sim.mujoco.ccd_iterations = overrides.mujoco_ccd_iterations


def _apply_rough_variant_overrides(
  cfg: ManagerBasedRlEnvCfg, rough: RoughVariantCfg
) -> None:
  if rough.terrain is not None:
    _apply_rough_terrain_overrides(cfg, rough.terrain)
  if rough.sensors:
    _apply_sensor_overrides(cfg, rough.sensors)
  if rough.sim is not None:
    _apply_sim_overrides(cfg, rough.sim)


def _scaling_for_terrain(
  profile: VelocityRobotProfile, terrain: TerrainName
) -> VelocityScaling:
  if (
    terrain == "Rough"
    and profile.rough is not None
    and profile.rough.scaling is not None
  ):
    return profile.rough.scaling
  return profile.scaling


def make_rough_env_cfg(
  profile: VelocityRobotProfile, *, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Build a rough-terrain environment from a compact robot profile."""
  rough = profile.rough
  selected_scaling = _scaling_for_terrain(profile, "Rough")
  cfg = _make_base_env_cfg(replace(profile, scaling=selected_scaling))
  if rough is not None:
    _apply_rough_variant_overrides(cfg, rough)
  _apply_play_overrides(
    cfg,
    play,
    command_ranges=selected_scaling.play_command_ranges,
    fixed_terrain_level=(
      cfg.scene.terrain.terrain_generator.num_rows - 1
      if rough is not None
      and rough.terrain is not None
      and cfg.scene.terrain is not None
      and cfg.scene.terrain.terrain_generator is not None
      and rough.terrain.max_level_stages
      else None
    ),
    fixed_terrain_type=(
      list(cfg.scene.terrain.terrain_generator.sub_terrains).index("hf_pyramid_slope")
      if rough is not None
      and rough.terrain is not None
      and cfg.scene.terrain is not None
      and cfg.scene.terrain.terrain_generator is not None
      and "hf_pyramid_slope" in cfg.scene.terrain.terrain_generator.sub_terrains
      else None
    ),
  )
  return cfg


def make_flat_env_cfg(
  profile: VelocityRobotProfile, *, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Build a flat-ground variant of a robot's velocity task."""
  flat_scaling = _scaling_for_terrain(profile, "Flat")
  cfg = _make_base_env_cfg(replace(profile, scaling=flat_scaling))
  _apply_play_overrides(
    cfg,
    play,
    command_ranges=flat_scaling.play_command_ranges,
  )
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)
  return cfg


_ENV_FACTORIES: dict[TerrainName, Callable[..., ManagerBasedRlEnvCfg]] = {
  "Flat": make_flat_env_cfg,
  "Rough": make_rough_env_cfg,
}


def register_velocity_profile(profile: VelocityRobotProfile) -> None:
  """Register all configured terrains for one robot profile."""
  for terrain in profile.terrains:
    experiment_name = (
      f"{profile.task_group.lower().replace('-', '_')}_"
      f"{profile.task_name.lower().replace('-', '_')}_"
      f"{terrain.lower()}_velocity"
    )
    runner_cfg = make_ppo_runner_cfg(
      experiment_name,
      max_iterations=_scaling_for_terrain(profile, terrain).max_iterations,
    )
    env_factory = _ENV_FACTORIES[terrain]
    register_mjlab_task(
      task_id=f"{profile.task_group}-{profile.task_name}-{terrain}",
      env_cfg=env_factory(profile),
      play_env_cfg=env_factory(profile, play=True),
      rl_cfg=runner_cfg,
      runner_cls=VelocityOnPolicyRunner,
    )
