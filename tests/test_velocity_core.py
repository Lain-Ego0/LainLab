"""Contract tests for the velocity environment builders."""

from dataclasses import fields, replace

import src.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.robots.opendoge.velocity import OPENDOGE_VELOCITY_PROFILES
from src.tasks.velocity import (
  RoughTerrainOverrides,
  RoughVariantCfg,
  SensorOverrideCfg,
  SimOverrides,
  make_rough_env_cfg,
)

_FORBIDDEN_OVERRIDE_FIELDS = {
  "actions",
  "commands",
  "events",
  "observations",
  "rewards",
  "terminations",
}


def test_flat_and_rough_share_base_config() -> None:
  flat = load_env_cfg("LainLab-OpenDoge-Flat")
  rough = load_env_cfg("LainLab-OpenDoge-Rough")

  assert flat.actions == rough.actions
  assert flat.decimation == rough.decimation
  assert flat.episode_length_s == rough.episode_length_s
  assert flat.events.keys() == rough.events.keys()
  assert flat.rewards.keys() == rough.rewards.keys()
  assert flat.viewer == rough.viewer
  assert flat.sim.mujoco.timestep == rough.sim.mujoco.timestep
  assert set(flat.observations["actor"].terms) - {"height_scan"} == set(
    rough.observations["actor"].terms
  ) - {"height_scan"}


def test_rough_variant_drives_runner_and_play_config() -> None:
  flat_rl = load_rl_cfg("LainLab-OpenDoge-Flat")
  rough_rl = load_rl_cfg("LainLab-OpenDoge-Rough")
  assert flat_rl.max_iterations == 9_000
  assert rough_rl.max_iterations == 10_000

  play_cfg = load_env_cfg("LainLab-OpenDoge-Rough", play=True)
  command = play_cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  assert command.ranges.lin_vel_x == (-0.6, 0.8)
  assert command.ranges.lin_vel_y == (-0.4, 0.4)
  assert command.ranges.ang_vel_z == (-0.7, 0.7)
  terrain = play_cfg.scene.terrain
  assert terrain is not None
  assert terrain.max_init_terrain_level == 2
  generator = terrain.terrain_generator
  assert generator is not None
  assert generator.curriculum is True
  assert generator.num_rows == 10
  assert play_cfg.events["randomize_terrain"].func.__name__ == "set_terrain_level"
  assert play_cfg.events["randomize_terrain"].params["fixed_level"] == 9


def test_rough_override_types_exclude_high_level_config_sections() -> None:
  override_types = (
    RoughVariantCfg,
    RoughTerrainOverrides,
    SensorOverrideCfg,
    SimOverrides,
  )
  for override_type in override_types:
    assert {field.name for field in fields(override_type)}.isdisjoint(
      _FORBIDDEN_OVERRIDE_FIELDS
    )


def test_rough_overrides_do_not_touch_env_behavior_config() -> None:
  profile = OPENDOGE_VELOCITY_PROFILES[0]
  rough = profile.rough
  assert rough is not None
  control_profile = replace(
    profile,
    rough=replace(rough, terrain=None, sensors=(), sim=None),
  )

  configured = make_rough_env_cfg(profile)
  control = make_rough_env_cfg(control_profile)

  assert configured.actions == control.actions
  assert configured.observations == control.observations
  assert configured.rewards == control.rewards
  assert configured.commands == control.commands
  assert configured.events == control.events
  assert configured.terminations == control.terminations

  assert configured.scene.terrain is not None
  assert control.scene.terrain is not None
  assert configured.scene.terrain.terrain_generator is not None
  assert control.scene.terrain.terrain_generator is not None
  assert configured.scene.terrain.terrain_generator.size == (6.0, 6.0)
  assert control.scene.terrain.terrain_generator.size != (6.0, 6.0)


# ``mjlab==1.6.0`` is pinned, so these are the stable reference values of
# ``mjlab.terrains.config.ROUGH_TERRAINS_CFG`` before any LainLab override.
_MJLAB_ROUGH_STEP_WIDTH = 0.3
_MJLAB_ROUGH_SCALE_WITH_DIFFICULTY = False


def _rough_sub_terrains(task_id: str):
  cfg = load_env_cfg(task_id)
  terrain = cfg.scene.terrain
  assert terrain is not None
  assert terrain.terrain_generator is not None
  return terrain.terrain_generator.sub_terrains


def test_opendoge_rough_overrides_do_not_leak_into_other_robots() -> None:
  """OpenDoge's terrain overrides must stay inside its own task.

  ``make_velocity_env_cfg`` shallow-copies mjlab's module-level generator, so the
  sub-terrain dict is shared with every other rough config unless the builder
  copies it before patching. A regression here silently re-tunes the terrain of
  every pre-existing rough task, which no per-robot config test would notice.
  """
  from mjlab.terrains.config import ROUGH_TERRAINS_CFG

  unitree = _rough_sub_terrains("Unitree-Go2-Rough")
  opendoge = _rough_sub_terrains("LainLab-OpenDoge-Rough")

  assert unitree is not ROUGH_TERRAINS_CFG.sub_terrains
  assert opendoge is not ROUGH_TERRAINS_CFG.sub_terrains
  assert unitree is not opendoge

  assert unitree["pyramid_stairs"].step_width == _MJLAB_ROUGH_STEP_WIDTH
  assert unitree["pyramid_stairs_inv"].step_width == _MJLAB_ROUGH_STEP_WIDTH
  assert (
    unitree["random_rough"].scale_with_difficulty is _MJLAB_ROUGH_SCALE_WITH_DIFFICULTY
  )

  assert opendoge["pyramid_stairs"].step_width == 0.22
  assert opendoge["pyramid_stairs_inv"].step_width == 0.22
  assert opendoge["random_rough"].scale_with_difficulty is True


def test_no_other_rough_task_inherits_opendoge_terrain() -> None:
  """Every non-OpenDoge rough task keeps mjlab's untouched terrain defaults.

  Covers the Unitree robots, the Go2 skill tasks and mjlab's own built-in
  velocity tasks, because they all share the same module-level generator dict.
  """
  covered = 0
  for task_id in sorted(list_tasks()):
    if "Rough" not in task_id or task_id.startswith("LainLab-OpenDoge"):
      continue
    sub = _rough_sub_terrains(task_id)
    assert sub["pyramid_stairs"].step_width == _MJLAB_ROUGH_STEP_WIDTH, task_id
    assert (
      sub["random_rough"].scale_with_difficulty is _MJLAB_ROUGH_SCALE_WITH_DIFFICULTY
    ), task_id
    covered += 1
  assert covered >= 11


def test_registration_leaves_mjlab_terrain_default_untouched() -> None:
  from mjlab.terrains.config import ROUGH_TERRAINS_CFG

  default = ROUGH_TERRAINS_CFG.sub_terrains
  assert default["pyramid_stairs"].step_width == _MJLAB_ROUGH_STEP_WIDTH
  assert (
    default["random_rough"].scale_with_difficulty is _MJLAB_ROUGH_SCALE_WITH_DIFFICULTY
  )
