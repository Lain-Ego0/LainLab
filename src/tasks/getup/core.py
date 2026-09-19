"""Shared fall-recovery task builder and registration."""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from src.tasks.getup.mdp import events as getup_events
from src.tasks.getup.mdp import rewards as getup_rewards
from src.tasks.rl import make_ppo_runner_cfg
from src.tasks.velocity.core import VelocityRobotProfile, _make_base_env_cfg


def _configure_getup_terrain(cfg: ManagerBasedRlEnvCfg) -> None:
  terrain = cfg.scene.terrain
  if terrain is None or terrain.terrain_generator is None:
    raise ValueError("Get-up requires a terrain generator")
  generator = terrain.terrain_generator
  generator.curriculum = False
  generator.num_rows = 4
  generator.num_cols = 6
  generator.size = (6.0, 6.0)
  generator.border_width = 10.0
  generator.difficulty_range = (0.0, 0.6)

  sub_terrains = dict(generator.sub_terrains)
  sub_terrains["pyramid_stairs"] = replace(
    sub_terrains["pyramid_stairs"],
    step_height_range=(0.0, 0.06),
    step_width=0.22,
  )
  sub_terrains["pyramid_stairs_inv"] = replace(
    sub_terrains["pyramid_stairs_inv"],
    step_height_range=(0.0, 0.06),
    step_width=0.22,
  )
  sub_terrains["hf_pyramid_slope"] = replace(
    sub_terrains["hf_pyramid_slope"],
    slope_range=(0.0, 0.6),
  )
  sub_terrains["hf_pyramid_slope_inv"] = replace(
    sub_terrains["hf_pyramid_slope_inv"],
    slope_range=(0.0, 0.6),
  )
  sub_terrains["random_rough"] = replace(
    sub_terrains["random_rough"],
    noise_range=(0.01, 0.06),
    scale_with_difficulty=True,
  )
  sub_terrains["wave_terrain"] = replace(
    sub_terrains["wave_terrain"],
    amplitude_range=(0.0, 0.12),
  )
  generator.sub_terrains = sub_terrains


def _configure_getup_commands(cfg: ManagerBasedRlEnvCfg) -> None:
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.heading_command = False
  command.rel_standing_envs = 1.0
  command.rel_heading_envs = 0.0
  command.rel_forward_envs = 0.0
  command.rel_world_envs = 0.0
  command.resampling_time_range = (1.0e9, 1.0e9)
  command.ranges.lin_vel_x = (0.0, 0.0)
  command.ranges.lin_vel_y = (0.0, 0.0)
  command.ranges.ang_vel_z = (0.0, 0.0)
  command.ranges.heading = None


def _configure_getup_events(
  cfg: ManagerBasedRlEnvCfg,
  *,
  fallen_height_range: tuple[float, float],
  joint_scale_range: tuple[float, float],
) -> None:
  startup_events = {
    name: cfg.events[name]
    for name in ("foot_friction", "encoder_bias", "base_com")
    if name in cfg.events
  }
  cfg.events = {
    "randomize_terrain": EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    ),
    "reset_base": EventTermCfg(
      func=getup_events.reset_fallen_root,
      mode="reset",
      params={"height_range": fallen_height_range},
    ),
    "reset_robot_joints": EventTermCfg(
      func=getup_events.reset_fallen_joints,
      mode="reset",
      params={"scale_range": joint_scale_range},
    ),
    **startup_events,
  }


def _configure_getup_rewards(
  cfg: ManagerBasedRlEnvCfg, *, base_height_target: float
) -> None:
  cfg.rewards = {
    "upright": RewardTermCfg(func=getup_rewards.upright_linear, weight=4.0),
    "base_height": RewardTermCfg(
      func=getup_rewards.base_height_tracking,
      weight=5.0,
      params={"target_height": base_height_target},
    ),
    "torques": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-0.0002),
    "dof_vel": RewardTermCfg(func=getup_rewards.joint_vel_l1, weight=-0.003),
    "joint_power": RewardTermCfg(func=getup_rewards.joint_power_l1, weight=-0.0001),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.02),
    "dof_acc": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-1.5e-7),
    "ang_vel_xy": RewardTermCfg(func=getup_rewards.gated_ang_vel_xy, weight=-0.03),
    "impact": RewardTermCfg(
      func=getup_rewards.impact,
      weight=-0.5,
      params={"target_height": base_height_target},
    ),
    "default_pos": RewardTermCfg(func=getup_rewards.default_pos_l1, weight=-0.08),
    "termination": RewardTermCfg(func=getup_rewards.getup_failure, weight=-7.0),
    "dof_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-1.0),
  }


def make_getup_env_cfg(
  profile: VelocityRobotProfile,
  *,
  base_height_target: float,
  fallen_height_range: tuple[float, float] = (0.04, 0.14),
  joint_scale_range: tuple[float, float] = (0.3, 1.7),
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build a terrain fall-recovery environment for one robot profile."""
  cfg = _make_base_env_cfg(profile)

  cfg.episode_length_s = 4.0
  cfg.curriculum = {}
  _configure_getup_commands(cfg)
  _configure_getup_terrain(cfg)
  _configure_getup_events(
    cfg,
    fallen_height_range=fallen_height_range,
    joint_scale_range=joint_scale_range,
  )
  _configure_getup_rewards(cfg, base_height_target=base_height_target)
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "out_of_terrain_bounds": TerminationTermCfg(
      func=velocity_mdp.out_of_terrain_bounds,
      time_out=True,
    ),
  }

  cfg.sim.nconmax = 64
  cfg.sim.njmax = 600
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.mujoco.ccd_iterations = 200

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg


def register_getup_profile(
  profile: VelocityRobotProfile,
  *,
  task_id: str,
  experiment_name: str,
  base_height_target: float,
  max_iterations: int = 5_000,
) -> None:
  """Register a train and play task for one robot profile."""
  env_cfg = make_getup_env_cfg(
    profile,
    base_height_target=base_height_target,
    play=False,
  )
  play_cfg = make_getup_env_cfg(
    profile,
    base_height_target=base_height_target,
    play=True,
  )
  rl_cfg = make_ppo_runner_cfg(
    experiment_name,
    max_iterations=max_iterations,
  )
  register_mjlab_task(
    task_id=task_id,
    env_cfg=env_cfg,
    play_env_cfg=play_cfg,
    rl_cfg=rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
  )
