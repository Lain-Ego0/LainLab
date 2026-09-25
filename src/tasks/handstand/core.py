"""Shared front-support handstand task builder and registration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from src.tasks.handstand.mdp import events as handstand_events
from src.tasks.handstand.mdp import rewards as handstand_rewards
from src.tasks.rl import make_ppo_runner_cfg
from src.tasks.velocity.core import VelocityRobotProfile, _make_base_env_cfg


def _configure_flat_terrain(cfg: ManagerBasedRlEnvCfg) -> None:
  """Force flat ground and drop the exteroceptive scan.

  Keeping this identical to the get-up task is what makes the two skills share
  one observation layout: both end up with a 48-field actor observation.
  """
  terrain = cfg.scene.terrain
  if terrain is None:
    raise ValueError("Handstand requires a terrain entity")
  terrain.terrain_type = "plane"
  terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)


def _configure_handstand_commands(cfg: ManagerBasedRlEnvCfg) -> None:
  """Zero the twist command: a handstand is a station-keeping task."""
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


def _configure_handstand_events(
  cfg: ManagerBasedRlEnvCfg,
  *,
  joint_scale_range: tuple[float, float],
  reset_velocity_range: dict[str, tuple[float, float]],
) -> None:
  startup_events = {
    name: cfg.events[name]
    for name in ("foot_friction", "encoder_bias", "base_com")
    if name in cfg.events
  }
  cfg.events = {
    "reset_base": cfg.events["reset_base"],
    "reset_robot_joints": EventTermCfg(
      func=handstand_events.reset_standing_joints,
      mode="reset",
      params={"scale_range": joint_scale_range},
    ),
    **startup_events,
  }
  cfg.events["reset_base"].params["velocity_range"] = reset_velocity_range


def _configure_handstand_rewards(
  cfg: ManagerBasedRlEnvCfg,
  *,
  base_height_target: float,
  rear_clearance_target: float,
  gravity_target: tuple[float, float, float],
  support_ids: tuple[int, ...],
  rear_ids: tuple[int, ...],
) -> None:
  cfg.rewards = {
    # Task tracking.
    "gravity_alignment": RewardTermCfg(
      func=handstand_rewards.gravity_alignment,
      weight=2.0,
      params={"target": gravity_target},
    ),
    "base_height": RewardTermCfg(
      func=handstand_rewards.base_height_tracking,
      weight=3.0,
      params={"target_height": base_height_target, "tolerance": 0.08},
    ),
    "rear_foot_clearance": RewardTermCfg(
      func=handstand_rewards.rear_foot_clearance,
      weight=2.0,
      params={
        "sensor_name": "foot_height_scan",
        "rear_ids": rear_ids,
        "target_height": rear_clearance_target,
        "tolerance": 0.15,
      },
    ),
    "support_contact": RewardTermCfg(
      func=handstand_rewards.support_feet_contact,
      weight=1.0,
      params={"sensor_name": "feet_ground_contact", "support_ids": support_ids},
    ),
    "rear_airborne": RewardTermCfg(
      func=handstand_rewards.rear_feet_airborne,
      weight=1.0,
      params={"sensor_name": "feet_ground_contact", "rear_ids": rear_ids},
    ),
    # Loophole closure: survive-and-do-nothing must be worse than trying.
    "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=1.0),
    "failure": RewardTermCfg(
      func=handstand_rewards.handstand_failure,
      weight=-5.0,
      params={"target": gravity_target},
    ),
    # Regularization.
    "torques": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-0.0005),
    "dof_vel": RewardTermCfg(func=envs_mdp.joint_vel_l2, weight=-0.002),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.02),
    "dof_acc": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-1.5e-7),
    "ang_vel_xy": RewardTermCfg(
      func=handstand_rewards.gated_ang_vel_xy,
      weight=-0.05,
      params={"target": gravity_target},
    ),
    "planar_vel": RewardTermCfg(
      func=handstand_rewards.planar_velocity_penalty,
      weight=-0.1,
    ),
    "dof_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-1.0),
  }


def _configure_handstand_sim(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.sim.nconmax = 64
  cfg.sim.njmax = 600
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.mujoco.ccd_iterations = 200


def _foot_index_groups(
  profile: VelocityRobotProfile,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
  """Split ``profile.foot_sites`` into (support, raised) indices.

  Order matters: reward terms index the same sensor frame order the profile
  declares, so this derives the split from names instead of hard-coding it.
  """
  support = tuple(
    index
    for index, name in enumerate(profile.foot_sites)
    if name.upper().startswith("F")
  )
  rear = tuple(
    index
    for index, name in enumerate(profile.foot_sites)
    if not name.upper().startswith("F")
  )
  if not support or not rear:
    raise ValueError(
      f"Handstand needs both front and rear feet, got {profile.foot_sites}"
    )
  return support, rear


def make_handstand_env_cfg(
  profile: VelocityRobotProfile,
  *,
  base_height_target: float,
  rear_clearance_target: float,
  gravity_target: tuple[float, float, float] = (1.0, 0.0, 0.0),
  episode_length_s: float = 10.0,
  joint_scale_range: tuple[float, float] = (0.85, 1.15),
  reset_velocity_range: dict[str, tuple[float, float]] | None = None,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build a flat-ground front-support handstand environment for one robot."""
  cfg = _make_base_env_cfg(profile)
  support_ids, rear_ids = _foot_index_groups(profile)

  cfg.episode_length_s = episode_length_s
  cfg.curriculum = {}
  _configure_handstand_commands(cfg)
  _configure_flat_terrain(cfg)
  _configure_handstand_events(
    cfg,
    joint_scale_range=joint_scale_range,
    reset_velocity_range=reset_velocity_range
    if reset_velocity_range is not None
    else {
      "x": (-0.3, 0.3),
      "y": (-0.3, 0.3),
      "roll": (-0.3, 0.3),
      "pitch": (-0.3, 0.3),
    },
  )
  _configure_handstand_rewards(
    cfg,
    base_height_target=base_height_target,
    rear_clearance_target=rear_clearance_target,
    gravity_target=gravity_target,
    support_ids=support_ids,
    rear_ids=rear_ids,
  )
  cfg.terminations = {
    # Any non-foot contact is a failed handstand, inherited from the velocity
    # task where it guards against belly-flopping. Kept as a termination, and
    # paired with the alive reward so early termination stays unprofitable.
    "illegal_contact": cfg.terminations["illegal_contact"],
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
  }
  _configure_handstand_sim(cfg)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg


def register_handstand_profile(
  profile: VelocityRobotProfile,
  *,
  task_id: str,
  experiment_name: str,
  base_height_target: float,
  rear_clearance_target: float,
  gravity_target: tuple[float, float, float] = (1.0, 0.0, 0.0),
  max_iterations: int = 5_000,
) -> None:
  """Register a train and play task for one robot profile."""
  env_cfg = make_handstand_env_cfg(
    profile,
    base_height_target=base_height_target,
    rear_clearance_target=rear_clearance_target,
    gravity_target=gravity_target,
    play=False,
  )
  play_cfg = make_handstand_env_cfg(
    profile,
    base_height_target=base_height_target,
    rear_clearance_target=rear_clearance_target,
    gravity_target=gravity_target,
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
