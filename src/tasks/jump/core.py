"""Shared in-place vertical jump task builder and registration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from src.tasks.jump.mdp import command as jump_command
from src.tasks.jump.mdp import rewards as jump_rewards
from src.tasks.rl import make_ppo_runner_cfg
from src.tasks.velocity.core import VelocityRobotProfile, _make_base_env_cfg

_DEFAULT_RESET_VELOCITY_RANGE = {
  "x": (-0.2, 0.2),
  "y": (-0.2, 0.2),
  "z": (-0.1, 0.1),
  "roll": (-0.2, 0.2),
  "pitch": (-0.2, 0.2),
}


def _configure_flat_terrain(cfg: ManagerBasedRlEnvCfg) -> None:
  """Force flat ground and drop the exteroceptive scan to keep the 48-field obs."""
  terrain = cfg.scene.terrain
  if terrain is None:
    raise ValueError("Jump requires a terrain entity")
  terrain.terrain_type = "plane"
  terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)


def _configure_jump_command(
  cfg: ManagerBasedRlEnvCfg, *, period_s: float, standing_height: float
) -> None:
  """Swap the twist command for the jump phase clock.

  The observation term keeps the name ``command`` and stays 3 wide, so the jump
  task ends up with exactly the same 48-field actor observation as walking,
  handstand and get-up; only the meaning of those three fields changes.
  """
  cfg.commands = {
    "jump": jump_command.JumpCommandCfg(
      entity_name="robot",
      contact_sensor_name="feet_ground_contact",
      resampling_time_range=(1.0e9, 1.0e9),
      period_s=period_s,
      standing_height=standing_height,
    )
  }
  for group in ("actor", "critic"):
    term = cfg.observations[group].terms["command"]
    term.params = {"command_name": "jump"}


def _configure_jump_rewards(
  cfg: ManagerBasedRlEnvCfg, *, standing_height: float, target_rise: float
) -> None:
  cfg.rewards = {
    # Task shaping. See `mdp/rewards.py` for why standing still earns nothing.
    "flight": RewardTermCfg(func=jump_rewards.flight, weight=6.0),
    "apex_height": RewardTermCfg(
      func=jump_rewards.apex_height,
      weight=3.0,
      params={
        "standing_height": standing_height,
        "target_rise": target_rise,
        "tolerance": 0.035,
      },
    ),
    "settle": RewardTermCfg(
      func=jump_rewards.settle,
      weight=2.0,
      params={"standing_height": standing_height, "tolerance": 0.06},
    ),
    "upright": RewardTermCfg(func=jump_rewards.upright, weight=1.0),
    "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=1.0),
    "failure": RewardTermCfg(func=jump_rewards.jump_failure, weight=-5.0),
    # Regularization.
    "soft_landing": RewardTermCfg(func=jump_rewards.soft_landing, weight=-0.5),
    "planar_vel": RewardTermCfg(func=jump_rewards.planar_velocity_penalty, weight=-0.2),
    "torques": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-0.0005),
    "dof_vel": RewardTermCfg(func=envs_mdp.joint_vel_l2, weight=-0.002),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.02),
    "dof_acc": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-1.5e-7),
    "dof_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-1.0),
  }


def make_jump_env_cfg(
  profile: VelocityRobotProfile,
  *,
  standing_height: float,
  target_rise: float = 0.05,
  period_s: float = 2.5,
  episode_length_s: float = 7.5,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build a flat-ground in-place vertical jump environment for one robot."""
  cfg = _make_base_env_cfg(profile)

  cfg.episode_length_s = episode_length_s
  cfg.curriculum = {}
  _configure_flat_terrain(cfg)
  _configure_jump_command(cfg, period_s=period_s, standing_height=standing_height)
  _configure_jump_rewards(cfg, standing_height=standing_height, target_rise=target_rise)
  # Everything else (friction/com/gain randomization, push_robot) stays as the
  # shared velocity environment set it up; only the reset changes.
  cfg.events["reset_base"].params["velocity_range"] = dict(
    _DEFAULT_RESET_VELOCITY_RANGE
  )
  # Start the robot already touching the ground. The shared velocity profile
  # drops it from 4-18 mm up, which would pre-load `peak_height` above the
  # standing height and hand out free apex reward before anything happens.
  cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.003)
  cfg.terminations = {
    # Base contact still ends an episode (falling over is a failure), but nothing
    # else is added: takeoff and landing legitimately slam the legs down, which
    # is what killed Go2's backflip attempt (see docs/go2_migration_matrix.md).
    "illegal_contact": cfg.terminations["illegal_contact"],
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
  }
  cfg.sim.nconmax = 64
  cfg.sim.njmax = 600
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.mujoco.ccd_iterations = 200

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg


def register_jump_profile(
  profile: VelocityRobotProfile,
  *,
  task_id: str,
  experiment_name: str,
  standing_height: float,
  target_rise: float = 0.05,
  period_s: float = 2.5,
  max_iterations: int = 5_000,
) -> None:
  """Register a train and play task for one robot profile."""
  env_cfg = make_jump_env_cfg(
    profile,
    standing_height=standing_height,
    target_rise=target_rise,
    period_s=period_s,
    play=False,
  )
  play_cfg = make_jump_env_cfg(
    profile,
    standing_height=standing_height,
    target_rise=target_rise,
    period_s=period_s,
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
