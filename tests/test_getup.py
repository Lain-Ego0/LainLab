"""Contract tests for the OpenDoge fall-recovery task."""

import src.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def test_opendoge_getup_is_registered() -> None:
  assert "LainLab-OpenDoge-Getup" in set(list_tasks())


def test_opendoge_getup_config() -> None:
  cfg = load_env_cfg("LainLab-OpenDoge-Getup")
  rl = load_rl_cfg("LainLab-OpenDoge-Getup")

  assert rl.max_iterations == 5_000
  assert cfg.episode_length_s == 4.0
  assert cfg.curriculum == {}
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  assert command.ranges.lin_vel_x == (0.0, 0.0)
  assert command.ranges.lin_vel_y == (0.0, 0.0)
  assert command.ranges.ang_vel_z == (0.0, 0.0)
  assert command.ranges.heading is None

  assert cfg.scene.terrain is not None
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  assert generator.curriculum is False
  assert generator.difficulty_range == (0.0, 0.6)

  assert set(cfg.events) == {
    "randomize_terrain",
    "reset_base",
    "reset_robot_joints",
    "foot_friction",
    "encoder_bias",
    "base_com",
  }
  assert cfg.events["reset_base"].func.__name__ == "reset_fallen_root"
  assert cfg.events["reset_robot_joints"].func.__name__ == "reset_fallen_joints"

  assert set(cfg.rewards) == {
    "upright",
    "base_height",
    "torques",
    "dof_vel",
    "joint_power",
    "action_rate",
    "dof_acc",
    "ang_vel_xy",
    "impact",
    "default_pos",
    "termination",
    "dof_pos_limits",
  }
