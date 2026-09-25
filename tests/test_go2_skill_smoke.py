"""MuJoCo Warp reset/step smoke test for completed Go2 skill tasks."""

import pytest
import src.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from src.tasks.robots.go2.skills.shared.contacts import source_vertical_contact


@pytest.fixture(autouse=True)
def _seed_domain_randomization() -> None:
  """Seed before every test in this module.

  These tests assert physical signs after settling (e.g. the vertical contact
  sign for the jump task), and the settled outcome depends on the randomized
  startup state. Without a per-test seed they inherit the RNG stream from
  whatever ran earlier in the session and pass or fail by test order.
  """
  torch.manual_seed(0)


def test_trot_single_environment_reset_step() -> None:
  cfg = load_env_cfg("Unitree-Go2-Trot-Flat")
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    observations, _ = env.reset()
    assert observations["actor"].shape == (1, 470)
    assert observations["critic"].shape == (1, 204)
    result = env.step(torch.zeros((1, 12)))
    observations, reward = result[0], result[1]
    assert observations["actor"].shape == (1, 470)
    assert observations["critic"].shape == (1, 204)
    assert torch.isfinite(observations["actor"]).all()
    assert torch.isfinite(observations["critic"]).all()
    assert torch.isfinite(reward).all()
  finally:
    env.close()


def test_jump_single_environment_reset_step() -> None:
  cfg = load_env_cfg("Unitree-Go2-Jump-Flat")
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    observations, _ = env.reset()
    assert observations["actor"].shape == (1, 470)
    assert observations["critic"].shape == (1, 210)
    result = env.step(torch.zeros((1, 12)))
    observations, reward = result[0], result[1]
    assert observations["actor"].shape == (1, 470)
    assert observations["critic"].shape == (1, 210)
    assert torch.isfinite(observations["actor"]).all()
    assert torch.isfinite(observations["critic"]).all()
    assert torch.isfinite(reward).all()
  finally:
    env.close()


def test_spring_jump_single_environment_reset_step() -> None:
  cfg = load_env_cfg("Unitree-Go2-Spring-Jump-Flat", play=True)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    observations, _ = env.reset()
    assert observations["actor"].shape == (1, 470)
    assert observations["critic"].shape == (1, 195)
    observations, reward, *_ = env.step(torch.zeros((1, 12)))
    assert torch.isfinite(observations["actor"]).all()
    assert torch.isfinite(observations["critic"]).all()
    assert torch.isfinite(reward).all()
  finally:
    env.close()


def test_dreamwaq_single_environment_reset_step() -> None:
  cfg = load_env_cfg("Unitree-Go2-DreamWaQ-Rough", play=True)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    observations, _ = env.reset()
    assert observations["actor"].shape == (1, 45)
    assert observations["critic"].shape == (1, 783)
    assert observations["history"].shape == (1, 225)
    assert observations["velocity"].shape == (1, 3)
    observations, reward, *_ = env.step(torch.zeros((1, 12)))
    assert all(torch.isfinite(value).all() for value in observations.values())
    assert torch.isfinite(reward).all()
  finally:
    env.close()


def test_jump_vertical_contact_sign_after_settling() -> None:
  cfg = load_env_cfg("Unitree-Go2-Jump-Flat", play=True)
  # This asserts the contact-frame sign convention, not robustness to domain
  # randomization, and only a subset of randomized startups settles into a
  # fully loaded stance. Pin the startup randomization off so the test measures
  # what it is named for instead of passing or failing by RNG draw.
  cfg.events = {
    name: term for name, term in cfg.events.items() if term.mode != "startup"
  }
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()
    for _ in range(12):
      env.step(torch.zeros((1, 12)))
    sensor = env.scene["feet_ground_contact"]
    force = sensor.data.force
    assert force is not None
    assert torch.all(force[0, :, 2] < -5.0)
    assert torch.all(source_vertical_contact(sensor, 5.0))
  finally:
    env.close()


def test_rear_stand_single_environment_reset_step() -> None:
  cfg = load_env_cfg("Unitree-Go2-Rear-Stand-Flat")
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    observations, _ = env.reset()
    assert observations["actor"].shape == (1, 45)
    assert observations["critic"].shape == (1, 86)
    result = env.step(torch.zeros((1, 12)))
    observations, reward = result[0], result[1]
    assert observations["actor"].shape == (1, 45)
    assert observations["critic"].shape == (1, 86)
    assert torch.isfinite(observations["actor"]).all()
    assert torch.isfinite(observations["critic"]).all()
    assert torch.isfinite(reward).all()
  finally:
    env.close()


def test_rear_stand_privileged_layout_and_randomization_ranges() -> None:
  cfg = load_env_cfg("Unitree-Go2-Rear-Stand-Flat")
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    observations, _ = env.reset()
    actor = observations["actor"]
    critic = observations["critic"]
    assert torch.equal(critic[:, 3:48], actor)
    domain = critic[:, 48:82]
    assert torch.all((0.2 <= domain[:, 0]) & (domain[:, 0] <= 1.2))
    assert torch.all((-1.0 <= domain[:, 1]) & (domain[:, 1] <= 2.0))
    assert torch.all((-0.05 <= domain[:, 2:5]) & (domain[:, 2:5] <= 0.05))
    assert torch.all((0.9 <= domain[:, 5:17]) & (domain[:, 5:17] <= 1.1))
    assert torch.all((0.9 <= domain[:, 17:29]) & (domain[:, 17:29] <= 1.1))
    assert torch.all((0.003 <= domain[:, 29]) & (domain[:, 29] <= 0.08))
    assert torch.all((0.01 <= domain[:, 30]) & (domain[:, 30] <= 0.1))
    assert torch.all((0.0 <= domain[:, 31]) & (domain[:, 31] <= 0.1))
    assert torch.all((0.0 <= domain[:, 32]) & (domain[:, 32] <= 0.3))
    assert torch.equal(domain[:, 32], domain[:, 33])
    assert torch.all((critic[:, 82:86] == 0.0) | (critic[:, 82:86] == 1.0))
  finally:
    env.close()


def test_handstand_single_environment_reset_step() -> None:
  cfg = load_env_cfg("Unitree-Go2-Handstand-Flat")
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    observations, _ = env.reset()
    assert observations["actor"].shape == (1, 48)
    assert observations["critic"].shape == (1, 89)
    assert torch.equal(observations["actor"][:, :3], torch.zeros((1, 3)))
    result = env.step(torch.zeros((1, 12)))
    observations, reward = result[0], result[1]
    assert observations["actor"].shape == (1, 48)
    assert observations["critic"].shape == (1, 89)
    assert torch.equal(observations["actor"][:, :3], torch.zeros((1, 3)))
    assert torch.isfinite(observations["actor"]).all()
    assert torch.isfinite(observations["critic"]).all()
    assert torch.isfinite(reward).all()
  finally:
    env.close()


def test_handstand_action_reset_delay_and_motor_offset_match_gym() -> None:
  cfg = load_env_cfg("Unitree-Go2-Handstand-Flat", play=True)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()
    action = env.action_manager.get_term("joint_pos")
    delay = action._source_delay
    # Gym's first delayed action after reset interpolates from last_actions=0.
    # The delay history must therefore be zero, not backfilled by the new action.
    assert torch.equal(delay._buffer.current_length, torch.ones(1, dtype=torch.long))
    assert torch.count_nonzero(delay._buffer.buffer) == 0

    # Gym torque target is default + action_scale * action + motor_zero_offset.
    robot = env.scene["robot"]
    robot.data.encoder_bias[:, action.target_ids] = 0.02
    env.action_manager.process_action(torch.zeros((1, 12)))
    delay.set_lags(torch.zeros(1, dtype=torch.long))
    delay._step_count[:] = 1
    action.apply_actions()
    expected = robot.data.default_joint_pos[:, action.target_ids] + 0.02
    assert torch.allclose(robot.data.joint_pos_target[:, action.target_ids], expected)
  finally:
    env.close()


def test_handstand_privileged_randomization_labels() -> None:
  cfg = load_env_cfg("Unitree-Go2-Handstand-Flat")
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    observations, _ = env.reset()
    actor = observations["actor"]
    critic = observations["critic"]
    assert torch.equal(critic[:, 3:51], actor)
    domain = critic[:, 51:85]
    assert torch.all((0.2 <= domain[:, 0]) & (domain[:, 0] <= 1.2))
    assert torch.all((-1.0 <= domain[:, 1]) & (domain[:, 1] <= 2.0))
    assert torch.all((-0.05 <= domain[:, 2:5]) & (domain[:, 2:5] <= 0.05))
    assert torch.all((0.9 <= domain[:, 5:17]) & (domain[:, 5:17] <= 1.1))
    assert torch.all((0.9 <= domain[:, 17:29]) & (domain[:, 17:29] <= 1.1))
    assert torch.all((0.005 <= domain[:, 29]) & (domain[:, 29] <= 0.015))
    assert torch.all((0.01 <= domain[:, 30]) & (domain[:, 30] <= 0.2))
    assert torch.all((0.0 <= domain[:, 31]) & (domain[:, 31] <= 0.2))
    assert torch.all((0.0 <= domain[:, 32]) & (domain[:, 32] <= 0.3))
    assert torch.equal(domain[:, 32], domain[:, 33])
    assert torch.all((critic[:, 85:89] == 0.0) | (critic[:, 85:89] == 1.0))
  finally:
    env.close()
