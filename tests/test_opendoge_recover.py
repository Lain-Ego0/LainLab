"""Contract tests for the OpenDoge recover-to-walk task.

This task exists to give the walk skill an expert whose training distribution
covers arriving mid-manoeuvre, so the properties that matter are: the walking
observation layout is unchanged, half the batch really does start fallen, and the
contact termination is gone (it would fire on the first step of every fall).
"""

import src.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK_ID = "LainLab-OpenDoge-Recover-Walk"
SHARED_ACTOR_DIM = 48
SHARED_CRITIC_DIM = 72
NUM_ENVS = 128


def _build(num_envs: int = NUM_ENVS) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(TASK_ID)
  cfg.scene.num_envs = num_envs
  return ManagerBasedRlEnv(cfg, device="cpu")


def test_recover_reuses_the_walking_observation_layout() -> None:
  """Same layout as the flat walking task, so one policy can serve both."""
  walk_cfg = load_env_cfg("LainLab-OpenDoge-Flat")
  recover_cfg = load_env_cfg(TASK_ID)
  assert list(recover_cfg.observations["actor"].terms) == list(
    walk_cfg.observations["actor"].terms
  )
  assert list(recover_cfg.observations["critic"].terms) == list(
    walk_cfg.observations["critic"].terms
  )

  env = _build(2)
  try:
    observations, _ = env.reset()
    actor = observations["actor"]
    assert isinstance(actor, torch.Tensor)
    assert actor.shape == (2, SHARED_ACTOR_DIM)
    critic = observations["critic"]
    assert isinstance(critic, torch.Tensor)
    assert critic.shape == (2, SHARED_CRITIC_DIM)
  finally:
    env.close()


def test_recover_drops_the_contact_termination() -> None:
  cfg = load_env_cfg(TASK_ID)
  # A fallen start is in contact with the ground, so `illegal_contact` cannot be
  # a termination here; a dense penalty replaces it.
  assert "illegal_contact" not in cfg.terminations
  assert "nonfoot_contact" in cfg.rewards
  assert cfg.rewards["nonfoot_contact"].weight < 0


def test_recover_batch_starts_from_both_regimes() -> None:
  """Half standing, half fallen -- the whole point of the task."""
  env = _build()
  try:
    env.reset()
    height = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    fallen_share = float((height < 0.14).float().mean())
    assert 0.2 < fallen_share < 0.8, fallen_share
    assert float(height.max()) > 0.13, float(height.max())
    assert float(height.min()) < 0.14, float(height.min())
  finally:
    env.close()


def test_recover_reset_step_is_finite() -> None:
  env = _build(4)
  try:
    env.reset()
    observations, reward, *_ = env.step(torch.zeros((4, 12)))
    actor = observations["actor"]
    assert isinstance(actor, torch.Tensor)
    assert torch.isfinite(actor).all()
    assert torch.isfinite(reward).all()
  finally:
    env.close()
