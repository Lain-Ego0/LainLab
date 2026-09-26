"""Contract tests for the OpenDoge in-place vertical jump task.

The jump must keep the same 48-field actor observation as the other skills, and
its shaping must not pay for standing still -- both are asserted here because
either one silently breaks the single-policy skill set.
"""

import src.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from src.tasks.jump.mdp.command import JumpCommand

TASK_ID = "LainLab-OpenDoge-Jump"
# 48 shared proprioceptive fields plus the jump's own twist command.
SHARED_ACTOR_DIM = 51
SHARED_CRITIC_DIM = 75


def _tensor(value: object) -> torch.Tensor:
  """Narrow a framework observation/reward value to a tensor for assertions."""
  assert isinstance(value, torch.Tensor)
  return value


def _build(num_envs: int = 2) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(TASK_ID)
  cfg.scene.num_envs = num_envs
  return ManagerBasedRlEnv(cfg, device="cpu")


def test_jump_matches_shared_observation_contract() -> None:
  cfg = load_env_cfg(TASK_ID)
  # The command term is replaced, but the observation term name, position and
  # width must stay identical to the velocity / handstand / get-up tasks.
  assert list(cfg.commands) == ["jump"]
  terms = list(cfg.observations["actor"].terms)
  # `command` stays the last of the shared 48; the twist block follows it, which
  # is what makes the unified observation's first 51 fields this task's own.
  assert terms[-2:] == ["command", "jump_twist"]
  assert cfg.observations["actor"].terms["command"].params["command_name"] == "jump"
  assert "height_scan" not in cfg.observations["actor"].terms

  env = _build(1)
  try:
    observations, _ = env.reset()
    assert _tensor(observations["actor"]).shape == (1, SHARED_ACTOR_DIM)
    assert _tensor(observations["critic"]).shape == (1, SHARED_CRITIC_DIM)
  finally:
    env.close()


def test_jump_reset_step_is_finite() -> None:
  env = _build(2)
  try:
    env.reset()
    observations, reward, *_ = env.step(torch.zeros((2, 12)))
    assert torch.isfinite(_tensor(observations["actor"])).all()
    assert torch.isfinite(_tensor(observations["critic"])).all()
    assert torch.isfinite(reward).all()
  finally:
    env.close()


def test_reset_drop_is_not_scored_as_a_takeoff() -> None:
  """The robot is placed on the ground at reset; that transient is not a jump."""
  env = _build(4)
  try:
    env.reset()
    term = env.command_manager.get_term("jump")
    assert isinstance(term, JumpCommand)
    assert not bool(term.left_ground.any()), term.left_ground
    for _ in range(20):
      env.step(torch.zeros((4, 12)))
    assert not bool(term.left_ground.any()), term.left_ground
  finally:
    env.close()


def test_standing_still_earns_no_jump_reward() -> None:
  """Flight, apex and settle must all be zero while the robot just stands."""
  env = _build(4)
  try:
    env.reset()
    for _ in range(20):
      env.step(torch.zeros((4, 12)))
    names = list(env.reward_manager.active_terms)
    values = dict(
      zip(names, env.reward_manager._step_reward.mean(dim=0).tolist(), strict=True)
    )
    for term in ("flight", "apex_height", "takeoff_simultaneity"):
      assert abs(values[term]) < 1e-6, (term, values[term])
    # Only the survival terms should be paying.
    assert values["alive"] > 0.9
    assert values["upright"] > 0.9
  finally:
    env.close()
