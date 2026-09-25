"""Contract tests for the OpenDoge handstand task.

The handstand only becomes part of the single-policy skill set if it keeps the
same observation layout as the flat velocity task and get-up, so that property
is asserted here rather than assumed.
"""

import pytest
import src.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK_ID = "LainLab-OpenDoge-Handstand"
# Flat velocity and get-up both produce this layout; the handstand must match.
SHARED_ACTOR_DIM = 48
SHARED_CRITIC_DIM = 72


def _tensor(value: object) -> torch.Tensor:
  """Narrow a framework observation/reward value to a tensor for assertions."""
  assert isinstance(value, torch.Tensor)
  return value


def _build(num_envs: int = 2) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(TASK_ID)
  cfg.scene.num_envs = num_envs
  return ManagerBasedRlEnv(cfg, device="cpu")


def test_handstand_matches_shared_observation_contract() -> None:
  cfg = load_env_cfg(TASK_ID)
  assert tuple(cfg.observations) == ("actor", "critic")
  assert "height_scan" not in cfg.observations["actor"].terms
  assert "height_scan" not in cfg.observations["critic"].terms

  env = _build(1)
  try:
    observations, _ = env.reset()
    assert _tensor(observations["actor"]).shape == (1, SHARED_ACTOR_DIM)
    assert _tensor(observations["critic"]).shape == (1, SHARED_CRITIC_DIM)
  finally:
    env.close()


def test_handstand_reset_step_is_finite() -> None:
  env = _build(2)
  try:
    env.reset()
    observations, reward, *_ = env.step(torch.zeros((2, 12)))
    assert torch.isfinite(_tensor(observations["actor"])).all()
    assert torch.isfinite(_tensor(observations["critic"])).all()
    assert torch.isfinite(reward).all()
  finally:
    env.close()


def test_handstand_starts_upright_and_terminates_on_body_contact() -> None:
  env = _build(2)
  try:
    env.reset()
    asset = env.scene["robot"]
    # A handstand must be built from standing, so the reset is near-upright with
    # the base close to the settled standing height.
    gravity = asset.data.projected_gravity_b
    assert torch.all(gravity[:, 2] < -0.9), gravity
    base_height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    assert torch.all((base_height > 0.10) & (base_height < 0.20)), base_height

    terminations = list(env.termination_manager.active_terms)
    assert "illegal_contact" in terminations
    assert "time_out" in terminations
  finally:
    env.close()


def test_alignment_reward_is_dense_across_the_pitch_up() -> None:
  """The alignment reward must not be flat away from the target pose.

  A Gaussian kernel with a small std is numerically ~0 at the standing pose,
  which starves the gradient that drives the pitch-up. The dot-product form is
  asserted to vary monotonically across the whole manoeuvre instead.
  """
  import math
  from types import SimpleNamespace

  from src.tasks.handstand.mdp import rewards as handstand_rewards

  class _StubEnv:
    def __init__(self, gravity: list[float]) -> None:
      asset = SimpleNamespace(
        data=SimpleNamespace(
          projected_gravity_b=torch.tensor([gravity], dtype=torch.float32)
        )
      )
      self.scene = {"robot": asset}
      self.device = "cpu"

  def alignment(pitch_deg: float) -> float:
    # projected_gravity_b for a pitch-only body rotation is [sin, 0, -cos].
    angle = math.radians(pitch_deg)
    env = _StubEnv([math.sin(angle), 0.0, -math.cos(angle)])
    value = handstand_rewards.gravity_alignment(env)  # type: ignore[arg-type]
    return float(value.item())

  assert alignment(0.0) == 0.0
  assert alignment(45.0) == pytest.approx(math.sqrt(0.5), abs=1e-5)
  assert alignment(90.0) == pytest.approx(1.0, abs=1e-5)
  samples = [alignment(deg) for deg in range(0, 91, 10)]
  assert samples == sorted(samples), samples
