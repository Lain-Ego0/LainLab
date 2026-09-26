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
from src.tasks.jump.mdp.state import JumpState, JumpStateCfg

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


def test_takeoff_survives_a_foot_landing_while_others_are_airborne() -> None:
  """A staggered takeoff must still be timed, and timed honestly.

  Clearing every foot's liftoff record whenever *any* foot lands deletes the
  record of feet that are still in the air. A takeoff is then only registered
  when all four lift together from a grounded stance, which reported a spread of
  exactly 0 in every case (`_update_takeoff` docstring has the measurements).
  """
  state = JumpState(1, "cpu", JumpStateCfg(period_s=1000.0, standing_height=0.151))
  true, false = True, False

  def step(contacts: tuple[bool, bool, bool, bool], index: int) -> None:
    contact = torch.tensor([contacts])
    state.update(
      None,
      dt=0.01,
      base_height=torch.tensor([0.151]),
      airborne=~contact.any(dim=-1),
      feet_contact=contact,
      step_index=torch.tensor([index]),
    )

  step((true, true, true, true), 0)  # grounded, records nothing
  step((false, true, true, true), 1)  # FR up
  step((false, false, true, true), 2)  # FL up
  step((false, false, false, false), 3)  # RR+RL up: takeoff complete
  assert bool(state.takeoff_complete[0])
  assert bool(state.takeoff_edge[0])
  assert float(state.takeoff_spread[0]) == 3.0 - 1.0

  step((true, false, false, false), 4)  # FR lands, the other three stay up
  # Second takeoff: FR pushes off again while the others are still airborne.
  step((false, false, false, false), 5)
  assert bool(state.takeoff_complete[0]), "staggered takeoff was not registered"
  assert float(state.takeoff_spread[0]) == 5.0 - 2.0


def test_flight_gate_only_spends_on_a_height_qualified_takeoff() -> None:
  """The one-flight budget belongs to the jump, not to the gait's suspension.

  A travel gait lifts all four feet without raising the base, so an
  "airborne inside the window" rule lets those hops spend the budget before the
  real jump: measured reward 0.0037 and peak rise down to 0.043 m.
  """
  cfg = JumpStateCfg(period_s=1000.0, standing_height=0.151, flight_window=(0.10, 0.22))
  state = JumpState(1, "cpu", cfg)
  margin_height = cfg.standing_height + cfg.takeoff_margin + 0.001

  def step(
    contacts: tuple[bool, bool, bool, bool], height: float, phase: float
  ) -> None:
    state.phase[0] = phase
    contact = torch.tensor([contacts])
    state.update(
      None,
      dt=0.01,
      base_height=torch.tensor([height]),
      airborne=~contact.any(dim=-1),
      feet_contact=contact,
      step_index=torch.tensor([0]),
    )

  grounded = (True, True, True, True)
  airborne = (False, False, False, False)
  in_window = 0.15

  step(grounded, cfg.standing_height, 0.0)
  # A gait hop inside the window: airborne, but the base never rises.
  step(airborne, cfg.standing_height + 0.004, in_window)
  step(grounded, cfg.standing_height, in_window)
  assert not bool(state.flight_used[0]), "a gait hop spent the jump budget"
  assert float(state.first_flight_gate()[0]) == 1.0

  # The real jump, also inside the window.
  step(airborne, margin_height, in_window)
  step(grounded, cfg.standing_height, in_window)
  assert bool(state.flight_used[0])
  assert float(state.first_flight_gate()[0]) == 0.0

  # A second jump in the same cycle does not earn a second budget.
  step(airborne, margin_height, in_window)
  step(grounded, cfg.standing_height, in_window)
  assert float(state.first_flight_gate()[0]) == 0.0

  # ...but the next cycle reopens it: park the clock just before the wrap so the
  # next update rolls over.
  step(grounded, cfg.standing_height, 0.99999)
  assert float(state.first_flight_gate()[0]) == 1.0
  step(airborne, margin_height, in_window)
  step(grounded, cfg.standing_height, in_window)
  assert float(state.first_flight_gate()[0]) == 0.0
