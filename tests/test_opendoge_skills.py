"""Contract tests for the single-policy multi-skill OpenDoge task.

These assert the properties the whole single-policy design rests on:

- the shared proprioceptive block is preserved, with the skill block appended
- every skill is actually sampled and gets a skill-specific reset distribution
- each sub-task's rewards fire only for their own skill
- get-up is exempt from the body-contact termination, the others are not
"""

import src.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from src.tasks.skills.layout import (
  EXPERT_OBS_DIM,
  SHARED_OBS_DIM,
  SKILL_ONE_HOT_SLICE,
  STUDENT_OBS_DIM,
)
from src.tasks.skills.mdp.command import (
  SKILL_NAMES,
  SkillCommandCfg,
  SkillCommandTerm,
)

TASK_ID = "LainLab-OpenDoge-Skills-Flat"
ACTOR_PROPRIO_DIM = 48  # includes the 3-wide `command` block
CRITIC_PROPRIO_DIM = 72  # 48 shared fields plus foot height/contact terms
TWIST_BLOCK_DIM = 3  # the jump's commanded [vx, vy, wz]
SKILL_BLOCK_DIM = len(SKILL_NAMES) + 2  # one-hot plus phase sin/cos
NUM_ENVS = 64


def _tensor(value: object) -> torch.Tensor:
  """Narrow a framework observation/reward value to a tensor for assertions."""
  assert isinstance(value, torch.Tensor)
  return value


def _build(num_envs: int = NUM_ENVS) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(TASK_ID)
  cfg.scene.num_envs = num_envs
  return ManagerBasedRlEnv(cfg, device="cpu")


def _skill_term(env: ManagerBasedRlEnv) -> SkillCommandTerm:
  term = env.command_manager.get_term("skill")
  assert isinstance(term, SkillCommandTerm)
  return term


def test_observation_extends_the_shared_proprioceptive_block() -> None:
  cfg = load_env_cfg(TASK_ID)
  terms = list(cfg.observations["actor"].terms)
  # The proprioceptive block must stay in the single-skill order, with the
  # command block still last among the shared fields.
  assert terms[:7] == [
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
  ]
  # The order is what keeps every expert driveable by slicing: the shared 48
  # fields first, then the jump's twist block, then the skill identity block.
  assert terms[7] == "jump_twist"
  assert terms[8] == "skill"
  assert "height_scan" not in terms

  env = _build(2)
  try:
    observations, _ = env.reset()
    assert _tensor(observations["actor"]).shape == (
      2,
      ACTOR_PROPRIO_DIM + TWIST_BLOCK_DIM + SKILL_BLOCK_DIM,
    )
    assert _tensor(observations["critic"]).shape == (
      2,
      CRITIC_PROPRIO_DIM + TWIST_BLOCK_DIM + SKILL_BLOCK_DIM,
    )
  finally:
    env.close()


def test_expert_observation_widths_match_the_unified_layout() -> None:
  """The slicing contract the whole clone pipeline rests on.

  Each expert's native observation must be a *prefix* of the unified actor
  observation, in the same field order. If a term is inserted before an
  expert's block, collecting that skill still runs -- it just labels every
  sample with an observation the expert never saw. The width constants therefore
  have to be checked against the environments, not trusted.
  """
  from mjlab.tasks.registry import load_env_cfg as load

  unified = list(load(TASK_ID).observations["actor"].terms)
  assert len(unified) == 9
  assert ACTOR_PROPRIO_DIM == SHARED_OBS_DIM
  assert SHARED_OBS_DIM + TWIST_BLOCK_DIM + SKILL_BLOCK_DIM == STUDENT_OBS_DIM

  # Walk, get-up and handstand read exactly the shared prefix.
  for task_id in (
    "LainLab-OpenDoge-Flat",
    "LainLab-OpenDoge-Getup",
    "LainLab-OpenDoge-Handstand",
  ):
    terms = list(load(task_id, play=True).observations["actor"].terms)
    assert terms == unified[: len(terms)], task_id

  # The jump expert reads the shared prefix plus the twist block right after it.
  jump_terms = list(
    load("LainLab-OpenDoge-Jump", play=True).observations["actor"].terms
  )
  assert jump_terms == unified[: len(jump_terms)]
  assert jump_terms[-1] == "jump_twist"

  # Term *names* alone do not pin the widths, so measure the built environments
  # too: the collector's constants are what actually get sliced.
  for task_id, expected in (
    ("LainLab-OpenDoge-Flat", EXPERT_OBS_DIM["walk"]),
    ("LainLab-OpenDoge-Jump", EXPERT_OBS_DIM["jump"]),
  ):
    cfg = load(task_id, play=True)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
      observations, _ = env.reset()
      assert _tensor(observations["actor"]).shape[1] == expected, task_id
    finally:
      env.close()


def test_every_skill_is_sampled_and_reset_is_skill_specific() -> None:
  env = _build()
  try:
    env.reset()
    term = _skill_term(env)
    present = {name: int((term.skill == i).sum()) for i, name in enumerate(SKILL_NAMES)}
    assert all(count > 0 for count in present.values()), present

    asset = env.scene["robot"]
    height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    getup = term.skill_is("getup")
    # Get-up must start fallen, everything else must start standing.
    assert float(height[getup].max()) < 0.14, height[getup]
    assert float(height[~getup].min()) > 0.13, height[~getup]
  finally:
    env.close()


def test_rewards_are_masked_per_skill() -> None:
  env = _build()
  try:
    env.reset()
    for _ in range(5):
      env.step(torch.zeros((NUM_ENVS, 12)))
    term = _skill_term(env)
    names = list(env.reward_manager.active_terms)
    values = env.reward_manager._step_reward
    for index, name in enumerate(SKILL_NAMES):
      own = term.skill == index
      assert bool(own.any())
      for term_index, term_name in enumerate(names):
        column = values[:, term_index]
        owner = term_name.split("/", 1)[0]
        if owner == name:
          continue
        # A term belonging to another skill must be exactly zero here.
        assert torch.all(column[own] == 0.0), (name, term_name, column[own])
  finally:
    env.close()


def test_command_block_is_skill_specific() -> None:
  """The command block must mean exactly one thing per skill.

  A regression guard: an earlier version sampled a walking velocity command for
  *every* environment at reset, so handstand and get-up environments observed a
  phantom velocity command where their experts expect zeros. That silently cost
  the get-up expert ~75 points of success rate inside the unified environment.
  """
  cfg = load_env_cfg(TASK_ID)
  command_cfg = cfg.commands["skill"]
  assert isinstance(command_cfg, SkillCommandCfg)
  ranges = command_cfg.walk_command_ranges
  env = _build()
  try:
    env.reset()
    term = _skill_term(env)
    for index, name in enumerate(SKILL_NAMES):
      own = term.skill == index
      if not bool(own.any()):
        continue
      command = term.command[own]
      if name == "walk":
        # Nonzero commands must respect the walking ranges.
        for axis, (low, high) in enumerate(ranges):
          column = command[:, axis]
          assert float(column.min()) >= low - 1e-6, (name, axis, column)
          assert float(column.max()) <= high + 1e-6, (name, axis, column)
      elif name in ("handstand", "getup"):
        assert torch.all(command == 0.0), (name, command)
      elif name == "jump":
        # The jump task's command block is the phase clock plus the takeoff
        # flag: [phase_sin, phase_cos, left_ground]. It must be a unit vector in
        # the first two slots, and the flag must start clear.
        assert torch.allclose(
          command[:, 0] ** 2 + command[:, 1] ** 2,
          torch.ones_like(command[:, 0]),
          atol=1e-4,
        ), command
        assert torch.all(command[:, 2] == 0.0), command
  finally:
    env.close()


def test_skill_override_survives_resets() -> None:
  """The viewer's skill picker must stick across resets.

  `force_skill` sets the current skill only; the reset event re-samples and would
  undo it. `set_skill_override` is the sticky variant the Viser GUI uses, so the
  viewer's Reset button can restart the chosen skill from its own reset
  distribution rather than losing the selection.
  """
  env = _build(32)
  try:
    env.reset()
    term = _skill_term(env)

    # A plain force is undone by the next reset.
    term.force_skill("jump")
    assert bool(term.skill_is("jump").all())
    env.reset()
    assert not bool(term.skill_is("jump").all()), "force_skill unexpectedly sticky"

    # An override is not.
    term.set_skill_override("handstand")
    assert bool(term.skill_is("handstand").all())
    for _ in range(3):
      env.reset()
      assert bool(term.skill_is("handstand").all())

    # A per-environment override pins only that environment, and leaves the rest
    # sampling freely (so some of them will also draw get-up by chance -- the
    # check is that they are free, not that they differ).
    term.set_skill_override(None)
    env.reset()
    term.set_skill_override("getup", torch.tensor([0]))
    for _ in range(3):
      env.reset()
      assert bool(term.skill_is("getup")[0])
    other_counts: list[int] = []
    for _ in range(4):
      env.reset()
      assert bool(term.skill_is("getup")[0])
      other_counts.append(int(term.skill[1:].eq(term.skill_index("getup")).sum()))
    assert len(set(other_counts)) > 1, f"other environments look pinned: {other_counts}"

    # Releasing restores ordinary sampling.
    term.set_skill_override(None)
    seen: set[int] = set()
    for _ in range(4):
      env.reset()
      seen |= set(term.skill.tolist())
    assert len(seen) >= 2, seen
  finally:
    env.close()


def test_contact_termination_is_disabled_only_for_getup() -> None:
  cfg = load_env_cfg(TASK_ID)
  assert list(cfg.terminations) == ["contact", "time_out"]
  params = cfg.terminations["contact"].params
  assert "getup" not in params["allowed_skills"]
  assert set(params["allowed_skills"]) == {"walk", "handstand", "jump"}


def test_jump_command_block_reports_takeoff() -> None:
  env = _build(2)
  try:
    env.reset()
    term = _skill_term(env)
    # Force both environments onto the jump skill: the command block must become
    # [phase_sin, phase_cos, left_ground] rather than a velocity command.
    ids = torch.arange(2)
    term.skill[:] = term.skill_index("jump")
    term._reset_skill_state(ids)  # noqa: SLF001
    term._update_command(ids)  # noqa: SLF001
    assert torch.allclose(
      term.command[:, 0] ** 2 + term.command[:, 1] ** 2,
      torch.ones(2),
      atol=1e-4,
    )
    assert torch.all(term.command[:, 2] == 0.0)
    # After the latch is set, the third slot reports the takeoff.
    term.left_ground[:] = True
    term._update_command(ids)  # noqa: SLF001
    assert torch.all(term.command[:, 2] == 1.0)
  finally:
    env.close()


def test_bc_artifact_loads_into_the_policy_the_evaluator_builds() -> None:
  """The clone and the evaluator must build the *same* network.

  `opendoge-bc` constructs the student itself, while `opendoge-eval` gets it from
  the runner, which resolves `RslRlModelCfg.class_name`. If those two disagree --
  a per-skill head added on one side only, a renamed module -- the checkpoint
  either fails to load or, worse, loads with a silently different architecture.
  This pins the contract the whole delivery rests on.
  """
  import tempfile
  from dataclasses import asdict
  from pathlib import Path

  from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
  from src.skills.bc import build_student

  env = _build(2)
  try:
    agent_cfg = load_rl_cfg(TASK_ID)
    assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
    # Deliberately no assertion about *which* architecture is registered: the
    # contract under test is that whatever `class_name` names can be built by
    # both sides and load the artifact.
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID)
    assert runner_cls is not None
    runner = runner_cls(wrapped, asdict(agent_cfg), device="cpu")
    actor = runner.alg.actor

    student = build_student("cpu")
    student_keys = set(student.state_dict())
    actor_keys = set(actor.state_dict())
    assert student_keys == actor_keys, student_keys ^ actor_keys

    # End to end: the artifact a clone produces loads into that policy.
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "student.pt"
      # Exactly the artifact `opendoge-bc` writes (actor only, no critic).
      torch.save(
        {"actor_state_dict": student.state_dict(), "infos": {}, "iter": 0},
        path,
      )
      loaded = runner.load(
        str(path), load_cfg={"actor": True}, strict=True, map_location="cpu"
      )
      assert loaded is not None
  finally:
    env.close()


def test_skill_residuals_start_as_a_no_op() -> None:
  """The registered architecture must *start* as the flat single-head policy.

  This is the whole reason for the residual form: independent per-skill heads
  measured worse on handover (aggregate 0.9255 vs 0.9389, `handstand->walk`
  0.491 vs 0.678), so the default has to begin from the flat behaviour and earn
  any specialisation from data. A non-zero residual init would silently give
  that up while still passing every other test.
  """
  from src.tasks.skills.rl import SharedResidualActor

  student = _experimental_actor(SharedResidualActor)
  state = student.state_dict()
  assert student.num_skills == len(SKILL_NAMES)
  assert any(key.startswith("mlp.") for key in state), "the shared trunk is gone"
  assert any(key.startswith("head.") for key in state), "the shared head is gone"
  residual_keys = [key for key in state if key.startswith("residuals.")]
  assert len(residual_keys) == 2 * len(SKILL_NAMES), residual_keys
  for key in residual_keys:
    assert float(state[key].abs().max()) == 0.0, f"{key} is not zero-initialised"

  # With the residuals at zero, every skill must produce the shared head's
  # output; the residual is the only per-skill part.
  obs = torch.zeros(len(SKILL_NAMES), STUDENT_OBS_DIM)
  obs[:, 0] = 0.3
  for index in range(len(SKILL_NAMES)):
    obs[index, SKILL_ONE_HOT_SLICE.start + index] = 1.0
  with torch.no_grad():
    actions = _tensor(student(_tensor_dict(obs)))
  assert student.skill_index(obs).tolist() == list(range(len(SKILL_NAMES)))
  assert actions.shape == (len(SKILL_NAMES), 12)

  # Identical observations, different skill token -> different actions. This is
  # the property a single shared output layer cannot have.
  obs = torch.zeros(len(SKILL_NAMES), STUDENT_OBS_DIM)
  obs[:, 0] = 0.3  # a joint position, so the trunk output is non-trivial
  for index in range(len(SKILL_NAMES)):
    obs[index, SKILL_ONE_HOT_SLICE.start + index] = 1.0
  with torch.no_grad():
    actions = (
      student({"actor": obs})
      if isinstance(student, dict)
      else student.__call__(_tensor_dict(obs))
    )
  actions = _tensor(actions)
  assert actions.shape == (len(SKILL_NAMES), 12)
  # Every pair of rows differs (the heads are independent), and the point of the
  # change: the one-hot actually selects.
  first = actions[0]
  assert not torch.allclose(first, actions[1])
  assert student.skill_index(obs).tolist() == list(range(len(SKILL_NAMES)))


def _experimental_actor(actor_class):
  """Build one of the non-default architectures with the student's settings."""
  from src.skills.bc import ACTION_DIM, HIDDEN_DIMS, OBS_DIM
  from tensordict import TensorDict

  return actor_class(
    TensorDict({"actor": torch.zeros(1, OBS_DIM)}, batch_size=[1]),
    {"actor": ["actor"]},
    "actor",
    ACTION_DIM,
    list(HIDDEN_DIMS),
    "elu",
    True,
    {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
  )


def _tensor_dict(obs: torch.Tensor):
  from tensordict import TensorDict

  return TensorDict({"actor": obs}, batch_size=[obs.shape[0]])
