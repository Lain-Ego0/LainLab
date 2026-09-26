"""Collect expert rollouts for behaviour cloning.

The data is gathered inside the *unified* multi-skill environment so the stored
observations already have the student layout. Each expert is a single-skill
policy whose native observation is a prefix of the unified observation -- same
field order, same command semantics, same noise -- so the expert is driven by
slicing, not by an adapter that could drift.

Usage::

  uv run opendoge-collect --skill walk --samples 400000
  uv run opendoge-collect --skill handstand \
      --checkpoint logs/opendoge_handstand/lainlab_opendoge_handstand/<run>/model_1100.pt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from tensordict import TensorDict

import src.tasks  # noqa: F401
from src.tasks.skills.layout import (
  EXPERT_OBS_DIM,
  SKILL_ONE_HOT_SLICE,
  STUDENT_OBS_DIM,
)
from src.tasks.skills.mdp.command import SKILL_NAMES, SkillCommandTerm

SKILLS_TASK_ID = "LainLab-OpenDoge-Skills-Flat"
DATA_ROOT = Path("logs/skills_data")


@dataclass(frozen=True)
class SkillDataset:
  """One skill's recorded transitions, in the unified student layout."""

  skill: str
  task_id: str
  expert_checkpoint: str
  obs: torch.Tensor
  action: torch.Tensor

  @property
  def samples(self) -> int:
    return int(self.obs.shape[0])


@dataclass(frozen=True)
class ExpertSource:
  """Where the expert policy for one skill comes from."""

  task_id: str
  checkpoint: Path | None


SKILL_SOURCES: dict[str, ExpertSource] = {
  "walk": ExpertSource(
    "LainLab-OpenDoge-Flat", Path("baseline/opendoge/flat/model.pt")
  ),
  "getup": ExpertSource(
    "LainLab-OpenDoge-Getup", Path("baseline/opendoge/getup/model.pt")
  ),
  "handstand": ExpertSource("LainLab-OpenDoge-Handstand", None),
  "jump": ExpertSource("LainLab-OpenDoge-Jump", None),
}


def load_expert_policy(task_id: str, checkpoint: Path, device: str):
  """Load a single-skill actor using the framework's own loading path."""
  cfg = load_env_cfg(task_id, play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    agent_cfg = load_rl_cfg(task_id)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)
  finally:
    env.close()
  return policy


def collect(
  skill: str,
  checkpoint: Path,
  *,
  num_envs: int,
  steps_per_env: int,
  device: str,
) -> SkillDataset:
  source = SKILL_SOURCES[skill]
  policy = load_expert_policy(source.task_id, checkpoint, device)

  cfg = load_env_cfg(SKILLS_TASK_ID)
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device=device)
  term = env.command_manager.get_term("skill")
  assert isinstance(term, SkillCommandTerm), "Expected the unified skill task"
  index = term.skill_index(skill)
  # One-hot weights so every environment is assigned this skill, which also
  # makes the reset distribution follow the skill.
  term.set_skill_weights(
    tuple(1.0 if i == index else 0.0 for i in range(len(term.skill_names)))
  )

  observations, _ = env.reset()
  expert_dim = EXPERT_OBS_DIM[skill]
  obs_chunks: list[torch.Tensor] = []
  action_chunks: list[torch.Tensor] = []
  for _ in range(steps_per_env):
    actor_obs = observations["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    with torch.inference_mode():
      expert_obs = TensorDict(
        {"actor": actor_obs[:, :expert_dim]}, batch_size=[num_envs]
      )
      actions = policy(expert_obs)
    obs_chunks.append(actor_obs.detach().to("cpu", torch.float16))
    action_chunks.append(actions.detach().to("cpu", torch.float16))
    observations, _, _, _, _ = env.step(actions)

  env.close()
  obs = torch.cat(obs_chunks).reshape(-1, obs_chunks[0].shape[-1])
  # Fail loudly on layout drift: a dataset with the wrong width still trains, it
  # just trains the wrong thing (the 48-field slice was used for the jump expert
  # after the twist block was inserted, which would have mislabelled every jump
  # sample).
  if obs.shape[1] != STUDENT_OBS_DIM:
    raise RuntimeError(
      f"unified observation is {obs.shape[1]}-dim, expected {STUDENT_OBS_DIM}; "
      "the student layout changed and EXPERT_OBS_DIM must be updated with it"
    )
  return SkillDataset(
    skill=skill,
    task_id=source.task_id,
    expert_checkpoint=str(checkpoint),
    obs=obs,
    action=torch.cat(action_chunks).reshape(-1, action_chunks[0].shape[-1]),
  )


def collect_handovers(
  target: str,
  checkpoints: dict[str, Path],
  *,
  num_envs: int,
  source_steps: int,
  target_steps: int,
  cycles: int,
  device: str,
) -> SkillDataset:
  """Collect the frames where one skill has to take over from another.

  The behaviour-cloning set alone teaches nothing about handover: every frame is
  collected with the environment pinned to a single skill, so the student only
  ever sees states its own skill produced. A flat shared output layer survives
  that by interpolating one continuous function across skills, but per-skill
  heads have no such continuity -- measured, transitions fell from 0.9389 to
  0.9255 and `handstand->walk` from 0.678 to 0.491 (docs section 5.8.2).

  So this collects the missing frames explicitly. Each environment alternates:

  - ``source_steps`` on a random skill *other* than the target, driven by **that**
    skill's expert, so the state the target inherits is a real state of the source
    behaviour;
  - ``target_steps`` on the target, driven by the target's expert and *recorded*.

  :meth:`SkillCommandTerm.force_skill` sets the skill and resets the skill's own
  state (phase clock, command block) without touching the robot pose, which is
  exactly the handover semantics -- it is the same thing the evaluation protocol
  does between its hold and settle windows.

  This is not the cross-skill DAgger that failed in docs section 5.3.1: DAgger
  labelled states the *student* had drifted into, which are off-manifold for every
  expert. Here the state was just produced by another expert and is on-manifold
  for it.
  """
  experts = {
    skill: load_expert_policy(SKILL_SOURCES[skill].task_id, path, device)
    for skill, path in checkpoints.items()
  }
  missing = [skill for skill in SKILL_NAMES if skill not in experts]
  if missing:
    raise SystemExit(f"handover collection needs every expert; missing {missing}")
  target_index = SKILL_NAMES.index(target)

  cfg = load_env_cfg(SKILLS_TASK_ID)
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device=device)
  term = env.command_manager.get_term("skill")
  assert isinstance(term, SkillCommandTerm), "Expected the unified skill task"
  # Uniform weights: the reset distribution has to follow whichever skill an
  # environment starts on, and we drive every skill explicitly below.
  term.set_skill_weights(tuple(1.0 for _ in SKILL_NAMES))

  ids_all = torch.arange(num_envs, device=device)
  observations, _ = _start_source_phase(
    env, term, ids_all, _sample_sources(num_envs, target_index, device)
  )

  obs_chunks: list[torch.Tensor] = []
  action_chunks: list[torch.Tensor] = []
  for _ in range(cycles):
    for step in range(source_steps + target_steps):
      actor_obs = observations["actor"]
      assert isinstance(actor_obs, torch.Tensor)
      on_target = term.skill == target_index
      with torch.inference_mode():
        actions = _expert_actions(experts, term.skill, actor_obs, device)
      # Record a frame only when the observation *itself* carries the target's
      # skill token, and label it with the target expert's action. Deriving the
      # condition from the token rather than from `term.skill` makes the two
      # agree by construction: for the first step after the switch the
      # observation was computed by the previous `env.step` while the skill was
      # still the source, so it carries the *source* token -- and labelling those
      # frames with the target expert trains the source skill's head on the
      # target's actions. Measured: exactly one such frame per environment
      # (2048 per dataset) drove the walk/validation MSE from 0.003 to 1.40 and
      # the handstand one to 2.6, because a handful of several-radian targets
      # dominate an MSE loss through the shared trunk.
      token = actor_obs[..., SKILL_ONE_HOT_SLICE].argmax(dim=-1)
      record = (token == target_index) & (step >= source_steps)
      if step < source_steps:
        record = torch.zeros_like(on_target)
      if bool(record.any()):
        obs_chunks.append(actor_obs[record].detach().to("cpu", torch.float16))
        action_chunks.append(actions[record].detach().to("cpu", torch.float16))
      observations, _, _, _, _ = env.step(actions)
      if step == source_steps - 1:
        # The handover itself: change the skill *without* resetting, which is
        # what the evaluation protocol does between its hold and settle windows.
        ids = ids_all[term.skill != target_index]
        if len(ids) > 0:
          term.force_skill(target, ids)
      elif step == source_steps + target_steps - 1:
        # Next cycle: put every environment back into a fresh source skill's own
        # reset distribution before rolling it out again.
        observations, _ = _start_source_phase(
          env, term, ids_all, _sample_sources(num_envs, target_index, device)
        )

  env.close()
  if not obs_chunks:
    raise SystemExit("handover collection produced no samples")
  obs = torch.cat(obs_chunks).reshape(-1, STUDENT_OBS_DIM)
  # Every frame must carry the target's token, or the labels below belong to a
  # different head. This is the check whose absence let 2048 poisoned frames
  # through and destroy a clone.
  tokens = obs[:, SKILL_ONE_HOT_SLICE].argmax(dim=-1)
  if int((tokens != target_index).sum()) != 0:
    raise RuntimeError(
      f"handover dataset for {target!r} contains "
      f"{int((tokens != target_index).sum())} frames labelled with another "
      "skill's token; those frames would train the wrong head"
    )
  return SkillDataset(
    skill=target,
    task_id=SKILL_SOURCES[target].task_id,
    expert_checkpoint="handover:"
    + ",".join(f"{skill}={path}" for skill, path in sorted(checkpoints.items())),
    obs=obs,
    action=torch.cat(action_chunks).reshape(-1, 12),
  )


def _sample_sources(num_envs: int, target_index: int, device: str) -> torch.Tensor:
  """A random skill index per environment, never the target."""
  others = [index for index in range(len(SKILL_NAMES)) if index != target_index]
  draw = torch.randint(len(others), (num_envs,), device=device)
  return torch.tensor(others, device=device)[draw]


def _expert_actions(
  experts: dict[str, Any],
  skill_index: torch.Tensor,
  actor_obs: torch.Tensor,
  device: str,
) -> torch.Tensor:
  """Drive every environment with the expert of the skill it is currently on."""
  actions = torch.zeros(
    (actor_obs.shape[0], 12), dtype=actor_obs.dtype, device=actor_obs.device
  )
  for index, skill in enumerate(SKILL_NAMES):
    ids = (skill_index == index).nonzero(as_tuple=True)[0]
    if len(ids) == 0:
      continue
    with torch.inference_mode():
      expert_obs = TensorDict(
        {"actor": actor_obs[ids][:, : EXPERT_OBS_DIM[skill]]},
        batch_size=[len(ids)],
      )
      actions[ids] = experts[skill](expert_obs)
  return actions


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--skill", choices=sorted(SKILL_SOURCES), required=True)
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--output", type=Path, default=None)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--steps-per-env", type=int, default=250)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--handovers",
    action="store_true",
    help=(
      "collect handover frames instead of steady-state frames: environments "
      "alternate between another skill (driven by that skill's expert) and the "
      "target skill, and only the target half is recorded. Needs --expert for "
      "all four skills."
    ),
  )
  parser.add_argument(
    "--expert",
    action="append",
    default=[],
    metavar="SKILL=PATH",
    help="expert checkpoint per skill, repeatable; required by --handovers.",
  )
  parser.add_argument("--source-steps", type=int, default=150)
  parser.add_argument("--target-steps", type=int, default=150)
  parser.add_argument("--cycles", type=int, default=20)
  args = parser.parse_args()

  configure_torch_backends()

  if args.handovers:
    checkpoints: dict[str, Path] = {}
    for item in args.expert:
      if "=" not in item:
        raise SystemExit(f"--expert expects SKILL=PATH, got {item!r}")
      skill, _, path = item.partition("=")
      if skill not in SKILL_SOURCES:
        raise SystemExit(f"unknown skill {skill!r} in --expert")
      if skill == args.skill:
        raise SystemExit(
          "--expert should list the *other* skills; the target "
          "expert is the source of the labels and is loaded from "
          "--checkpoint"
        )
      checkpoints[skill] = Path(path)
    target_checkpoint = args.checkpoint or SKILL_SOURCES[args.skill].checkpoint
    if target_checkpoint is None:
      raise SystemExit(f"--checkpoint is required for skill {args.skill!r}")
    checkpoints[args.skill] = Path(target_checkpoint)
    for skill, path in checkpoints.items():
      if not path.is_file():
        raise SystemExit(f"checkpoint not found for {skill}: {path}")
    dataset = collect_handovers(
      args.skill,
      checkpoints,
      num_envs=args.num_envs,
      source_steps=args.source_steps,
      target_steps=args.target_steps,
      cycles=args.cycles,
      device=args.device,
    )
    output = args.output or (DATA_ROOT / f"{args.skill}_handover.pt")
  else:
    source = SKILL_SOURCES[args.skill]
    checkpoint = args.checkpoint or source.checkpoint
    if checkpoint is None:
      raise SystemExit(f"--checkpoint is required for skill {args.skill!r}")
    if not Path(checkpoint).is_file():
      raise SystemExit(f"checkpoint not found: {checkpoint}")
    dataset = collect(
      args.skill,
      Path(checkpoint),
      num_envs=args.num_envs,
      steps_per_env=args.steps_per_env,
      device=args.device,
    )
    output = args.output or (DATA_ROOT / f"{args.skill}.pt")

  output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(dataset, output)
  summary = {
    "skill": dataset.skill,
    "task_id": dataset.task_id,
    "expert_checkpoint": dataset.expert_checkpoint,
    "samples": dataset.samples,
    "obs_dim": int(dataset.obs.shape[1]),
    "action_dim": int(dataset.action.shape[1]),
    "output": str(output),
  }
  print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()


def _start_source_phase(
  env: ManagerBasedRlEnv,
  term: SkillCommandTerm,
  ids_all: torch.Tensor,
  source_skill: torch.Tensor,
) -> tuple[dict, dict]:
  """Put each environment into its source skill's own reset distribution.

  This cannot be done with `force_skill` alone: that only sets the skill and its
  bookkeeping, leaving the robot in whatever pose it was in. Driving, say, the
  get-up expert from a *standing* pose is off-manifold -- it was trained only
  from fallen resets -- and its wild actions then poison both the source rollout
  and the handover frames that follow. Measured once: cloning on handover data
  collected that way drove the handstand validation MSE from 0.0007 to 2.63 and
  its success rate to 0.08.

  `set_skill_override` pins the skill through the reset event (which otherwise
  samples a skill), then `env.reset(env_ids=...)` applies that skill's harvested
  reset distribution. The override is released immediately afterwards so a later
  mid-episode reset does not drag the environment back to the source skill.
  """
  for skill in SKILL_NAMES:
    mask = source_skill == SKILL_NAMES.index(skill)
    if bool(mask.any()):
      term.set_skill_override(skill, ids_all[mask])
  observations, extras = env.reset(env_ids=ids_all)
  term.set_skill_override(None, ids_all)
  return observations, extras
