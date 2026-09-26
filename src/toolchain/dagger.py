"""DAgger rounds for the multi-skill student.

Plain behaviour cloning fails on get-up even when validation error looks tiny
(2 degrees per step): the student drifts off the expert's state distribution and
the error compounds until the rise fails. DAgger fixes exactly that by labelling
the states the *student* visits with the expert's action.

Each round:

1. roll out in the unified environment, acting with the student and falling back
   to the expert with probability ``expert_prob`` to keep trajectories near the
   expert manifold,
2. always record the expert's action at the visited state,
3. accumulate those labels with the original expert data and retrain.

Usage::

  uv run opendoge-dagger --student logs/skills_data/student.pt \\
      --expert-data logs/skills_data/walk.pt logs/skills_data/getup.pt \\
      --rounds 2 --steps-per-env 200
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from tensordict import TensorDict

import src.tasks  # noqa: F401
from src.tasks.multiskill.mdp.command import SkillCommandTerm
from src.toolchain.bc import ACTION_DIM, OBS_DIM, build_student, train
from src.toolchain.collect import (
  EXPERT_OBS_DIM,
  SKILL_SOURCES,
  SKILLS_TASK_ID,
  SkillDataset,
  load_expert_policy,
)


@dataclass(frozen=True)
class ExpertSpec:
  """Expert checkpoint for one skill on this DAgger run."""

  skill: str
  checkpoint: Path


def _student_action(model, obs: torch.Tensor, device: str) -> torch.Tensor:
  batch = TensorDict(
    {"actor": obs.to(device, torch.float32)}, batch_size=[obs.shape[0]]
  )
  with torch.inference_mode():
    return model(batch)


def _expert_action(policy, obs: torch.Tensor, skill: str, device: str) -> torch.Tensor:
  batch = TensorDict(
    {"actor": obs[:, : EXPERT_OBS_DIM[skill]].to(device, torch.float32)},
    batch_size=[obs.shape[0]],
  )
  with torch.inference_mode():
    return policy(batch)


def collect_round(
  spec: ExpertSpec,
  student,
  expert_policy,
  *,
  num_envs: int,
  steps_per_env: int,
  expert_prob: float,
  device: str,
) -> SkillDataset:
  """Roll out one skill, acting on-policy and labelling with the expert."""
  cfg = load_env_cfg(SKILLS_TASK_ID)
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device=device)
  term = env.command_manager.get_term("skill")
  assert isinstance(term, SkillCommandTerm)
  index = term.skill_index(spec.skill)
  term.set_skill_weights(
    tuple(1.0 if i == index else 0.0 for i in range(len(term.skill_names)))
  )

  observations, _ = env.reset()
  obs_chunks: list[torch.Tensor] = []
  label_chunks: list[torch.Tensor] = []
  for _ in range(steps_per_env):
    actor_obs = observations["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    label = _expert_action(expert_policy, actor_obs, spec.skill, device)
    if expert_prob > 0.0:
      use_expert = torch.rand(num_envs, device=device) < expert_prob
      student_act = _student_action(student, actor_obs, device)
      action = torch.where(use_expert[:, None], label, student_act)
    else:
      action = _student_action(student, actor_obs, device)
    obs_chunks.append(actor_obs.detach().to("cpu", torch.float16))
    label_chunks.append(label.detach().to("cpu", torch.float16))
    observations, _, _, _, _ = env.step(action)

  env.close()
  return SkillDataset(
    skill=spec.skill,
    task_id=SKILL_SOURCES[spec.skill].task_id,
    expert_checkpoint=str(spec.checkpoint),
    obs=torch.cat(obs_chunks).reshape(-1, OBS_DIM),
    action=torch.cat(label_chunks).reshape(-1, ACTION_DIM),
  )


def collect_transitions(
  experts: dict[str, object],
  student,
  *,
  skills: list[str],
  num_envs: int,
  hold_steps: int,
  collect_steps: int,
  device: str,
) -> dict[str, SkillDataset]:
  """Collect *approach* states for every ordered skill pair.

  Cloning only ever sees each expert on its own reset distribution, so nothing
  in the data covers "arriving at this skill from another one" -- which is
  exactly where the first single-policy attempt was weak (jump -> handstand
  scored 0.36 while every skill scored > 0.95 on its own). This walks every
  ordered pair: hold the source skill long enough to establish it, switch to the
  target, and record the target expert's action on the states the student
  actually visits during that approach.

  Acting with the student (rather than the expert) is deliberate, and is the
  difference between this and simply recording more expert rollouts.
  """
  cfg = load_env_cfg(SKILLS_TASK_ID)
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device=device)
  term = env.command_manager.get_term("skill")
  assert isinstance(term, SkillCommandTerm)
  ids = torch.arange(num_envs, device=device)

  collected: dict[str, list[SkillDataset]] = {skill: [] for skill in skills}
  pairs = [(a, b) for a in skills for b in skills if a != b]
  for source, target in pairs:
    observation, _ = env.reset()
    term.force_skill(source, ids)
    for _ in range(hold_steps):
      hold_obs = observation["actor"]
      assert isinstance(hold_obs, torch.Tensor)
      with torch.no_grad():
        observation, _, _, _, _ = env.step(_student_action(student, hold_obs, device))
    term.force_skill(target, ids)
    obs_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    for _ in range(collect_steps):
      actor_obs = observation["actor"]
      assert isinstance(actor_obs, torch.Tensor)
      label = _expert_action(experts[target], actor_obs, target, device)
      obs_chunks.append(actor_obs.detach().to("cpu", torch.float16))
      label_chunks.append(label.detach().to("cpu", torch.float16))
      with torch.no_grad():
        observation, _, _, _, _ = env.step(_student_action(student, actor_obs, device))
    collected[target].append(
      SkillDataset(
        skill=target,
        task_id=SKILL_SOURCES[target].task_id,
        expert_checkpoint=f"transition:{source}->{target}",
        obs=torch.cat(obs_chunks).reshape(-1, OBS_DIM),
        action=torch.cat(label_chunks).reshape(-1, ACTION_DIM),
      )
    )
  env.close()
  merged: dict[str, SkillDataset] = {}
  for skill, datasets in collected.items():
    if not datasets:
      continue
    merged[skill] = SkillDataset(
      skill=skill,
      task_id=SKILL_SOURCES[skill].task_id,
      expert_checkpoint="transition-aggregate",
      obs=torch.cat([d.obs for d in datasets]),
      action=torch.cat([d.action for d in datasets]),
    )
  return merged


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--student", type=Path, required=True)
  parser.add_argument(
    "--expert-data",
    type=Path,
    nargs="+",
    required=True,
    help="original per-skill datasets to keep in the training pool",
  )
  parser.add_argument(
    "--skill", nargs="+", default=["walk", "handstand", "getup", "jump"]
  )
  parser.add_argument(
    "--expert",
    action="append",
    default=[],
    metavar="SKILL=PATH",
    help="override the expert checkpoint for one skill (repeatable)",
  )
  parser.add_argument("--rounds", type=int, default=2)
  parser.add_argument(
    "--include-transitions",
    action="store_true",
    help="also collect every ordered skill-pair approach state (transition DAgger)",
  )
  parser.add_argument("--hold-steps", type=int, default=200)
  parser.add_argument("--collect-steps", type=int, default=150)
  parser.add_argument("--steps-per-env", type=int, default=200)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--expert-prob", type=float, default=0.5)
  parser.add_argument("--epochs", type=int, default=25)
  parser.add_argument("--batch-size", type=int, default=8192)
  parser.add_argument("--output", type=Path, default=None)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()

  configure_torch_backends()
  output = args.output or args.student
  overrides: dict[str, Path] = {}
  for entry in args.expert:
    if "=" not in entry:
      raise SystemExit(f"--expert expects SKILL=PATH, got {entry!r}")
    skill, path = entry.split("=", 1)
    overrides[skill] = Path(path)

  def expert_path(skill: str) -> Path:
    override = overrides.get(skill)
    if override is not None:
      return override
    default = SKILL_SOURCES[skill].checkpoint
    if default is None:
      raise SystemExit(
        f"no expert checkpoint for skill {skill!r}; pass --expert {skill}=<path>"
      )
    return default

  base_data = [
    torch.load(path, map_location="cpu", weights_only=False)
    for path in args.expert_data
  ]
  student = build_student(args.device)
  checkpoint = torch.load(args.student, map_location="cpu", weights_only=False)
  student.load_state_dict(checkpoint["actor_state_dict"])
  student.eval()

  experts: dict[str, object] = {}
  for skill in args.skill:
    experts[skill] = load_expert_policy(
      SKILL_SOURCES[skill].task_id, expert_path(skill), args.device
    )

  aggregated = {dataset.skill: dataset for dataset in base_data}
  rounds_log: list[dict[str, object]] = []
  for round_index in range(args.rounds):
    # Anneal the expert's share of the actions: early rounds stay close to the
    # expert manifold, later rounds push into the student's own distribution.
    expert_prob = args.expert_prob * (0.5**round_index)
    round_summary: dict[str, object] = {
      "round": round_index,
      "expert_prob": expert_prob,
    }
    for skill in args.skill:
      spec = ExpertSpec(skill, expert_path(skill))
      fresh = collect_round(
        spec,
        student,
        experts[skill],
        num_envs=args.num_envs,
        steps_per_env=args.steps_per_env,
        expert_prob=expert_prob,
        device=args.device,
      )
      existing = aggregated.get(skill)
      if existing is not None:
        fresh = SkillDataset(
          skill=skill,
          task_id=fresh.task_id,
          expert_checkpoint=fresh.expert_checkpoint,
          obs=torch.cat((existing.obs, fresh.obs)),
          action=torch.cat((existing.action, fresh.action)),
        )
      aggregated[skill] = fresh
      round_summary[f"{skill}_samples"] = fresh.samples

    if args.include_transitions:
      fresh_transitions = collect_transitions(
        experts,
        student,
        skills=args.skill,
        num_envs=args.num_envs,
        hold_steps=args.hold_steps,
        collect_steps=args.collect_steps,
        device=args.device,
      )
      for skill, extra in fresh_transitions.items():
        existing = aggregated[skill]
        aggregated[skill] = SkillDataset(
          skill=skill,
          task_id=existing.task_id,
          expert_checkpoint=existing.expert_checkpoint,
          obs=torch.cat((existing.obs, extra.obs)),
          action=torch.cat((existing.action, extra.action)),
        )
      round_summary["transition_samples"] = {
        skill: int(extra.samples) for skill, extra in fresh_transitions.items()
      }

    datasets = [aggregated[skill] for skill in args.skill]
    training = train(
      datasets,
      output=output,
      epochs=args.epochs,
      batch_size=args.batch_size,
      learning_rate=1e-3,
      val_fraction=0.02,
      device=args.device,
      seed=round_index,
    )
    student.load_state_dict(
      torch.load(output, map_location="cpu", weights_only=False)["actor_state_dict"]
    )
    student.eval()
    round_summary["validation_mse"] = training["validation_mse"]
    rounds_log.append(round_summary)
    print(json.dumps(round_summary, indent=2, ensure_ascii=False))

  report: dict[str, object] = {
    "rounds": rounds_log,
    "output": str(output),
    "samples": {skill: aggregated[skill].samples for skill in args.skill},
  }
  print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
