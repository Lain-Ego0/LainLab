"""Headless policy evaluation for the OpenDoge skills.

Reward curves are not acceptance evidence: a policy can farm a shaped reward
while doing nothing recognisable. This script runs a checkpoint deterministically
and reports the physical quantities that define each skill, so a task can be
accepted or rejected on measurements.

Usage::

  uv run opendoge-eval LainLab-OpenDoge-Handstand \
      --checkpoint logs/opendoge_handstand/lainlab_opendoge_handstand/<run>/model_4999.pt
  uv run opendoge-eval LainLab-OpenDoge-Jump --checkpoint ... --num-envs 64
  uv run opendoge-eval LainLab-OpenDoge-Skills-Flat --checkpoint ...
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

import src.tasks  # noqa: F401
from src.tasks.jump.mdp.command import JumpCommand
from src.tasks.skills.mdp.command import SKILL_NAMES, SkillCommandTerm

STANDING_HEIGHT = 0.151
HANDSTAND_HEIGHT = 0.220
HANDSTAND_ALIGNMENT_THRESHOLD = 0.9  # ~25 degrees off vertical
JUMP_RISE_TARGET = 0.05


def _contact_flags(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  flags = (found > 0).to(torch.float32)
  if flags.dim() == 3:
    flags = flags.amax(dim=-1)
  return flags


def _height(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  return robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def _alignment(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Handstand alignment: 1 when the body is pitched to vertical."""
  robot = env.scene["robot"]
  target = torch.tensor([1.0, 0.0, 0.0], device=env.device)
  return torch.clamp(robot.data.projected_gravity_b @ target, -1.0, 1.0)


def _upright(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Nominal uprightness: 1 when the body is level, used for walk/get-up."""
  robot = env.scene["robot"]
  target = torch.tensor([0.0, 0.0, -1.0], device=env.device)
  return torch.clamp(robot.data.projected_gravity_b @ target, -1.0, 1.0)


def evaluate(
  task_id: str,
  checkpoint: Path,
  *,
  num_envs: int,
  steps: int,
  device: str,
) -> dict[str, object]:
  configure_torch_backends()
  cfg = load_env_cfg(task_id, play=True)
  cfg.scene.num_envs = num_envs
  agent_cfg = load_rl_cfg(task_id)
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  names = ["walk", "handstand", "getup", "jump"]
  per_skill: dict[str, dict[str, list]] = {
    name: {
      "height": [],
      "alignment": [],
      "feet_down": [],
      "drift": [],
      "airborne": [],
      "upright": [],
      "velocity_error": [],
      "foot_contact": [],
    }
    for name in names
  }
  start_xy: torch.Tensor | None = None
  jump_takeoffs = 0
  jump_recoveries = 0
  was_airborne = torch.zeros(num_envs, dtype=torch.bool, device=device)
  was_recovered = torch.zeros(num_envs, dtype=torch.bool, device=device)

  observation, _ = env.reset()
  # Single-skill tasks call their command term `twist` or `jump`; the unified
  # task calls it `skill`. Discover it rather than assuming.
  active_commands = list(env.command_manager.active_terms)
  skill_term_name = active_commands[0] if active_commands else "twist"
  skill_term = env.command_manager.get_term(skill_term_name)

  for _ in range(steps):
    with torch.no_grad():
      actions = policy(observation)
    observation, _, _, _, _ = env.step(actions)
    height = _height(env)
    alignment = _alignment(env)
    upright = _upright(env)
    command = env.command_manager.get_command(skill_term_name)
    assert command is not None
    velocity_error = torch.norm(
      command[:, :2] - env.scene["robot"].data.root_link_lin_vel_b[:, :2], dim=1
    )
    contact = _contact_flags(env, "feet_ground_contact")
    feet_down = (contact > 0.5).sum(dim=1).float()
    airborne = feet_down < 0.5
    robot = env.scene["robot"]
    if start_xy is None:
      start_xy = robot.data.root_link_pos_w[:, :2].clone()

    masks: dict[str, torch.Tensor]
    if isinstance(skill_term, SkillCommandTerm):
      masks = {name: skill_term.skill_is(name) for name in names}
    else:
      single = task_id.rsplit("-", 1)[-1].lower()
      key = "jump" if single == "jump" else "handstand"
      masks = {name: torch.full_like(airborne, name == key) for name in names}

    drift = torch.norm(robot.data.root_link_pos_w[:, :2] - start_xy, dim=1)
    for name, mask in masks.items():
      if not bool(mask.any()):
        continue
      bucket = per_skill[name]
      bucket["height"].extend(height[mask].tolist())
      bucket["alignment"].extend(alignment[mask].tolist())
      bucket["feet_down"].extend(feet_down[mask].tolist())
      bucket["drift"].extend(drift[mask].tolist())
      bucket["airborne"].extend(airborne[mask].float().tolist())
      bucket["upright"].extend(upright[mask].tolist())
      bucket["velocity_error"].extend(velocity_error[mask].tolist())
      bucket["foot_contact"].append(contact[mask].mean(dim=0))

    if isinstance(skill_term, (SkillCommandTerm, JumpCommand)):
      left_ground = skill_term.left_ground
      new_takeoff = left_ground & ~was_airborne
      jump_takeoffs += int(new_takeoff.sum())
      # Edge-triggered: count each environment once for having come back to a
      # settled stance after a takeoff, not every step it stays there.
      recovered = (
        left_ground
        & (feet_down == 4.0)
        & (height < STANDING_HEIGHT + 0.03)
        & (height > STANDING_HEIGHT - 0.03)
      )
      jump_recoveries += int((recovered & ~was_recovered).sum())
      was_recovered |= recovered
      was_airborne = left_ground.clone()

  report: dict[str, object] = {
    "task_id": task_id,
    "checkpoint": str(checkpoint),
    "num_envs": num_envs,
    "steps": steps,
  }
  skill_reports: dict[str, dict[str, object]] = {}
  for name, bucket in per_skill.items():
    if not bucket["height"]:
      continue
    height_t = torch.tensor(bucket["height"])
    alignment_t = torch.tensor(bucket["alignment"])
    feet_t = torch.tensor(bucket["feet_down"])
    airborne_t = torch.tensor(bucket["airborne"])
    drift_t = torch.tensor(bucket["drift"])
    entry: dict[str, object] = {
      "samples": float(height_t.numel()),
      "height_mean": float(height_t.mean()),
      "height_max": float(height_t.max()),
      "alignment_mean": float(alignment_t.mean()),
      "fraction_near_vertical": float(
        (alignment_t > HANDSTAND_ALIGNMENT_THRESHOLD).float().mean()
      ),
      "fraction_all_feet_down": float((feet_t == 4.0).float().mean()),
      "fraction_airborne": float(airborne_t.mean()),
      "drift_mean": float(drift_t.mean()),
      "drift_max": float(drift_t.max()),
      "upright_mean": float(torch.tensor(bucket["upright"]).mean()),
      "fraction_standing": float(
        (
          (torch.tensor(bucket["height"]) > 0.13)
          & (torch.tensor(bucket["upright"]) > 0.9)
        )
        .float()
        .mean()
      ),
      "velocity_error_mean": float(torch.tensor(bucket["velocity_error"]).mean()),
    }
    if bucket["foot_contact"]:
      # Sensor frame order follows the robot profile: FR, FL, RR, RL, so the
      # first two entries are the front feet and the last two the rear pair.
      mean_contact = torch.stack(bucket["foot_contact"]).mean(dim=0)
      entry["foot_contact_fraction"] = {
        name: float(value)
        for name, value in zip(("FR", "FL", "RR", "RL"), mean_contact, strict=True)
      }
    if name == "handstand":
      entry["passes_handstand"] = float(
        (
          (alignment_t > HANDSTAND_ALIGNMENT_THRESHOLD)
          & (torch.tensor(bucket["height"]) > 0.19)
        )
        .float()
        .mean()
      )
    if name == "jump":
      entry["peak_rise"] = float(height_t.max() - STANDING_HEIGHT)
      entry["passes_jump_height"] = float(
        (height_t.max() - STANDING_HEIGHT) >= JUMP_RISE_TARGET
      )
    skill_reports[name] = entry
  report["skills"] = skill_reports
  report["jump_takeoffs"] = jump_takeoffs
  report["jump_recoveries"] = jump_recoveries
  report["jump_recovery_ratio"] = (
    jump_recoveries / jump_takeoffs if jump_takeoffs else 0.0
  )
  env.close()
  return report


def transitions(
  task_id: str,
  checkpoint: Path,
  *,
  num_envs: int,
  hold_steps: int,
  settle_steps: int,
  measure_steps: int,
  device: str,
) -> dict[str, object]:
  """Script every ordered skill transition and score the destination skill.

  Holding one skill and then commanding another is the property that a
  single-policy multi-skill controller is actually bought for, and it is the
  one thing behaviour cloning cannot provide (the demonstration data contains no
  switches). Each pair is scored on the destination skill's own success metric
  measured over the last ``measure_steps`` steps.
  """
  configure_torch_backends()
  cfg = load_env_cfg(task_id, play=True)
  cfg.scene.num_envs = num_envs
  agent_cfg = load_rl_cfg(task_id)
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  term = env.command_manager.get_term("skill")
  assert isinstance(term, SkillCommandTerm), "transitions need the unified task"
  skills = list(SKILL_NAMES)
  ids = torch.arange(num_envs, device=device)
  matrix: dict[str, dict[str, float]] = {}

  for source in skills:
    matrix[source] = {}
    for target in skills:
      observation, _ = env.reset()
      term.force_skill(source, ids)
      for _ in range(hold_steps):
        with torch.no_grad():
          observation, _, _, _, _ = env.step(policy(observation))
      term.force_skill(target, ids)
      for _ in range(settle_steps):
        with torch.no_grad():
          observation, _, _, _, _ = env.step(policy(observation))
      heights, upright, align, feet_down = [], [], [], []
      for _ in range(measure_steps):
        with torch.no_grad():
          observation, _, _, _, _ = env.step(policy(observation))
        robot = env.scene["robot"]
        heights.append(
          (robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).cpu()
        )
        gravity = robot.data.projected_gravity_b
        upright.append((-gravity[:, 2]).cpu())
        align.append((gravity @ torch.tensor([1.0, 0.0, 0.0], device=device)).cpu())
        contact = _contact_flags(env, "feet_ground_contact")
        feet_down.append((contact > 0.5).sum(dim=1).cpu())
      height_t = torch.cat(heights)
      upright_t = torch.cat(upright)
      align_t = torch.cat(align)
      feet_t = torch.cat(feet_down)
      matrix[source][target] = _skill_success(
        target, height_t, upright_t, align_t, feet_t
      )

  env.close()
  return {
    "task_id": task_id,
    "checkpoint": str(checkpoint),
    "num_envs": num_envs,
    "hold_steps": hold_steps,
    "settle_steps": settle_steps,
    "measure_steps": measure_steps,
    "success": matrix,
    "mean_off_diagonal": _off_diagonal_mean(matrix),
  }


def _skill_success(
  skill: str,
  height: torch.Tensor,
  upright: torch.Tensor,
  align: torch.Tensor,
  feet_down: torch.Tensor,
) -> float:
  """Fraction of measured steps that count as performing ``skill``."""
  if skill == "walk":
    return float(((upright > 0.9) & (height > 0.13) & (feet_down < 4)).float().mean())
  if skill == "handstand":
    return float(
      ((align > HANDSTAND_ALIGNMENT_THRESHOLD) & (height > 0.19)).float().mean()
    )
  if skill == "getup":
    return float(((upright > 0.9) & (height > 0.13)).float().mean())
  if skill == "jump":
    peak = float(height.max() - STANDING_HEIGHT)
    return float(peak >= JUMP_RISE_TARGET)
  raise ValueError(skill)


def _off_diagonal_mean(matrix: dict[str, dict[str, float]]) -> float:
  values = [
    value
    for source, row in matrix.items()
    for target, value in row.items()
    if source != target
  ]
  return sum(values) / len(values) if values else 0.0


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task_id", help="registered task id")
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--steps", type=int, default=1500)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--transitions",
    action="store_true",
    help="score every ordered skill transition instead of per-skill rollouts",
  )
  parser.add_argument("--hold-steps", type=int, default=250)
  parser.add_argument("--settle-steps", type=int, default=250)
  parser.add_argument("--measure-steps", type=int, default=200)
  args = parser.parse_args()
  if args.transitions:
    report = transitions(
      args.task_id,
      args.checkpoint,
      num_envs=args.num_envs,
      hold_steps=args.hold_steps,
      settle_steps=args.settle_steps,
      measure_steps=args.measure_steps,
      device=args.device,
    )
  else:
    report = evaluate(
      args.task_id,
      args.checkpoint,
      num_envs=args.num_envs,
      steps=args.steps,
      device=args.device,
    )
  print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
