"""Collect expert rollouts for behaviour cloning.

The data is gathered inside the *unified* multi-skill environment so the stored
observations already have the 54-field student layout. Each expert is a
single-skill policy whose native observation is exactly the first 48 fields of
the unified observation -- same field order, same command semantics, same noise
-- so the expert is driven by slicing, not by an adapter that could drift.

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

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from tensordict import TensorDict

import src.tasks  # noqa: F401
from src.tasks.skills.mdp.command import SkillCommandTerm

SKILLS_TASK_ID = "LainLab-OpenDoge-Skills-Flat"
# The unified actor observation is the shared 48-field proprioceptive block
# followed by the skill identity block.
SHARED_OBS_DIM = 48
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
  obs_chunks: list[torch.Tensor] = []
  action_chunks: list[torch.Tensor] = []
  for _ in range(steps_per_env):
    actor_obs = observations["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    with torch.inference_mode():
      expert_obs = TensorDict(
        {"actor": actor_obs[:, :SHARED_OBS_DIM]}, batch_size=[num_envs]
      )
      actions = policy(expert_obs)
    obs_chunks.append(actor_obs.detach().to("cpu", torch.float16))
    action_chunks.append(actions.detach().to("cpu", torch.float16))
    observations, _, _, _, _ = env.step(actions)

  env.close()
  return SkillDataset(
    skill=skill,
    task_id=source.task_id,
    expert_checkpoint=str(checkpoint),
    obs=torch.cat(obs_chunks).reshape(-1, obs_chunks[0].shape[-1]),
    action=torch.cat(action_chunks).reshape(-1, action_chunks[0].shape[-1]),
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--skill", choices=sorted(SKILL_SOURCES), required=True)
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--output", type=Path, default=None)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--steps-per-env", type=int, default=250)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()

  configure_torch_backends()
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
