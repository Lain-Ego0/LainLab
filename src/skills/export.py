"""Export a trained skill policy to ONNX.

Deployment needs the policy as a single ONNX graph. For the multi-skill task the
actor input is the 57-field observation (48 shared proprioceptive fields, the
3-field jump twist command and the one-hot skill token), so the exported graph is
the whole controller: the caller only has to supply the skill token.

Usage::

  uv run opendoge-export LainLab-OpenDoge-Skills-Flat \\
      --checkpoint logs/opendoge_skills_ft/.../model_3000.pt \\
      --output baseline/opendoge/skills/policy.onnx
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

import src.tasks  # noqa: F401


def export(
  task_id: str, checkpoint: Path, output: Path, device: str
) -> dict[str, object]:
  configure_torch_backends()
  cfg = load_env_cfg(task_id, play=True)
  cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(task_id)
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_onnx(
      str(output.parent), filename=output.name, verbose=False
    )
  finally:
    env.close()
  return {
    "task_id": task_id,
    "checkpoint": str(checkpoint),
    "onnx": str(output),
    "onnx_bytes": output.stat().st_size if output.is_file() else 0,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task_id")
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  report = export(args.task_id, args.checkpoint, args.output, args.device)
  print(report)


if __name__ == "__main__":
  main()
