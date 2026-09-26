"""Offline behaviour cloning into one skill-conditioned student policy.

Produces a checkpoint the PPO runner can load directly
(``--agent.resume True --agent.load-checkpoint student.pt``): the state dict is
built from RSL-RL's own ``MLPModel``, so key names, normalization semantics and
the action distribution are exactly what the training loop expects.

The student is validated per skill. A single shared trunk can represent four
mutually exclusive behaviours only if it does not sacrifice one for another, so
the per-skill validation error is the acceptance signal -- an aggregate number
would hide the skill that got dropped.

Usage::

  uv run opendoge-bc --data logs/skills_data/walk.pt logs/skills_data/getup.pt \\
      logs/skills_data/handstand.pt logs/skills_data/jump.pt \\
      --output logs/skills_data/student.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from src.skills.collect import SKILLS_TASK_ID, SkillDataset
from src.tasks.skills.layout import STUDENT_OBS_DIM
from src.tasks.skills.rl import SkillConditionedActor

OBS_DIM = STUDENT_OBS_DIM
ACTION_DIM = 12
HIDDEN_DIMS = (512, 256, 128)


def build_student(device: str) -> SkillConditionedActor:
  """Construct the student with the same architecture and keys as PPO's actor.

  The *class* is resolved from the task's registered actor config, not written
  here: the clone has to be the network `opendoge-eval` builds, and duplicating
  the name is how the two drift apart. The kwargs below mirror the config the
  runner passes; `test_bc_artifact_loads_into_the_policy_the_evaluator_builds`
  asserts the resulting state dicts are identical, so a kwarg that diverges
  fails the suite instead of silently producing an unloadable artifact.
  """
  from mjlab.rl import RslRlOnPolicyRunnerCfg
  from mjlab.tasks.registry import load_rl_cfg

  agent_cfg = load_rl_cfg(SKILLS_TASK_ID)
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
  dummy = TensorDict({"actor": torch.zeros(1, OBS_DIM)}, batch_size=[1])
  actor_class = resolve_callable(agent_cfg.actor.class_name)
  model = actor_class(
    obs=dummy,
    obs_groups={"actor": ["actor"]},
    obs_set="actor",
    output_dim=ACTION_DIM,
    hidden_dims=list(HIDDEN_DIMS),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )
  return model.to(device)


def _batch(
  obs: torch.Tensor,
  action: torch.Tensor,
  indices: torch.Tensor,
  device: str,
) -> tuple[TensorDict, torch.Tensor]:
  obs_batch = obs[indices].to(device, torch.float32)
  action_batch = action[indices].to(device, torch.float32)
  return TensorDict({"actor": obs_batch}, batch_size=[obs_batch.shape[0]]), action_batch


def train(
  datasets: list[SkillDataset],
  *,
  output: Path,
  epochs: int,
  batch_size: int,
  learning_rate: float,
  val_fraction: float,
  device: str,
  seed: int,
) -> dict[str, object]:
  generator = torch.Generator().manual_seed(seed)
  train_indices: dict[str, torch.Tensor] = {}
  val_indices: dict[str, torch.Tensor] = {}
  obs_by_skill: dict[str, torch.Tensor] = {}
  action_by_skill: dict[str, torch.Tensor] = {}
  # Several datasets may target the same skill (steady state plus handovers).
  # Concatenate them rather than letting the last one win: an overwrite here is
  # silent and would drop the handover frames that per-skill heads need, leaving
  # a clone that trains fine and cannot hand over.
  pooled: dict[str, list[SkillDataset]] = {}
  for dataset in datasets:
    pooled.setdefault(dataset.skill, []).append(dataset)
  skills = [skill for skill in pooled]
  for skill, group in pooled.items():
    obs = torch.cat([dataset.obs for dataset in group])
    action = torch.cat([dataset.action for dataset in group])
    permutation = torch.randperm(obs.shape[0], generator=generator)
    val_count = max(int(obs.shape[0] * val_fraction), 1)
    val_indices[skill] = permutation[:val_count]
    train_indices[skill] = permutation[val_count:]
    obs_by_skill[skill] = obs
    action_by_skill[skill] = action
  model = build_student(device)

  # Seed the observation normalizer from the whole dataset, then freeze it for
  # the rest of cloning. If it keeps updating while the network trains, the
  # network is chasing a moving input distribution and generalisation collapses
  # (measured: get-up validation MSE was ~80x the handstand's).
  normalizer = model.obs_normalizer
  seed_normalizer = getattr(normalizer, "update", None)
  if callable(seed_normalizer):
    model.train()
    with torch.no_grad():
      for skill in skills:
        obs = obs_by_skill[skill].to(device, torch.float32)
        for chunk in obs.split(65536):
          seed_normalizer(chunk)
  model.eval()  # Freezes the normalizer; the MLP itself has no train-only layers.

  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
  per_skill_batch = max(batch_size // len(skills), 1)

  history: list[dict[str, float]] = []
  best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
  best_score = float("inf")
  stale_epochs = 0
  patience = 4
  for epoch in range(epochs):
    epoch_loss = 0.0
    batches = 0
    for skill in skills:
      pool = train_indices[skill]
      num_batches = max(pool.numel() // per_skill_batch, 1)
      for _ in range(num_batches):
        pick = torch.randint(pool.numel(), (per_skill_batch,), generator=generator)
        obs_batch, action_batch = _batch(
          obs_by_skill[skill], action_by_skill[skill], pool[pick], device
        )
        prediction = model(obs_batch)
        loss = F.mse_loss(prediction, action_batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += float(loss.item())
        batches += 1
    scheduler.step()

    # Per-skill validation drives the stopping decision: an aggregate would hide
    # the one skill that the shared trunk is sacrificing.
    running_validation: dict[str, float] = {}
    with torch.inference_mode():
      for skill in skills:
        obs_batch, action_batch = _batch(
          obs_by_skill[skill], action_by_skill[skill], val_indices[skill], device
        )
        running_validation[skill] = float(
          F.mse_loss(model(obs_batch), action_batch).item()
        )
    score = max(running_validation.values())
    history.append(
      {
        "epoch": float(epoch),
        "train_mse": epoch_loss / max(batches, 1),
        "val_worst": score,
        **{f"val_{k}": v for k, v in running_validation.items()},
      }
    )
    if score < best_score - 1e-5:
      best_score = score
      stale_epochs = 0
      best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
      stale_epochs += 1
      if stale_epochs >= patience:
        break

  model.load_state_dict(best_state)
  validation: dict[str, float] = {}
  residual_std = torch.zeros(ACTION_DIM, device=device)
  residual_count = 0
  with torch.inference_mode():
    for skill in skills:
      obs_batch, action_batch = _batch(
        obs_by_skill[skill], action_by_skill[skill], val_indices[skill], device
      )
      prediction = model(obs_batch)
      # Report error in joint-offset radians, not in normalized action units:
      # the action scale is 0.25 rad per unit.
      validation[skill] = float(F.mse_loss(prediction, action_batch).item())
      residual_std += (action_batch - prediction).std(dim=0)
      residual_count += 1
  residual_std /= max(residual_count, 1)
  # Start PPO with the exploration scale that matches how well cloning fit the
  # experts, instead of the default unit noise that would wreck the policy at
  # the first update.
  std_param = getattr(model.distribution, "std_param", None)
  if std_param is not None:
    with torch.no_grad():
      std_param.copy_(residual_std.clamp(min=0.1))

  output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(
    {"actor_state_dict": model.state_dict(), "infos": {}, "iter": 0},
    output,
  )
  return {
    "output": str(output),
    "skills": skills,
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "validation_mse": validation,
    "validation_rmse_radians": {
      skill: (value**0.5) * 0.25 for skill, value in validation.items()
    },
    "residual_std": [round(float(v), 4) for v in residual_std.tolist()],
    "history_tail": history[-3:],
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data", type=Path, nargs="+", required=True)
  parser.add_argument(
    "--output", type=Path, default=Path("logs/skills_data/student.pt")
  )
  parser.add_argument("--epochs", type=int, default=20)
  parser.add_argument("--batch-size", type=int, default=4096)
  parser.add_argument("--learning-rate", type=float, default=1.0e-3)
  parser.add_argument("--val-fraction", type=float, default=0.02)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  datasets = [
    torch.load(path, map_location="cpu", weights_only=False) for path in args.data
  ]
  report = train(
    datasets,
    output=args.output,
    epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    val_fraction=args.val_fraction,
    device=args.device,
    seed=args.seed,
  )
  print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
