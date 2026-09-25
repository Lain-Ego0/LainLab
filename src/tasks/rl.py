"""Shared RSL-RL configuration factories."""

from dataclasses import dataclass
from typing import Any

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class RslRlPpoWithSymmetryAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """Expose the symmetry extension supported by the installed RSL-RL PPO."""

  symmetry_cfg: dict[str, Any] | None = None


def make_ppo_runner_cfg(
  experiment_name: str,
  *,
  max_iterations: int = 10_000,
  learning_rate: float = 1.0e-3,
  entropy_coef: float = 0.01,
  save_interval: int = 100,
  symmetry_cfg: dict[str, Any] | None = None,
) -> RslRlOnPolicyRunnerCfg:
  """Create the common PPO setup used by LainLab locomotion tasks."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoWithSymmetryAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=entropy_coef,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=learning_rate,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      symmetry_cfg=symmetry_cfg,
    ),
    experiment_name=experiment_name,
    save_interval=save_interval,
    num_steps_per_env=24,
    max_iterations=max_iterations,
  )
