"""Networks and runner used by the multi-skill task.

The task is one policy over four mutually exclusive behaviours, which is only
sound if the network can represent them without sacrificing one for another. A
single shared output layer cannot: measured (docs section 5.8.1), adding the
512k-sample moving-jump dataset to the clone pushed the walk velocity error from
0.0569 to 0.0649 m/s while handstand and get-up stayed identical, and up-weighting
the walk data did not recover it (0.0642 at 2x, 0.0685 at 3x). The trunk has to
be shared -- that is what makes it one policy -- but the *output* does not, so
each skill gets its own head.

The runner is unchanged in what it optimises; it exists because a
behaviour-cloning artifact contains only the actor, while the stock training path
demands a complete checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, HiddenState
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict

from src.tasks.skills.layout import SKILL_NAMES, SKILL_ONE_HOT_SLICE


class SkillOnPolicyRunner(MjlabOnPolicyRunner):
  """``MjlabOnPolicyRunner`` that can resume from an actor-only checkpoint."""

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    if load_cfg is None:
      load_cfg = {"actor": True}
    return super().load(
      path, load_cfg=load_cfg, strict=strict, map_location=map_location
    )


class SkillHeadedActor(MLPModel):
  """``MLPModel`` whose output layer is one head per skill.

  The trunk, the observation normalizer, the distribution and therefore every
  state-dict key of the hidden layers are exactly the baseline's; only the final
  ``Linear`` is replaced by ``len(SKILL_NAMES)`` of them, selected by the one-hot
  skill token that the unified observation already carries.

  Registered as the task's actor via ``RslRlModelCfg.class_name``, so the same
  class is built by PPO training, by ``opendoge-bc`` and by ``opendoge-eval``.
  """

  def __init__(
    self,
    obs,
    obs_groups,
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    num_skills: int = len(SKILL_NAMES),
  ) -> None:
    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims,
      activation,
      obs_normalization,
      distribution_cfg,
    )
    dims = list(hidden_dims)
    if any(dim <= 0 for dim in dims):
      raise ValueError(
        f"SkillHeadedActor needs explicit hidden dims, got {hidden_dims}; the "
        "trunk is rebuilt from them, so the -1 'infer' convention is not supported"
      )
    if num_skills != len(SKILL_NAMES):
      raise ValueError(f"expected {len(SKILL_NAMES)} skills, got {num_skills}")

    # Same trunk as the baseline MLP: every hidden layer, stopping before the
    # output layer that the heads replace.
    self.mlp = MLP(self._get_latent_dim(), dims[-1], dims[:-1], activation)
    heads = [nn.Linear(dims[-1], output_dim) for _ in range(num_skills)]
    self.heads = nn.ModuleList(heads)
    for head in heads:
      nn.init.orthogonal_(head.weight, gain=1.0)
      nn.init.zeros_(head.bias)

  @property
  def num_skills(self) -> int:
    return len(self.heads)

  def raw_observation(self, obs: TensorDict) -> torch.Tensor:
    """The model input before normalization: the same concatenation as the latent.

    The skill token has to be read from here, not from the normalized latent,
    because normalization rescales it and a one-hot stops being one-hot.
    """
    return torch.cat([obs[group] for group in self.obs_groups], dim=-1)

  def skill_index(self, actor_obs: torch.Tensor) -> torch.Tensor:
    """Which head each row of the raw actor observation selects."""
    return actor_obs[..., SKILL_ONE_HOT_SLICE].argmax(dim=-1)

  def head_output(self, latent: torch.Tensor, actor_obs: torch.Tensor) -> torch.Tensor:
    """Apply the selected head to the trunk features.

    Written as a one-hot weighted sum rather than an index lookup so the graph
    stays differentiable and exports to ONNX without a gather; for a genuine
    one-hot it is exactly head selection.
    """
    trunk = self.mlp(latent)
    one_hot = actor_obs[..., SKILL_ONE_HOT_SLICE].to(trunk.dtype)
    heads = list(self.heads)
    out = one_hot[..., 0:1] * heads[0](trunk)
    for index in range(1, self.num_skills):
      out = out + one_hot[..., index : index + 1] * heads[index](trunk)
    return out

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    # Mirrors `MLPModel.forward`; the raw observation is needed alongside the
    # normalized latent because the skill token must not be normalized before it
    # selects a head.
    obs = unpad_trajectories(obs, masks) if masks is not None else obs
    latent = self.get_latent(obs, masks, hidden_state)
    mlp_output = self.head_output(latent, self.raw_observation(obs))
    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(mlp_output)
        return self.distribution.sample()
      return self.distribution.deterministic_output(mlp_output)
    return mlp_output
