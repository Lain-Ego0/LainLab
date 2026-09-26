"""Networks and runner used by the multi-skill task.

One policy over four mutually exclusive behaviours is only sound if the network
can represent them without sacrificing one for another, and the measurements
(docs section 5.8) pin down the trade-off:

- a single shared output layer interpolates one continuous function across
  skills, which is a *good* inductive bias for handover (transition aggregate
  0.9389, `handstand->walk` 0.678) but leaves each skill fighting for capacity;
- one head per skill removes that fight but destroys the continuity: with no
  handover samples in the cloning set the aggregate fell to 0.9255 and
  `handstand->walk` to 0.491. Adding handover samples recovered only part of it
  (0.9301 / 0.552).

So the shipped default is the flat shared output layer (``MLPModel``): it is the
best-measured variant on handover. The per-skill head and the zero-initialised
per-skill residual below are kept as measured-but-rejected alternatives, not as
the default (docs section 5.8.3).

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

from src.tasks.multiskill.layout import SKILL_NAMES, SKILL_ONE_HOT_SLICE


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


class SkillConditionedActor(MLPModel):
  """``MLPModel`` whose output can be specialised per skill.

  The trunk, the observation normalizer, the distribution and therefore every
  state-dict key of the hidden layers are exactly the baseline's; only the final
  ``Linear`` is replaced, by whatever :meth:`head_output` builds. Subclasses
  decide how the one-hot skill token that the unified observation already
  carries is used.

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
    self.output_dim = output_dim
    dims = list(hidden_dims)
    if any(dim <= 0 for dim in dims):
      raise ValueError(
        f"{type(self).__name__} needs explicit hidden dims, got {hidden_dims}; "
        "the trunk is rebuilt from them, so the -1 'infer' convention is not "
        "supported"
      )
    if num_skills != len(SKILL_NAMES):
      raise ValueError(f"expected {len(SKILL_NAMES)} skills, got {num_skills}")

    # Same trunk as the baseline MLP: every hidden layer, stopping before the
    # output layer that the heads replace.
    self.mlp = MLP(self._get_latent_dim(), dims[-1], dims[:-1], activation)
    self.trunk_dim = dims[-1]
    self.num_skills = num_skills
    self.build_heads()

  def build_heads(self) -> None:
    raise NotImplementedError

  def raw_observation(self, obs: TensorDict) -> torch.Tensor:
    """The model input before normalization: the same concatenation as the latent.

    The skill token has to be read from here, not from the normalized latent,
    because normalization rescales it and a one-hot stops being one-hot.
    """
    return torch.cat([obs[group] for group in self.obs_groups], dim=-1)

  def skill_index(self, actor_obs: torch.Tensor) -> torch.Tensor:
    """Which skill's head each row of the raw actor observation selects."""
    return actor_obs[..., SKILL_ONE_HOT_SLICE].argmax(dim=-1)

  def skill_one_hot(self, actor_obs: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return actor_obs[..., SKILL_ONE_HOT_SLICE].to(like.dtype)

  def head_output(self, latent: torch.Tensor, actor_obs: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError

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


class SkillHeadedActor(SkillConditionedActor):
  """One independent head per skill (the measured ablation, not the default).

  Kept because it is what established that per-skill capacity alone costs
  handover quality; see the module docstring for the numbers.
  """

  def build_heads(self) -> None:
    heads = [nn.Linear(self.trunk_dim, self.output_dim) for _ in range(self.num_skills)]
    self.heads = nn.ModuleList(heads)
    for head in heads:
      nn.init.orthogonal_(head.weight, gain=1.0)
      nn.init.zeros_(head.bias)

  def head_output(self, latent: torch.Tensor, actor_obs: torch.Tensor) -> torch.Tensor:
    """Apply the selected head to the trunk features.

    Written as a one-hot weighted sum rather than an index lookup so the graph
    stays differentiable and exports to ONNX without a gather; for a genuine
    one-hot it is exactly head selection.
    """
    trunk = self.mlp(latent)
    one_hot = self.skill_one_hot(actor_obs, trunk)
    heads = list(self.heads)
    out = one_hot[..., 0:1] * heads[0](trunk)
    for index in range(1, self.num_skills):
      out = out + one_hot[..., index : index + 1] * heads[index](trunk)
    return out


class SharedResidualActor(SkillConditionedActor):
  """One shared head plus a per-skill residual, residual initialised to zero.

  At initialisation every residual contributes exactly zero, so the network is
  the flat single-head policy: handover behaviour starts from the variant that
  measured best (aggregate 0.9389, `handstand->walk` 0.678) instead of from four
  unrelated functions (0.9255 / 0.491). Training then adds per-skill
  specialisation only where the data asks for it.
  """

  def build_heads(self) -> None:
    self.head = nn.Linear(self.trunk_dim, self.output_dim)
    nn.init.orthogonal_(self.head.weight, gain=1.0)
    nn.init.zeros_(self.head.bias)
    residuals = [
      nn.Linear(self.trunk_dim, self.output_dim) for _ in range(self.num_skills)
    ]
    self.residuals = nn.ModuleList(residuals)
    for residual in residuals:
      # Zero, not orthogonal: the residual has to be a no-op at initialisation,
      # which is the whole point of the architecture.
      nn.init.zeros_(residual.weight)
      nn.init.zeros_(residual.bias)

  def head_output(self, latent: torch.Tensor, actor_obs: torch.Tensor) -> torch.Tensor:
    trunk = self.mlp(latent)
    one_hot = self.skill_one_hot(actor_obs, trunk)
    out = self.head(trunk)
    residuals = list(self.residuals)
    for index in range(1, self.num_skills):
      out = out + one_hot[..., index : index + 1] * residuals[index](trunk)
    out = out + one_hot[..., 0:1] * residuals[0](trunk)
    return out
