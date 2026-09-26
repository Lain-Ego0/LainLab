"""Skill-conditioning helpers shared by the multi-skill MDP terms."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv

from src.tasks.skills.mdp.command import SkillCommandTerm

DEFAULT_COMMAND_NAME = "skill"
COMMAND_PARAM_KEYS = ("command_name",)


def skill_term(
  env: ManagerBasedRlEnv, command_name: str = DEFAULT_COMMAND_NAME
) -> SkillCommandTerm:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, SkillCommandTerm), (
    f"Command term {command_name!r} is not a SkillCommandTerm"
  )
  return term


def skill_mask(
  env: ManagerBasedRlEnv,
  skill: str,
  command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
  """Float mask, 1 where the environment runs ``skill``."""
  return skill_term(env, command_name).skill_is(skill).float()


def skill_identity_observation(
  env: ManagerBasedRlEnv,
  command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
  """``[one_hot(skill), phase_sin, phase_cos]``, shape ``[B, K + 2]``."""
  term = skill_term(env, command_name)
  one_hot = torch.nn.functional.one_hot(
    term.skill, num_classes=len(term.skill_names)
  ).to(torch.float32)
  return torch.cat((one_hot, term.skill_phase()), dim=-1)


def jump_twist_observation(
  env: ManagerBasedRlEnv,
  command_name: str = DEFAULT_COMMAND_NAME,
) -> torch.Tensor:
  """The jump's commanded ``[vx, vy, wz]``, zeros for every other skill.

  Reproduces the jump task's own ``jump_twist`` term so the first 51 fields of
  the unified observation are that expert's native observation.
  """
  return skill_term(env, command_name).twist


def retarget_command_params(
  func: Callable[..., Any] | type, params: dict[str, Any]
) -> dict[str, Any]:
  """Point a harvested reward term at the unified command term.

  Sub-task rewards are copied verbatim from the single-skill configurations,
  where the command term is called ``twist`` (walking) or ``jump``. In the
  unified environment there is exactly one command term, ``skill``, so only that
  name has to be rewritten -- no reward formula is touched.

  Two spellings exist and both must be covered: some terms carry an explicit
  ``command_name`` parameter, while others (the jump rewards) take it as a
  default argument. Class-based terms are left alone; they read their
  configuration in ``__init__`` and already carry an explicit parameter.
  """
  retargeted = dict(params)
  for key in COMMAND_PARAM_KEYS:
    if key in retargeted:
      retargeted[key] = DEFAULT_COMMAND_NAME
  if inspect.isfunction(func) and "command_name" in inspect.signature(func).parameters:
    retargeted.setdefault("command_name", DEFAULT_COMMAND_NAME)
  return retargeted


def masked_by_skill(
  func: Callable[..., torch.Tensor] | type,
  skill: str,
  command_name: str = DEFAULT_COMMAND_NAME,
) -> Callable[..., torch.Tensor] | type:
  """Wrap a reward term so it only fires for environments running ``skill``.

  Two forms must be preserved because mjlab reward terms can be either:

  - a plain function, called as ``func(env, **params)``, or
  - a class, instantiated by the reward manager as ``cls(cfg=term_cfg, env=env)``
    and then called as ``instance(env, **params)``, optionally stateful with its
    own ``reset``.

  Wrapping a class in a plain closure would make the manager call the class
  constructor with the reward parameters, so classes get a class wrapper that
  keeps the original construction and delegates ``reset``.
  """
  if inspect.isclass(func):

    class MaskedSkillTerm:
      def __init__(self, cfg: object, env: ManagerBasedRlEnv) -> None:
        self._inner = func(cfg=cfg, env=env)  # type: ignore[call-arg]
        self._skill = skill
        self._command_name = command_name

      def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        value = self._inner(env, **params)
        return value * skill_mask(env, self._skill, self._command_name)

      def reset(self, env_ids: object = None) -> dict[str, Any]:
        inner_reset = getattr(self._inner, "reset", None)
        if inner_reset is None:
          return {}
        return inner_reset(env_ids=env_ids)

    MaskedSkillTerm.__name__ = f"{skill}__{getattr(func, '__name__', 'term')}"
    MaskedSkillTerm.__qualname__ = MaskedSkillTerm.__name__
    return MaskedSkillTerm

  def masked_term(env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
    return func(env, **params) * skill_mask(env, skill, command_name)

  masked_term.__name__ = f"{skill}__{getattr(func, '__name__', 'term')}"
  masked_term.__qualname__ = masked_term.__name__
  return masked_term
