"""Recover-to-walk task: the walking MDP with a mixed standing/fallen reset.

Why this exists
---------------

Cloning walks perfectly on its own but is the weakest *destination* in the
single-policy transition matrix (`handstand -> walk` scored 0.686 while every
skill scored >0.95 on its own). The expert demonstrations only ever show the
walking controller starting from a nominal stance, so nothing teaches the policy
what to do when the walk token arrives mid-manoeuvre -- and a crossed-out
cross-skill DAgger experiment showed that labelling those states with the
single-skill walking expert produces meaningless targets (98% of the collected
handstand-target samples were airborne).

The fix is an expert whose *training distribution covers the approach states*:
the walk task's reward set and velocity command, but reset from a random fall
half of the time. Its manifold is strictly larger than the walking expert's, and
it is valid everywhere the walk token can be handed over.

The contact termination is deliberately dropped -- a fallen start would trip it
immediately -- and replaced with a dense non-foot contact penalty.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from src.tasks.recover.mdp import events as recover_events
from src.tasks.rl import make_ppo_runner_cfg
from src.tasks.velocity.core import (
  VelocityRobotProfile,
  make_flat_env_cfg,
)

DEFAULT_FALLEN_PROBABILITY = 0.5
"""Fraction of each batch that starts from a random fall rather than standing."""


def make_recover_walk_env_cfg(
  profile: VelocityRobotProfile,
  *,
  fallen_probability: float = DEFAULT_FALLEN_PROBABILITY,
  nonfoot_penalty_weight: float = -0.002,
  episode_length_s: float = 10.0,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build a flat-ground recover-to-walk environment for one robot."""
  cfg = make_flat_env_cfg(profile, play=False)
  cfg.episode_length_s = episode_length_s

  # Mixed reset: standing for part of the batch, random fall for the rest.
  startup_events = {
    name: cfg.events[name]
    for name in ("foot_friction", "encoder_bias", "base_com")
    if name in cfg.events
  }
  cfg.events = {
    "reset_base": EventTermCfg(
      func=recover_events.reset_mixed_root,
      mode="reset",
      params={"fallen_probability": fallen_probability},
    ),
    "reset_robot_joints": EventTermCfg(
      func=recover_events.reset_mixed_joints,
      mode="reset",
      params={"fallen_probability": fallen_probability},
    ),
    **startup_events,
  }

  # No contact termination: it would fire on the first step of every fallen
  # start. A dense penalty takes its place.
  cfg.terminations.pop("illegal_contact", None)
  cfg.terminations["time_out"] = TerminationTermCfg(
    func=envs_mdp.time_out, time_out=True
  )
  cfg.rewards["nonfoot_contact"] = RewardTermCfg(
    func=recover_events.nonfoot_contact_penalty, weight=nonfoot_penalty_weight
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg


def register_recover_profile(
  profile: VelocityRobotProfile,
  *,
  task_id: str,
  experiment_name: str,
  fallen_probability: float = DEFAULT_FALLEN_PROBABILITY,
  max_iterations: int = 5_000,
) -> None:
  """Register a train and play task for one robot profile."""
  env_cfg = make_recover_walk_env_cfg(
    profile, fallen_probability=fallen_probability, play=False
  )
  play_cfg = make_recover_walk_env_cfg(
    profile, fallen_probability=fallen_probability, play=True
  )
  rl_cfg = make_ppo_runner_cfg(experiment_name, max_iterations=max_iterations)
  register_mjlab_task(
    task_id=task_id,
    env_cfg=env_cfg,
    play_env_cfg=play_cfg,
    rl_cfg=rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
  )
