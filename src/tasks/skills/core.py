"""Single-policy multi-skill task builder.

The composition rule is deliberately narrow: every sub-task's reward terms are
*harvested* from the already-validated single-skill environment builder and then
wrapped in a skill mask. No reward formula, weight or parameter is retyped here,
so a skill cannot drift between its own task and the unified one.

What changes for the unified environment:

- one ``skill`` command term replaces ``twist``, holding a 3-wide skill-specific
  command block (the observation term keeps its name, position and width, so the
  proprioceptive layout stays at 48 fields)
- one extra observation block, ``[one_hot(skill), phase_sin, phase_cos]``
- per-skill reset dispatch, because get-up must start fallen while everything
  else starts standing
- the body-contact termination is masked off for get-up, where contact is the
  point of the task
"""

from __future__ import annotations

from typing import Any

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.getup import make_getup_env_cfg
from src.tasks.handstand import make_handstand_env_cfg
from src.tasks.jump import make_jump_env_cfg
from src.tasks.rl import make_ppo_runner_cfg
from src.tasks.skills.mdp import curriculums as skills_curriculums
from src.tasks.skills.mdp import events as skills_events
from src.tasks.skills.mdp import terminations as skills_terminations
from src.tasks.skills.mdp import utils as skills_utils
from src.tasks.skills.mdp.command import SKILL_NAMES, SkillCommandCfg
from src.tasks.skills.mdp.events import StandingReset
from src.tasks.skills.rl import SkillOnPolicyRunner
from src.tasks.velocity.core import VelocityRobotProfile, _make_base_env_cfg

SKILL_COMMAND_NAME = "skill"


def _harvest_rewards(
  rewards: dict[str, RewardTermCfg],
  skill: str,
  *,
  skip: tuple[str, ...] = (),
) -> dict[str, RewardTermCfg]:
  """Wrap a single-skill reward dict so it only fires for ``skill``."""
  harvested: dict[str, RewardTermCfg] = {}
  for name, term in rewards.items():
    if name in skip:
      continue
    harvested[f"{skill}/{name}"] = RewardTermCfg(
      func=skills_utils.masked_by_skill(term.func, skill, SKILL_COMMAND_NAME),
      weight=term.weight,
      params=skills_utils.retarget_command_params(term.func, term.params),
    )
  return harvested


def _harvest_standing_reset(
  cfg: ManagerBasedRlEnvCfg, *, fallen_probability: float = 0.0
) -> StandingReset:
  """Read a sub-task's standing reset instead of restating it.

  Either form the framework uses is supported: a multiplicative
  ``scale_range`` (the handstand task) or an additive ``position_range`` off the
  default pose (the shared velocity task, and therefore walking and jumping).
  """
  base = cfg.events.get("reset_base")
  joints = cfg.events.get("reset_robot_joints")
  params: dict[str, Any] = base.params if base is not None else {}
  pose_range = params.get("pose_range", {})
  velocity_range = params.get("velocity_range") or {}
  joint_params: dict[str, Any] = joints.params if joints is not None else {}
  return StandingReset(
    height_offset_range=tuple(pose_range.get("z", (0.0, 0.0))),
    joint_scale_range=(
      tuple(joint_params["scale_range"]) if "scale_range" in joint_params else None
    ),
    joint_offset_range=(
      tuple(joint_params["position_range"])
      if "position_range" in joint_params
      else None
    ),
    velocity_range={k: tuple(v) for k, v in velocity_range.items()},
    fallen_probability=fallen_probability,
  )


def _skills_command_cfg(
  profile: VelocityRobotProfile,
  *,
  standing_height: float,
  jump_period_s: float,
  switch_prob: float,
  play: bool,
) -> SkillCommandCfg:
  return SkillCommandCfg(
    entity_name="robot",
    contact_sensor_name="feet_ground_contact",
    resampling_time_range=(1.0e9, 1.0e9),
    skill_names=SKILL_NAMES,
    walk_command_ranges=(
      profile.play_command_ranges if play else profile.command_ranges
    ),
    jump_period_s=jump_period_s,
    standing_height=standing_height,
    switch_prob=switch_prob,
  )


def make_skills_env_cfg(
  profile: VelocityRobotProfile,
  *,
  base_height_target: float,
  rear_clearance_target: float,
  standing_height: float,
  jump_target_rise: float = 0.05,
  jump_period_s: float = 2.5,
  gravity_target: tuple[float, float, float] = (1.0, 0.0, 0.0),
  episode_length_s: float = 10.0,
  switch_prob: float = 0.0,
  switch_stages: tuple[tuple[int, float], ...] = (),
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the unified flat-ground multi-skill environment for one robot.

  ``switch_stages`` ramps the mid-episode skill-switch probability over
  environment steps; leave it empty to keep each episode on one skill.
  """
  cfg = _make_base_env_cfg(profile)
  assert isinstance(cfg.commands["twist"], UniformVelocityCommandCfg)

  # Harvest each sub-task's rewards before anything is replaced.
  walk_cfg = cfg
  walk_rewards = cfg.rewards
  handstand_cfg = make_handstand_env_cfg(
    profile,
    base_height_target=base_height_target,
    rear_clearance_target=rear_clearance_target,
    gravity_target=gravity_target,
  )
  handstand_rewards = handstand_cfg.rewards
  getup_rewards = make_getup_env_cfg(
    profile, base_height_target=standing_height
  ).rewards
  jump_cfg = make_jump_env_cfg(
    profile, standing_height=standing_height, target_rise=jump_target_rise
  )
  jump_rewards = jump_cfg.rewards

  # Reset distributions come from each sub-task, never from a shared default.
  standing_resets = {
    "walk": _harvest_standing_reset(walk_cfg),
    "handstand": _harvest_standing_reset(handstand_cfg),
    "jump": _harvest_standing_reset(jump_cfg),
    # Get-up always starts fallen; the standing reset above is overwritten.
    # A mixed walking start was tried and rejected: see
    # `docs/opendoge_multiskill.md` §5.6.
    "getup": StandingReset(fallen_probability=1.0),
  }

  # Flat ground, no exteroception: keeps the 48-field proprioceptive layout.
  terrain = cfg.scene.terrain
  if terrain is None:
    raise ValueError("Skills require a terrain entity")
  terrain.terrain_type = "plane"
  terrain.terrain_generator = None
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)

  # One skill command term, plus the skill identity observation block.
  cfg.commands = {
    SKILL_COMMAND_NAME: _skills_command_cfg(
      profile,
      standing_height=standing_height,
      jump_period_s=jump_period_s,
      switch_prob=switch_prob,
      play=play,
    )
  }
  skill_observation = ObservationTermCfg(
    func=skills_utils.skill_identity_observation,
    params={"command_name": SKILL_COMMAND_NAME},
  )
  for group in ("actor", "critic"):
    cfg.observations[group].terms["command"] = ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": SKILL_COMMAND_NAME},
    )
    cfg.observations[group].terms["skill"] = skill_observation

  # Masked reward composition: nothing is retyped, only re-scoped.
  rewards: dict[str, RewardTermCfg] = {}
  rewards.update(_harvest_rewards(walk_rewards, "walk"))
  rewards.update(
    _harvest_rewards(
      handstand_rewards,
      "handstand",
      # The handstand's survival bonus is already provided by the walk set;
      # keeping one global alive term avoids stacking it four times.
      skip=("alive",),
    )
  )
  rewards.update(_harvest_rewards(getup_rewards, "getup"))
  rewards.update(_harvest_rewards(jump_rewards, "jump", skip=("alive",)))
  cfg.rewards = rewards

  # Terminations: contact ends walk/handstand/jump but not get-up. `time_out`
  # stays unmasked so every skill shares one episode budget.
  contact = cfg.terminations.get("illegal_contact")
  cfg.terminations = {
    "contact": TerminationTermCfg(
      func=skills_terminations.contact_termination,
      params={
        "sensor_name": "nonfoot_ground_touch",
        "force_threshold": contact.params.get("force_threshold", 10.0)
        if contact is not None
        else 10.0,
        "allowed_skills": ("walk", "handstand", "jump"),
        "command_name": SKILL_COMMAND_NAME,
      },
    ),
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
  }

  # Events: skill-dispatched reset, plus the startup randomization the shared
  # velocity environment already installed.
  startup_events = {
    name: cfg.events[name]
    for name in ("foot_friction", "encoder_bias", "base_com")
    if name in cfg.events
  }
  cfg.events = {
    "reset_skill": EventTermCfg(
      func=skills_events.reset_skill_episode,
      mode="reset",
      params={"command_name": SKILL_COMMAND_NAME, "standing": standing_resets},
    ),
    **startup_events,
  }

  cfg.episode_length_s = episode_length_s
  cfg.curriculum = {}
  if switch_stages and switch_prob == 0.0 and not play:
    cfg.curriculum["skill_switch"] = CurriculumTermCfg(
      func=skills_curriculums.skill_switch_probability,
      params={"command_name": SKILL_COMMAND_NAME, "stages": list(switch_stages)},
    )
  # More contacts than the single-skill tasks: a fallen get-up start plus
  # the handstand's tucked legs overflow 64 contact slots.
  cfg.sim.nconmax = 128
  cfg.sim.njmax = 800
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.mujoco.ccd_iterations = 200

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg


def register_skills_profile(
  profile: VelocityRobotProfile,
  *,
  task_id: str,
  experiment_name: str,
  base_height_target: float,
  rear_clearance_target: float,
  standing_height: float,
  jump_target_rise: float = 0.05,
  jump_period_s: float = 2.5,
  gravity_target: tuple[float, float, float] = (1.0, 0.0, 0.0),
  episode_length_s: float = 10.0,
  switch_prob: float = 0.0,
  switch_stages: tuple[tuple[int, float], ...] = (),
  max_iterations: int = 5_000,
  learning_rate: float = 1.0e-3,
  entropy_coef: float = 0.01,
  runner_cls: type | None = None,
) -> None:
  """Register a train and play task for one robot profile."""
  kwargs = {
    "base_height_target": base_height_target,
    "rear_clearance_target": rear_clearance_target,
    "standing_height": standing_height,
    "jump_target_rise": jump_target_rise,
    "jump_period_s": jump_period_s,
    "gravity_target": gravity_target,
    "episode_length_s": episode_length_s,
    "switch_prob": switch_prob,
    "switch_stages": switch_stages,
  }
  env_cfg = make_skills_env_cfg(profile, **kwargs, play=False)
  play_cfg = make_skills_env_cfg(profile, **kwargs, play=True)
  rl_cfg = make_ppo_runner_cfg(
    experiment_name,
    max_iterations=max_iterations,
    learning_rate=learning_rate,
    entropy_coef=entropy_coef,
  )
  register_mjlab_task(
    task_id=task_id,
    env_cfg=env_cfg,
    play_env_cfg=play_cfg,
    rl_cfg=rl_cfg,
    runner_cls=runner_cls or SkillOnPolicyRunner,
  )
