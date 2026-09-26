"""OpenDoge single-policy multi-skill task.

One policy, one network, four skills on the same robot:

``walk / handstand / getup / jump``

All targets come from the per-skill tasks, which take them from
``docs/opendoge_multiskill_targets.md``:

- ``standing_height`` 0.151 m (settled standing base height)
- handstand base height 0.220 m, raised rear feet 0.341 m, gravity target [1,0,0]
- jump apex 0.05 m above standing on a 2.5 s phase clock
"""

from src.tasks.multiskill import register_skills_profile

from .handstand import HANDSTAND_BASE_HEIGHT_TARGET, HANDSTAND_REAR_CLEARANCE_TARGET
from .jump import JUMP_PERIOD_S, JUMP_STANDING_HEIGHT, JUMP_TARGET_RISE
from .velocity import OPENDOGE_VELOCITY_PROFILES

OPENDOGE_SKILLS_PROFILE = OPENDOGE_VELOCITY_PROFILES[0]

SKILLS_EPISODE_LENGTH_S = 10.0
# Fine-tuning starts from a behaviour-cloned policy, so the learning rate is a
# third of the from-scratch value and exploration is kept tight around it.
SKILLS_LEARNING_RATE = 3.0e-4
SKILLS_ENTROPY_COEF = 0.002
# Mid-episode skill switching ramps in over training: skills first, transitions
# later. The probability is **per environment step** (100 Hz here), so it must be
# tiny to mean anything sensible: 0.0015 is about one switch every 6.7 s, which
# is roughly once per episode. A value like 0.05 would switch every 0.2 s and
# make every skill's settling reward unachievable.
# `common_step_counter` advances by 24 per iteration.
SKILLS_SWITCH_STAGES = (
  (0, 0.0),
  (800 * 24, 0.0005),
  (1800 * 24, 0.0015),
)


def register_skills_tasks(
  *, switch_prob: float = 0.0, max_iterations: int = 5_000
) -> None:
  register_skills_profile(
    OPENDOGE_SKILLS_PROFILE,
    task_id="LainLab-OpenDoge-Skills-Flat",
    experiment_name="lainlab_opendoge_skills",
    base_height_target=HANDSTAND_BASE_HEIGHT_TARGET,
    rear_clearance_target=HANDSTAND_REAR_CLEARANCE_TARGET,
    standing_height=JUMP_STANDING_HEIGHT,
    jump_target_rise=JUMP_TARGET_RISE,
    jump_period_s=JUMP_PERIOD_S,
    episode_length_s=SKILLS_EPISODE_LENGTH_S,
    switch_prob=switch_prob,
    switch_stages=SKILLS_SWITCH_STAGES,
    max_iterations=max_iterations,
    learning_rate=SKILLS_LEARNING_RATE,
    entropy_coef=SKILLS_ENTROPY_COEF,
  )
