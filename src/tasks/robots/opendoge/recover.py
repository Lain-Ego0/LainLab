"""OpenDoge recover-to-walk task.

Same reward set and velocity command as the flat walking task, but half of every
batch starts from a random fall, so the expert covers "arrive at the walk token
from any pose" -- the states the single policy actually meets when a skill is
switched at runtime.
"""

from src.tasks.recover import register_recover_profile

from .velocity import OPENDOGE_VELOCITY_PROFILES

OPENDOGE_RECOVER_PROFILE = OPENDOGE_VELOCITY_PROFILES[0]


def register_recover_tasks() -> None:
  register_recover_profile(
    OPENDOGE_RECOVER_PROFILE,
    task_id="LainLab-OpenDoge-Recover-Walk",
    experiment_name="lainlab_opendoge_recover_walk",
    fallen_probability=0.5,
    max_iterations=5_000,
  )
