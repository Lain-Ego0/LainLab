"""OpenDoge fall-recovery task."""

from src.tasks.getup import register_getup_profile

from .velocity import OPENDOGE_VELOCITY_PROFILES

OPENDOGE_GETUP_PROFILE = OPENDOGE_VELOCITY_PROFILES[0]


def register_getup_tasks() -> None:
  register_getup_profile(
    OPENDOGE_GETUP_PROFILE,
    task_id="LainLab-OpenDoge-Getup",
    experiment_name="lainlab_opendoge_getup",
    base_height_target=0.158,
    max_iterations=5_000,
  )
