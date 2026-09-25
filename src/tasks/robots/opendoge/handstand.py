"""OpenDoge front-support handstand task.

Targets are the calibrated values from ``docs/opendoge_multiskill_targets.md``
(regenerate with ``uv run opendoge-calibrate --write``):

- body pitched to vertical, so ``projected_gravity_b -> [1, 0, 0]``
- base height ``0.220 m`` in the handstand (settled standing height is 0.151 m)
- raised rear feet held ``0.341 m`` above the ground

Static effort at that pose is 0.76% of the actuator limits, so the task is
balance-limited rather than torque-limited.
"""

from src.tasks.handstand import register_handstand_profile

from .velocity import OPENDOGE_VELOCITY_PROFILES

OPENDOGE_HANDSTAND_PROFILE = OPENDOGE_VELOCITY_PROFILES[0]

HANDSTAND_BASE_HEIGHT_TARGET = 0.220
HANDSTAND_REAR_CLEARANCE_TARGET = 0.341
HANDSTAND_GRAVITY_TARGET = (1.0, 0.0, 0.0)


def register_handstand_tasks() -> None:
  register_handstand_profile(
    OPENDOGE_HANDSTAND_PROFILE,
    task_id="LainLab-OpenDoge-Handstand",
    experiment_name="lainlab_opendoge_handstand",
    base_height_target=HANDSTAND_BASE_HEIGHT_TARGET,
    rear_clearance_target=HANDSTAND_REAR_CLEARANCE_TARGET,
    gravity_target=HANDSTAND_GRAVITY_TARGET,
    max_iterations=5_000,
  )
