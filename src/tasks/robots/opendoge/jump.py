"""OpenDoge in-place vertical jump task.

Targets are the calibrated values from ``docs/opendoge_multiskill_targets.md``
(regenerate with ``uv run opendoge-calibrate --write``):

- usable leg-extension stroke from standing is only ``0.0342 m``
- the whole push-off lasts about ``26.5 ms``, i.e. 2.6 control steps at 100 Hz,
  so the takeoff is pre-programmed against a phase clock rather than closed-loop
- design target: apex ``0.03 - 0.08 m`` above standing, recover to standing

The settled standing base height is ``0.151 m``; ``target_rise`` is measured
from there.

Task rewards are confined to windows of the phase cycle (flight in
``[0.10, 0.40]``, recovery in ``[0.45, 1.0]``). Without those windows the policy
farms the flight reward by bouncing continuously -- air time becomes its own
objective -- so the cycle is what forces "jump once, then recover to standing".
"""

from src.tasks.jump import register_jump_profile

from .velocity import OPENDOGE_VELOCITY_PROFILES

OPENDOGE_JUMP_PROFILE = OPENDOGE_VELOCITY_PROFILES[0]

JUMP_STANDING_HEIGHT = 0.151
JUMP_TARGET_RISE = 0.05
JUMP_PERIOD_S = 2.5


def register_jump_tasks() -> None:
  register_jump_profile(
    OPENDOGE_JUMP_PROFILE,
    task_id="LainLab-OpenDoge-Jump",
    experiment_name="lainlab_opendoge_jump",
    standing_height=JUMP_STANDING_HEIGHT,
    target_rise=JUMP_TARGET_RISE,
    period_s=JUMP_PERIOD_S,
    max_iterations=5_000,
  )
