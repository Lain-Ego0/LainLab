"""The unified observation layout, defined once.

The whole single-policy design rests on one invariant (docs section 1): the
unified actor observation *starts with* each expert's native observation, field
for field, so an expert is driven by slicing rather than by an adapter. Every
consumer -- the collector, the actor architecture, the evaluator, the tests --
reads the widths from here, because getting one of them wrong does not raise:
it silently labels data with an observation the expert never saw.

Layout::

  [shared 48][jump twist 3][skill 6]
  \\_________/\\____________/\\______/
   |          |              one_hot(skill 4) + (phase_sin, phase_cos)
   |          the jump's commanded [vx, vy, wz]; zeros for other skills
   proprioception + command, identical to the walk / handstand / get-up tasks
"""

from __future__ import annotations

from src.tasks.skills.mdp.command import SKILL_NAMES

SHARED_OBS_DIM = 48
"""Proprioception plus the command block: the walk/get-up/handstand observation."""

JUMP_TWIST_DIM = 3
"""The jump's commanded ``[vx, vy, wz]``; zeros while a non-jump skill runs."""

SKILL_BLOCK_DIM = len(SKILL_NAMES) + 2
"""``one_hot(skill)`` followed by ``(phase_sin, phase_cos)``."""

STUDENT_OBS_DIM = SHARED_OBS_DIM + JUMP_TWIST_DIM + SKILL_BLOCK_DIM
"""Width of the unified actor observation the student policy consumes."""

SKILL_ONE_HOT_SLICE = slice(
  SHARED_OBS_DIM + JUMP_TWIST_DIM, SHARED_OBS_DIM + JUMP_TWIST_DIM + len(SKILL_NAMES)
)
"""Where the one-hot skill token lives inside the unified observation."""

EXPERT_OBS_DIM: dict[str, int] = {
  "walk": SHARED_OBS_DIM,
  "getup": SHARED_OBS_DIM,
  "handstand": SHARED_OBS_DIM,
  "jump": SHARED_OBS_DIM + JUMP_TWIST_DIM,
}
"""Native actor-observation width of each expert.

Walk, get-up and handstand read the shared fields; the jump expert additionally
consumes the twist block, which sits immediately after them.
"""
