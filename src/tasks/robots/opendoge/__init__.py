"""OpenDoge robot task composition package."""

from .getup import register_getup_tasks as register_getup_tasks
from .handstand import register_handstand_tasks as register_handstand_tasks
from .jump import register_jump_tasks as register_jump_tasks
from .skills import register_skills_tasks as register_skills_tasks
from .velocity import OPENDOGE_VELOCITY_PROFILES as OPENDOGE_VELOCITY_PROFILES
from .velocity import register_velocity_tasks as register_velocity_tasks
