"""LainLab task registrations.

Importing this package registers every LainLab task in mjlab's shared registry.
Generic task types live under task-type packages; robot-specific composition
lives under ``tasks.robots``.
"""

from . import amp as amp
from . import handstand as handstand
from . import jump as jump
from . import recover as recover
from . import skills as skills
from . import tracking as tracking
from . import velocity as velocity
from .robots import g1 as g1
from .robots import go2 as go2
from .robots import opendoge as opendoge
from .robots import unitree as unitree

unitree.register_velocity_tasks()
go2.register_tasks()
opendoge.register_velocity_tasks()
opendoge.register_getup_tasks()
opendoge.register_handstand_tasks()
opendoge.register_jump_tasks()
opendoge.register_recover_tasks()
opendoge.register_skills_tasks()
g1.register_tracking_tasks()
