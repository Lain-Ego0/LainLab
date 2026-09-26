"""Single-policy multi-skill task type.

Composes the per-skill MDPs into one environment whose observation is the shared
48-field proprioceptive block plus a skill identity block, so a single network
can serve every skill.
"""

from .core import make_skills_env_cfg as make_skills_env_cfg
from .core import register_skills_profile as register_skills_profile
from .mdp.command import SKILL_NAMES as SKILL_NAMES
from .mdp.command import SkillCommandCfg as SkillCommandCfg
from .mdp.command import SkillCommandTerm as SkillCommandTerm
