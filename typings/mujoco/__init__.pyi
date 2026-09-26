"""Partial type stubs for the untyped ``mujoco`` wheel.

The published ``mujoco`` package ships neither a ``py.typed`` marker nor any
``.pyi`` files, so a type checker sees none of its members (``mujoco.MjModel``
becomes "not a known attribute"). This hand-written stub declares only the
members LainLab actually uses, which removes those false positives.

Extend it when new code starts using another member. It is consumed solely via
``[tool.pyright] stubPath = "typings"``; it is never imported, so it has no
runtime effect on training, simulation or playback.
"""

from typing import Any, Self

class MjModel:
  njnt: int
  nv: int
  body_mass: Any
  jnt_type: Any
  jnt_qposadr: Any
  jnt_dofadr: Any
  @staticmethod
  def from_file(filename: str, assets: dict[str, bytes] | None = None) -> Self: ...
  @staticmethod
  def from_xml_path(path: str) -> Self: ...
  def compile(self) -> Any: ...

class MjData:
  qpos: Any
  qvel: Any
  qacc: Any
  qfrc_inverse: Any
  site_xpos: Any
  ncon: int
  nefc: int
  efc_force: Any
  def __init__(self, model: MjModel) -> None: ...

class MjSpec:
  @staticmethod
  def from_file(filename: str, assets: dict[str, bytes] | None = None) -> Self: ...
  def compile(self) -> Any: ...

class mjtObj:
  mjOBJ_SITE: int
  mjOBJ_JOINT: int
  mjOBJ_BODY: int
  mjOBJ_ACTUATOR: int

class mjtJoint:
  mjJNT_HINGE: int
  mjJNT_SLIDE: int
  mjJNT_FREE: int

class mjtGeom:
  mjGEOM_CAPSULE: int
  mjGEOM_CYLINDER: int

def mj_name2id(model: MjModel, objtype: Any, name: str) -> int: ...
def mj_id2name(model: MjModel, objtype: Any, index: int) -> str | None: ...
def mj_forward(model: MjModel, data: MjData) -> None: ...
def mj_inverse(model: MjModel, data: MjData) -> None: ...
def mj_resetDataKeyframe(model: MjModel, data: MjData, key: int) -> None: ...
def mj_normalizeQuat(model: MjModel, qpos: Any) -> None: ...
def mju_quat2Mat(dest: Any, quat: Any) -> None: ...

viewer: Any
