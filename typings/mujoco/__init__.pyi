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

class mjtJoint:
  mjJNT_HINGE: int

def mj_name2id(model: MjModel, objtype: Any, name: str) -> int: ...
def mj_id2name(model: MjModel, objtype: Any, index: int) -> str | None: ...
def mj_forward(model: MjModel, data: MjData) -> None: ...
def mj_inverse(model: MjModel, data: MjData) -> None: ...
def mju_quat2Mat(dest: Any, quat: Any) -> None: ...

viewer: Any
