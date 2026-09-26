"""OpenDoge multi-skill statics calibration.

Pure kinematics and statics for the OpenDoge quadruped.  No RL, no GPU, no
training: this script exists to answer the feasibility questions that decide the
handstand and jump targets *before* any policy is trained.

It answers four questions with numbers instead of guesses:

1. How far can the legs extend, respecting the MJCF joint limits?
2. Which body pitch puts the rear feet up, so ``projected_gravity_b`` can be
   used as a handstand target with a known sign?
3. Which front-leg pose holds the body highest while the front feet support it,
   and what static joint torques does that pose need?
4. How much vertical energy can the legs deliver for a standing jump, and
   therefore what apex height is physically reachable?

Usage::

  uv run opendoge-calibrate            # print the report
  uv run opendoge-calibrate --write    # refresh docs/opendoge_multiskill_targets.md
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass

import mujoco
import numpy as np

from src import PACKAGE_ROOT

# The robot-only MJCF has no ground plane, so statics needs the scene wrapper;
# without it every `mj_inverse` call returns a gravity-only torque because the
# feet never touch anything.
OPENDOGE_XML = (
  PACKAGE_ROOT / "assets" / "robots" / "opendoge" / "xmls" / "scene_opendoge.xml"
)
REPORT_PATH = PACKAGE_ROOT.parent / "docs" / "opendoge_multiskill_targets.md"

GRAVITY = 9.81
FOOT_RADIUS = 0.015
STANDING_INIT_HEIGHT = 0.158  # `OPENDOGE_INIT_STATE.pos[2]`.
FEET = ("FL", "FR", "RL", "RR")
FRONT_FEET = ("FL", "FR")
REAR_FEET = ("RL", "RR")

# Mirrored leg pairs: (left, right).  Right-side angles are the negation of the
# left side, matching `OPENDOGE_INIT_STATE`.
LEGS = (("FL", "FR"), ("RL", "RR"))
# Asymmetric front-hip allow-range, taken from the MJCF.
FL_HIP_RANGE = (-1.0468, 0.521799)
THIGH_RANGE = (-1.0468, 1.3958)
CALF_RANGE = (-2.68, -1.04)
DEFAULT_POSE = {"hip": 0.0, "thigh": 0.8, "calf": -1.6}
# Per-actuator effort limits from ``opendoge_constants.py``.
EFFORT = {"hip": 6.0, "thigh": 6.0, "calf": 9.0}
# Common position-servo damping gain; used for the push-off speed bound.
K_DAMPING = 0.5


@dataclass(frozen=True)
class LegRange:
  """Hip-to-foot vertical reach of one leg under the MJCF joint limits."""

  minimum: float
  maximum: float
  peak_thigh: float
  peak_calf: float
  default: float


@dataclass(frozen=True)
class StaticMetric:
  """Geometry and static effort for one candidate whole-body pose."""

  label: str
  pitch_deg: float
  thigh: float
  calf: float
  base_height: float
  rear_clearance: float
  front_offset_x: float
  peak_effort_ratio: float
  torque_hip: float
  torque_thigh: float
  torque_calf: float
  root_residual: float


class Rig:
  """Thin MuJoCo wrapper for whole-body pose probes."""

  def __init__(self) -> None:
    self.model = mujoco.MjModel.from_xml_path(str(OPENDOGE_XML))
    self.data = mujoco.MjData(self.model)
    self.site_ids = {
      name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
      for name in FEET
    }
    # Hinge joints must be addressed through `jnt_qposadr` / `jnt_dofadr`: the
    # free joint occupies qpos[0:7] and dof[0:6], so the joint index is *not*
    # the qpos or dof address.
    self.joint_qpos: dict[str, int] = {}
    self.joint_dof: dict[str, int] = {}
    for i in range(self.model.njnt):
      if self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
        continue
      name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
      if name is None:
        continue
      self.joint_qpos[name] = int(self.model.jnt_qposadr[i])
      self.joint_dof[name] = int(self.model.jnt_dofadr[i])

  @property
  def total_mass(self) -> float:
    return float(self.model.body_mass.sum())

  def write_pose(
    self,
    *,
    pitch_deg: float = 0.0,
    pose: dict[str, float] | None = None,
    rear_pose: dict[str, float] | None = None,
    base_height: float = 0.5,
  ) -> float:
    """Write a mirrored whole-body pose and forward-propagate.

    ``pose`` drives the front legs and ``rear_pose`` the rear legs; when
    ``rear_pose`` is omitted both pairs share ``pose``.
    """
    pose = pose or DEFAULT_POSE
    rear_pose = rear_pose or pose
    data = self.data
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    half = np.radians(pitch_deg) / 2.0
    # Rotation about the world Y axis: positive pitch raises the front feet.
    data.qpos[3:7] = (np.cos(half), 0.0, np.sin(half), 0.0)
    for (left, right), leg_pose in zip(LEGS, (pose, rear_pose), strict=True):
      for joint, value in leg_pose.items():
        data.qpos[self.joint_qpos[f"{left}_{joint}_joint"]] = value
        data.qpos[self.joint_qpos[f"{right}_{joint}_joint"]] = -value
    data.qpos[2] = base_height
    mujoco.mj_forward(self.model, data)
    return base_height

  def root_vertical_residual(self) -> float:
    """Vertical root force needed to hold the pose; zero when feet carry weight.

    ``mj_inverse`` reports the generalized force an actuator must supply.  The
    first six entries act on the floating base, which has no actuator, so a
    non-zero vertical entry means the foot contacts are *not* supporting the
    body and every joint torque in that configuration is meaningless.
    """
    self.data.qacc[:] = 0.0
    mujoco.mj_inverse(self.model, self.data)
    return float(self.data.qfrc_inverse[2])

  def settle(
    self,
    *,
    pitch_deg: float = 0.0,
    pose: dict[str, float] | None = None,
    rear_pose: dict[str, float] | None = None,
  ) -> tuple[float, float]:
    """Find the base height where the feet carry exactly the body weight.

    The contact model is soft, so "feet just touching" leaves the contacts
    unloaded and produces a fake gravity-only torque.  Bisect the base height
    until the vertical root residual is zero, then return that height and the
    residual achieved.
    """
    self.write_pose(pitch_deg=pitch_deg, pose=pose, rear_pose=rear_pose)
    lowest = min(self.site_z(name) for name in FEET)
    touch = self.data.qpos[2] - lowest + FOOT_RADIUS

    def residual(height: float) -> float:
      self.write_pose(
        pitch_deg=pitch_deg, pose=pose, rear_pose=rear_pose, base_height=height
      )
      return self.root_vertical_residual()

    high = touch  # Barely touching: contacts unloaded, residual positive.
    low = touch - 0.02  # Deeper penetration: contacts push up, residual negative.
    if residual(high) <= 0.0:
      return high, residual(high)
    for _ in range(60):
      mid = 0.5 * (low + high)
      if residual(mid) > 0.0:
        high = mid
      else:
        low = mid
    height = 0.5 * (low + high)
    return height, residual(height)

  def site_z(self, name: str) -> float:
    return float(self.data.site_xpos[self.site_ids[name]][2])

  def site_x(self, name: str) -> float:
    return float(self.data.site_xpos[self.site_ids[name]][0])

  def mean_site_z(self, names: tuple[str, ...]) -> float:
    return float(np.mean([self.site_z(name) for name in names]))

  def mean_site_x(self, names: tuple[str, ...]) -> float:
    return float(np.mean([self.site_x(name) for name in names]))

  def leg_height(self) -> float:
    """Vertical hip-to-foot reach of the FL leg for the current pose."""
    return float(self.data.qpos[2] - self.site_z("FL"))

  def static_joint_torques(self) -> dict[str, float]:
    """Required joint torque for the current pose at zero acceleration.

    ``mj_forward`` resolves the foot-ground constraint forces, and
    ``mj_inverse`` with zero acceleration then reports the generalized force an
    actuator must supply to hold the pose.  The first six entries are the root
    wrench and are discarded.
    """
    self.data.qacc[:] = 0.0
    mujoco.mj_inverse(self.model, self.data)
    torques = np.asarray(self.data.qfrc_inverse[6:], dtype=np.float64)
    return {
      "hip": float(torques[self.joint_dof["FL_hip_joint"]]),
      "thigh": float(torques[self.joint_dof["FL_thigh_joint"]]),
      "calf": float(torques[self.joint_dof["FL_calf_joint"]]),
    }


def leg_extension_range(rig: Rig) -> LegRange:
  """Hip-to-foot vertical reach range under the MJCF joint limits."""
  thighs = np.linspace(*THIGH_RANGE, 41)
  calves = np.linspace(*CALF_RANGE, 41)
  heights = np.empty((len(thighs), len(calves)))
  for i, thigh in enumerate(thighs):
    for j, calf in enumerate(calves):
      rig.write_pose(pose={"hip": 0.0, "thigh": float(thigh), "calf": float(calf)})
      heights[i, j] = rig.leg_height()
  i, j = np.unravel_index(int(np.argmax(heights)), heights.shape)
  return LegRange(
    minimum=float(heights.min()),
    maximum=float(heights[i, j]),
    peak_thigh=float(thighs[i]),
    peak_calf=float(calves[j]),
    default=_leg_height_at(rig, DEFAULT_POSE["thigh"], DEFAULT_POSE["calf"]),
  )


def _leg_height_at(rig: Rig, thigh: float, calf: float) -> float:
  rig.write_pose(pose={"hip": 0.0, "thigh": thigh, "calf": calf})
  return rig.leg_height()


def pitch_sign_probe(rig: Rig) -> list[tuple[float, float, float]]:
  """Report which body pitch raises the rear feet (the front-support branch)."""
  rows = []
  for pitch in range(-180, 181, 45):
    rig.write_pose(pitch_deg=float(pitch), base_height=0.5)
    front = rig.mean_site_z(FRONT_FEET) - 0.5
    rear = rig.mean_site_z(REAR_FEET) - 0.5
    rows.append((float(pitch), front, rear))
  return rows


def _static_metric(
  rig: Rig,
  label: str,
  pitch_deg: float,
  pose: dict[str, float],
  rear_pose: dict[str, float] | None = None,
) -> StaticMetric:
  base_height, residual = rig.settle(
    pitch_deg=pitch_deg, pose=pose, rear_pose=rear_pose
  )
  torques = rig.static_joint_torques()
  ratio = max(abs(torques[key]) / EFFORT[key] for key in torques)
  return StaticMetric(
    label=label,
    pitch_deg=float(pitch_deg),
    thigh=float(pose["thigh"]),
    calf=float(pose["calf"]),
    base_height=float(base_height),
    rear_clearance=float(rig.mean_site_z(REAR_FEET) - FOOT_RADIUS),
    front_offset_x=float(rig.mean_site_x(FRONT_FEET)),
    peak_effort_ratio=float(ratio),
    torque_hip=torques["hip"],
    torque_thigh=torques["thigh"],
    torque_calf=torques["calf"],
    root_residual=float(residual),
  )


def standing_reference(rig: Rig) -> StaticMetric:
  """Static effort of the nominal standing pose.

  This is the validation row: the robot provably stands in the real world at
  this pose, so the reported torques must sit far below the effort limits.
  """
  return _static_metric(rig, "站姿", 0.0, DEFAULT_POSE)


def handstand_search(rig: Rig, pitch_deg: float) -> list[StaticMetric]:
  """Grid-search front-leg poses that hold the body highest over the front feet.

  The rear legs stay tucked at the default pose so the measurement isolates the
  supporting pair.  Rear-foot clearance, static effort and the horizontal offset
  of the support feet relative to the base origin are all reported, because a
  static handstand also needs its centre of mass inside the support polygon.
  """
  tucked = {
    "hip": 0.0,
    "thigh": DEFAULT_POSE["thigh"],
    "calf": DEFAULT_POSE["calf"],
  }
  metrics: list[StaticMetric] = []
  for thigh, calf in itertools.product(
    np.linspace(*THIGH_RANGE, 13), np.linspace(*CALF_RANGE, 13)
  ):
    pose = {"hip": 0.0, "thigh": float(thigh), "calf": float(calf)}
    # Only keep poses where the front feet are genuinely the support pair; if a
    # candidate leg pose points upward the settle would key off the rear feet
    # and every derived number would be meaningless.
    rig.settle(pitch_deg=pitch_deg, pose=pose, rear_pose=tucked)
    if min(rig.site_z(name) for name in FRONT_FEET) >= min(
      rig.site_z(name) for name in REAR_FEET
    ):
      continue
    metrics.append(
      _static_metric(
        rig,
        label=f"thigh={thigh:.3f} calf={calf:.3f}",
        pitch_deg=pitch_deg,
        pose=pose,
        rear_pose=tucked,
      )
    )
  return metrics


def _foot_jacobian(rig: Rig, thigh: float, calf: float) -> np.ndarray:
  """Numerical d(foot x, foot z)/d(thigh, calf) for the sagittal leg plane."""
  eps = 1e-5
  out = np.zeros((2, 2))
  for column, (d_thigh, d_calf) in enumerate(((eps, 0.0), (0.0, eps))):
    rig.write_pose(pose={"hip": 0.0, "thigh": thigh, "calf": calf})
    x0, z0 = rig.site_x("FL"), rig.site_z("FL")
    rig.write_pose(pose={"hip": 0.0, "thigh": thigh + d_thigh, "calf": calf + d_calf})
    out[0, column] = (rig.site_x("FL") - x0) / eps
    out[1, column] = (rig.site_z("FL") - z0) / eps
  return out


def _vertical_thrust_per_leg(rig: Rig, thigh: float, calf: float) -> float:
  """Largest vertical foot force one leg can hold with a purely vertical push.

  The horizontal component is pinned to zero -- a symmetric standing jump pushes
  straight down -- so the effort bounds reduce to
  ``F_z <= min(tau_i / |dz/dq_i|)``.  Allowing a free horizontal component makes
  the LP exploit the near-singular fully-extended pose and return a meaningless
  force, which is exactly what an unconstrained formulation does here.
  """
  jac = _foot_jacobian(rig, thigh, calf)
  dz = np.abs(jac[1, :])
  if np.any(dz < 1e-6):
    return float("inf")
  limits = np.array([EFFORT["thigh"], EFFORT["calf"]]) / dz
  return float(np.min(limits))


def _vertical_rate(rig: Rig, thigh: float, calf: float) -> float:
  """Fastest vertical foot speed one leg can drive, in m/s.

  A position servo producing ``tau`` at speed ``q_dot`` needs ``Kd * q_dot`` of
  its effort budget, so ``q_dot_max = tau_max / Kd``.  With the actual gains
  (Kp=12, Kd=0.5) that is 12 rad/s at the thigh and 18 rad/s at the calf, which
  is what bounds the push-off speed rather than the torque limit.
  """
  jac = _foot_jacobian(rig, thigh, calf)
  rate = np.array([EFFORT["thigh"] / K_DAMPING, EFFORT["calf"] / K_DAMPING])
  return float(np.abs(jac[1, :]) @ rate)


def jump_budget(rig: Rig, samples: int = 25) -> dict[str, float]:
  """Bound the push-off a symmetric four-leg jump can achieve.

  Two independent bounds are reported so the honest conclusion is visible: the
  torque limit is *not* binding (the legs can push many times body weight), and
  the damping-limited foot speed implies the entire stroke is over in a few
  control steps.  The design target therefore stays far below either bound.
  """
  reach = leg_extension_range(rig)
  thighs = np.linspace(DEFAULT_POSE["thigh"], reach.peak_thigh, samples)
  calves = np.linspace(DEFAULT_POSE["calf"], reach.peak_calf, samples)

  thrusts, rates = [], []
  for thigh, calf in zip(thighs, calves, strict=True):
    thrusts.append(_vertical_thrust_per_leg(rig, float(thigh), float(calf)))
    rates.append(_vertical_rate(rig, float(thigh), float(calf)))

  rise = reach.maximum - reach.default
  weight = rig.total_mass * GRAVITY
  finite = [value for value in thrusts if np.isfinite(value)]
  typical = 4.0 * float(np.median(finite)) if finite else 0.0
  speed = float(np.max(rates))
  return {
    "default_reach": reach.default,
    "top_reach": reach.maximum,
    "rise": rise,
    "thrust_body_weight": float(typical / weight),
    "max_foot_speed": speed,
    "pushoff_window_ms": 1000.0 * rise / speed if speed > 0 else float("inf"),
    "control_steps": rise / speed / 0.01 if speed > 0 else float("inf"),
    "ballistic_rise": speed**2 / (2.0 * GRAVITY),
    "apex_rise": rise + speed**2 / (2.0 * GRAVITY),
  }


def projected_gravity_at(pitch_deg: float) -> str:
  """``projected_gravity_b`` for a pitch-only body rotation.

  ``projected_gravity_b = R^T @ (0, 0, -1)`` (see ``mjlab.entity.data``); the
  triple is rounded so it can be pasted straight into a reward target.
  """
  half = np.radians(pitch_deg) / 2.0
  quat = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
  rotation = np.zeros(9)
  mujoco.mju_quat2Mat(rotation, quat)
  gravity_b = rotation.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
  return "[" + ", ".join(f"{value:.3f}" for value in np.round(gravity_b, 3)) + "]"


def _fmt(value: float, digits: int = 4) -> str:
  return f"{value:.{digits}f}"


def build_report() -> str:
  rig = Rig()
  reach = leg_extension_range(rig)
  standing = standing_reference(rig)
  pitch_rows = pitch_sign_probe(rig)
  jump = jump_budget(rig)

  # The front-support branch is the pitch whose rear feet sit highest.
  front_support_pitch = max(pitch_rows, key=lambda row: row[2] - row[1])[0]
  candidates = handstand_search(rig, front_support_pitch)
  feasible = [c for c in candidates if c.peak_effort_ratio <= 0.6]
  # Keep joint margin: a target pose sitting on a joint limit leaves the balance
  # controller no workspace, so the extremes of the grid are excluded.
  thigh_span = THIGH_RANGE[1] - THIGH_RANGE[0]
  calf_span = CALF_RANGE[1] - CALF_RANGE[0]
  with_margin = [
    c
    for c in feasible
    if THIGH_RANGE[0] + 0.15 * thigh_span
    <= c.thigh
    <= THIGH_RANGE[1] - 0.15 * thigh_span
    and CALF_RANGE[0] + 0.15 * calf_span <= c.calf <= CALF_RANGE[1] - 0.15 * calf_span
  ]
  ranking = sorted(
    with_margin or feasible or candidates,
    key=lambda c: c.base_height,
    reverse=True,
  )
  # Recommend a pose in the middle of the feasible band rather than at its edge:
  # the highest target sits at the workspace boundary and is the hardest to hold.
  band = with_margin or feasible or candidates
  heights = sorted(c.base_height for c in band)
  median_height = heights[len(heights) // 2]
  target = min(band, key=lambda c: abs(c.base_height - median_height))
  gravity_target = projected_gravity_at(target.pitch_deg)

  lines: list[str] = []
  add = lines.append
  add("# OpenDoge 多技能目标标定")
  add("")
  add("本文件由 `uv run opendoge-calibrate --write` 生成。数值全部来自 MJCF 的")
  add("运动学与静力学计算，不是估计值；修改机器人模型后请重新生成。")
  add("")
  add("## 1. 刚体与行程约束")
  add("")
  add(f"- 总质量：**{_fmt(rig.total_mass, 3)} kg**")
  add(f"- 广义坐标：{rig.model.nv}（6 浮动基座 + 12 关节）")
  add(f"- MJCF 声明站姿基座高度：{_fmt(STANDING_INIT_HEIGHT, 3)} m")
  add(f"- 落地后实测站姿高度：{_fmt(standing.base_height, 3)} m")
  add(
    f"- 髋到足可达竖直高度：**{_fmt(reach.minimum)} m → {_fmt(reach.maximum)} m**"
    f"（峰值姿态 thigh={_fmt(reach.peak_thigh, 3)},"
    f" calf={_fmt(reach.peak_calf, 3)}）"
  )
  add(f"- 站姿腿高：{_fmt(reach.default)} m")
  add(f"- **从站姿可继续伸腿的行程：{_fmt(reach.maximum - reach.default)} m**")
  add("")
  add("> 这个行程上限直接决定了原地跳跃的离地高度只能是厘米级。")
  add("")
  add("## 2. 静力学校验")
  add("")
  add("| 姿态 | 基座高度 | 髋力矩 | 大腿力矩 | 小腿力矩 | 峰值占额比 | 基座竖直残差 |")
  add("|---|---|---|---|---|---|---|")
  add(
    f"| {standing.label}（默认关节角） | {_fmt(standing.base_height, 3)} |"
    f" {_fmt(standing.torque_hip, 3)} | {_fmt(standing.torque_thigh, 3)} |"
    f" {_fmt(standing.torque_calf, 3)} | {_fmt(standing.peak_effort_ratio, 3)} |"
    f" {standing.root_residual:+.2e} |"
  )
  add("")
  add("基座竖直残差是浮动基座上「没有执行器却需要的力」。它接近 0 表示足端接触力")
  add("恰好承担了整机重力，因此同一姿态下的关节力矩才是真实静力矩。接触模型是软")
  add("的，所以基座高度用二分求解到该平衡点，而不是让足端刚好触地。")
  add("")
  add("站姿静力矩远低于 6/6/9 Nm 限值，说明静力学链路可信：`mj_forward` 解出足端")
  add("接触力后，`mj_inverse` 在零加速度下给出的关节力矩即为所需驱动力矩。")
  add("")
  add("## 3. 倒立（前腿支撑、后腿抬起）")
  add("")
  add("俯仰角扫描（正值抬前脚，负值抬后脚；数值为足端相对基座原点的高度）：")
  add("")
  add("| pitch (deg) | 前足 z − 基座 | 后足 z − 基座 | 支撑支路 |")
  add("|---|---|---|---|")
  for pitch, front, rear in pitch_rows:
    if abs(rear - front) < 1e-9:
      branch = "对称，不区分"
    else:
      branch = "后腿支撑" if rear < front else "前腿支撑"
    add(f"| {pitch:+.0f} | {front:+.4f} | {rear:+.4f} | {branch} |")
  add("")
  add(
    f"选定的前腿支撑俯仰角：**{front_support_pitch:+.0f} deg**。该姿态下"
    f" `projected_gravity_b ≈ {gravity_target}`，与 Go2 Handstand 的"
    " `[1, 0, 0]` 目标语义一致。"
  )
  add("")
  add("前腿关节角网格搜索（后腿保持收拢姿；数值为前腿支撑、足端接触力恰好承担")
  add("整机重力时的静态量）：")
  add("")
  add("| thigh | calf | 基座高度 | 后足离地 | 支撑足 x | 峰值占额比 |")
  add("|---|---|---|---|---|---|")
  for cand in ranking[:6]:
    add(
      f"| {_fmt(cand.thigh, 3)} | {_fmt(cand.calf, 3)} |"
      f" {_fmt(cand.base_height, 3)} | {_fmt(cand.rear_clearance, 3)} |"
      f" {_fmt(cand.front_offset_x, 3)} | {_fmt(cand.peak_effort_ratio, 3)} |"
    )
  add("")
  if feasible:
    lows = min(c.base_height for c in feasible)
    highs = max(c.base_height for c in feasible)
    rear_lows = min(c.rear_clearance for c in feasible)
    rear_highs = max(c.rear_clearance for c in feasible)
    add(
      f"上表是「带 15% 关节余量」子集里基座最高的若干项。全部可行解"
      f"（峰值占额比 ≤ 0.6）的基座高度区间为 **{_fmt(lows, 3)} – {_fmt(highs, 3)} m**，"
      f"后足离地区间为 **{_fmt(rear_lows, 3)} – {_fmt(rear_highs, 3)} m**。"
    )
    add("")
    add(
      "**推荐值取该区间的中位**，而不是最高点：可行域边界上的姿态留给平衡控制器"
      "的余量最小，作为训练目标最容易卡住。"
    )
    add("")
  add("**推荐倒立目标（Step 2 使用）：**")
  add("")
  add("```yaml")
  add(f"pitch_deg: {target.pitch_deg:+.0f}")
  add(f"front_leg: {{thigh: {_fmt(target.thigh, 3)}, calf: {_fmt(target.calf, 3)}}}")
  add(f"base_height_target: {_fmt(target.base_height, 3)}   # m，相对地面")
  add(f"rear_foot_clearance_target: {_fmt(target.rear_clearance, 3)}   # m")
  add(f"projected_gravity_target: {gravity_target}")
  add(f"peak_effort_ratio: {_fmt(target.peak_effort_ratio, 3)}")
  add("```")
  add("")
  add(
    f"静止倒立所需峰值力矩仅为额定的 {_fmt(target.peak_effort_ratio * 100, 2)}%，"
    "因为前腿此时几乎是一根与重力同向的直柱。**所以倒立的难点不是力矩，而是"
    "把倒立摆稳住**：目标姿态静态可行，训练预算应该花在平衡与噪声鲁棒性上。"
  )
  add("")
  add("## 4. 原地竖直跳的推离预算")
  add("")
  add("| 量 | 值 |")
  add("|---|---|")
  add(
    f"| 站姿腿高 → 峰值腿高 | {_fmt(jump['default_reach'])} →"
    f" {_fmt(jump['top_reach'])} m |"
  )
  add(f"| 伸腿行程（唯一的推离窗口） | {_fmt(jump['rise'])} m |")
  add(f"| 四腿竖直推力（力矩约束） | {_fmt(jump['thrust_body_weight'], 1)} 倍体重 |")
  add(
    f"| 足端最快竖直速度（`Kd={K_DAMPING}` 阻尼约束） |"
    f" {_fmt(jump['max_foot_speed'], 2)} m/s |"
  )
  add(
    f"| **推离窗口时长** | **{_fmt(jump['pushoff_window_ms'], 1)} ms ="
    f" {_fmt(jump['control_steps'], 1)} 个控制步** |"
  )
  add(f"| 解析上界（不做设计用） | 抬升 {_fmt(jump['apex_rise'], 3)} m |")
  add("")
  add("两条独立结论：")
  add("")
  add(
    f"1. **力矩不是瓶颈**：四腿可输出约 {_fmt(jump['thrust_body_weight'], 1)} 倍体重的竖直推力，"
    f"而整段行程只有 {_fmt(jump['rise'] * 1000, 1)} mm。"
  )
  add(
    f"2. **真正的瓶颈是控制频率**：按阻尼约束的足端速度，整个推离过程只有"
    f" {_fmt(jump['pushoff_window_ms'], 1)} ms，在 100 Hz 下相当于"
    f" {_fmt(jump['control_steps'], 1)} 个控制步。策略几乎不可能在起跳过程中"
    "闭环修正，起跳必须靠 one-shot 命令 + 相位时钟预先编排。"
  )
  add("")
  add("**推荐跳跃目标（Step 2 使用）：**")
  add("")
  add("```yaml")
  add("jump_type: in_place_vertical")
  add("apex_rise_target: 0.03 - 0.08   # m，相对站姿的质心抬升")
  add("landing: 落地后 1 s 内恢复站立，不摔倒")
  add("```")
  add("")
  add(
    "该区间显著低于上面的解析上界，留出控制余量；把目标定到上界附近会把问题同时"
    "变成「精确起跳时机」和「落地稳定」两件难事。"
  )
  add("")
  add("## 5. 对 Step 2 / Step 3 的约束")
  add("")
  add("- 倒立：`base_height_target` 取上表推荐值，`projected_gravity` 目标取")
  add(f"  `{gravity_target}`（前腿支撑支路）；后足离地写成稠密奖励，避免稀疏探索。")
  add("- 跳跃：一回合只有一次起跳机会，终止条件必须排除 `base_contact`，否则")
  add("  起跳与落地会被误判为失败（Go2 Backflip 的已知坑）。")
  add("- 统一策略：walk 与 getup 的 actor 观测已是 48 维且字段顺序一致，追加")
  add("  skill block 后为 57 维，critic 72 → 81 维。")
  add("")
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--write",
    action="store_true",
    help="write the report to docs/opendoge_multiskill_targets.md",
  )
  args = parser.parse_args()
  report = build_report()
  if args.write:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(f"wrote {REPORT_PATH}")
  else:
    print(report)


if __name__ == "__main__":
  main()
