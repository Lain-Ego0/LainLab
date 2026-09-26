"""Local CSV-to-NPZ conversion for G1 motion-tracking tasks."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig
from tqdm import tqdm

from src.assets.robots import (
  G1_23DOF_ACTION_SCALE,
  G1_ACTION_SCALE,
  get_g1_23dof_robot_cfg,
  get_g1_robot_cfg,
)
from src.tasks.robots.g1.tracking import make_g1_tracking_env_cfg

RobotName = Literal["g1", "g1_23dof"]

G1_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

G1_23DOF_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
)


def resolve_output_path(input_file: str | Path, output_name: str | Path) -> Path:
  """Resolve a local output path while preserving the existing CLI semantics."""
  input_path = Path(input_file).expanduser()
  output_path = Path(output_name).expanduser()
  if output_path.parent == Path("."):
    output_path = input_path.parent / output_path
  if output_path.suffix != ".npz":
    output_path = Path(f"{output_path}.npz")
  return output_path


def _robot_setup(robot: RobotName):
  if robot == "g1":
    env_cfg = make_g1_tracking_env_cfg(get_g1_robot_cfg, G1_ACTION_SCALE)
    return env_cfg.scene, G1_JOINT_NAMES
  env_cfg = make_g1_tracking_env_cfg(
    get_g1_23dof_robot_cfg,
    G1_23DOF_ACTION_SCALE,
    reduced_dof=True,
  )
  return env_cfg.scene, G1_23DOF_JOINT_NAMES


def convert_csv_to_npz(
  robot: RobotName,
  input_file: str,
  output_name: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  line_range: tuple[int, int] | None = None,
  progress: Callable[[str, int, int], None] | None = None,
) -> Path:
  """Convert a motion CSV to a local tracking NPZ and return its path."""
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING] CUDA is unavailable; falling back to CPU.")
    device = "cpu"

  scene_cfg, joint_names = _robot_setup(robot)
  scene = Scene(scene_cfg, device=device)
  model = scene.compile()
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  motion = MotionLoader(
    motion_file=input_file,
    # MotionLoader's annotations say int, but its arithmetic and mjlab's own
    # public converter accept fractional frame rates.
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    line_range=line_range,
  )
  if motion.motion_dof_poss.shape[1] != len(joint_names):
    raise ValueError(
      f"{robot} expects {len(joint_names)} joint columns, but the CSV contains "
      f"{motion.motion_dof_poss.shape[1]}"
    )

  robot_entity: Entity = scene["robot"]
  joint_indexes = robot_entity.find_joints(joint_names, preserve_order=True)[0]
  data: dict[str, Any] = {
    "fps": np.asarray([output_fps]),
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }

  renderer = None
  frames = []
  if render:
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot",
      distance=2.0,
      elevation=-5.0,
      azimuth=20,
    )
    renderer = OffscreenRenderer(model=sim.mj_model, cfg=viewer_cfg, scene=scene)
    renderer.initialize()

  scene.reset()
  if progress:
    progress("converting", 0, motion.output_frames)
  for frame_index in tqdm(
    range(motion.output_frames),
    desc="Processing frames",
    unit="frame",
    ncols=100,
  ):
    (
      (
        base_pos,
        base_quat,
        base_lin_vel,
        base_ang_vel,
        joint_pos,
        joint_vel,
      ),
      _,
    ) = motion.get_next_state()

    root_state = robot_entity.data.default_root_state.clone()
    root_state[:, 0:3] = base_pos
    root_state[:, :2] += scene.env_origins[:, :2]
    root_state[:, 3:7] = base_quat
    root_state[:, 7:10] = base_lin_vel
    root_state[:, 10:13] = base_ang_vel
    robot_entity.write_root_state_to_sim(root_state)

    sim_joint_pos = robot_entity.data.default_joint_pos.clone()
    sim_joint_vel = robot_entity.data.default_joint_vel.clone()
    sim_joint_pos[:, joint_indexes] = joint_pos
    sim_joint_vel[:, joint_indexes] = joint_vel
    robot_entity.write_joint_state_to_sim(sim_joint_pos, sim_joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    if renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    data["joint_pos"].append(robot_entity.data.joint_pos[0].cpu().numpy().copy())
    data["joint_vel"].append(robot_entity.data.joint_vel[0].cpu().numpy().copy())
    data["body_pos_w"].append(robot_entity.data.body_link_pos_w[0].cpu().numpy().copy())
    data["body_quat_w"].append(
      robot_entity.data.body_link_quat_w[0].cpu().numpy().copy()
    )
    data["body_lin_vel_w"].append(
      robot_entity.data.body_link_lin_vel_w[0].cpu().numpy().copy()
    )
    data["body_ang_vel_w"].append(
      robot_entity.data.body_link_ang_vel_w[0].cpu().numpy().copy()
    )

    if progress and (
      (frame_index + 1) % 10 == 0 or frame_index + 1 == motion.output_frames
    ):
      progress("converting", frame_index + 1, motion.output_frames)

  if progress:
    progress("saving", motion.output_frames, motion.output_frames)
  for key in (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ):
    data[key] = np.stack(data[key], axis=0)

  output_path = resolve_output_path(input_file, output_name)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output_path, **data)
  print(f"[INFO] Motion saved locally: {output_path}")

  if render:
    import mediapy as media

    video_path = output_path.with_suffix(".mp4")
    media.write_video(video_path, frames, fps=output_fps)
    print(f"[INFO] Motion video saved locally: {video_path}")

  return output_path


def main(
  input_file: str,
  output_name: str,
  robot: RobotName = "g1",
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  line_range: tuple[int, int] | None = None,
) -> None:
  """Replay a G1 motion CSV and save a training-ready NPZ locally."""
  convert_csv_to_npz(
    robot=robot,
    input_file=input_file,
    output_name=output_name,
    input_fps=input_fps,
    output_fps=output_fps,
    device=device,
    render=render,
    line_range=line_range,
  )
