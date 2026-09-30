// AMP uses the wbc_fsm State_MJAmp checkpoint, motor order and gains.
export const g1JointNames = [
  ...["left", "right"].flatMap((side) => ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"].map((joint) => `${side}_${joint}_joint`)),
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  ...["left", "right"].flatMap((side) => ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"].map((joint) => `${side}_${joint}_joint`)),
];
export const g1MotorOrder = Array.from({ length: 29 }, (_, i) => i);
const pose = [-.312, 0, 0, .669, -.363, 0, -.312, 0, 0, .669, -.363, 0, 0, 0, 0, .2, .2, 0, .6, 0, 0, 0, .2, -.2, 0, .6, 0, 0, 0];
const armature = [.025101925, .025101925, .010177520, .025101925, .00721945, .00721945,
  .025101925, .025101925, .010177520, .025101925, .00721945, .00721945, .010177520, .00721945, .00721945,
  .003609725, .003609725, .003609725, .003609725, .003609725, .0021812, .0021812,
  .003609725, .003609725, .003609725, .003609725, .003609725, .0021812, .0021812];
const kp = armature.map((a) => a * (20 * Math.PI) ** 2);
const kd = armature.map((a) => 4 * a * 20 * Math.PI);
const maxTorque = [139, 139, 88, 139, 50, 50, 139, 139, 88, 139, 50, 50, 88, 50, 50, 25, 25, 25, 25, 25, 10, 10, 25, 25, 25, 25, 25, 10, 10];
export const g1Policies = {
  g1Amp: {
    robot: "g1", mode: "g1", name: "G1 · AMP 平地行走", file: "/policies/g1-amp.onnx",
    inputSize: 384, defaultPose: pose, order: g1MotorOrder, kp, kd, historyLength: 4,
    actionScale: kp.map((gain, i) => .25 * maxTorque[i] / gain),
    commandRanges: { vx: [-.8, 2.5], vy: [-1, 1], yaw: [-3.14, 3.14] },
    note: "AMP · 4 帧历史；W/S 前后，A/D 横移，Q/E 转向。",
  },
};

export function createG1State() {
  return { history: [] };
}

export function g1Observation(policy, state, { q, dq, gyro, gravity, command, action }) {
  const frame = [...gyro, ...gravity, ...command,
    ...policy.order.map((i) => q[i] - policy.defaultPose[i]), ...policy.order.map((i) => dq[i]), ...action];
  if (!state.history.length) state.history = Array.from({ length: policy.historyLength }, () => [...frame]);
  state.history.push(frame);
  state.history.shift();
  return Float32Array.from(state.history.flat(), (v) => Math.max(-100, Math.min(100, v)));
}

export function g1Targets(policy, action) {
  const targets = [...policy.defaultPose];
  policy.order.forEach((motor, i) => { targets[motor] += action[i] * policy.actionScale[motor]; });
  return targets;
}
