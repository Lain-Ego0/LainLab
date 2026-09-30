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
// G1DWAQ_Lab/LeggedLabDeploy/configs/g1_dwaq_phase.yaml: policy order is
// Isaac Lab's interleaved joint order; poses and gains below are stored in motor order.
const dwaqOrder = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28];
const dwaqMotorValues = (values) => {
  const motors = new Array(29);
  dwaqOrder.forEach((motor, index) => { motors[motor] = values[index]; });
  return motors;
};
export const g1Policies = {
  g1Amp: {
    robot: "g1", mode: "g1", name: "G1 · AMP 平地行走", file: "/policies/g1-amp.onnx",
    inputSize: 384, defaultPose: pose, order: g1MotorOrder, kp, kd, historyLength: 4,
    actionScale: kp.map((gain, i) => .25 * maxTorque[i] / gain),
    commandRanges: { vx: [-.8, 2.5], vy: [-1, 1], yaw: [-3.14, 3.14] },
    note: "AMP · 4 帧历史；W/S 前后，A/D 横移，Q/E 转向。",
  },
  g1DwaqPhase: {
    robot: "g1", mode: "g1", name: "G1 · DWAQ 越障", file: "/policies/g1-dwaq-phase.onnx",
    inputSize: 500, order: dwaqOrder, historyLength: 5,
    defaultPose: dwaqMotorValues([-.2, -.2, 0, 0, 0, 0, 0, 0, 0, .42, .42, .35, .35, -.23, -.23, .18, -.18, 0, 0, 0, 0, .87, .87, 0, 0, 0, 0, 0, 0]),
    kp: dwaqMotorValues([200, 200, 200, 150, 150, 200, 150, 150, 200, 200, 200, 100, 100, 20, 20, 100, 100, 20, 20, 50, 50, 50, 50, 40, 40, 40, 40, 40, 40]),
    kd: dwaqMotorValues([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
    actionScale: new Array(29).fill(.25), gait: { period: .8, offset: .5 },
    commandRanges: { vx: [-.4, .7], vy: [-.4, .4], yaw: [-1.57, 1.57] },
    note: "DreamWaQ · 带步态相位的 29 自由度盲走策略；建议从低速、低台阶开始。",
  },
};

export function createG1State() {
  return { history: [] };
}

export function g1Observation(policy, state, { q, dq, gyro, gravity, command, action, elapsed = 0 }) {
  const frame = [...gyro, ...gravity, ...command,
    ...policy.order.map((i) => q[i] - policy.defaultPose[i]), ...policy.order.map((i) => dq[i]), ...action];
  if (policy.gait) {
    const left = (elapsed / policy.gait.period % 1) * 2 * Math.PI;
    const right = left + policy.gait.offset * 2 * Math.PI;
    // Match the training environment: sin(left/right), then cos(left/right).
    frame.push(Math.sin(left), Math.sin(right), Math.cos(left), Math.cos(right));
  }
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
