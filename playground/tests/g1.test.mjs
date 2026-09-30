import test from "node:test";
import assert from "node:assert/strict";
import { g1Policies, createG1State, g1Observation, g1Targets } from "../src/g1.js";

const sample = (policy) => ({ q: [...policy.defaultPose], dq: new Array(29).fill(0), gyro: [1, 2, 3],
  gravity: [0, 0, -1], command: [0, 0, 0], action: new Float32Array(29), elapsed: 0 });

test("G1 history starts with the standing observation, then advances oldest first", () => {
  const policy = g1Policies.g1Amp, state = createG1State(), input = sample(policy);
  const initial = g1Observation(policy, state, input);
  assert.equal(initial.length, 384);
  assert.deepEqual([...initial.slice(0, 96)], [...initial.slice(288)]);
  assert.deepEqual([...initial.slice(0, 9)], [1, 2, 3, 0, 0, -1, 0, 0, 0]);
  input.q[6] += .5;
  const next = g1Observation(policy, state, input);
  assert.equal(next[9 + 6], 0);
  assert.equal(next[288 + 9 + 6], .5);
});

test("AMP torque-normalized action produces one quarter of the configured motor torque", () => {
  const policy = g1Policies.g1Amp;
  const action = new Float32Array(29); action[0] = 1;
  const targets = g1Targets(policy, action);
  assert.ok(Math.abs((targets[0] - policy.defaultPose[0]) * policy.kp[0] - 139 / 4) < 1e-10);
});

test("DWAQ uses interleaved joints and 5 frames with training-order gait phases", () => {
  const policy = g1Policies.g1DwaqPhase, state = createG1State(), input = sample(policy);
  input.q[6] += .3; // Right hip pitch is the second policy joint.
  input.dq[12] = 2; // Waist yaw is the third policy joint.
  input.action[2] = .75; // Previous actions already use policy order.
  const obs = g1Observation(policy, state, input);
  assert.equal(obs.length, 500);
  assert.deepEqual([...obs.slice(0, 100)], [...obs.slice(400)]);
  assert.ok(Math.abs(obs[410] - .3) < 1e-6);
  assert.equal(obs[440], 2);
  assert.equal(obs[469], .75);
  const near = (actual, expected) => actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < 1e-6));
  near(obs.slice(496), [0, 0, 1, -1]);
  input.elapsed = .2;
  const next = g1Observation(policy, state, input);
  near(next.slice(496), [1, -1, 0, 0]);
  near(next.slice(396, 400), [0, 0, 1, -1]);
});

test("DWAQ targets and PD gains map to motor order without AMP torque scaling", () => {
  const policy = g1Policies.g1DwaqPhase;
  assert.deepEqual(policy.defaultPose.slice(0, 6), [-.2, 0, 0, .42, -.23, 0]);
  assert.deepEqual(policy.kp.slice(0, 6), [200, 150, 150, 200, 20, 20]);
  assert.deepEqual(policy.kd.slice(0, 6), [5, 5, 5, 5, 2, 2]);
  const action = new Float32Array(29); action[1] = 2; action[12] = -1;
  const target = g1Targets(policy, action);
  assert.ok(Math.abs(target[6] - .3) < 1e-10);
  assert.ok(Math.abs(target[22] - .10) < 1e-10);
  assert.equal(target[0], policy.defaultPose[0]);
  assert.equal(new Set(policy.order).size, 29);
});
