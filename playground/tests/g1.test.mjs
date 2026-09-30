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
