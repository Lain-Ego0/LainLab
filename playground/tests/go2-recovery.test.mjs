import test from "node:test";
import assert from "node:assert/strict";
import { createRecoveryDecision, measureRecoverySupport, recoveryTiltSpeed, stepRecoveryDecision } from "../src/go2-recovery.js";

const standing = { up: .99, height: .28, tiltSpeed: .2, verticalSpeed: 0, supportFeet: 4, bodyContact: false };
const fallen = { ...standing, up: 0, height: .10, supportFeet: 0, bodyContact: true };
function advance(state, sample, seconds) {
  const events = [];
  for (let step = 0; step < Math.round(seconds / .02); step++) {
    const event = stepRecoveryDecision(state, sample, .02);
    if (event) events.push(event);
  }
  return events;
}
function recovering() {
  const state = createRecoveryDecision();
  assert.deepEqual(advance(state, fallen, .16), ["recover"]);
  return state;
}

test("a supported low stance below the former 31 cm cutoff completes recovery", () => {
  const state = recovering();
  assert.deepEqual(advance(state, { ...standing, height: .25 }, .38), ["stabilize"]);
  assert.equal(state.active, true);
  assert.deepEqual(advance(state, { ...standing, height: .25 }, .02), ["resume"]);
  assert.equal(state.active, false);
  assert.equal(state.phase, "normal");
});

test("turning and a moderate slope do not block a supported standing pose", () => {
  const gravity = [.5, 0, -Math.sqrt(.75)];
  const yawGyro = gravity.map((value) => value * 4);
  const tiltSpeed = recoveryTiltSpeed(yawGyro, gravity);
  assert.ok(tiltSpeed < 1e-10);
  assert.equal(recoveryTiltSpeed([2, 0, 0], [0, 0, -1]), 2);
  const state = recovering();
  assert.deepEqual(advance(state, { ...standing, up: -gravity[2], tiltSpeed }, .5), ["stabilize", "resume"]);
});

test("brief contact loss and threshold jitter do not keep restarting confirmation", () => {
  const state = recovering();
  for (let step = 0; step < 45 && state.active; step++) {
    const sample = step % 6 === 5 ? { ...standing, supportFeet: 1, tiltSpeed: 3.2 }
      : step % 3 === 1 ? { ...standing, up: .80, height: .21 } : standing;
    stepRecoveryDecision(state, sample, .02);
  }
  assert.equal(state.active, false);
});

test("a brief loss of all foot contacts does not reset progress on uneven terrain", () => {
  const state = recovering();
  advance(state, standing, .3);
  // The fallback sample sees an obstacle under the torso until foot support
  // returns. It is not evidence that the body suddenly fell by 18 cm.
  assert.deepEqual(advance(state, { ...standing, supportFeet: 0, height: .1 }, .08), []);
  assert.equal(state.phase, "stabilizing");
  assert.deepEqual(advance(state, standing, .16), ["resume"]);
});

test("airborne, belly-supported and rolling poses never count as recovered", () => {
  for (const sample of [
    { ...standing, supportFeet: 0 },
    { ...standing, bodyContact: true },
    { ...standing, up: .5 },
    { ...standing, tiltSpeed: 4 },
    { ...standing, verticalSpeed: 2 },
    { ...standing, height: .14 },
  ]) {
    const state = recovering();
    assert.deepEqual(advance(state, sample, 2), []);
    assert.equal(state.phase, "recovering");
  }
});

test("losing support or falling again cancels confirmation instead of timing out to normal", () => {
  for (const sample of [{ ...standing, supportFeet: 0 }, fallen]) {
    const state = recovering();
    advance(state, standing, .3);
    assert.deepEqual(advance(state, sample, .2), ["retry"]);
    assert.equal(state.uprightTime, 0);
    assert.equal(state.active, true);
    assert.deepEqual(advance(state, standing, .2), ["stabilize"]);
    assert.equal(state.active, true);
  }
});

test("brief landing compression and intended two-leg stands do not trigger recovery", () => {
  const state = createRecoveryDecision();
  advance(state, fallen, .1);
  advance(state, standing, .1);
  advance(state, { ...standing, height: .18 }, .5);
  advance(state, { ...standing, intentionalStand: true, up: .1, supportFeet: 2 }, 2);
  advance(state, { ...standing, height: .6, verticalSpeed: 1, supportFeet: 0 }, .5);
  assert.equal(state.active, false);
  assert.deepEqual(advance(state, fallen, .16), ["recover"]);
});

test("a resumed policy gets a cooldown, and a later sustained fall still recovers", () => {
  const state = recovering();
  advance(state, standing, .4);
  advance(state, fallen, .9);
  assert.equal(state.active, false);
  assert.deepEqual(advance(state, fallen, .3), ["recover"]);
});

test("clearance follows actual foot support on steps, not an obstacle underneath the torso", () => {
  const contact = (geom2, z, overrides = {}) => ({ geom1: 10, geom2, efc_address: 0,
    dist: 0, pos: [0, 0, z], frame: [0, 0, 1], ...overrides });
  const setup = { baseZ: .46, groundZ: .30, footGeomIds: new Set([0, 1, 2, 3]), baseBodyId: 1,
    terrainGeomIds: new Set([10]), geomBodyIds: [2, 3, 4, 5, 1] };
  const sensors = measureRecoverySupport({ ...setup, contacts: [
    contact(0, .16), contact(0, .16), contact(1, .20, { geom1: 1, geom2: 10 }),
    contact(2, .3, { frame: [1, 0, 0] }), // Touching a wall.
    contact(3, .3, { efc_address: -1 }), // Inactive proximity contact.
    contact(4, 0, { geom1: 2 }), // Robot self-contact.
  ] });
  assert.equal(sensors.supportFeet, 2);
  assert.equal(sensors.bodyContact, false);
  assert.ok(Math.abs(sensors.height - .28) < 1e-10);
  assert.deepEqual(advance(recovering(), { ...standing, ...sensors }, .5), ["stabilize", "resume"]);
  const belly = measureRecoverySupport({ ...setup, contacts: [contact(4, .3)] });
  assert.equal(belly.bodyContact, true);
  assert.equal(belly.supportFeet, 0);
});
