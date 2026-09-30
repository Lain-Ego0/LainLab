// Decision logic is independent of ONNX and MuJoCo so recorded/synthetic sensor
// sequences can exercise the same state machine used during playback.
export function createRecoveryDecision() {
  return { active: false, phase: "normal", fallenTime: 0, uprightTime: 0, unsupportedTime: 0, cooldown: 0 };
}

export function measureRecoverySupport({ baseZ, groundZ, footGeomIds, baseBodyId, terrainGeomIds, geomBodyIds, contacts }) {
  const feet = new Map();
  let bodyContact = false;
  for (const contact of contacts) {
    if (contact.efc_address < 0 || contact.dist > .01) continue;
    const robotGeom = terrainGeomIds.has(contact.geom1) ? contact.geom2
      : terrainGeomIds.has(contact.geom2) ? contact.geom1 : -1;
    if (robotGeom < 0) continue;
    if (geomBodyIds[robotGeom] === baseBodyId) bodyContact = true;
    // A foot touching a vertical wall does not support a standing robot.
    if (!footGeomIds.has(robotGeom) || Math.abs(contact.frame[2]) < .5) continue;
    feet.set(robotGeom, Math.max(feet.get(robotGeom) ?? -Infinity, contact.pos[2]));
  }
  const supportZ = feet.size ? [...feet.values()].reduce((sum, z) => sum + z, 0) / feet.size : groundZ;
  return { supportFeet: feet.size, bodyContact, height: baseZ - supportZ };
}

export function recoveryTiltSpeed(gyro, gravity) {
  // Rotation around gravity is yaw. Only rotation that changes tilt matters.
  return Math.hypot(gyro[1] * gravity[2] - gyro[2] * gravity[1],
    gyro[2] * gravity[0] - gyro[0] * gravity[2], gyro[0] * gravity[1] - gyro[1] * gravity[0]);
}

export function stepRecoveryDecision(state, sample, dt) {
  const { up, height, tiltSpeed, verticalSpeed, supportFeet, bodyContact, intentionalStand = false } = sample;
  if (!Number.isFinite(dt) || dt <= 0) throw new Error("Invalid recovery decision timestep");
  if (![up, height, tiltSpeed, verticalSpeed, supportFeet].every(Number.isFinite)) {
    state.fallenTime = state.uprightTime = state.unsupportedTime = 0;
    if (state.active && state.phase === "stabilizing") {
      state.phase = "recovering";
      return "retry";
    }
    return null;
  }

  if (!state.active) {
    state.cooldown = Math.max(0, state.cooldown - dt);
    const collapsed = height < .19 && (bodyContact || supportFeet < 2);
    const fallen = intentionalStand ? up < -.35 || (height < .12 && bodyContact)
      : (up < .45 && height < .34) || collapsed;
    state.fallenTime = fallen && state.cooldown === 0 ? state.fallenTime + dt : 0;
    if (state.fallenTime + 1e-9 < (intentionalStand ? .25 : .16)) return null;
    state.active = true;
    state.phase = "recovering";
    state.fallenTime = state.uprightTime = state.unsupportedTime = 0;
    return "recover";
  }

  // Enter standing confirmation at 22 cm; once there, use relaxed bounds to
  // avoid restarting on every small oscillation. Torso-supported or airborne
  // poses cannot finish recovery, regardless of their height and orientation.
  const standing = up >= .82 && height >= .22 && height <= .65 && tiltSpeed <= 2
    && Math.abs(verticalSpeed) <= .65 && supportFeet >= 2 && !bodyContact;
  const holding = up >= .75 && height >= .20 && height <= .70 && tiltSpeed <= 3
    && Math.abs(verticalSpeed) <= 1 && supportFeet >= 2 && !bodyContact;
  if (state.phase === "recovering") {
    if (!standing) return null;
    state.phase = "stabilizing";
    state.uprightTime = dt;
    state.unsupportedTime = 0;
    return "stabilize";
  }

  if (holding) {
    state.uprightTime += dt;
    state.unsupportedTime = 0;
  } else {
    state.unsupportedTime += dt;
    state.uprightTime = Math.max(0, state.uprightTime - dt * .5);
    // With no foot contact, clearance falls back to the terrain under the
    // torso; that estimate can jump over an obstacle during a brief dropout.
    if (bodyContact || up < .65 || (height < .17 && supportFeet > 0) || state.unsupportedTime > .12 + 1e-9) {
      state.phase = "recovering";
      state.uprightTime = state.unsupportedTime = 0;
      return "retry";
    }
  }
  if (holding && state.uprightTime + 1e-9 >= .4) {
    Object.assign(state, createRecoveryDecision(), { cooldown: 1 });
    return "resume";
  }
  return null;
}
