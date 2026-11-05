// Low-level robot interface over the OpenArm MuJoCo WASM page.
//
// Everything above this file (tools, planner, UI) talks to the robot only
// through this module, so there is exactly one place that knows how the
// upstream page stores its state. Three habits here are deliberate:
//
//   * The harness drives the physics with a fixed step, it does not wait on
//     requestAnimationFrame. The upstream page only steps the model while it
//     is painting, and a browser pane that is not on screen paints nothing -
//     which is exactly how a run used to hang half way, and why the same
//     coordinates succeeded twice and then failed once.
//   * Motion waits for the measurement, not for the clock. Every primitive
//     polls the real end-effector error until it settles or the deadline
//     passes, then reports the numbers it measured.
//   * Nothing writes an object's pose. The bottle is a free body for the whole
//     run; if the gripper does not hold it, the run fails and says so.

import { poseWorldToLocal } from "/sim/web/ik.js";

// MuJoCo stores gainprm and biasprm as mjtNum[MJNGAIN * nu] with MJNGAIN = 10,
// so one actuator's parameters are a 10-wide slice, not 3. This file used 3 and
// wrote into neighbouring actuators' entries; the read-back check agreed only
// because it read the same wrong offsets.
const MJNGAIN = 10;
const SIDES = ["left", "right"];

/**
 * Robot base frame -> this scene's world frame, measured by comparing pure
 * forward kinematics against the recorded end-effector positions.
 *
 * It is a *frame* conversion, not a physical distance: it also absorbs the
 * difference between the recording's link7 origin and this model's control
 * point. Use it to move data between the two, never as a measurement of where
 * anything physically is.
 */
export const BASE_TO_WORLD = [0.185, 0.0, 0.792];
const FIXED_DT = 1 / 60; // control period, not the model timestep (0.001 s)

export function norm(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

/** Give the page a chance to paint, whether or not it is on screen. */
function yieldToPage() {
  return new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    requestAnimationFrame(finish); // fires when the pane is visible
    setTimeout(finish, 16); // fires when it is not
  });
}

function slerpQuat(a, target, t) {
  let b = [...target];
  let dot = a.reduce((sum, value, i) => sum + value * b[i], 0);
  if (dot < 0) {
    b = b.map((value) => -value);
    dot = -dot;
  }
  if (dot > 0.9995) {
    const mixed = a.map((value, i) => value + t * (b[i] - value));
    const length = Math.hypot(...mixed);
    return mixed.map((value) => value / length);
  }
  const angle = Math.acos(Math.max(-1, Math.min(1, dot)));
  const denominator = Math.sin(angle);
  const first = Math.sin((1 - t) * angle) / denominator;
  const second = Math.sin(t * angle) / denominator;
  return a.map((value, i) => first * value + second * b[i]);
}

export class Robot {
  constructor(app, mujoco) {
    this.app = app;
    this.mujoco = mujoco;
    this.sides = SIDES;
    this.startPose = null;
    this.#resolve("task_bottle", "bottle_free",
      ["bottle_body", "bottle_neck", "bottle_cap"], this.bottle = {});
    this.#resolve("target_box", null, ["box_floor"], this.box = {});
  }

  static async attach(app, mujoco, calib = null) {
    const robot = new Robot(app, mujoco);
    if (robot.bottle.qpos < 0 || robot.box.bodyId < 0) {
      throw new Error("任务场景缺少瓶子或盒子；请确认已加载 bottle_to_box.xml。");
    }
    robot.sceneChanges = robot.applySceneOptions(calib);
    robot.startPose = [...app.mjData.qpos.subarray(robot.bottle.qpos, robot.bottle.qpos + 7)];
    return robot;
  }

  /**
   * Turn off the parts of the upstream cell that the real robot does not have.
   *
   * The lab robot is a mobile bimanual with a table; the cell scene wraps the
   * same arm in an enclosure of walls, a roof and rails. Those extra surfaces
   * are collision geometry, and replaying a real episode with them enabled put
   * the right arm 0.84 rad behind its commanded joints - it was pressed against
   * a wall that does not exist. With them off the same replay tracks to
   * 0.009 rad, better than the real robot's own 0.005-0.032 rad. Which ones to
   * drop is a calibration decision and lives in calibrate.json.
   */
  applySceneOptions(calib = null) {
    // Kept so anything holding the robot - replay, tools, telemetry - reads the
    // same numbers the scene was built from, instead of carrying its own copy.
    this.calibration = calib;
    const spec = calib?.scene;
    if (!spec) return null;
    const changed = [];
    // The real controller gains, identified from the deployment logs. The
    // upstream model carries gains from the motor datasheets; the robot runs a
    // different loop, and the gripper is five times softer than the model says
    // (kp 2 rather than 10). That difference is what decides whether a grasp
    // presses a bottle over or settles on it.
    const gains = spec.actuator_gains ?? calib?.actuators;
    if (gains?.kp) {
      for (const side of ["left", "right"]) {
        for (const key of ["j1", "j2", "j3", "j4", "j5", "j6", "j7"]) {
          const actuator = this.app.mjModel.actuator(`${side}_joint${key.slice(1)}_ctrl`);
          const id = actuator.id;
          const kp = gains.kp[key] ?? null;
          const kd = gains.kd?.[key] ?? null;
          if (kp !== null) {
            this.app.mjModel.actuator_gainprm[id * MJNGAIN] = kp;
            this.app.mjModel.actuator_biasprm[id * MJNGAIN + 1] = -kp;
          }
          if (kd !== null) this.app.mjModel.actuator_biasprm[id * MJNGAIN + 2] = -kd;
          actuator.delete?.();
        }
        // The model's own finger actuator carries forcerange +-7 N.m; the real
        // motor stalls at the 0.2466 N.m measured in the deployment logs. The
        // scene adds a force-limited actuator on the same joint, so the
        // original one is neutralised and the limited one does the work.
        const original = this.app.mjModel.actuator(`${side}_finger1_ctrl`);
        this.app.mjModel.actuator_gainprm[original.id * MJNGAIN] = 0;
        this.app.mjModel.actuator_biasprm[original.id * MJNGAIN + 1] = 0;
        this.app.mjModel.actuator_biasprm[original.id * MJNGAIN + 2] = 0;
        original.delete?.();
        const limited = this.app.mjModel.actuator(`${side}_finger_limit`);
        if (limited) {
          this.app.mjModel.actuator_gainprm[limited.id * MJNGAIN] = gains.kp.gripper;
          this.app.mjModel.actuator_biasprm[limited.id * MJNGAIN + 1] = -gains.kp.gripper;
          this.app.mjModel.actuator_biasprm[limited.id * MJNGAIN + 2] = -(gains.kd?.gripper ?? 0);
          // route the gripper commands to it instead
          this.app.controller.arms[side].gripperActId = limited.id;
          this.app.controller.arms[side].gripperRange = [
            this.app.mjModel.actuator_ctrlrange[limited.id * 2],
            this.app.mjModel.actuator_ctrlrange[limited.id * 2 + 1],
          ];
          limited.delete?.();
        }
      }
      changed.push(`actuators kp=${gains.kp.j1}/${gains.kp.gripper}`);
    }
    // The lifter is how far the arms sit above the table. The upstream cell
    // puts them 0.335 m up; the lab robot is 0.15 m higher than that, and
    // getting it wrong pressed the right hand into the tabletop for 1300
    // frames of every replay - which reads as "the servos are too soft" and
    // is not.
    if (spec.lifter_height?.value) {
      const joint = this.mujoco.mj_name2id(
        this.app.mjModel, this.mujoco.mjtObj.mjOBJ_JOINT.value, "openarm_lifter_joint",
      );
      if (joint >= 0) {
        const qpos = this.app.mjModel.jnt_qposadr[joint];
        this.lifterJointId = joint;
        this.app.mjData.qpos[qpos] = spec.lifter_height.value;
        const actuator = this.app.mjModel.actuator("lifter_ctrl");
        this.lifterActuatorId = actuator.id;
        actuator.delete?.();
        this.app.mjData.ctrl[this.lifterActuatorId] = spec.lifter_height.value;
        this.mujoco.mj_forward(this.app.mjModel, this.app.mjData);
        // `app.lifterHeight` is what applyTeleop feeds the lifter actuator, so
        // it has to carry the calibrated value or every applyTargets would
        // drop the arms back down to the cell's default.
        this.app.lifterHeight = spec.lifter_height.value;
        this.lifterHeight = spec.lifter_height.value;
        changed.push(`lifter=${spec.lifter_height.value}`);
      }
    }
    for (const name of spec.disable_cell_collisions ?? []) {
      const id = this.geomId(name);
      if (id < 0) continue;
      this.app.mjModel.geom_contype[id] = 0;
      this.app.mjModel.geom_conaffinity[id] = 0;
      changed.push(name);
    }
    return changed;
  }

  #resolve(bodyName, jointName, geomNames, into) {
    const { mujoco, app } = this;
    into.bodyId = mujoco.mj_name2id(app.mjModel, mujoco.mjtObj.mjOBJ_BODY.value, bodyName);
    into.geomIds = geomNames.map((name) => this.geomId(name));
    if (jointName) {
      const joint = mujoco.mj_name2id(app.mjModel, mujoco.mjtObj.mjOBJ_JOINT.value, jointName);
      into.jointId = joint;
      into.qpos = joint >= 0 ? app.mjModel.jnt_qposadr[joint] : -1;
      into.dof = joint >= 0 ? app.mjModel.jnt_dofadr[joint] : -1;
    }
  }

  geomId(name) {
    return this.mujoco.mj_name2id(this.app.mjModel, this.mujoco.mjtObj.mjOBJ_GEOM.value, name);
  }

  actuatorId(name) {
    const { app } = this;
    const actuator = app.mjModel.actuator(name);
    const id = actuator.id;
    actuator.delete?.();
    return id;
  }

  // --- physics -------------------------------------------------------------

  /**
   * Advance the model by `seconds`, one 1 ms model step at a time.
   *
   * `onControlTick` is called at the recording's rate - 30 Hz, the same as the
   * lab data - rather than once per model step, so a simulated episode comes
   * out at the same sampling rate as a real one and the two are directly
   * comparable as training data.
   */
  #step(seconds) {
    const { mjModel, mjData } = this.app;
    const until = mjData.time + seconds;
    let guard = 0;
    while (mjData.time < until) {
      this.app.controller.applyGravityComp(mjData);
      this.mujoco.mj_step(mjModel, mjData);
      if (++guard > 200000) break; // never spin forever on a stalled model
    }
    if (!this.onControlTick) return;
    const period = 1 / (this.controlHz ?? 30);
    while (mjData.time - (this.lastTick ?? -Infinity) >= period) {
      this.lastTick = (this.lastTick ?? mjData.time) + period;
      this.onControlTick(mjData.time);
    }
  }

  /** Advance by `seconds` of simulated time, letting the page paint meanwhile. */
  async simulate(seconds) {
    const dt = 1 / 60;
    const steps = Math.max(1, Math.round(seconds / dt));
    for (let i = 0; i < steps; i++) {
      this.#step(dt);
      if (i % 8 === 7) await yieldToPage();
    }
  }

  // --- state ---------------------------------------------------------------

  tcp(side) {
    return this.app.controller.arms[side].pose(this.app.mjData);
  }

  tcpError(side, pos, quat) {
    return this.app.controller.arms[side].poseError(this.app.mjData, pos, quat);
  }

  time() {
    return this.app.mjData.time;
  }

  /**
   * Put the scene back to the state captured right after the model loaded.
   *
   * It waits for the arms to actually arrive. Setting the park target and
   * returning immediately left them mid-swing, and the next tool then took the
   * hand's half-way position as its starting point - which is how a "reach"
   * came to be planned from a place the arm had never been asked to be.
   */
  async reset({ settleSeconds = 4, stow = true, arms = true } = {}) {
    const { app, mujoco } = this;
    if (arms) {
      app.reset();
      if (stow) this.stow();
    }
    // The arms travel home along a path that crosses the bottle's column, and
    // at servo speed that is not a nudge: measured on a plain reset(), the
    // bottle ended up 426 mm away and on the floor before anything had been
    // asked to run. Pressing "reset" destroyed the scene it was resetting.
    //
    // So the bottle is held out of the way for the move and put back
    // afterwards. The order is the whole fix - restoring it first is what let
    // the sweep hit it. It is also why a replay used to pass arms = false, a
    // workaround at the call site for a problem that belongs here.
    const park = () => {
      app.mjData.qpos[this.bottle.qpos] = 5;
      app.mjData.qpos[this.bottle.qpos + 1] = 5;
      app.mjData.qpos[this.bottle.qpos + 2] = -1;
      app.mjData.qvel.fill(0, this.bottle.dof, this.bottle.dof + 6);
      mujoco.mj_forward(app.mjModel, app.mjData);
    };
    const restore = () => {
      app.mjData.qpos.set(this.startPose, this.bottle.qpos);
      app.mjData.qvel.fill(0, this.bottle.dof, this.bottle.dof + 6);
      mujoco.mj_forward(app.mjModel, app.mjData);
    };
    if (arms) {
      park();
      await this.settleBoth({ seconds: settleSeconds });
    }
    restore();
  }

  /** Step until both hands are at the pose they were last told to reach. */
  async settleBoth({ tolerance = 0.015, seconds = 4 } = {}) {
    const steps = Math.round(seconds / FIXED_DT);
    for (let i = 0; i < steps; i++) {
      const done = this.sides.every((side) => {
        const target = this.app.targets?.[side]?.pos;
        if (!target) return true;
        return Math.hypot(...this.tcpError(side, target, this.tcp(side).quat).errorP) <= tolerance;
      });
      if (done) return { converged: true, seconds: (i * FIXED_DT).toFixed(2) };
      this.#step(FIXED_DT);
      if (i % 8 === 7) await yieldToPage();
    }
    return { converged: false, seconds };
  }

  /**
   * Park both hands clear of the task area so an idle scene reads cleanly.
   *
   * The right hand parks high and back towards the body, not out to the side:
   * the bottle stands at y = -0.30, and a stow that swept outwards passed
   * straight through it - which is what tipped the bottle over on every reset
   * until the sweep was moved off the object's column.
   */
  stow() {
    const { app } = this;
    app.teleop.arms.left.pos = [0.216, 0.345, -0.07];
    app.teleop.arms.left.grip = 0;
    app.teleop.arms.right.pos = [0.216, -0.020, 0.050];
    app.teleop.arms.right.grip = 0;
    app.applyTargets();
  }

  /** World pose of a task body, straight from the simulation. */
  bodyPose(which) {
    const id = this[which].bodyId;
    const p = this.app.mjData.xpos.subarray(id * 3, id * 3 + 3);
    return { pos: [...p] };
  }

  bottlePose() {
    return this.bodyPose("bottle");
  }

  /** Measured actuator force on the gripper joint, in newton-metres. */
  gripForce(side) {
    const id = this.actuatorId(`${side}_finger1_ctrl`);
    const force = this.app.mjData.actuator_force;
    if (!force || id < 0) return null;
    return force[id];
  }

  /**
   * Geom ids of one hand's finger pads, split per finger.
   *
   * The upstream model names its collision meshes rather than placing pad
   * sites, so the pads are found by name once per model load.
   */
  #padGeoms(side) {
    this.padIds ??= {};
    if (this.padIds[side]) return this.padIds[side];
    const inner = [];
    const outer = [];
    for (let g = 0; g < this.app.mjModel.ngeom; g++) {
      const name = this.mujoco.mj_id2name(this.app.mjModel, this.mujoco.mjtObj.mjOBJ_GEOM.value, g);
      if (!name || !name.includes(`_${side}_collision`)) continue;
      if (name.startsWith("finger_inner")) inner.push(g);
      else if (name.startsWith("finger_outer")) outer.push(g);
    }
    this.padIds[side] = { inner, outer };
    return this.padIds[side];
  }

  /** World position of the middle of the two finger pads. */
  padCenter(side) {
    const { inner, outer } = this.#padGeoms(side);
    if (!inner.length || !outer.length) return null;
    const mean = (ids) => {
      const sum = [0, 0, 0];
      for (const g of ids) {
        sum[0] += this.app.mjData.geom_xpos[g * 3];
        sum[1] += this.app.mjData.geom_xpos[g * 3 + 1];
        sum[2] += this.app.mjData.geom_xpos[g * 3 + 2];
      }
      return sum.map((v) => v / ids.length);
    };
    const a = mean(inner);
    const b = mean(outer);
    return a.map((v, i) => (v + b[i]) / 2);
  }

  /** Unit vector from the wrist down the hand, in world coordinates. */
  approachAxis(side) {
    const { quat } = this.tcp(side);
    const [w, x, y, z] = quat;
    // third column of the rotation matrix, negated: the hand's local -z
    const m02 = 2 * (x * z + y * w);
    const m12 = 2 * (y * z - x * w);
    const m22 = 1 - 2 * (x * x + y * y);
    return [-m02, -m12, -m22];
  }

  /**
   * Put the middle of the pads on a point, by measuring where they actually
   * are and moving the wrist by the difference.
   *
   * The alternative - computing the wrist target from a nominal "pads are N mm
   * below the control point" constant - is wrong twice over: the real offset is
   * 114 mm rather than the 92 mm a first guess gives, and it changes with the
   * wrist angle. Measured feedback has neither problem, and a real arm has the
   * same information from its own kinematics.
   *
   * The loop tolerates the arm's own tracking error rather than fighting it:
   * these position actuators settle about 10 mm short of any target, so a
   * tolerance below that would never be met and the correction would keep
   * chasing. What a grasp really needs is checked afterwards, by lifting.
   */
  async alignPads(side, target, quat, { tolerance = 0.014, rounds = 8, axisMask = [1, 1, 1] } = {}) {
    let last = null;
    for (let round = 0; round < rounds; round++) {
      const pads = this.padCenter(side);
      const wrist = this.tcp(side);
      if (!pads) break;
      // Only the axes the caller asked for count towards convergence, and only
      // those are corrected: lining up sideways above the bottle and then going
      // straight down keeps the open fingers from sweeping through it.
      const active = target.map((v, i) => (axisMask[i] ? v : pads[i]));
      last = norm(pads, active);
      if (last <= tolerance) {
        return { error: last, converged: true, rounds: round };
      }
      const corrected = wrist.pos.map((v, i) => v + axisMask[i] * (target[i] - pads[i]));
      try {
        await this.moveTcp(side, corrected, {
          seconds: 0.8, quat, tolerance: 0.004, rotationTolerance: 0.35,
        });
      } catch {
        // Under-tracking is expected here; the next round re-measures and
        // corrects from wherever the arm actually ended up.
      }
    }
    if (last !== null && last <= tolerance) return { error: last, converged: true, rounds };
    throw new Error(
      `${side} 手没能把指垫对到目标点：偏离 ${((last ?? NaN) * 1000).toFixed(1)} mm。`,
    );
  }

  /**
   * Total magnitude of the contacts the gripper currently has on the bottle.
   * Returns null when this build of the WASM binding does not expose contacts,
   * so callers report "not measured" instead of a made-up number.
   */
  bottleContactForce() {
    const data = this.app.mjData;
    if (!data.contact || data.ncon === undefined) return null;
    const bottleGeoms = new Set(this.bottle.geomIds);
    let total = 0;
    let count = 0;
    for (let i = 0; i < data.ncon; i++) {
      const contact = data.contact[i];
      if (!contact || contact.geom1 === undefined) break;
      if (!bottleGeoms.has(contact.geom1) && !bottleGeoms.has(contact.geom2)) continue;
      const force = contact.force;
      if (!force || force.length < 3) continue;
      total += Math.hypot(force[0], force[1], force[2]);
      count += 1;
    }
    return { total, count };
  }

  /** Commanded grip: 0 fully open, 1 fully closed. */
  grip(side) {
    return this.app.teleop.arms[side].grip;
  }

  setGrip(side, value) {
    this.app.teleop.arms[side].grip = Math.max(0, Math.min(1, value));
    this.app.applyTargets();
  }

  /** Mean finger joint angle: 0 rad closed, 0.7854 rad fully open. */
  fingerAngle(side) {
    const { app, mujoco } = this;
    const angles = [];
    for (const n of [1, 2]) {
      const joint = mujoco.mj_name2id(
        app.mjModel, mujoco.mjtObj.mjOBJ_JOINT.value, `openarm_${side}_finger_joint${n}`,
      );
      if (joint >= 0) angles.push(Math.abs(app.mjData.qpos[app.mjModel.jnt_qposadr[joint]]));
    }
    return angles.length ? angles.reduce((a, b) => a + b, 0) / angles.length : null;
  }

  /** Gap between the pads, in metres, inferred from the finger angles. */
  gripWidth(side) {
    const angle = this.fingerAngle(side);
    if (angle === null) return null;
    return 2 * (0.018 + 0.024 * Math.sin(angle));
  }

  /**
   * Close the gripper while keeping the middle of the pads on a target point.
   *
   * Closing pushes back on the wrist, and this arm's wrist joints are soft
   * enough (kp = 30) that the pads slide about 20 mm sideways while the jaws
   * travel - which is more than a 56 mm bottle's clearance, so the fingers
   * graze past it. Aligning once and then closing cannot fix that, because the
   * error appears during the close; correcting between steps can, and the
   * correction is the same measured pad position the rest of the harness uses.
   */
  async closeOnTarget(side, padTarget, quat, {
    step = 0.08, settle = 0.10, rounds = 14, drift = 0.006, blockedGap = 0.06,
  } = {}) {
    const OPEN = 0.7854;
    await this.alignPads(side, padTarget, quat, { tolerance: drift });
    let blocked = 0;
    let last = null;
    for (let i = 0; i < rounds; i++) {
      this.setGrip(side, Math.min(1, this.grip(side) + step));
      const from = this.app.mjData.time;
      while (this.app.mjData.time - from < settle) this.#step(1 / 120);
      const angle = this.fingerAngle(side);
      const commanded = OPEN * (1 - this.grip(side));
      last = await this.alignPads(side, padTarget, quat, { tolerance: drift, rounds: 2 })
        .catch((error) => ({ error: NaN, note: error.message }));
      if (angle !== null && angle - commanded > blockedGap) blocked += 1;
      else blocked = 0;
      if (blocked >= 2) {
        return {
          stalled: true, angle, commanded, width: this.gripWidth(side), grip: this.grip(side),
          padDrift: last?.error ?? null,
        };
      }
      if (i % 3 === 2) await yieldToPage();
    }
    return {
      stalled: false, angle: this.fingerAngle(side), commanded: OPEN * (1 - this.grip(side)),
      width: this.gripWidth(side), grip: this.grip(side), padDrift: last?.error ?? null,
    };
  }

  /**
   * Close the gripper a step at a time and report what stopped it.
   *
   * A blind close cannot tell "holding the bottle" from "closed on air", and
   * the difference only showed up later as a mysterious 30 mm drift during the
   * lift. The test here is not "has the angle stopped changing" - the fingers
   * lag their command, so that fires while they are still travelling - but
   * "is the angle behind the angle it was told to reach". Something in the
   * jaws is the only thing that can keep the fingers short of their command,
   * and how far short they are measures what they closed on.
   */
  async closeUntilStall(side, { step = 0.05, settle = 0.10, timeout = 3.0, blockedGap = 0.06 } = {}) {
    const OPEN = 0.7854;
    const startedAt = this.fingerAngle(side);
    const rounds = Math.max(4, Math.round(timeout / settle));
    let blockedRounds = 0;
    for (let i = 0; i < rounds; i++) {
      this.setGrip(side, Math.min(1, this.grip(side) + step));
      const from = this.app.mjData.time;
      while (this.app.mjData.time - from < settle) this.#step(1 / 120);
      const angle = this.fingerAngle(side);
      const commanded = OPEN * (1 - this.grip(side));
      if (angle !== null && angle - commanded > blockedGap) blockedRounds += 1;
      else blockedRounds = 0;
      if (blockedRounds >= 2) {
        return {
          stalled: true, angle, commanded, width: this.gripWidth(side),
          grip: this.grip(side), startedAt,
        };
      }
      if (i % 4 === 3) await yieldToPage();
    }
    return {
      stalled: false, angle: this.fingerAngle(side), commanded: OPEN * (1 - this.grip(side)),
      width: this.gripWidth(side), grip: this.grip(side), startedAt,
    };
  }

  // --- motion --------------------------------------------------------------

  setTarget(side, worldPos, worldQuat) {
    const { app } = this;
    const origin = app.controller.originPose(app.mjData);
    const local = poseWorldToLocal(origin, worldPos, worldQuat);
    app.teleop.arms[side].pos = local.pos;
    app.teleop.arms[side].quat = local.quat;
    app.applyTargets();
  }

  /**
   * Go to a pose by rising, crossing, then descending - never in a straight
   * line from wherever the hand happens to be.
   *
   * The straight-line version is what kept knocking the bottle over: from the
   * park pose to the grasp point the hand travelled diagonally, and that
   * diagonal passes straight through the space the bottle occupies. Measured,
   * the bottle ended up 154 mm away before the fingers had even closed.
   */
  async approachFromAbove(side, target, quat, { safeZ, seconds = 1.4, tolerance = 0.010 } = {}) {
    const start = this.tcp(side).pos;
    // The cruise height is fixed, not "at least the height we are already at":
    // the park pose sits at 1.39 m, and this arm cannot hold the 60-degree
    // grasp orientation that high - it lands 15 deg short. Rising or dropping
    // to the cruise height first, while still clear of the bottle, keeps the
    // sideways leg inside the region where the orientation is reachable.
    const cruise = safeZ;
    await this.moveTcp(side, [start[0], start[1], cruise], { seconds: 0.9, tolerance });
    await this.moveTcp(side, [target[0], target[1], cruise], { seconds, quat, tolerance });
    if (Math.abs(cruise - target[2]) > 0.002) {
      await this.moveTcp(side, target, { seconds: 0.8, quat, tolerance });
    }
    return this.settle(side, target, quat, { tolerance });
  }

  /**
   * Drive one TCP to a world pose and wait until it is actually there.
   * Throws with the measured numbers when it never settles, so a caller can
   * report a real failure instead of continuing on an arm that is elsewhere.
   */
  async moveTcp(side, target, {
    seconds = 1.6, quat = null, tolerance = 0.008, rotationTolerance = 0.20, arc = 0,
  } = {}) {
    const from = this.tcp(side);
    const fromTarget = this.app.targets?.[side]?.pos ?? from.pos;
    const goalQuat = quat ?? from.quat;
    const steps = Math.max(1, Math.round(seconds / FIXED_DT));
    for (let i = 1; i <= steps; i++) {
      const u = i / steps;
      const eased = u * u * (3 - 2 * u);
      const point = fromTarget.map((value, j) => value + (target[j] - value) * eased);
      if (arc) point[2] += arc * Math.sin(Math.PI * eased);
      this.setTarget(side, point, slerpQuat(from.quat, goalQuat, eased));
      this.#step(FIXED_DT);
      if (i % 8 === 7) await yieldToPage();
    }
    return this.settle(side, target, goalQuat, { tolerance, rotationTolerance });
  }

  /**
   * Step until the real end-effector error is inside tolerance.
   *
   * The rotation gate is deliberately loose. The wrist joints run on
   * kp = 30 with kv = 1.5, so they hold a few degrees of error under load no
   * matter what is asked of them; a tighter gate would fail moves that are
   * perfectly good for a grasp, and the achieved rotation is reported instead.
   */
  async settle(side, target, quat, { tolerance = 0.008, rotationTolerance = 0.20, seconds = 3 } = {}) {
    const steps = Math.round(seconds / FIXED_DT);
    let last = this.tcpError(side, target, quat);
    for (let i = 0; i < steps; i++) {
      last = this.tcpError(side, target, quat);
      const position = Math.hypot(...last.errorP);
      const rotation = Math.hypot(...last.errorR);
      if (position <= tolerance && rotation <= rotationTolerance) {
        return { error: position, rotation, converged: true };
      }
      this.#step(FIXED_DT);
      if (i % 8 === 7) await yieldToPage();
    }
    const actual = this.tcp(side);
    const error = norm(actual.pos, target);
    throw new Error(
      `${side} 臂未到达目标位姿：位置误差 ${(error * 100).toFixed(1)} cm，`
      + `姿态误差 ${((Math.hypot(...last.errorR) * 180) / Math.PI).toFixed(1)}°；`
      + `实际 (${actual.pos.map((v) => v.toFixed(3)).join(", ")}) m，`
      + `目标 (${target.map((v) => v.toFixed(3)).join(", ")}) m。`,
    );
  }
}
