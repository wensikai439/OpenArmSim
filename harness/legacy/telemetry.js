// Diagnostics the platform keeps but does not draw.
//
// The interface shows the operator three things: the conversation, the camera,
// and the stage board. Everything else the harness can measure is useful for
// judging a run - and would ruin the page if it were on it. So it lives here,
// readable on demand:
//
//     __lab.telemetry.contacts()      what is touching what
//     __lab.telemetry.arms()          commanded vs achieved, per joint
//     __lab.telemetry.objects()       where the task objects are
//     __lab.telemetry.views()         the same moment from every camera
//     __lab.telemetry.snapshot()      all of the above, as JSON
//
// Same idea as the agent's perception having a wider field than the panel: the
// instrument reads more than the operator needs to see.

import * as perception from "./perception.js";

const TASK_GEOMS = ["bottle_body", "bottle_neck", "bottle_cap"];

export function createTelemetry({ robot, app, harness, calib }) {
  const { mujoco } = robot;
  const geomName = (id) => mujoco.mj_id2name(app.mjModel, mujoco.mjtObj.mjOBJ_GEOM.value, id) ?? `geom#${id}`;
  const bodyName = (id) => mujoco.mj_id2name(app.mjModel, mujoco.mjtObj.mjOBJ_BODY.value, id) ?? `body#${id}`;

  function geomId(name) {
    return mujoco.mj_name2id(app.mjModel, mujoco.mjtObj.mjOBJ_GEOM.value, name);
  }

  /**
   * Every contact, with the geoms named.
   *
   * The binding's contact array is not a plain JS array in every build, so this
   * tries both shapes and returns an empty list rather than throwing - a
   * diagnostic that breaks the run it is diagnosing is worse than none.
   */
  function contacts({ involvingTaskObjects = false } = {}) {
    const data = app.mjData;
    if (!data.contact || data.ncon === undefined) return { available: false, contacts: [] };
    const task = new Set(TASK_GEOMS.map(geomId));
    const out = [];
    for (let i = 0; i < data.ncon; i++) {
      let c = null;
      try {
        c = data.contact.get ? data.contact.get(i) : data.contact[i];
      } catch {
        continue;
      }
      if (!c || c.geom1 === undefined) continue;
      const a = c.geom1;
      const b = c.geom2;
      if (involvingTaskObjects && !task.has(a) && !task.has(b)) continue;
      const force = c.force;
      out.push({
        a: geomName(a),
        b: geomName(b),
        bodyA: bodyName(app.mjModel.geom_bodyid[a]),
        bodyB: bodyName(app.mjModel.geom_bodyid[b]),
        force: force && force.length >= 3
          ? Number(Math.hypot(force[0], force[1], force[2]).toFixed(4))
          : null,
      });
    }
    return { available: true, count: out.length, contacts: out };
  }

  /** Commanded, achieved, and the gap, for every arm joint. */
  function arms() {
    const out = {};
    for (const side of ["left", "right"]) {
      const arm = app.controller.arms[side];
      const joints = arm.qposIds.map((q, i) => {
        const commanded = app.mjData.ctrl[arm.actIds[i]];
        const achieved = app.mjData.qpos[q];
        return {
          joint: i + 1,
          commanded: Number(commanded.toFixed(4)),
          achieved: Number(achieved.toFixed(4)),
          error: Number((commanded - achieved).toFixed(4)),
        };
      });
      const finger = mujoco.mj_name2id(
        app.mjModel, mujoco.mjtObj.mjOBJ_JOINT.value, `openarm_${side}_finger_joint1`,
      );
      out[side] = {
        joints,
        worstJoint: joints.reduce((w, j) => (Math.abs(j.error) > Math.abs(w.error) ? j : w), joints[0]),
        gripper: {
          achievedRad: finger >= 0 ? Number(app.mjData.qpos[app.mjModel.jnt_qposadr[finger]].toFixed(4)) : null,
          commandedGrip: Number(robot.grip(side).toFixed(3)),
        },
        tcp: robot.tcp(side).pos.map((v) => Number(v.toFixed(4))),
      };
    }
    return out;
  }

  /** Where the task objects are, with orientation, not just position. */
  function objects() {
    const pose = (which) => {
      const id = robot[which].bodyId;
      const m = app.mjData.xmat.subarray(id * 9, id * 9 + 9);
      // tilt of the body's local z away from world z - reads "standing" or
      // "lying down" without anyone having to interpret a matrix
      const tiltDeg = (Math.acos(Math.min(1, Math.max(-1, m[8]))) * 180) / Math.PI;
      return {
        pos: [app.mjData.xpos[id * 3], app.mjData.xpos[id * 3 + 1], app.mjData.xpos[id * 3 + 2]]
          .map((v) => Number(v.toFixed(4))),
        tiltDeg: Number(tiltDeg.toFixed(1)),
        posture: tiltDeg < 20 ? "standing" : tiltDeg > 70 ? "lying" : "tilted",
      };
    };
    return { bottle: pose("bottle"), box: pose("box"), time: Number(robot.time().toFixed(3)) };
  }

  /**
   * The same instant from every camera the model has, as data URLs.
   *
   * Not shown in the page - this is for looking at when a run goes wrong and
   * the window cannot explain why.
   */
  function views({ hideRobot = false } = {}) {
    const out = {};
    const names = ["camera_head_left", "camera_head_right", "camera_wrist_left", "camera_wrist_right", "camera_ceiling"];
    for (const name of names) {
      const id = mujoco.mj_name2id(app.mjModel, mujoco.mjtObj.mjOBJ_CAMERA.value, name);
      if (id < 0) continue;
      const shot = perception.sampleFromModelCamera(app, mujoco, id, { hideRobot });
      if (shot) out[name] = { image: perception.frameToDataUrl(shot), width: shot.width, height: shot.height };
    }
    out.perception_a = { image: perception.frameToDataUrl(perception.sample(app, harness.cameras.a, { hideRobot })) };
    out.perception_b = { image: perception.frameToDataUrl(perception.sample(app, harness.cameras.b, { hideRobot })) };
    return out;
  }

  function snapshot() {
    return {
      time: robot.time(),
      arms: arms(),
      objects: objects(),
      contacts: contacts(),
      taskContacts: contacts({ involvingTaskObjects: true }),
      scene: robot.sceneChanges ?? null,
    };
  }

  return { contacts, arms, objects, views, snapshot };
}
