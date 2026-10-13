import { BASE_TO_WORLD } from "./robot.js";
// Episode record. The previous version kept keyframes in page memory and lost
// them on refresh, which made every run unreproducible. This writes the whole
// run - configuration, per-step measurements and keyframes - to runs/ on the
// server when the operator asks for it.

export class Episode {
  constructor(meta) {
    this.meta = meta;
    this.startedAt = new Date().toISOString();
    this.steps = [];
    this.frames = [];
    this.trajectory = [];
    this.currentStage = null;
    this.controlHz = 30;
  }

  /**
   * One control-rate sample: the measured state and the commanded action, both
   * in the recordings' 37-number layout. This is what a training loader reads.
   */
  sample(robot, stage = this.currentStage) {
    const state = Episode.stateVector(robot);
    const action = Episode.actionVector(robot);
    this.trajectory.push({
      t: Number(robot.time().toFixed(4)),
      stage,
      state,
      action,
    });
  }

  record(step, robot = null) {
    if (step.stage !== undefined && step.stage !== null) this.currentStage = step.stage;
    const entry = { index: this.steps.length, at: new Date().toISOString(), ...step };
    if (robot) entry.state37 = Episode.stateVector(robot);
    this.steps.push(entry);
  }

  /**
   * The robot's state in the same 37-number layout the lab recordings use, so
   * a simulated episode can be turned into training data by the same converter
   * the real ones go through.
   *
   *   left joints (7) | left gripper | right joints (7) | right gripper |
   *   epos_0 quat (4) | epos_0 xyz (3) | epos_1 quat (4) | epos_1 xyz (3) |
   *   body pose (7, zeros for a fixed base)
   *
   * The gripper channels are written in the recordings' convention, which is
   * more negative when the jaws are wide, not the model's 0..0.7854 rad.
   */
  static actionVector(robot) {
    const out = [];
    for (const side of ["left", "right"]) {
      const arm = robot.app.controller.arms[side];
      for (let j = 0; j < 7; j++) out.push(robot.app.mjData.ctrl[arm.actIds[j]]);
      // The recordings' action channel runs -0.9 (open) to +0.05 (closed).
      out.push(-0.9 + 0.95 * robot.grip(side));
    }
    const epos = (side) => [
      ...robot.tcp(side).quat,
      ...robot.tcp(side).pos.map((x, i) => Number((x - BASE_TO_WORLD[i]).toFixed(6))),
    ];
    out.push(...epos("left"), ...epos("right"), 1, 0, 0, 0, 0, 0, 0);
    return out.map((v) => Number(Number(v).toFixed(6)));
  }

  static stateVector(robot, baseToWorld = BASE_TO_WORLD) {
    // End-effector in the *recording's* frame, not this scene's: subtract the
    // frame conversion so a simulated episode and a recorded one are in the
    // same coordinates. (See Robot.BASE_TO_WORLD for what that constant is and
    // is not.)
    const inBase = (v) => v.map((x, i) => Number((x - baseToWorld[i]).toFixed(6)));
    // Gripper from the finger joint that actually moved, in the recordings'
    // convention (about -1.06 wide, -0.44 closed). An earlier version wrote the
    // commanded grip instead, which is exactly the channel that disagrees with
    // reality whenever something is between the jaws.
    const gripperState = (side) => {
      const arm = robot.app.controller.arms[side];
      const joint = robot.mujoco.mj_name2id(
        robot.app.mjModel, robot.mujoco.mjtObj.mjOBJ_JOINT.value, `openarm_${side}_finger_joint1`,
      );
      const angle = joint >= 0 ? Math.abs(robot.app.mjData.qpos[robot.app.mjModel.jnt_qposadr[joint]]) : 0;
      return Number((-0.44 - 0.62 * (angle / 0.7854)).toFixed(6));
    };
    const joints = (side) => robot.app.controller.arms[side].qposIds
      .map((i) => Number(robot.app.mjData.qpos[i].toFixed(6)));
    const pose = (side) => ({
      quat: robot.tcp(side).quat.map((v) => Number(v.toFixed(6))),
      xyz: inBase(robot.tcp(side).pos),
    });
    const left = pose("left");
    const right = pose("right");
    return [
      ...joints("left"), gripperState("left"),
      ...joints("right"), gripperState("right"),
      ...left.quat, ...left.xyz,
      ...right.quat, ...right.xyz,
      1, 0, 0, 0, 0, 0, 0,
    ];
  }

  frame(label, dataUrl, simTime) {
    this.frames.push({
      label,
      time: Number(simTime.toFixed(3)),
      step: this.steps.length,
      dataUrl,
    });
    return this.frames.length - 1;
  }

  get finished() {
    return this.steps.length > 0;
  }

  summary() {
    const failed = this.steps.filter((step) => !step.ok);
    return {
      steps: this.steps.length,
      keyframes: this.frames.length,
      failed: failed.length,
      last: this.steps.at(-1)?.tool ?? null,
      outcome: failed.length ? "failed" : "ok",
    };
  }

  toJSON() {
    return {
      id: this.meta.id,
      task: this.meta.task,
      planner: this.meta.planner,
      started_at: this.startedAt,
      finished_at: new Date().toISOString(),
      simulator: this.meta.simulator,
      calibration: this.meta.calibration,
      geometry: this.meta.geometry,
      stages: this.meta.stages,
      summary: this.summary(),
      trajectory_hz: this.controlHz,
      steps: this.steps,
      frames: this.frames,
      trajectory: this.trajectory,
    };
  }

  async save() {
    const response = await fetch("/api/episode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(this.toJSON()),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error ?? `HTTP ${response.status}`);
    return payload;
  }
}
