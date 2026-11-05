// Replaying a recorded real episode in the simulator.
//
// This is the part of real-to-sim that the hand-authored tool set was
// standing in for. The lab robot recorded 148 successful runs of exactly this
// task, at 30 Hz, with all fourteen arm joints and both grippers - and those
// joints are the same joints this model has, so they can be commanded straight
// into the simulation. Replaying them answers a question that no amount of
// scripted motion can: when the real trajectory is fed to the sim, does the
// bottle end up where the real one ended up?
//
// Two conversions are needed and both are checked, not assumed:
//
//   * Coordinates. The recordings are in the robot's own base frame, where the
//     table top is z = 0.40. The cell scene puts the same table at z = 1.005
//     with the arms 0.185 further back (the upstream pedestal -> cell offset),
//     so comparing end-effector paths needs that shift.
//   * Grippers. The recorded signal runs from about -1.0 (open) to -0.45
//     (closed), which is a command, not a joint angle. It is mapped affinely
//     onto the model's 0..0.7854 rad finger travel.

// Robot base frame -> scene world frame, from Robot.BASE_TO_WORLD.
//
// What this is: the constant that has to be subtracted before the recorded
// end-effector positions and this model's agree. With it, pure forward
// kinematics over the recorded joint angles reproduces the recorded positions
// to 0.0000 m across 849 frames.
//
// What that result does and does not say. The recording's `epos` is itself
// computed from joint angles, by the lab machine's `openarm_kinematics`
// ("根据 URDF 文件 openarm_bimanual.urdf 实现"), not measured by an external
// tracker. So a zero residual says two software implementations of the same
// kinematics agree - it does not say the physical robot is installed where the
// URDF says, that its joint zeros are right, that its links are machined to
// drawing, or that nothing deflects under load. Treat this as a frame
// conversion for moving data between the two, not as a measurement of
// anything physical.
//
// The upstream note gives pedestal -> cell as (0.185, 0, 0.605); x agrees, z
// differs by 37 mm because the lab robot is a mobile bimanual whose table sits
// at a different height in its own frame.
import { BASE_TO_WORLD } from "./robot.js";

const BASE_TO_CELL = BASE_TO_WORLD;
// The recording carries the gripper twice, and the two channels are different
// quantities:
//
//   *_action_gripper   what was sent to the motors - the command
//   *_gripper          what the encoders read back - the result
//
// They are also on different scales. Commanded wide open, the action channel
// sits at -0.90 while the state channel reads about -1.06; sent shut, the
// action goes to +0.05 and the state only follows as far as the jaws are
// allowed to travel.
//
// This matters more than it looks. On this episode the right hand is commanded
// shut at t = 9.33 s and its encoder stops at -0.58, because the bottle is
// between the jaws. **That gap between command and result is the grasp.** An
// earlier version of this file fed the state channel back in as the command, so
// the simulation was told to close to exactly the width the real fingers had
// stopped at - it was handed the answer - and it duly shut the jaws to 94 % of
// their travel around empty air and left the bottle standing on the table.
// Use the action channel to drive the fingers and let contact decide where they
// stop; the state channel is only good for labelling which stage we are in.
const CMD_OPEN = -0.90;
const CMD_CLOSED = 0.05;
const OPEN_ANGLE = 0.7854;

/**
 * A recorded gripper signal -> how far open the sim's fingers should be, 0..1.
 *
 * Affine between the channel's endpoints and clamped, so a reading from the
 * other channel (or any excursion) pins the jaws wide rather than
 * extrapolating past the model's joint limits.
 */
export function gripperOpenness(value) {
  const t = (value - CMD_CLOSED) / (CMD_OPEN - CMD_CLOSED);
  return Math.max(0, Math.min(1, t));
}

export function toCell([x, y, z]) {
  return [x + BASE_TO_CELL[0], y + BASE_TO_CELL[1], z + BASE_TO_CELL[2]];
}

/**
 * Which of the seven stages the recording is in, read off the recorded
 * commands themselves: the right hand closes to take the bottle, then the left
 * closes while the right is still holding, then the right opens, then the left
 * opens again over the basket.
 *
 * The action channel is used rather than the encoder channel because it is the
 * cleaner signal for this: the operator's command is exactly -0.90 or exactly
 * +0.05, where the encoder drifts by a few hundredths under load and would put
 * a threshold in the middle of the noise.
 */
export function stageAt(episode, frame) {
  const g0 = episode.left_action_gripper[frame];
  const g1 = episode.right_action_gripper[frame];
  const rightClosed = gripperOpenness(g1) < 0.35;
  const leftClosed = gripperOpenness(g0) < 0.35;
  const leftStartedClosing = episode.left_action_gripper.findIndex((v) => gripperOpenness(v) < 0.35);
  const rightOpenedAgain = episode.right_action_gripper.findIndex(
    (v, i) => i > leftStartedClosing && leftStartedClosing >= 0 && gripperOpenness(v) > 0.6,
  );
  const leftOpened = episode.left_action_gripper.findIndex(
    (v, i) => rightOpenedAgain >= 0 && i > rightOpenedAgain && gripperOpenness(v) > 0.6,
  );
  if (leftOpened >= 0 && frame >= leftOpened) return 6;
  if (rightOpenedAgain >= 0 && frame >= rightOpenedAgain) return 5;
  if (leftClosed) return 4;
  if (rightClosed) return 3;
  if (frame > 0) return 1;
  return 0;
}

export class Replay {
  constructor(robot, trajectory) {
    this.robot = robot;
    this.trajectory = trajectory;
  }

  get episodeIds() {
    return Object.keys(this.trajectory.episodes);
  }

  /**
   * Command the model's joints straight from the recording for one control
   * period, then let the physics run.
   */
  #applyFrame(episode, frame, { kinematic }) {
    const { app } = this.robot;
    // The lifter carries the arms and is part of the calibrated scene. Writing
    // only the fourteen arm joints left it wherever the last reset put it -
    // back at the cell's default - so the replay ran with the arms 0.15 m
    // lower than the calibration says and drove them through the tabletop.
    //
    // Written, never commanded, in *both* modes. Its position actuator settles
    // 45 mm short of its target under the weight of both arms, and that 45 mm
    // lands straight in the z of everything the replay measures. The shortfall
    // is a property of the simulator's actuator model, not of the real
    // pedestal, which holds its commanded height; leaving it in the loop would
    // charge the arm servos for a pedestal that is not theirs.
    const lifterJoint = this.robot.lifterJointId;
    if (lifterJoint !== undefined && lifterJoint >= 0 && this.robot.lifterHeight) {
      app.mjData.qpos[app.mjModel.jnt_qposadr[lifterJoint]] = this.robot.lifterHeight;
      app.mjData.qvel[app.mjModel.jnt_dofadr[lifterJoint]] = 0;
      // Its actuator is sent to the same place even though the joint is
      // written. Left at the model's default the servo pulls the pedestal back
      // towards zero, and the replay spends the episode writing 0.15 and the
      // actuator pulling against it - which pushed the end-effector comparison
      // from 5 mm to 425 mm the first time this was tried.
      if (this.robot.lifterActuatorId >= 0) {
        app.mjData.ctrl[this.robot.lifterActuatorId] = this.robot.lifterHeight;
      }
    }
    const arms = ["left", "right"];
    // Two different channels, and they are not interchangeable:
    //   *_joints        what the encoders measured that frame
    //   *_action_joints what the policy asked for that frame
    // Kinematic replay pins the measured positions; commanded replay has to
    // send the commands, or it is just replaying the result into a controller
    // and calls zero tracking error a success. An earlier version sent the
    // measured positions down the commanded branch.
    const measured = [episode.left_joints[frame], episode.right_joints[frame]];
    const commanded = [episode.left_action_joints[frame], episode.right_action_joints[frame]];
    const joints = kinematic ? measured : commanded;
    // The fingers are driven from the command in both modes. Kinematic replay
    // pins the arm joints because the simulated servos lag the recording by
    // more than the discrepancy being measured - but the fingers are the one
    // place where contact is the whole point, so they are never written
    // directly, only asked.
    const grips = [episode.left_action_gripper[frame], episode.right_action_gripper[frame]];
    arms.forEach((side, i) => {
      const arm = app.controller.arms[side];
      if (kinematic) {
        // Kinematic replay: the arms are written straight to the recorded
        // joint positions instead of being asked for through the position
        // actuators. Measured, the actuators lag the recording by up to
        // 100 mm, which is larger than the discrepancy the replay exists to
        // measure - driving the arms directly removes the simulator's control
        // error from the comparison and leaves only the scene and the physics
        // of the objects. The fingers are still commanded normally, so the
        // grasp itself stays a contact result.
        const current = joints[i];
        for (let j = 0; j < 7; j++) {
          app.mjData.qpos[arm.qposIds[j]] = current[j];
          app.mjData.ctrl[arm.actIds[j]] = current[j];
        }
      } else {
        for (let j = 0; j < 7; j++) app.mjData.ctrl[arm.actIds[j]] = joints[i][j];
      }
      const [lo, hi] = arm.gripperRange;
      const end = Math.abs(lo) > Math.abs(hi) ? lo : hi;
      app.mjData.ctrl[arm.gripperActId] = end * gripperOpenness(grips[i]);
    });
  }

  /**
   * Play one episode and measure how far the simulated hands drift from the
   * recorded ones. A large deviation is the honest answer to "does this scene
   * reproduce the real one", and it is reported per phase rather than averaged
   * into one number that hides where it goes wrong.
   */
  async run(episodeId, {
    sampleEvery = 3, onProgress, onStage, onFrame, kinematic = true, autoCalibrate = false,
  } = {}) {
    const episode = this.trajectory.episodes[episodeId];
    if (!episode) throw new Error(`没有第 ${episodeId} 条轨迹。`);
    // The frame constant is known exactly from the forward-kinematics probe
    // (BASE_TO_CELL below), so the self-calibration pass is off by default: it
    // is a whole extra episode of simulation and one more place for the run to
    // diverge from the measurement.
    if (autoCalibrate && !this.fittedOffset) {
      // Pass one: the same replay with the objects set aside, purely to measure
      // the constant between the recorded end-effector and the model's control
      // point. Doing it here rather than trusting the constant in the source
      // means the scene and the calibration cannot drift apart silently.
      await this.#calibrationPass(episodeId, episode);
    }
    const { app, mujoco } = this.robot;
    const stepsPerFrame = Math.max(1, Math.round(1 / (episode.fps * app.mjModel.opt.timestep)));

    // The page's own update cycle writes the joints from the teleop targets
    // every frame, which silently overwrites everything the replay just
    // commanded - the arms went back to their park pose the first time this
    // ran. Holding the override for the duration is the smallest fix that
    // keeps the upstream page untouched.
    const realApplyTargets = app.applyTargets;
    const realUpdate = app.update;
    app.applyTargets = () => {};
    // Taking a keyframe calls the page's update, and its catch-up loop steps
    // the physics until it reaches the frame's target time - stepping without
    // the replay's per-step joint write. The arms drifted between samples, and
    // the comparison then blamed the scene for it. Pinning the target time to
    // the current time makes that loop a no-op while still syncing the meshes.
    app.update = (dt, now, xrFrame) => {
      app.simTarget = app.mjData.time;
      return realUpdate.call(app, 0, now, xrFrame);
    };
    try {
      return await this.#play(episodeId, episode, stepsPerFrame,
        { sampleEvery, onProgress, onStage, onFrame, kinematic });
    } finally {
      app.applyTargets = realApplyTargets;
      app.update = realUpdate;
    }
  }

  /** One physics step at a fixed frame, used while the scene is being set up. */
  #applyFrameSettle(episode, frame, kinematic) {
    this.#applyFrame(episode, frame, { kinematic });
    this.robot.app.controller.applyGravityComp(this.robot.app.mjData);
    this.robot.mujoco.mj_step(this.robot.app.mjModel, this.robot.app.mjData);
  }

  /** One pass with nothing in the way, to fit the frame constant. */
  async #calibrationPass(episodeId, episode) {
    const { app, mujoco } = this.robot;
    const bottle = this.robot.bottle;
    const boxBody = this.robot.box.bodyId;
    const savedQpos = [...app.mjData.qpos.subarray(bottle.qpos, bottle.qpos + 7)];
    const savedBox = [app.mjModel.body_pos[boxBody * 3], app.mjModel.body_pos[boxBody * 3 + 1], app.mjModel.body_pos[boxBody * 3 + 2]];
    app.mjData.qpos[bottle.qpos] = 5;
    app.mjData.qpos[bottle.qpos + 1] = 5;
    app.mjData.qpos[bottle.qpos + 2] = -1;
    app.mjModel.body_pos[boxBody * 3] = 5;
    const previous = this.fittedOffset;
    try {
      const pass = await this.#play(episodeId, episode, 1, { sampleEvery: 6, kinematic: true });
      const fit = {};
      for (const side of ["left", "right"]) {
        const medians = [0, 1, 2].map((axis) => {
          const values = [];
          for (const entry of pass.trace) {
            const residual = entry.measured[side].map((v, i) => v - entry.recorded[side][i]);
            values.push(rotateVector(entry.quats[side], residual)[axis]);
          }
          return medianOf(values);
        });
        const spread = Math.hypot(medianOf([...pass.trace].map((entry) => {
          const residual = entry.measured[side].map((v, i) => v - entry.recorded[side][i]);
          const tool = rotateVector(entry.quats[side], residual);
          return Math.hypot(...tool.map((v, i) => v - medians[i]));
        })));
        fit[side] = { offset_mm: medians.map((v) => v * 1000), spread_mm: spread * 1000 };
      }
      this.fittedOffset = fit;
    } finally {
      this.fittedOffset = this.fittedOffset ?? previous;
      app.mjModel.body_pos[boxBody * 3] = savedBox[0];
      app.mjData.qpos.set(savedQpos, bottle.qpos);
      mujoco.mj_forward(app.mjModel, app.mjData);
    }
  }

  /**
   * One frame of the object's story: where the bottle is, how upright it is,
   * how wide each gripper is, and whether either gripper is touching it.
   *
   * This is the half of a replay the end-effector comparison cannot see. That
   * comparison answers "did the hands go where the recording says"; it says
   * nothing about whether the bottle was picked up, carried and put in the
   * box. On this episode the two answers came apart completely - the arms
   * tracked the recording to a few millimetres while the fingers shut on empty
   * air and the bottle never left the table. A replay that reports only the
   * first number would have called that a pass.
   */
  #observeObject(rows, events, frame, episode) {
    const { app, mujoco } = this.robot;
    const data = app.mjData;
    const id = this.robot.bottle.bodyId;
    const m = data.xmat.subarray(id * 9, id * 9 + 9);
    const tilt = (Math.acos(Math.min(1, Math.max(-1, m[8]))) * 180) / Math.PI;
    const pos = [data.xpos[id * 3], data.xpos[id * 3 + 1], data.xpos[id * 3 + 2]];
    const touching = { left: false, right: false };
    if (data.ncon !== undefined) {
      const name = (g) => mujoco.mj_id2name(app.mjModel, mujoco.mjtObj.mjOBJ_GEOM.value, g) ?? "";
      for (let i = 0; i < data.ncon; i++) {
        let c = null;
        try { c = data.contact?.get ? data.contact.get(i) : data.contact?.[i]; } catch { continue; }
        if (!c || c.geom1 === undefined) continue;
        for (const [a, b] of [[c.geom1, c.geom2], [c.geom2, c.geom1]]) {
          const na = name(a);
          if (!na.startsWith("finger_") || !name(b).startsWith("bottle")) continue;
          if (na.includes("_left_")) touching.left = true;
          if (na.includes("_right_")) touching.right = true;
        }
      }
    }
    const row = {
      frame,
      t: frame / episode.fps,
      pos,
      tilt,
      grip: { left: this.robot.gripWidth("left"), right: this.robot.gripWidth("right") },
      touching,
    };
    rows.push(row);
    const origin = rows[0];
    const away = Math.hypot(pos[0] - origin.pos[0], pos[1] - origin.pos[1]);
    if (!events.firstMoved && away > 0.010) events.firstMoved = { frame, t: row.t, away_mm: away * 1000 };
    if (!events.firstTipped && tilt > 10) events.firstTipped = { frame, t: row.t, tilt };
    if (!events.firstFingerContact && (touching.left || touching.right)) {
      events.firstFingerContact = {
        frame, t: row.t, side: touching.right ? "right" : "left",
        grip_mm: (touching.right ? row.grip.right : row.grip.left) * 1000,
        tilt,
      };
    }
    if (!events.maxTilt || tilt > events.maxTilt.tilt) events.maxTilt = { frame, t: row.t, tilt };
    if (!events.lifted && pos[2] > origin.pos[2] + 0.03) {
      events.lifted = { frame, t: row.t, height_mm: (pos[2] - origin.pos[2]) * 1000 };
    }
  }

  async #play(episodeId, episode, stepsPerFrame, { sampleEvery, onProgress, onStage, onFrame, kinematic }) {
    const { app, mujoco } = this.robot;
    const bottle = this.robot.bottle;
    const start = [...this.robot.startPose];

    // Order matters here, and finding that out cost a long time. The lifter
    // first, then the bottle, then the arms:
    //
    //   * the reset must not park the arms (`arms: false`) - the model's home
    //     keyframe sits right on top of the bottle's column, and the first
    //     version flung it to x = -13 m before the episode even started;
    //   * putting the arms at the recording's first pose *before* restoring
    //     the bottle left the solver to resolve a loaded step with the arms
    //     already moved, and the bottle came out of it at 4 m/s - 118 m across
    //     the floor by the end. With this order the same episode pushes it
    //     gently instead.
    this.robot.reset({ arms: false });
    const lifterJoint = mujoco.mj_name2id(app.mjModel, mujoco.mjtObj.mjOBJ_JOINT.value, "openarm_lifter_joint");
    if (lifterJoint >= 0 && this.robot.lifterHeight) {
      app.mjData.qpos[app.mjModel.jnt_qposadr[lifterJoint]] = this.robot.lifterHeight;
      app.mjData.qvel[app.mjModel.jnt_dofadr[lifterJoint]] = 0;
    }
    app.mjData.qpos.set(start, bottle.qpos);
    app.mjData.qvel.fill(0, bottle.dof, bottle.dof + 6);
    for (const side of ["left", "right"]) {
      const arm = app.controller.arms[side];
      const first = episode[`${side}_joints`][0];
      for (let j = 0; j < 7; j++) {
        app.mjData.qpos[arm.qposIds[j]] = first[j];
        app.mjData.qvel[arm.dofIds[j]] = 0;
      }
      // The fingers too. They are driven by ctrl, not written, so whatever they
      // happened to be holding when the run started stayed with them: a replay
      // launched right after the page had been sitting idle closed on nothing
      // and began the episode with the jaws already shut, and the same episode
      // then reported its first contact 6 s later than it does from a cold
      // start. Everything the episode does not itself define is pinned here.
      const [lo, hi] = arm.gripperRange;
      const end = Math.abs(lo) > Math.abs(hi) ? lo : hi;
      const want = end * gripperOpenness(episode[`${side}_action_gripper`][0]);
      for (const joint of [`openarm_${side}_finger_joint1`, `openarm_${side}_finger_joint2`]) {
        const id = mujoco.mj_name2id(app.mjModel, mujoco.mjtObj.mjOBJ_JOINT.value, joint);
        if (id < 0) continue;
        app.mjData.qpos[app.mjModel.jnt_qposadr[id]] = want;
        app.mjData.qvel[app.mjModel.jnt_dofadr[id]] = 0;
      }
    }
    mujoco.mj_forward(app.mjModel, app.mjData);
    const trace = [];
    const byStage = new Map();
    const toolOffsets = { left: [], right: [] };
    let lastStage = -1;
    // The object's story, kept separately from the end-effector comparison.
    // Whether the hands go where the recording says is one question; whether
    // the object does what it did is another, and it is the one that decides
    // if the task is reproduced. A replay that tracks the arms to 3 mm and
    // leaves the bottle standing has not replayed the episode.
    const objectFrames = [];
    const events = { firstMoved: null, firstTipped: null, firstFingerContact: null, maxTilt: null, lifted: null };
    const boxPose = this.robot.calibration?.objects?.box;
    const boxHalf = boxPose ? [
      boxPose.inner_size.value[0] / 2,
      boxPose.inner_size.value[1] / 2,
    ] : null;

    for (let frame = 0; frame < episode.frames; frame++) {
      // Write the frame's joints and settle the kinematics *before* stepping,
      // then measure. Sampling after the step reads the next frame's pose, so
      // the comparison carried a speed-dependent few millimetres that had
      // nothing to do with the scene.
      this.#applyFrame(episode, frame, { kinematic });
      mujoco.mj_kinematics(app.mjModel, app.mjData);
      const measured = {};
      const stage = stageAt(episode, frame);
      if (stage !== lastStage) {
        onStage?.(stage, frame, frame / episode.fps);
        lastStage = stage;
      }
      const sampled = frame % sampleEvery === 0;
      const recorded = {};
      const deviation = {};
      const quats = {};
      for (const side of ["left", "right"]) {
        const pose = this.robot.tcp(side);
        measured[side] = pose.pos;
        quats[side] = pose.quat;
        recorded[side] = toCell(side === "left" ? episode.left_ee_xyz[frame] : episode.right_ee_xyz[frame]);
        const raw = measured[side].map((v, i) => v - recorded[side][i]);
        // The recorded end-effector is the robot's link7 origin; the model's
        // control point is the wrist it hangs off. Those two are not the same
        // point, so the raw difference carries a constant offset that says
        // nothing about whether the scene matches. Expressing it in the hand's
        // own frame separates the two: a constant part is the TCP convention,
        // and only what is left over is a real disagreement.
        if (sampled) toolOffsets[side].push(rotateVector(pose.quat, raw));
        deviation[side] = Math.hypot(...raw);
      }
      for (let k = 0; k < stepsPerFrame; k++) {
        this.#applyFrame(episode, frame, { kinematic });
        app.controller.applyGravityComp(app.mjData);
        mujoco.mj_step(app.mjModel, app.mjData);
      }
      this.#observeObject(objectFrames, events, frame, episode);
      if (!sampled) continue;
      const entry = {
        t: frame / episode.fps, frame, stage,
        measured, recorded, deviation, quats,
        bottle: this.robot.bottlePose().pos,
      };
      trace.push(entry);
      for (const side of ["left", "right"]) {
        const bucket = byStage.get(stage) ?? [];
        bucket.push(deviation[side]);
        byStage.set(stage, bucket);
      }
      onFrame?.(entry);
      if (frame % (sampleEvery * 20) === 0) await new Promise((r) => setTimeout(r, 0));
      onProgress?.(frame / episode.frames);
    }

    const summary = [...byStage.entries()].sort((a, b) => a[0] - b[0]).map(([stage, values]) => ({
      stage,
      samples: values.length,
      mean_mm: (values.reduce((a, b) => a + b, 0) / values.length) * 1000,
      max_mm: Math.max(...values) * 1000,
    }));

    // Fit one constant offset per hand, then measure again with it removed.
    const fitted = this.fittedOffset ?? {};
    const calibrated = new Map();
    for (const side of ["left", "right"]) {
      const offsets = toolOffsets[side];
      const median = [0, 1, 2].map((axis) => medianOf(offsets.map((o) => o[axis])));
      const spread = Math.hypot(...[0, 1, 2].map(
        (axis) => medianOf(offsets.map((o) => Math.abs(o[axis] - median[axis]))),
      ));
      fitted[side] = { offset_mm: median.map((v) => v * 1000), spread_mm: spread * 1000 };
      for (const entry of trace) {
        const residual = entry.measured[side].map((v, i) => v - entry.recorded[side][i]);
        const tool = rotateVector(entry.quats[side], residual);
        const corrected = hypot3(tool.map((v, i) => v - median[i]));
        const bucket = calibrated.get(entry.stage) ?? [];
        bucket.push(corrected);
        calibrated.set(entry.stage, bucket);
      }
    }
    const calibratedSummary = [...calibrated.entries()].sort((a, b) => a[0] - b[0]).map(([stage, values]) => ({
      stage,
      samples: values.length,
      mean_mm: (values.reduce((a, b) => a + b, 0) / values.length) * 1000,
      max_mm: Math.max(...values) * 1000,
    }));
    return {
      trace, summary, calibratedSummary, fitted, kinematic,
      frames: episode.frames, seconds: episode.frames / episode.fps,
      bottle: this.robot.bottlePose().pos,
      objects: objectFrames,
      events,
      outcome: this.#outcome(objectFrames, events, boxPose, boxHalf),
    };
  }

  /**
   * Did the episode actually happen? Not "did the arms track" - did the bottle
   * get picked up, carried and left in the box.
   *
   * The two are independent questions and this replay is the proof: the arms
   * tracked the recording to 3.6 mm mean and the bottle never moved. An
   * end-effector residual is a scene-consistency check; this is the task.
   */
  #outcome(objectFrames, events, boxPose, boxHalf) {
    const last = objectFrames[objectFrames.length - 1];
    const reasons = [];
    if (!events.firstFingerContact) reasons.push("两只手的手指全程没有碰到瓶子");
    if (!events.lifted) reasons.push("瓶子始终没有离开台面");
    let inBox = null;
    if (boxPose && boxHalf && last) {
      const [bx, by, bz] = boxPose.pose.value;
      const inside = Math.abs(last.pos[0] - bx) < boxHalf[0]
        && Math.abs(last.pos[1] - by) < boxHalf[1]
        && last.pos[2] < bz + boxPose.wall_height.value;
      inBox = { inside, xy_mm: [last.pos[0] - bx, last.pos[1] - by].map((v) => v * 1000) };
      if (!inside) reasons.push("瓶子最后不在白盒内");
    }
    return {
      passed: reasons.length === 0,
      inBox,
      finalTilt: last?.tilt ?? null,
      displaced_mm: last ? Math.hypot(last.pos[0] - objectFrames[0].pos[0], last.pos[1] - objectFrames[0].pos[1]) * 1000 : null,
      reasons,
    };
  }
}

function rotateVector(quat, vector) {
  const [w, x, y, z] = quat;
  const m = [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ];
  return [0, 1, 2].map((c) => vector[0] * m[c] + vector[1] * m[3 + c] + vector[2] * m[6 + c]);
}

function medianOf(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0;
}

function hypot3(v) {
  return Math.hypot(v[0], v[1], v[2]);
}
