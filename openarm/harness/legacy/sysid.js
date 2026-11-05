// System identification for the arm's servos.
//
// The gap this closes: with the scene corrected, the simulated arm tracks its
// commands *better* than the real one does - 0.001-0.009 rad against the real
// robot's 0.005-0.032 rad. That sounds like a good thing and is the opposite.
// A policy trained in a simulator whose servos are three times stiffer than
// the hardware learns to rely on precision the robot does not have, and the
// failure only shows up on the real machine.
//
// SimplerEnv's tools/sysid does exactly this identification for its robots -
// stiffness and damping, by simulated annealing - and this follows the same
// idea with the same two parameters. The difference is that it needs no
// special dataset: the recordings already contain the pair that identifies a
// servo, the command that was sent (action) and the position that resulted
// (observation.state), 30 times a second for 29 seconds.
//
// What is being matched is the *distribution* of tracking error, not a
// trajectory. Nothing here needs the simulator to be right about anything
// else: it only asks the servos to be as sloppy as the real ones.

// MuJoCo's gainprm/biasprm have a per-actuator stride of MJNGAIN = 10, not 3.
const MJNGAIN = 10;

/** Read the model's position-actuator gains, per arm and joint. */
export function readActuatorGains(robot) {
  const out = {};
  for (const side of ["left", "right"]) {
    out[side] = [];
    for (let j = 0; j < 7; j++) {
      const actuator = robot.app.mjModel.actuator(`${side}_joint${j + 1}_ctrl`);
      const id = actuator.id;
      out[side].push({
        kp: robot.app.mjModel.actuator_gainprm[id * MJNGAIN],
        kv: -robot.app.mjModel.actuator_biasprm[id * MJNGAIN + 2],
        forcerange: [robot.app.mjModel.actuator_forcerange[id * 2], robot.app.mjModel.actuator_forcerange[id * 2 + 1]],
      });
      actuator.delete?.();
    }
  }
  return out;
}

/** Scale both gains by the same factor, keeping gainprm and biasprm consistent. */
export function applyActuatorScale(robot, base, scale) {
  for (const side of ["left", "right"]) {
    for (let j = 0; j < 7; j++) {
      const actuator = robot.app.mjModel.actuator(`${side}_joint${j + 1}_ctrl`);
      const id = actuator.id;
      const kp = base[side][j].kp * scale.kp;
      const kv = base[side][j].kv * scale.kv;
      robot.app.mjModel.actuator_gainprm[id * MJNGAIN] = kp;
      robot.app.mjModel.actuator_biasprm[id * MJNGAIN + 1] = -kp;
      robot.app.mjModel.actuator_biasprm[id * MJNGAIN + 2] = -kv;
      actuator.delete?.();
    }
  }
}

/**
 * Replay the recorded *commands* through the simulated servos and compare how
 * they respond with how the real ones did.
 *
 * Two earlier versions of this asked the wrong question and both had to go:
 *
 *   * RMS(command - state) inside each system and compare the two numbers. Two
 *     systems can share an RMS and differ completely in lag and in trajectory.
 *   * Drive the simulator with the recorded commands and compare its joint
 *     positions with the real robot's. This one is worse than it looks: the
 *     real joint sits at (command - the real robot's own tracking error), and
 *     the simulator cannot know that error, so the only way to shrink this
 *     score is to make the simulated servos infinitely stiff and let the
 *     simulator sit exactly on the command. The identification duly walked
 *     straight off the end of its search bracket, every round picking the
 *     stiffest setting on offer.
 *
 * What is compared now is the two *error signals*: the simulator's own
 * (command - position) against the real robot's (action - state), on the same
 * commands and the same clock. Zero means the two servos responded to the same
 * input the same way. Being too stiff and being too soft both move it away
 * from zero, and so does lag, because the comparison is of the signals rather
 * than of their magnitudes.
 *
 * The position difference is still measured and reported - it is the honest
 * answer to "given the same commands, where does the simulation end up" - but
 * it is not what the search minimises.
 */
export async function measureSimTracking(robot, episode, {
  settleFrames = 60, maxFrames = 0, keeper = null,
} = {}) {
  const { app, mujoco } = robot;
  const bottle = robot.bottle;
  const saved = {
    qpos: [...app.mjData.qpos.subarray(bottle.qpos, bottle.qpos + 7)],
  };
  const boxBody = robot.box.bodyId;
  const boxPos = [app.mjModel.body_pos[boxBody * 3], app.mjModel.body_pos[boxBody * 3 + 1], app.mjModel.body_pos[boxBody * 3 + 2]];
  // Clear the scene: a bottle in the arm's path shows up here as servo error,
  // and this measurement is meant to be about the servos.
  app.mjData.qpos[bottle.qpos] = 5;
  app.mjData.qpos[bottle.qpos + 1] = 5;
  app.mjData.qpos[bottle.qpos + 2] = -1;
  app.mjModel.body_pos[boxBody * 3] = 5;

  const realUpdate = app.update;
  app.update = (dt, now, xrFrame) => {
    app.simTarget = app.mjData.time;
    return realUpdate.call(app, 0, now, xrFrame);
  };

  const stepsPerFrame = Math.max(1, Math.round(1 / (episode.fps * app.mjModel.opt.timestep)));
  const positionGap = { left: Array(7).fill(0), right: Array(7).fill(0) };
  const trackingGap = { left: Array(7).fill(0), right: Array(7).fill(0) };
  let samples = 0;
  let worstFrame = null;

  // The lifter carries both arms and is part of the calibrated scene. It has to
  // be pinned here for the same reason the replay pins it: left at the cell's
  // default the arms sit 0.15 m too low, the fingers press into the tabletop
  // for the whole run, and every joint reads a tracking error of a third of a
  // radian - which is a collision with the furniture, not a servo. The first
  // run of this identification came back at 393 mm RMS and would have been read
  // as "the servos need to be much softer".
  const holdLifter = () => {
    const joint = robot.lifterJointId;
    if (joint === undefined || joint < 0 || !robot.lifterHeight) return;
    app.mjData.qpos[app.mjModel.jnt_qposadr[joint]] = robot.lifterHeight;
    app.mjData.qvel[app.mjModel.jnt_dofadr[joint]] = 0;
  };

  try {
    // Start from the recording's own first pose and let the arms arrive there,
    // otherwise the first second of travel lands in the measurement.
    for (let f = 0; f < 150; f++) {
      holdLifter();
      for (const side of ["left", "right"]) {
        const arm = app.controller.arms[side];
        const first = episode[`${side}_action_joints`][0];
        for (let j = 0; j < 7; j++) {
          app.mjData.qpos[arm.qposIds[j]] = first[j];
          app.mjData.qvel[arm.dofIds[j]] = 0;
          app.mjData.ctrl[arm.actIds[j]] = first[j];
        }
      }
      for (let k = 0; k < stepsPerFrame; k++) {
        app.controller.applyGravityComp(app.mjData);
        mujoco.mj_step(app.mjModel, app.mjData);
      }
    }
    const last = maxFrames > 0 ? Math.min(episode.frames, maxFrames) : episode.frames;
    for (let f = 0; f < last; f++) {
      holdLifter();
      for (const side of ["left", "right"]) {
        const arm = app.controller.arms[side];
        const commanded = episode[`${side}_action_joints`][f];
        for (let j = 0; j < 7; j++) app.mjData.ctrl[arm.actIds[j]] = commanded[j];
      }
      for (let k = 0; k < stepsPerFrame; k++) {
        app.controller.applyGravityComp(app.mjData);
        mujoco.mj_step(app.mjModel, app.mjData);
      }
      if (f < settleFrames) continue;
      samples += 1;
      for (const side of ["left", "right"]) {
        const arm = app.controller.arms[side];
        const measured = episode[`${side}_joints`][f];
        const commanded = episode[`${side}_action_joints`][f];
        for (let j = 0; j < 7; j++) {
          // Two different errors, and they are not interchangeable.
          //
          //   positionGap  where the simulated joint ended up relative to the
          //                real one - informative, but it cannot be driven to
          //                zero: the real joint sits at (command - the real
          //                robot's own error) and the simulator has no way to
          //                know that error. Making the servos infinitely stiff
          //                minimises *this* one, which is why the first run of
          //                this identification walked straight off the end of
          //                its search bracket.
          //   trackingGap  how the simulator's own command-to-position error
          //                differs from the real robot's. Zero means the two
          //                servos behaved the same way under the same commands.
          //                Too stiff, too soft and wrong lag are all penalised,
          //                and the minimum sits at the matched servo.
          //
          // Both are reported; only trackingGap is minimised.
          const actual = app.mjData.qpos[arm.qposIds[j]];
          const d = actual - measured[j];
          positionGap[side][j] += d * d;
          const eSim = app.mjData.ctrl[arm.actIds[j]] - actual;
          const eReal = commanded[j] - measured[j];
          const e = eSim - eReal;
          trackingGap[side][j] += e * e;
          if (!worstFrame || Math.abs(d) > Math.abs(worstFrame.d)) {
            worstFrame = { side, joint: j, frame: f, d };
          }
        }
      }
      if (f % 120 === 0) await new Promise((r) => setTimeout(r, 0));
    }
  } finally {
    app.update = realUpdate;
    app.mjData.qpos.set(saved.qpos, bottle.qpos);
    app.mjModel.body_pos[boxBody * 3] = boxPos[0];
    mujoco.mj_forward(app.mjModel, app.mjData);
  }
  const sqrt = (a) => a.map((sum) => Math.sqrt(sum / Math.max(1, samples)));
  return {
    rms: { left: sqrt(trackingGap.left), right: sqrt(trackingGap.right) },
    position: { left: sqrt(positionGap.left), right: sqrt(positionGap.right) },
    samples,
    worstFrame,
  };
}

/** The real robot's own tracking error, for reference alongside the above. */
export function realTrackingError(episode) {
  const out = {};
  for (const side of ["left", "right"]) {
    const commanded = episode[`${side}_action_joints`];
    const measured = episode[`${side}_joints`];
    out[side] = Array.from({ length: 7 }, (_, j) => {
      let sum = 0;
      for (let f = 0; f < measured.length; f++) {
        const d = commanded[f][j] - measured[f][j];
        sum += d * d;
      }
      return Math.sqrt(sum / measured.length);
    });
  }
  return out;
}

/**
 * Score one gain setting, on the tracking-error signal.
 *
 * Zero is perfect: under the same commands the simulated servo and the real
 * one were wrong by the same amount in the same direction at the same instant.
 * A servo that is too stiff is wrong in a different way, not in a smaller way,
 * and this score says so.
 */
function score(simulated) {
  let sum = 0;
  let n = 0;
  let worst = 0;
  let worstAt = null;
  for (const side of ["left", "right"]) {
    for (let j = 0; j < 7; j++) {
      const value = simulated[side][j];
      sum += value * value;
      n += 1;
      if (value > worst) {
        worst = value;
        worstAt = { side, joint: j + 1 };
      }
    }
  }
  return { rms: Math.sqrt(sum / n), worst, worstAt };
}

/**
 * Find the servo gains whose simulated servo *behaves* most like the real one.
 *
 * A coarse grid over (kp, kv), then two rounds of refinement around the winner
 * with the bracket tightened threefold each time - the same "reduce the search
 * range" step SimplerEnv's procedure asks for between rounds.
 *
 * What the winning scale means. It is not a claim that the real controller is
 * stiffer or softer than the deployment logs say - those gains are applied as
 * the starting point, and the scale multiplies them. It is the factor that has
 * to be put on MuJoCo's idealised position actuator to make it lag the way the
 * real one lags, which also has to cover everything the simulator does not
 * model: bus latency, joint compliance, gearbox friction. So a scale above 1
 * is expected rather than alarming, and the number to look at is not the scale
 * but the tracking gap that is left when it is applied.
 *
 * `episodes` may hold more than one recording; the score is averaged over them,
 * because gains tuned on a single run can fit that run's particular speed and
 * load and nothing else.
 */
export async function identifyServos(robot, episodes, {
  kpGrid = [0.4, 0.7, 1.0, 1.5, 2.2, 3.2],
  kvGrid = [0.4, 0.7, 1.0, 1.5, 2.2],
  refineRounds = 3,
  maxFrames = 480,
  onProgress = null,
} = {}) {
  const list = Array.isArray(episodes) ? episodes : [episodes];
  const base = readActuatorGains(robot);
  const evaluated = [];
  let best = null;

  const evaluate = async (kp, kv) => {
    applyActuatorScale(robot, base, { kp, kv });
    const perEpisode = [];
    for (const episode of list) {
      const result = await measureSimTracking(robot, episode, { maxFrames });
      perEpisode.push(result);
    }
    const scores = perEpisode.map((r) => score(r.rms));
    const entry = {
      kp, kv,
      rms: scores.reduce((a, s) => a + s.rms, 0) / scores.length,
      worst: Math.max(...scores.map((s) => s.worst)),
      measured: perEpisode.map((r) => r.rms),
      // where the simulated joints landed relative to the real ones - reported,
      // not minimised (see measureSimTracking)
      position: perEpisode.map((r) => r.position),
      samples: perEpisode.map((r) => r.samples),
    };
    evaluated.push(entry);
    if (!best || entry.rms < best.rms) best = entry;
    onProgress?.(entry, evaluated.length);
    return entry;
  };

  for (const kp of kpGrid) {
    for (const kv of kvGrid) await evaluate(kp, kv);
  }

  let span = 0.3;
  for (let round = 0; round < refineRounds; round++) {
    const centre = { kp: best.kp, kv: best.kv };
    for (const dk of [-span, 0, span]) {
      for (const dv of [-span, 0, span]) {
        if (dk === 0 && dv === 0) continue;
        await evaluate(Math.max(0.02, centre.kp + dk), Math.max(0.02, centre.kv + dv));
      }
    }
    span /= 3;
  }

  applyActuatorScale(robot, base, { kp: best.kp, kv: best.kv });
  return { best, evaluated, base, episodes: list.length };
}
