// The tool set the agent is allowed to use, in the shape Pigey gives it: a
// heterogeneous box of primitives, each wrapping either a deterministic motion
// or a measurement. The planner (LLM or local) only picks from this list; it
// never emits joint commands, and it never decides whether a stage passed.
//
// Two rules keep the board honest:
//
//   * A stage is ticked by the tool that measured it, never by the planner's
//     prose. Every tool returns the numbers it measured alongside ok/failed.
//   * Nothing writes an object's pose. The bottle is a free body for the whole
//     run; the grasp is confirmed by lifting it and watching whether the bottle
//     comes with it, and the handover is confirmed by letting go with one hand
//     and watching whether the other one was really holding on.

import * as perception from "./perception.js";

export const STAGES = [
  "初始场景",
  "定位瓶子",
  "接近瓶子",
  "夹取验证",
  "双手交接",
  "放入白盒",
  "落点验证",
];

// Grasp geometry. The bottle stands up, so a grasp is described as a height up
// its body rather than a point along it: the right hand takes it low and the
// left hand takes it higher, and the band has to be wide enough that two
// grippers fit on the body at once.
//
//   padReach   wrist to the middle of the pads, measured from the model
//   rightBand  height above the bottle's base where the right hand grips
//   leftBand   the same for the left hand, above the right hand's band
export const GEOMETRY = {
  padReach: 0.114,
  rightBand: 0.046,
  leftBand: 0.146,
  hoverMargin: 0.055,
  liftHeight: 0.065,
  carryClearance: 0.120,
  releaseClearance: 0.060,
  gripClose: 1.0,
  tolerance: 0.008,
};

export class ToolError extends Error {}

const round = (value, digits = 3) => Number(value.toFixed(digits));
const metres = (p) => `(${p.map((v) => v.toFixed(3)).join(", ")}) m`;

export function createTools({ robot, app, calib, onImage }) {
  const tableZ = calib.table.top_z.value;
  const boxXY = calib.objects.box.pose.value;
  const boxHalf = [
    calib.objects.box.inner_size.value[0] / 2,
    calib.objects.box.inner_size.value[1] / 2,
  ];

  // How the hand is turned when it takes hold of the bottle. A straight-down
  // wrist is not reachable at the pick point - the solver lands 21.6 deg short
  // of it, which is what made the first collision grasp close on empty air -
  // so the orientation is one that was measured to be reachable, and the tool
  // checks the achieved rotation again before it closes.
  const orientation = {
    right: normaliseQuat(calib.grasp.orientation.value),
    left: normaliseQuat(calib.grasp.orientation_left?.value ?? calib.grasp.orientation.value),
  };

  /** Unit vector from the wrist towards the object, for a given hand rotation. */
  function approachOf(q) {
    const [w, x, y, z] = q;
    return [
      -(2 * (x * z + y * w)),
      -(2 * (y * z - x * w)),
      -(1 - 2 * (x * x + y * y)),
    ];
  }

  /** Where the wrist has to be for the middle of the pads to reach `point`. */
  function wristFor(point, side) {
    const a = approachOf(orientation[side]);
    return point.map((value, i) => value - GEOMETRY.padReach * a[i]);
  }

  /**
   * How far above a grip band the hand has to hover so its open fingers clear
   * the top of the bottle.
   *
   * The hand comes in at an angle, so a fixed clearance puts the pads beside
   * the bottle rather than above it: at 90 mm the pads sat 72 mm below the cap,
   * inside the bottle's column, and the arm stalled 27 mm short of a target it
   * was physically resting against. The clearance has to be the height of the
   * bottle above the band, plus a margin.
   */
  function hoverClearance(band) {
    return (calib.objects.bottle.height.value - band) + GEOMETRY.hoverMargin;
  }

  /** Wrist height at which the pads clear the top of the bottle. */
  function cruiseWristZ(side) {
    const a = approachOf(orientation[side]);
    return tableZ + calib.objects.bottle.height.value - GEOMETRY.padReach * a[2] + GEOMETRY.hoverMargin;
  }

  const perceptionCamera = app.camera.clone();
  perception.aimCamera(perceptionCamera, {
    intrinsics: calib.perception_camera.intrinsics,
    pose: perception.lookAtPose(calib.perception_camera.pose),
  });
  const perceptionCameraB = app.camera.clone();
  perception.aimCamera(perceptionCameraB, {
    intrinsics: calib.perception_camera_b.intrinsics,
    pose: perception.lookAtPose(calib.perception_camera_b.pose),
  });

  // Latest measurements, so a later tool can verify against what an earlier
  // one actually saw rather than against what it intended.
  const state = { detection: null, held: { left: false, right: false } };

  function frame() {
    return perception.sample(app, perceptionCamera);
  }

  /**
   * A frame with the robot's own body masked out, used only for detection.
   * The operators still see the normal frame in the keyframes.
   */
  function detectionFrame(camera = perceptionCamera) {
    // Self-masking is a calibration choice, not a hard-coded behaviour: the
    // real system can do it too (it knows its own geometry), but it changes
    // what the agent's observation looks like, so it is recorded in
    // calibrate.json -> perception_camera.agent_observation.self_mask_robot.
    const masked = calib.perception_camera?.agent_observation?.self_mask_robot ?? true;
    return perception.sample(app, camera, { hideRobot: masked });
  }

  function image(shot, note) {
    const url = perception.frameToDataUrl(shot);
    onImage?.(url, note);
    return url;
  }

  /** Where on the table objects are allowed to be, as image-space bounds. */
  function objectWindow(shot, camera = perceptionCamera, region = calib.object_region) {
    return perception.regionWindow(
      camera, region, tableZ,
      { width: shot.width, height: shot.height },
    );
  }

  function findBottle(shot, camera = perceptionCamera, region) {
    return perception.darkestBlob(shot, {
      threshold: 0.18,
      minPixels: 120,
      window: objectWindow(shot, camera, region),
    });
  }

  /**
   * The bottle's axis in world coordinates, from both cameras.
   *
   * A single view gives a ray, not a point, and the old code made up the
   * missing depth by intersecting that ray with a known plane. That only works
   * for something lying flat; a standing bottle's blob sits at mid-height, so
   * the plane assumption put the estimate tens of millimetres out - more than
   * the clearance the open gripper has.
   *
   * The two cameras are not equally good at both axes, so each is used for
   * what it is good at: A looks almost straight down, which pins the bottle's
   * x to its image column, while its row - the depth - is the weak one. B
   * looks in from the side, where that depth is well conditioned. The leftover
   * disagreement between the two is reported rather than averaged away, so a
   * bad correspondence shows up as a number instead of a confident wrong point.
   */
  /**
   * The bottle's axis in the world, from both cameras.
   *
   * Not the blob's centroid: the centroid sits on the visible shell, not on
   * the axis, and its offset depends on which way the camera is looking. For a
   * cylinder standing on a known plane there is an exact construction instead -
   * the two silhouette edges are the two tangent lines, and where they meet the
   * table plane they are exactly one radius either side of the axis, whatever
   * the view. Their midpoint is the axis, so this does not depend on the
   * bottle's radius or on the camera angle.
   *
   * Two of them are averaged, and the spread between the two is reported: a
   * bad detection disagrees, and the caller is told by how much.
   */
  function bottleAxis(a, b) {
    const axisA = silhouetteAxis(perceptionCamera, a.blob, a.size);
    const axisB = silhouetteAxis(perceptionCameraB, b.blob, b.size);
    if (!axisA || !axisB) throw new ToolError("这一帧里轮廓边缘落在台面外，无法定位。");

    // The construction is only as good as the assumption behind it, and the
    // assumption is the view: the tangent lines land one radius either side of
    // the axis only while the camera looks steeply down. Measured, the
    // near-vertical camera lands 3 mm out and the shallow one 25 mm, because
    // from a shallow angle the bottle's shadow merges into its silhouette. So
    // the steeper camera decides, and the other one is kept as a cross-check.
    const verticalA = Math.abs(perception.rayFromPixel(perceptionCamera, a.blob.centroid, a.size).direction[2]);
    const verticalB = Math.abs(perception.rayFromPixel(perceptionCameraB, b.blob.centroid, b.size).direction[2]);
    const primary = verticalA >= verticalB ? axisA : axisB;
    const check = verticalA >= verticalB ? axisB : axisA;
    return {
      point: [primary[0], primary[1], tableZ + calib.objects.bottle.height.value / 2],
      residual: Math.hypot(primary[0] - check[0], primary[1] - check[1]),
      per_camera: { a: axisA, b: axisB },
      primary: verticalA >= verticalB ? "a" : "b",
      view_verticality: { a: round(verticalA, 3), b: round(verticalB, 3) },
    };
  }

  function silhouetteAxis(camera, blob, size) {
    const row = blob.centroid[1];
    const left = perception.unprojectToPlane(camera, [blob.bbox.x, row], tableZ, size);
    const right = perception.unprojectToPlane(
      camera, [blob.bbox.x + blob.bbox.width - 1, row], tableZ, size,
    );
    if (!left || !right) return null;
    return [(left[0] + right[0]) / 2, (left[1] + right[1]) / 2];
  }

  // The box is a declared fixture, so its interior is a region too: the bottle
  // has to be found there after the drop, not back in the pick zone.
  const boxRegion = {
    x: [boxXY[0] - boxHalf[0], boxXY[0] + boxHalf[0]],
    y: [boxXY[1] - boxHalf[1], boxXY[1] + boxHalf[1]],
  };

  /** Where on the bottle one hand should take hold, for a given grip band. */
  function gripPoint(centre, band) {
    return [centre[0], centre[1], tableZ + band];
  }

  const tools = {
    reset: {
      stage: 0,
      describe: "把场景、双臂和物体恢复到初始状态，并清空上一轮的所有判定。",
      alias: "GoHome（Pigey 里是 GoHome + 场景重置）",
      run: async () => {
        await robot.reset();
        state.detection = null;
        state.held = { left: false, right: false };
        await robot.simulate(0.4);
        const bottle = robot.bottlePose().pos;
        return {
          ok: true,
          stage: 0,
          summary: `场景已复位，瓶子躺在 ${metres(bottle)}。`,
          data: {
            bottle: bottle.map((v) => round(v)),
            table_z: tableZ,
            note: "瓶子是自由刚体，复位只写它的起始 qpos，之后完全由物理决定。",
          },
        };
      },
    },

    observe: {
      stage: null,
      describe: "取一帧双目画面和机器人本体状态。看不到物体的真实坐标。",
      alias: "Perceive 的场景部分",
      run: async () => {
        const shotA = perception.sample(app, perceptionCamera, { hideRobot: true });
        const shotB = perception.sample(app, perceptionCameraB, { hideRobot: true });
        const url = perception.stereoToDataUrl(shotA, shotB, "感知相机 A · 俯视", "感知相机 B · 侧视");
        onImage?.(url, "观察（双目）");
        const pose = (side) => robot.tcp(side).pos.map((v) => round(v, 4));
        return {
          ok: true,
          summary: "已取到双目画面与末端位姿。",
          data: {
            image: url,
            cameras: [calib.perception_camera.name, calib.perception_camera_b.name],
            time: round(robot.time(), 2),
            tcp: { left: pose("left"), right: pose("right") },
            grip: { left: round(robot.grip("left"), 2), right: round(robot.grip("right"), 2) },
            contact: contactSummary(robot),
          },
        };
      },
    },

    perceive: {
      stage: 1,
      describe: "两路相机各找一次深色物体，用两条射线交会算出瓶轴的三维位置。",
      alias: "Perceive",
      run: async () => {
        const shot = frame();
        const maskA = detectionFrame(perceptionCamera);
        const maskB = detectionFrame(perceptionCameraB);
        const sizeA = { width: maskA.width, height: maskA.height };
        const sizeB = { width: maskB.width, height: maskB.height };
        const blobA = findBottle(maskA, perceptionCamera);
        const blobB = findBottle(maskB, perceptionCameraB);
        const url = image(shot, "感知");
        if (!blobA && !blobB) throw new ToolError("两路相机里都没有找到足够大的深色区域，无法定位瓶子。");

        let world;
        let how;
        let residual = null;
        if (blobA && blobB) {
          const hit = bottleAxis({ blob: blobA, size: sizeA }, { blob: blobB, size: sizeB });
          world = hit.point;
          residual = hit.residual;
          how = `轮廓边缘解算（主：相机 ${hit.primary.toUpperCase()}，垂直度 ${hit.view_verticality.a} / ${hit.view_verticality.b}）`;
          if (residual > 0.040) {
            throw new ToolError(
              `两路相机量出来的瓶轴差了 ${(residual * 1000).toFixed(0)} mm，超过 40 mm：`
              + "说明两幅画面里框住的不是同一个物体（多半有一路被挡、或被工位区域切掉了）。",
            );
          }
        } else {
          // One camera still gives a usable answer for an object whose height
          // is known, but the caller is told that is what happened.
          const blob = blobA ?? blobB;
          const camera = blobA ? perceptionCamera : perceptionCameraB;
          const size = blobA ? sizeA : sizeB;
          const onAxis = perception.unprojectToPlane(
            camera, blob.centroid, tableZ + calib.objects.bottle.height.value / 2, size);
          if (!onAxis) throw new ToolError("射线与瓶身高度平面没有交点。");
          world = onAxis;
          how = `${blobA ? "相机 A" : "相机 B"} 单视角 + 已知瓶高假设`;
        }

        state.detection = { id: "#1", world, pixels: (blobA?.pixels ?? 0) + (blobB?.pixels ?? 0) };
        return {
          ok: true,
          stage: 1,
          summary: `找到 #1：${how}，瓶轴 ${metres(world.map((v) => round(v)))}`
            + (residual === null ? "。" : `，两路相机互差 ${(residual * 1000).toFixed(1)} mm。`),
          data: {
            image: url,
            detection: {
              id: "#1",
              method: how,
              pixels: { a: blobA?.pixels ?? 0, b: blobB?.pixels ?? 0 },
              bbox: { a: blobA?.bbox ?? null, b: blobB?.bbox ?? null },
              world: world.map((v) => round(v)),
              camera_disagreement_mm: residual === null ? "未测得（只用了一路）" : round(residual * 1000, 1),
            },
            fixture: {
              note: "白盒是工装，位置来自 calibrate.json，不是视觉检测",
              xy: boxXY.slice(0, 2),
              inner: calib.objects.box.inner_size.value,
            },
          },
        };
      },
    },

    reach: {
      stage: 2,
      describe: "把一只手的末端移到抓取点正上方，先不接触任何东西。",
      alias: "Pick 的接近段",
      run: async ({ side, target, band = null }) => {
        if (!state.detection) throw new ToolError("还没有定位到瓶子，请先调用 perceive。");
        const holdBand = band ?? defaultBand(side);
        const point = gripPoint(resolveTarget(target, state), holdBand);
        const wrist = wristFor(point, side);
        const hover = [wrist[0], wrist[1], wrist[2] + hoverClearance(holdBand)];
        const result = await robot.approachFromAbove(side, hover, orientation[side], {
          safeZ: cruiseWristZ(side), seconds: 1.8,
        });
        return {
          ok: true,
          stage: 2,
          summary: `${side} 手到达预抓取位 ${metres(hover.map((v) => round(v)))}，`
            + `位置误差 ${(result.error * 1000).toFixed(1)} mm，姿态误差 ${((result.rotation * 180) / Math.PI).toFixed(1)}°。`,
          data: {
            side,
            hover: hover.map((v) => round(v)),
            error_mm: round(result.error * 1000, 1),
            rotation_deg: round((result.rotation * 180) / Math.PI, 2),
          },
        };
      },
    },

    grasp: {
      stage: 3,
      describe: "下降到抓取高度、把指垫对到瓶轴上、闭爪，再抬起来看瓶子是否跟着手走。",
      alias: "Pick",
      run: async ({ side, target, band = null }) => {
        if (!state.detection) throw new ToolError("还没有定位到瓶子，请先调用 perceive。");
        const holdBand = band ?? defaultBand(side);
        const point = gripPoint(resolveTarget(target, state), holdBand);
        const wrist = wristFor(point, side);

        robot.setGrip(side, 0);
        await robot.approachFromAbove(
          side, [wrist[0], wrist[1], wrist[2] + hoverClearance(holdBand)], orientation[side],
          { safeZ: cruiseWristZ(side), seconds: 1.5 },
        );
        // Close the loop on where the pads really are, not on a nominal offset.
        // Two stages on purpose: the hand comes down at 60 degrees, so its
        // open fingers travel sideways as they descend and would sweep through
        // the bottle's upper body on the way. Line up over it first, then drop
        // straight down.
        await robot.alignPads(side, point, orientation[side], { axisMask: [1, 1, 0], tolerance: 0.005 });
        const aligned = await robot.alignPads(side, point, orientation[side]);
        const contactAt = robot.padCenter(side);
        const before = robot.bottlePose().pos;

        const close = await robot.closeUntilStall(side);
        const contact = contactSummary(robot);
        // The fingers can only fall short of their command if something is
        // between them. If they reached it, they closed on air.
        if (!close.stalled) {
          state.held[side] = false;
          throw new ToolError(
            `夹爪合到了指令位置都没有被挡住，说明两指之间没有东西：`
            + `指关节 ${close.angle.toFixed(3)} rad / 指令 ${close.commanded.toFixed(3)} rad，`
            + `张开度约 ${((close.width ?? 0) * 1000).toFixed(0)} mm；`
            + `指垫停在 ${metres(contactAt.map((v) => round(v)))}。`,
          );
        }

        const gripWrist = robot.tcp(side).pos;
        const lifted = [gripWrist[0], gripWrist[1], gripWrist[2] + GEOMETRY.liftHeight];
        await robot.moveTcp(side, lifted, {
          seconds: 1.4, quat: orientation[side], tolerance: GEOMETRY.tolerance,
        });
        await robot.simulate(0.6);

        const after = robot.bottlePose().pos;
        const rise = after[2] - before[2];
        // Compare against where the wrist actually went, not where it was told
        // to go: the actuators track to about a centimetre, and charging that
        // to the grasp made a good hold look like a 25 mm slip.
        const liftedWrist = robot.tcp(side).pos;
        const expected = before.map((v, i) => v + (liftedWrist[i] - gripWrist[i]));
        const drift = Math.hypot(after[0] - expected[0], after[1] - expected[1], after[2] - expected[2]);
        const grip = robot.gripForce(side);

        if (rise < 0.030 || drift > 0.030) {
          state.held[side] = false;
          throw new ToolError(
            `闭爪后瓶子没有跟着走：抬升 ${(rise * 1000).toFixed(0)} mm，`
            + `跟随偏差 ${(drift * 1000).toFixed(0)} mm，指垫落在 ${metres(contactAt.map((v) => round(v)))}，`
            + `接触 ${describeContact(contact)}。`,
          );
        }
        // Squeeze the last of the travel after the stall: the stall is where
        // the fingers met the bottle, not the firmest hold they can take, and
        // the bottle later slid out during a fast carry at stall pressure.
        robot.setGrip(side, GEOMETRY.gripClose);
        await robot.simulate(0.4);

        state.held[side] = true;
        return {
          ok: true,
          stage: 3,
          summary: `${side} 手夹住瓶子：抬升 ${(rise * 1000).toFixed(0)} mm，跟随偏差 ${(drift * 1000).toFixed(0)} mm。`,
          data: {
            side,
            pad_point: contactAt.map((v) => round(v, 4)),
            pad_alignment_mm: round(aligned.error * 1000, 1),
            grip_width_mm: round((close.width ?? 0) * 1000, 1),
            finger_angle_rad: round(close.angle, 4),
            finger_blocked_by_rad: round(close.angle - close.commanded, 4),
            rise_mm: round(rise * 1000, 1),
            drift_mm: round(drift * 1000, 1),
            bottle: after.map((v) => round(v, 4)),
            grip_actuator_force: grip === null ? "未测得" : round(grip, 3),
            bottle_contact: contact,
          },
        };
      },
    },

    verify_grasp: {
      stage: null,
      describe: "再抬一次，确认瓶子仍然跟着手，相对手没有滑移。",
      alias: "Pick 的验证段（Pigey 用夹爪传感器+腕部相机确认抓取）",
      run: async ({ side }) => {
        const from = robot.tcp(side).pos;
        const before = robot.bottlePose().pos;
        await robot.moveTcp(side, [from[0], from[1], from[2] + 0.040],
          { seconds: 1.0, tolerance: 0.020 });
        await robot.simulate(0.5);
        const after = robot.bottlePose().pos;
        const wrist = robot.tcp(side).pos;
        const observed = [
          before[0] + (wrist[0] - from[0]),
          before[1] + (wrist[1] - from[1]),
          before[2] + (wrist[2] - from[2]),
        ];
        const drift = Math.hypot(after[0] - observed[0], after[1] - observed[1], after[2] - observed[2]);
        if (drift > 0.020) {
          state.held[side] = false;
          throw new ToolError(`${side} 手在继续抬升时瓶子滑移了 ${(drift * 1000).toFixed(0)} mm。`);
        }
        return {
          ok: true,
          summary: `${side} 手握持稳定，继续抬升时滑移 ${(drift * 1000).toFixed(0)} mm。`,
          data: { side, drift_mm: round(drift * 1000, 1), bottle: after.map((v) => round(v, 4)) },
        };
      },
    },

    handover: {
      stage: 4,
      describe: "左手张开等着不动，右手把瓶子送到左手两指之间；左手合上确实夹住之后，右手才松开。",
      run: async () => {
        if (!state.held.right) throw new ToolError("右手没有握住瓶子，不能交接。");
        const q = orientation;
        const [handoverX, handoverY] = calib.grasp.handover_xy.value;
        const carryPadZ = tableZ + GEOMETRY.carryClearance + GEOMETRY.leftBand;

        // The left hand parks, open, and does not move again. Asking it to
        // servo onto a bottle that someone else is holding never worked: its
        // soft wrist drifts about 20 mm under the closing fingers, which is
        // more than the bottle's clearance. Moving the bottle into a hand that
        // is standing still uses the right arm, which has already shown it can
        // hold a pose to 13 mm.
        robot.setGrip("left", 0);
        const holdWrist = wristFor([handoverX, handoverY, carryPadZ], "left");
        await robot.approachFromAbove("left", holdWrist, q.left, { safeZ: cruiseWristZ("left"), seconds: 1.8 });
        const leftPads = robot.padCenter("left");
        if (!leftPads) throw new ToolError("读不到左手指垫位置。");

        // Carry the bottle over, then slide it sideways into the open jaws by
        // the measured difference between the two hands. The right hand's pads
        // are on the bottle, the left hand's pads are where the bottle has to
        // end up, so the move is just the difference between the two.
        const centre = resolveTarget("#1", state);
        const rightNow = robot.tcp("right").pos;
        const lift = wristFor([centre[0], centre[1], tableZ + GEOMETRY.carryClearance + GEOMETRY.rightBand], "right");
        await robot.moveTcp("right", [rightNow[0], rightNow[1], lift[2]],
          { seconds: 2.0, quat: q.right, tolerance: 0.020 });
        await robot.moveTcp("right", lift, { seconds: 2.4, quat: q.right, tolerance: 0.020 });
        await robot.simulate(0.4);

        const handles = [];
        let rightPads = robot.padCenter("right");
        for (let i = 0; i < 4; i++) {
          const base = rightPads[2] - GEOMETRY.rightBand;
          const shift = [
            leftPads[0] - rightPads[0],
            leftPads[1] - rightPads[1],
            (leftPads[2] - GEOMETRY.leftBand) - base,
          ];
          handles.push(shift.map((v) => round(v * 1000, 1)));
          if (Math.hypot(...shift) < 0.008) break;
          const wrist = robot.tcp("right").pos;
          // Loose tolerance: this arm tracks to about 5 mm, and the loop's
          // own measured pad difference is what decides convergence.
          await robot.moveTcp("right", wrist.map((v, k) => v + shift[k]),
            { seconds: 1.6, quat: q.right, tolerance: 0.015 });
          await robot.simulate(0.3);
          rightPads = robot.padCenter("right");
        }
        const placement = Math.hypot(
          leftPads[0] - rightPads[0], leftPads[1] - rightPads[1],
          (leftPads[2] - GEOMETRY.leftBand) - (rightPads[2] - GEOMETRY.rightBand),
        );

        const leftClose = await robot.closeUntilStall("left");
        const leftContact = contactSummary(robot);
        if (!leftClose.stalled) {
          throw new ToolError(
            `瓶子送到位了但左手没夹住：合到指令位置都没被挡住，`
            + `指关节 ${leftClose.angle.toFixed(3)} rad / 指令 ${leftClose.commanded.toFixed(3)} rad。右手没有松开。`,
          );
        }
        const beforeRelease = robot.bottlePose().pos;
        const rightTcp = robot.tcp("right").pos;

        // Only now does the right hand let go, and the bottle has to stay put.
        robot.setGrip("right", 0);
        await robot.simulate(0.6);
        const afterRelease = robot.bottlePose().pos;
        const sag = Math.hypot(...afterRelease.map((v, i) => v - beforeRelease[i]));
        if (sag > 0.025) {
          state.held.left = false;
          state.held.right = false;
          throw new ToolError(
            `右手松开后瓶子没有被左手接住：下沉 ${(sag * 1000).toFixed(0)} mm，`
            + `左手接触 ${describeContact(leftContact)}。`,
          );
        }
        state.held.left = true;
        state.held.right = false;

        await robot.moveTcp("right", [rightTcp[0], rightTcp[1] - 0.20, rightTcp[2] + 0.10],
          { seconds: 1.4, quat: q.right, tolerance: 0.025 });
        await robot.simulate(0.4);
        const carried = robot.bottlePose().pos;
        const leftDrift = Math.hypot(...carried.map((v, i) => v - afterRelease[i]));
        if (leftDrift > 0.020) {
          state.held.left = false;
          throw new ToolError(`左手在右手撤离时没有拿稳，瓶子偏了 ${(leftDrift * 1000).toFixed(0)} mm。`);
        }

        return {
          ok: true,
          stage: 4,
          summary: `交接完成：右手把瓶子送进左手（对位偏差 ${(placement * 1000).toFixed(0)} mm），`
            + `左手合上后右手才松开，撤离时瓶子只偏 ${(leftDrift * 1000).toFixed(0)} mm，现在 ${metres(carried.map((v) => round(v)))}。`,
          data: {
            grip_bands_from_base: { right: GEOMETRY.rightBand, left: GEOMETRY.leftBand },
            left_hold_point: leftPads.map((v) => round(v, 4)),
            right_pads_after_placing: rightPads.map((v) => round(v, 4)),
            placement_error_mm: round(placement * 1000, 1),
            placement_steps_mm: handles,
            left_grip_width_mm: round((leftClose.width ?? 0) * 1000, 1),
            release_sag_mm: round(sag * 1000, 1),
            withdraw_drift_mm: round(leftDrift * 1000, 1),
            held_by: "left",
            bottle: carried.map((v) => round(v, 4)),
            bottle_contact: leftContact,
          },
        };
      },
    },

    place: {
      stage: 5,
      describe: "左手把瓶子移到白盒上方、降到盒内，松开并撤走。",
      alias: "Place",
      run: async ({ side = "left" }) => {
        if (!state.held[side]) throw new ToolError(`${side} 手没有握住瓶子，不能放置。`);
        const above = wristFor(
          [boxXY[0], boxXY[1], tableZ + GEOMETRY.carryClearance + GEOMETRY.leftBand], side);
        await robot.moveTcp(side, above,
          { seconds: 1.9, quat: orientation[side], tolerance: 0.020 });
        const releasePoint = [boxXY[0], boxXY[1], tableZ + GEOMETRY.releaseClearance + GEOMETRY.leftBand];
        await robot.alignPads(side, releasePoint, orientation[side]);
        await robot.simulate(0.3);

        robot.setGrip(side, 0);
        await robot.simulate(0.5);
        const retract = wristFor([boxXY[0], boxXY[1], tableZ + 0.30 + GEOMETRY.leftBand], side);
        await robot.moveTcp(side, retract,
          { seconds: 1.3, quat: orientation[side], tolerance: 0.025 });
        await robot.simulate(1.6);
        const landed = robot.bottlePose().pos;
        state.held[side] = false;
        return {
          ok: true,
          stage: 5,
          summary: `已松爪，瓶子落到 ${metres(landed.map((v) => round(v)))}。`,
          data: {
            side,
            released_at: releasePoint.map((v) => round(v, 4)),
            bottle: landed.map((v) => round(v, 4)),
          },
        };
      },
    },

    verify_goal: {
      stage: 6,
      describe: "从相机画面确认瓶子落在白盒范围内，并且稳稳停在盒底之上。",
      alias: "Done",
      run: async () => {
        const shot = frame();
        const mask = detectionFrame(perceptionCamera);
        const url = image(shot, "验证落点");
        const blob = findBottle(mask, perceptionCamera, boxRegion);
        const landed = robot.bottlePose().pos;
        const insideBox = Math.abs(landed[0] - boxXY[0]) < boxHalf[0]
          && Math.abs(landed[1] - boxXY[1]) < boxHalf[1];
        const aboveFloor = landed[2] > tableZ && landed[2] < tableZ + 0.25;

        // Seeing it in the box is the second opinion; the box's floor decides
        // the height, so the plane assumption is valid here even lying down.
        const seen = blob
          ? perception.unprojectToPlane(perceptionCamera, blob.centroid, tableZ + 0.02,
            { width: mask.width, height: mask.height })
          : null;
        const seenInside = seen
          ? Math.abs(seen[0] - boxXY[0]) < boxHalf[0] + 0.04 && Math.abs(seen[1] - boxXY[1]) < boxHalf[1] + 0.04
          : false;

        if (!insideBox || !aboveFloor) {
          throw new ToolError(
            `瓶子不在盒内：位置 ${metres(landed.map((v) => round(v)))}，盒心 ${metres(boxXY)}。`,
          );
        }
        return {
          ok: true,
          stage: 6,
          summary: `瓶子位于白盒内，落点 ${metres(landed.map((v) => round(v)))}；相机里${seenInside ? "也" : "未"}能再次看到它。`,
          data: {
            image: url,
            bottle: landed.map((v) => round(v, 4)),
            box_center: boxXY,
            box_inner: calib.objects.box.inner_size.value,
            camera_estimate: seen ? seen.map((v) => round(v)) : null,
            camera_agrees: seenInside,
          },
        };
      },
    },

    home: {
      stage: null,
      describe: "把两只手收回初始停放位置。",
      alias: "GoHome",
      run: async () => {
        robot.setGrip("left", 0);
        robot.setGrip("right", 0);
        robot.stow();
        await robot.simulate(0.8);
        return { ok: true, summary: "双臂已收回停放位置。", data: {} };
      },
    },
  };

  return {
    tools,
    state,
    cameras: { a: perceptionCamera, b: perceptionCameraB },
    geometry: { ...GEOMETRY },
    stages: STAGES,
    /** A single-line catalogue the planner is given, derived from the registry. */
    catalogue() {
      return Object.entries(tools).map(([name, tool]) => ({
        name, describe: tool.describe, alias: tool.alias,
      }));
    },
  };
}

function defaultBand(side) {
  return side === "left" ? GEOMETRY.leftBand : GEOMETRY.rightBand;
}

function normaliseQuat(q) {
  const length = Math.hypot(...q) || 1;
  return q.map((value) => value / length);
}

function contactSummary(robot) {
  const contact = robot.bottleContactForce();
  if (contact === null) return null;
  return { count: contact.count, total_n: round(contact.total, 2) };
}

function describeContact(contact) {
  if (!contact) return "未测得（这个 WASM 构建没有暴露接触力）";
  return `${contact.count} 点 ${contact.total_n} N`;
}

function resolveTarget(target, state) {
  if (Array.isArray(target) && target.length === 3) return target;
  if (target === "#1" && state.detection) return state.detection.world;
  throw new ToolError(`无法解析目标 ${JSON.stringify(target)}；请先调用 perceive。`);
}
