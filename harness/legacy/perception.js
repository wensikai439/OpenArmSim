// Perception: what the harness is allowed to know, and where it came from.
//
// The agent never reads an object's pose out of mjData. It gets pixels from a
// camera described by intrinsics and extrinsics, finds the dark bottle in
// those pixels, and unprojects the measurement onto the table plane. That
// keeps the harness honest in the same way the real pipeline is: if the camera
// is mis-calibrated, or the bottle is out of view, perception reports a miss
// instead of a perfect answer.
//
// The white box is a fixture, not a detection. A white box on a white table is
// not separable by colour, so its pose comes from the calibration file, and
// every place that uses it says so.

const LUMA = (r, g, b) => (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;

/** Column-major 4x4 multiply, three.js Matrix4 layout. */
function multiply(a, b) {
  const out = new Array(16);
  for (let column = 0; column < 4; column++) {
    for (let row = 0; row < 4; row++) {
      out[column * 4 + row] = a[row] * b[column * 4]
        + a[4 + row] * b[column * 4 + 1]
        + a[8 + row] * b[column * 4 + 2]
        + a[12 + row] * b[column * 4 + 3];
    }
  }
  return out;
}

function transform(m, v) {
  return [
    m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3],
    m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3],
    m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3],
    m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3],
  ];
}

/**
 * Rebuild a camera's world and inverse-world matrices from its own
 * position/quaternion.
 *
 * This is not redundant with updateMatrixWorld: the upstream page's
 * OrbitControls writes camera positions directly, and the camera the harness
 * uses is a clone that the renderer does not always refresh, so
 * matrixWorldInverse can be left holding an older pose. Projection and
 * unprojection then disagree by centimetres while both look locally correct.
 */
export function refreshCamera(camera) {
  camera.scale?.set?.(1, 1, 1);
  camera.updateMatrix();
  camera.matrixWorld.copy(camera.matrix);
  camera.matrixWorldInverse?.copy(camera.matrixWorld).invert();
  return camera;
}

const subtract = (a, b) => a.map((v, i) => v - b[i]);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

function unit(v) {
  const length = Math.hypot(...v) || 1;
  return v.map((value) => value / length);
}

/**
 * Turn a look-from / look-at pair into the position and quaternion a camera
 * needs. Writing a camera as "stand here, look there" is what makes the
 * calibration file editable by hand; a real camera's extrinsic matrix
 * converts into the same two fields.
 */
export function lookAtPose({ lookFrom, lookAt, up = [0, 0, 1] }) {
  const forward = unit(subtract(lookAt, lookFrom));
  const right = unit(cross(forward, up));
  const trueUp = cross(right, forward);
  const back = forward.map((v) => -v); // a camera looks down its own -z
  const m = [
    right[0], trueUp[0], back[0],
    right[1], trueUp[1], back[1],
    right[2], trueUp[2], back[2],
  ];
  const trace = m[0] + m[4] + m[8];
  let q;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    q = [0.25 * s, (m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s];
  } else if (m[0] > m[4] && m[0] > m[8]) {
    const s = Math.sqrt(1 + m[0] - m[4] - m[8]) * 2;
    q = [(m[7] - m[5]) / s, 0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s];
  } else if (m[4] > m[8]) {
    const s = Math.sqrt(1 + m[4] - m[0] - m[8]) * 2;
    q = [(m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25 * s, (m[3] + m[7]) / s];
  } else {
    const s = Math.sqrt(1 + m[8] - m[0] - m[4]) * 2;
    q = [(m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[3] + m[7]) / s, 0.25 * s];
  }
  return { position: [...lookFrom], quaternion: q, up };
}

/**
 * Pinhole projection matrix from measured intrinsics. Unlike a fov-based
 * three.js camera this carries fx, fy, cx and cy independently, so the numbers
 * a real camera is calibrated with drop in unchanged.
 */
export function projectionFor({ fx, fy, cx, cy, width, height }, near = 0.01, far = 100) {
  const m = new Array(16).fill(0);
  m[0] = (2 * fx) / width;
  m[5] = (2 * fy) / height;
  m[8] = 1 - (2 * cx) / width;
  m[9] = (2 * cy) / height - 1;
  m[10] = -(far + near) / (far - near);
  m[11] = -1;
  m[14] = (-2 * far * near) / (far - near);
  return m;
}

export function quatToMatrix(q) {
  const [w, x, y, z] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w), 0,
    2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w), 0,
    2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y), 0,
    0, 0, 0, 1,
  ];
}

/**
 * Point a three.js camera at a calibrated pose. The projection comes from the
 * intrinsics, not from a field of view, so an asymmetric principal point is
 * reproduced exactly.
 */
export function aimCamera(camera, camera_spec) {
  const { intrinsics, pose } = camera_spec;
  camera.projectionMatrix.fromArray(projectionFor(intrinsics));
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  camera.matrixAutoUpdate = true;
  camera.position.set(...pose.position);
  const [w, x, y, z] = pose.quaternion;
  camera.quaternion.set(x, y, z, w);
  camera.up.set(...(pose.up ?? [0, 0, 1])); // MuJoCo world is z-up
  return refreshCamera(camera);
}

/**
 * Grab one frame and return top-down RGBA pixels.
 *
 * The upstream page only copies MuJoCo into the three.js scene from inside its
 * own animation loop, and a browser pane that is not on screen never runs that
 * loop - the render then comes out empty. So the scene is synced here first,
 * with dt = 0: that refreshes the meshes without stepping the physics, which
 * the harness drives itself.
 */
export function sample(app, camera, { hideRobot = false } = {}) {
  const renderer = app.renderer;
  const gl = renderer.getContext();
  const width = renderer.domElement.width;
  const height = renderer.domElement.height;
  const groups = app.mjvOption?.geomgroup;
  const saved = hideRobot && groups ? Array.from(groups) : null;
  try {
    // The upstream model puts the arm's visual geoms in group 2 and its
    // collision geoms in group 3; the cell and the task objects are in groups
    // 1, 4 and 0. Hiding 2 and 3 is self-filtering: a real perception stack
    // knows its own body from kinematics and masks it out, and without it a
    // matte black gripper and a matte black bottle are the same pixels.
    if (saved) {
      groups[2] = 0;
      groups[3] = 0;
    }
    app.update(0, performance.now());
    refreshCamera(camera);
    renderer.render(app.scene, camera);
    refreshCamera(camera);
    const raw = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, raw);
    // GL returns rows bottom-up; flip so row 0 is the top of the image.
    const flipped = new Uint8Array(raw.length);
    const stride = width * 4;
    for (let row = 0; row < height; row++) {
      flipped.set(raw.subarray((height - 1 - row) * stride, (height - row) * stride), row * stride);
    }
    return { data: flipped, width, height };
  } finally {
    if (saved) saved.forEach((value, i) => { groups[i] = value; });
  }
}

/** Largest run of dark pixels inside `window`, how the matte black bottle reads. */
export function darkestBlob(frame, { threshold = 0.10, minPixels = 60, window: area = null } = {}) {
  const { data, width, height } = frame;
  const x0 = Math.max(0, Math.floor(area?.x ?? 0));
  const y0 = Math.max(0, Math.floor(area?.y ?? 0));
  const x1 = Math.min(width, Math.ceil(area ? area.x + area.width : width));
  const y1 = Math.min(height, Math.ceil(area ? area.y + area.height : height));
  if (x1 - x0 < 2 || y1 - y0 < 2) return null;

  const mask = new Uint8Array(width * height);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = y * width + x;
      const p = i * 4;
      if (LUMA(data[p], data[p + 1], data[p + 2]) < threshold) mask[i] = 1;
    }
  }

  const seen = new Uint8Array(width * height);
  const stack = [];
  let best = null;
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || seen[start]) continue;
    stack.length = 0;
    stack.push(start);
    seen[start] = 1;
    let count = 0;
    let minX = width;
    let maxX = -1;
    let minY = height;
    let maxY = -1;
    let sumX = 0;
    let sumY = 0;
    while (stack.length) {
      const index = stack.pop();
      const x = index % width;
      const y = (index - x) / width;
      count += 1;
      sumX += x;
      sumY += y;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (x > 0 && mask[index - 1] && !seen[index - 1]) { seen[index - 1] = 1; stack.push(index - 1); }
      if (x < width - 1 && mask[index + 1] && !seen[index + 1]) { seen[index + 1] = 1; stack.push(index + 1); }
      if (y > 0 && mask[index - width] && !seen[index - width]) { seen[index - width] = 1; stack.push(index - width); }
      if (y < height - 1 && mask[index + width] && !seen[index + width]) { seen[index + width] = 1; stack.push(index + width); }
    }
    if (count < minPixels || count <= (best?.pixels ?? 0)) continue;
    best = {
      pixels: count,
      bbox: { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 },
      centroid: [sumX / count, sumY / count],
      base: [sumX / count, maxY],
    };
  }
  return best;
}

/**
 * Ray through a pixel, intersected with the horizontal plane at `planeZ`.
 *
 * `base` rather than `centroid` is the right pixel for an object standing on
 * that plane: the lowest visible pixel of the object touches the plane, so its
 * ray lands at the object's footprint. The centroid of a tall object projects
 * too high and unprojects to a point behind it.
 */
export function unprojectToPlane(camera, pixel, planeZ, frameSize) {
  refreshCamera(camera);
  const width = frameSize?.width ?? camera.viewportWidth;
  const height = frameSize?.height ?? camera.viewportHeight;
  const inverse = multiply(camera.matrixWorld.elements, camera.projectionMatrixInverse.elements);
  const ndc = (px, py, pz) => [(px / width) * 2 - 1, 1 - (py / height) * 2, pz, 1];

  const near = transform(inverse, ndc(pixel[0], pixel[1], -1));
  const far = transform(inverse, ndc(pixel[0], pixel[1], 1));
  const nearPoint = near.slice(0, 3).map((v) => v / near[3]);
  const farPoint = far.slice(0, 3).map((v) => v / far[3]);

  const direction = farPoint.map((v, i) => v - nearPoint[i]);
  if (Math.abs(direction[2]) < 1e-9) return null;
  const t = (planeZ - nearPoint[2]) / direction[2];
  if (t < 0) return null;
  return nearPoint.map((v, i) => v + direction[i] * t);
}

/**
 * Where a rectangle on the table lands in the image.
 *
 * Perception is restricted to the cell's object area with this. A matte black
 * bottle and a matte black gripper are only a factor of two apart in albedo,
 * so no threshold separates them across a whole frame; the cell declaring
 * where objects are placed is what a real workcell does too, and it is written
 * down in the calibration file rather than hidden in the detector.
 */
export function regionWindow(camera, region, planeZ, frameSize, pad = 6) {
  const corners = [
    [region.x[0], region.y[0]], [region.x[1], region.y[0]],
    [region.x[1], region.y[1]], [region.x[0], region.y[1]],
  ].map(([x, y]) => projectPoint(camera, [x, y, planeZ], frameSize));
  const xs = corners.map((c) => c[0]);
  const ys = corners.map((c) => c[1]);
  return {
    x: Math.min(...xs) - pad,
    y: Math.min(...ys) - pad,
    width: Math.max(...xs) - Math.min(...xs) + 2 * pad,
    height: Math.max(...ys) - Math.min(...ys) + 2 * pad,
  };
}

export function projectPoint(camera, world, frameSize) {
  refreshCamera(camera);
  const width = frameSize?.width ?? camera.viewportWidth;
  const height = frameSize?.height ?? camera.viewportHeight;
  const m = multiply(camera.projectionMatrix.elements, camera.matrixWorldInverse.elements);
  const clip = transform(m, [world[0], world[1], world[2], 1]);
  const w = clip[3] || 1;
  return [((clip[0] / w) + 1) * 0.5 * width, (1 - (clip[1] / w)) * 0.5 * height];
}

/**
 * The world-space ray through a pixel.
 *
 * One camera gives a direction only. For a bottle lying on a known plane that
 * is enough, because the plane supplies the missing depth; for anything
 * standing up it is not, and the two rays from a calibrated pair have to be
 * intersected instead.
 */
export function rayFromPixel(camera, pixel, frameSize) {
  refreshCamera(camera);
  const width = frameSize?.width ?? camera.viewportWidth;
  const height = frameSize?.height ?? camera.viewportHeight;
  const inverse = multiply(camera.matrixWorld.elements, camera.projectionMatrixInverse.elements);
  const ndc = (pz) => [(pixel[0] / width) * 2 - 1, 1 - (pixel[1] / height) * 2, pz, 1];
  const near = transform(inverse, ndc(-1));
  const far = transform(inverse, ndc(1));
  const origin = near.slice(0, 3).map((v) => v / near[3]);
  const target = far.slice(0, 3).map((v) => v / far[3]);
  return { origin, direction: unit(target.map((v, i) => v - origin[i])) };
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/**
 * Midpoint of the shortest segment between two rays: the classic stereo
 * intersection. `residual` is how far apart the two rays pass, which is the
 * honest error bar on the estimate — a wrong correspondence shows up as a
 * large residual instead of as a confident wrong point.
 */
export function triangulate(a, b) {
  const w0 = subtract(a.origin, b.origin);
  const bb = dot(a.direction, b.direction);
  const d = dot(a.direction, w0);
  const e = dot(b.direction, w0);
  const denom = 1 - bb * bb;
  const s = Math.abs(denom) < 1e-9 ? 0 : (bb * e - d) / denom;
  const t = Math.abs(denom) < 1e-9 ? e : (e - bb * d) / denom;
  const pa = a.origin.map((v, i) => v + s * a.direction[i]);
  const pb = b.origin.map((v, i) => v + t * b.direction[i]);
  return {
    point: pa.map((v, i) => (v + pb[i]) / 2),
    residual: Math.hypot(...pa.map((v, i) => v - pb[i])),
  };
}

/**
 * Sample through one of the model's own cameras, by id.
 *
 * Used by the diagnostics rather than by perception: the robot's own cameras
 * move with it and two of the three have no calibrated intrinsics, so they are
 * not what the agent localises from - but they are what a person looks at when
 * a run goes wrong, and they cost nothing to have ready.
 */
export function sampleFromModelCamera(app, mujoco, cameraId, options = {}) {
  const data = app.mjData;
  const position = [data.cam_xpos[cameraId * 3], data.cam_xpos[cameraId * 3 + 1], data.cam_xpos[cameraId * 3 + 2]];
  const m = data.cam_xmat.subarray(cameraId * 9, cameraId * 9 + 9);
  const target = [position[0] - m[2], position[1] - m[5], position[2] - m[8]];
  const camera = app.camera.clone();
  camera.up.set(m[1], m[4], m[7]);
  camera.position.set(...position);
  camera.lookAt(target[0], target[1], target[2]);
  camera.fov = app.mjModel.cam_fovy[cameraId];
  camera.updateProjectionMatrix();
  return sample(app, camera, options);
}

/** Draw the sampled frame into a canvas so the UI can show what the agent saw. */
export function frameToDataUrl(frame) {
  return canvasFor(frame).toDataURL("image/png");
}

function canvasFor(frame) {
  const canvas = document.createElement("canvas");
  canvas.width = frame.width;
  canvas.height = frame.height;
  const context = canvas.getContext("2d");
  const image = context.createImageData(frame.width, frame.height);
  image.data.set(frame.data);
  context.putImageData(image, 0, 0);
  return canvas;
}

/**
 * The agent's observation as one picture: both calibrated views side by side.
 *
 * A single frame is what the interface used to show, and a single frame is not
 * what the agent uses - depth comes from comparing the two. Putting them in
 * one image makes the observation honest about that, and makes a disagreement
 * between the views visible at a glance.
 */
export function stereoToDataUrl(a, b, labelA = "A", labelB = "B") {
  const gap = 6;
  const canvas = document.createElement("canvas");
  canvas.width = a.width + b.width + gap;
  canvas.height = Math.max(a.height, b.height) + 20;
  const context = canvas.getContext("2d");
  context.fillStyle = "#11161c";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(canvasFor(a), 0, 20);
  context.drawImage(canvasFor(b), a.width + gap, 20);
  context.fillStyle = "#d97757";
  context.font = "13px monospace";
  context.fillText(labelA, 8, 14);
  context.fillText(labelB, a.width + gap + 8, 14);
  return canvas.toDataURL("image/png");
}
