// ── Pose-edit math ────────────────────────────────────────────────────────────
// Pure, dependency-free helpers for the 3D pose editor. The editor's source of
// truth is a "control skeleton" of joint world-positions in three.js scene space.
// That space is the SAME as exportBVH's pre-yaw raw space, so whatever you edit is
// what gets exported:
//
//   worldLandmark {x,y,z} → scene [-x*SCALE, -y*SCALE, z*SCALE]   (see exportBVH mp())
//
// Edits are written back into frame.worldLandmarks (and 2D landmarks for the canvas
// /JSON), and exportBVH/exportJSON then consume them unchanged. Yaw/gravity are NOT
// applied here — exportBVH recomputes yaw from the (edited) positions.

const SCALE = 100;

// ── Small vector / quaternion helpers (mirrors exportBVH.js) ───────────────────
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (v, s) => [v[0] * s, v[1] * s, v[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const len = (v) => Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2);
export const norm = (v) => { const l = len(v) || 1e-8; return [v[0] / l, v[1] / l, v[2] / l]; };
export const avg = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
export const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

// Rotate vector v by quaternion q = [w,x,y,z].
export function quatRotate([w, x, y, z], [vx, vy, vz]) {
  // t = 2 * cross(q.xyz, v)
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  // v + w*t + cross(q.xyz, t)
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx),
  ];
}

// ── Coordinate conversion (exact inverse pair) ────────────────────────────────
export function sceneFromWorld(lm) {
  return [-lm.x * SCALE, -lm.y * SCALE, lm.z * SCALE];
}
// Falls back to the 2D landmark (z=0) when no world landmark, matching exportBVH mp().
export function scenePosFor(frame, idx) {
  const w = frame.worldLandmarks?.[idx];
  if (w) return sceneFromWorld(w);
  const lm = frame.landmarks?.[idx];
  if (lm) return [-lm.x * SCALE, -lm.y * SCALE, 0];
  return [0, 0, 0];
}
const round4 = (n) => parseFloat(n.toFixed(4));
export function worldFromScene([X, Y, Z], v = 1) {
  return { x: round4(-X / SCALE), y: round4(-Y / SCALE), z: round4(Z / SCALE), v };
}

// ── MediaPipe joint indices the editor tracks ─────────────────────────────────
export const MP = {
  nose: 0, earL: 7, earR: 8,
  shoL: 11, shoR: 12, elbL: 13, elbR: 14, wriL: 15, wriR: 16,
  hipL: 23, hipR: 24, kneeL: 25, kneeR: 26, ankL: 27, ankR: 28,
  heelL: 29, heelR: 30, toeL: 31, toeR: 32,
};

// Every MediaPipe index we read/write (used for whole-body root rotate + write-back).
export const TRACKED_MP = Object.values(MP);

// ── Derived torso joints (computed exactly like exportBVH P()) ─────────────────
// Keyed by name; never written back — exportBVH recomputes them from the MP joints.
function deriveTorso(pos) {
  const hips = avg(pos[MP.hipL], pos[MP.hipR]);
  const chest = avg(pos[MP.shoL], pos[MP.shoR]);          // = spine2 / shoulders
  const earMid = avg(pos[MP.earL], pos[MP.earR]);         // = head
  return {
    hips,
    chest,
    head: earMid,
    spine: avg(hips, chest),
    neck: avg(chest, earMid),
  };
}

// ── Build control-skeleton positions for a frame ──────────────────────────────
// Returns a flat object: numeric keys = MP joint scene positions, string keys =
// derived torso joints. This object is the live, editable state.
export function buildControlPositions(frame) {
  const pos = {};
  for (const idx of TRACKED_MP) pos[idx] = scenePosFor(frame, idx);
  Object.assign(pos, deriveTorso(pos));
  return pos;
}

// Recompute the derived torso joints after MP joints move.
export function refreshDerived(pos) {
  Object.assign(pos, deriveTorso(pos));
  return pos;
}

// ── Bones to draw (pairs of position keys) ────────────────────────────────────
export const BONES = [
  ['hips', 'spine'], ['spine', 'chest'], ['chest', 'neck'], ['neck', 'head'],
  ['chest', MP.shoL], ['chest', MP.shoR],
  [MP.shoL, MP.elbL], [MP.elbL, MP.wriL],
  [MP.shoR, MP.elbR], [MP.elbR, MP.wriR],
  ['hips', MP.hipL], ['hips', MP.hipR],
  [MP.hipL, MP.kneeL], [MP.kneeL, MP.ankL], [MP.ankL, MP.heelL], [MP.heelL, MP.toeL],
  [MP.hipR, MP.kneeR], [MP.kneeR, MP.ankR], [MP.ankR, MP.heelR], [MP.heelR, MP.toeR],
];

// ── Editable joints + their rotate subtrees ───────────────────────────────────
// `key`     position key the gizmo attaches to (the rotation pivot).
// `rotate`  MP indices that swing about the pivot when rotating (strict descendants).
// `move`    whether free-move is allowed (single-point drag).
// `root`    move translates the whole figure instead of a single joint.
export const EDIT_TARGETS = [
  { key: 'head',     label: 'Head',       rotate: [MP.nose, MP.earL, MP.earR], move: false },
  { key: 'neck',     label: 'Neck',       rotate: [MP.nose, MP.earL, MP.earR], move: false },
  { key: 'chest',    label: 'Chest / torso', rotate: upperBody(),              move: true },
  { key: MP.shoL,    label: 'L shoulder', rotate: [MP.elbL, MP.wriL],          move: true },
  { key: MP.elbL,    label: 'L elbow',    rotate: [MP.wriL],                   move: true },
  { key: MP.wriL,    label: 'L wrist',    rotate: [],                          move: true },
  { key: MP.shoR,    label: 'R shoulder', rotate: [MP.elbR, MP.wriR],          move: true },
  { key: MP.elbR,    label: 'R elbow',    rotate: [MP.wriR],                   move: true },
  { key: MP.wriR,    label: 'R wrist',    rotate: [],                          move: true },
  { key: MP.hipL,    label: 'L hip',      rotate: [MP.kneeL, MP.ankL, MP.heelL, MP.toeL], move: true },
  { key: MP.kneeL,   label: 'L knee',     rotate: [MP.ankL, MP.heelL, MP.toeL], move: true },
  { key: MP.ankL,    label: 'L ankle',    rotate: [MP.heelL, MP.toeL],         move: true },
  { key: MP.toeL,    label: 'L foot',     rotate: [],                          move: true },
  { key: MP.hipR,    label: 'R hip',      rotate: [MP.kneeR, MP.ankR, MP.heelR, MP.toeR], move: true },
  { key: MP.kneeR,   label: 'R knee',     rotate: [MP.ankR, MP.heelR, MP.toeR], move: true },
  { key: MP.ankR,    label: 'R ankle',    rotate: [MP.heelR, MP.toeR],         move: true },
  { key: MP.toeR,    label: 'R foot',     rotate: [],                          move: true },
  { key: 'hips',     label: 'Hips (root)', rotate: TRACKED_MP,                 move: true, root: true },
];

// Upper-body MP joints (everything from the chest up) — chest lean rotate subtree.
function upperBody() {
  return [MP.shoL, MP.shoR, MP.elbL, MP.elbR, MP.wriL, MP.wriR,
          MP.nose, MP.earL, MP.earR];
}

export const EDIT_TARGET_BY_KEY = Object.fromEntries(EDIT_TARGETS.map((t) => [String(t.key), t]));

// ── Length-preserving subtree rotation (the Rotate tool) ──────────────────────
// Rotates every MP joint in target.rotate about the pivot by dq=[w,x,y,z]. Pure
// rotation about a fixed pivot preserves all bone lengths within the subtree.
// `pivot` is the scene position to rotate around (the joint at drag start).
export function rotateSubtree(pos, target, dq, pivot) {
  for (const idx of target.rotate) {
    if (!pos[idx]) continue;
    pos[idx] = add(pivot, quatRotate(dq, sub(pos[idx], pivot)));
  }
  return refreshDerived(pos);
}

// ── Free move (the Move tool) ─────────────────────────────────────────────────
// Root targets translate the whole figure; others set a single MP joint.
export function moveJoint(pos, target, newScenePos) {
  if (target.root) {
    const delta = sub(newScenePos, pos[target.key]);
    for (const idx of TRACKED_MP) pos[idx] = add(pos[idx], delta);
  } else if (typeof target.key === 'number') {
    pos[target.key] = newScenePos.slice();
  } else if (Array.isArray(target.rotate) && target.rotate.length) {
    // Derived group joint (e.g. chest): translate the whole group so the torso
    // leans/bends rather than moving a single point.
    const delta = sub(newScenePos, pos[target.key]);
    for (const idx of target.rotate) if (pos[idx]) pos[idx] = add(pos[idx], delta);
  }
  return refreshDerived(pos);
}

// ── Write edited control positions back into a frame ──────────────────────────
// Only MP-indexed joints are written (derived torso joints are recomputed by the
// exporter). Returns a NEW { landmarks, worldLandmarks } pair (immutable update).
//
// The editor builds scene positions from `worldLandmarks` when present (metric, the
// space the BVH exporter consumes), else from the 2D `landmarks` (normalised screen
// coords). Write back ONLY to whichever space we edited — the two are different
// coordinate systems, so cross-writing world-metric values into the 2D screen-space
// `landmarks` is what previously mangled the FrameInspector preview.
export function writeBackFrame(frame, pos) {
  const hasWorld = !!frame.worldLandmarks;
  const worldLandmarks = hasWorld ? frame.worldLandmarks.map((lm) => ({ ...lm })) : null;
  const landmarks = frame.landmarks.map((lm) => ({ ...lm }));

  for (const idx of TRACKED_MP) {
    const p = pos[idx];
    if (!p) continue;
    if (hasWorld && worldLandmarks[idx]) {
      Object.assign(worldLandmarks[idx], worldFromScene(p, worldLandmarks[idx].v ?? 1));
    } else if (!hasWorld && landmarks[idx]) {
      landmarks[idx].x = round4(-p[0] / SCALE);
      landmarks[idx].y = round4(-p[1] / SCALE);
    }
  }
  return { landmarks, worldLandmarks };
}
