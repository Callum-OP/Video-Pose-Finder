import * as THREE from 'three';
import { MP } from './poseEditMath.js';

// ── Forward retarget: control skeleton → rigged humanoid (glTF or FBX) ─────────
// The control skeleton (joint world-positions, our edit source of truth) drives a
// Mixamo-rigged humanoid as a *display-only* skin. We never read the rig back —
// edits are written from the control skeleton straight to worldLandmarks — so this
// mapping only has to look right, not round-trip.
//
// Method: each frame we build a torso basis (up/right/fwd) from the control
// skeleton and from the rig's bind pose, derive the rotation A that maps the bind
// orientation onto the current control orientation, orient the pelvis (whole body)
// by A so the character turns/leans/lies to match the captured pose, then swing
// each limb bone so its bind child-direction points along its control segment.
//
// Bone names are matched by their Mixamo *suffix* so this works for glTF
// ("mixamorig:LeftArm") and FBX ("mixamorig:LeftArm", "mixamorigLeftArm",
// "Armature|mixamorig:LeftArm", or bare "LeftArm") alike.

// Strip namespace/prefix decoration down to the bare Mixamo bone suffix.
function normName(name = '') {
  let n = name;
  const bar = n.lastIndexOf('|'); if (bar >= 0) n = n.slice(bar + 1);
  const colon = n.lastIndexOf(':'); if (colon >= 0) n = n.slice(colon + 1);
  n = n.replace(/^mixamorig[:_]?/i, '');
  return n;
}

// Bone suffix → control segment [fromKey, toKey] it should point along.
const SEGMENTS = {
  Spine: ['hips', 'chest'], Spine1: ['hips', 'chest'], Spine2: ['hips', 'chest'],
  Neck: ['chest', 'head'], Head: ['chest', 'head'],
  LeftShoulder: ['chest', MP.shoL], LeftArm: [MP.shoL, MP.elbL], LeftForeArm: [MP.elbL, MP.wriL],
  RightShoulder: ['chest', MP.shoR], RightArm: [MP.shoR, MP.elbR], RightForeArm: [MP.elbR, MP.wriR],
  LeftUpLeg: [MP.hipL, MP.kneeL], LeftLeg: [MP.kneeL, MP.ankL], LeftFoot: [MP.ankL, MP.toeL],
  RightUpLeg: [MP.hipR, MP.kneeR], RightLeg: [MP.kneeR, MP.ankR], RightFoot: [MP.ankR, MP.toeR],
};

// Bone suffix → the child whose bind direction defines the bone's "forward".
const CHILD = {
  Spine: 'Spine1', Spine1: 'Spine2', Spine2: 'Neck', Neck: 'Head', Head: 'HeadTop_End',
  LeftShoulder: 'LeftArm', LeftArm: 'LeftForeArm', LeftForeArm: 'LeftHand',
  RightShoulder: 'RightArm', RightArm: 'RightForeArm', RightForeArm: 'RightHand',
  LeftUpLeg: 'LeftLeg', LeftLeg: 'LeftFoot', LeftFoot: 'LeftToeBase',
  RightUpLeg: 'RightLeg', RightLeg: 'RightFoot', RightFoot: 'RightToeBase',
};

// Parents before children, so each bone's current parent world rotation is known.
const ORDER = [
  'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
  'LeftShoulder', 'LeftArm', 'LeftForeArm',
  'RightShoulder', 'RightArm', 'RightForeArm',
  'LeftUpLeg', 'LeftLeg', 'LeftFoot',
  'RightUpLeg', 'RightLeg', 'RightFoot',
];

// Torso bones. MediaPipe gives only hips + shoulders (no mid-spine), so we orient
// the pelvis from the hip line and the chest from the shoulder line, then SLERP that
// difference across the spine sections (fraction below) so the torso twists/leans
// through three sections instead of moving as one rigid block. Neck/Head ride the
// chest (fraction 1).
const TORSO = new Set(['Spine', 'Spine1', 'Spine2', 'Neck', 'Head']);
const TORSO_FRACTION = { Spine: 0.34, Spine1: 0.67, Spine2: 1, Neck: 1, Head: 1 };

// Upper-arm bones get a roll correction: a T-pose arm (pointing sideways at bind)
// swung down to its target picks an arbitrary roll, twisting the skinned mesh. We
// roll the bone so its side axis tracks the elbow bend plane. Maps bone → the
// grandchild control joint (wrist) that, with the elbow, defines the bend.
const ROLL_TO_BEND = { LeftArm: MP.wriL, RightArm: MP.wriR };

const v3 = (a) => new THREE.Vector3(a[0], a[1], a[2]);

function torsoBasis(up, right) {
  const u = up.clone().normalize();
  const f = new THREE.Vector3().crossVectors(right, u).normalize(); // forward
  const r = new THREE.Vector3().crossVectors(u, f).normalize();     // re-orthogonalised right
  return new THREE.Matrix4().makeBasis(r, u, f);
}

// Capture the rig's bind pose once, after the model is in the scene and world
// matrices are updated. Returns the data poseRig() consumes each frame, or null if
// the rig isn't a recognised Mixamo skeleton.
export function buildRigBindData(root) {
  root.updateMatrixWorld(true);

  // Index every node by its bare Mixamo suffix (bones win over plain Object3Ds).
  const byName = {};
  root.traverse((o) => {
    const key = normName(o.name);
    if (!key) return;
    if (!byName[key] || o.isBone) byName[key] = o;
  });
  const get = (suffix) => byName[suffix];

  if (!get('Hips') || !get('Spine2')) return null;

  const wp = (suffix) => get(suffix)?.getWorldPosition(new THREE.Vector3());
  const wq = (suffix) => get(suffix)?.getWorldQuaternion(new THREE.Quaternion());

  const hips = wp('Hips');
  const chest = wp('Spine2');
  const rigBasis = torsoBasis(chest.clone().sub(hips), wp('LeftArm').clone().sub(wp('RightArm')));

  // Per-bone bind data for a chained-swing solve:
  //   parentName     normalised parent suffix (to read the posed parent each frame)
  //   bindLocal      bone rotation relative to its parent at bind
  //   localChildAxis unit child-direction expressed in the bone's local frame
  const bind = {};
  for (const name of ORDER) {
    const bone = get(name);
    if (!bone) continue;
    const boneWorldQ = wq(name);
    const parentWorldQ = bone.parent?.getWorldQuaternion(new THREE.Quaternion()) ?? new THREE.Quaternion();
    const entry = {
      bone,
      parentName: normName(bone.parent?.name),
      bindWorldQuat: boneWorldQ.clone(),
      bindLocal: parentWorldQ.clone().invert().multiply(boneWorldQ),
      localChildAxis: null,
      localSide: null,
    };
    // The child defines the bone's forward axis for the swing solve. Torso bones
    // are oriented by bind quaternion and don't need it; a missing child (e.g. a
    // non-Mixamo head tip name) just means that bone follows its parent.
    const child = get(CHILD[name]);
    if (child) {
      const childDirWorld = child.getWorldPosition(new THREE.Vector3())
        .sub(bone.getWorldPosition(new THREE.Vector3())).normalize();
      const localChildAxis = childDirWorld.clone().applyQuaternion(boneWorldQ.clone().invert());
      // A deterministic local axis perpendicular to the bone, used as the roll
      // reference for bend-plane correction.
      let up = new THREE.Vector3(0, 1, 0);
      if (Math.abs(localChildAxis.dot(up)) > 0.9) up.set(0, 0, 1);
      entry.localChildAxis = localChildAxis;
      entry.localSide = new THREE.Vector3().crossVectors(localChildAxis, up).normalize();
    }
    bind[name] = entry;
  }

  // Pose-invariant height proxy (torso + one leg) for auto-scaling the rig to the
  // control figure regardless of file units (glTF metres vs FBX centimetres).
  const head = wp('Head') ?? wp('HeadTop_End');
  const d = (a, b) => (a && b ? a.distanceTo(b) : 0);
  const bindProxy =
    d(hips, chest) + d(chest, head) +
    d(wp('LeftUpLeg'), wp('LeftLeg')) + d(wp('LeftLeg'), wp('LeftFoot'));

  const hipsBone = get('Hips');
  return {
    bind, rigBasis,
    hipsBone,
    hipsBindPos: hips.clone(),
    hipsBindQuat: wq('Hips'),
    hipsParentBindQuat: hipsBone.parent?.getWorldQuaternion(new THREE.Quaternion()) ?? new THREE.Quaternion(),
    bindProxy: bindProxy || 1,
    fingers: { left: buildHandBind(get, 'Left'), right: buildHandBind(get, 'Right') },
  };
}

// ── Finger driving ────────────────────────────────────────────────────────────
// Each finger is a 3-bone chain (knuckle → mid → tip) that curls about a single
// hinge — the axis across the knuckles. The captured `fingerAngles` (degrees of
// flexion per joint) drive the bend; the export already uses the same angles.
const FINGERS = {
  Thumb:  { bones: ['HandThumb1', 'HandThumb2', 'HandThumb3'],  keys: ['mcp', 'ip', 'ip'],   scale: [1, 1, 0.4] },
  Index:  { bones: ['HandIndex1', 'HandIndex2', 'HandIndex3'],  keys: ['mcp', 'pip', 'dip'] },
  Middle: { bones: ['HandMiddle1', 'HandMiddle2', 'HandMiddle3'], keys: ['mcp', 'pip', 'dip'] },
  Ring:   { bones: ['HandRing1', 'HandRing2', 'HandRing3'],     keys: ['mcp', 'pip', 'dip'] },
  Pinky:  { bones: ['HandPinky1', 'HandPinky2', 'HandPinky3'],  keys: ['mcp', 'pip', 'dip'] },
};
// Sign of the bend about the knuckle axis (flexion curls fingers toward the palm).
const FINGER_SIGN = -1;

function buildHandBind(get, side) {
  const idx = get(side + 'HandIndex1');
  const pinky = get(side + 'HandPinky1');
  if (!idx || !pinky) return null;
  const wpos = (b) => b.getWorldPosition(new THREE.Vector3());
  const knuckleAxis = wpos(pinky).sub(wpos(idx)).normalize();   // world curl axis
  const out = {};
  for (const [finger, def] of Object.entries(FINGERS)) {
    const chain = [];
    for (let i = 0; i < def.bones.length; i++) {
      const bone = get(side + def.bones[i]);
      if (!bone) break;
      const parentWorld = bone.parent.getWorldQuaternion(new THREE.Quaternion());
      const boneWorld = bone.getWorldQuaternion(new THREE.Quaternion());
      chain.push({
        bone,
        bindLocal: parentWorld.clone().invert().multiply(boneWorld),
        hinge: knuckleAxis.clone().applyQuaternion(parentWorld.clone().invert()), // into parent-local
        key: def.keys[i],
        scale: def.scale?.[i] ?? 1,
      });
    }
    if (chain.length) out[finger.toLowerCase()] = chain;
  }
  return out;
}

// Pose the rig fingers from a frame's handData.{left,right}.fingerAngles.
export function poseFingers(rigData, handData) {
  if (!rigData?.fingers || !handData) return;
  for (const side of ['left', 'right']) {
    const hand = rigData.fingers[side];
    const angles = handData[side]?.fingerAngles;
    if (!hand || !angles) continue;
    for (const finger in hand) {
      const fa = angles[finger];
      if (!fa) continue;
      for (const j of hand[finger]) {
        const deg = Math.max(0, Math.min(90, (fa[j.key] ?? 0) * j.scale));
        const rot = new THREE.Quaternion().setFromAxisAngle(j.hinge, FINGER_SIGN * deg * Math.PI / 180);
        j.bone.quaternion.copy(rot.multiply(j.bindLocal));   // bend in the parent frame
      }
    }
  }
}

// Pose the rig for one frame from control positions (string/number keyed scene vecs).
export function poseRig(rigData, pos) {
  if (!rigData) return;
  const { bind, rigBasis, hipsBindQuat, hipsParentBindQuat, hipsBone } = rigData;

  // Two torso orientations: pelvis from the hip line, chest from the shoulder line
  // (both share the hips→shoulders up). Their difference is the torso twist/lean,
  // which we distribute across the spine sections. Each maps the rig's bind torso
  // basis onto the control orientation, so both include full body facing.
  const up = v3(pos.chest).sub(v3(pos.hips));
  const rbInv = rigBasis.clone().invert();
  const pelvisBasis = torsoBasis(up, v3(pos[MP.hipL]).sub(v3(pos[MP.hipR])));
  const chestBasis  = torsoBasis(up, v3(pos[MP.shoL]).sub(v3(pos[MP.shoR])));
  const Apelvis = new THREE.Quaternion().setFromRotationMatrix(
    new THREE.Matrix4().multiplyMatrices(pelvisBasis, rbInv));
  const Achest = new THREE.Quaternion().setFromRotationMatrix(
    new THREE.Matrix4().multiplyMatrices(chestBasis, rbInv));

  // Head orientation from the face points (ear line + neck→head up) so rotating the
  // head/neck control actually turns the rig head. Falls back to the chest when the
  // face frame is degenerate (missing/edge-on ears).
  let Ahead = Achest;
  const earSpan = v3(pos[MP.earL]).sub(v3(pos[MP.earR]));
  const headUp = v3(pos.head).sub(v3(pos.neck));
  if (earSpan.lengthSq() > 1e-6 && headUp.lengthSq() > 1e-6) {
    const headBasis = torsoBasis(headUp, earSpan);
    Ahead = new THREE.Quaternion().setFromRotationMatrix(
      new THREE.Matrix4().multiplyMatrices(headBasis, rbInv));
  }

  // Orient the pelvis to the hip line (turns the whole body to face the pose).
  const hipsWorld = Apelvis.clone().multiply(hipsBindQuat);
  if (hipsBone) {
    hipsBone.quaternion.copy(hipsParentBindQuat.clone().invert().multiply(hipsWorld));
  }

  const tmpFrom = new THREE.Vector3();
  const tmpTo = new THREE.Vector3();

  for (const name of ORDER) {
    const b = bind[name];
    if (!b) continue;

    // Read the parent's CURRENT world rotation straight from the scene graph (it
    // recomputes from the locals we've set so far). This is robust to extra/
    // intermediate bones the name table doesn't know about — e.g. a LeftShoulder2
    // between shoulder and arm — which would otherwise break the parent chain and
    // fling the arms. getWorldQuaternion() updates the parent's world matrix first.
    const parentWorld = b.bone.parent.getWorldQuaternion(new THREE.Quaternion());

    let desiredWorld;
    if (TORSO.has(name)) {
      // Spine sections interpolate pelvis→chest so the torso bends/twists through
      // three sections; the neck blends chest→head and the head takes the face
      // orientation, so head/neck edits actually turn the rig head.
      let A;
      if (name === 'Head') A = Ahead;
      else if (name === 'Neck') A = Achest.clone().slerp(Ahead, 0.5);
      else A = Apelvis.clone().slerp(Achest, TORSO_FRACTION[name] ?? 1);
      desiredWorld = A.clone().multiply(b.bindWorldQuat);
    } else {
      // Where the bone would sit if it kept its bind pose under the posed parent.
      const restWorld = parentWorld.clone().multiply(b.bindLocal);
      const seg = SEGMENTS[name];
      tmpFrom.copy(v3(pos[seg?.[0]] ?? [0, 0, 0]));
      tmpTo.copy(v3(pos[seg?.[1]] ?? [0, 0, 0]));
      const targetDir = tmpTo.sub(tmpFrom);
      // A bone with no usable child/target just follows its parent at rest.
      if (!seg || !b.localChildAxis || targetDir.lengthSq() < 1e-8) {
        b.bone.quaternion.copy(parentWorld.clone().invert().multiply(restWorld));
        continue;
      }
      targetDir.normalize();

      // Swing ONLY: rotate the bone's current child-direction onto the target,
      // inheriting the parent's roll. This keeps twist continuous down the chain
      // (the previous per-bone minimal rotation left roll free → mangled arms).
      const restDir = b.localChildAxis.clone().applyQuaternion(restWorld);
      const swing = new THREE.Quaternion().setFromUnitVectors(restDir, targetDir);
      desiredWorld = swing.multiply(restWorld);

      // Upper-arm roll correction: twist about the bone axis so the bone's side
      // tracks the elbow bend plane, instead of the swing's arbitrary roll (which
      // twists the skinned arm). Skipped when the arm is near-straight (no plane).
      const grand = ROLL_TO_BEND[name];
      if (grand !== undefined) {
        const f = targetDir;                                  // bone axis (unit)
        const bend = v3(pos[grand]).sub(v3(pos[seg[1]]));     // elbow → wrist
        const hinge = new THREE.Vector3().crossVectors(f, bend);  // ⟂ bend plane
        if (hinge.lengthSq() > 1e-5) {
          hinge.normalize();
          const curSide = b.localSide.clone().applyQuaternion(desiredWorld);
          const proj = (vv) => vv.clone().addScaledVector(f, -vv.dot(f));
          const curS = proj(curSide), tgtS = proj(hinge);
          if (curS.lengthSq() > 1e-8 && tgtS.lengthSq() > 1e-8) {
            curS.normalize(); tgtS.normalize();
            let ang = Math.acos(Math.max(-1, Math.min(1, curS.dot(tgtS))));
            if (new THREE.Vector3().crossVectors(curS, tgtS).dot(f) < 0) ang = -ang;
            desiredWorld = new THREE.Quaternion().setFromAxisAngle(f, ang).multiply(desiredWorld);
          }
        }
      }
    }

    b.bone.quaternion.copy(parentWorld.clone().invert().multiply(desiredWorld));
  }
}
