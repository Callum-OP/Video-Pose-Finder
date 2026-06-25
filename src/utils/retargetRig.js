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

// The control skeleton has no independent spine/neck articulation (spine, chest, etc.
// are derived linearly from hips + shoulders), so the torso is rigid. Orient these
// bones by the full torso rotation A (like the hips) rather than by a single
// direction — a direction leaves the bone's roll about its axis undetermined, which
// fed the shoulders a twisted chest frame and mangled the arms.
const TORSO = new Set(['Spine', 'Spine1', 'Spine2', 'Neck', 'Head']);

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
    const child = get(CHILD[name]);
    if (!bone || !child) continue;
    const boneWorldQ = wq(name);
    const childDirWorld = child.getWorldPosition(new THREE.Vector3())
      .sub(bone.getWorldPosition(new THREE.Vector3())).normalize();
    const parentWorldQ = bone.parent?.getWorldQuaternion(new THREE.Quaternion()) ?? new THREE.Quaternion();
    const localChildAxis = childDirWorld.clone().applyQuaternion(boneWorldQ.clone().invert());
    // A deterministic local axis perpendicular to the bone, used as the roll
    // reference for bend-plane correction.
    let up = new THREE.Vector3(0, 1, 0);
    if (Math.abs(localChildAxis.dot(up)) > 0.9) up.set(0, 0, 1);
    const localSide = new THREE.Vector3().crossVectors(localChildAxis, up).normalize();
    bind[name] = {
      bone,
      parentName: normName(bone.parent?.name),
      bindLocal: parentWorldQ.clone().invert().multiply(boneWorldQ),
      localChildAxis,
      localSide,
    };
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
  };
}

// Pose the rig for one frame from control positions (string/number keyed scene vecs).
export function poseRig(rigData, pos) {
  if (!rigData) return;
  const { bind, rigBasis, hipsBindQuat, hipsParentBindQuat, hipsBone } = rigData;

  // Control-space torso basis for this frame.
  const up = v3(pos.chest).sub(v3(pos.hips));
  const right = v3(pos[MP.shoL]).sub(v3(pos[MP.shoR]));
  const ctrlBasis = torsoBasis(up, right);

  // A maps rig bind directions into the current control orientation — INCLUDING
  // full body facing — so the whole character turns / leans / lies to match the
  // captured pose (not just the limbs). A = ctrlBasis · rigBasis⁻¹.
  const A = new THREE.Matrix4().multiplyMatrices(ctrlBasis, rigBasis.clone().invert());
  const Aq = new THREE.Quaternion().setFromRotationMatrix(A);

  // Orient the pelvis (and thus the whole body) to the control torso.
  const hipsWorld = Aq.clone().multiply(hipsBindQuat);
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
    // Where the bone would sit if it kept its bind pose under the posed parent.
    const restWorld = parentWorld.clone().multiply(b.bindLocal);

    let desiredWorld;
    if (TORSO.has(name)) {
      // Rigid torso — inherit the parent (and thus the A-oriented hips) unchanged.
      desiredWorld = restWorld;
    } else {
      const seg = SEGMENTS[name];
      if (!seg) continue;
      tmpFrom.copy(v3(pos[seg[0]]));
      tmpTo.copy(v3(pos[seg[1]]));
      const targetDir = tmpTo.sub(tmpFrom);
      if (targetDir.lengthSq() < 1e-8) continue;
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
