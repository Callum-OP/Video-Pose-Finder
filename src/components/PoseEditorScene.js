import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';
import {
  MP, BONES, EDIT_TARGETS, EDIT_TARGET_BY_KEY, TRACKED_MP,
  rotateSubtree, moveJoint,
} from '../utils/poseEditMath';
import { buildRigBindData, poseRig } from '../utils/retargetRig';

// Default character candidates, tried in order — drop a Mixamo-rigged character.fbx
// in public/models to use your own; the bundled Xbot.glb is the fallback. Paths are
// prefixed with Vite's BASE_URL so they resolve under the app's base (e.g. on Pages).
const BASE = import.meta.env.BASE_URL || '/';
const DEFAULT_MODELS = [`${BASE}models/character.fbx`, `${BASE}models/character.glb`];

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

// ── Imperative three.js layer for the 3D pose editor ──────────────────────────
// Owns the scene/camera/renderer, the control-skeleton visual (capsule mannequin
// + pickable joint spheres), the optional rigged GLB skin, and the rotate/move
// gizmo. The React component drives it through a tiny imperative API and receives
// committed edits via the onEdit callback.

const ACCENT = 0x7c6cff;
const JOINT_COL = 0x39e8a0;
const SEL_COL = 0xf5a623;
const v3 = (a) => new THREE.Vector3(a[0], a[1], a[2]);
const arr = (q) => [q.w, q.x, q.y, q.z];

export class PoseEditorScene {
  constructor(container, { onEdit, onSelect, modelUrls = DEFAULT_MODELS } = {}) {
    this.container = container;
    this.onEdit = onEdit || (() => {});
    this.onSelect = onSelect || (() => {});
    this.pos = null;            // live control positions for the current frame
    this.tool = 'rotate';
    this.selectedKey = null;
    this.showMesh = true;
    this.rigData = null;
    this._rigScale = null;
    this._dragSnapshot = null;
    this._framed = false;

    const w = container.clientWidth || 800;
    const h = container.clientHeight || 600;

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.shadowMap.enabled = false;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0a0f);

    this.camera = new THREE.PerspectiveCamera(45, w / h, 1, 5000);
    this.camera.position.set(0, 40, 320);

    // Lights
    this.scene.add(new THREE.HemisphereLight(0xbfc7ff, 0x141420, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 1.4);
    key.position.set(120, 220, 180);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x8899ff, 0.4);
    fill.position.set(-150, 60, -120);
    this.scene.add(fill);

    // Ground grid (XZ plane). Figure roughly spans ±100 units.
    this.grid = new THREE.GridHelper(600, 30, 0x2a2a3a, 0x1b1b27);
    this.scene.add(this.grid);

    // ── Controls ──────────────────────────────────────────────────────────
    this.orbit = new OrbitControls(this.camera, this.renderer.domElement);
    this.orbit.enableDamping = true;
    this.orbit.target.set(0, 0, 0);

    this.transform = new TransformControls(this.camera, this.renderer.domElement);
    this.transform.setSize(0.9);
    this.scene.add(this.transform.getHelper());
    this.gizmoPivot = new THREE.Object3D();
    this.scene.add(this.gizmoPivot);

    this.transform.addEventListener('dragging-changed', (e) => {
      this.orbit.enabled = !e.value;
      if (e.value) this._beginDrag();
      else this._endDrag();
    });
    this.transform.addEventListener('objectChange', () => this._onGizmoChange());

    // ── Figure (capsule mannequin) + joint pick spheres ────────────────────
    this.figure = new THREE.Group();
    this.scene.add(this.figure);
    this._bones = BONES.map(() => this._makeLimb());
    this._bones.forEach((m) => this.figure.add(m));

    this.jointGroup = new THREE.Group();
    this.scene.add(this.jointGroup);
    this._jointMeshes = new Map();
    for (const t of EDIT_TARGETS) {
      const isRoot = !!t.root;
      const geo = new THREE.SphereGeometry(isRoot ? 7 : 5, 16, 12);
      const mat = new THREE.MeshStandardMaterial({
        color: JOINT_COL, emissive: JOINT_COL, emissiveIntensity: 0.25,
        roughness: 0.5, metalness: 0,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.userData.targetKey = String(t.key);
      this.jointGroup.add(mesh);
      this._jointMeshes.set(String(t.key), mesh);
    }

    // ── Optional rigged humanoid skin (FBX or glTF) ────────────────────────
    this.rigGroup = new THREE.Group();
    this.scene.add(this.rigGroup);
    this._loadRig(modelUrls);

    // ── Picking ────────────────────────────────────────────────────────────
    this.raycaster = new THREE.Raycaster();
    this._pointer = new THREE.Vector2();
    this._onPointerDown = (e) => this._pick(e);
    this.renderer.domElement.addEventListener('pointerdown', this._onPointerDown);

    this._onResize = () => this.resize();
    window.addEventListener('resize', this._onResize);

    this._animate = this._animate.bind(this);
    this._raf = requestAnimationFrame(this._animate);
  }

  _makeLimb() {
    const geo = new THREE.CylinderGeometry(1, 1, 1, 12, 1, false);
    const mat = new THREE.MeshStandardMaterial({ color: ACCENT, roughness: 0.6, metalness: 0.05 });
    return new THREE.Mesh(geo, mat);
  }

  async _loadRig(urls) {
    let root = null;
    for (const url of urls) {
      try {
        const lower = url.toLowerCase();
        if (lower.endsWith('.fbx')) {
          root = await new FBXLoader().loadAsync(url);
        } else {
          root = (await new GLTFLoader().loadAsync(url)).scene;
        }
        if (root) { console.log(`[PoseEditor] Loaded character: ${url}`); break; }
      } catch {
        // Try the next candidate.
      }
    }
    if (!root) {
      console.warn('[PoseEditor] No character model loaded — using mannequin');
      this.showMesh = false;
      this._applyMeshVisibility();
      return;
    }

    this.rigData = buildRigBindData(root);
    if (!this.rigData) console.warn('[PoseEditor] Model is not a recognised Mixamo rig — using mannequin');
    root.traverse((o) => {
      if (o.isMesh) {
        o.frustumCulled = false;
        o.material = new THREE.MeshStandardMaterial({ color: 0xcfd3e6, roughness: 0.75, metalness: 0.05 });
      }
    });
    this.rigGroup.add(root);
    this._rigRoot = root;
    this._applyMeshVisibility();
    if (this.pos) this._update();
  }

  // Scale the rig to the control figure once, so glTF (metres) and FBX (often
  // centimetres) both fit. Uses a pose-invariant torso+leg length proxy.
  _ensureRigScale() {
    if (this._rigScale != null || !this.rigData || !this.pos) return;
    const p = this.pos;
    const head = p.head ?? p[MP.nose];
    const controlProxy =
      dist(p.hips, p.chest) + dist(p.chest, head) +
      dist(p[MP.hipL], p[MP.kneeL]) + dist(p[MP.kneeL], p[MP.ankL]);
    const s = controlProxy / this.rigData.bindProxy;
    this._rigScale = Number.isFinite(s) && s > 0 ? s : 1;
    this.rigGroup.scale.setScalar(this._rigScale);
  }

  // ── Public API ────────────────────────────────────────────────────────────
  setPose(pos) {
    this.pos = pos;
    if (this.selectedKey) this._placeGizmo();
    this._update();
    if (!this._framed) { this._frameCamera(); this._framed = true; }
  }

  setTool(tool) {
    this.tool = tool;
    if (this.transform.object) this.transform.setMode(tool === 'move' ? 'translate' : 'rotate');
    // Wrists/feet/chest/head have no meaningful free-move vs rotate restriction here;
    // the gizmo simply reflects the active tool.
    this._refreshGizmoForSelection();
  }

  setShowMesh(on) {
    this.showMesh = on;
    this._applyMeshVisibility();
  }

  select(key) {
    this.selectedKey = key ? String(key) : null;
    this.onSelect(this.selectedKey);
    this._refreshGizmoForSelection();
    this._paintSelection();
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  dispose() {
    cancelAnimationFrame(this._raf);
    window.removeEventListener('resize', this._onResize);
    this.renderer.domElement.removeEventListener('pointerdown', this._onPointerDown);
    // NB: three r169's TransformControls.dispose() calls this.traverse() but the
    // control is no longer an Object3D, so it throws. Tear it down manually: drop
    // DOM listeners (disconnect) and dispose/remove its helper object instead.
    this.transform.detach();
    try { this.transform.disconnect(); } catch { /* older API */ }
    const helper = this.transform.getHelper?.();
    if (helper) {
      helper.traverse((c) => {
        c.geometry?.dispose?.();
        const m = c.material;
        if (m) (Array.isArray(m) ? m : [m]).forEach((x) => x.dispose?.());
      });
      this.scene.remove(helper);
    }
    this.orbit.dispose();
    this.scene.traverse((o) => {
      if (o.geometry) o.geometry.dispose?.();
      if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose?.());
    });
    this.renderer.dispose();
    if (this.renderer.domElement.parentNode === this.container) {
      this.container.removeChild(this.renderer.domElement);
    }
  }

  // ── Internals ───────────────────────────────────────────────────────────
  _applyMeshVisibility() {
    const haveRig = !!this._rigRoot;
    const meshOn = this.showMesh && haveRig;
    this.rigGroup.visible = meshOn;
    // Hide mannequin limbs when the skin is shown; keep joint spheres for picking.
    this._bones.forEach((m) => { m.visible = !meshOn; });
  }

  _frameCamera() {
    if (!this.pos) return;
    const c = v3(this.pos.chest);
    this.orbit.target.copy(c);
    this.camera.position.set(c.x, c.y + 30, c.z + 320);
    this.orbit.update();
    // Drop the ground grid to the lower foot so it doesn't cut through the figure.
    const footY = Math.min(this.pos[31]?.[1] ?? 0, this.pos[32]?.[1] ?? 0);
    if (Number.isFinite(footY)) this.grid.position.y = footY;
  }

  _setLimb(mesh, a, b, radius) {
    const A = v3(a), B = v3(b);
    const dir = new THREE.Vector3().subVectors(B, A);
    const dist = dir.length() || 1e-6;
    mesh.position.copy(A).addScaledVector(dir, 0.5);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
    mesh.scale.set(radius, dist, radius);
  }

  _update() {
    if (!this.pos) return;
    const pos = this.pos;
    // Limbs
    BONES.forEach(([ka, kb], i) => {
      const a = pos[ka], b = pos[kb];
      if (!a || !b) return;
      const torso = typeof ka === 'string' && typeof kb === 'string';
      this._setLimb(this._bones[i], a, b, torso ? 7 : 5);
    });
    // Joint spheres
    for (const t of EDIT_TARGETS) {
      const mesh = this._jointMeshes.get(String(t.key));
      const p = pos[t.key];
      if (mesh && p) mesh.position.set(p[0], p[1], p[2]);
    }
    // Rigged skin
    if (this.rigGroup.visible && this.rigData) {
      this._ensureRigScale();
      poseRig(this.rigData, pos);
      // Co-locate the rig's hips with the control hips (rig is uniformly scaled).
      const s = this._rigScale ?? 1;
      const hb = this.rigData.hipsBindPos;
      this.rigGroup.position.set(
        pos.hips[0] - hb.x * s,
        pos.hips[1] - hb.y * s,
        pos.hips[2] - hb.z * s,
      );
    }
  }

  _pick(e) {
    if (this.transform.dragging) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this._pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this._pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this._pointer, this.camera);
    const hits = this.raycaster.intersectObjects([...this._jointMeshes.values()], false);
    if (hits.length) this.select(hits[0].object.userData.targetKey);
  }

  _refreshGizmoForSelection() {
    const key = this.selectedKey;
    if (!key) { this.transform.detach(); return; }
    const target = EDIT_TARGET_BY_KEY[key];
    if (!target) { this.transform.detach(); return; }
    // Rotate tool needs a non-empty subtree (wrists/feet have none → fall back to move).
    const canRotate = this.tool === 'rotate' && target.rotate.length > 0;
    const mode = (this.tool === 'move' || !canRotate) ? 'translate' : 'rotate';
    if (mode === 'translate' && !target.move && !target.root) { this.transform.detach(); return; }
    this._placeGizmo();
    this.transform.setMode(mode);
    this.transform.attach(this.gizmoPivot);
  }

  _placeGizmo() {
    const target = EDIT_TARGET_BY_KEY[this.selectedKey];
    if (!target || !this.pos) return;
    const p = this.pos[target.key];
    if (p) { this.gizmoPivot.position.set(p[0], p[1], p[2]); this.gizmoPivot.quaternion.identity(); }
  }

  _beginDrag() {
    // Snapshot the control positions so each gizmo update is computed from a fixed
    // start state (no cumulative float drift), and reset the pivot transform.
    this._dragSnapshot = {};
    for (const k of [...TRACKED_MP, 'hips', 'spine', 'chest', 'neck', 'head']) {
      if (this.pos[k]) this._dragSnapshot[k] = this.pos[k].slice();
    }
    this._dragPivot = this.gizmoPivot.position.clone();
    this.gizmoPivot.quaternion.identity();
  }

  _onGizmoChange() {
    if (!this._dragSnapshot || !this.selectedKey) return;
    const target = EDIT_TARGET_BY_KEY[this.selectedKey];
    if (!target) return;
    // Work from the snapshot each change.
    for (const k in this._dragSnapshot) this.pos[k] = this._dragSnapshot[k].slice();

    if (this.transform.getMode() === 'rotate') {
      const dq = arr(this.gizmoPivot.quaternion);
      const pivot = [this._dragPivot.x, this._dragPivot.y, this._dragPivot.z];
      rotateSubtree(this.pos, target, dq, pivot);
    } else {
      const np = [this.gizmoPivot.position.x, this.gizmoPivot.position.y, this.gizmoPivot.position.z];
      moveJoint(this.pos, target, np);
    }
    this._update();
  }

  _endDrag() {
    this._dragSnapshot = null;
    this._placeGizmo();           // re-zero the pivot at the new joint location
    if (this.pos) this.onEdit(this.pos);
  }

  _paintSelection() {
    for (const [key, mesh] of this._jointMeshes) {
      const sel = key === this.selectedKey;
      mesh.material.color.setHex(sel ? SEL_COL : JOINT_COL);
      mesh.material.emissive.setHex(sel ? SEL_COL : JOINT_COL);
      mesh.material.emissiveIntensity = sel ? 0.6 : 0.25;
    }
  }

  _animate() {
    this._raf = requestAnimationFrame(this._animate);
    this.orbit.update();
    this.renderer.render(this.scene, this.camera);
  }
}
