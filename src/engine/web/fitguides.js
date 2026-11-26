// Fitting guides for the preview: soft colour markers on the chest of the base body
// (left / right apex points + centre line), like the marks on a tailor's dress form.
// They help to line up a bra / swimsuit top: both sides at the same height and the
// same distance from the centre. Preview only: never written into the VRM file.
import * as THREE from 'three';

export const FIT_DEFAULTS = () => ({
  on: false,          // show the guides
  color: '#ff3d8b',   // marker colour
  size: 2.4,          // marker diameter, cm
  soft: 0.6,          // 0 = hard edge, 1 = fully soft (gaussian-like)
  opacity: 0.85,
  height: 0,          // shift up (+) / down (−), cm, relative to the automatic height
  spread: 1,          // distance between the two side markers, ×automatic
  center: true,       // centre marker + level line + centre axis
  onTop: true,        // visible through clothes (otherwise hidden by them like a real mark)
  // lower centre line: a thin soft symmetry axis over the briefs / underwear area
  lower: false,
  lowerColor: '#3dc8ff',
  lowerWidth: 3,      // line width, mm
  lowerSoft: 0.6,     // 0 = hard edge, 1 = fully soft
  lowerLength: 1,     // length, ×automatic (waistband → crotch)
});

const STORE_KEY = 'mmd2vrm_fitguides';
const BODY_CATS = new Set(['skin', 'body']);

export function loadFitOptions() {
  try { return { ...FIT_DEFAULTS(), ...(JSON.parse(localStorage.getItem(STORE_KEY) || '{}')) }; }
  catch (e) { return FIT_DEFAULTS(); }
}
export function saveFitOptions(o) { try { localStorage.setItem(STORE_KEY, JSON.stringify(o)); } catch (e) {} }

// soft round marker: full colour in the core, smooth fall-off to the edge
function makeMarkerTexture(soft) {
  const S = 128, c = document.createElement('canvas'); c.width = c.height = S;
  const g = c.getContext('2d'), r = S / 2;
  const core = Math.max(0, Math.min(0.95, 1 - soft));
  const grad = g.createRadialGradient(r, r, 0, r, r, r);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(core, 'rgba(255,255,255,1)');
  // ease-out tail so a soft marker still has a readable middle
  for (let i = 1; i <= 6; i++) {
    const t = i / 6, pos = core + (1 - core) * t;
    grad.addColorStop(Math.min(1, pos), `rgba(255,255,255,${(1 - t) * (1 - t)})`);
  }
  g.fillStyle = grad; g.fillRect(0, 0, S, S);
  // a tiny crisp dot in the middle: the exact point to align a seam / cup centre to
  g.fillStyle = 'rgba(255,255,255,1)'; g.beginPath(); g.arc(r, r, S * 0.035, 0, Math.PI * 2); g.fill();
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
// centre marker: a ring, so it is not mistaken for a side marker
function makeRingTexture(soft) {
  const S = 128, c = document.createElement('canvas'); c.width = c.height = S;
  const g = c.getContext('2d'), r = S / 2;
  const blur = 2 + soft * 10;
  g.filter = `blur(${blur}px)`;
  g.strokeStyle = 'rgba(255,255,255,1)'; g.lineWidth = S * 0.12;
  g.beginPath(); g.arc(r, r, r * 0.55, 0, Math.PI * 2); g.stroke();
  g.filter = 'none'; g.fillStyle = 'rgba(255,255,255,1)';
  g.beginPath(); g.arc(r, r, S * 0.035, 0, Math.PI * 2); g.fill();
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

// soft vertical strip: full colour on the axis, smooth fall-off to the sides and the ends
function makeStripTexture(soft) {
  const Wd = 64, Ht = 256, c = document.createElement('canvas'); c.width = Wd; c.height = Ht;
  const g = c.getContext('2d'), img = g.createImageData(Wd, Ht);
  const core = Math.max(0.02, 1 - soft);
  for (let y = 0; y < Ht; y++) for (let x = 0; x < Wd; x++) {
    const u = Math.abs((x + 0.5) / Wd * 2 - 1), v = Math.abs((y + 0.5) / Ht * 2 - 1);
    const a = u <= core ? 1 : Math.max(0, 1 - (u - core) / (1 - core + 1e-6)) ** 2;
    const e = v < 0.85 ? 1 : Math.max(0, 1 - (v - 0.85) / 0.15);   // soft ends
    const i = (y * Wd + x) * 4; img.data[i] = img.data[i + 1] = img.data[i + 2] = 255; img.data[i + 3] = Math.round(255 * a * e);
  }
  g.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

export class FitGuides {
  constructor(scene) {
    this.root = new THREE.Group(); this.root.name = 'fitGuides'; this.root.renderOrder = 9000;
    scene.add(this.root);
    this.opts = loadFitOptions();
    this.anchors = [];   // { obj, bone, local: Vector3, quat: Quaternion }
    this.vrm = null; this.bodyMeshes = []; this.innerMeshes = [];
    this.info = null;    // last placement (for tests / status line)
  }
  set(patch) { Object.assign(this.opts, patch); saveFitOptions(this.opts); this.rebuild(); }
  reset() { this.opts = FIT_DEFAULTS(); this.opts.on = true; saveFitOptions(this.opts); this.rebuild(); }

  // vrm: loaded VRM; bodyMeshes: meshes of the base body (skin). Empty → all meshes.
  // innerMeshes: underwear / swimsuit — used where the skin mesh is cut out under them
  attach(vrm, bodyMeshes, innerMeshes) { this.vrm = vrm; this.bodyMeshes = bodyMeshes || []; this.innerMeshes = innerMeshes || []; this.rebuild(); }
  detach() { this.vrm = null; this.clear(); }

  clear() {
    for (const a of this.anchors) { a.obj.geometry?.dispose(); a.obj.material?.map?.dispose(); a.obj.material?.dispose(); }
    this.root.clear(); this.anchors = []; this.info = null;
  }

  rebuild() {
    this.clear();
    if (!this.vrm || !(this.opts.on || this.opts.lower)) return;
    this.info = {};
    if (this.opts.on) { const p = this.place(); if (p) { this.info.chest = p; this.buildChest(p); } }
    if (this.opts.lower) { const q = this.placeLower(); if (q) { this.info.lower = q; this.buildLower(q); } }
    this.update();
  }

  buildLower(q) {
    const o = this.opts, w = o.lowerWidth / 1000, pts = q.points;
    // ribbon that follows the body surface along the centre line
    const pos = [], uv = [], idx = [];
    pts.forEach((pt, i) => {
      const l = q.bone.worldToLocal(pt.clone().add(new THREE.Vector3(-w / 2, 0, 0)));
      const r = q.bone.worldToLocal(pt.clone().add(new THREE.Vector3(w / 2, 0, 0)));
      pos.push(l.x, l.y, l.z, r.x, r.y, r.z);
      const v = 1 - i / (pts.length - 1); uv.push(0, v, 1, v);
      if (i) { const a = (i - 1) * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2)); geo.setIndex(idx);
    const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: makeStripTexture(o.lowerSoft), color: new THREE.Color(o.lowerColor),
      transparent: true, opacity: o.opacity, depthTest: !o.onTop, depthWrite: false, toneMapped: false, side: THREE.DoubleSide,
      polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4 }));
    m.renderOrder = 9000; this.root.add(m); this.anchors.push({ obj: m, bone: q.bone, line: true });
  }

  buildChest(p) {
    const o = this.opts, col = new THREE.Color(o.color);
    const mat = (tex) => new THREE.MeshBasicMaterial({ map: tex, color: col, transparent: true, opacity: o.opacity,
      depthTest: !o.onTop, depthWrite: false, toneMapped: false, side: THREE.DoubleSide,
      polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4 });
    const d = o.size / 100;
    const add = (obj, world, size) => {
      obj.renderOrder = 9000; if (size) obj.scale.setScalar(size);
      const local = p.bone.worldToLocal(world.clone());
      const q = p.bone.getWorldQuaternion(new THREE.Quaternion()).invert();
      this.root.add(obj); this.anchors.push({ obj, bone: p.bone, local, quat: q });
    };
    for (const w of [p.left, p.right]) add(new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat(makeMarkerTexture(o.soft))), w, d);
    if (o.center) {
      add(new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat(makeRingTexture(o.soft))), p.center, d * 0.8);
      const lm = () => new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: o.opacity * 0.55, depthTest: !o.onTop, depthWrite: false, toneMapped: false });
      // level line through both markers (shows if one side sits higher) and the centre axis
      const lvl = new THREE.Line(new THREE.BufferGeometry().setFromPoints([p.left, p.center, p.right].map(v => p.bone.worldToLocal(v.clone()))), lm());
      const ax = p.torso * 0.16;
      const axis = new THREE.Line(new THREE.BufferGeometry().setFromPoints(
        [p.center.clone().add(new THREE.Vector3(0, ax, 0)), p.center.clone().add(new THREE.Vector3(0, -ax, 0))].map(v => p.bone.worldToLocal(v.clone()))), lm());
      for (const l of [lvl, axis]) { l.renderOrder = 8999; this.root.add(l); this.anchors.push({ obj: l, bone: p.bone, line: true }); }
    }
  }

  // lower centre line: from just above the hips joint down to where the legs split,
  // sampled along the midline and snapped onto the body surface
  placeLower() {
    const hum = this.vrm.humanoid; if (!hum) return null;
    const raw = (n) => hum.getRawBoneNode?.(n) || null;
    const hips = raw('hips'), ll = raw('leftUpperLeg'), rl = raw('rightUpperLeg'), la = raw('leftUpperArm'), ra = raw('rightUpperArm');
    if (!hips || !ll || !rl) return null;
    this.vrm.scene.updateMatrixWorld(true);
    const W = (n) => n.getWorldPosition(new THREE.Vector3());
    const H = W(hips), LL = W(ll), RL = W(rl);
    const torso = la && ra ? Math.max(0.05, (W(la).y + W(ra).y) / 2 - H.y) : Math.abs(LL.x - RL.x) * 2.5;
    const cx = (LL.x + RL.x) / 2, legY = (LL.y + RL.y) / 2;
    // waistband ≈ halfway between the hips joint and the leg joints; down to just below the leg joints
    const top = legY + Math.max(0.5 * (H.y - legY), torso * 0.12);
    const len = (top - (legY - torso * 0.25)) * this.opts.lowerLength;
    const meshes = this.surfaceMeshes(), N = 24, pts = [];
    for (let i = 0; i <= N; i++) {
      const y = top - len * i / N, z = this.surfaceZ(meshes, cx, y);
      if (z === null) { if (pts.length >= 3) break; continue; }   // stops where the legs split
      pts.push(new THREE.Vector3(cx, y, z + 0.002));
    }
    if (pts.length < 2) {   // no body surface: straight line in front of the hips
      const z = H.z + torso * 0.15; pts.length = 0;
      pts.push(new THREE.Vector3(cx, top, z), new THREE.Vector3(cx, top - len, z));
    }
    return { bone: hips, points: pts, snapped: meshes.length > 0 };
  }

  // follow the chest bone while the model moves (idle / walk / VMD)
  update() {
    if (!this.anchors.length) return;
    const bq = new THREE.Quaternion(), bs = new THREE.Vector3(), bp = new THREE.Vector3();
    for (const a of this.anchors) {
      a.bone.matrixWorld.decompose(bp, bq, bs);
      if (a.line) { a.obj.matrixAutoUpdate = false; a.obj.matrix.copy(a.bone.matrixWorld); a.obj.matrixWorldNeedsUpdate = true; continue; }
      a.obj.position.copy(a.local).applyMatrix4(a.bone.matrixWorld);
      a.obj.quaternion.copy(bq).multiply(a.quat);   // keeps the rest-pose facing (toward the viewer)
    }
  }

  // automatic placement from the skeleton, then snapped onto the body surface
  place() {
    const hum = this.vrm.humanoid; if (!hum) return null;
    const raw = (n) => hum.getRawBoneNode?.(n) || null;
    const bone = raw('upperChest') || raw('chest') || raw('spine');
    const la = raw('leftUpperArm'), ra = raw('rightUpperArm'), hips = raw('hips');
    if (!bone || !la || !ra || !hips) return null;
    this.vrm.scene.updateMatrixWorld(true);
    const W = (n) => n.getWorldPosition(new THREE.Vector3());
    const L = W(la), R = W(ra), H = W(hips), C = W(bone);
    const shoulderY = (L.y + R.y) / 2, torso = Math.max(0.05, shoulderY - H.y);
    const cx = (L.x + R.x) / 2, halfW = Math.abs(L.x - R.x) / 2;
    // ≈ 0.55 of half the shoulder span from the centre (dress-form proportions)
    const dx = halfW * 0.55 * this.opts.spread;
    // VRM looks toward +Z (VRM 0.x is turned by VRMUtils.rotateVRM0)
    const sx = Math.sign(L.x - R.x) || 1;
    const meshes = this.surfaceMeshes();
    // height: the most forward point of the body in the chest band, searched on both sides
    // (like the apex mark on a dress form); without a surface — 27% of shoulder→hip below the shoulders
    let best = null;
    for (let i = 0; i <= 32; i++) {
      const yy = shoulderY - torso * (0.10 + 0.45 * i / 32);
      const zl = this.surfaceZ(meshes, cx + sx * dx, yy), zr = this.surfaceZ(meshes, cx - sx * dx, yy);
      if (zl === null || zr === null) continue;
      const z = (zl + zr) / 2; if (!best || z > best.z + 1e-4) best = { y: yy, z };
    }
    const y = (best ? best.y : shoulderY - torso * 0.27) + this.opts.height / 100;
    const front = (x) => this.surfaceZ(meshes, x, y) ?? (C.z + torso * 0.12);
    const off = 0.002;
    return {
      bone, torso,
      left: new THREE.Vector3(cx + sx * dx, y, front(cx + sx * dx) + off),
      right: new THREE.Vector3(cx - sx * dx, y, front(cx - sx * dx) + off),
      center: new THREE.Vector3(cx, y, front(cx) + off),
      snapped: meshes.length > 0,
    };
  }

  surfaceMeshes() {
    let ms = this.bodyMeshes.filter(m => m.isMesh);
    if (!ms.length) { ms = []; this.vrm.scene.traverse(o => { if (o.isMesh && o.visible) ms.push(o); }); }
    // skinned meshes are ray-cast with their bone matrices: make sure those are current
    for (const m of [...ms, ...this.innerMeshes]) if (m.isSkinnedMesh) m.skeleton.update();
    this._box = new THREE.Box3().setFromObject(this.vrm.scene);
    return ms;
  }
  surfaceZ(meshes, x, y) {
    if (!meshes.length) return null;
    const rc = new THREE.Raycaster(new THREE.Vector3(x, y, this._box.max.z + 1), new THREE.Vector3(0, 0, -1));
    let hit = rc.intersectObjects(meshes, false)[0];
    // skin is often cut out under underwear: then the underwear surface is the body line
    if (!hit && this.innerMeshes.length) hit = rc.intersectObjects(this.innerMeshes, false)[0];
    return hit ? hit.point.z : null;
  }
}
