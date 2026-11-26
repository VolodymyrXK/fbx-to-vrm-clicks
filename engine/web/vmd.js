// VMD (MikuMikuDance motion) reader and player for the VRM preview.
// MMD bones have no rest orientation: a key is a rotation in parent-aligned world axes relative to the
// MMD rest pose (A-pose arms).  three-vrm normalized bones work the same way relative to the T-pose, so
// a key only needs (1) a change of axes MMD -> VRM and (2) a correction for the arm rest difference.
import * as THREE from 'three';

const sjis = new TextDecoder('shift-jis');
function str(dv, off, len) {
  const b = new Uint8Array(dv.buffer, dv.byteOffset + off, len);
  let n = b.indexOf(0); if (n < 0) n = len;
  return sjis.decode(b.subarray(0, n)).trim();
}

const fin = (x, d = 0) => (Number.isFinite(x) && Math.abs(x) < 1e6 ? x : d);

export function parseVMD(buf) {
  if (!buf || buf.byteLength < 54) throw new Error('Это не файл движения MMD (.vmd)');
  const dv = new DataView(buf);
  const head = str(dv, 0, 30);
  if (!head.startsWith('Vocaloid Motion Data')) throw new Error('Это не файл движения MMD (.vmd)');
  const nameLen = head.endsWith('0002') ? 20 : 10;
  let p = 30 + nameLen;
  const need = (n) => { if (p + n > dv.byteLength) throw new Error('Файл .vmd повреждён'); };
  need(4); const nb = dv.getUint32(p, true); p += 4;
  if (nb > 5e6 || p + nb * 111 > dv.byteLength) throw new Error('Файл .vmd повреждён (обрезан)');
  const bones = new Map();
  let last = 0;
  for (let i = 0; i < nb; i++) {
    need(111);
    const name = str(dv, p, 15), frame = dv.getUint32(p + 15, true);
    const pos = [fin(dv.getFloat32(p + 19, true)), fin(dv.getFloat32(p + 23, true)), fin(dv.getFloat32(p + 27, true))];
    let rot = [fin(dv.getFloat32(p + 31, true)), fin(dv.getFloat32(p + 35, true)), fin(dv.getFloat32(p + 39, true)), fin(dv.getFloat32(p + 43, true), 1)];
    const ln = Math.hypot(...rot);
    rot = ln > 1e-6 ? rot.map(x => x / ln) : [0, 0, 0, 1];
    if (frame > 1e6) throw new Error('Файл .vmd повреждён');
    const ip = new Uint8Array(buf, dv.byteOffset + p + 47, 64).slice(0, 16);
    p += 111;
    if (!bones.has(name)) bones.set(name, []);
    bones.get(name).push({ frame, pos, rot, ip });
    if (frame > last) last = frame;
  }
  const morphs = new Map();
  if (p + 4 <= dv.byteLength) {
    const nm = dv.getUint32(p, true); p += 4;
    if (p + nm * 23 > dv.byteLength) throw new Error('Файл .vmd повреждён (обрезан)');
    for (let i = 0; i < nm; i++) {
      const name = str(dv, p, 15), frame = dv.getUint32(p + 15, true), w = fin(dv.getFloat32(p + 19, true));
      if (frame > 1e6) throw new Error('Файл .vmd повреждён');
      p += 23;
      if (!morphs.has(name)) morphs.set(name, []);
      morphs.get(name).push({ frame, w });
      if (frame > last) last = frame;
    }
  }
  for (const l of bones.values()) l.sort((a, b) => a.frame - b.frame);
  for (const l of morphs.values()) l.sort((a, b) => a.frame - b.frame);
  return { bones, morphs, frames: last };
}

function bezier(x1, y1, x2, y2, x) {
  if (x1 === y1 && x2 === y2) return x;
  let lo = 0, hi = 1, t = x;
  for (let i = 0; i < 18; i++) {
    t = (lo + hi) / 2;
    const s = 1 - t, fx = 3 * s * s * t * x1 + 3 * s * t * t * x2 + t * t * t;
    if (fx < x) lo = t; else hi = t;
  }
  const s = 1 - t;
  return 3 * s * s * t * y1 + 3 * s * t * t * y2 + t * t * t;
}
function curve(ip, c, x) {
  return bezier(ip[c] / 127, ip[4 + c] / 127, ip[8 + c] / 127, ip[12 + c] / 127, x);
}

// standard MMD bone names -> VRM humanoid bones
const SIDE = { left: '左', right: '右' };
const FINGERS = [['Thumb', '親指', ['０', '１', '２']], ['Index', '人差指', ['１', '２', '３']],
  ['Middle', '中指', ['１', '２', '３']], ['Ring', '薬指', ['１', '２', '３']], ['Little', '小指', ['１', '２', '３']]];
const SEG0 = ['Proximal', 'Intermediate', 'Distal'];

const _q = new THREE.Quaternion(), _q2 = new THREE.Quaternion();
const _v = new THREE.Vector3(), _v2 = new THREE.Vector3(), _v3 = new THREE.Vector3();

export class VmdPlayer {
  constructor(motion, vrm, info) {
    this.m = motion; this.vrm = vrm; this.info = info;
    this.v0 = vrm.meta?.metaVersion === '0';
    this.scale = info.scale || 0.08;
    this.rest = {};
    for (const [k, q] of Object.entries(info.rest_fix || {})) this.rest[k] = new THREE.Quaternion(...q);
    const hum = vrm.humanoid;
    this.has = (n) => !!hum.getNormalizedBoneNode(n);
    const thumb1 = hum.humanBones.leftThumbMetacarpal ? ['Metacarpal', 'Proximal', 'Distal'] : SEG0;
    // name map
    this.map = [];
    const add = (vrmBone, ...jp) => { if (this.has(vrmBone)) this.map.push([vrmBone, jp]); };
    add('neck', '首'); add('head', '頭'); add('chest', '上半身2'); add('upperChest', '上半身3');
    for (const [s, j] of Object.entries(SIDE)) {
      add(s + 'Shoulder', j + '肩');
      add(s + 'UpperArm', j + '腕', j + '腕捩');
      add(s + 'LowerArm', j + 'ひじ', j + '手捩');
      add(s + 'Hand', j + '手首');
      add(s + 'Eye', j + '目', '両目');
      add(s + 'Toes', j + 'つま先');
      for (const [fn, jn, segs] of FINGERS) {
        const names = fn === 'Thumb' ? thumb1 : SEG0;
        segs.forEach((seg, i) => add(s + fn + names[i], j + jn + seg));
      }
    }
    this.ik = {};
    for (const [s, j] of Object.entries(SIDE)) {
      this.ik[s] = { keys: motion.bones.get(j + '足ＩＫ') || motion.bones.get(j + '足IK') };
      this.ik[s].fk = !!(motion.bones.get(j + '足') || []).length;
    }
    this.legs = [];
    for (const [s, j] of Object.entries(SIDE)) this.legs.push([s, j]);
    this.restCaptured = false;
  }
  sample(name, frame) {
    const l = this.m.bones.get(name);
    if (!l || !l.length) return null;
    if (frame <= l[0].frame) return { pos: l[0].pos, rot: l[0].rot };
    const last = l[l.length - 1];
    if (frame >= last.frame) return { pos: last.pos, rot: last.rot };
    let lo = 0, hi = l.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (l[mid].frame <= frame) lo = mid; else hi = mid; }
    const a = l[lo], b = l[hi], x = (frame - a.frame) / Math.max(1, b.frame - a.frame);
    const pos = [0, 1, 2].map(c => a.pos[c] + (b.pos[c] - a.pos[c]) * curve(b.ip, c, x));
    _q.set(...a.rot); _q2.set(...b.rot);
    const r = _q.clone().slerp(_q2, curve(b.ip, 3, x));
    return { pos, rot: [r.x, r.y, r.z, r.w] };
  }
  mmdQ(name, frame) {
    const s = this.sample(name, frame);
    return s ? new THREE.Quaternion(...s.rot) : new THREE.Quaternion();
  }
  // MMD axes -> axes of the normalized bones as three-vrm exposes them
  toVrmQ(q) {
    // MMD -> VRM1 glTF: mirror Z  => (-x, -y, z, w);  VRM0 model space is VRM1 turned 180 deg about Y
    return this.v0 ? new THREE.Quaternion(q.x, -q.y, -q.z, q.w) : new THREE.Quaternion(-q.x, -q.y, q.z, q.w);
  }
  toVrmV(p) {
    const s = this.scale;
    return this.v0 ? new THREE.Vector3(-p[0] * s, p[1] * s, p[2] * s) : new THREE.Vector3(p[0] * s, p[1] * s, -p[2] * s);
  }
  restOf(vrmBone) {
    const side = vrmBone.startsWith('left') ? 'left' : vrmBone.startsWith('right') ? 'right' : null;
    if (!side) return null;
    const rest = vrmBone.slice(side.length);
    if (rest === 'UpperArm') return this.rest[side + 'UpperArm'];
    if (rest === 'LowerArm' || rest === 'Hand' || /Thumb|Index|Middle|Ring|Little/.test(rest)) return this.rest[side + 'LowerArm'];
    return null;
  }
  parentRestOf(vrmBone) {
    const side = vrmBone.startsWith('left') ? 'left' : vrmBone.startsWith('right') ? 'right' : null;
    if (!side) return null;
    const rest = vrmBone.slice(side.length);
    if (rest === 'LowerArm') return this.rest[side + 'UpperArm'];
    if (rest === 'Hand' || /Thumb|Index|Middle|Ring|Little/.test(rest)) return this.rest[side + 'LowerArm'];
    return null;
  }
  captureRest() {
    const hum = this.vrm.humanoid, sc = this.vrm.scene;
    hum.resetNormalizedPose();
    sc.updateMatrixWorld(true);
    const P = (n) => { const o = hum.getNormalizedBoneNode(n); return o ? sc.worldToLocal(o.getWorldPosition(new THREE.Vector3())) : null; };
    this.restModel = {};
    for (const s of ['left', 'right']) for (const b of ['UpperLeg', 'LowerLeg', 'Foot']) this.restModel[s + b] = P(s + b);
    this.restCaptured = true;
  }
  apply(t) {
    const hum = this.vrm.humanoid; if (!hum) return;
    if (!this.restCaptured) this.captureRest();
    const frame = (t * 30) % Math.max(1, this.m.frames);
    hum.resetNormalizedPose();
    const B = (n) => hum.getNormalizedBoneNode(n);
    // root: 全ての親 * センター * グルーブ, then 下半身 = hips, 上半身 relative to it = spine
    const root = this.sample('全ての親', frame), cen = this.sample('センター', frame), grv = this.sample('グルーブ', frame);
    const qRoot = this.mmdQ('全ての親', frame).multiply(this.mmdQ('センター', frame)).multiply(this.mmdQ('グルーブ', frame));
    const qLower = this.mmdQ('下半身', frame), qUpper = this.mmdQ('上半身', frame);
    const off = [0, 1, 2].map(c => (root ? root.pos[c] : 0) + (cen ? cen.pos[c] : 0) + (grv ? grv.pos[c] : 0));
    const hips = B('hips');
    if (hips) {
      hips.quaternion.copy(this.toVrmQ(qRoot.clone().multiply(qLower)));
      hips.position.add(this.toVrmV(off));
    }
    const spine = B('spine');
    if (spine) spine.quaternion.copy(this.toVrmQ(qLower.clone().invert().multiply(qUpper)));
    for (const [vb, jp] of this.map) {
      let q = new THREE.Quaternion();
      let any = false;
      for (const n of jp) if (this.m.bones.has(n)) { q.multiply(this.mmdQ(n, frame)); any = true; }
      if (!any) continue;
      const O = this.restOf(vb), Op = this.parentRestOf(vb);
      if (O || Op) {
        const a = Op ? Op.clone().invert() : new THREE.Quaternion();
        q = a.multiply(q).multiply(O ? O.clone() : new THREE.Quaternion());
      }
      B(vb).quaternion.copy(this.toVrmQ(q));
    }
    // legs: FK keys if present, else two-bone IK toward 足ＩＫ
    for (const [s, j] of this.legs) {
      const ik = this.ik[s];
      if (ik.fk || !ik.keys) {
        for (const [vb, n] of [[s + 'UpperLeg', j + '足'], [s + 'LowerLeg', j + 'ひざ'], [s + 'Foot', j + '足首']])
          if (B(vb) && this.m.bones.has(n)) B(vb).quaternion.copy(this.toVrmQ(this.mmdQ(n, frame)));
        continue;
      }
      this.solveLeg(s, j, frame, root);
    }
    // expressions: every MMD morph is also a custom expression of the same name
    const em = this.vrm.expressionManager;
    if (em) for (const [name, l] of this.m.morphs) {
      if (!em.getExpression(name)) continue;
      let w = l[0].w;
      for (let i = 0; i < l.length; i++) {
        if (l[i].frame > frame) {
          const a = l[i - 1] || l[i], b = l[i];
          w = a.w + (b.w - a.w) * ((frame - a.frame) / Math.max(1, b.frame - a.frame));
          break;
        }
        w = l[i].w;
      }
      em.setValue(name, Math.max(0, Math.min(1, w)));
    }
  }
  solveLeg(s, j, frame, root) {
    const hum = this.vrm.humanoid, sc = this.vrm.scene;
    const ul = hum.getNormalizedBoneNode(s + 'UpperLeg'), ll = hum.getNormalizedBoneNode(s + 'LowerLeg'),
      ft = hum.getNormalizedBoneNode(s + 'Foot');
    if (!ul || !ll || !ft) return;
    const k = this.sample(j + '足ＩＫ', frame) || this.sample(j + '足IK', frame);
    const R = this.restModel;
    const ikOff = [0, 1, 2].map(c => (k ? k.pos[c] : 0) + (root ? root.pos[c] : 0));
    const target = sc.localToWorld(R[s + 'Foot'].clone().add(this.toVrmV(ikOff)));
    sc.updateMatrixWorld(true);
    const a = ul.getWorldPosition(new THREE.Vector3());
    const L1 = R[s + 'UpperLeg'].distanceTo(R[s + 'LowerLeg']) * sc.getWorldScale(_v3).y;
    const L2 = R[s + 'LowerLeg'].distanceTo(R[s + 'Foot']) * sc.getWorldScale(_v3).y;
    const dvec = target.clone().sub(a);
    let D = dvec.length();
    D = Math.min(Math.max(D, Math.abs(L1 - L2) + 1e-4), L1 + L2 - 1e-4);
    const dir = dvec.normalize();
    const hipsQ = hum.getNormalizedBoneNode('hips').getWorldQuaternion(new THREE.Quaternion());
    const fwd = new THREE.Vector3(0, 0, this.v0 ? -1 : 1).applyQuaternion(hipsQ);
    const n = fwd.clone().sub(dir.clone().multiplyScalar(fwd.dot(dir))).normalize();
    const alpha = Math.acos(Math.min(1, Math.max(-1, (L1 * L1 + D * D - L2 * L2) / (2 * L1 * D))));
    const knee = a.clone().add(dir.clone().multiplyScalar(L1 * Math.cos(alpha))).add(n.clone().multiplyScalar(L1 * Math.sin(alpha)));
    const ankle = a.clone().add(dir.clone().multiplyScalar(D));
    const sceneQ = sc.getWorldQuaternion(new THREE.Quaternion());
    const restDir = (from, to) => R[to].clone().sub(R[from]).applyQuaternion(sceneQ).normalize();
    const restFwd = new THREE.Vector3(0, 0, this.v0 ? -1 : 1).applyQuaternion(sceneQ);
    const frameQ = (d0, f0, d1, f1) => {
      const basis = (d, f) => {
        const x = d.clone().normalize(), z = f.clone().sub(x.clone().multiplyScalar(f.dot(x))).normalize();
        const y = z.clone().cross(x);
        return new THREE.Matrix4().makeBasis(x, y, z);
      };
      const m0 = basis(d0, f0), m1 = basis(d1, f1);
      return new THREE.Quaternion().setFromRotationMatrix(m1.multiply(m0.transpose()));
    };
    const setWorld = (node, qWorld) => {
      const pq = node.parent.getWorldQuaternion(new THREE.Quaternion());
      node.quaternion.copy(pq.invert().multiply(qWorld));
      node.updateMatrixWorld(true);
    };
    setWorld(ul, frameQ(restDir(s + 'UpperLeg', s + 'LowerLeg'), restFwd, knee.clone().sub(a), n).multiply(sceneQ));
    setWorld(ll, frameQ(restDir(s + 'LowerLeg', s + 'Foot'), restFwd, ankle.clone().sub(knee), n).multiply(sceneQ));
    if (k) {
      const qik = this.mmdQ(j + '足ＩＫ', frame);
      setWorld(ft, sceneQ.clone().multiply(this.toVrmQ(qik)));
    }
  }
}
