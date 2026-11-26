import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMSpringBoneLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { parseVMD, VmdPlayer } from './vmd.js';
import { FitGuides } from './fitguides.js';

// ------------------------------------------------------------------ state
const DEFAULTS = () => ({
  scale: 0.08, tpose: true, spring: true, outline: false, custom_blendshapes: true,
  max_texture: 2048, physics: 'soft', meta: { title: '', author: '' },
  humanoid: {}, materials: {}, chains: {}, presets: {}, renames: {}, splits: {}, body_fill: [], wardrobe: [], ru_names: false, bone_lang: 'orig', tex_quality: 'normal', tex_sharpen: 0, upscale: false, repair: false, anime_face: true, target_mb: 0, strip_hidden: false,
});
const app = {
  settings: DEFAULTS(), info: null, vrm: null, gltf: null, nodes: [],
  matMeshes: new Map(), chainJoints: new Map(), tab: 'parts', sel: null,
  anim: 'idle', busy: false, loadedVersion: -1,
};
window.__app = app;
window.__open = (p) => openPath(p);
window.__open2 = (p, m) => openPath(p, m);
window.__rebuild = () => rebuild(0);
window.__waitIdle = () => waitIdle();

const $ = (s, el = document) => el.querySelector(s);
const h = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (k === 'checked' || k === 'disabled' || k === 'value' || k === 'selected') e[k] = v;
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false)
    e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
};

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({ error: 'HTTP ' + r.status }));
  if (!r.ok || j.error) throw new Error(j.error || ('HTTP ' + r.status));
  return j;
}
function toast(msg, err = false, ms = 4500) {
  let box = document.getElementById('toasts');
  if (!box) { box = h('div', { id: 'toasts' }); document.body.append(box); }
  const t = h('div', { class: 'toast' + (err ? ' err' : '') }, msg);
  box.append(t); while (box.children.length > 4) box.firstChild.remove();
  setTimeout(() => t.remove(), ms);
}
function localStorageGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
function localStorageSet(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
function setBusy(on, text = 'Конвертация…', light = false) {
  app.busy = on;
  if (light) {  // rebuilds: keep the model visible and turnable, show a small badge
    $('#busy').hidden = true;
    const b = $('#rebuildBadge');
    clearInterval(app.badgeTimer);
    if (on) { const t0 = Date.now(); b.hidden = false; b.textContent = text;
      app.badgeTimer = setInterval(() => { b.textContent = text + ' ' + ((Date.now() - t0) / 1000).toFixed(0) + ' с'; }, 500); }
    else b.hidden = true;
    return;
  }
  $('#busy').hidden = !on; $('#busyText').textContent = text;
}
setInterval(() => api('/api/ping', {}).catch(() => {}), 3000);

// ------------------------------------------------------------------ three
const canvas = $('#c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x202330);
const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 100);
camera.position.set(0, 1.2, 4);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0.9, 0);
controls.update();
app.camera = camera; app.controls = controls;
scene.add(new THREE.HemisphereLight(0xffffff, 0x445066, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 2.2); sun.position.set(1, 2, 2.5); scene.add(sun);
const grid = new THREE.GridHelper(4, 16, 0x3b4152, 0x2b2f3b); scene.add(grid);
const helperRoot = new THREE.Group(); helperRoot.renderOrder = 10000; scene.add(helperRoot);
const jointHelpers = new THREE.Group(); const colliderHelpers = new THREE.Group();
helperRoot.add(jointHelpers, colliderHelpers);
jointHelpers.visible = false; colliderHelpers.visible = false;
const selRoot = new THREE.Group(); scene.add(selRoot);
const pivot = new THREE.Group(); scene.add(pivot);  // spins the model in 'spin' test
const fitGuides = new FitGuides(scene); app.fitGuides = fitGuides;  // clothing fitting guides: chest marks + lower centre line (preview only)

function resize() {
  const v = $('#view'); const w = v.clientWidth, hh = v.clientHeight;
  renderer.setSize(w, hh, false); camera.aspect = w / Math.max(1, hh); camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const dt = Math.min(clock.getDelta(), 1 / 20);
  const t = clock.elapsedTime;
  if (app.vrm) { pose(app.vrm, t, dt); app.vrm.update(dt); updateSelection(t); if (app.tab === 'bones') updateSkeleton(); fitGuides.update(); }
  updateBALabels();
  if (app.orig && app.showOrig) {  // "before": the same motion
    origRoot.rotation.y = pivot.rotation.y;
    pose(app.orig, t, dt);
    app.orig.update(dt);
  }
  controls.update();
  renderer.render(scene, camera);
});

// ------------------------------------------------------------------ test animations
function pose(vrm, t, dt) {
  const hum = vrm.humanoid; if (!hum) return;
  hum.resetNormalizedPose();
  const b = (n) => hum.getNormalizedBoneNode(n);
  const mode = app.anim;
  if (vrm === app.vrm) pivot.rotation.y = mode === 'spin' ? (pivot.rotation.y + dt * 2.5) : 0;
  if (mode === 'tpose') return;
  if (mode === 'vmd') {
    const pl = vrm === app.vrm ? app.player : app.origPlayer;
    if (vrm !== app.vrm) { if (pl) try { pl.apply(t - app.vmdStart); } catch (e) {} return; }
    if (app.player) {
      try { app.player.apply(t - app.vmdStart); } catch (e) { console.error(e); app.anim = 'idle'; hum.resetNormalizedPose(); vrm.springBoneManager?.reset(); }
    }
    return;
  }
  if (mode.startsWith('b:')) {
    const a = ANIMS.byId[mode.slice(2)];
    if (a) poseBuiltin(b, a, app.animFreeze != null ? app.animFreeze : t - (app.animStart || 0), vrm.meta?.metaVersion === '0');
  } else poseBody(b, mode, t);
  // three-vrm keeps VRM 0.x normalized bones in VRM0 axes: mirror X/Z rotations
  if (vrm.meta?.metaVersion === '0') {
    for (const name of Object.keys(hum.humanBones)) {
      const n = hum.getNormalizedBoneNode(name);
      if (n) { n.rotation.x *= -1; n.rotation.z *= -1; }
    }
  }
}
// built-in animations (anims.json): eased key poses, Euler XYZ degrees on VRM 1.0 normalized bones
const ANIMS = { list: [], byId: {} };
fetch('anims.json').then(r => r.json()).then(d => {
  ANIMS.list = d.anims; for (const a of d.anims) ANIMS.byId[a.id] = a; buildAnimMenu(); if (app.info) renderPanel();
}).catch(e => console.error('anims.json', e));
const easeK = (x) => x * x * (3 - 2 * x);
const _qa = new THREE.Quaternion(), _qb = new THREE.Quaternion();
// keys: [[t, ...values]]; returns {i, u}: segment and eased fraction at time t
function keySeg(keys, t) {
  if (t <= keys[0][0]) return { i: 0, u: 0 };
  for (let i = 1; i < keys.length; i++) if (t <= keys[i][0])
    return { i: i - 1, u: easeK((t - keys[i - 1][0]) / Math.max(1e-6, keys[i][0] - keys[i - 1][0])) };
  return { i: keys.length - 1, u: 0 };
}
function poseBuiltin(b, a, t, vrm0) {
  const T = a.loop ? (t % a.duration) : Math.min(t, a.duration);
  for (const [bone, keys] of Object.entries(a.tracks)) {
    const n = b(bone); if (!n) continue;
    const { i, u } = keySeg(keys, T), k0 = keys[i], k1 = keys[Math.min(i + 1, keys.length - 1)];
    _qa.set(k0[1], k0[2], k0[3], k0[4]); _qb.set(k1[1], k1[2], k1[3], k1[4]);
    n.quaternion.copy(_qa).slerp(_qb, u);
  }
  const hk = a.hips_pos, { i, u } = keySeg(hk, T), h0 = hk[i], h1 = hk[Math.min(i + 1, hk.length - 1)], hips = b('hips');
  const hp = [1, 2, 3].map(k => h0[k] + (h1[k] - h0[k]) * u);
  if (hips) { hips.position.x += vrm0 ? -hp[0] : hp[0]; hips.position.y += hp[1]; hips.position.z += vrm0 ? -hp[2] : hp[2]; }
}
function buildAnimMenu() {
  const sel = $('#animSel'); if (!sel) return;
  sel.innerHTML = ''; sel.append(h('option', { value: '' }, '📚 Ещё движения (' + ANIMS.list.length + ')…'));
  const groups = {};
  for (const a of ANIMS.list) (groups[a.group] ||= []).push(a);
  for (const [g, as] of Object.entries(groups)) sel.append(h('optgroup', { label: g }, ...as.map(a => h('option', { value: a.id }, a.name))));
}
function playBuiltin(id) {
  if (!ANIMS.byId[id]) return;
  app.anim = 'b:' + id; app.animStart = clock.elapsedTime; pivot.rotation.y = 0;
  document.querySelectorAll('#anim button').forEach(x => x.classList.remove('on'));
  const sel = $('#animSel'); if (sel) { sel.value = id; sel.classList.add('on'); }
}
window.__playAnim = playBuiltin;
function poseBody(b, mode, t) {
  const arms = (down) => {
    b('leftUpperArm') && (b('leftUpperArm').rotation.z = -down);
    b('rightUpperArm') && (b('rightUpperArm').rotation.z = down);
    b('leftLowerArm') && (b('leftLowerArm').rotation.y = 0.15);
    b('rightLowerArm') && (b('rightLowerArm').rotation.y = -0.15);
  };
  arms(1.25);
  const s = Math.sin, hips = b('hips'), spine = b('spine');
  if (mode === 'idle') {
    if (spine) spine.rotation.x = 0.02 * s(t * 1.6);
    if (b('head')) b('head').rotation.y = 0.15 * s(t * 0.5);
  } else if (mode === 'sway') {
    if (hips) { hips.rotation.y = 0.7 * s(t * 2.2); hips.rotation.z = 0.12 * s(t * 2.2 + 1); }
    if (spine) spine.rotation.y = 0.25 * s(t * 2.2 + 0.6);
  } else if (mode === 'walk') {
    const p = t * 6;
    for (const [side, k] of [['left', 1], ['right', -1]]) {
      const ul = b(side + 'UpperLeg'), ll = b(side + 'LowerLeg'), ua = b(side + 'UpperArm');
      if (ul) ul.rotation.x = 0.55 * s(p) * k;
      if (ll) ll.rotation.x = Math.max(0, 0.8 * s(p + (k > 0 ? 0 : Math.PI) - 0.8));
      if (ua) ua.rotation.x = -0.4 * s(p) * k;
    }
    if (hips) { hips.position.y += 0.03 * Math.abs(s(p)); hips.rotation.y = 0.12 * s(p); }
  } else if (mode === 'jump') {
    const ph = (t * 1.6) % 1, up = Math.max(0, s(ph * Math.PI));
    if (hips) hips.position.y += 0.35 * up;
    for (const side of ['left', 'right']) {
      const ul = b(side + 'UpperLeg'), ll = b(side + 'LowerLeg');
      const crouch = 1 - up;
      if (ul) ul.rotation.x = -0.5 * crouch; if (ll) ll.rotation.x = 0.9 * crouch;
    }
    arms(1.25 - 0.9 * up);
  } else if (mode === 'spin') {
    if (spine) spine.rotation.x = 0.05;
  }
}
for (const btn of document.querySelectorAll('#anim button[data-anim]')) btn.onclick = () => {
  app.anim = btn.dataset.anim;
  document.querySelectorAll('#anim button').forEach(x => x.classList.toggle('on', x === btn));
  const sel = $('#animSel'); if (sel) { sel.value = ''; sel.classList.remove('on'); }
  pivot.rotation.y = 0;
};
$('#animSel').onchange = (e) => { if (e.target.value) playBuiltin(e.target.value); };
$('#btnResetPhys').onclick = () => { app.vrm?.springBoneManager?.reset(); app.orig?.springBoneManager?.reset(); };
$('#showJoints').onchange = (e) => jointHelpers.visible = e.target.checked;
$('#showColliders').onchange = (e) => colliderHelpers.visible = e.target.checked;

// ------------------------------------------------------------------ loading
async function loadVRM(fit) {
  const buf = await (await fetch('/api/vrm?v=' + app.info.version)).arrayBuffer();
  jointHelpers.clear(); colliderHelpers.clear();
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser, {
    springBonePlugin: new VRMSpringBoneLoaderPlugin(parser, {
      jointHelperRoot: jointHelpers, colliderHelperRoot: colliderHelpers }) }));
  const gltf = await new Promise((res, rej) => loader.parse(buf, '', res, rej));
  const vrm = gltf.userData.vrm;
  if (app.vrm) { pivot.remove(app.vrm.scene); VRMUtils.deepDispose(app.vrm.scene); }
  VRMUtils.rotateVRM0(vrm);
  vrm.scene.traverse(o => { o.frustumCulled = false; });
  pivot.add(vrm.scene);
  app.vrm = vrm; app.gltf = gltf; app.loadedVersion = app.info.version;
  // node index -> object
  const nNodes = gltf.parser.json.nodes.length;
  app.nodes = await Promise.all([...Array(nNodes).keys()].map(i => gltf.parser.getDependency('node', i)));
  // materials -> meshes
  app.matMeshes = new Map();
  vrm.scene.traverse(o => {
    if (!o.isMesh) return;
    const m0 = Array.isArray(o.material) ? o.material[0] : o.material;
    let idx = gltf.parser.associations.get(m0)?.materials;
    if (idx === undefined) idx = (gltf.parser.json.materials || []).findIndex(m => m.name === m0.name);
    if (idx === undefined || idx < 0) return;
    if (!app.matMeshes.has(idx)) app.matMeshes.set(idx, []);
    app.matMeshes.get(idx).push(o);
  });
  fitT.meas = null;
  if (fitT.clear) computeClearance();
  applyFitOverlay();
  // chains -> joints
  const jointByBone = new Map();
  for (const j of vrm.springBoneManager?.joints || []) jointByBone.set(j.bone, j);
  app.chainJoints = new Map();
  for (const c of app.info.chains || []) {
    const js = c.nodes.map(i => jointByBone.get(app.nodes[i])).filter(Boolean);
    app.chainJoints.set(c.id, js);
    for (const j of js) j.userData = { ...(j.userData || {}), origColliders: j.colliderGroups };
  }
  // VRM 0.x: a branch (pendant on a tail) swings with its host chain in the file — the preview does the same
  if (app.settings.vrm_version !== 1) for (const c of app.info.chains || []) {
    if (!c.branch_of || !app.chainJoints.has(c.branch_of)) continue;
    app.chainJoints.get(c.branch_of).push(...app.chainJoints.get(c.id)); app.chainJoints.set(c.id, []);
  }
  const aniso = renderer.capabilities.getMaxAnisotropy();
  vrm.scene.traverse(o => { if (!o.isMesh) return; for (const m of [].concat(o.material)) for (const k of ['map', 'shadeMultiplyTexture']) {
    const t = m[k]; if (t && t.anisotropy !== aniso) { t.anisotropy = aniso; t.needsUpdate = true; } } });
  vrm.springBoneManager?.reset();
  if (app.motion) { app.player = new VmdPlayer(app.motion, vrm, app.info); }
  if (app.showOrig && app.origSrc !== app.info.src) loadOriginal(false);
  applyLiveVisibility();
  attachFitGuides();
  if (fit) fitCamera();
  $('#empty').style.display = 'none';
  if (app.tab === 'bones') { setSeeThrough(true); collectSkeleton(); }
  if (fit && app.tab === 'expr') focusFace();
}
function fitCamera() {
  const box = new THREE.Box3().setFromObject(app.vrm.scene);
  const size = box.getSize(new THREE.Vector3()), c = box.getCenter(new THREE.Vector3());
  const d = size.y / 2 / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.15;
  controls.target.copy(c); camera.position.set(c.x, c.y + size.y * 0.05, c.z + d);
  camera.near = d / 100; camera.far = d * 20; camera.updateProjectionMatrix();
}

async function applyInfo(info, fit = false) {
  app.info = info;
  $('#btnSave').disabled = !info.has_vrm;
  const m = info.model || {};
  $('#status').textContent = info.has_vrm
    ? `${m.name || ''} · ${m.vertices} вершин · ${info.size_mb} МБ`
    : `${m.name || ''} · не хватает костей: ${info.missing.join(', ')}`;
  if (!app.settings.meta.title && m.name) app.settings.meta.title = '';
  if (info.has_vrm && info.version !== app.loadedVersion) {
    try { await loadVRM(fit); } catch (e) { console.error(e); toast('Ошибка загрузки превью: ' + e.message, true); }
  } else if (!info.has_vrm && app.vrm) {
    pivot.remove(app.vrm.scene); app.vrm = null; fitGuides.detach(); $('#empty').style.display = '';
  }
  if (!info.has_vrm) { app.tab = 'bones'; }
  renderTabs(); renderPanel(); updateChip();
  if (info.has_vrm) { skelRoot.visible = app.tab === 'bones'; legend.hidden = app.tab !== 'bones'; }
}

async function openPath(path, member) {
  setBusy(true, 'Чтение и конвертация модели…');
  try {
    app.settings = { ...DEFAULTS(), physics: app.settings.physics, max_texture: app.settings.max_texture, tex_quality: app.settings.tex_quality, tex_sharpen: app.settings.tex_sharpen,
                     bone_lang: app.settings.bone_lang,
                     outline: app.settings.outline, scale: app.settings.scale, vrm_version: app.settings.vrm_version || 0 };
    app.sel = null; app.loadedVersion = -1;
    const info = await api('/api/open', { path, member, settings: app.settings, use_saved: true });
    if (info.saved_settings) {
      const d = DEFAULTS();
      app.settings = { ...d, ...info.saved_settings, meta: { ...d.meta, ...(info.saved_settings.meta || {}) } };
      toast('Загружены ваши прошлые правки для этой модели');
    }
    app.dirty = false; app.savedAt = null; history.length = 0; app.committed = JSON.stringify(app.settings);
    await applyInfo(info, true);
    if (!localStorageGet('mmd2vrm_seen')) { toast('Проверьте вкладки Части → Физика → Лицо, затем «Сохранить VRM». Кружок на вкладке показывает, где есть что проверить.', false, 9000); localStorageSet('mmd2vrm_seen', '1'); }
  } catch (e) { toast(e.message, true, 8000); }
  setBusy(false);
}
let rebuildTimer = null, rebuildRunning = false, rebuildAgain = false;
const history = [];
function snapshot() {
  // remember the state BEFORE the edit that is about to be applied
  const cur = JSON.stringify(app.settings);
  if (app.committed && app.committed !== cur && history[history.length - 1] !== app.committed) {
    history.push(app.committed); if (history.length > 60) history.shift();
    future.length = 0;
  }
  app.committed = cur;
}
const future = [];  // redo stack
addEventListener('keydown', (e) => {
  if (!(e.ctrlKey || e.metaKey) || /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || '')) return;
  const k = e.key.toLowerCase();
  const redo = (k === 'z' && e.shiftKey) || k === 'y';
  if (k !== 'z' && k !== 'y') return;
  e.preventDefault();
  const cur = JSON.stringify(app.settings);
  if (redo) {
    const nxt = future.pop();
    if (!nxt) { toast('Нечего повторять'); return; }
    history.push(cur);
    app.settings = JSON.parse(nxt); app.committed = nxt; undoing = true; rebuild(0); renderPanel(); toast('Повторено (Ctrl+Shift+Z)');
    return;
  }
  const prev = history.pop();
  if (!prev) { toast('Нечего отменять'); return; }
  future.push(cur); if (future.length > 60) future.shift();
  app.settings = JSON.parse(prev); app.committed = prev; undoing = true; rebuild(0); renderPanel(); toast('Отменено (Ctrl+Z) · повторить: Ctrl+Shift+Z');
});
let undoing = false;
function rebuild(delay = 350) {
  if (!app.info) { undoing = false; return; }  // no model yet: the choice is kept and used on open
  if (!undoing) snapshot(); undoing = false;
  app.dirty = true;
  clearTimeout(rebuildTimer);
  rebuildTimer = setTimeout(async () => {
    if (rebuildRunning) { rebuildAgain = true; return; }
    rebuildRunning = true; setBusy(true, '⟳ Применяю правки…', true);
    try { await applyInfo(await api('/api/rebuild', { settings: app.settings })); }
    catch (e) { toast(e.message, true, 8000); }
    setBusy(false, '', true); rebuildRunning = false;
    if (rebuildAgain) { rebuildAgain = false; rebuild(0); }
  }, delay);
}
const ddMenu = $('#ddOpen .dd-menu');
$('#btnOpen').onclick = (e) => { e.stopPropagation(); ddMenu.hidden = !ddMenu.hidden; };
addEventListener('click', () => { ddMenu.hidden = true; });
$('#cardFile').onclick = () => openFileDialog();
$('#cardFolder').onclick = () => $('#btnFolder').click();
$('#miFile').onclick = () => openFileDialog();
async function openFileDialog() {
  ddMenu.hidden = true;
  try {
    const r = await api('/api/browse', {});
    if (r.items) { const ok = r.items.filter(i => !i.error); if (ok.length === 1) openPath(ok[0].path, ok[0].member); else showScan(r); }
    else if (r.path) openPath(r.path);
  }
  catch (e) { toast(e.message, true); }
}
async function doSave(andLaunch) {
  setBusy(true, 'Сохранение…');
  try {
    const r = await api('/api/save', { settings: app.settings });
    if (r.saved) {
      app.dirty = false; app.savedAt = new Date(); app.savedPath = r.saved; updateChip();
      toast('Сохранено: ' + r.saved + ` (${r.size_mb} МБ)` + (r.anims?.length ? ` + анимаций .vrma: ${r.anims.length}` : '') + '. Правки запомнены — при следующем открытии модели они подхватятся.', false, 8000);
      if (r.anim_error) toast('Анимации не сохранились: ' + r.anim_error, true, 8000);
      if (andLaunch) { setBusy(false); await launch(false); return; }
    }
  } catch (e) { toast(e.message, true, 8000); }
  setBusy(false);
}
$('#btnSave').onclick = () => saveDialog();
function saveDialog() {
  if (!app.info) return;
  const s = app.settings, I = app.info;
  modalBody.innerHTML = '';
  const title = h('input', { value: s.meta.title || I.model?.name || '', style: 'flex:1' });
  const author = h('input', { value: s.meta.author || '', style: 'flex:1', placeholder: 'ваше имя или автор модели' });
  const fmt = selectEl({ 0: 'VRM 0.x — Astra, VSeeFace, большинство программ', 1: 'VRM 1.0 — новые программы, точнее физика' }, String(s.vrm_version || 0), () => {});
  const size = selectEl({ 0: 'без ограничения', 25: '25 МБ (VRChat Quest / cluster)', 50: '50 МБ', 100: '100 МБ' }, String(s.target_mb || 0), () => {});
  const apply = () => {
    const nv = +fmt.value, nm = +size.value;
    const rebuildNeeded = nv !== (s.vrm_version || 0) || nm !== (s.target_mb || 0);
    s.export_anims = ANIMS.list.map(a => a.id).filter(i => chosen.has(i));
    s.meta.title = title.value.trim(); s.meta.author = author.value.trim(); s.vrm_version = nv; s.target_mb = nm;
    return rebuildNeeded;
  };
  const chosen = new Set(Array.isArray(s.export_anims) ? s.export_anims : []);
  const animBox = h('div', { class: 'checks anim-checks' });
  const groupsA = {};
  for (const a of ANIMS.list) (groupsA[a.group] ||= []).push(a);
  for (const [g, as] of Object.entries(groupsA)) {
    animBox.append(h('div', { class: 'cat-h' }, h('span', { class: 'item-n' }, g)));
    for (const a of as) animBox.append(check(a.name, chosen.has(a.id), (v) => { v ? chosen.add(a.id) : chosen.delete(a.id); cnt.textContent = chosen.size ? '(' + chosen.size + ')' : ''; }));
  }
  const cnt = h('span', {}, chosen.size ? '(' + chosen.size + ')' : '');
  const animSec = h('details', {}, h('summary', {}, '🎞 Анимации рядом с VRM (.vrma) ', cnt),
    h('div', { class: 'note' }, 'Выбранные движения сохранятся отдельными файлами «имя.движение.vrma» рядом с VRM — для программ, которые понимают VRM Animation (VSeeFace, three-vrm, Warudo и др.).'),
    h('div', { class: 'btns' }, h('button', { onclick: () => { for (const a of ANIMS.list) chosen.add(a.id); animSec.querySelectorAll('input').forEach(x => x.checked = true); cnt.textContent = '(' + chosen.size + ')'; } }, 'Все'),
      h('button', { onclick: () => { chosen.clear(); animSec.querySelectorAll('input').forEach(x => x.checked = false); cnt.textContent = ''; } }, 'Ни одной')),
    animBox);
  modalBody.append(h('h2', {}, '💾 Сохранить VRM'),
    h('div', { class: 'row' }, h('label', {}, 'Название'), title),
    h('div', { class: 'row' }, h('label', {}, 'Автор'), author),
    h('div', { class: 'row' }, h('label', {}, 'Формат'), fmt),
    h('div', { class: 'row' }, h('label', {}, 'Размер'), size),
    animSec,
    I.model?.comment ? h('details', {}, h('summary', {}, 'Условия использования модели (readme / 利用規約)'),
      h('div', { class: 'log' }, I.model.comment)) : null,
    h('div', { class: 'note' }, 'Проверьте условия автора исходной модели: многие запрещают конвертацию и распространение.'),
    h('div', { class: 'btns' },
      h('button', { class: 'accent', onclick: () => { const rb = apply(); closeModal(); if (rb) { rebuild(0); setTimeout(() => waitIdle().then(() => doSave(false)), 50); } else doSave(false); } }, 'Сохранить…'),
      h('button', { onclick: () => { const rb = apply(); closeModal(); if (rb) { rebuild(0); setTimeout(() => waitIdle().then(() => doSave(true)), 50); } else doSave(true); } }, 'Сохранить и проверить в программе'),
      h('button', { onclick: closeModal }, 'Отмена')));
  modal.hidden = false;
}
function waitIdle() {
  return new Promise(res => { const t = setInterval(() => { if (!app.busy && !rebuildRunning) { clearInterval(t); res(); } }, 100); });
}
addEventListener('beforeunload', (e) => { if (app.dirty) { e.preventDefault(); e.returnValue = ''; } });
function updateChip() {
  const c = $('#chip'), I = app.info;
  if (!I) { c.hidden = true; return; }
  c.hidden = false;
  const n = issues().length;
  c.className = 'chip ' + (I.missing?.length ? 'bad' : n ? 'warn' : 'ok');
  c.textContent = (I.model?.name || 'модель') + (n ? ` · ⚠ ${n}` : ' · ✓') + (app.dirty ? ' · есть несохранённые правки' : app.savedAt ? ' · сохранено ' + app.savedAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '');
  $('#btnSave').classList.toggle('dirty', !!app.dirty);
  $('#btnSave').textContent = '💾 Сохранить VRM' + (I.size_mb ? ` · ${I.size_mb} МБ` : '');
}
$('#chip').onclick = () => { const is = issues(); if (is.length && is[0].tab) setTab(is[0].tab); };
// ---- folder scan
const modal = $('#modal'), modalBody = $('#modalBody');
function closeModal() { modal.hidden = true; modalBody.innerHTML = ''; }
modal.addEventListener('click', (e) => { if (e.target === modal) closeModal(); });
$('#btnFolder').onclick = async () => {
  setBusy(true, 'Поиск моделей в папке и архивах…');
  let r;
  try { r = await api('/api/scan', {}); } catch (e) { setBusy(false); toast(e.message, true); return; }
  setBusy(false);
  if (r.cancelled) return;
  showScan(r);
};
function showScan(r) {
  const items = r.items || [];
  const fmt = (n) => n > 1048576 ? (n / 1048576).toFixed(1) + ' МБ' : Math.round(n / 1024) + ' КБ';
  const checks = [];
  modalBody.innerHTML = '';
  modalBody.append(h('h2', {}, `Найдено моделей: ${items.length}`), h('div', { class: 'note' }, r.root +
    ' — включая подпапки и zip-архивы. Текстуры программа будет искать по всей этой папке.'));
  if (!items.length) modalBody.append(h('div', { class: 'warn' }, 'Файлов .pmx / .pmd / .fbx не найдено (в том числе внутри .zip/.rar/.7z).'));
  for (const it of items) {
    const cb = h('input', { type: 'checkbox', checked: true }); checks.push([cb, it]);
    modalBody.append(h('div', { class: 'mrow' }, cb,
      h('div', {}, h('div', {}, it.name), h('div', { class: 'path', title: it.rel }, it.rel)),
      h('span', { class: 'size' }, fmt(it.size || 0)),
      h('button', { onclick: () => { closeModal(); openPath(it.path, it.member); } }, 'Открыть')));
  }
  modalBody.append(h('div', { class: 'btns', style: 'margin-top:12px' },
    h('button', { class: 'accent', disabled: !items.length, onclick: async () => {
      const sel = checks.filter(([cb]) => cb.checked).map(([, it]) => it);
      if (!sel.length) return;
      closeModal(); setBusy(true, `Конвертация ${sel.length} моделей…`);
      try {
        const res = await api('/api/batch', { items: sel, settings: app.settings });
        if (!res.cancelled) {
          const ok = res.results.filter(x => !x.error).length;
          modalBody.innerHTML = '';
          modalBody.append(h('h2', {}, `Готово: ${ok} из ${res.results.length}`), h('div', { class: 'note' }, 'Папка: ' + res.outdir),
            ...res.results.map(x => h('div', { class: x.error ? 'bad' : 'ok' }, (x.error ? '✗ ' : '✓ ') + x.src + (x.error ? ' — ' + x.error : ''))),
            h('div', { class: 'btns' }, h('button', { onclick: closeModal }, 'Закрыть')));
          modal.hidden = false;
        }
      } catch (e) { toast(e.message, true, 8000); }
      setBusy(false);
    } }, 'Конвертировать отмеченные в VRM…'),
    h('button', { onclick: closeModal }, 'Закрыть')));
  modal.hidden = false;
}

// ---- drag & drop: files and whole folders from Explorer are copied into the program's cache
async function readEntry(entry, prefix, out) {
  if (entry.isFile) {
    const f = await new Promise((res, rej) => entry.file(res, rej));
    out.push([prefix + f.name, f]);
  } else if (entry.isDirectory) {
    const rd = entry.createReader();
    for (;;) {
      const batch = await new Promise((res, rej) => rd.readEntries(res, rej));
      if (!batch.length) break;
      for (const e of batch) await readEntry(e, prefix + entry.name + '/', out);
    }
  }
}
async function dropFiles(dt) {
  const files = [];
  const entries = [...(dt.items || [])].map(it => it.webkitGetAsEntry && it.webkitGetAsEntry()).filter(Boolean);
  if (entries.length) { for (const e of entries) await readEntry(e, '', files); }
  else for (const f of dt.files || []) files.push([f.name, f]);
  if (!files.length) return;
  const vmd = files.find(([n]) => /\.vmd$/i.test(n));
  if (vmd && files.length === 1) { loadMotion(vmd[1]); return; }
  const total = files.reduce((a, [, f]) => a + f.size, 0);
  if (total > 4 * 1024 ** 3) { toast('Слишком много данных (больше 4 ГБ)', true); return; }
  const session = Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  setBusy(true, 'Копирование файлов…');
  try {
    let done = 0;
    for (const [rel, f] of files) {
      const r = await fetch('/api/upload?session=' + session + '&rel=' + encodeURIComponent(rel),
        { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: f });
      if (!r.ok) throw new Error('Не удалось скопировать ' + rel);
      done += f.size; $('#busyText').textContent = `Копирование файлов… ${Math.round(100 * done / Math.max(1, total))}%`;
    }
    const r = await api('/api/upload_done', { session });
    setBusy(false);
    if (r.items.length === 1) { openPath(r.items[0].path, r.items[0].member); }
    else showScan(r);
    if (files.length === 1 && /\.(pmx|pmd|fbx)$/i.test(files[0][0]))
      toast('Перетащен только файл модели: если текстуры не нашлись, перетащите всю папку или архив', false, 7000);
  } catch (e) { setBusy(false); toast(e.message, true); }
}
addEventListener('dragover', e => e.preventDefault());
addEventListener('drop', e => {
  e.preventDefault();
  const txt = e.dataTransfer.getData('text/plain') || e.dataTransfer.getData('text/uri-list');
  if (!(e.dataTransfer.files && e.dataTransfer.files.length) && txt && /\.(pmx|pmd|fbx|vrm)\s*$/i.test(txt)) {
    openPath(decodeURIComponent(txt.trim().replace(/^file:\/\/\/?/, ''))); return;
  }
  dropFiles(e.dataTransfer);
});

// ---- VMD motion
async function loadMotion(file) {
  try {
    const m = parseVMD(await file.arrayBuffer());
    app.motion = m; app.motionName = file.name;
    if (app.vrm) { app.vrm.humanoid?.resetNormalizedPose(); app.vrm.springBoneManager?.reset(); app.player = new VmdPlayer(m, app.vrm, app.info); }
    if (app.orig) { app.orig.humanoid?.resetNormalizedPose(); app.orig.springBoneManager?.reset(); app.origPlayer = new VmdPlayer(m, app.orig, app.info); }
    app.anim = 'vmd'; app.vmdStart = clock.elapsedTime;
    document.querySelectorAll('#anim button').forEach(x => x.classList.toggle('on', x.id === 'btnVmd'));
    toast(`Движение ${file.name}: ${(m.frames / 30).toFixed(1)} с, костей ${m.bones.size}, морфов ${m.morphs.size}`);
  } catch (e) { toast(e.message, true); }
}
const vmdInput = h('input', { type: 'file', accept: '.vmd', style: 'display:none', onchange: (e) => { if (e.target.files[0]) loadMotion(e.target.files[0]); e.target.value = ''; } });
document.body.append(vmdInput);
$('#btnVmd').onclick = () => {
  if (app.motion && app.anim !== 'vmd') {
    app.anim = 'vmd'; app.vmdStart = clock.elapsedTime;
    document.querySelectorAll('#anim button').forEach(x => x.classList.toggle('on', x.id === 'btnVmd'));
  } else vmdInput.click();
};
window.__loadMotionBytes = (buf, name) => loadMotion(new File([buf], name || 'test.vmd'));

// ---- open the result in an avatar program (Astra, VSeeFace, ...)
async function launch(pick) {
  if (!app.info) return;
  setBusy(true, 'Сборка VRM и запуск программы…');
  try {
    const r = await api('/api/launch', { settings: app.settings, pick });
    if (r.launched) toast('Запущено: ' + r.launched.split(/[\\/]/).pop() + '. Файл: ' + r.vrm, false, 8000);
  } catch (e) { toast(e.message, true, 8000); }
  setBusy(false);
}
$('#btnLaunch').onclick = (e) => launch(e.shiftKey);
$('#btnLaunchPick').onclick = () => launch(true);

// ---- before / after
const origRoot = new THREE.Group(); scene.add(origRoot);
const baLabels = [h('div', { class: 'ba-label' }, 'ДО'), h('div', { class: 'ba-label after' }, 'ПОСЛЕ')];
for (const l of baLabels) { l.hidden = true; document.body.append(l); }
function updateBALabels() {
  const on = !!(app.showOrig && app.orig && app.vrm);
  baLabels.forEach(l => l.hidden = !on);
  if (!on) return;
  const r = canvas.getBoundingClientRect(), v = new THREE.Vector3();
  [[app.orig.scene, baLabels[0]], [app.vrm.scene, baLabels[1]]].forEach(([obj, el]) => {
    const box = new THREE.Box3().setFromObject(obj); v.set((box.min.x + box.max.x) / 2, box.max.y, (box.min.z + box.max.z) / 2);
    v.project(camera);
    el.style.left = (r.left + (v.x + 1) / 2 * r.width) + 'px'; el.style.top = (r.top + (1 - v.y) / 2 * r.height - 28) + 'px';
  });
}
async function loadOriginal(fit = true) {
  origRoot.clear(); app.origSrc = app.info.src;
  try {
    const resp = await fetch('/api/original?v=' + app.info.version);
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    app.origHasPhys = resp.headers.get('X-Has-Physics') === '1';
    const buf = await resp.arrayBuffer();
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    const gltf = await new Promise((res, rej) => loader.parse(buf, '', res, rej));
    const v = gltf.userData.vrm; VRMUtils.rotateVRM0(v);
    v.scene.traverse(o => { o.frustumCulled = false; });
    const box = new THREE.Box3().setFromObject(app.vrm.scene);
    origRoot.position.x = -(box.max.x - box.min.x) - 0.25;
    origRoot.add(v.scene); app.orig = v;
    v.springBoneManager?.reset();
    app.origPlayer = app.motion ? new VmdPlayer(app.motion, v, app.info) : null;
    if (!fit) return;
    const all = new THREE.Box3().setFromObject(app.vrm.scene).union(new THREE.Box3().setFromObject(v.scene));
    const size = all.getSize(new THREE.Vector3()), c = all.getCenter(new THREE.Vector3());
    const d = Math.max(size.y, size.x / camera.aspect) / 2 / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.15;
    controls.target.copy(c); camera.position.set(c.x, c.y + size.y * 0.05, c.z + d); controls.update();
  } catch (e) { toast('Не удалось показать исходную модель: ' + e.message, true); }
}
$('#btnOrig').onclick = async () => {
  app.showOrig = !app.showOrig;
  $('#btnOrig').classList.toggle('on', app.showOrig);
  if (app.showOrig && app.vrm) {
    setBusy(true, 'Сборка исходной модели…'); await loadOriginal(); setBusy(false);
    if (app.orig) toast(app.origHasPhys
      ? 'Слева — «до»: модель без ваших правок, с физикой из исходного файла. Справа — «после». Обе двигаются одинаково — включите «Идёт» или «Кружится».'
      : 'Слева — «до» без ваших правок. ⚠ В исходной модели физики нет — слева волосы и одежда неподвижны, справа физика добавлена программой.', false, 9000);
  } else { origRoot.clear(); app.orig = null; app.origPlayer = null; app.origSrc = null; if (app.vrm) fitCamera(); }
};

// ------------------------------------------------------------------ selection / picking
const ray = new THREE.Raycaster(); const ndc = new THREE.Vector2();
let downXY = null;
canvas.addEventListener('pointerdown', e => { downXY = [e.clientX, e.clientY]; if (WD.brushUid && e.button === 0) brushDown(e); });
canvas.addEventListener('pointermove', e => { if (WD.brushUid) brushMove(e); });
addEventListener('pointerup', () => { if (WD.painting) brushUp(); });
canvas.addEventListener('pointerup', e => {
  if (!downXY || !app.vrm || e.button !== 0) return;
  if (WD.brushUid) return;
  if (Math.hypot(e.clientX - downXY[0], e.clientY - downXY[1]) > 4) return;
  const r = canvas.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const meshes = []; app.vrm.scene.traverse(o => { if (o.isMesh && o.visible) meshes.push(o); });
  const hit = ray.intersectObjects(meshes, false)[0];
  // a joint dot right under the cursor (skeleton view / physics) wins over the surface behind it
  const joint = (app.tab === 'bones' || app.tab === 'physics') && !app.pickFor ? jointAtScreen(e.clientX - r.left, e.clientY - r.top, r) : null;
  if (app.tab === 'bones' && !app.pickFor) {
    const b = joint || (hit && skinBoneAt(hit));
    if (b) selectBoneNode(b); else select(null);
    return;
  }
  if (app.tab === 'physics') {
    // a joint dot counts only when it is not hidden behind the surface that was clicked
    let jn = joint;
    if (jn && hit) { const wp = jn.getWorldPosition(new THREE.Vector3()); if (wp.distanceTo(camera.position) > hit.distance + 0.04) jn = null; }
    const sb = hit && skinBoneAt(hit);
    const b = jn || sb;
    let cid = b ? chainOfNode(b) : null;
    // the skin of a body bone (face, chest, legs) is "no physics" — guess a nearby chain only for unmapped helpers
    if (!cid && hit && !sb) cid = nearestChain(hit.point, 0.04);  // unskinned mesh: the closest joint, if very close
    if (cid) {
      const ch = app.info.chains.find(x => x.id === cid);
      app.physOpen ||= {}; if (ch) app.physOpen[ch.kind] = true;
      app.physFilter = ''; app.flashT = clock.elapsedTime;
      select({ type: 'chain', id: cid }); return;
    }
    if (hit) {
      const mi = matOfMesh(hit.object);
      toast('Здесь нет физики' + (mi ? ': «' + mi.name + '» держится на теле' : '') + '. Перейти к части — вкладка «Части».');
    }
    return;
  }
  if (!hit) { select(null); return; }
  if (app.tab === 'bones' && app.pickFor) {
    const v = new THREE.Vector3(), leafOk = /Eye|Distal|Toes|jaw/.test(app.pickFor);  // these humanoid bones are leaves themselves
    // tier 1: the body skeleton (mapped bones and their parents); tier 2: other body bones with bones below them.
    // Tier 2 wins only when clearly closer, so tongue / hair roots / brows don't steal a click on the face.
    let b1 = null, d1 = 1e9, b2 = null, d2 = 1e9;
    for (const b of app.skel?.bones || []) {
      if (app.skel.phys.has(b) || !app.skel.body?.has(b)) continue;
      b.getWorldPosition(v); const d = v.distanceTo(hit.point);
      if (leafOk) { if (d < d1) { d1 = d; b1 = b; } continue; }
      if (app.skel.leafHuman.has(b)) continue;
      if (app.skel.trunk.has(b)) { if (d < d1) { d1 = d; b1 = b; } }
      else if (app.skel.deep.has(b) && d < d2) { d2 = d; b2 = b; }
    }
    const best = b2 && d2 < d1 * 0.4 ? b2 : b1;
    const ni = best ? app.nodes.indexOf(best) : -1;
    const name = ni >= 0 ? app.info.bones[ni] : null;
    if (name) { app.settings.humanoid[app.pickFor] = name; toast(hbLabel(app.pickFor) + ' → ' + ru(name)); app.pickFor = null; rebuild(0); }
    return;
  }
  if (app.tab === 'bones') return;  // no jump to "Части" while looking at the skeleton
  for (const [idx, ms] of app.matMeshes) if (ms.includes(hit.object)) {
    const mi = app.info.materials.find(m => m.gltf_index === idx);
    if (mi) { if (app.tab !== 'parts') setTab('parts'); select({ type: 'mat', id: mi.name }); }
    return;
  }
});
function matOfMesh(obj) {
  for (const [idx, ms] of app.matMeshes) if (ms.includes(obj)) return app.info.materials.find(m => m.gltf_index === idx);
  return null;
}
// the bone that carries most of the clicked triangle (its skin weights)
function skinBoneAt(hit) {
  const o = hit.object;
  if (!o?.isSkinnedMesh || !hit.face) return null;
  const si = o.geometry.attributes.skinIndex, sw = o.geometry.attributes.skinWeight;
  if (!si || !sw) return null;
  const acc = new Map();
  for (const vi of [hit.face.a, hit.face.b, hit.face.c])
    for (let k = 0; k < 4; k++) {
      const w = sw.getComponent(vi, k); if (w <= 0) continue;
      const b = o.skeleton.bones[si.getComponent(vi, k)]; if (b) acc.set(b, (acc.get(b) || 0) + w);
    }
  let best = null, bw = 0;
  for (const [b, w] of acc) if (w > bw) { bw = w; best = b; }
  return best;
}
// nearest drawn joint within ~14 px of the cursor
function jointAtScreen(x, y, rect) {
  const list = app.tab === 'bones' ? (app.skel?.bones || []) : [...new Set(app.info.chains.flatMap(c => c.nodes).map(i => app.nodes[i]).filter(Boolean))];
  let best = null, bd = 14; const v = new THREE.Vector3();
  for (const b of list) {
    b.getWorldPosition(v); v.project(camera);
    if (v.z > 1) continue;
    const px = (v.x + 1) / 2 * rect.width, py = (1 - v.y) / 2 * rect.height;
    const d = Math.hypot(px - x, py - y); if (d < bd) { bd = d; best = b; }
  }
  return best;
}
function chainOfNode(b) {
  for (let o = b; o; o = o.parent) {
    const i = app.nodes.indexOf(o); if (i < 0) continue;
    // a branch (pendant) first, then its host chain
    const cs = app.info.chains.filter(c => c.nodes.includes(i));
    if (cs.length) return (cs.find(c => c.branch_of) || cs[0]).id;
  }
  return null;
}
function selectBoneNode(b) {
  const i = app.nodes.indexOf(b); const name = i >= 0 ? app.info.bones[i] : null;
  if (!name) return;
  const hb = Object.entries(app.info.humanoid_nodes || {}).find(([, n]) => n === i)?.[0];
  app.flashT = clock.elapsedTime;
  if (hb) { app.openHb = true; select({ type: 'bone', id: hb, node: i }); return; }
  app.openNames = true; boneFilter = name;
  select({ type: 'node', id: name, node: i });
}
function nearestChain(p, maxd = 0.12) {
  let best = null, bd = maxd; const v = new THREE.Vector3();
  for (const c of app.info.chains) for (const i of c.nodes) {
    const o = app.nodes[i]; if (!o) continue; o.getWorldPosition(v);
    const d = v.distanceTo(p); if (d < bd) { bd = d; best = c.id; }
  }
  return best;
}
function select(s) {
  app.sel = s; renderPanel();
  const el = $('#panel .item.sel') || $('#panel .fam-h.sel') || $('#panel .lbl.sel') || $('#panel .bonecard');
  el?.scrollIntoView({ block: 'center' });
  if (el) { el.classList.add('flash'); setTimeout(() => el.classList.remove('flash'), 900); }
}
function famName(id) {
  let r = ru(id);
  for (let i = 0; i < 3; i++) r = r.replace(/(^|\s)([a-zA-Z]|\d+[a-zA-Z]?|Л|П|L|R)(?=\s|$)/g, ' ');
  return r.replace(/[_\s]+/g, ' ').trim();
}
// strands of one hairstyle / pleats of one skirt share a "family" name: hair_base_L_a_01 -> hair base
function chainFamily(id) {
  let s = id.replace(/[左右]/g, ' ').replace(/([a-z])([A-Z])/g, '$1 $2');
  const toks = s.split(/[_\.\-\s]+|\d+/).map(t => t.trim()).filter(Boolean)
    .filter(t => !/^(l|r|left|right|jnt|joint|bone|bn|end|tip|[a-z])$/i.test(t));
  return toks.slice(0, 2).join(' ').toLowerCase() || id;
}
const tint = new THREE.Color(0.25, 0.45, 1.0);
const selLine = new THREE.LineSegments(new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0xff7ab6, depthTest: false, transparent: true }));
selLine.renderOrder = 9999; selRoot.add(selLine);
const selDots = new THREE.Points(new THREE.BufferGeometry(),
  new THREE.PointsMaterial({ color: 0xffd36b, size: 7, sizeAttenuation: false, depthTest: false }));
selDots.renderOrder = 9999; selRoot.add(selDots);
function updateSelection(t) {
  // material highlight via emissive pulse
  const pulse = 0.55 + 0.45 * Math.sin(t * 6);
  for (const [idx, ms] of app.matMeshes) {
    const mi = app.info.materials.find(m => m.gltf_index === idx);
    const on = app.sel?.type === 'mat' && mi?.name === app.sel.id ||
               app.sel?.type === 'cat' && mi?.category === app.sel.id;
    for (const mesh of ms) for (const mat of [].concat(mesh.material)) {
      if (!mat.emissive) continue;
      if (!mat.userData.origEmissive) mat.userData.origEmissive = mat.emissive.clone();
      if (on) mat.emissive.copy(tint).multiplyScalar(pulse); else mat.emissive.copy(mat.userData.origEmissive);
    }
  }
  // chain / bone lines
  const pts = [], dots = []; const v = new THREE.Vector3(), w = new THREE.Vector3();
  const chainIds = app.sel?.type === 'chain' ? [app.sel.id]
    : app.sel?.type === 'kind' ? app.info.chains.filter(c => c.kind === app.sel.id).map(c => c.id)
    : app.sel?.type === 'family' ? app.info.chains.filter(c => c.kind + '|' + chainFamily(c.id) === app.sel.id).map(c => c.id) : [];
  const fl = app.flashT !== undefined ? Math.max(0, 1 - (t - app.flashT) / 1.2) : 0;
  selDots.material.size = 7 + 10 * fl * (0.5 + 0.5 * Math.sin(t * 20));
  for (const id of chainIds) {
    const c = app.info.chains.find(x => x.id === id); if (!c) continue;
    const set = new Set(c.nodes);
    for (const i of c.nodes) {
      const o = app.nodes[i]; if (!o) continue; o.getWorldPosition(v); dots.push(v.x, v.y, v.z);
      const p = o.parent; const pi = app.nodes.indexOf(p);
      if (set.has(pi)) { p.getWorldPosition(w); pts.push(w.x, w.y, w.z, v.x, v.y, v.z); }
    }
  }
  if (app.sel?.type === 'node') {
    const o = app.nodes[app.sel.node];
    if (o) { o.getWorldPosition(v); dots.push(v.x, v.y, v.z); if (o.parent?.isBone) { o.parent.getWorldPosition(w); pts.push(w.x, w.y, w.z, v.x, v.y, v.z); } }
  }
  if (app.sel?.type === 'bone') {
    const i = app.info.humanoid_nodes?.[app.sel.id];
    const o = i !== undefined ? app.nodes[i] : null;
    if (o) { o.getWorldPosition(v); dots.push(v.x, v.y, v.z); }
  }
  selLine.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  selDots.geometry.setAttribute('position', new THREE.Float32BufferAttribute(dots, 3));
}
// base-body meshes for the fitting guides: the 'skin' part (visible or not — the guides sit on the body under clothes)
function bodyMeshes(inner) {
  const out = [];
  for (const [idx, ms] of app.matMeshes) {
    const mi = app.info.materials.find(m => m.gltf_index === idx);
    const cat = app.settings.materials[mi?.name]?.category || mi?.category;
    if (inner ? (cat === 'clothes' && /underwear|swim|bra|lingerie/.test(mi?.sublabel || '')) : cat === 'skin') out.push(...ms);
  }
  return out;
}
function attachFitGuides() {
  if (!app.vrm) return;
  // rest pose: placement is measured before the test animation moves the body
  app.vrm.humanoid?.resetNormalizedPose(); app.vrm.update(0);
  fitGuides.attach(app.vrm, bodyMeshes(), bodyMeshes(true));
}
function applyLiveVisibility() {
  for (const [idx, ms] of app.matMeshes) {
    const mi = app.info.materials.find(m => m.gltf_index === idx);
    const ov = app.settings.materials[mi?.name];
    for (const mesh of ms) mesh.visible = !(ov && ov.visible === false);
  }
}

const ru = (n) => (app.settings.renames[n]) || app.info?.bones_ru?.[n] || n;
function boneLang() { return app.settings.bone_lang || (app.settings.ru_names ? 'ru' : 'orig'); }
// the name a bone gets in the file: own name > chosen translation > original
function finalName(n) {
  if (app.settings.renames[n]) return app.settings.renames[n];
  const l = boneLang();
  return l === 'ru' ? (app.info?.bones_ru?.[n] || n) : l === 'en' ? (app.info?.bones_en?.[n] || n) : n;
}
function boneLangControl() {
  const cur = boneLang();
  const seg = h('div', { class: 'seg small' },
    ...[['orig', 'Оригинал'], ['ru', 'Русский'], ['en', 'English']].map(([k, t]) =>
      h('button', { class: cur === k ? 'on' : '', onclick: () => {
        app.settings.bone_lang = k; app.settings.ru_names = false; rebuild(0);
        toast(k === 'orig' ? 'Имена костей — оригинальные' : 'Имена костей переведены: ' + t);
      } }, t)));
  return h('div', { class: 'row' }, h('label', {}, 'Имена костей'), seg);
}
// ------------------------------------------------------------------ overrides helpers
function matOv(name) { return app.settings.materials[name] || (app.settings.materials[name] = {}); }
function chainOv(id) { return app.settings.chains[id] || (app.settings.chains[id] = {}); }
let mirrorEdits = true;
function mirrorId(id) {
  const cands = [];
  const esc = (x) => x.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const swap = (str, a, b) => str.replace(new RegExp(esc(a), 'g'), '\u0000').replace(new RegExp(esc(b), 'g'), a).replace(/\u0000/g, b);
  for (const [a, b] of [['左', '右'], ['_L_', '_R_'], ['_l_', '_r_'], ['Left', 'Right'], ['left', 'right'],
                        ['L_', 'R_'], ['_L', '_R'], ['.L', '.R'], ['_l', '_r'], [' L ', ' R ']]) {
    if (id.includes(a) || id.includes(b)) cands.push(swap(id, a, b));
  }
  return cands.find(x => x !== id && app.info.chains.some(c => c.id === x)) || null;
}
let physSaveTimer = null;
function physEdited() {  // slider / preset edits: no reconversion, but undo + dirty + autosave
  snapshot(); app.dirty = true; updateChip();
  // a rebuild that is running now was started with older settings: its result would bring the old values back
  if (rebuildRunning) rebuildAgain = true;
  clearTimeout(physSaveTimer);
  physSaveTimer = setTimeout(() => { api('/api/settings', { settings: app.settings }).catch(() => {}); }, 600);
}
function setChainParam(c, key, val, noMirror, keepPreset) {
  if (mirrorEdits && !noMirror) {
    const mid = mirrorId(c.id); const mc = mid && app.info.chains.find(x => x.id === mid);
    if (mc) setChainParam(mc, key, val, true, keepPreset);
  }
  const ov = chainOv(c.id);
  ov[key] = val; c[key] = val;
  if (!keepPreset && ['stiffness', 'gravity', 'drag'].includes(key)) {
    ov.preset = 'custom';  // a preset (or auto) was in use, then a slider moved: now it is the user's own setting
    updatePresetSelects();
  }
  const map = { stiffness: 'stiffness', gravity: 'gravityPower', drag: 'dragForce', hitRadius: 'hitRadius' };
  for (const j of app.chainJoints.get(c.id) || []) j.settings[map[key]] = val;
}

// ------------------------------------------------------------------ panels
function renderTabs() {
  document.querySelectorAll('#tabs button').forEach(b => {
    b.classList.toggle('on', b.dataset.tab === app.tab);
    const d = b.querySelector('.dot'); if (d) d.className = 'dot ' + (app.info ? stepState(b.dataset.tab) : '');
  });
}
// what still needs attention, with where to go and a one-click fix when there is one
function issues() {
  const I = app.info; if (!I) return [];
  const out = [];
  if (I.missing?.length) out.push({ tab: 'bones', bad: true, text: 'Не найдены обязательные кости: ' + I.missing.map(hbLabel).join(', '),
    fix: ['Восстановить скелет', () => { app.settings.repair = true; rebuild(0); }] });
  const lim = app.settings.target_mb;
  if (lim && I.size_mb > lim) out.push({ tab: 'more', text: `Файл ${I.size_mb} МБ — больше выбранных ${lim} МБ` });
  const noCol = (I.chains || []).filter(c => c.enabled && ['hair', 'tail', 'skirt'].includes(c.kind) && c.colliders && c.n_colliders === 0);
  if (noCol.length) out.push({ tab: 'physics', text: `${noCol.length} цепочек без столкновений с телом — могут проходить сквозь модель`,
    fix: ['Показать', () => { setTab('physics'); select({ type: 'chain', id: noCol[0].id }); }] });
  const tex = (I.warnings || []).filter(w => /текстур/i.test(w));
  if (tex.length) out.push({ tab: 'parts', text: tex.length === 1 ? tex[0] : `${tex.length} предупреждений о текстурах (см. Ещё → Журнал)` });
  for (const w of (I.warnings || []).filter(w => !/текстур/i.test(w)).slice(0, 3)) out.push({ tab: 'more', text: w });
  return out;
}
function stepState(tab) {
  const is = issues().filter(x => x.tab === tab);
  if (is.some(x => x.bad)) return 'bad';
  if (is.length) return 'warn';
  return (tab === 'more' || tab === 'wardrobe') ? '' : 'ok';
}
function issuesBar() {
  const is = issues(); if (!is.length) return null;
  const mine = is.filter(x => x.tab === app.tab), others = is.filter(x => x.tab !== app.tab);
  const box = h('div', { class: 'issues' });
  for (const x of mine) box.append(h('div', { class: 'issue' + (x.bad ? ' bad' : '') }, h('span', {}, '⚠ ' + x.text),
    x.fix ? h('button', { onclick: x.fix[1] }, x.fix[0]) : null));
  if (others.length) box.append(h('div', { class: 'issue other' }, `Ещё ${others.length} — на вкладках: ` +
    [...new Set(others.map(x => ({ parts: 'Части', physics: 'Физика', expr: 'Лицо', bones: 'Скелет', more: 'Ещё' })[x.tab]))].join(', ')));
  return box;
}
document.querySelectorAll('#tabs button').forEach(b => b.onclick = () => { setTab(b.dataset.tab); });
function setTab(tab) {
  if (tab === 'settings' || tab === 'log') tab = 'more';
  const prev = app.tab;
  app.tab = tab; renderTabs(); renderPanel(); applyTabView(prev);
}

// ---- per-tab view: see-through body with bones on "Скелет", close-up of the face on "Выражения"
const skelRoot = new THREE.Group(); skelRoot.renderOrder = 10001; scene.add(skelRoot);
const skelLines = new THREE.LineSegments(new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false, transparent: true, opacity: 0.95 }));
const skelDots = new THREE.Points(new THREE.BufferGeometry(),
  new THREE.PointsMaterial({ vertexColors: true, size: 6, sizeAttenuation: false, depthTest: false, transparent: true }));
skelLines.renderOrder = skelDots.renderOrder = 10001;
skelRoot.add(skelLines, skelDots); skelRoot.visible = false;
const legend = h('div', { id: 'legend', hidden: true },
  h('span', { class: 'dot', style: 'background:#5be08a' }), 'кости гуманоида ',
  h('span', { class: 'dot', style: 'background:#ff7ab6' }), 'физика ',
  h('span', { class: 'dot', style: 'background:#8a93a8' }), 'прочие');
$('#view').append(legend);
function setSeeThrough(on) {
  if (!app.vrm) return;
  app.vrm.scene.traverse(o => {
    if (!o.isMesh) return;
    for (const m of [].concat(o.material)) {
      if (on) {
        if (!m.userData.seeThrough) m.userData.seeThrough = { t: m.transparent, o: m.opacity, d: m.depthWrite };
        m.transparent = true; m.opacity = 0.22; m.depthWrite = false;
      } else if (m.userData.seeThrough) {
        const st = m.userData.seeThrough; m.transparent = st.t; m.opacity = st.o; m.depthWrite = st.d;
        delete m.userData.seeThrough;
      }
      m.needsUpdate = true;
    }
  });
}
function collectSkeleton() {
  const bones = new Set();
  app.vrm.scene.traverse(o => { if (o.isSkinnedMesh) for (const b of o.skeleton.bones) bones.add(b); });
  const human = new Set(Object.values(app.info.humanoid_nodes || {}).map(i => app.nodes[i]).filter(Boolean));
  const phys = new Set();
  for (const c of app.info.chains || []) for (const i of c.nodes) if (app.nodes[i]) phys.add(app.nodes[i]);
  for (const b of human) bones.add(b);
  // bones that can be a humanoid bone: no physics bones and nothing hanging below them (hair/skirt/ribbons)
  const body = new Set();
  const under = (b) => { for (let p = b; p; p = p.parent) if (phys.has(p)) return true; return false; };
  for (const b of bones) if (!under(b)) body.add(b);
  // "trunk": humanoid bones, their ancestors, and body bones with at least two levels of body bones below
  const leafHuman = new Set(Object.entries(app.info.humanoid_nodes || {}).filter(([k]) => /Eye|Distal|Toes|jaw/.test(k)).map(([, i]) => app.nodes[i]).filter(Boolean));
  const trunk = new Set();
  for (const b of human) for (let p = b; p && p.isBone; p = p.parent) trunk.add(p);
  const isDeep = (b, d) => d === 0 || b.children.some(x => body.has(x) && isDeep(x, d - 1));
  const deep = new Set([...body].filter(b => isDeep(b, 2)));
  app.skel = { bones: [...bones], human, phys, body, trunk, leafHuman, deep };
}
const _v1 = new THREE.Vector3(), _v2 = new THREE.Vector3();
function updateSkeleton() {
  if (!app.skel || !app.vrm) return;
  const { bones, human, phys } = app.skel;
  const set = new Set(bones), pts = [], cols = [], dp = [], dc = [];
  const col = (b) => human.has(b) ? [0.36, 0.88, 0.54] : phys.has(b) ? [1, 0.48, 0.71] : [0.54, 0.58, 0.66];
  for (const b of bones) {
    b.getWorldPosition(_v1); dp.push(_v1.x, _v1.y, _v1.z); dc.push(...col(b));
    if (b.parent && set.has(b.parent)) {
      b.parent.getWorldPosition(_v2); pts.push(_v2.x, _v2.y, _v2.z, _v1.x, _v1.y, _v1.z);
      const c = col(b); cols.push(...c, ...c);
    }
  }
  skelLines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  skelLines.geometry.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  skelDots.geometry.setAttribute('position', new THREE.Float32BufferAttribute(dp, 3));
  skelDots.geometry.setAttribute('color', new THREE.Float32BufferAttribute(dc, 3));
}
function focusFace() {
  const head = app.vrm?.humanoid?.getNormalizedBoneNode('head');
  if (!head) return;
  if (!app.camSaved) app.camSaved = { p: camera.position.clone(), t: controls.target.clone() };
  head.getWorldPosition(_v1);
  const box = new THREE.Box3().setFromObject(app.vrm.scene);
  const hgt = box.max.y - box.min.y;
  _v1.y += hgt * 0.035;
  controls.target.copy(_v1);
  camera.position.set(_v1.x, _v1.y + hgt * 0.01, _v1.z + hgt * 0.3);
  controls.update();
}
function applyTabView(prev) {
  const bones = app.tab === 'bones' && !!app.vrm;
  setSeeThrough(bones);
  skelRoot.visible = bones; legend.hidden = !bones;
  if (bones) { collectSkeleton(); updateSkeleton(); }
  if (app.tab === 'expr' && app.vrm) {
    if (app.anim === 'walk' || app.anim === 'jump' || app.anim === 'spin') {
      app.anim = 'idle';
      document.querySelectorAll('#anim button').forEach(x => x.classList.toggle('on', x.dataset.anim === 'idle'));
    }
    focusFace();
  }
  else if (prev === 'expr' && app.camSaved) {
    camera.position.copy(app.camSaved.p); controls.target.copy(app.camSaved.t); controls.update(); app.camSaved = null;
  }
}
window.__setTab = setTab;
$('#vFit').onclick = () => { app.camSaved = null; if (app.vrm) fitCamera(); };
$('#vFace').onclick = () => { app.camSaved = null; focusFace(); app.camSaved = null; };

function slider(label, val, min, max, step, onInput, fmt = (x) => (+x).toFixed(2), ends = null) {
  const out = h('span', { class: 'val' }, fmt(val));
  const inp = h('input', { type: 'range', min, max, step, value: val,
    oninput: (e) => { out.textContent = fmt(e.target.value); onInput(+e.target.value); },
    onchange: () => { if (inp.closest('#panel') && app.tab === 'physics') physEdited(); } });
  const row = h('div', { class: 'row' }, h('label', { title: ends?.hint || '' }, label), inp, out);
  if (!ends) return row;
  return h('div', { class: 'prow', title: ends.hint || '' }, row,
    h('div', { class: 'ends' }, h('span', {}, '◂ ' + ends.lo), h('span', {}, ends.hi + ' ▸')));
}
function selectEl(options, value, onChange) {
  return h('select', { onchange: (e) => onChange(e.target.value) },
    ...Object.entries(options).map(([k, v]) => h('option', { value: k, selected: k === value }, v)));
}
function check(label, val, onChange) {
  return h('label', { class: 'chk' }, h('input', { type: 'checkbox', checked: !!val, onchange: (e) => onChange(e.target.checked) }), label);
}

function renderPanel() {
  const P = $('#panel'); P.innerHTML = '';
  if (!app.info) {
    P.append(h('div', { class: 'note' }, 'Откройте модель .pmx, .pmd или .fbx. Программа сама найдёт кости, выражения, физику и разложит модель на части.'));
    if (app.tab === 'more') P.append(...settingsPanel());
    return;
  }
  const f = { parts: partsPanel, physics: physicsPanel, bones: bonesPanel, expr: exprPanel,
              more: morePanel }[app.tab] || partsPanel;
  const ib = issuesBar(); if (ib) P.append(ib);
  P.append(...f().filter(Boolean));
}

const BLEND = { opaque: 'Непрозрачный', cutout: 'Вырез (дырки по маске)', blend: 'Полупрозрачный' };
// ---- fitting tools (preview only: nothing of this goes into the VRM) -------------------------------
// ▦ grid on the body, ◐ clothing clearance (red = through the body, yellow = touching), 📏 measurements
const fitT = { grid: false, clear: false, meas: null };
const BODY_CATS = new Set(['skin', 'face']);
function meshCat(mesh) {
  for (const [idx, ms] of app.matMeshes) if (ms.includes(mesh)) return app.info.materials.find(m => m.gltf_index === idx)?.category;
  return null;
}
function gridMaterial(orig) {
  const m = new THREE.MeshLambertMaterial({ color: 0xf1e3da });
  m.onBeforeCompile = (sh) => {
    sh.vertexShader = 'varying vec3 vGridP;\n' + sh.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvGridP = position;');
    sh.fragmentShader = 'varying vec3 vGridP;\n' + sh.fragmentShader.replace('#include <dithering_fragment>', `#include <dithering_fragment>
      vec3 q2 = vGridP / 0.02; vec3 g2 = abs(fract(q2 - 0.5) - 0.5) / fwidth(q2);
      vec3 q10 = vGridP / 0.10; vec3 g10 = abs(fract(q10 - 0.5) - 0.5) / fwidth(q10);
      float l2 = 1.0 - min(min(min(g2.x, g2.y), g2.z), 1.0);
      float l10 = 1.0 - min(min(min(g10.x, g10.y), g10.z) * 0.6, 1.0);
      gl_FragColor.rgb = mix(gl_FragColor.rgb, vec3(0.35, 0.42, 0.62), l2 * 0.45);
      gl_FragColor.rgb = mix(gl_FragColor.rgb, vec3(0.16, 0.22, 0.48), l10 * 0.85);`);
  };
  m.userData.fitTool = true;
  return m;
}
function restPositions(mesh) {  // vertex positions of the bind pose in scene space
  const p = mesh.geometry.attributes.position, out = new Float32Array(p.count * 3), v = new THREE.Vector3();
  mesh.updateWorldMatrix(true, false);
  const M = mesh.isSkinnedMesh ? mesh.bindMatrix.clone().premultiply(mesh.matrixWorld) : mesh.matrixWorld;
  for (let i = 0; i < p.count; i++) { v.fromBufferAttribute(p, i).applyMatrix4(M); out.set([v.x, v.y, v.z], i * 3); }
  return out;
}
function computeClearance() {
  // body points (skin, generated body) on a 1 cm hash grid
  const cell = 0.01, grid = new Map(), body = [];
  for (const [idx, ms] of app.matMeshes) {
    const mi = app.info.materials.find(m => m.gltf_index === idx);
    if (!mi || !BODY_CATS.has(mi.category)) continue;
    for (const mesh of ms) {
      const P = restPositions(mesh), N = mesh.geometry.attributes.normal;
      for (let i = 0; i < P.length / 3; i++) {
        const k = Math.floor(P[i * 3] / cell) + ',' + Math.floor(P[i * 3 + 1] / cell) + ',' + Math.floor(P[i * 3 + 2] / cell);
        const e = [P[i * 3], P[i * 3 + 1], P[i * 3 + 2], N ? N.getX(i) : 0, N ? N.getY(i) : 1, N ? N.getZ(i) : 0];
        (grid.get(k) || grid.set(k, []).get(k)).push(e); body.push(e);
      }
    }
  }
  if (!body.length) return 0;
  let bad = 0;
  for (const [idx, ms] of app.matMeshes) {
    const mi = app.info.materials.find(m => m.gltf_index === idx);
    if (!mi || BODY_CATS.has(mi.category) || mi.category === 'hair' || mi.category === 'eyes') continue;
    for (const mesh of ms) {
      const P = restPositions(mesh), n = P.length / 3, col = new Float32Array(n * 3).fill(1);
      for (let i = 0; i < n; i++) {
        const x = P[i * 3], y = P[i * 3 + 1], z = P[i * 3 + 2];
        const cx = Math.floor(x / cell), cy = Math.floor(y / cell), cz = Math.floor(z / cell);
        // vote of the body points within 1.2 cm: "inside" only when most of them see this point behind
        // their surface (one stray normal at the edge of a skin patch must not paint it red)
        let bd = 1e9, inn = 0, outn = 0;
        for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
          const l = grid.get((cx + dx) + ',' + (cy + dy) + ',' + (cz + dz)); if (!l) continue;
          for (const e of l) {
            const d = (e[0] - x) ** 2 + (e[1] - y) ** 2 + (e[2] - z) ** 2; if (d < bd) bd = d;
            if (d > 0.012 * 0.012) continue;
            const side = (x - e[0]) * e[3] + (y - e[1]) * e[4] + (z - e[2]) * e[5];
            if (side < -0.001) inn++; else outn++;
          }
        }
        if (bd > 1e8) continue;
        if (inn >= 3 && inn > outn * 1.5) { col.set([1, 0.22, 0.22], i * 3); bad++; }   // through the body
        else if (Math.sqrt(bd) < 0.005) col.set([1, 0.85, 0.3], i * 3);                // touching (< 5 mm)
      }
      mesh.geometry.setAttribute('color', new THREE.BufferAttribute(col, 3));
      const om = mesh.userData.origMat || mesh.material;
      const m = new THREE.MeshLambertMaterial({ map: om.map || null, vertexColors: true, side: om.side,
        transparent: om.transparent, alphaTest: om.alphaTest || 0 });
      m.userData.fitTool = true;
      mesh.userData.clearMat = m;
    }
  }
  return bad;
}
function applyFitOverlay() {
  if (!app.vrm) return;
  for (const [, ms] of app.matMeshes) for (const mesh of ms) {
    if (!mesh.userData.origMat) mesh.userData.origMat = mesh.material;
    const cat = meshCat(mesh);
    let mat = mesh.userData.origMat;
    if (fitT.clear && mesh.userData.clearMat) mat = mesh.userData.clearMat;
    else if (fitT.grid && cat === 'skin') mat = mesh.userData.gridMat ||= gridMaterial(mat);
    mesh.material = mat;
  }
}
function measure() {
  // circumference = perimeter of the convex hull of a 1 cm slice of the body (as a tape measure goes)
  const pts = [];
  for (const [idx, ms] of app.matMeshes) {
    const mi = app.info.materials.find(m => m.gltf_index === idx);
    if (!mi || ['hair', 'eyes'].includes(mi.category)) continue;
    const bodyOnly = BODY_CATS.has(mi.category);
    for (const mesh of ms) { const P = restPositions(mesh); for (let i = 0; i < P.length; i += 3) pts.push([P[i], P[i + 1], P[i + 2], bodyOnly]); }
  }
  const restBone = (name) => {
    const b = app.vrm.humanoid?.getRawBoneNode(name); if (!b) return null;
    for (const [, ms] of app.matMeshes) for (const mesh of ms) {
      if (!mesh.isSkinnedMesh) continue;
      const i = mesh.skeleton.bones.indexOf(b);
      if (i >= 0) return new THREE.Vector3().setFromMatrixPosition(mesh.skeleton.boneInverses[i].clone().invert().premultiply(mesh.matrixWorld));
    }
    return b.getWorldPosition(new THREE.Vector3());
  };
  const hips = restBone('hips'), chest = restBone('upperChest') || restBone('chest'), lul = restBone('leftUpperLeg');
  const lsh = restBone('leftUpperArm'), rsh = restBone('rightUpperArm'), lh = restBone('leftHand'), lf = restBone('leftFoot');
  if (!hips || !chest) return null;
  const cx = hips.x, cz = hips.z;
  const hull = (P) => {  // 2D convex hull perimeter (monotone chain)
    if (P.length < 3) return 0;
    P.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lo = [], up = [];
    for (const p of P) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); }
    for (const p of P.slice().reverse()) { while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], p) <= 0) up.pop(); up.push(p); }
    const H = lo.slice(0, -1).concat(up.slice(0, -1)); let per = 0;
    for (let i = 0; i < H.length; i++) { const a = H[i], b = H[(i + 1) % H.length]; per += Math.hypot(a[0] - b[0], a[1] - b[1]); }
    return per;
  };
  const slice = (y, onlyBody) => hull(pts.filter(p => Math.abs(p[1] - y) < 0.005 && Math.abs(p[0] - cx) < 0.22 && Math.abs(p[2] - cz) < 0.22 && (!onlyBody || p[3])).map(p => [p[0], p[2]]));
  const scan = (y0, y1, pick, onlyBody) => { let best = null; for (let y = y0; y <= y1; y += 0.005) { const c = slice(y, onlyBody); if (c > 0.2 && (best === null || pick(c, best.c))) best = { y, c }; } return best; };
  const shoulderY = lsh ? lsh.y : chest.y + 0.08;
  const legY = lul ? lul.y : hips.y - 0.08;
  const res = {};
  for (const [key, onlyBody] of [['outer', false], ['body', true]]) {
    const bust = scan(chest.y - 0.08, Math.min(chest.y + 0.08, shoulderY - 0.05), (a, b) => a > b, onlyBody);
    const spine = restBone('spine');
    const w0 = Math.min(hips.y, spine ? spine.y : hips.y) - 0.02, w1 = Math.max(w0 + 0.04, chest.y - 0.05);
    const waist = scan(w0, w1, (a, b) => a < b, onlyBody);
    const hip = scan(legY - 0.1, hips.y + 0.02, (a, b) => a > b, onlyBody);
    res[key] = { bust: bust?.c, waist: waist?.c, hips: hip?.c };
  }
  let y0 = Infinity, y1 = -Infinity;
  for (const p of pts) { if (p[1] < y0) y0 = p[1]; if (p[1] > y1) y1 = p[1]; }
  res.height = pts.length ? y1 - y0 : null;
  res.shoulders = lsh && rsh ? lsh.distanceTo(rsh) : null;
  res.arm = lsh && lh ? lsh.distanceTo(lh) : null;
  res.leg = lul && lf ? lul.y - lf.y : null;
  return res;
}
function fitSection() {
  const cm = (v) => (v ? Math.round(v * 100) + ' см' : '—');
  const M = fitT.meas;
  const table = M ? h('div', { class: 'meas' },
    ...[['Рост', cm(M.height)],
      ['Обхват груди', cm(M.outer.bust) + (M.body.bust ? '  ·  по коже ' + cm(M.body.bust) : '')],
      ['Обхват талии', cm(M.outer.waist) + (M.body.waist ? '  ·  по коже ' + cm(M.body.waist) : '')],
      ['Обхват бёдер', cm(M.outer.hips) + (M.body.hips ? '  ·  по коже ' + cm(M.body.hips) : '')],
      ['Ширина плеч (между суставами)', cm(M.shoulders)], ['Длина руки', cm(M.arm)], ['Длина ноги', cm(M.leg)]]
      .map(([a, b]) => h('div', { class: 'mrow' }, h('span', {}, a), h('b', {}, b)))) : null;
  return h('details', { class: 'sec', open: app.openFit },
    h('summary', { onclick: () => { app.openFit = !app.openFit; } }, '📏 Примерка: сетка, просвет одежды, мерки'),
    h('div', { class: 'note' }, 'Только для просмотра — в сохранённый VRM это не попадает.'),
    h('div', { class: 'btns' },
      h('button', { class: fitT.grid ? 'accent' : '', title: 'Клетка 2 см (крупные линии — 10 см) на коже и достроенном теле: видно, где одежда тянется или проваливается',
        onclick: () => { fitT.grid = !fitT.grid; applyFitOverlay(); renderPanel(); } }, '▦ Сетка на теле'),
      h('button', { class: fitT.clear ? 'accent' : '', title: 'Красное — одежда проходит сквозь тело, жёлтое — ближе 5 мм к телу',
        onclick: () => {
          if (!fitT.clear) { const n = computeClearance(); toast(n ? 'Сквозь тело проходит вершин одежды: ' + n + ' (красные)' : 'Одежда нигде не проходит сквозь тело'); }
          fitT.clear = !fitT.clear; applyFitOverlay(); renderPanel(); } }, '◐ Просвет одежды'),
      h('button', { onclick: () => { fitT.meas = measure(); if (!fitT.meas) toast('Не хватает костей для мерок', true); renderPanel(); } }, '📏 Измерить')),
    fitT.clear ? h('div', { class: 'note' }, '🟥 одежда проходит сквозь тело · 🟨 ближе 5 мм · белое — с запасом. Пересчёт: выключите и включите снова.') : null,
    table,
    M ? h('div', { class: 'note' }, 'Обхваты — как сантиметровой лентой (по выпуклому контуру среза 1 см). «По коже» — только кожа и достроенное тело, без одежды.') : null);
}
// rebuilding the body where clothing / shoes / hair hid it: smooth, natural form, skin texture of the model
const BODY_PARTS = { feet: 'Ступни с пальцами', legs: 'Ноги', torso: 'Торс', arms: 'Руки', head: 'Голова (затылок, макушка)' };
function fitGuidesSection() {
  const o = fitGuides.opts, set = (patch, redraw) => { fitGuides.set(patch); if (redraw) renderPanel(); };
  const out = [];
  out.push(h('div', { class: 'note' }, 'Метки на базовом теле для примерки одежды. Только для превью: в VRM не попадают.'));
  out.push(h('div', { class: 'cat-h' }, h('span', { class: 'item-n' }, 'Верх: метки на груди')));
  out.push(h('div', { class: 'note' }, 'Левая и правая точки и центр: видно, на одной ли высоте стоят обе стороны лифа или купальника и одинаково ли они отстоят от центра.'));
  out.push(h('div', { class: 'prow' }, check('Показать метки', o.on, (v) => set({ on: v }, true))));
  if (!o.on) return out;
  if (app.vrm && fitGuides.info && !fitGuides.info.chest) out.push(h('div', { class: 'note warn' }, 'Не нашлись кости груди или плеч — проверьте вкладку «Скелет».'));
  else if (fitGuides.info?.chest && !fitGuides.info.chest.snapped) out.push(h('div', { class: 'note' }, 'Тело не распознано как «Кожа/тело» — метки поставлены по скелету, без прилипания к поверхности.'));
  out.push(h('div', { class: 'row' }, h('label', {}, 'Цвет'),
    h('input', { type: 'color', value: o.color, oninput: (e) => set({ color: e.target.value }) }),
    h('span', { class: 'grow' })));
  out.push(slider('Размер', o.size, 0.8, 6, 0.1, (v) => set({ size: v }), (x) => (+x).toFixed(1) + ' см',
    { lo: 'точка', hi: 'пятно', hint: 'Диаметр метки' }));
  out.push(slider('Мягкость края', o.soft, 0, 1, 0.05, (v) => set({ soft: v }), (x) => Math.round(+x * 100) + '%',
    { lo: 'чёткий', hi: 'размытый', hint: 'Как плавно метка растворяется к краю' }));
  out.push(slider('Высота', o.height, -8, 8, 0.25, (v) => set({ height: v }), (x) => (+x > 0 ? '+' : '') + (+x).toFixed(2) + ' см',
    { lo: 'ниже', hi: 'выше', hint: 'Сдвиг обеих меток относительно автоматической высоты' }));
  out.push(slider('Расстояние', o.spread, 0.6, 1.5, 0.01, (v) => set({ spread: v }), (x) => '×' + (+x).toFixed(2),
    { lo: 'ближе к центру', hi: 'шире', hint: 'Расстояние между левой и правой меткой' }));
  out.push(h('div', { class: 'prow' }, check('Центр и линия уровня', o.center, (v) => set({ center: v }))));
  return out;
}
function fitLowerSection() {
  const o = fitGuides.opts, set = (patch, redraw) => { fitGuides.set(patch); if (redraw) renderPanel(); };
  const out = [h('div', { class: 'cat-h' }, h('span', { class: 'item-n' }, 'Низ: осевая линия'))];
  out.push(h('div', { class: 'note' }, 'Тонкая мягкая линия симметрии по центру в области трусов / нижнего белья — для контроля центровки и посадки низа.'));
  out.push(h('div', { class: 'prow' }, check('Показать осевую линию', o.lower, (v) => set({ lower: v }, true))));
  if (!o.lower) return out;
  if (app.vrm && fitGuides.info && !fitGuides.info.lower) out.push(h('div', { class: 'note warn' }, 'Не нашлись кости таза и бёдер — проверьте вкладку «Скелет».'));
  out.push(h('div', { class: 'row' }, h('label', {}, 'Цвет'),
    h('input', { type: 'color', value: o.lowerColor, oninput: (e) => set({ lowerColor: e.target.value }) }), h('span', { class: 'grow' })));
  out.push(slider('Толщина', o.lowerWidth, 1, 12, 0.5, (v) => set({ lowerWidth: v }), (x) => (+x).toFixed(1) + ' мм',
    { lo: 'тонкая', hi: 'широкая', hint: 'Ширина линии' }));
  out.push(slider('Мягкость края', o.lowerSoft, 0, 1, 0.05, (v) => set({ lowerSoft: v }), (x) => Math.round(+x * 100) + '%',
    { lo: 'чёткая', hi: 'размытая', hint: 'Как плавно линия растворяется к краям' }));
  out.push(slider('Длина', o.lowerLength, 0.4, 1.4, 0.05, (v) => set({ lowerLength: v }), (x) => '×' + (+x).toFixed(2),
    { lo: 'короче', hi: 'длиннее', hint: 'От пояса вниз; линия сама обрывается там, где расходятся ноги' }));
  return out;
}
function fitCommonSection() {
  const o = fitGuides.opts, set = (patch) => fitGuides.set(patch);
  if (!o.on && !o.lower) return [];
  return [h('div', { class: 'cat-h' }, h('span', { class: 'item-n' }, 'Общее')),
    slider('Непрозрачность', o.opacity, 0.15, 1, 0.05, (v) => set({ opacity: v }), (x) => Math.round(+x * 100) + '%'),
    h('div', { class: 'prow' }, check('Видно сквозь одежду', o.onTop, (v) => set({ onTop: v }))),
    h('div', { class: 'btns' }, h('button', { onclick: () => { fitGuides.reset(); renderPanel(); } }, 'Сбросить ориентиры'))];
}
function bodyFillSection() {
  const s = app.settings, on = new Set(Array.isArray(s.body_fill) ? s.body_fill : []);
  const list = h('div', { class: 'checks' });
  for (const [k, label] of Object.entries(BODY_PARTS))
    list.append(check(label, on.has(k), (v) => {
      const cur = new Set(Array.isArray(app.settings.body_fill) ? app.settings.body_fill : []); v ? cur.add(k) : cur.delete(k);
      app.settings.body_fill = Object.keys(BODY_PARTS).filter(x => cur.has(x)); rebuild(0);
    }));
  return h('details', { class: 'sec', open: app.openFill || on.size > 0 },
    h('summary', { onclick: () => { app.openFill = !app.openFill; } }, '🧍 Достроить тело' + (on.size ? ' (' + on.size + ')' : '')),
    h('div', { class: 'note' }, 'Если сняли обувь, головной убор или одежду, а под ними тела нет — программа достроит его по скелету и форме модели, с текстурой её кожи. Достраивается только там, где поверхности нет: под оставшейся кожей и одеждой ничего не меняется.'),
    list,
    h('div', { class: 'note' }, 'Сначала скройте ненужное (✂ разделите материал, если обувь или повязка в одном материале с телом), потом отметьте части. Форма — гладкая, без мелких деталей.'));
}
// ------------------------------------------------------------------ 👕 wardrobe
const WD = { list: null, loading: false, brushUid: null, painting: false, radius: 0.04, strength: 0.004, dir: 1, stroke: [], saveSel: new Set(), saveName: '' };
async function wdLoad(force) {
  if ((WD.list && !force) || WD.loading) return;
  WD.loading = true;
  try { WD.list = await api('/api/wardrobe/list', {}); } catch (e) { toast(e.message, true); }
  WD.loading = false; if (app.tab === 'wardrobe') renderPanel();
}
const wdUid = () => 'w' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
function wdSwatch(it) {
  if (it.src === 'catalog') {
    const c = (WD.list?.catalog || []).find(x => x.uid === it.id);
    return c?.thumb ? '/api/wardrobe/thumb?uid=' + encodeURIComponent(it.id) : '/api/wardrobe/swatch?id=tshirt&color=%23555a66';
  }
  const g = (WD.list?.builtin || []).find(x => x.id === it.id) || {};
  const q = new URLSearchParams({ id: it.id, color: it.color || g.color || '', color2: it.color2 || g.color2 || '', pattern: it.pattern || g.pattern || '' });
  return '/api/wardrobe/swatch?' + q;
}
function wdAdd(it) {
  const s = app.settings; s.wardrobe = Array.isArray(s.wardrobe) ? s.wardrobe : [];
  s.wardrobe.push({ uid: wdUid(), size: 1, shift: 0, ...it });
  renderPanel(); rebuild(0);
}
function wdOwnClothes() { return (app.info?.materials || []).filter(m => m.category === 'clothes' && !m.wardrobe && !/ · хвост$/.test(m.name)); }
function wdStrip(on) {
  const s = app.settings, own = wdOwnClothes();
  for (const m of own) matOv(m.name).visible = !on;
  // the tail / animal ears are body parts: they stay even when they share a material with the clothes
  s.wd_keep = on && s.wd_keep_tail !== false ? own.map(m => m.name) : [];
  const bf = new Set(Array.isArray(s.body_fill) ? s.body_fill : []);
  if (on) for (const p of ['torso', 'legs', 'arms']) bf.add(p);
  s.body_fill = Object.keys(BODY_PARTS).filter(x => bf.has(x));
  applyLiveVisibility(); renderPanel(); rebuild(0);
}
function wardrobePanel() {
  wdLoad();
  const s = app.settings, out = [], items = Array.isArray(s.wardrobe) ? s.wardrobe : [];
  for (const it of items) if (!it.uid) it.uid = wdUid();  // older settings
  out.push(h('div', { class: 'note' }, 'Встроенная одежда строится по фигуре этой модели. Одежду любой модели можно сохранить в «Мои вещи» и надеть на другую — она подстроится под её пропорции и формы.'));
  // ---- the model's own clothes
  const own = wdOwnClothes(), hidden = own.filter(m => app.settings.materials[m.name]?.visible === false).length;
  out.push(h('div', { class: 'cat-h' }, h('span', { class: 'item-n' }, 'Своя одежда модели'), h('span', { class: 'cnt' }, own.length ? `(${own.length - hidden} из ${own.length} видно)` : '(нет)')));
  out.push(h('div', { class: 'btns' },
    h('button', { disabled: !own.length || hidden === own.length, title: 'Скрыть одежду модели и достроить тело под ней (торс, руки, ноги)', onclick: () => wdStrip(true) }, '🙈 Снять одежду модели'),
    h('button', { disabled: !hidden, onclick: () => wdStrip(false) }, '↺ Вернуть')));
  out.push(h('div', { class: 'prow', title: 'Хвост и звериные уши — часть тела: если они в одном материале с одеждой, при снятии одежды они останутся. Снимите галочку, чтобы убрать их вместе с одеждой.' },
    check('Хвост и уши остаются', s.wd_keep_tail !== false, (v) => {
      s.wd_keep_tail = v;
      if (hidden) { s.wd_keep = v ? own.filter(m => app.settings.materials[m.name]?.visible === false).map(m => m.name) : []; rebuild(0); }
    })));
  // ---- worn
  out.push(h('div', { class: 'cat-h' }, h('span', { class: 'item-n' }, 'Надето из гардероба'), h('span', { class: 'cnt' }, items.length ? `(${items.length})` : '')));
  if (!items.length) out.push(h('div', { class: 'note' }, 'Ничего. Выберите вещь в каталоге ниже.'));
  else if (items.length > 1) out.push(h('div', { class: 'note' }, 'Порядок = слои: вещь ниже в списке надета поверх (↑↓ — переставить). Бельё всегда снизу.'));
  for (const it of items) out.push(wdItemEditor(it));
  // ---- catalog
  if (!WD.list) { out.push(h('div', { class: 'note' }, 'Загружаю каталог…')); return out; }
  const tops = WD.list.builtin.filter(g => g.kind === 'top'), bottoms = WD.list.builtin.filter(g => g.kind === 'bottom');
  const tile = (g, src) => h('button', { class: 'wd-tile', title: 'Надеть', onclick: () => wdAdd(src === 'catalog' ? { src, id: g.uid, label: g.name } : { src, id: g.id }) },
    h('img', { src: src === 'catalog' ? (g.thumb ? '/api/wardrobe/thumb?uid=' + encodeURIComponent(g.uid) : '/api/wardrobe/swatch?id=tshirt&color=%23555a66') : wdSwatch({ src, id: g.id }), loading: 'lazy' }),
    g.name,
    src === 'catalog' ? h('button', { class: 'wd-del icon', title: 'Удалить из каталога', onclick: async (e) => {
      e.stopPropagation(); if (!confirm('Удалить «' + g.name + '» из каталога?')) return;
      await api('/api/wardrobe/delete', { uid: g.uid }); await wdLoad(true); } }, '✕') : null);
  out.push(h('details', { class: 'sec', open: app.openWdTop !== false }, h('summary', { onclick: (e) => { app.openWdTop = !e.currentTarget.parentNode.open; } }, '👚 Верх — встроенное (' + tops.length + ')'),
    h('div', { class: 'wd-grid' }, ...tops.map(g => tile(g, 'builtin')))));
  out.push(h('details', { class: 'sec', open: app.openWdBot !== false }, h('summary', { onclick: (e) => { app.openWdBot = !e.currentTarget.parentNode.open; } }, '👖 Низ — встроенное (' + bottoms.length + ')'),
    h('div', { class: 'wd-grid' }, ...bottoms.map(g => tile(g, 'builtin')))));
  const cat = WD.list.catalog || [];
  out.push(h('details', { class: 'sec', open: true }, h('summary', {}, '🧳 Мои вещи (' + cat.length + ')'),
    cat.length ? h('div', { class: 'wd-grid' }, ...cat.map(g => tile(g, 'catalog'))) : h('div', { class: 'note' }, 'Пока пусто. Сохраните одежду этой модели в каталог (ниже) — потом её можно надеть на другую модель.'),
    h('div', { class: 'note' }, 'Папка каталога: ' + (WD.list.folder || ''))));
  // ---- save this model's clothes
  const cands = (app.info.materials || []).filter(m => m.category !== 'skin' && m.category !== 'face' && m.category !== 'eyes' && m.category !== 'hair');
  const box = h('div', { class: 'checks' });
  for (const m of cands) box.append(check(matTitle(m) + (m.wardrobe ? ' (из гардероба)' : ''), WD.saveSel.has(m.name), (v) => { v ? WD.saveSel.add(m.name) : WD.saveSel.delete(m.name); }));
  const nm = h('input', { value: WD.saveName, placeholder: 'название, например «Рубашка Минт»', style: 'flex:1', oninput: (e) => { WD.saveName = e.target.value; } });
  out.push(h('details', { class: 'sec' }, h('summary', {}, '💾 Сохранить одежду этой модели в «Мои вещи»'),
    h('div', { class: 'note' }, 'Отметьте части одежды. Если в одном материале и одежда, и тело или хвост — сначала разделите материал (✂ на вкладке «Части»). Миниатюра — текущий вид окна.'),
    box, h('div', { class: 'row' }, h('label', {}, 'Название'), nm),
    h('div', { class: 'btns' }, h('button', { class: 'accent', onclick: wdSave }, '💾 В каталог'))));
  return out;
}
function wdItemEditor(it) {
  const g = it.src === 'builtin' ? (WD.list?.builtin || []).find(x => x.id === it.id) || {} : {};
  const title = it.src === 'builtin' ? (g.name || it.id) : (it.label || (WD.list?.catalog || []).find(x => x.uid === it.id)?.name || 'вещь из каталога');
  const brushing = WD.brushUid === it.uid;
  const set = (k, v, now) => { it[k] = v; if (now) rebuild(0); else { clearTimeout(WD.t); WD.t = setTimeout(() => rebuild(0), 500); } };
  const el = h('div', { class: 'wd-item' + (brushing ? ' brush' : '') },
    h('div', { class: 'wd-head' }, h('img', { src: wdSwatch(it) }), h('b', {}, title),
      h('button', { class: 'icon', title: 'Глубже (под другими вещами)', onclick: () => wdMove(it, -1) }, '↑'),
      h('button', { class: 'icon', title: 'Поверх других вещей', onclick: () => wdMove(it, 1) }, '↓'),
      h('button', { class: brushing ? 'accent' : '', title: 'Кисть: ведите по вещи, чтобы отодвинуть её от тела или подтянуть к нему', onclick: () => wdBrush(brushing ? null : it.uid) }, '🖌'),
      h('button', { class: 'icon', title: 'Снять', onclick: () => { if (brushing) wdBrush(null); app.settings.wardrobe = app.settings.wardrobe.filter(x => x !== it); renderPanel(); rebuild(0); } }, '✕')));
  if (it.src === 'builtin') {
    el.append(h('div', { class: 'wd-colors' }, h('label', {}, 'Цвет'),
      h('input', { type: 'color', value: it.color || g.color || '#ffffff', onchange: (e) => { set('color', e.target.value, true); renderPanel(); } }),
      h('input', { type: 'color', value: it.color2 || g.color2 || '#333333', title: 'второй цвет (полоска, клетка, горошек)', onchange: (e) => { set('color2', e.target.value, true); renderPanel(); } }),
      selectEl(WD.list?.patterns || {}, it.pattern || g.pattern || 'solid', (v) => { set('pattern', v, true); renderPanel(); })));
  }
  el.append(slider('Размер', it.size ?? 1, 0.7, 1.6, 0.02, (v) => set('size', v), (x) => '×' + (+x).toFixed(2), { lo: 'в обтяжку', hi: 'больше', hint: 'Насколько вещь отстоит от тела' }));
  el.append(slider('Свободнее', it.loose ?? (it.src === 'builtin' ? 0.3 : 0), 0, 1.6, 0.05, (v) => set('loose', v), (x) => Math.round(+x * 100) + '%', { lo: 'по фигуре', hi: 'свободно', hint: 'Свободная вещь не повторяет изгибы, а висит' }));
  el.append(slider('Сдвиг', it.shift ?? 0, -0.15, 0.15, 0.005, (v) => set('shift', v), (x) => (+x > 0 ? '↑' : +x < 0 ? '↓' : '') + Math.abs(Math.round(+x * 100)) + '%', { lo: 'ниже', hi: 'выше', hint: 'Сдвинуть вещь вверх или вниз по телу' }));
  if (it.src === 'builtin') el.append(slider('Длина', it.length ?? 1, 0.4, 1.6, 0.02, (v) => set('length', v), (x) => '×' + (+x).toFixed(2), { lo: 'короче', hi: 'длиннее', hint: 'Длина рукавов, штанин и низа' }));
  if (brushing) {
    el.append(h('div', { class: 'note' }, '🖌 Ведите по вещи левой кнопкой. Модель стоит в T-позе; вращать — правой кнопкой мыши. Мазков: ' + (it.brush?.length || 0)),
      h('div', { class: 'seg' }, h('button', { class: WD.dir > 0 ? 'on' : '', onclick: () => { WD.dir = 1; renderPanel(); } }, '↥ Отодвинуть от тела'),
        h('button', { class: WD.dir < 0 ? 'on' : '', onclick: () => { WD.dir = -1; renderPanel(); } }, '↧ Подтянуть к телу')),
      slider('Радиус', WD.radius * 100, 1, 15, 0.5, (v) => { WD.radius = v / 100; }, (x) => (+x).toFixed(1) + ' см'),
      slider('Сила', WD.strength * 1000, 0.5, 8, 0.5, (v) => { WD.strength = v / 1000; }, (x) => (+x).toFixed(1) + ' мм'),
      h('div', { class: 'btns' }, h('button', { disabled: !it.brush?.length, onclick: () => { it.brush = []; renderPanel(); rebuild(0); } }, 'Стереть мазки'),
        h('button', { onclick: () => wdBrush(null) }, 'Готово')));
  }
  return el;
}
function wdMove(it, d) {
  const a = app.settings.wardrobe, i = a.indexOf(it), j = i + d;
  if (i < 0 || j < 0 || j >= a.length) return;
  [a[i], a[j]] = [a[j], a[i]]; renderPanel(); rebuild(0);
}
async function wdSave() {
  if (!WD.saveSel.size) { toast('Отметьте части одежды', true); return; }
  let thumb = null;
  try {
    const c = renderer.domElement, w = c.width, hh = c.height, sz = Math.min(w, hh);
    const t = document.createElement('canvas'); t.width = t.height = 256;
    t.getContext('2d').drawImage(c, (w - sz) / 2, (hh - sz) / 2, sz, sz, 0, 0, 256, 256);
    thumb = t.toDataURL('image/png');
  } catch (e) {}
  setBusy(true, 'Сохраняю одежду в каталог…');
  try {
    const r = await api('/api/wardrobe/save', { name: WD.saveName.trim(), materials: [...WD.saveSel], thumb, settings: app.settings });
    toast((r.log || []).join('\n') || 'Сохранено', false, 8000);
    WD.saveSel.clear(); WD.saveName = ''; await wdLoad(true);
  } catch (e) { toast(e.message, true, 8000); }
  setBusy(false);
}
// ---- brush: strokes in the preview space (T-pose, VRM 1.0 axes, metres); the server bakes them
function wdBrush(uid) {
  WD.brushUid = uid;
  if (uid) {
    app.anim = 'tpose'; pivot.rotation.y = 0;
    document.querySelectorAll('#anim button').forEach(x => x.classList.toggle('on', x.dataset.anim === 'tpose'));
    controls.mouseButtons.LEFT = null; controls.mouseButtons.RIGHT = THREE.MOUSE.ROTATE;
  } else {
    controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE; controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
    const c = $('#brushCursor'); if (c) c.style.display = 'none';
  }
  renderPanel();
}
function wdMeshes(uid) {
  const out = [];
  for (const m of app.info?.materials || []) if (m.wardrobe === uid && app.matMeshes.has(m.gltf_index)) out.push(...app.matMeshes.get(m.gltf_index));
  return out;
}
function brushHit(e) {
  const r = canvas.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  // only where the garment is the visible surface (not through the clothes on top of it)
  const all = []; app.vrm.scene.traverse(o => { if (o.isMesh && o.visible) all.push(o); });
  const hit = ray.intersectObjects(all, false)[0];
  return hit && wdMeshes(WD.brushUid).includes(hit.object) ? hit : null;
}
function brushCursor(e, hit) {
  let c = $('#brushCursor');
  if (!c) { c = h('div', { id: 'brushCursor' }); $('#view').append(c); }
  if (!hit) { c.style.display = 'none'; return; }
  const r = canvas.getBoundingClientRect(), vr = $('#view').getBoundingClientRect();
  const p2 = hit.point.clone().add(new THREE.Vector3(WD.radius, 0, 0).applyQuaternion(camera.quaternion)).project(camera);
  const px = (p2.x + 1) / 2 * r.width + r.left, py = (1 - p2.y) / 2 * r.height + r.top;
  const rad = Math.hypot(px - e.clientX, py - e.clientY);
  Object.assign(c.style, { display: 'block', left: (e.clientX - vr.left) + 'px', top: (e.clientY - vr.top) + 'px', width: 2 * rad + 'px', height: 2 * rad + 'px' });
}
function brushDab(hit) {
  const it = (app.settings.wardrobe || []).find(x => x.uid === WD.brushUid); if (!it) return;
  const last = WD.stroke[WD.stroke.length - 1];
  if (last && hit.point.distanceTo(last) < WD.radius * 0.6) return;
  if ((it.brush?.length || 0) >= 3000) { if (!WD.capWarned) { toast('Слишком много мазков (3000) — сотрите часть или увеличьте радиус', true); WD.capWarned = true; } return; }
  WD.stroke.push(hit.point.clone());
  const st = { p: [+hit.point.x.toFixed(4), +hit.point.y.toFixed(4), +hit.point.z.toFixed(4)], r: WD.radius, s: WD.dir * WD.strength };
  (it.brush ||= []).push(st);
  // live: move the vertices of the mesh right away (the server repeats it exactly on the next rebuild)
  for (const mesh of wdMeshes(WD.brushUid)) {
    const pos = mesh.geometry.attributes.position, nor = mesh.geometry.attributes.normal;
    const lp = mesh.worldToLocal(hit.point.clone()), v = new THREE.Vector3(), n = new THREE.Vector3();
    const sc = mesh.getWorldScale(new THREE.Vector3()).x || 1, rr = WD.radius / sc;
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i); const d = v.distanceTo(lp); if (d >= rr) continue;
      const w = 1 - d / rr, f = st.s / sc * w * w * (3 - 2 * w);
      n.fromBufferAttribute(nor, i); v.addScaledVector(n, f); pos.setXYZ(i, v.x, v.y, v.z);
    }
    pos.needsUpdate = true; mesh.geometry.computeBoundingSphere();
  }
}
function brushDown(e) {
  const hit = brushHit(e); if (!hit) return;
  WD.painting = true; WD.stroke = []; brushDab(hit);
}
function brushMove(e) {
  const hit = brushHit(e); brushCursor(e, hit);
  if (WD.painting && hit && (e.buttons & 1)) brushDab(hit);
}
function brushUp() {
  WD.painting = false; WD.stroke = [];
  clearTimeout(WD.bt); WD.bt = setTimeout(() => { renderPanel(); rebuild(0); }, 900);
}
function partsPanel() {
  const I = app.info, out = [];
  out.push(h('div', { class: 'note' }, 'Части разложены автоматически по названиям материалов и текстур. Клик по модели выбирает часть. Категорию, видимость и прозрачность можно поменять.'));
  out.push(bodyFillSection());
  out.push(fitSection());
  out.push(h('details', { class: 'sec', open: app.openGuides }, h('summary', { onclick: () => { app.openGuides = !app.openGuides; } }, '📍 Ориентиры посадки одежды'), ...fitGuidesSection(), ...fitLowerSection(), ...fitCommonSection()));
  const bycat = {};
  for (const m of I.materials) (bycat[m.category] ||= []).push(m);
  for (const cat of Object.keys(I.categories)) {
    const ms = bycat[cat]; if (!ms) continue;
    const allHidden = ms.every(m => app.settings.materials[m.name]?.visible === false);
    const head = h('div', { class: 'cat-h' },
      h('span', { class: 'item-n', style: 'cursor:pointer', onclick: () => select({ type: 'cat', id: cat }) }, I.categories[cat]),
      h('span', { class: 'cnt' }, `(${ms.length})`), h('span', { class: 'grow' }),
      check('показать', !allHidden, (v) => { for (const m of ms) matOv(m.name).visible = v; applyLiveVisibility(); renderPanel(); rebuild(); }));
    const list = h('div', { class: 'list' });
    for (const m of ms) {
      const ov = app.settings.materials[m.name] || {};
      const sel = app.sel?.type === 'mat' && app.sel.id === m.name;
      list.append(h('div', { class: 'item' + (sel ? ' sel' : ''), onclick: () => select({ type: 'mat', id: m.name }) },
        h('input', { type: 'checkbox', checked: ov.visible !== false, onclick: (e) => e.stopPropagation(),
          onchange: (e) => { matOv(m.name).visible = e.target.checked; applyLiveVisibility(); renderPanel(); rebuild(); } }),
        h('span', { class: 'name', title: m.texture }, matTitle(m),
          !m.piece_of && m.name_ru && m.name_ru !== m.name ? h('span', { class: 'sub' }, '  ' + m.name) : null,
          h('span', { class: 'sub' }, '  ' + (m.faces || 0) + ' полиг.')),
        m.piece_of ? h('span', { class: 'tag k-piece', title: 'кусок материала «' + m.piece_of + '»' }, '✂') : null,
        m.as_skin ? h('span', { class: 'tag k-flesh', title: 'покрашено как «' + m.as_skin + '»' }, '🎨') : null,
        m.sublabel ? h('span', { class: 'tag', title: 'распознано по слову: ' + (m.matched || '') }, I.ru_labels?.[m.sublabel] || m.sublabel) : null,
        m.blend && m.blend !== 'opaque' ? h('span', { class: 'tag' }, m.blend === 'cutout' ? 'вырез' : 'прозр.') : null));
      if (sel) list.append(matEditor(m));
    }
    out.push(h('div', { class: 'cat' }, head, list));
  }
  out.push(h('details', { class: 'sec', open: app.openQuality }, h('summary', { onclick: () => { app.openQuality = !app.openQuality; } }, 'Качество картинки: текстуры, ✨ ИИ-улучшение, тени лица'), ...qualitySection()));
  return out;
}
function qualitySection() {
  const s = app.settings, out = [];
  out.push(h('h3', {}, 'Качество картинки'));
  out.push(h('div', { class: 'row' }, h('label', {}, 'Текстуры'),
    selectEl({ normal: 'Обычные (JPEG, до 2048 px)', best: 'Максимальные (без сжатия, до 4096 px)' }, s.tex_quality || 'normal',
      (v) => { s.tex_quality = v; s.max_texture = v === 'best' ? 4096 : 2048; rebuild(0); })));
  out.push(h('div', { class: 'note' }, 'Края прозрачных текстур (волосы, ресницы, кружево) всегда очищаются от тёмной каймы, уменьшенные текстуры подтягиваются по резкости.'));
  out.push(h('div', { class: 'prow', title: 'Подчёркивает линии и мелкие детали на всех текстурах. Работает без видеокарты.' },
    slider('Резкость текстур', s.tex_sharpen || 0, 0, 1, 0.05, (v) => { s.tex_sharpen = v; }, (x) => Math.round(+x * 100) + '%'),
    h('div', { class: 'ends' }, h('span', {}, '◂ как в исходнике'), h('span', {}, 'очень чётко ▸'))));
  const sharpRow = out[out.length - 1].querySelector('input');
  sharpRow.addEventListener('change', () => rebuild(0));
  app.upModel ||= (s.upscale === 'best' ? 'best' : 'fast');
  out.push(h('div', { class: 'row' }, h('label', {}, 'Нейросеть'),
    selectEl({ fast: 'Быстрая (аниме, x2–x4)', best: 'Максимальное качество (аниме-рисунок, медленнее)' }, app.upModel,
      (v) => { app.upModel = v; if (s.upscale) { s.upscale = v; rebuild(0); } })));
  const upBtn = h('button', { class: 'accent', onclick: async () => {
    setBusy(true, 'ИИ-улучшение текстур (Real-ESRGAN)… при первом запуске скачивается ~45 МБ');
    try {
      const r = await api('/api/upscale', { model: app.upModel });
      s.upscale = app.upModel; rebuild(0);
      toast(`Улучшено текстур: ${r.done}` + (r.skipped ? `, пропущено крупных (≥2048 px): ${r.skipped}` : ''), false, 7000);
    } catch (e) { toast(e.message, true, 9000); }
    setBusy(false);
  } }, '✨ Улучшить текстуры (ИИ)');
  out.push(h('div', { class: 'btns' }, upBtn,
    s.upscale ? check('использовать улучшенные', true, (v) => { s.upscale = v; rebuild(0); }) : null));
  out.push(h('div', { class: 'note' }, 'Нейросеть Real-ESRGAN увеличивает мелкие текстуры в 2–4 раза с дорисовкой деталей; края прозрачных частей (кончики волос, кружево, ресницы) тоже увеличиваются нейросетью — без размытой каймы. Нужна видеокарта с Vulkan; результат запоминается и для других моделей с теми же текстурами. Без видеокарты помогает «Резкость текстур».'));
  out.push(h('div', { class: 'row' }, h('label', {}, ''), check('Плоские аниме-тени на лице', s.anime_face !== false, (v) => { s.anime_face = v; rebuild(0); })));
  out.push(h('div', { class: 'row' }, h('label', {}, ''), check('Контур как в MMD (толщина и цвет из PMX)', s.outline, (v) => { s.outline = v; rebuild(0); })));
  return out;
}
function matTitle(m) {
  if (m.wardrobe) return '👕 ' + m.name.replace(/^WD /, '');
  if (m.piece_of) {
    const bn = m.piece_of_ru || m.piece_of;
    return bn + ' · ' + m.name.split('#').pop() + ' — ' + (m.piece_label || '');
  }
  return m.name_ru && m.name_ru !== m.name ? m.name_ru : m.name;
}
function setSplit(name, on) {
  app.settings.splits ||= {};
  if (on) app.settings.splits[name] = true;
  else {
    delete app.settings.splits[name];
    for (const k of Object.keys(app.settings.materials)) if (k.startsWith(name + '#')) delete app.settings.materials[k];
  }
  app.sel = null; rebuild(0);
  toast(on ? 'Материал разделён на куски: кликните по нужному куску на модели и скройте или настройте его отдельно.' : 'Куски собраны обратно в один материал.', false, 6000);
}
// "make it skin": the piece keeps its shape (a swimsuit cup is the breast surface itself) but is painted
// in the tone of another material — no hole where clothing was the body
function skinRow(m, ov) {
  const I = app.info, others = I.materials.filter(x => x.name !== m.name && x.visible !== false);
  const skins = [...others.filter(x => x.category === 'skin'), ...others.filter(x => x.category === 'face' && !x.sublabel)];
  const opts = { '': '— как есть —' };
  for (const x of [...skins, ...others.filter(x => !skins.includes(x))]) opts[x.name] = (skins.includes(x) ? '🧴 ' : '') + matTitle(x);
  return h('div', {},
    h('div', { class: 'row', title: 'Деталь остаётся на месте и сохраняет форму, но красится в цвет выбранного материала. Нужно, когда одежда и есть поверхность тела: скрытый купальник оставил бы дыру, а повязка — пустоту на хвосте.' },
      h('label', {}, 'Покрасить как'),
      selectEl(opts, ov.as_skin || '', (v) => { if (v) { ov.as_skin = v; ov.visible = true; } else delete ov.as_skin; rebuild(); })),
    h('div', { class: 'note' }, ov.as_skin ? '🎨 Сейчас эта деталь покрашена как «' + (matTitle(I.materials.find(x => x.name === ov.as_skin) || { name: ov.as_skin })) + '»: форма та же, текстура — ровный цвет.'
      : 'Если при скрытии детали появляется дыра (одежда была частью тела) — не скрывайте её, а выберите «Покрасить как» кожу или соседнюю часть.'));
}
function matEditor(m) {
  const ov = matOv(m.name);
  return h('div', { class: 'edit' },
    h('div', { class: 'title' }, matTitle(m), m.name_en && !m.piece_of ? h('span', { class: 'muted' }, ' · ' + m.name_en) : null),
    h('div', { class: 'note' }, 'Текстура: ' + (m.texture || '—')),
    m.n_pieces > 1 ? h('div', { class: 'btns' }, h('button', { class: 'accent', title: 'Например, все аксессуары в одном меше: после разделения каждый кусок можно скрыть или настроить отдельно',
      onclick: () => setSplit(m.name, true) }, '✂ Разделить на куски (' + m.n_pieces + ')')) : null,
    m.piece_of ? h('div', { class: 'note' }, '✂ Кусок материала «' + m.piece_of + '» — держится на кости «' + (m.piece_label || '?') + '».') : null,
    m.piece_of ? h('div', { class: 'btns' }, h('button', { onclick: () => setSplit(m.piece_of, false) }, 'Собрать куски обратно')) : null,
    skinRow(m, ov),
    h('div', { class: 'row' }, h('label', {}, 'Категория'),
      selectEl(app.info.categories, ov.category || m.category, (v) => { ov.category = v; m.category = v; renderPanel(); rebuild(); })),
    h('div', { class: 'row' }, h('label', {}, 'Прозрачность'),
      selectEl(BLEND, ov.blend || m.blend || 'opaque', (v) => { ov.blend = v; rebuild(); })),
    h('div', { class: 'row' }, h('label', {}, ''), check('двусторонний', ov.double_sided ?? m.double_sided, (v) => { ov.double_sided = v; rebuild(); })),
    h('div', { class: 'row' }, h('label', {}, ''), check('контур', ov.outline ?? m.outline, (v) => { ov.outline = v; rebuild(); })),
    h('div', { class: 'btns' }, h('button', { onclick: () => { delete app.settings.materials[m.name]; rebuild(); } }, 'Сбросить к авто')));
}

function physicsPanel() {
  const I = app.info, out = [];
  out.push(h('div', { class: 'row' }, h('label', {}, 'Общий стиль'),
    selectEl(I.physics_presets, app.settings.physics, (v) => {
      const tuned = Object.values(app.settings.chains).some(o => ['stiffness', 'gravity', 'drag', 'hitRadius'].some(k => k in o));
      if (tuned && !confirm('Сменить стиль физики? Ваши ползунки жёсткости/гравитации/затухания у цепочек будут заменены значениями стиля. Включение/выключение цепочек и типы сохранятся.')) { renderPanel(); return; }
      app.settings.physics = v;
      for (const o of Object.values(app.settings.chains)) for (const k of ['stiffness', 'gravity', 'drag', 'hitRadius', 'preset']) delete o[k];
      rebuild(0); })));
  out.push(h('div', { class: 'btns' }, h('button', { class: 'accent', title: 'Программа прогоняет каждую группу прядей через симуляцию ходьбы и поворота и подбирает настройки, при которых пряди не проходят сквозь тело, не болтаются пружиной и держат форму', onclick: async () => {
    setBusy(true, 'Автоподбор физики: симуляция движений…');
    try {
      const r = await api('/api/autotune', { settings: app.settings });
      for (const [id, v] of Object.entries(r.chains || {})) Object.assign(chainOv(id), v, { preset: 'auto' });
      rebuild(0);
      toast('Подобрано для ' + Object.keys(r.chains || {}).length + ' цепочек. Ручные ползунки по-прежнему работают.', false, 7000);
    } catch (e) { toast(e.message, true); }
    setBusy(false);
  } }, '🤖 Автоподбор физики')));
  out.push(h('details', { class: 'sec help' }, h('summary', {}, 'Волосы или юбка проходят сквозь тело?'),
    h('div', { class: 'note' }, '1) Нажмите «🤖 Автоподбор физики». 2) Включите «коллайдеры» внизу экрана — увидите, чем тело отталкивает пряди. ' +
      '3) Выберите прядь и увеличьте «Толщина пряди» — она будет держаться дальше от тела. 4) Проверьте кнопками «Идёт» и «Кружится».'),
    h('div', { class: 'btns' }, h('button', { onclick: () => { const c = $('#showColliders'); c.checked = !c.checked; c.dispatchEvent(new Event('change')); } }, 'Показать / скрыть коллайдеры'))));
  out.push(h('div', { class: 'row' }, check('⇆ Менять левую и правую сторону вместе', mirrorEdits, (v) => { mirrorEdits = v; })));
  out.push(h('div', { class: 'note' }, 'Кликните по волосам, юбке или хвосту на модели — цепочка подсветится и откроется в списке. Пряди одной причёски собраны в группы.'));
  if (!I.chains?.length) { out.push(h('div', { class: 'muted' }, 'В модели нет физики.')); return out; }
  const bykind = {};
  const q = (app.physFilter || '').toLowerCase();
  out.push(h('input', { class: 'search', placeholder: 'Поиск цепочки…', value: app.physFilter || '',
    oninput: (e) => { app.physFilter = e.target.value; const pos = e.target.selectionStart; renderPanel(); const el = $('#panel .search'); el.focus(); el.setSelectionRange(pos, pos); } }));
  for (const c of I.chains) if (!q || c.id.toLowerCase().includes(q) || ru(c.id).toLowerCase().includes(q)) (bykind[c.kind] ||= []).push(c);
  app.physOpen ||= {};
  for (const [kind, cs] of Object.entries(bykind)) {
    const selK = app.sel?.type === 'kind' && app.sel.id === kind;
    const open = q || app.physOpen[kind] !== false;
    out.push(h('div', { class: 'cat-h fold' + (open ? ' open' : '') },
      h('span', { class: 'arrow', onclick: () => { app.physOpen[kind] = !open; renderPanel(); } }, open ? '▾' : '▸'),
      h('span', { class: 'tag k-' + kind, onclick: () => { app.physOpen[kind] = !open; renderPanel(); } }, I.kinds[kind]),
      h('span', { class: 'cnt' }, `${cs.length} цепоч.`),
      h('span', { class: 'grow' }),
      h('button', { onclick: () => select(selK ? null : { type: 'kind', id: kind }) }, selK ? 'Свернуть' : 'Настроить все')));
    if (selK) out.push(groupEditor(kind, cs));
    if (!open) continue;
    const list = h('div', { class: 'list' });
    const fams = {};
    for (const c of cs) (fams[chainFamily(c.id)] ||= []).push(c);
    const multi = Object.keys(fams).length > 1;
    const ordered = multi ? Object.entries(fams).flatMap(([f, fc]) => fc.map((c, i) => [c, i === 0 ? [f, fc] : null])) : cs.map(c => [c, null]);
    for (const [c, famHead] of ordered) {
      if (famHead && famHead[1].length > 1) {
        const [f, fc] = famHead, fid = kind + '|' + f;
        const selF = app.sel?.type === 'family' && app.sel.id === fid;
        list.append(h('div', { class: 'fam-h' + (selF ? ' sel' : '') },
          h('span', { class: 'name' }, famName(fc[0].id) || f, h('span', { class: 'sub' }, '  ' + f)),
          h('span', { class: 'cnt' }, fc.length + ' шт.'),
          h('button', { onclick: () => select(selF ? null : { type: 'family', id: fid }) }, selF ? 'Свернуть' : 'Настроить группу')));
        if (selF) list.append(groupEditor(kind, fc, 'Группа «' + f + '» — ' + fc.length + ' цепочек'));
      }
      const sel = app.sel?.type === 'chain' && app.sel.id === c.id;
      list.append(h('div', { class: 'item' + (sel ? ' sel' : ''), onclick: () => select({ type: 'chain', id: c.id }) },
        h('input', { type: 'checkbox', checked: c.enabled, title: 'включена', onclick: (e) => e.stopPropagation(),
          onchange: (e) => { chainOv(c.id).enabled = e.target.checked; rebuild(); } }),
        h('span', { class: 'name', title: c.id }, ru(c.id)),
        h('span', { class: 'sub' }, c.nodes.length + ' кост.' + (c.branch_of ? ' · на «' + ru(c.branch_of) + '»' : '')),
        c.kind !== c.auto_kind ? h('span', { class: 'tag', title: 'тип изменён вручную' }, '✎') : null));
      if (sel) list.append(chainEditor(c));
    }
    out.push(list);
  }
  return out;
}
// natural names of the 4 spring parameters for each kind of part: [label, left end, right end, hint]
const PARAM_TEXT = {
  hair: { stiffness: ['Упругость прядей', 'мягкие', 'держат форму', 'Как быстро прядь возвращается в причёску'],
          gravity: ['Тяжесть волос', 'лёгкие, парят', 'тяжёлые, висят', 'Насколько пряди тянет вниз'],
          drag: ['Плавность', 'качаются долго', 'сразу успокаиваются', 'Гасит раскачку: больше — концы не «пружинят»'],
          hitRadius: ['Толщина пряди', 'тонкая', 'толстая', 'Насколько далеко прядь держится от тела и головы'] },
  tail: { stiffness: ['Упругость хвоста', 'вялый', 'пружинит назад', 'Сила, с которой хвост возвращается в свою форму'],
          gravity: ['Вес хвоста', 'лёгкий', 'тяжёлый', 'Насколько хвост тянет вниз'],
          drag: ['Гашение раскачки', 'машет долго', 'сразу затихает', 'Больше — хвост не болтается после движения'],
          hitRadius: ['Толщина хвоста', 'тонкий', 'толстый', 'Отступ хвоста от ног и тела при столкновении'] },
  skirt: { stiffness: ['Плотность ткани', 'лёгкая, летит', 'держит форму', 'Плиссе и плотные юбки — больше'],
           gravity: ['Тяжесть подола', 'лёгкий', 'тяжёлый', 'Насколько подол тянет вниз'],
           drag: ['Сопротивление воздуха', 'развевается', 'спокойная', 'Гасит колыхание ткани'],
           hitRadius: ['Отступ от ног', 'вплотную', 'далеко', 'Больше — ноги реже проходят сквозь подол'] },
  cloth: { stiffness: ['Плотность ткани', 'мягкая', 'держит форму', 'Рукава, воротник, полы плаща'],
           gravity: ['Тяжесть ткани', 'лёгкая', 'тяжёлая', 'Насколько ткань тянет вниз'],
           drag: ['Сопротивление воздуха', 'колышется', 'спокойная', 'Гасит колыхание ткани'],
           hitRadius: ['Отступ от тела', 'вплотную', 'далеко', 'Больше — тело реже проходит сквозь ткань'] },
  ribbon: { stiffness: ['Упругость ленты', 'мягкая', 'пружинит', 'Банты — больше, длинные ленты — меньше'],
            gravity: ['Тяжесть ленты', 'лёгкая', 'тяжёлая', 'Насколько лента тянет вниз'],
            drag: ['Плавность', 'вьётся долго', 'сразу успокаивается', 'Гасит раскачку ленты'],
            hitRadius: ['Ширина ленты', 'узкая', 'широкая', 'Отступ ленты от тела при столкновении'] },
  accessory: { stiffness: ['Жёсткость крепления', 'болтается', 'закреплено', 'Насколько подвеска держит своё положение'],
               gravity: ['Вес подвески', 'лёгкая', 'тяжёлая', 'Насколько подвеску тянет вниз'],
               drag: ['Гашение раскачки', 'качается долго', 'сразу затихает', 'Больше — не раскачивается после движения'],
               hitRadius: ['Размер подвески', 'маленькая', 'большая', 'Отступ от тела при столкновении'] },
  breast: { stiffness: ['Упругость', 'мягкая', 'упругая', 'Как быстро возвращается в форму'],
            gravity: ['Провисание', 'нет', 'сильное', 'Насколько тянет вниз'],
            drag: ['Затухание колебаний', 'колышется', 'почти неподвижна', 'Гасит колебания'],
            hitRadius: ['Радиус', 'маленький', 'большой', 'Размер для столкновений'] },
  flesh: { stiffness: ['Плотность тела', 'мягкое', 'плотное', 'Бёдра, икры, живот: насколько мягко покачиваются'],
           gravity: ['Провисание', 'нет', 'заметное', 'Обычно 0: тело не должно проседать'],
           drag: ['Затухание колебаний', 'колышется', 'почти неподвижно', 'Гасит колебания'],
           hitRadius: ['Радиус', 'маленький', 'большой', 'Размер для столкновений'] },
  ear: { stiffness: ['Жёсткость ушей', 'мягкие, висят', 'стоят', 'Насколько уши держат форму'],
         gravity: ['Провисание', 'нет', 'сильное', 'Насколько уши тянет вниз'],
         drag: ['Затухание', 'дрожат долго', 'сразу успокаиваются', 'Гасит покачивание ушей'],
         hitRadius: ['Толщина уха', 'тонкое', 'толстое', 'Отступ уха от головы при столкновении'] },
};
PARAM_TEXT.other = { stiffness: ['Жёсткость', 'мягко', 'держит форму', 'Как быстро часть возвращается в форму'],
                     gravity: ['Тяжесть', 'лёгкая', 'тяжёлая', 'Насколько тянет вниз'],
                     drag: ['Плавность', 'качается долго', 'сразу успокаивается', 'Гасит раскачку'],
                     hitRadius: ['Толщина', 'тонкая', 'толстая', 'Отступ от тела при столкновении'] };
function paramText(kind, key) { return (PARAM_TEXT[kind] || PARAM_TEXT.other)[key]; }
function paramSliders(get, set, kind) {
  const one = (key, min, max, step, fmt) => {
    const [label, lo, hi, hint] = paramText(kind, key);
    return slider(label, get(key), min, max, step, (v) => set(key, v), fmt, { lo, hi, hint });
  };
  return [
    one('stiffness', 0, 8, 0.05), one('gravity', 0, 2, 0.02), one('drag', 0, 1, 0.01),
    one('hitRadius', 0, 0.1, 0.001, (x) => (+x * 100).toFixed(1) + ' см'),
  ];
}
const PARAM_HINT = 'Наведите на название параметра — появится подсказка. Подписи под ползунком показывают, что будет слева и справа.';
function chainPreset(c) {
  const ov = app.settings.chains[c.id] || {};
  if (ov.preset) return ov.preset;
  return ['stiffness', 'gravity', 'drag'].some(k => k in ov) ? 'custom' : '';
}
function groupPreset(cs) {
  const ps = new Set(cs.map(chainPreset));
  return ps.size === 1 ? [...ps][0] : 'mixed';
}
function presetOptions(kind, value) {
  const list = app.info.part_presets?.[kind] || [];
  const opts = { '': 'Авто (по стилю физики)' };
  for (const p of list) opts[p.id] = p.name;
  opts.auto = 'Подобрано автоподбором';
  opts.custom = 'Свои настройки';
  if (value === 'mixed') opts.mixed = 'Разные у цепочек группы';
  return opts;
}
function presetRow(kind, value, apply, getValue) {
  const list = app.info.part_presets?.[kind] || [];
  if (!list.length) return null;
  const sel = selectEl(presetOptions(kind, value), value, (id) => {
    const p = list.find(x => x.id === id);
    if (p) apply(p);
    else if (id === '') apply(null);
    else { sel.value = getValue(); }  // 'custom' / 'mixed' / 'auto' are states, not choices
  });
  sel.classList.add('preset-sel');
  sel._get = getValue; sel._kind = kind;
  return h('div', { class: 'row' }, h('label', {}, 'Заготовка'), sel);
}
function updatePresetSelects() {
  for (const sel of document.querySelectorAll('#panel select.preset-sel')) {
    const v = sel._get();
    if (![...sel.options].some(o => o.value === v)) {
      sel.innerHTML = '';
      for (const [k, t] of Object.entries(presetOptions(sel._kind, v))) sel.append(h('option', { value: k }, t));
    }
    sel.value = v;
  }
}
function applyPreset(cs, p) {
  for (const c of cs) {
    const ov = chainOv(c.id);
    if (p) { for (const k of ['stiffness', 'gravity', 'drag']) setChainParam(c, k, p[k], true, true); ov.preset = p.id; }
    else { for (const k of ['stiffness', 'gravity', 'drag', 'preset']) delete ov[k]; }
  }
  if (!p) { rebuild(0); return; }
  physEdited(); renderPanel(); toast('Заготовка «' + p.name + '»' + (cs.length > 1 ? ' — для ' + cs.length + ' цепочек' : ''));
}
function chainEditor(c) {
  const mid = mirrorId(c.id);
  return h('div', { class: 'edit' },
    mid && mirrorEdits ? h('div', { class: 'note' }, '⇆ Меняется вместе с «' + ru(mid) + '»') : null,
    c.branch_of ? h('div', { class: 'note warnline' }, '↳ Висит на «' + ru(c.branch_of) + '». ' + (app.settings.vrm_version === 1
      ? 'В VRM 1.0 качается со своими настройками.'
      : 'В VRM 0.x формат не позволяет отделить её от «' + ru(c.branch_of) + '»: в файле она качается вместе с ним (но сталкивается с телом) — и выключить её отдельно тоже нельзя. Свои настройки и выключение работают при сохранении в VRM 1.0.')) : null,
    h('div', { class: 'row' }, h('label', {}, 'Тип'),
      selectEl(app.info.kinds, c.kind, (v) => { chainOv(c.id).kind = v; for (const k of ['stiffness', 'gravity', 'drag', 'preset']) delete chainOv(c.id)[k]; rebuild(); })),
    presetRow(c.kind, chainPreset(c), (p) => {
      const mid = mirrorEdits && mirrorId(c.id); const mc = mid && app.info.chains.find(x => x.id === mid);
      applyPreset(mc ? [c, mc] : [c], p);
    }, () => chainPreset(c)),
    ...paramSliders((k) => c[k], (k, v) => setChainParam(c, k, v), c.kind),
    c.length > 5 ? h('div', { class: 'note' }, `Длинная цепочка (${c.length} костей): программа сама добавила жёсткости и затухания к концу, чтобы кончик не болтался как пружина.`) : null,
    h('div', { class: 'row' }, h('label', {}, ''), check(`столкновения (${c.n_colliders})`, c.colliders, (v) => { chainOv(c.id).colliders = v; rebuild(); })),
    h('div', { class: 'btns' }, h('button', { onclick: () => { delete app.settings.chains[c.id]; rebuild(); } }, 'Сбросить к авто')));
}
function groupEditor(kind, cs, title) {
  const avg = (k) => cs.reduce((a, c) => a + c[k], 0) / cs.length;
  return h('div', { class: 'edit' },
    h('div', { class: 'title' }, title || 'Все цепочки: ' + app.info.kinds[kind]),
    h('div', { class: 'note' }, 'Изменения применяются сразу ко всем цепочкам этой группы.'),
    presetRow(kind, groupPreset(cs), (p) => applyPreset(cs, p), () => groupPreset(cs)),
    ...paramSliders(avg, (k, v) => { for (const c of cs) setChainParam(c, k, v, true); }, kind),
    h('div', { class: 'note' }, PARAM_HINT),
    h('div', { class: 'btns' },
      h('button', { onclick: () => { for (const c of cs) chainOv(c.id).enabled = true; rebuild(); } }, 'Включить все'),
      h('button', { onclick: () => { for (const c of cs) chainOv(c.id).enabled = false; rebuild(); } }, 'Выключить все'),
      h('button', { onclick: () => { for (const c of cs) delete app.settings.chains[c.id]; rebuild(); } }, 'Сбросить')));
}

const HB_RU = { hips: 'Таз', spine: 'Позвоночник', chest: 'Грудь', upperChest: 'Верх груди', neck: 'Шея', head: 'Голова',
  leftEye: 'Глаз Л', rightEye: 'Глаз П', jaw: 'Челюсть' };
function hbLabel(b) {
  if (HB_RU[b]) return HB_RU[b];
  const side = b.startsWith('left') ? ' Л' : ' П'; const r = b.replace(/^(left|right)/, '');
  const m = { Shoulder: 'Плечо', UpperArm: 'Плечо (рука)', LowerArm: 'Предплечье', Hand: 'Кисть', UpperLeg: 'Бедро',
    LowerLeg: 'Голень', Foot: 'Стопа', Toes: 'Пальцы ног' };
  if (m[r]) return m[r] + side;
  const f = { Thumb: 'Большой', Index: 'Указат.', Middle: 'Средний', Ring: 'Безым.', Little: 'Мизинец' };
  const p = { Proximal: '1', Intermediate: '2', Distal: '3' };
  const mm = r.match(/^(Thumb|Index|Middle|Ring|Little)(Proximal|Intermediate|Distal)$/);
  return mm ? `${f[mm[1]]} ${p[mm[2]]}${side}` : b;
}
function bonesPanel() {
  const I = app.info, out = [];
  const repairBtn = h('button', { class: 'accent', onclick: () => { app.settings.repair = !app.settings.repair; rebuild(0); } },
    app.settings.repair ? 'Отменить восстановление скелета' : 'Восстановить скелет');
  if (I.missing?.length) out.push(h('div', { class: 'edit' }, h('div', { class: 'bad' },
    'Не найдены обязательные кости: ' + I.missing.map(hbLabel).join(', ')),
    h('div', { class: 'note' }, 'Нажмите «Восстановить скелет» — программа сама поставит недостающие локти, запястья, колени, лодыжки, шею и перераспределит веса. Или укажите кости вручную ниже.'),
    h('div', { class: 'btns' }, repairBtn)));
  else out.push(h('div', { class: 'btns' }, repairBtn,
    h('span', { class: 'note' }, 'Если руки/ноги гнутся одним куском — нажмите, программа добавит недостающие суставы.')));
  if (I.repaired?.length) out.push(h('div', { class: 'ok' }, 'Восстановлены: ' + I.repaired.map(hbLabel).join(', ')));
  else out.push(h('div', { class: 'note' }, 'Соответствие костей модели гуманоидному скелету VRM. * — обязательные. Начните вводить имя кости — появится подсказка. Клик по строке показывает кость на модели.'));
  const card = boneCard();
  if (card) out.push(card);
  else out.push(h('div', { class: 'note' }, '🖱 Кликните по кости (точке скелета) или по телу на модели — кость выберется и найдётся в списке.'));
  const nHum = Object.values(I.humanoid || {}).filter(Boolean).length;
  if (!I.missing?.length) out.push(h('div', { class: 'okline' }, `✓ Скелет распознан: ${nHum} костей гуманоида`));
  out.push(boneLangControl());
  out.push(h('div', { class: 'note' }, 'Перевод применяется сразу — так кости будут называться и в сохранённом VRM.'));
  const dl = h('datalist', { id: 'boneList' }, ...I.bones.map(n => h('option', { value: n, label: ru(n) })));
  out.push(dl);
  const grid = h('div', { class: 'hb' });
  const groups = [['Тело', I.all_humanoid.filter(b => !/(Thumb|Index|Middle|Ring|Little)/.test(b))],
                  ['Пальцы', I.all_humanoid.filter(b => /(Thumb|Index|Middle|Ring|Little)/.test(b))]];
  for (const [title, bones] of groups) {
    grid.append(h('h3', { style: 'grid-column:1/4' }, title));
    for (const b of bones) {
      const req = I.required.includes(b), miss = I.missing?.includes(b);
      const changed = b in app.settings.humanoid;
      grid.append(h('span', { class: 'lbl' + (req ? ' req' : '') + (miss ? ' missing' : '') + (app.sel?.type === 'bone' && app.sel.id === b ? ' sel' : ''), title: b,
        style: 'cursor:pointer', onclick: () => select({ type: 'bone', id: b }) }, hbLabel(b)),
        h('input', { list: 'boneList', value: I.humanoid[b] || '', class: changed ? 'changed' : '', placeholder: '—',
          onfocus: () => select({ type: 'bone', id: b }),
          onchange: (e) => { const v = e.target.value.trim(); if (v && !I.bones.includes(v)) { toast('Нет такой кости: ' + v, true); return; }
            app.settings.humanoid[b] = v || null; rebuild(0); } }),
        h('span', { class: 'sub ru' }, h('button', { class: 'pick' + (app.pickFor === b ? ' on' : ''), title: 'Указать кость кликом по модели',
          onclick: () => { app.pickFor = app.pickFor === b ? null : b; renderPanel(); if (app.pickFor) toast('Кликните по модели рядом с костью: ' + hbLabel(b)); } }, '🎯'),
          ' ', I.humanoid[b] ? ru(I.humanoid[b]) : ''));
    }
  }
  const manual = h('details', { class: 'sec', open: !!I.missing?.length || app.openHb }, h('summary', { onclick: () => { app.openHb = !app.openHb; } }, 'Ручное сопоставление костей'), grid,
    h('div', { class: 'btns' }, h('button', { onclick: () => { if (confirm('Вернуть автоматическое сопоставление всех костей?')) { app.settings.humanoid = {}; rebuild(0); } } }, 'Сбросить сопоставление к авто')));
  out.push(manual);
  out.push(h('details', { class: 'sec', open: app.openNames }, h('summary', { onclick: () => { app.openNames = !app.openNames; } }, `Все кости модели (${I.bones.length}): имена и переименование`), ...allBonesSection()));
  return out;
}
let boneFilter = '';
// the bone clicked on the model: what it is and where it is used
function boneCard() {
  const I = app.info, sel = app.sel;
  if (!sel || (sel.type !== 'node' && !(sel.type === 'bone' && sel.node !== undefined))) return null;
  const i = sel.node, name = I.bones[i];
  if (name === undefined) return null;
  const hb = Object.entries(I.humanoid_nodes || {}).find(([, n]) => n === i)?.[0];
  const ch = I.chains.find(c => c.nodes.includes(i));
  const role = hb ? '🦴 кость гуманоида: ' + hbLabel(hb) : ch ? '🌀 физика: ' + ru(ch.id) + ' (' + I.kinds[ch.kind] + ')' : '· вспомогательная кость тела';
  return h('div', { class: 'edit bonecard' },
    h('div', { class: 'title' }, finalName(name), finalName(name) !== name ? h('span', { class: 'sub' }, '  ' + name) : null),
    h('div', { class: 'note' }, role),
    h('div', { class: 'row' }, h('label', {}, 'Своё имя'),
      h('input', { value: app.settings.renames[name] || '', placeholder: ru(name), style: 'flex:1',
        onchange: (e) => { const v = e.target.value.trim(); if (v) app.settings.renames[name] = v; else delete app.settings.renames[name]; rebuild(); } })),
    h('div', { class: 'btns' },
      ch ? h('button', { onclick: () => { setTab('physics'); app.physOpen ||= {}; app.physOpen[ch.kind] = true; app.flashT = clock.elapsedTime; select({ type: 'chain', id: ch.id }); } }, 'Настроить физику этой части') : null,
      h('button', { onclick: () => select(null) }, 'Снять выбор')));
}
function allBonesSection() {
  const I = app.info, out = [];
  out.push(h('div', { class: 'note' }, 'Своё имя для отдельной кости задаётся в поле справа — оно важнее перевода. Кириллицу понимают не все программы для аватаров; английский безопаснее.'));
  const listEl = h('div', { class: 'list' });
  const draw = () => {
    listEl.innerHTML = '';
    const q = boneFilter.toLowerCase();
    const rows = I.bones.filter(n => !q || n.toLowerCase().includes(q) || ru(n).toLowerCase().includes(q));
    for (const n of rows.slice(0, 150)) {
      listEl.append(h('div', { class: 'item brow' + (app.sel?.type === 'node' && app.sel.id === n ? ' sel' : ''),
        onclick: (e) => { if (e.target.tagName !== 'INPUT') { app.flashT = clock.elapsedTime; app.sel = { type: 'node', id: n, node: I.bones.indexOf(n) }; draw(); } } },
        h('span', { class: 'name', title: n }, finalName(n), h('span', { class: 'sub' }, finalName(n) !== n ? '  ' + n : '')),
        h('input', { value: app.settings.renames[n] || '', placeholder: 'своё имя', style: 'width:120px',
          onchange: (e) => { const v = e.target.value.trim(); if (v) app.settings.renames[n] = v; else delete app.settings.renames[n]; rebuild(); } })));
    }
    if (rows.length > 150) listEl.append(h('div', { class: 'note' }, `…и ещё ${rows.length - 150}, уточните поиск`));
  };
  out.push(h('div', { class: 'row' }, h('input', { placeholder: 'поиск: волосы, юбка, hair, 髪…', value: boneFilter, style: 'flex:1',
    oninput: (e) => { boneFilter = e.target.value; draw(); } })));
  draw(); out.push(listEl);
  return out;
}

const PRESET_RU = { a: 'А', i: 'И', u: 'У', e: 'Э', o: 'О', blink: 'Моргание', blink_l: 'Моргание Л', blink_r: 'Моргание П',
  joy: 'Радость', angry: 'Злость', sorrow: 'Грусть', fun: 'Веселье' };
function exprPanel() {
  const I = app.info, out = [];
  const pad = h('div', { class: 'pad' });
  const play = (name) => { if (!em) return; let t0 = performance.now();
    const step = () => { const k = (performance.now() - t0) / 900; const v = k < 0.5 ? k * 2 : Math.max(0, 2 - k * 2);
      em.setValue(name, v); if (k < 1) requestAnimationFrame(step); else em.setValue(name, 0); }; step(); };
  for (const [lbl, nm] of [['😊', 'happy'], ['😠', 'angry'], ['😢', 'sad'], ['😆', 'relaxed'], ['😉 моргнуть', 'blink'], ['А', 'aa'], ['И', 'ih'], ['У', 'ou'], ['Э', 'ee'], ['О', 'oh']])
    pad.append(h('button', { title: 'Показать на модели', onclick: () => play(nm) }, lbl));
  out.push(h('h3', {}, 'Проверить'), pad);
  out.push(h('div', { class: 'note' }, 'Ниже — из каких морфов модели собрано каждое стандартное выражение VRM. Ползунок справа держит выражение на модели.'));
  out.push(h('datalist', { id: 'morphList' }, ...I.morphs.map(n => h('option', { value: n }))));
  const em = app.vrm?.expressionManager;
  const map = { a: 'aa', i: 'ih', u: 'ou', e: 'ee', o: 'oh', blink: 'blink', blink_l: 'blinkLeft', blink_r: 'blinkRight',
    joy: 'happy', angry: 'angry', sorrow: 'sad', fun: 'relaxed' };
  out.push(h('h3', {}, 'Стандартные'));
  for (const p of I.preset_names) {
    const cur = app.settings.presets[p] ?? I.presets[p] ?? [];
    const name = map[p];
    out.push(h('div', { class: 'row xrow' }, h('label', {}, PRESET_RU[p] || p),
      h('input', { list: 'morphList', value: cur.join(', '), placeholder: 'нет', style: 'flex:1; min-width:0',
        onchange: (e) => { app.settings.presets[p] = e.target.value.split(',').map(s => s.trim()).filter(Boolean); rebuild(0); } }),
      h('input', { type: 'range', min: 0, max: 1, step: 0.01, value: 0, style: 'width:56px; flex:none',
        oninput: (e) => em?.setValue(name, +e.target.value) })));
  }
  if (em) {
    const custom = em.expressions.filter(x => !Object.values(map).includes(x.expressionName) && x.expressionName !== 'neutral'
      && !['lookUp', 'lookDown', 'lookLeft', 'lookRight'].includes(x.expressionName));
    if (custom.length) out.push(h('details', { class: 'sec' }, h('summary', {}, `Остальные морфы модели (${custom.length})`),
      ...custom.map(x => slider(x.expressionName, 0, 0, 1, 0.01, (v) => em.setValue(x.expressionName, v)))));
  }
  out.push(h('div', { class: 'btns' },
    h('button', { onclick: () => { em?.expressions.forEach(x => em.setValue(x.expressionName, 0)); renderPanel(); } }, 'Сбросить ползунки'),
    h('button', { onclick: () => { app.settings.presets = {}; rebuild(0); } }, 'Сопоставление — к авто')));
  const s = app.settings;
  out.push(h('h3', {}, 'Настройки лица'));
  out.push(h('div', { class: 'row' }, check('Мимика из костей век и челюсти (для моделей без выражений)', s.face_from_bones !== false, (v) => { s.face_from_bones = v; rebuild(0); })));
  out.push(h('div', { class: 'row' }, check('Все морфы модели — отдельными выражениями', s.custom_blendshapes, (v) => { s.custom_blendshapes = v; rebuild(0); })));
  out.push(h('div', { class: 'row' }, check('Плоские аниме-тени на лице', s.anime_face !== false, (v) => { s.anime_face = v; rebuild(0); })));
  return out;
}

function settingsPanel() {
  const s = app.settings, out = [];
  const num = (label, key, step, onCh) => h('div', { class: 'row' }, h('label', {}, label),
    h('input', { type: 'number', value: s[key], step, min: 0, style: 'width:90px', onchange: (e) => {
      const v = +e.target.value;
      if (!(v > 0) && key === 'scale') { e.target.value = s[key]; toast('Масштаб должен быть больше нуля', true); return; }
      if (!(v >= 0)) { e.target.value = s[key]; return; }
      s[key] = v; onCh(); } }));
  out.push(h('h3', {}, 'Конвертация'));
  out.push(h('div', { class: 'row' }, h('label', {}, 'Формат'),
    selectEl({ 0: 'VRM 0.x (совместимость)', 1: 'VRM 1.0 (новые программы)' }, String(s.vrm_version || 0), (v) => { s.vrm_version = +v; rebuild(0); })));
  out.push(num('Масштаб (м/ед.)', 'scale', 0.005, () => rebuild(0)));
  if (app.vrm) { const bx = new THREE.Box3().setFromObject(app.vrm.scene); out.push(h('div', { class: 'note' }, `Рост модели сейчас ≈ ${(bx.max.y - bx.min.y).toFixed(2)} м. Обычно 1,4–1,8 м.`)); }
  out.push(h('div', { class: 'row' }, h('label', {}, 'Макс. текстура'),
    selectEl({ 0: 'оригинал', 1024: '1024', 2048: '2048', 4096: '4096' }, String(s.max_texture), (v) => { s.max_texture = +v; rebuild(0); })));
  out.push(h('div', { class: 'row' }, h('label', {}, ''), check('T-поза (нужно для VRM)', s.tpose, (v) => { s.tpose = v; rebuild(0); })));
  out.push(h('div', { class: 'row' }, h('label', {}, ''), check('Физика (spring bones)', s.spring, (v) => { s.spring = v; rebuild(0); })));
  out.push(boneLangControl());
  out.push(h('h3', {}, 'Информация о модели'));
  out.push(h('div', { class: 'row' }, h('label', {}, 'Название'),
    h('input', { value: s.meta.title || app.info?.model?.name || '', style: 'flex:1', onchange: (e) => s.meta.title = e.target.value })));
  out.push(h('div', { class: 'row' }, h('label', {}, 'Автор'),
    h('input', { value: s.meta.author || '', style: 'flex:1', onchange: (e) => s.meta.author = e.target.value })));
  out.push(h('div', { class: 'note' }, 'Проверьте условия использования исходной модели (readme / 利用規約): многие авторы запрещают конвертацию и распространение.'));
  if (app.info?.model?.comment) out.push(h('details', { class: 'sec' }, h('summary', {}, 'Условия автора модели (readme / 利用規約)'), h('div', { class: 'log' }, app.info.model.comment)));
  out.push(h('h3', {}, 'Размер файла'));
  out.push(h('div', { class: 'row' }, h('label', {}, 'Не больше'),
    selectEl({ 0: 'без ограничения', 25: '25 МБ (VRChat Quest / cluster)', 50: '50 МБ', 100: '100 МБ' }, String(s.target_mb || 0), (v) => { s.target_mb = +v; rebuild(0); })));
  out.push(h('div', { class: 'row' }, h('label', {}, ''), check('Убирать физику скрытых частей', !!s.strip_hidden, (v) => { s.strip_hidden = v; rebuild(0); })));
  out.push(h('div', { class: 'note' }, 'Скрытые части не попадают в файл. С галочкой у них ещё и отключается физика (пряди и ленты, на которых ничего не видно), модель работает легче.'));
  out.push(h('h3', {}, 'Шаблоны'));
  out.push(h('div', { class: 'note' }, 'Шаблон сохраняет физику, категории частей, выражения и настройки. Его можно применить к другой модели — например, ко всем персонажам из одной игры: правки переносятся по совпадающим именам материалов и костей.'));
  out.push(h('div', { class: 'btns' },
    h('button', { onclick: async () => { try { const r = await api('/api/template_save', { settings: app.settings }); if (r.saved) toast('Шаблон сохранён: ' + r.saved); } catch (e) { toast(e.message, true); } } }, 'Сохранить как шаблон…'),
    h('button', { onclick: async () => {
      try {
        const r = await api('/api/template_load', {});
        if (r.settings) { app.settings = { ...app.settings, ...r.settings }; rebuild(0); toast('Шаблон применён: ' + r.path.split(/[\\/]/).pop()); }
      } catch (e) { toast(e.message, true); } } }, 'Применить шаблон…')));
  out.push(h('h3', { class: 'danger' }, 'Опасная зона'));
  out.push(h('div', { class: 'btns' }, h('button', { onclick: () => {
    if (!confirm('Сбросить все ручные правки этой модели (части, физику, кости, выражения)? Отменить можно Ctrl+Z.')) return;
    const keep = { scale: s.scale, max_texture: s.max_texture, physics: s.physics, vrm_version: s.vrm_version,
      target_mb: s.target_mb, tex_quality: s.tex_quality, tex_sharpen: s.tex_sharpen, upscale: s.upscale, bone_lang: s.bone_lang, meta: s.meta };
    app.settings = { ...DEFAULTS(), ...keep }; rebuild(0); } }, 'Сбросить все ручные правки')));
  return out;
}
function morePanel() {
  const out = settingsPanel();
  out.push(h('details', { class: 'sec', open: app.openLog }, h('summary', { onclick: () => { app.openLog = !app.openLog; } }, 'Журнал конвертации'), ...logPanel()));
  return out;
}
function logPanel() {
  const I = app.info, out = [];
  if (I.warnings?.length) { out.push(h('h3', {}, 'Предупреждения')); for (const w of I.warnings) out.push(h('div', { class: 'warn' }, '• ' + w)); }
  out.push(h('h3', {}, 'Отчёт конвертации'));
  out.push(h('div', { class: 'log' }, (I.log || []).filter(l => !/Готово: None/.test(l)).join('\n')));
  return out;
}

// ------------------------------------------------------------------ startup
renderPanel();
(async () => {
  try {
    const st = await api('/api/state');
    if (st.startup?.files?.length) openPath(st.startup.files[0]);
    else if (!st.empty) {
      if (st.settings) app.settings = { ...DEFAULTS(), ...st.settings };  // page reload: keep the edits
      app.committed = JSON.stringify(app.settings);
      await applyInfo(st, true);
    }
  } catch (e) { console.error(e); }
})();
window.__rebuild = () => rebuild(0);
