"""Read an existing VRM (0.x or 1.0) back into the internal MMD-like model, so it can be edited
(physics, expressions, materials, T-pose fixes) and saved again as VRM 0.x or 1.0.

Geometry is returned in MMD axes and units (1 unit = 0.08 m, character facing -Z, left at +X),
the same as pmx.py / fbx.py produce, so the converter treats it like any other source."""
import json
import struct

import numpy as np

import pmx

SCALE = 0.08

PRESET_TO_MMD = {
    # VRM 0.x
    "a": "あ", "i": "い", "u": "う", "e": "え", "o": "お", "blink": "まばたき",
    "blink_l": "ウィンク", "blink_r": "ウィンク右", "joy": "笑い", "angry": "怒り",
    "sorrow": "困る", "fun": "にこり", "neutral": "",
    # VRM 1.0
    "aa": "あ", "ih": "い", "ou": "う", "ee": "え", "oh": "お", "blinkLeft": "ウィンク",
    "blinkRight": "ウィンク右", "happy": "笑い", "sad": "困る", "relaxed": "にこり", "surprised": "驚き",
}

_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class VRMError(ValueError):
    pass


def _glb(data):
    if len(data) < 20 or data[:4] != b"glTF":
        raise VRMError("Это не файл VRM/GLB")
    total = struct.unpack_from("<I", data, 8)[0]
    p = 12
    js, bin_ = None, b""
    while p + 8 <= min(total, len(data)):
        ln, typ = struct.unpack_from("<II", data, p)
        chunk = data[p + 8:p + 8 + ln]
        if typ == 0x4E4F534A:
            js = json.loads(chunk.decode("utf-8"))
        elif typ == 0x004E4942:
            bin_ = bytes(chunk)
        p += 8 + ln
    if js is None:
        raise VRMError("В файле нет описания glTF")
    return js, bin_


class _Acc:
    def __init__(self, js, bin_):
        self.js, self.bin = js, bin_

    def view(self, vi):
        bv = self.js["bufferViews"][vi]
        if bv.get("buffer", 0) != 0:
            raise VRMError("Внешние буферы не поддерживаются")
        o = bv.get("byteOffset", 0)
        return self.bin[o:o + bv["byteLength"]], bv.get("byteStride")

    def read(self, ai):
        a = self.js["accessors"][ai]
        dt = np.dtype(_CT[a["componentType"]])
        nc = _NC[a["type"]]
        cnt = a["count"]
        if "bufferView" in a:
            raw, stride = self.view(a["bufferView"])
            off = a.get("byteOffset", 0)
            item = dt.itemsize * nc
            if stride and stride != item:
                buf = np.frombuffer(raw, np.uint8, count=stride * (cnt - 1) + item, offset=off)
                rows = np.lib.stride_tricks.as_strided(buf, (cnt, item), (stride, 1))
                arr = np.ascontiguousarray(rows).view(dt).reshape(cnt, nc)
            else:
                arr = np.frombuffer(raw, dt, count=cnt * nc, offset=off).reshape(cnt, nc).copy()
        else:
            arr = np.zeros((cnt, nc), dt)
        sp = a.get("sparse")
        if sp:
            iv, _ = self.view(sp["indices"]["bufferView"])
            idx = np.frombuffer(iv, _CT[sp["indices"]["componentType"]], count=sp["count"],
                                offset=sp["indices"].get("byteOffset", 0)).astype(np.int64)
            vv, _ = self.view(sp["values"]["bufferView"])
            vals = np.frombuffer(vv, dt, count=sp["count"] * nc, offset=sp["values"].get("byteOffset", 0))
            arr = arr.astype(dt, copy=True)
            arr[idx] = vals.reshape(-1, nc)
        if a.get("normalized") and dt.kind in "iu":
            arr = arr.astype(np.float32) / float(np.iinfo(dt).max)
        return arr


def _node_local(n):
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T
    t = np.array(n.get("translation", [0, 0, 0]), float)
    x, y, z, w = n.get("rotation", [0, 0, 0, 1])
    s = np.array(n.get("scale", [1, 1, 1]), float)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3] = R * s
    M[:3, 3] = t
    return M


def load(path, log=print):
    with open(path, "rb") as f:
        data = f.read()
    js, bin_ = _glb(data)
    ext = js.get("extensions", {})
    v1 = "VRMC_vrm" in ext
    if not v1 and "VRM" not in ext:
        raise VRMError("Это glTF без расширения VRM: нет описания гуманоида")
    acc = _Acc(js, bin_)
    nodes = js.get("nodes", [])
    nn = len(nodes)
    parent = [-1] * nn
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            if 0 <= c < nn:
                parent[c] = i
    world = [None] * nn

    def W(i, depth=0):
        if world[i] is None:
            L = _node_local(nodes[i])
            world[i] = L if parent[i] < 0 or depth > 512 else W(parent[i], depth + 1) @ L
        return world[i]

    for i in range(nn):
        W(i)
    # glTF (metres) -> MMD axes and units
    flip = np.array([1.0, 1.0, -1.0]) if v1 else np.array([-1.0, 1.0, 1.0])

    def to_mmd(p):
        return p * flip / SCALE

    skins = js.get("skins", [])
    bone_nodes = set()
    for sk in skins:
        bone_nodes.update(sk.get("joints", []))
    hb = (ext["VRMC_vrm"]["humanoid"]["humanBones"] if v1 else
          {b["bone"]: {"node": b["node"]} for b in ext["VRM"].get("humanoid", {}).get("humanBones", [])})
    for k, v in hb.items():
        if isinstance(v, dict) and isinstance(v.get("node"), int):
            bone_nodes.add(v["node"])
    # spring joints and collider nodes are bones too
    if v1:
        sb = ext.get("VRMC_springBone", {})
        for s in sb.get("springs", []):
            for j in s.get("joints", []):
                bone_nodes.add(j["node"])
        for c in sb.get("colliders", []):
            bone_nodes.add(c["node"])
    else:
        sa = ext["VRM"].get("secondaryAnimation", {})
        for g in sa.get("boneGroups", []):
            bone_nodes.update(g.get("bones", []))
        for cg in sa.get("colliderGroups", []):
            bone_nodes.add(cg["node"])
    bone_nodes = {b for b in bone_nodes if 0 <= b < nn}
    for b in list(bone_nodes):  # with all ancestors (keeps one connected tree)
        p = parent[b]
        guard = 0
        while p >= 0 and p not in bone_nodes and guard <= nn:
            guard += 1
            bone_nodes.add(p)
            p = parent[p]
    # spring bone subtrees (VRM0 roots swing whole subtrees)
    if not v1:
        for g in ext["VRM"].get("secondaryAnimation", {}).get("boneGroups", []):
            st = list(g.get("bones", []))
            while st:
                x = st.pop()
                if 0 <= x < nn:
                    bone_nodes.add(x)
                    st.extend(nodes[x].get("children", []))
    # parents before children
    depth = {}

    def dep(i):
        chain = []
        x = i
        while x not in depth:
            if x in chain or len(chain) > nn:
                raise VRMError("Иерархия узлов в VRM зациклена")
            chain.append(x)
            if parent[x] < 0 or parent[x] not in bone_nodes:
                depth[x] = 0
                chain.pop()
                break
            x = parent[x]
        for y in reversed(chain):
            depth[y] = depth[parent[y]] + 1
        return depth[i]

    order = sorted(bone_nodes, key=lambda i: (dep(i), i))
    bidx = {n: k for k, n in enumerate(order)}
    m = pmx.Model()
    meta = ext["VRMC_vrm"].get("meta", {}) if v1 else ext["VRM"].get("meta", {})
    m.name = meta.get("name") or meta.get("title") or "VRM"
    m.comment = "Imported from VRM %s" % ("1.0" if v1 else "0.x")
    used_names = set()
    for n_ in order:
        b = pmx.Bone()
        nm = nodes[n_].get("name") or "node_%d" % n_
        base, k = nm, 1
        while nm in used_names:
            k += 1
            nm = "%s_%d" % (base, k)
        used_names.add(nm)
        b.name, b.name_en = nm, ""
        b.pos = to_mmd(W(n_)[:3, 3]).astype(float)
        p = parent[n_]
        b.parent = bidx.get(p, -1)
        b.layer = 0
        b.flags = 0x0002 | 0x0008 | 0x0010
        m.bones.append(b)
    m.humanoid_hint = {}
    for k, v in hb.items():
        if not (isinstance(v, dict) and v.get("node") in bidx):
            continue
        if v1:  # VRM 1.0 thumb names -> the 0.x names the converter works with
            if k.endswith("ThumbMetacarpal"):
                k = k[:-len("ThumbMetacarpal")] + "ThumbProximal"
            elif k.endswith("ThumbProximal"):
                k = k[:-len("ThumbProximal")] + "ThumbIntermediate"
        m.humanoid_hint[k] = bidx[v["node"]]

    def nearest_bone(n_):
        guard = 0
        while n_ >= 0 and n_ not in bidx and guard <= nn:
            n_ = parent[n_]
            guard += 1
        return bidx.get(n_, 0)

    # ---- textures
    images = js.get("images", [])
    tex_of_image = {}
    m.texture_blobs = {}

    def texture_index(tinfo):
        if not tinfo:
            return -1
        t = js.get("textures", [])[tinfo.get("index", -1)] if 0 <= tinfo.get("index", -1) < len(js.get("textures", [])) else None
        if not t or "source" not in t:
            return -1
        src = t["source"]
        if src in tex_of_image:
            return tex_of_image[src]
        im = images[src] if 0 <= src < len(images) else None
        if not im or "bufferView" not in im:
            return -1
        raw, _ = acc.view(im["bufferView"])
        ext_ = ".jpg" if "jpeg" in im.get("mimeType", "") else ".png"
        ti = len(m.textures)
        m.textures.append((im.get("name") or "image_%d" % src) + ext_)
        m.texture_blobs[ti] = bytes(raw)
        tex_of_image[src] = ti
        return ti

    gl_mats = js.get("materials", [])
    mtoon0 = {mp.get("name"): mp for mp in ext.get("VRM", {}).get("materialProperties", [])} if not v1 else {}

    def lin2srgb(c):
        c = max(0.0, min(1.0, float(c)))
        return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055

    # ---- meshes
    P, N, UV, J, WT, IDX = [], [], [], [], [], []
    mats = []
    vbase = 0
    target_map = {}  # (mesh index, target index) -> list of (global vertex ids, deltas)
    target_names = {}
    for ni, node in enumerate(nodes):
        if "mesh" not in node:
            continue
        mi = node["mesh"]
        mesh = js["meshes"][mi]
        names_t = (mesh.get("extras") or {}).get("targetNames") or []
        sk = skins[node["skin"]] if "skin" in node and node["skin"] < len(skins) else None
        if sk:
            ibm = acc.read(sk["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1) \
                if "inverseBindMatrices" in sk else np.tile(np.eye(4), (len(sk["joints"]), 1, 1))
            jm = np.array([W(j) for j in sk["joints"]]) @ ibm  # joint matrices at rest
        for pi, prim in enumerate(mesh.get("primitives", [])):
            if prim.get("mode", 4) != 4:
                continue
            at = prim["attributes"]
            pos = acc.read(at["POSITION"]).astype(np.float64)
            nv = len(pos)
            nor = acc.read(at["NORMAL"]).astype(np.float64) if "NORMAL" in at else np.tile([0, 1, 0], (nv, 1)).astype(float)
            uv = acc.read(at["TEXCOORD_0"]).astype(np.float32) if "TEXCOORD_0" in at else np.zeros((nv, 2), np.float32)
            if sk and "JOINTS_0" in at and "WEIGHTS_0" in at:
                jj = acc.read(at["JOINTS_0"]).astype(np.int64)
                ww = acc.read(at["WEIGHTS_0"]).astype(np.float64)
                jj = np.clip(jj, 0, len(sk["joints"]) - 1)
                s = ww.sum(axis=1, keepdims=True)
                ww = np.where(s > 1e-8, ww / np.maximum(s, 1e-8), np.array([[1.0, 0, 0, 0]]))
                Mv = np.einsum("nk,nkab->nab", ww, jm[jj])
                bj = np.array([nearest_bone(sk["joints"][x]) for x in range(len(sk["joints"]))])[jj]
            else:
                Mv = np.tile(W(ni), (nv, 1, 1))
                bj = np.zeros((nv, 4), np.int64)
                bj[:, 0] = nearest_bone(ni)
                ww = np.zeros((nv, 4))
                ww[:, 0] = 1.0
            wp = np.einsum("nab,nb->na", Mv[:, :3, :3], pos) + Mv[:, :3, 3]
            wn = np.einsum("nab,nb->na", Mv[:, :3, :3], nor)
            P.append(to_mmd(wp))
            N.append(wn * flip)
            UV.append(uv)
            J.append(bj.astype(np.int32))
            WT.append(ww.astype(np.float32))
            if "indices" in prim:
                ind = acc.read(prim["indices"]).reshape(-1).astype(np.int64)
            else:
                ind = np.arange(nv, dtype=np.int64)
            ind = ind[:len(ind) // 3 * 3].reshape(-1, 3)[:, ::-1].reshape(-1)  # handedness flip
            IDX.append(ind + vbase)
            # material
            gi = prim.get("material")
            gm = gl_mats[gi] if gi is not None and 0 <= gi < len(gl_mats) else {}
            mt = pmx.Material()
            mt.name = gm.get("name") or "material_%d" % len(mats)
            mt.name_en = ""
            pbr = gm.get("pbrMetallicRoughness", {})
            bc = pbr.get("baseColorFactor", [1, 1, 1, 1])
            mt.diffuse = (lin2srgb(bc[0]), lin2srgb(bc[1]), lin2srgb(bc[2]), float(bc[3]))
            mp = mtoon0.get(gm.get("name"))
            if mp and "_Color" in mp.get("vectorProperties", {}):
                c = mp["vectorProperties"]["_Color"]
                mt.diffuse = (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
            mt.specular, mt.spec_power = (0.0, 0.0, 0.0), 1.0
            mt.ambient = tuple(float(x) * 0.4 for x in mt.diffuse[:3])
            mt.flags = 0x01 if gm.get("doubleSided") else 0
            mt.edge_color, mt.edge_size = (0, 0, 0, 1), 0.0
            mt.texture = texture_index(pbr.get("baseColorTexture"))
            mt.sphere, mt.sphere_mode, mt.toon_shared, mt.toon, mt.memo = -1, 0, 1, 0, ""
            mt.index_count = int(len(ind))
            mats.append(mt)
            # morph targets (deltas through the same skinning matrix)
            for ti_, tgt in enumerate(prim.get("targets", [])):
                if "POSITION" not in tgt:
                    continue
                d = acc.read(tgt["POSITION"]).astype(np.float64)
                d = np.einsum("nab,nb->na", Mv[:, :3, :3], d) * flip / SCALE
                nz = np.nonzero(np.abs(d).max(axis=1) > 1e-7)[0]
                if len(nz):
                    target_map.setdefault((mi, ti_), []).append((nz + vbase, d[nz].astype(np.float32)))
                if ti_ < len(names_t):
                    target_names[(mi, ti_)] = names_t[ti_]
                else:
                    target_names.setdefault((mi, ti_), (prim.get("extras") or {}).get("targetNames", [None] * 999)[ti_]
                                            if ti_ < len((prim.get("extras") or {}).get("targetNames", [])) else "morph_%d_%d" % (mi, ti_))
            vbase += nv
    if not P:
        raise VRMError("В VRM нет сеток")
    m.positions = np.concatenate(P).astype(np.float32)
    m.normals = np.concatenate(N).astype(np.float32)
    m.uvs = np.concatenate(UV).astype(np.float32)
    m.joints = np.concatenate(J).astype(np.int32)
    m.weights = np.concatenate(WT).astype(np.float32)
    m.indices = np.concatenate(IDX).astype(np.int64)
    m.materials = mats

    # ---- expressions -> vertex morphs
    node_mesh = {i: n["mesh"] for i, n in enumerate(nodes) if "mesh" in n}

    def morph_from(binds):
        acc_ = {}
        for mesh_i, t_i, w in binds:
            for ix, d in target_map.get((mesh_i, t_i), []):
                for a, b in zip(ix.tolist(), d * w):
                    acc_[a] = acc_.get(a, 0) + b
        if not acc_:
            return None
        ks = np.array(sorted(acc_), np.int64)
        return ks, np.array([acc_[k] for k in ks.tolist()], np.float32)

    exprs = []
    if v1:
        ex = ext["VRMC_vrm"].get("expressions", {})
        for group, items in (("preset", ex.get("preset", {})), ("custom", ex.get("custom", {}))):
            for name, e in items.items():
                binds = [(node_mesh.get(b["node"]), b["index"], float(b.get("weight", 1.0)))
                         for b in e.get("morphTargetBinds", []) if b.get("node") in node_mesh]
                nm = PRESET_TO_MMD.get(name, name) if group == "preset" else name
                exprs.append((nm, binds))
    else:
        for g in ext["VRM"].get("blendShapeMaster", {}).get("blendShapeGroups", []):
            mesh_nodes = js.get("meshes", [])
            binds = [(b["mesh"], b["index"], float(b.get("weight", 100)) / 100.0) for b in g.get("binds", [])
                     if 0 <= b.get("mesh", -1) < len(mesh_nodes)]
            pn = g.get("presetName", "unknown")
            nm = PRESET_TO_MMD.get(pn) if pn not in ("unknown", "") else None
            exprs.append((nm or g.get("name") or "expression", binds))
    seen = set()
    for nm, binds in exprs:
        if not nm or nm in seen:
            continue
        r = morph_from(binds)
        if r is None:
            continue
        seen.add(nm)
        mo = pmx.Morph()
        mo.name, mo.name_en, mo.panel, mo.type, mo.data = nm, "", 4, 1, r
        m.morphs.append(mo)
    if not m.morphs:  # no expressions: raw blend shapes
        for key, parts in target_map.items():
            mo = pmx.Morph()
            mo.name, mo.name_en, mo.panel, mo.type = target_names.get(key) or "morph_%d_%d" % key, "", 4, 1
            mo.data = (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
            m.morphs.append(mo)

    # ---- spring bones -> MMD rigid bodies (dynamic joints, static colliders)
    def rb(bone, pos_mmd, r_m, mode, name):
        b = pmx.RigidBody()
        b.name, b.name_en, b.bone = name, "", bone
        b.group, b.mask, b.shape = (2 if mode else 1), 0, 0
        b.size = (max(r_m, 0.005) / SCALE, 0.0, 0.0)
        b.pos = tuple(float(x) for x in pos_mmd)
        b.rot = (0.0, 0.0, 0.0)
        b.mass, b.lin_damp, b.ang_damp = 1.0, 0.5, 0.5
        b.restitution, b.friction, b.mode = 0.0, 0.5, mode
        return b

    rbs = []
    if v1:
        sb = ext.get("VRMC_springBone", {})
        for s in sb.get("springs", []):
            for j in s.get("joints", [])[:-1] or s.get("joints", []):
                if j["node"] in bidx:
                    bi = bidx[j["node"]]
                    rbs.append(rb(bi, m.bones[bi].pos, float(j.get("hitRadius", 0.02)), 1, "spring_%d" % bi))
        for c in sb.get("colliders", []):
            if c["node"] not in bidx:
                continue
            bi = bidx[c["node"]]
            shp = c.get("shape", {})
            sph = shp.get("sphere") or shp.get("capsule") or {}
            pts = [sph.get("offset", [0, 0, 0])] + ([sph["tail"]] if "tail" in sph else [])
            for k, o in enumerate(pts):
                wp = W(c["node"]) @ np.array(list(o) + [1.0])
                rbs.append(rb(bi, to_mmd(wp[:3]), float(sph.get("radius", 0.05)), 0, "collider_%d_%d" % (bi, k)))
    else:
        sa = ext["VRM"].get("secondaryAnimation", {})
        for g in sa.get("boneGroups", []):
            st = list(g.get("bones", []))
            while st:
                x = st.pop()
                if x in bidx:
                    bi = bidx[x]
                    rbs.append(rb(bi, m.bones[bi].pos, float(g.get("hitRadius", 0.02)), 1, "spring_%d" % bi))
                    st.extend(nodes[x].get("children", []))
        for cg in sa.get("colliderGroups", []):
            if cg.get("node") not in bidx:
                continue
            bi = bidx[cg["node"]]
            for k, c in enumerate(cg.get("colliders", [])):
                o = c.get("offset", {})
                # VRM0 stores collider offsets in Unity axes (z flipped)
                wp = W(cg["node"]) @ np.array([o.get("x", 0), o.get("y", 0), -o.get("z", 0), 1.0])
                rbs.append(rb(bi, to_mmd(wp[:3]), float(c.get("radius", 0.05)), 0, "collider_%d_%d" % (bi, k)))
    m.rigid_bodies = rbs
    m.suggested_scale = SCALE
    m.source_format = "vrm"
    log("  VRM %s: вершин %d, костей %d, материалов %d, выражений %d, физика: %d тел"
        % ("1.0" if v1 else "0.x", len(m.positions), len(m.bones), len(m.materials), len(m.morphs), len(rbs)))
    return m
