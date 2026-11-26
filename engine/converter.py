"""MMD (PMX) -> VRM 0.x converter."""
import copy
import io
import json
import math
import re
import os
import struct
import unicodedata

import numpy as np
from PIL import Image

import pmx
import names_en
import names_ru
import vocab

VERSION = "2.3.0"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def nk(s):
    return unicodedata.normalize("NFKC", s or "").strip()


def unique_names(names, fallback):
    used = set()
    out = []
    for i, n in enumerate(names):
        n = (n or "").strip() or ("%s_%d" % (fallback, i))
        base, k = n, 1
        while n in used:
            k += 1
            n = "%s_%d" % (base, k)
        used.add(n)
        out.append(n)
    return out


def srgb_to_linear(c):
    c = max(0.0, min(1.0, float(c)))
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def rot_between(a, b):
    """Rotation matrix taking direction a onto direction b."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # 180 degrees: rotate around any perpendicular axis
        axis = np.cross(a, [0, 0, 1]) if abs(a[2]) < 0.9 else np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def euler_mmd(rx, ry, rz):
    """MMD rigid body rotation (YawPitchRoll => R = Ry*Rx*Rz)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def mat_quat(R):
    """rotation matrix -> quaternion (x, y, z, w)"""
    R = np.asarray(R, float)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        S = math.sqrt(t + 1.0) * 2
        q = ((R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S, 0.25 * S)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = (0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S, (R[2, 1] - R[1, 2]) / S)
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = ((R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S, (R[0, 2] - R[2, 0]) / S)
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = ((R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S, (R[1, 0] - R[0, 1]) / S)
    return [float(x) for x in q]


def quat_mat(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def quat_pow(q, t):
    """slerp(identity, q, t)"""
    x, y, z, w = q
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    w = max(-1.0, min(1.0, w))
    ang = math.acos(w)
    s = math.sin(ang)
    if s < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    na = ang * t
    k = math.sin(na) / s
    return (x * k, y * k, z * k, math.cos(na))


def bake_bone_morph(bones, parent, order, data, P, jo, we):
    """Pose the skeleton with a bone morph and return per-vertex offsets (N,3)."""
    nb = len(bones)
    lt = np.zeros((nb, 3))
    lq = [(0.0, 0.0, 0.0, 1.0)] * nb
    for b, t, q in data:
        if 0 <= b < nb:
            lt[b] += t
            lq[b] = quat_mul(q, lq[b])
    # 付与 (grant) rotation / translation
    for b in order:
        bn = bones[b]
        g = bn.grant_parent
        if 0 <= g < nb and bn.grant_weight != 0:
            if bn.flags & 0x0100:
                lq[b] = quat_mul(quat_pow(lq[g], bn.grant_weight), lq[b])
            if bn.flags & 0x0200:
                lt[b] = lt[b] + lt[g] * bn.grant_weight
    pos = np.array([bn.pos for bn in bones], float)
    Rw = np.zeros((nb, 3, 3))
    Pw = np.zeros((nb, 3))
    for b in order:
        R = quat_mat(lq[b])
        p = parent[b]
        if p < 0:
            Rw[b] = R
            Pw[b] = pos[b] + lt[b]
        else:
            Rw[b] = Rw[p] @ R
            Pw[b] = Pw[p] + Rw[p] @ (pos[b] - pos[p] + lt[b])
    moved = np.zeros(nb, bool)
    moved |= np.abs(Pw - pos).max(axis=1) > 1e-7
    moved |= np.abs(Rw - np.eye(3)).reshape(nb, -1).max(axis=1) > 1e-7
    vm = (moved[jo] & (we > 0)).any(axis=1)
    out = np.zeros_like(P)
    if not vm.any():
        return out
    v = P[vm]
    j = jo[vm]
    w = we[vm]
    loc = v[:, None, :] - pos[j]                       # (k,4,3)
    nv = np.einsum("nkab,nkb->nka", Rw[j], loc) + Pw[j]  # (k,4,3)
    out[vm] = np.einsum("nk,nka->na", w, nv) - v
    return out


# --------------------------------------------------------------------------
# facial expressions from bones (FBX / game rips without blendshapes)
# --------------------------------------------------------------------------
import re as _re

_RX_LID = _re.compile(r"(?i)eye_?lid|まぶた|瞼|眼睑|眼皮")
_RX_UP = _re.compile(r"(?i)(^|[^a-z])(u|up|upper|top|t)([^a-z]|$)|upper|上")
_RX_DN = _re.compile(r"(?i)(^|[^a-z])(d|dn|down|lower|bottom|b)([^a-z]|$)|lower|下")
# a ring of numbered bones around each eye (eyeLf01..08 / eye_L_01 ...): the real eyelid edge in many
# game rigs, while a bone called "eyeLid" there is only the skin fold above it
_RX_EYERING = _re.compile(r"(?i)^(?:[a-z]+_)?eye_?(lf|rt|l|r|left|right)_?(\d{1,2})_?(?:joint|jnt|bone)?$")
_RX_EYELASH = _re.compile(r"(?i)eye_?(lf|rt|l|r|left|right)?_?eyelash|eyelash")
_RX_LIDSKIP = _re.compile(r"(?i)furrow|crease|fold|_end$|_tip$|先")
_RX_JAW = _re.compile(r"(?i)(^|[^a-z])jaw([^a-z]|$)|^jaw|jaw$|顎|あご|下颌|下巴")
_RX_LOWMOUTH = _re.compile(r"(?i)lower_?lip|lip_?(lower|bottom|dn|d)([^a-z]|$)|下唇|(^|[^a-z])(c_|l_|r_)?d_?chin|chin_?(d|lower)")
_RX_BROW = _re.compile(r"(?i)brow|眉")
_RX_CHEEK = _re.compile(r"(?i)cheek|頬|脸颊")
_RX_CORNER = _re.compile(r"(?i)mouth_?corner|lip_?corner|口角")


def _quat_x(theta):
    return (math.sin(theta / 2), 0.0, 0.0, math.cos(theta / 2))


# --------------------------------------------------------------------------
# VRM 1.0 output (VRMC_vrm / VRMC_springBone / VRMC_materials_mtoon)
# --------------------------------------------------------------------------
VRM1_PRESET = {"joy": "happy", "angry": "angry", "sorrow": "sad", "fun": "relaxed", "surprised": "surprised",
               "a": "aa", "i": "ih", "u": "ou", "e": "ee", "o": "oh", "blink": "blink",
               "blink_l": "blinkLeft", "blink_r": "blinkRight", "neutral": "neutral",
               "lookup": "lookUp", "lookdown": "lookDown", "lookleft": "lookLeft", "lookright": "lookRight"}


def _mtoon1(vm, gm):
    """VRM0 MToon parameters -> VRMC_materials_mtoon 1.0 (same look: shading thresholds matched)"""
    fp, vp, tp = vm["floatProperties"], vm["vectorProperties"], vm.get("textureProperties", {})
    t0, s0 = float(fp.get("_ShadeToony", 0.9)), float(fp.get("_ShadeShift", 0.0))
    toony = 1.0 - (1.0 - t0) * (1.0 - s0) / 2.0
    shift = -1.0 + toony - s0
    shade = vp.get("_ShadeColor", [0.8, 0.8, 0.8, 1])[:3]
    ext = {
        "specVersion": "1.0",
        "transparentWithZWrite": fp.get("_BlendMode") == 3,
        "renderQueueOffsetNumber": 0,
        "shadeColorFactor": [srgb_to_linear(c) for c in shade],
        "shadingShiftFactor": round(shift, 4),
        "shadingToonyFactor": round(min(1.0, max(0.0, toony)), 4),
        "giEqualizationFactor": 0.9,
        "matcapFactor": [0, 0, 0],
        "parametricRimColorFactor": [0, 0, 0],
        "parametricRimFresnelPowerFactor": 1.0,
        "parametricRimLiftFactor": 0.0,
        "rimLightingMixFactor": 1.0,
        "outlineWidthMode": "none",
        "outlineWidthFactor": 0.0,
        "outlineColorFactor": [0, 0, 0],
        "outlineLightingMixFactor": 1.0,
        "uvAnimationScrollXSpeedFactor": 0.0, "uvAnimationScrollYSpeedFactor": 0.0,
        "uvAnimationRotationSpeedFactor": 0.0,
    }
    if "_ShadeTexture" in tp:
        ext["shadeMultiplyTexture"] = {"index": tp["_ShadeTexture"]}
    if fp.get("_OutlineWidthMode", 0) == 1 and fp.get("_OutlineWidth", 0) > 0:
        ext["outlineWidthMode"] = "worldCoordinates"
        ext["outlineWidthFactor"] = round(float(fp["_OutlineWidth"]) * 0.01, 5)  # cm -> m
        ext["outlineColorFactor"] = [srgb_to_linear(c) for c in vp.get("_OutlineColor", [0, 0, 0])[:3]]
        ext["outlineLightingMixFactor"] = float(fp.get("_OutlineLightingMix", 1.0))
    rq = int(vm.get("renderQueue", 2000))
    if gm.get("alphaMode") == "BLEND":
        if ext["transparentWithZWrite"]:
            ext["renderQueueOffsetNumber"] = int(min(9, max(0, rq - 2501)))
        else:
            ext["renderQueueOffsetNumber"] = int(min(0, max(-9, (rq - 3000) - 9)))
    return ext


def to_vrm1(gltf, vrm0, mesh_nodes, title, author):
    """rewrite a finished VRM0 glTF dict into VRM 1.0 (geometry must already be in +Z facing space)"""
    nodes = gltf["nodes"]
    meta0 = vrm0["meta"]
    meta = {
        "name": meta0.get("title") or "Model",
        "version": meta0.get("version", "1.0"),
        "authors": [meta0.get("author") or "Unknown"],
        "references": [meta0.get("reference", "")] if meta0.get("reference") else [],
        "licenseUrl": "https://vrm.dev/licenses/1.0/",
        "avatarPermission": "onlyAuthor",
        "allowExcessivelyViolentUsage": False,
        "allowExcessivelySexualUsage": False,
        "commercialUsage": "personalNonProfit",
        "allowPoliticalOrReligiousUsage": False,
        "allowAntisocialOrHateUsage": False,
        "creditNotation": "required",
        "allowRedistribution": False,
        "modification": "prohibited",
        "otherLicenseUrl": "",
    }
    thumb = {"ThumbProximal": "ThumbMetacarpal", "ThumbIntermediate": "ThumbProximal"}  # 0.x -> 1.0 names
    hb = {}
    for b in vrm0["humanoid"]["humanBones"]:
        k = b["bone"]
        for a, c in thumb.items():
            if k.endswith(a):
                k = k[:-len(a)] + c
                break
        hb[k] = {"node": int(b["node"])}
    fp0 = vrm0["firstPerson"]
    rng = lambda d: {"inputMaxValue": float(d["xRange"]), "outputScale": float(d["yRange"])}
    look = {"offsetFromHeadBone": [0.0, 0.06, 0.0],
            "type": "bone" if fp0["lookAtTypeName"] == "Bone" else "expression",
            "rangeMapHorizontalInner": rng(fp0["lookAtHorizontalInner"]),
            "rangeMapHorizontalOuter": rng(fp0["lookAtHorizontalOuter"]),
            "rangeMapVerticalDown": rng(fp0["lookAtVerticalDown"]),
            "rangeMapVerticalUp": rng(fp0["lookAtVerticalUp"])}
    # expressions
    mat_index = {m["name"]: i for i, m in enumerate(gltf["materials"])}
    preset, custom = {}, {}
    for g in vrm0["blendShapeMaster"]["blendShapeGroups"]:
        e = {"morphTargetBinds": [{"node": int(mesh_nodes[b["mesh"]]), "index": int(b["index"]),
                                   "weight": round(float(b["weight"]) / 100.0, 4)} for b in g["binds"]],
             "materialColorBinds": [{"material": mat_index[v["materialName"]], "type": "color",
                                     "targetValue": [srgb_to_linear(x) for x in v["targetValue"][:3]] +
                                                    [float(v["targetValue"][3])]}
                                    for v in g.get("materialValues", []) if v["materialName"] in mat_index],
             "isBinary": bool(g.get("isBinary")),
             "overrideBlink": "none", "overrideLookAt": "none", "overrideMouth": "none"}
        p = VRM1_PRESET.get(g.get("presetName", "unknown"))
        if not p and g.get("name") in ("驚き", "びっくり", "surprised", "Surprised"):
            p = "surprised"
        if p and p not in preset:
            if p in ("happy", "angry", "sad", "relaxed", "surprised"):
                e["overrideBlink"] = "blend"
            preset[p] = e
        else:
            name = g["name"] or "custom"
            base, k = name, 1
            while name in custom or name in VRM1_PRESET.values():
                k += 1
                name = "%s_%d" % (base, k)
            custom[name] = e
    vrmc = {"specVersion": "1.0", "meta": meta, "humanoid": {"humanBones": hb},
            "firstPerson": {"meshAnnotations": [{"node": int(n), "type": "auto"} for n in mesh_nodes]},
            "lookAt": look, "expressions": {"preset": preset, "custom": custom}}
    # spring bones: VRM0 groups of subtree roots -> linear chains of joints
    sa = vrm0["secondaryAnimation"]
    colliders, cgroups = [], []
    for gi, cg in enumerate(sa["colliderGroups"]):
        sph = [(np.array([c["offset"]["x"], c["offset"]["y"], -c["offset"]["z"]], float), float(c["radius"]))
               for c in cg["colliders"]]
        idx = []
        if len(sph) >= 2 and max(r for _, r in sph) - min(r for _, r in sph) < 1e-4:
            # spheres in a row with one radius: one capsule covers the gaps between them
            colliders.append({"node": int(cg["node"]), "shape": {"capsule": {
                "offset": [float(x) for x in sph[0][0]], "radius": sph[0][1],
                "tail": [float(x) for x in sph[-1][0]]}}})
            idx.append(len(colliders) - 1)
        else:
            for o, r in sph:
                colliders.append({"node": int(cg["node"]), "shape": {"sphere": {
                    "offset": [float(x) for x in o], "radius": r}}})
                idx.append(len(colliders) - 1)
        cgroups.append({"name": "colliders_%d" % gi, "colliders": idx})
    springs = []
    used = set()
    branches = {int(k): v for k, v in (sa.pop("_branches", None) or {}).items()}
    # a switched-off branch (pendant) of a chain: its bones must not swing with the host chain
    for r_, v_ in branches.items():
        if v_.get("disabled"):
            st_ = [r_]
            while st_:
                x_ = st_.pop()
                used.add(x_)
                st_.extend(nodes[x_].get("children", []))
    for bg0 in sa["boneGroups"]:
        for root in bg0["bones"]:
            stack = [([int(root)], bg0)]
            while stack:
                chain, pbg = stack.pop()
                # a pendant on a tail keeps its own settings, and so do all its strands
                bg = branches.get(chain[0], pbg)
                if bg.get("disabled"):
                    continue
                cur = chain[-1]
                while True:
                    kids = [c for c in nodes[cur].get("children", []) if c not in used]
                    if not kids:
                        break
                    kids.sort(key=lambda c: c in branches)  # the host follows its own continuation
                    for k2 in kids[1:]:
                        stack.append(([k2], bg))
                    cur = kids[0]
                    chain.append(cur)
                chain = [c for c in chain if c not in used]
                if len(chain) < 2:
                    continue
                used.update(chain)
                j = {"hitRadius": float(bg["hitRadius"]), "stiffness": float(bg["stiffiness"]),
                     "gravityPower": float(bg["gravityPower"]), "gravityDir": [0.0, -1.0, 0.0],
                     "dragForce": float(bg["dragForce"])}
                nj = len(chain)
                joints = []
                for ji, c in enumerate(chain):
                    jj = dict(j, node=int(c))
                    if nj >= 6:  # long chain: calmer, stiffer tip instead of a loose spring
                        t = ji / (nj - 1)
                        jj["dragForce"] = round(min(1.0, j["dragForce"] * (1.0 + 0.6 * t)), 4)
                        jj["stiffness"] = round(j["stiffness"] * (1.0 + 0.4 * t), 4)
                    joints.append(jj)
                springs.append({"name": "%s_%s" % (bg.get("comment", "spring"), nodes[chain[0]].get("name", "")),
                                "joints": joints,
                                "colliderGroups": [int(x) for x in bg["colliderGroups"]]})
    ext = {"VRMC_vrm": vrmc}
    used_ext = ["VRMC_vrm", "VRMC_materials_mtoon", "KHR_materials_unlit"]
    if springs:
        ext["VRMC_springBone"] = {"specVersion": "1.0", "colliders": colliders,
                                  "colliderGroups": cgroups, "springs": springs}
        used_ext.insert(1, "VRMC_springBone")
    for gm, vm in zip(gltf["materials"], vrm0["materialProperties"]):
        gm.setdefault("extensions", {})["VRMC_materials_mtoon"] = _mtoon1(vm, gm)
    gltf["extensions"] = ext
    gltf["extensionsUsed"] = used_ext
    return gltf


def flatten_face_normals(m, P, N, h, children, jo, we, mat_ov, log=print, blend=0.65):
    """Anime faces need 'flat' shading: blend the face-skin normals toward a sphere around the head so
    that the nose / cheek geometry does not cast round 3D-like shadows (in-place on N)."""
    head = h.get("head")
    if head is None or not len(P):
        return 0
    nb = len(children)
    hsub = np.zeros(nb, bool)
    st = [head]
    while st:
        k = st.pop()
        if not hsub[k]:
            hsub[k] = True
            st.extend(children[k])
    prim = jo[np.arange(len(jo)), np.argmax(we, axis=1)]
    vm = np.zeros(len(P), bool)
    off = 0
    names = unique_names([mt.name for mt in m.materials], "material")
    for k, mt in enumerate(m.materials):
        seg = m.indices[off:off + mt.index_count]
        off += mt.index_count
        texname = m.textures[mt.texture] if 0 <= mt.texture < len(m.textures) else ""
        cat = (mat_ov.get(names[k]) or {}).get("category") or guess_category(mt.name, mt.name_en, texname)
        c2, sub, _w = vocab.material_category(*[x for x in (mt.name, mt.name_en, texname) if x])
        if cat in ("face", "skin") and not (sub and c2 == cat):
            vm[seg] = True
    vm &= (prim >= 0) & hsub[np.clip(prim, 0, nb - 1)]
    idx = np.nonzero(vm)[0]
    if len(idx) < 30:
        return 0
    Q = P[idx]
    lo, hi = Q.min(axis=0), Q.max(axis=0)
    c = (lo + hi) / 2
    c[2] += 0.5 * (hi[1] - lo[1])  # MMD faces -Z: the sphere centre sits behind the face
    sph = Q - c
    sph /= np.maximum(np.linalg.norm(sph, axis=1, keepdims=True), 1e-9)
    n0 = N[idx]
    n0 = n0 / np.maximum(np.linalg.norm(n0, axis=1, keepdims=True), 1e-9)
    ok = np.einsum("ij,ij->i", n0, sph) > 0.0  # only outward facing skin (not mouth interior)
    nn = n0 * (1 - blend) + sph * blend
    nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
    N[idx[ok]] = nn[ok]
    log("  нормали лица выровнены для плоских аниме-теней: %d вершин" % int(ok.sum()))
    return int(ok.sum())


def synth_face_morphs(m, log=print):
    """Create bone morphs (まばたき/ウィンク/ウィンク右/あいうえお) for models that have
    eyelid / jaw bones but no such morphs.  Returns list of created names."""
    have = {nk(mo.name) for mo in m.morphs}
    nb = len(m.bones)
    if not nb:
        return []
    parent = [b.parent for b in m.bones]
    names = [b.name for b in m.bones]
    pos = np.array([b.pos for b in m.bones], float)
    P = m.positions.astype(np.float64)
    jo, we = m.joints, m.weights
    # descendants + subtree weighted vertex centroid
    kids = [[] for _ in range(nb)]
    for i, p in enumerate(parent):
        if 0 <= p < nb and p != i:
            kids[p].append(i)

    def subtree(i):
        out, st = [], [i]
        while st:
            k = st.pop()
            out.append(k)
            st.extend(kids[k])
        return out

    vw_cache = {}

    def centroid(i):
        if i not in vw_cache:
            sub = np.zeros(nb, bool)
            sub[subtree(i)] = True
            w = (we * sub[np.clip(jo, 0, nb - 1)] * (jo >= 0)).sum(axis=1)
            s = w.sum()
            vw_cache[i] = (P.T @ w / s, s) if s > 1e-6 else (None, 0.0)
        return vw_cache[i]

    def tops(cands):
        cs = set(cands)
        res = []
        for i in cands:
            p, ok = parent[i], True
            seen = 0
            while 0 <= p < nb and seen < nb:
                if p in cs:
                    ok = False
                    break
                p = parent[p]
                seen += 1
            if ok and centroid(i)[1] > 0:
                res.append(i)
        return res

    def rot_entry(b, hinge, theta):
        q = _quat_x(theta)
        R = quat_mat(q)
        t = R @ (pos[b] - hinge) + hinge - pos[b]
        return (b, tuple(t), q)

    def solve_theta(tip, hinge, dy):
        """angle about X through hinge so that tip moves by dy vertically (minimal |theta|)."""
        r = tip - hinge
        best = None
        for deg in np.arange(0.5, 85.0, 0.5):
            for sgn in (1, -1):
                th = math.radians(deg) * sgn
                y2 = r[1] * math.cos(th) - r[2] * math.sin(th)
                err = abs((y2 - r[1]) - dy)
                if best is None or err < best[0] - 1e-9:
                    best = (err, th)
        return best[1]

    created = []
    new = []
    # ---------------- eyelids ----------------
    lids = [i for i, n in enumerate(names) if _RX_LID.search(n) and not _RX_LIDSKIP.search(n)]
    groups = {}
    for i in tops(lids):
        n = names[i]
        c, _ = centroid(i)
        side = "L" if c[0] > 0 else "R"   # MMD: character's left at +X
        up = bool(_RX_UP.search(n)) and not _RX_DN.search(n)
        dn = bool(_RX_DN.search(n)) and not up
        if not (up or dn):
            groups.setdefault((side, "?"), []).append(i)
            continue
        groups.setdefault((side, "U" if up else "D"), []).append(i)
    # eye rings win over unmarked "lid" bones (on such rigs those are the skin fold above the eye): upper half
    # of the ring closes down, lower half rises a little. Marked upper / lower lid bones keep priority.
    ring, ring_ent = {}, {}
    for i, n in enumerate(names):
        mm = _RX_EYERING.match(n)
        if mm and centroid(i)[1] > 0:
            sd = "L" if pos[i][0] > 0 else "R"
            tok = mm.group(1).lower()
            if (tok.startswith("l") and sd != "L") or (tok.startswith("r") and sd != "R"):
                continue  # the name's side disagrees with where the bone is: not a ring we understand
            ring.setdefault(sd, []).append(i)

    def ancestors(i):
        out, p_, seen = set(), parent[i], 0
        while 0 <= p_ < nb and seen < nb:
            out.add(p_)
            p_ = parent[p_]
            seen += 1
        return out

    if len(ring.get("L", [])) >= 5 and len(ring.get("R", [])) >= 5:
        rs = set(ring["L"]) | set(ring["R"])
        for side in ("L", "R"):
            if (side, "U") in groups:
                continue
            rb = ring[side]
            px = [pos[i][0] for i in rb]
            py = [pos[i][1] for i in rb]
            if np.ptp(px) <= 1e-3 or np.ptp(py) <= 0.2 * np.ptp(px):
                continue  # all in one spot / a flat row: eyeball or highlight bones, not a lid ring
            cy = float(np.mean(py))
            xs = sorted(rb, key=lambda i: pos[i][0])
            corners = {xs[0], xs[-1]}
            ups = [i for i in rb if i not in corners and pos[i][1] > cy]
            dns = [i for i in rb if i not in corners and pos[i][1] <= cy]
            if not ups or not dns:
                continue
            gap = float(np.mean([pos[i][1] for i in ups])) - float(np.mean([pos[i][1] for i in dns]))
            if gap <= 1e-4:
                continue
            # loose lash bones over the upper lid go down with it
            for i, n in enumerate(names):
                if _RX_EYELASH.search(n) and i not in rs and (pos[i][0] > 0) == (side == "L") \
                        and pos[i][1] > cy and centroid(i)[1] > 0:
                    ups.append(i)
            folds = [i for i in groups.get((side, "?"), []) if pos[i][1] > cy]
            ent = [(i, -0.92 * gap) for i in ups] + [(i, 0.08 * gap) for i in dns] + \
                  [(i, -0.25 * gap) for i in folds]
            # a bone under another moved bone already moves with it: its own offset is only the difference
            want = {}
            for i, dy in ent + [(c_, 0.0) for c_ in corners]:  # corners stay put even under a moved bone
                want.setdefault(i, dy)
            local = {}
            for i in sorted(want, key=lambda k: len(ancestors(k))):
                local[i] = want[i] - sum(local.get(a_, 0.0) for a_ in ancestors(i))
            ent = [(i, dy) for i, dy in local.items() if abs(dy) > 1e-9]
            groups.pop((side, "?"), None)
            groups.pop((side, "D"), None)
            ring_ent[side] = (ent, gap)
        if ring_ent:
            log("  моргание: по кольцу костей вокруг глаз (%s)" % " + ".join(str(len(ring[s_])) for s_ in sorted(ring_ent)))
    for side in ("L", "R"):  # unmarked lid bones: the upper ones (above the lower lids / the only ones)
        unk = groups.pop((side, "?"), [])
        if not unk:
            continue
        if (side, "U") in groups:
            continue
        dns = groups.get((side, "D"), [])
        if dns:
            ly = np.mean([centroid(i)[0][1] for i in dns])
            unk = [i for i in unk if centroid(i)[0][1] > ly]
        if unk:
            groups[(side, "U")] = unk
    eye_ent, eye_par = {}, {}

    def eye_entries(side, f):
        if side in ring_ent:
            return [(i, (0.0, dy * f, 0.0), (0.0, 0.0, 0.0, 1.0)) for i, dy in ring_ent[side][0]]
        ups, dns, gap = eye_par[side]
        ent = []
        for lst, dy in ((ups, -0.85 * gap * f), (dns, 0.15 * gap * f)):
            for i in lst:
                tip = centroid(i)[0]
                hinge = pos[i]
                if np.linalg.norm(tip - hinge) < 0.15 * max(gap, 1e-3):
                    ent.append((i, (0.0, dy, 0.0), (0.0, 0.0, 0.0, 1.0)))
                else:
                    ent.append(rot_entry(i, hinge, solve_theta(tip, hinge, dy)))
        return ent

    for side in ("L", "R"):
        if side in ring_ent:
            eye_par[side] = (None, None, ring_ent[side][1])
            eye_ent[side] = eye_entries(side, 1.0)
            continue
        ups = groups.get((side, "U"), [])
        if not ups:
            continue
        dns = groups.get((side, "D"), [])
        cu = np.mean([centroid(i)[0] for i in ups], axis=0)
        if dns:
            cd = np.mean([centroid(i)[0] for i in dns], axis=0)
            gap = max(cu[1] - cd[1], 0.0)
        else:
            gap = 0.0
        if gap <= 1e-4:
            # no lower lid: vertical extent of the vertices that follow the upper lid
            sub = np.zeros(nb, bool)
            for i in ups:
                sub[subtree(i)] = True
            wl = (we * sub[np.clip(jo, 0, nb - 1)] * (jo >= 0)).sum(axis=1)
            ys = P[wl > 0.5, 1]
            gap = 0.7 * float(np.ptp(ys)) if len(ys) > 3 else 0.0
            if gap <= 1e-4:
                ext = np.ptp([centroid(i)[0][0] for i in ups]) if len(ups) > 1 else 0.0
                gap = 0.35 * max(ext, 0.05)
        eye_par[side] = (ups, dns, gap)
        eye_ent[side] = eye_entries(side, 1.0)
    if eye_ent:
        if nk("まばたき") not in have and nk("瞬き") not in have and len(eye_ent) == 2:
            new.append(("まばたき", "blink", eye_ent["L"] + eye_ent["R"]))
        if "L" in eye_ent and nk("ウィンク") not in have and nk("ウインク") not in have:
            new.append(("ウィンク", "wink", eye_ent["L"]))
        if "R" in eye_ent and nk("ウィンク右") not in have and nk("ウインク右") not in have:
            new.append(("ウィンク右", "wink_r", eye_ent["R"]))
    # ---------------- jaw / mouth ----------------
    jaws = [i for i, n in enumerate(names) if _RX_JAW.search(n)]
    jaw = None
    best = 0.0
    for i in jaws:
        c, s = centroid(i)
        if s > best:
            jaw, best = i, s
    if jaw is not None:
        hinge = pos[jaw]
        jsub = set(subtree(jaw))
        lowm = tops([i for i, n in enumerate(names)
                     if i not in jsub and _RX_LOWMOUTH.search(n) and not _RX_UP.search(n.replace("Lip", ""))])
        corners = tops([i for i, n in enumerate(names) if i not in jsub and _RX_CORNER.search(n)
                        and not _re.search(r"(?i)(^|_)(u|d)_", n)])
        c, _ = centroid(jaw)
        # sign: chin (jaw centroid) has to move down
        r = c - hinge
        sgn = 1.0 if (r[1] * math.cos(0.2) - r[2] * math.sin(0.2)) < r[1] else -1.0
        for name, deg in (("あ", 11.0), ("い", 3.0), ("う", 4.0), ("え", 6.0), ("お", 9.0)):
            if nk(name) in have:
                continue
            th = math.radians(deg) * sgn
            ent = [rot_entry(jaw, hinge, th)]
            ent += [rot_entry(i, hinge, th) for i in lowm]
            ent += [rot_entry(i, hinge, th * 0.5) for i in corners]
            new.append((name, name, ent))
    # ---------------- emotions: brows, mouth corners, cheeks, half-closed eyes ----------------
    def side_of(i):
        return "L" if centroid(i)[0][0] > 0 else "R"

    brows = tops([i for i, n in enumerate(names) if _RX_BROW.search(n)])
    corners = tops([i for i, n in enumerate(names) if _RX_CORNER.search(n) and not _re.search(r"(?i)inner|thick", n)])
    cheeks = tops([i for i, n in enumerate(names) if _RX_CHEEK.search(n)])
    if (brows or corners) and not any(mo.type == 1 for mo in m.morphs):
        bl = [centroid(i)[0] for i in brows if side_of(i) == "L"]
        br = [centroid(i)[0] for i in brows if side_of(i) == "R"]
        if bl and br:
            u = abs(np.mean(bl, axis=0)[0] - np.mean(br, axis=0)[0])
        elif eye_par:
            u = max(eye_par[k][2] for k in eye_par) * 4.0
        else:
            u = 0.0
        if u > 1e-4:
            bx = {s_: np.mean([centroid(i)[0][0] for i in brows if side_of(i) == s_]) if any(side_of(i) == s_ for i in brows) else 0.0
                  for s_ in ("L", "R")}

            def brow_moves(inner_up, outer_up, inward=0.0):
                ent = []
                for i in brows:
                    c = centroid(i)[0]
                    sd = side_of(i)
                    inner = abs(c[0]) < abs(bx[sd])
                    dy = (inner_up if inner else outer_up) * u
                    dx = -np.sign(c[0]) * inward * u if inner else 0.0
                    ent.append((i, (float(dx), float(dy), 0.0), (0.0, 0.0, 0.0, 1.0)))
                return ent

            def corner_moves(up, out=0.0):
                return [(i, (float(np.sign(centroid(i)[0][0]) * out * u), float(up * u), 0.0), (0.0, 0.0, 0.0, 1.0))
                        for i in corners]

            def cheek_moves(up):
                return [(i, (0.0, float(up * u), 0.0), (0.0, 0.0, 0.0, 1.0)) for i in cheeks]

            def eyes(f):
                return [e for sd in eye_par for e in eye_entries(sd, f)]

            emo = [("笑い", "joy", brow_moves(0.06, 0.06) + corner_moves(0.12, 0.04) + cheek_moves(0.05) + eyes(0.45)),
                   ("怒り", "angry", brow_moves(-0.10, 0.04, 0.05) + corner_moves(-0.04, -0.01) + eyes(0.15)),
                   ("困る", "sorrow", brow_moves(0.16, -0.06, -0.02) + corner_moves(-0.07) + eyes(0.25)),
                   ("にこり", "fun", brow_moves(0.03, 0.03) + corner_moves(0.05, 0.02) + cheek_moves(0.02)),
                   ("驚き", "surprised", brow_moves(0.14, 0.12) + eyes(-0.15))]
            for name, en, ent in emo:
                if nk(name) not in have and ent:
                    new.append((name, en, ent))
    for name, en, data in new:
        mo = pmx.Morph()
        mo.name, mo.name_en, mo.panel, mo.type, mo.data = name, en, 1 if en in ("blink", "wink", "wink_r") else 3, 2, data
        m.morphs.append(mo)
        created.append(name)
    if created:
        log("  выражения из костей: %s" % ", ".join(created))
    return created


# --------------------------------------------------------------------------
# humanoid mapping
# --------------------------------------------------------------------------
REQUIRED = ["hips", "spine", "chest", "neck", "head",
            "leftUpperArm", "leftLowerArm", "leftHand",
            "rightUpperArm", "rightLowerArm", "rightHand",
            "leftUpperLeg", "leftLowerLeg", "leftFoot",
            "rightUpperLeg", "rightLowerLeg", "rightFoot"]


def map_humanoid(find):
    def get(*cands):
        for c in cands:
            i = find.get(nk(c))
            if i is not None:
                return i
        return None

    h = {
        "hips": get("下半身", "腰"),
        "spine": get("上半身"),
        "chest": get("上半身2"),
        "upperChest": get("上半身3"),
        "neck": get("首"),
        "head": get("頭"),
        "jaw": None,
    }
    for side, jp in (("left", "左"), ("right", "右")):
        h[side + "Eye"] = get(jp + "目")
        h[side + "Shoulder"] = get(jp + "肩")
        h[side + "UpperArm"] = get(jp + "腕")
        h[side + "LowerArm"] = get(jp + "ひじ", jp + "肘")
        h[side + "Hand"] = get(jp + "手首")
        dl = (get(jp + "足D"), get(jp + "ひざD", jp + "膝D"), get(jp + "足首D"))
        if any(x is not None for x in dl) and not all(x is not None for x in dl):
            # incomplete D chain: per bone, prefer the D bone (it carries the weights)
            h[side + "UpperLeg"] = dl[0] if dl[0] is not None else get(jp + "足")
            h[side + "LowerLeg"] = dl[1] if dl[1] is not None else get(jp + "ひざ", jp + "膝")
            h[side + "Foot"] = dl[2] if dl[2] is not None else get(jp + "足首")
            h[side + "Toes"] = get(jp + "足先EX")
        elif all(x is not None for x in dl):
            # D-bones carry the mesh weights in most modern models
            h[side + "UpperLeg"], h[side + "LowerLeg"], h[side + "Foot"] = dl
            h[side + "Toes"] = get(jp + "足先EX")
        else:
            h[side + "UpperLeg"] = get(jp + "足")
            h[side + "LowerLeg"] = get(jp + "ひざ", jp + "膝")
            h[side + "Foot"] = get(jp + "足首")
            h[side + "Toes"] = None
        S = side
        t0, t1, t2 = get(jp + "親指0"), get(jp + "親指1"), get(jp + "親指2")
        if t0 is not None:
            th = (t0, t1, t2)
        else:
            th = (t1, t2, None)
        h[S + "ThumbProximal"], h[S + "ThumbIntermediate"], h[S + "ThumbDistal"] = th
        for fin, jpf in (("Index", ("人指", "人差指")), ("Middle", ("中指",)),
                         ("Ring", ("薬指",)), ("Little", ("小指",))):
            for k, part in enumerate(("Proximal", "Intermediate", "Distal")):
                h[S + fin + part] = get(*[jp + f + str(k + 1) for f in jpf])
    h = {k: v for k, v in h.items() if v is not None}
    # a bone must not be used twice
    seen = {}
    for k in list(h):
        if h[k] in seen:
            del h[k]
        else:
            seen[h[k]] = k
    return h


EN_BONES = {
    "hips": ["hips", "pelvis", "hip", "lowerbody", "waist"],
    "spine": ["spine", "spine0", "spine01", "upperbody", "torso"],
    "chest": ["chest", "spine1", "spine02", "upperbody2"],
    "upperChest": ["upperchest", "spine2", "spine03", "upperbody3"],
    "neck": ["neck", "neck1", "neck01"],
    "head": ["head"],
    "Eye": ["eye"],
    "Shoulder": ["shoulder", "clavicle", "collar"],
    "UpperArm": ["upperarm", "arm", "uparm"],
    "LowerArm": ["lowerarm", "forearm", "elbow", "lowarm"],
    "Hand": ["hand", "wrist"],
    "UpperLeg": ["upperleg", "upleg", "thigh", "leg", "legupper"],
    "LowerLeg": ["lowerleg", "calf", "knee", "shin", "leglower", "leg"],
    "Foot": ["foot", "ankle"],
    "Toes": ["toes", "toebase", "toe", "toe0", "ball"],
}
_PREFIXES = ("mixamorig", "ccbase", "jbipc", "jbip", "jsec", "jadj", "bip001", "bip01", "def", "bone",
             "valvebiped", "bip", "jnt")


def _en_key(name):
    """'J_Bip_L_UpperArm' -> ('left', 'upperarm'); 'Spine1' -> (None, 'spine1')."""
    n = nk(name).rsplit(":", 1)[-1].lower()  # drop namespaces like 'mixamorig:'
    side = None
    toks = [t for t in "".join(c if c.isalnum() else " " for c in n).split() if t]
    if not toks:
        return None, ""
    rest = []
    for i, t in enumerate(toks):
        if t in ("l", "left", "lf") and side is None:
            side = "left"
            continue
        if t in ("r", "right", "rt") and side is None:
            side = "right"
            continue
        rest.append(t)
    word = "".join(rest)
    for pre in ("left", "right"):
        if word.startswith(pre) and side is None:
            side, word = pre, word[len(pre):]
        if word.endswith(pre) and side is None:
            side, word = pre, word[:-len(pre)]
    changed = True
    while changed:
        changed = False
        for p in _PREFIXES:
            if word.startswith(p) and len(word) > len(p):
                word = word[len(p):]
                changed = True
    for suf in ("jnt", "jt", "bone"):
        if word.endswith(suf) and len(word) > len(suf):
            word = word[:-len(suf)]
    return side, word


def map_humanoid_geometry(pos, parent, wsum, height, log=print):
    """Humanoid bones from skeleton shape alone (rips with bone_001-style names).
    pos: bone positions (MMD axes: y up, character's left at +X, facing -Z); wsum: vertex weight per bone."""
    nb = len(parent)
    pos = np.asarray(pos, float)
    kids = [[] for _ in range(nb)]
    for i, p in enumerate(parent):
        if 0 <= p < nb and p != i:
            kids[p].append(i)
    # bones that deform something (directly or below)
    sub_w = np.array(wsum, float).copy()
    order = []
    st = [i for i in range(nb) if not (0 <= parent[i] < nb)]
    seen = set()
    while st:
        x = st.pop()
        if x in seen:
            continue
        seen.add(x)
        order.append(x)
        st.extend(kids[x])
    for x in reversed(order):
        p = parent[x]
        if 0 <= p < nb:
            sub_w[p] += sub_w[x]
    deform = [i for i in range(nb) if sub_w[i] > 1e-6]
    if len(deform) < 10:
        return {}
    H = float(height) if height > 0 else float(np.ptp(pos[deform, 1]) or 1.0)
    ground = float(pos[deform, 1].min())

    def path_to_root(i):
        out = []
        while 0 <= i < nb and i not in out:
            out.append(i)
            i = parent[i]
        return out

    def lca(a, b):
        pa = path_to_root(a)
        sa = set(pa)
        for x in path_to_root(b):
            if x in sa:
                return x
        return None

    def path(a, b):  # a ancestor of b -> [a, ..., b]
        p = path_to_root(b)
        if a not in p:
            return []
        return list(reversed(p[:p.index(a) + 1]))

    dw = [i for i in deform if wsum[i] > 1e-6] or deform
    low = [i for i in dw if pos[i, 1] < ground + 0.15 * H]
    fl = [i for i in low if pos[i, 0] > 0.02 * H]
    fr = [i for i in low if pos[i, 0] < -0.02 * H]
    if not fl or not fr:
        return {}
    def leg_ok(f, hp):
        return len([i for i in path(hp, f)[1:] if abs(pos[i, 0]) > 0.03 * H]) >= 3
    hips = None
    for footL in sorted(fl, key=lambda i: pos[i, 1])[:12]:
        for footR in sorted(fr, key=lambda i: pos[i, 1])[:12]:
            hp = lca(footL, footR)
            if hp is not None and pos[hp, 1] > ground + 0.3 * H and leg_ok(footL, hp) and leg_ok(footR, hp):
                hips = hp
                break
        if hips is not None:
            break
    if hips is None:
        return {}
    up = [i for i in dw if pos[i, 1] > pos[hips, 1]]
    if not up:
        return {}
    cx = (pos[footL, 0] + pos[footR, 0]) / 2  # body centre line (hips may sit off-centre in rips)

    def chain_len(c):
        n_, x = 1, c
        while True:
            nx = [k for k in kids[x] if sub_w[k] > 0]
            if not nx:
                return n_
            x = max(nx, key=lambda k: sub_w[k])
            n_ += 1

    def finger_score(i, sgn):
        """a wrist: several short chains of 2+ bones that all point away from the body"""
        sc = 0
        for c in kids[i]:
            if sub_w[c] <= 0 or chain_len(c) < 2:
                continue
            d = pos[c] - pos[i]
            far = c
            for _ in range(4):
                nx = [k for k in kids[far] if sub_w[k] > 0]
                if not nx:
                    break
                far = max(nx, key=lambda k: sub_w[k])
            v = pos[far] - pos[i]
            L = np.linalg.norm(v)
            if 0 < L < 0.12 * H and v[0] * sgn > 0.35 * L and np.linalg.norm(d) < 0.08 * H:
                sc += 1
        return sc

    def find_hand(sgn):
        cands = [i for i in up if (pos[i, 0] - cx) * sgn > 0.1 * H]
        if not cands:
            return None
        best = max(cands, key=lambda i: (min(finger_score(i, sgn), 5), (pos[i, 0] - cx) * sgn))
        if finger_score(best, sgn) >= 3:
            return best
        return max(cands, key=lambda i: (pos[i, 0] - cx) * sgn)

    handL, handR = find_hand(1), find_hand(-1)
    if handL is None or handR is None:
        return {}
    top = lca(handL, handR)
    if top is None:
        return {}
    if hips not in path_to_root(top):
        # MMD style: lower body and upper body are siblings under a centre bone
        hc = lca(hips, top)
        if hc is None or abs(pos[hc, 1] - pos[hips, 1]) > 0.15 * H:
            return {}
        hips = hc
    h = {"hips": hips}
    # spine
    S = path(hips, top)[1:]
    if not S:
        return {}
    if len(S) == 1:
        h["spine"] = S[0]
    elif len(S) == 2:
        h["spine"], h["chest"] = S
    else:
        h["spine"], h["chest"], h["upperChest"] = S[0], S[len(S) // 2] if len(S) > 3 else S[1], S[-1]
    # neck / head: child of `top` that leads upward the most
    cands = [c for c in kids[top] if sub_w[c] > 0 and c not in path_to_root(handL) and c not in path_to_root(handR)]
    if cands:
        def maxy(c):
            st_, m_ = [c], -1e9
            while st_:
                x = st_.pop()
                m_ = max(m_, pos[x, 1])
                st_.extend(kids[x])
            return m_
        neck = max(cands, key=maxy)
        if pos[neck, 1] > pos[top, 1] - 0.02 * H:
            h["neck"] = neck
            # head: first bone above the neck where face / hair bones branch out
            cur_, head, best_ = neck, None, 0
            for _ in range(6):
                nxt = [c for c in kids[cur_] if sub_w[c] > 0]
                if not nxt:
                    break
                cur_ = max(nxt, key=lambda c: sub_w[c])
                nk_ = len([c for c in kids[cur_] if sub_w[c] > 0])
                if nk_ > best_:
                    head, best_ = cur_, nk_
            if head is None:
                nxt = [c for c in kids[neck] if sub_w[c] > 0]
                head = max(nxt, key=lambda c: sub_w[c]) if nxt else None
            if head is not None and pos[head, 1] >= pos[neck, 1] - 0.01 * H:
                h["head"] = head
    # legs
    for side, foot in (("left", footL), ("right", footR)):
        P = path(hips, foot)[1:]
        lat = [i for i in P if abs(pos[i, 0] - cx) > 0.03 * H]
        P = []
        for i in lat:  # helpers sitting on the same joint (腰キャンセル ...): keep the deforming one
            if P and np.linalg.norm(pos[i] - pos[P[-1]]) < 0.004 * H:
                if wsum[i] >= wsum[P[-1]]:
                    P[-1] = i
            else:
                P.append(i)
        if len(P) < 3:
            continue
        ul = P[0]
        rest = P[1:]
        ky = ground + (pos[ul, 1] - ground) * 0.48
        knee = min(rest, key=lambda i: abs(pos[i, 1] - ky))
        after = rest[rest.index(knee) + 1:]
        if not after:
            continue
        ay = ground + 0.05 * H
        inner = [i for i in after if any(sub_w[c] > 0 for c in kids[i])] or after
        ankle = min(inner, key=lambda i: abs(pos[i, 1] - ay))
        h[side + "UpperLeg"], h[side + "LowerLeg"], h[side + "Foot"] = ul, knee, ankle
        toes = after[after.index(ankle) + 1:]
        if toes and pos[toes[0], 2] < pos[ankle, 2]:
            h[side + "Toes"] = toes[0]
    # arms
    fingers_of = {}
    for side, tip in (("left", handL), ("right", handR)):
        P = path(top, tip)[1:]
        if len(P) < 3:
            continue
        # drop helper bones sitting on top of another bone (肩P, twist roots ...)
        Q = []
        for i in P:
            if Q and np.linalg.norm(pos[i] - pos[Q[-1]]) < 0.004 * H:
                if wsum[i] > wsum[Q[-1]]:  # keep the bone that really deforms
                    Q[-1] = i
            else:
                Q.append(i)
        sgn_ = 1 if side == "left" else -1
        if finger_score(tip, sgn_) >= 3:  # tip is already the wrist
            hand = tip
            cand = Q
            D = np.linalg.norm(pos[tip] - pos[top]) / 0.79
            A = cand[:cand.index(hand)] if hand in cand else []
            nkid = None
        else:
            hand = None
        tipd = lambda i: np.linalg.norm(pos[tip] - pos[i])
        cand = [i for i in Q if i != tip] if hand is None else [i for i in Q]
        if len(cand) < 3:
            continue
        if hand is None:
            D = tipd(top)  # chest centre -> finger tip
            hand = min(cand, key=lambda i: abs(tipd(i) - 0.21 * D))  # wrist -> middle finger tip
        if hand != tip:
            near = [i for i in cand if abs(tipd(i) - 0.21 * D) < 0.13 * D]
            nkid = lambda i: len([c for c in kids[i] if sub_w[c] > 0])
            if near and max(nkid(i) for i in near) >= 3:  # the wrist is where the fingers fork
                hand = max(near, key=lambda i: (nkid(i) >= 3, -tipd(i) if nkid(i) >= 3 else 0))
        if hand not in cand:
            continue
        A = cand[:cand.index(hand)]
        if len(A) < 2:
            continue
        best = None
        for bi, ua_ in enumerate(A):
            for la_ in A[bi + 1:]:
                L1 = np.linalg.norm(pos[la_] - pos[ua_])
                L2 = np.linalg.norm(pos[hand] - pos[la_])
                if L1 + L2 <= 1e-9 or L2 < 0.2 * D:  # half a forearm (twist bone) is too short
                    continue
                sc = abs(L1 - 1.1 * L2) / (L1 + L2) + 0.5 * abs((L1 + L2) / D - 0.6)
                # twist bones are plain links (one child); real joints branch (twist helpers, sleeves)
                sc += 0.15 * sum(1 for x in (ua_, la_) if len([c for c in kids[x] if sub_w[c] > 0]) <= 1)
                if best is None or sc < best[0]:
                    best = (sc, ua_, la_)
        if best is None:
            continue
        _, ua, la = best
        B = A[:A.index(ua) + 1]
        before = B[:B.index(ua)]
        if before:
            h[side + "Shoulder"] = before[-1]
        h[side + "UpperArm"], h[side + "LowerArm"], h[side + "Hand"] = ua, la, hand
        fingers_of[side] = hand
    names_f = ["Thumb", "Index", "Middle", "Ring", "Little"]
    for side, hand in fingers_of.items():
        fc = [c for c in kids[hand] if sub_w[c] > 0]
        if len(fc) < 3:
            continue
        fc.sort(key=lambda c: pos[c, 2])  # MMD faces -Z: index finger is in front, little finger behind
        if len(fc) > 5:
            fc = fc[:5]
        if len(fc) == 5:  # the thumb starts closest to the wrist
            th = min(fc, key=lambda c: np.linalg.norm(pos[c] - pos[hand]))
            fc.remove(th)
            fc.insert(0, th)
        for fn, c in zip(names_f if len(fc) == 5 else names_f[1:] if len(fc) == 4 else names_f, fc):
            chain = [c]
            while len(chain) < 3:
                nx = [k for k in kids[chain[-1]] if sub_w[k] > 0]
                if not nx:
                    break
                chain.append(max(nx, key=lambda k: sub_w[k]))
            for seg, b in zip(("Proximal", "Intermediate", "Distal"), chain):
                h[side + fn + seg] = b
    log("  скелет распознан по форме: %d костей" % len(h))
    return h


def map_humanoid_en(names, h):
    """Fill missing humanoid bones from English / game-style names."""
    idx = {}
    for i, n in enumerate(names):
        side, word = _en_key(n)
        if word:
            idx.setdefault((side, word), i)
    used = set(h.values())
    for vb, words in EN_BONES.items():
        targets = [(None, vb)] if vb[0].islower() else [("left", "left" + vb), ("right", "right" + vb)]
        for side, key in targets:
            if key in h:
                continue
            for w in words:
                i = idx.get((side, w))
                if i is not None and i not in used:
                    h[key] = i
                    used.add(i)
                    break
    # fingers: Mixamo LeftHandIndex1, Unreal index_01_l, VRoid J_Bip_L_Index1, Biped Finger1/Finger11/Finger12
    import re as _re
    fing = {"thumb": "Thumb", "index": "Index", "middle": "Middle", "ring": "Ring", "pinky": "Little",
            "little": "Little"}
    parts = ("Proximal", "Intermediate", "Distal")
    for i, n in enumerate(names):
        side, word = _en_key(n)
        if side is None or i in used:
            continue
        word = word.replace("hand", "", 1) if word.startswith("hand") else word
        mm = _re.fullmatch(r"(thumb|index|middle|ring|pinky|little)(?:finger|f)?0?([1-3])", word)
        if mm:
            f, seg = fing[mm.group(1)], int(mm.group(2)) - 1
        else:
            mm = _re.fullmatch(r"finger([0-4])([0-9]?)", word)
            if not mm:
                continue
            f = ["Thumb", "Index", "Middle", "Ring", "Little"][int(mm.group(1))]
            seg = int(mm.group(2) or 0)
            if seg > 2:
                continue
        key = side + f + parts[seg]
        if key not in h:
            h[key] = i
            used.add(i)
    return h


# --------------------------------------------------------------------------
# material categories
# --------------------------------------------------------------------------
CATEGORIES = ["skin", "face", "eyes", "hair", "clothes", "accessory", "other"]
CATEGORY_NAMES = {"skin": "Тело (кожа)", "face": "Лицо", "eyes": "Глаза", "hair": "Волосы",
                  "clothes": "Одежда", "accessory": "Аксессуары", "other": "Прочее"}
_CAT_WORDS = [
    ("face", ("eyeshadow", "brow", "眉", "まつ", "睫", "lash", "口", "mouth", "歯", "teeth", "舌",
              "tongue", "顔", "颜", "脸", "face", "頬", "cheek", "表情", "blush")),
    ("eyes", ("目", "眼", "瞳", "eye", "iris", "pupil", "ハイライト", "highlight", "白目")),
    ("hair", ("髪", "发", "髮", "hair", "前髪", "後髪", "bang")),
    ("accessory", ("acc", "飾", "饰", "リボン", "ribbon", "帽", "hat", "cap", "角", "horn", "眼鏡",
                   "glass", "武器", "weapon", "gun", "sword", "剣", "翼", "wing", "尻尾", "tail",
                   "耳飾", "earring", "ネックレス", "necklace", "bag", "鞄", "包", "gadget", "jewel",
                   "ヘッドフォン", "headphone", "crown", "冠", "花", "flower", "badge")),
    ("clothes", ("服", "衣", "cloth", "裙", "スカート", "skirt", "shirt", "上着", "coat", "jacket",
                 "パンツ", "pants", "靴", "shoe", "boot", "袜", "靴下", "sock", "手袋", "glove", "dress",
                 "制服", "uniform", "水着", "swim", "bikini", "下着", "under", "bra", "stocking",
                 "tights", "タイツ", "裤", "鞋", "外套", "belt", "ベルト", "cape", "マント", "hood",
                 "sleeve", "袖", "collar", "襟", "apron", "vest", "top", "bottom")),
    ("skin", ("肌", "skin", "体", "身体", "皮肤", "body", "素体", "手", "hand", "neck", "首",
              "腕", "arm", "leg", "脚", "足")),
]


def guess_category(*names):
    cat, _sub, _w = vocab.material_category(*[n for n in names if n])
    if cat in CATEGORY_NAMES:
        return cat
    text = " ".join(nk(n).lower() for n in names if n)
    for cat, words in _CAT_WORDS:
        if any(w in text for w in words):
            return cat
    return "other"


# --------------------------------------------------------------------------
# blend shape presets
# --------------------------------------------------------------------------
PRESETS = [
    ("neutral", "Neutral", []),
    ("a", "A", [["あ"]]),
    ("i", "I", [["い"]]),
    ("u", "U", [["う"]]),
    ("e", "E", [["え"]]),
    ("o", "O", [["お"]]),
    ("blink", "Blink", [["まばたき", "瞬き"]]),
    ("blink_l", "Blink_L", [["ウィンク", "ウインク", "ウィンク2"]]),
    ("blink_r", "Blink_R", [["ウィンク右", "ウインク右", "ウィンク2右"]]),
    ("joy", "Joy", [["笑い", "笑い目", "smile", "eye_smile"], ["にっこり", "にやり", "口角上げ", "∧", "ワ"]]),
    ("angry", "Angry", [["怒り", "怒り眉", "angry"], ["怒り目", "じと目", "ジト目", "ｷﾘｯ"], ["口角下げ", "への字", "む"]]),
    ("sorrow", "Sorrow", [["困る", "困り眉", "悲しい", "sad"], ["困り目", "泣き", "悲しい目", "瞳小"], ["口角下げ", "への字"]]),
    ("fun", "Fun", [["にこり", "上", "喜び"], ["笑い", "なごみ", "笑い目"], ["にやり", "にっこり", "口角上げ"]]),
]


# --------------------------------------------------------------------------
# textures
# --------------------------------------------------------------------------
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tga", ".dds", ".spa", ".sph", ".gif", ".tif", ".tiff")
_index_cache = {}


def _file_index(base_dir):
    if base_dir not in _index_cache:
        by_name, by_stem = {}, {}
        n_files = 0
        root_depth = base_dir.rstrip(os.sep).count(os.sep)
        for dp, dn, fn in os.walk(base_dir):
            if dp.count(os.sep) - root_depth >= 6:
                dn[:] = []
            n_files += len(fn)
            if n_files > 30000:
                break
            for f in fn:
                full = os.path.join(dp, f)
                by_name.setdefault(f.lower(), full)
                st, ext = os.path.splitext(f)
                if ext.lower() in IMG_EXT:
                    by_stem.setdefault(st.lower(), full)
        _index_cache[base_dir] = (by_name, by_stem)
    return _index_cache[base_dir]


EXTRA_TEX_ROOTS = []  # folder chosen with "Открыть папку" — searched for textures too


def resolve_path(base_dir, rel):
    if rel and os.path.isabs(rel) and os.path.isfile(rel):
        return rel
    p = _resolve_exact(base_dir, rel)
    if p:
        return p
    q = _resolve_index(base_dir, rel)
    if q:
        return q
    for root in EXTRA_TEX_ROOTS:  # only for models opened from the scanned folder
        if root and os.path.isdir(root) and os.path.abspath(base_dir).startswith(os.path.abspath(root)):
            q = _resolve_index(root, rel)
            if q:
                return q
    return None


def _resolve_index(base_dir, rel):
    by_name, by_stem = _file_index(base_dir)
    fname = rel.replace("\\", "/").split("/")[-1].strip().lower()
    if not fname:
        return None
    if fname in by_name:
        return by_name[fname]
    stem, ext = os.path.splitext(fname)
    if ext not in IMG_EXT:
        stem = fname  # name without extension
    return by_stem.get(stem)


def _resolve_exact(base_dir, rel):
    rel = rel.replace("\\", "/").strip()
    p = os.path.join(base_dir, *rel.split("/"))
    if os.path.isfile(p):
        return p
    cur = base_dir
    for part in rel.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            cur = os.path.dirname(cur)
            continue
        try:
            entries = os.listdir(cur)
        except OSError:
            return None
        low = part.lower()
        hit = next((e for e in entries if e.lower() == low), None)
        if hit is None:
            return None
        cur = os.path.join(cur, hit)
    return cur if os.path.isfile(cur) else None


_TEX_CACHE = {}


def load_texture_cached(path, max_size=0, quality="normal", upscale_dir=None, sharpen=0.0):
    try:
        st = os.stat(path)
        key = (os.path.normcase(os.path.realpath(path)), st.st_mtime, st.st_size, max_size, quality,
               upscale_dir or "", round(float(sharpen or 0), 2))
    except OSError:
        key = None
    if key and key in _TEX_CACHE:
        return _TEX_CACHE[key]
    raw = None
    if upscale_dir:
        with open(path, "rb") as f:
            raw = upscaled_raw(f.read(), upscale_dir)
    res = load_texture(path, max_size, raw=raw, quality=quality, sharpen=sharpen)
    if key:
        if len(_TEX_CACHE) > 64:
            _TEX_CACHE.clear()
        _TEX_CACHE[key] = res
    return res


_PMX_CACHE = {}


def _finite(o):
    """last line of defence: no NaN/inf may reach the JSON chunk"""
    if isinstance(o, float):
        return o if math.isfinite(o) else 0.0
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return o


_HUMAN_BONE_SET = set(["hips", "spine", "chest", "upperChest", "neck", "head", "leftEye", "rightEye", "jaw"] +
                      [sd + b for sd in ("left", "right") for b in
                       ("Shoulder", "UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot", "Toes")] +
                      [sd + f + p for sd in ("left", "right")
                       for f in ("Thumb", "Index", "Middle", "Ring", "Little")
                       for p in ("Proximal", "Intermediate", "Distal")])
_CHAIN_NUM = {"stiffness": (0.0, 8.0), "gravity": (0.0, 5.0), "drag": (0.0, 1.0), "hitRadius": (0.0, 0.5)}


def _clean_overrides(ov):
    """user / sidecar overrides come from JSON: keep only well-typed, finite values"""
    if not isinstance(ov, dict):
        return {}
    out = {}
    h = ov.get("humanoid")
    if isinstance(h, dict):
        out["humanoid"] = {k: (v if isinstance(v, str) else "") for k, v in h.items()
                           if k in _HUMAN_BONE_SET and (v is None or isinstance(v, str))}
    ch = ov.get("chains")
    if isinstance(ch, dict):
        cc = {}
        for k, o in ch.items():
            if not isinstance(o, dict):
                continue
            d = {}
            for key, val in o.items():
                if key in _CHAIN_NUM:
                    try:
                        f = float(val)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(f):
                        lo, hi = _CHAIN_NUM[key]
                        d[key] = min(hi, max(lo, f))
                elif key == "kind" and isinstance(val, str):
                    d[key] = val
                elif key in ("enabled", "colliders"):
                    d[key] = bool(val)
            cc[str(k)] = d
        out["chains"] = cc
    mt = ov.get("materials")
    if isinstance(mt, dict):
        out["materials"] = {str(k): v for k, v in mt.items() if isinstance(v, dict)}
    pr = ov.get("presets")
    if isinstance(pr, dict):
        out["presets"] = {str(k): [x for x in v if isinstance(x, str)] for k, v in pr.items()
                          if isinstance(v, list)}
    rn = ov.get("renames")
    if isinstance(rn, dict):
        out["renames"] = {str(k): v.strip() for k, v in rn.items() if isinstance(v, str) and v.strip()}
    bf = ov.get("body_fill")
    if isinstance(bf, list):
        out["body_fill"] = [str(x) for x in bf if isinstance(x, str)]
    sp = ov.get("splits")
    if isinstance(sp, dict):
        out["splits"] = {str(k): True for k, v in sp.items() if v}
    wk = ov.get("wd_keep")
    if isinstance(wk, list):
        out["wd_keep"] = [str(x) for x in wk if isinstance(x, str)]
    wd = ov.get("wardrobe")
    if isinstance(wd, list):
        items = []
        for it in wd[:24]:
            if not isinstance(it, dict):
                continue
            d = {}
            for key in ("src", "id", "color", "color2", "pattern", "label", "uid"):
                if isinstance(it.get(key), str):
                    d[key] = it[key][:200]
            for key in ("color", "color2"):
                if key in d and not re.fullmatch(r"#[0-9a-fA-F]{6}", d[key]):
                    del d[key]
            if "pattern" in d and not re.fullmatch(r"[a-z]{2,12}", d["pattern"]):
                del d["pattern"]
            for key, lo, hi in (("size", 0.5, 2.0), ("loose", 0.0, 2.0), ("shift", -0.5, 0.5), ("length", 0.2, 2.0)):
                try:
                    f = float(it.get(key))
                    if math.isfinite(f):
                        d[key] = min(hi, max(lo, f))
                except (TypeError, ValueError):
                    pass
            br = []
            for st in (it.get("brush") or [])[:3000]:
                try:
                    p_ = [float(x) for x in st["p"]][:3]
                    r_, s_ = float(st["r"]), float(st["s"])
                    if len(p_) == 3 and all(math.isfinite(x) for x in p_ + [r_, s_]):
                        br.append({"p": p_, "r": min(0.2, max(0.002, r_)), "s": min(0.05, max(-0.05, s_))})
                except (TypeError, ValueError, KeyError):
                    pass
            if br:
                d["brush"] = br
            if d.get("id"):
                items.append(d)
        out["wardrobe"] = items
    return out


# --------------------------------------------------------------------------
# splitting one material into its separate pieces (all accessories in one mesh -> one per piece)
# --------------------------------------------------------------------------
_ISLAND_CACHE = {}


def material_pieces(m, k, max_pieces=64):
    """Pieces of material k: [(triangle row indices, dominant bone)], biggest first.
    Pieces = parts of the mesh not connected to each other (vertices on UV seams are welded by position);
    crumbs (< 0.3 % of the faces) join a piece hanging on the same bone."""
    key = (id(m.indices), id(m.positions), len(m.indices), k, m.materials[k].name, m.materials[k].index_count)
    if key in _ISLAND_CACHE:
        return _ISLAND_CACHE[key]
    off = sum(mt.index_count for mt in m.materials[:k])
    cnt = m.materials[k].index_count // 3 * 3
    tri = np.asarray(m.indices[off:off + cnt], np.int64).reshape(-1, 3)
    if len(tri) == 0:
        return []
    used, inv = np.unique(tri.reshape(-1), return_inverse=True)
    P = np.asarray(m.positions)[used]
    ext = float(np.ptp(np.asarray(m.positions), axis=0).max()) or 1.0
    q = np.round(P / (ext * 1e-5)).astype(np.int64)
    _, weld = np.unique(q, axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    t = weld[inv.reshape(-1, 3)]
    lab = np.arange(weld.max() + 1)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [0, 2]]])
    for _ in range(200):  # min-label propagation with pointer jumping
        old_ = lab.copy()
        mn = np.minimum(lab[e[:, 0]], lab[e[:, 1]])
        np.minimum.at(lab, e[:, 0], mn)
        np.minimum.at(lab, e[:, 1], mn)
        for _j in range(8):
            lab = lab[lab]
        if np.array_equal(lab, old_):
            break
    tl = lab[t[:, 0]]
    ids, tri_isl = np.unique(tl, return_inverse=True)
    J = np.asarray(m.joints)
    Wt = np.asarray(m.weights)
    vb = J[np.arange(len(J)), np.argmax(Wt, axis=1)]
    pieces = []
    for i in range(len(ids)):
        rows = np.nonzero(tri_isl == i)[0]
        vs = np.unique(tri[rows].reshape(-1))
        bones, cnts = np.unique(vb[vs], return_counts=True)
        pieces.append([rows, int(bones[np.argmax(cnts)])])
    total = len(tri)
    big = [p for p in pieces if len(p[0]) >= max(12, total * 0.003)]
    small = [p for p in pieces if len(p[0]) < max(12, total * 0.003)]
    if not big:
        big, small = pieces, []
    by_bone = {}
    for p in big:
        by_bone.setdefault(p[1], p)
    def join(dst, p):
        dst[0] = np.concatenate([dst[0], p[0]])

    # crumbs without a big piece on their bone: one piece per bone (a strap on the tail stays its own part)
    for p in small:
        host = by_bone.get(p[1])
        if host is not None:
            join(host, p)
        else:
            by_bone[p[1]] = p
            big.append(p)
    big.sort(key=lambda p: (-len(p[0]), int(p[0].min())))
    if len(big) > max_pieces - 1:
        keep, over = big[:max_pieces - 1], big[max_pieces - 1:]
        rest = []
        for p in over:  # too many: the smallest ones join a kept piece on the same bone, if any
            host = next((q for q in keep if q[1] == p[1]), None)
            if host is not None:
                join(host, p)
            else:
                rest.append(p)
        big = keep
        if rest:
            big.append([np.concatenate([p[0] for p in rest]), -1])
    out = [(np.sort(p[0]), p[1]) for p in big]
    if len(_ISLAND_CACHE) > 256:
        _ISLAND_CACHE.clear()
    _ISLAND_CACHE[key] = out
    return out


_BODY_CACHE = {}
_ALPHA_CACHE = {}  # texture path -> alpha channel (or None)


def vertex_opaque(m, mats, base_dir):
    """per vertex: does the texture under it show something (alpha >= 128)? Transparent skin (the faded
    top of a stocking, a cut-out) does not count as skin."""
    from PIL import Image
    ok = np.ones(len(m.positions), bool)
    off = 0
    for k, mt in enumerate(m.materials):
        cnt = mt.index_count
        if k in mats and 0 <= mt.texture < len(m.textures):
            p_ = resolve_path(base_dir, m.textures[mt.texture]) if base_dir else None
            if p_:
                try:
                    a = _ALPHA_CACHE.get(p_, False)
                    if a is False:
                        img = Image.open(p_)
                        a = (np.asarray(img.convert("RGBA").getchannel("A"))
                             if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info else None)
                        if len(_ALPHA_CACHE) > 32:
                            _ALPHA_CACHE.clear()
                        _ALPHA_CACHE[p_] = a
                    if a is not None:
                        hh, ww = a.shape
                        vs = np.unique(np.asarray(m.indices[off:off + cnt]))
                        uv = np.asarray(m.uvs)[vs]
                        av = a[(np.clip(np.mod(uv[:, 1], 1.0), 0, 0.9999) * hh).astype(int),
                               (np.clip(np.mod(uv[:, 0], 1.0), 0, 0.9999) * ww).astype(int)]
                        ok[vs[av < 128]] = False
                except Exception:
                    pass
        off += cnt
    return ok


def _build_fill(m, parts, h, skin, other, hidden, best, rel_, log, base_dir):
    """generated body with normals, UV and weights, or None"""
    import bodyfill
    opaque = vertex_opaque(m, skin, base_dir)
    res = bodyfill.build_body(m, h, set(parts), skin, other, log, hidden_mats=hidden, opaque=opaque)
    if res is None:
        return None
    V, N, T, J, W = res
    skin_sel = np.zeros(len(m.positions), bool)
    off = 0
    for k, mt in enumerate(m.materials):
        if k == best:
            skin_sel[np.asarray(m.indices[off:off + mt.index_count])] = True
        off += mt.index_count
    Hm = float(np.ptp(np.asarray(m.positions)[:, 1])) or 20.0
    Pm = np.asarray(m.positions, float)
    # plain skin: texels close to the typical skin tone (no blush, shadow, lines, tattoos) — the body
    # far from the seams takes its texture only from there, so it has an even tone
    plain = skin_sel.copy()
    tp_ = resolve_path(base_dir, rel_) if (rel_ and base_dir) else None
    if tp_:
        try:
            from PIL import Image
            img_ = np.asarray(Image.open(tp_).convert("RGB")).astype(np.float32)
            hh_, ww_ = img_.shape[:2]
            vs_ = np.nonzero(skin_sel)[0]
            uv_ = np.asarray(m.uvs)[vs_]
            col = img_[(np.clip(np.mod(uv_[:, 1], 1.0), 0, 0.9999) * hh_).astype(int),
                       (np.clip(np.mod(uv_[:, 0], 1.0), 0, 0.9999) * ww_).astype(int)]
            med = np.median(col, axis=0)
            dist = np.linalg.norm(col - med, axis=1)
            keep = dist < max(14.0, float(np.percentile(dist, 55)))
            if keep.sum() > 50:
                plain[:] = False
                plain[vs_[keep]] = True
        except Exception:
            pass
    UV = bodyfill.transfer_uv(V, Pm, np.asarray(m.uvs), skin_sel, Hm, plain_sel=plain)
    # soft seams: next to the real skin the generated surface takes over its normals, so light and
    # shadow continue across the edge instead of drawing a line
    all_skin = np.zeros(len(Pm), bool)
    off = 0
    for k, mt in enumerate(m.materials):
        if k in skin:
            all_skin[np.asarray(m.indices[off:off + mt.index_count])] = True
        off += mt.index_count
    S_ = np.nonzero(all_skin)[0]
    if len(S_):
        j_, d_ = bodyfill.nearest(V, Pm[S_], 0.03 * Hm)
        ok_ = j_ >= 0
        w_ = np.zeros(len(V))
        w_[ok_] = np.clip(1.0 - d_[ok_] / (0.03 * Hm), 0.0, 1.0) ** 1.5
        Ns = np.asarray(m.normals, float)[S_]
        Nb = N.copy()
        Nb[ok_] = (1 - w_[ok_, None]) * N[ok_] + w_[ok_, None] * Ns[j_[ok_]]
        N = Nb / np.maximum(np.linalg.norm(Nb, axis=1), 1e-9)[:, None]
    # a triangle whose corners took UVs from far-apart places of the atlas would smear the whole texture
    # between them: such triangles get the UV of one plain-skin texel
    if len(T):
        tu = UV[T]
        spread = tu.max(axis=1) - tu.min(axis=1)
        wide = (spread > 0.05).any(axis=1)
        if wide.any():
            pv = np.nonzero(plain)[0]
            if len(pv):
                c_uv = np.median(np.asarray(m.uvs)[pv], axis=0)
                pick = pv[np.argmin(np.linalg.norm(np.asarray(m.uvs)[pv] - c_uv, axis=1))]
                # each such triangle gets its own vertices so the neighbours keep their UVs
                tw = T[wide]
                nv = len(V)
                V = np.concatenate([V, V[tw.reshape(-1)]])
                N = np.concatenate([N, N[tw.reshape(-1)]])
                J = np.concatenate([J, J[tw.reshape(-1)]])
                W = np.concatenate([W, W[tw.reshape(-1)]])
                UV = np.concatenate([UV, np.repeat(np.asarray(m.uvs)[pick][None, :], tw.size, 0).astype(UV.dtype)])
                T = T.copy()
                T[wide] = nv + np.arange(tw.size).reshape(-1, 3)
    return V, N, UV, T, J, W


def add_body_fill(m, parts, ov, log=print, base_dir=None):
    """Append a generated body (bodyfill.py) as a new material "BodyFill" that uses the skin texture."""
    import bodyfill
    names = [b.name for b in m.bones]
    find = {}
    for i, n in enumerate(names):
        find.setdefault(nk(n), i)
    h = map_humanoid_en(names, map_humanoid(find))
    for vb, bname in (ov.get("humanoid") or {}).items():
        if bname in names:
            h[vb] = names.index(bname)
    mo = ov.get("materials") or {}
    skin, other, hidden = set(), set(), set()
    for k, mt in enumerate(m.materials):
        if (mo.get(mt.name) or {}).get("visible", True) is False:
            other.add(k)  # hidden clothes still tell the size of the body under them
            hidden.add(k)
            continue
        cat = (mo.get(mt.name) or {}).get("category") or guess_category(mt.name, mt.name_en, "")
        (skin if cat in ("skin", "face") or (mo.get(mt.name) or {}).get("as_skin") else other).add(k)
    if not skin:
        log("  ! достройка тела: нет материала кожи — отметьте категорию «Кожа» у материала тела")
        return
    # the skin material: the biggest visible material of category "skin"
    best, bf = None, -1
    for k in skin:
        mt = m.materials[k]
        if guess_category(mt.name, mt.name_en, "") == "skin" and mt.index_count > bf:
            best, bf = k, mt.index_count
    if best is None:
        best = max(skin, key=lambda k: m.materials[k].index_count) if skin else 0
    mt_ = m.materials[best]
    rel_ = m.textures[mt_.texture] if 0 <= mt_.texture < len(m.textures) else ""
    ckey = (id(m.positions), len(m.positions), tuple(sorted(parts)), tuple(sorted(skin)), tuple(sorted(hidden)),
            tuple(sorted(h.items())), tuple(mt.name for mt in m.materials), best, rel_, base_dir)
    if ckey in _BODY_CACHE:
        res = _BODY_CACHE[ckey]
        if res is not None:
            log("  достройка тела: %d треугольников (из кэша)" % len(res[3]))
    else:
        res = _build_fill(m, parts, h, skin, other, hidden, best, rel_, log, base_dir)
        if len(_BODY_CACHE) > 8:
            _BODY_CACHE.clear()
        _BODY_CACHE[ckey] = res
    if res is None:
        return
    V, N, UV, T, J, W = res
    n0 = len(m.positions)
    m.positions = np.concatenate([m.positions, V.astype(m.positions.dtype)])
    m.normals = np.concatenate([m.normals, N.astype(m.normals.dtype)])
    m.uvs = np.concatenate([m.uvs, UV.astype(m.uvs.dtype)])
    m.joints = np.concatenate([m.joints, J.astype(m.joints.dtype)])
    m.weights = np.concatenate([m.weights, W.astype(m.weights.dtype)])
    m.indices = np.concatenate([np.asarray(m.indices), (T + n0).reshape(-1).astype(np.asarray(m.indices).dtype)])
    bm = copy.copy(m.materials[best])
    bm.name = "BodyFill"
    bm.name_en = "Body fill"
    bm.index_count = len(T) * 3
    bm.flags = bm.flags & ~0x01 & ~0x10  # one-sided: it is a closed body; no outline (no seam line)
    if hasattr(bm, "edge_size"):
        bm.edge_size = 0.0
    m.materials = list(m.materials) + [bm]


def _src_humanoid(m, ov):
    """humanoid bones of the source model, found the same way as the conversion does (names, the VRM/FBX hint,
    the shape of the skeleton when names do not help, the user's manual choices)"""
    names = [b.name for b in m.bones]
    find = {}
    for i, n in enumerate(names):
        find.setdefault(nk(n), i)
    h = map_humanoid_en(names, map_humanoid(find))
    hint = getattr(m, "humanoid_hint", None)
    if hint:
        for k_, v_ in hint.items():
            if 0 <= v_ < len(names):
                for k2 in [k2 for k2, v2 in h.items() if v2 == v_ and k2 != k_]:
                    del h[k2]
                h[k_] = v_
    if sum(1 for b in REQUIRED if b not in h) >= 4:
        try:
            pos = np.array([b.pos for b in m.bones], float)
            parent = np.array([b.parent for b in m.bones])
            wsum_ = np.zeros(len(names))
            okj = (m.joints >= 0) & (m.joints < len(names))
            np.add.at(wsum_, m.joints[okj], m.weights[okj])
            hg = map_humanoid_geometry(pos, parent, wsum_, float(np.ptp(m.positions[:, 1])) if len(m.positions) else 0,
                                       lambda *a: None)
            used_ = set(h.values())
            for k_, v_ in hg.items():
                if k_ not in h and v_ not in used_:
                    h[k_] = v_
                    used_.add(v_)
        except Exception:
            pass
    for vb, bname in (ov.get("humanoid") or {}).items():
        if not bname:
            h.pop(vb, None)
            continue
        if bname in names:
            i = names.index(bname)
            for k in [k for k, v in h.items() if v == i and k != vb]:
                del h[k]
            h[vb] = i
    return h


def add_wardrobe(m, ov, log=print):
    """Put on clothes from the wardrobe tab (built-in, generated to the body / from the catalog)."""
    import wardrobe
    h = _src_humanoid(m, ov)
    mo = ov.get("materials") or {}
    cats = {}
    for k, mt in enumerate(m.materials):
        texname = m.textures[mt.texture] if 0 <= mt.texture < len(m.textures) else ""
        auto = "skin" if mt.name == "BodyFill" else guess_category(mt.name, mt.name_en, texname)
        cats[k] = (mo.get(mt.name) or {}).get("category") or auto
        if (mo.get(mt.name) or {}).get("as_skin"):
            cats[k] = "skin"
        if cats[k] == "skin" and (mo.get(mt.name) or {}).get("visible", True) is False and mt.name != "BodyFill":
            cats[k] = "hidden_skin"
    wardrobe.apply_wardrobe(m, h, ov["wardrobe"], cats, log)


TAIL_SUFFIX = " · хвост"


def keep_body_parts(m, names, log=print):
    """The tail and animal ears are part of the body: when the wardrobe takes the model's clothes off and the tail
    sits in the same material as the clothes (often: underwear), its triangles go to their own visible material
    "<name> · хвост", so the tail stays."""
    kind = np.array([(_kind_strict(b.name) or "") for b in m.bones] + [""])
    J = np.asarray(m.joints)
    W = np.asarray(m.weights, float)
    Jc = np.where(J < 0, len(kind) - 1, np.minimum(J, len(kind) - 1))
    Wn = np.where(J < 0, 0.0, W)
    Wn = Wn / np.maximum(Wn.sum(1, keepdims=True), 1e-9)
    tailw = (Wn * np.isin(kind[Jc], ["tail", "ear"])).sum(1)
    new_mats, new_idx = [], []
    off = 0
    moved = 0
    extra = []
    for mt in m.materials:
        seg = np.asarray(m.indices[off:off + mt.index_count]).reshape(-1, 3)
        off += mt.index_count
        if mt.name in names and len(seg):
            sel = tailw[seg].mean(1) > 0.5
            if sel.any() and not sel.all():
                body = copy.copy(mt)
                body.index_count = int((~sel).sum() * 3)
                tail = copy.copy(mt)
                tail.name = mt.name + TAIL_SUFFIX
                tail.name_en = (mt.name_en or "") + " tail"
                tail.index_count = int(sel.sum() * 3)
                new_mats.append(body)
                new_idx.append(seg[~sel].reshape(-1))
                extra.append((tail, seg[sel].reshape(-1)))
                moved += int(sel.sum())
                continue
        new_mats.append(mt)
        new_idx.append(seg.reshape(-1))
    if not extra:
        return
    for mt, ix in extra:
        new_mats.append(mt)
        new_idx.append(ix)
    m.materials = new_mats
    m.indices = np.concatenate(new_idx).astype(np.asarray(m.indices).dtype)
    log("  хвост / уши оставлены на теле: %d треугольников" % moved)


def apply_wardrobe_brush(m, P, N, vrm1, log=print):
    """Brush strokes of the wardrobe tab, applied to the finished garments in the VRM T-pose space.
    Strokes are stored in the preview's space (VRM 1.0 axes, metres): {p, r: radius, s: push (+out / -in)}."""
    owner = getattr(m, "_wd_owner", None)
    if not owner:
        return
    to_file = np.array([1.0, 1.0, 1.0]) if vrm1 else np.array([-1.0, 1.0, -1.0])
    off = 0
    for mt in m.materials:
        cnt = mt.index_count
        o = owner.get(mt.name)
        if o and o.get("brush"):
            vs = np.unique(np.asarray(m.indices[off:off + cnt]))
            V = P[vs]
            Nn = N[vs] / np.maximum(np.linalg.norm(N[vs], axis=1), 1e-9)[:, None]
            disp = np.zeros(len(vs))
            for st in o["brush"]:
                p = np.asarray(st["p"], float) * to_file
                d = np.linalg.norm(V - p, axis=1)
                w = np.clip(1.0 - d / max(float(st["r"]), 1e-6), 0, 1)
                disp += float(st["s"]) * w * w * (3 - 2 * w)
            disp = np.clip(disp, -0.04, 0.08)
            P[vs] = V + Nn * disp[:, None]
            log("  гардероб: кисть — %s (%d мазков)" % (mt.name, len(o["brush"])))
        off += cnt


def prepare_source(src, overrides, with_wardrobe=True, log=print):
    """the model as the converter sees it right before conversion (splits, wardrobe clothes), and its humanoid
    bones — for saving clothes of this model into the wardrobe catalog"""
    ov = _clean_overrides(overrides)
    m = copy.copy(load_pmx_cached(src, log))
    m.morphs = list(m.morphs)
    apply_splits(m, ov.get("splits"), log)
    if with_wardrobe and ov.get("wardrobe"):
        add_wardrobe(m, ov, log)
    return m, _src_humanoid(m, ov), ov


def apply_splits(m, splits, log=print):
    """Replace every material listed in splits by one material per piece ("name#1", "name#2" ...).
    Returns {new material name: (original name, label)}."""
    if not splits:
        return {}
    names = [mt.name for mt in m.materials]
    new_mats, new_idx, remap, info = [], [], {}, {}
    off = 0
    for k, mt in enumerate(m.materials):
        cnt = mt.index_count
        seg = np.asarray(m.indices[off:off + cnt])
        off += cnt
        pieces = material_pieces(m, k) if names[k] in splits else []
        if len(pieces) < 2:
            remap[k] = [len(new_mats)]
            new_mats.append(mt)
            new_idx.append(seg)
            continue
        tri = seg[:cnt // 3 * 3].reshape(-1, 3)
        remap[k] = []
        for i, (rows, bone) in enumerate(pieces):
            pm = copy.copy(mt)
            pm.name = "%s#%d" % (mt.name, i + 1)
            pm.index_count = len(rows) * 3
            label = (names_ru.translate(m.bones[bone].name) if 0 <= bone < len(m.bones) else "мелкие куски")
            info[pm.name] = (mt.name, label)
            remap[k].append(len(new_mats))
            new_mats.append(pm)
            new_idx.append(tri[rows].reshape(-1))
        log("  материал «%s» разделён на %d кусков" % (mt.name, len(pieces)))
    m.materials = new_mats
    m.indices = np.concatenate(new_idx) if new_idx else m.indices
    morphs = []
    for mo in m.morphs:
        if mo.type == 8 and mo.data:
            mo2 = copy.copy(mo)
            data = []
            for d in mo.data:
                if d["mat"] < 0:
                    data.append(d)
                else:
                    for nk_ in remap.get(d["mat"], []):
                        d2 = dict(d)
                        d2["mat"] = nk_
                        data.append(d2)
            mo2.data = data
            morphs.append(mo2)
        else:
            morphs.append(mo)
    m.morphs = morphs
    return info


def load_pmx_cached(path, log=print):
    st = os.stat(path)
    key = (os.path.normcase(os.path.realpath(path)), st.st_mtime, st.st_size)
    if key not in _PMX_CACHE:
        _PMX_CACHE.clear()
        if path.lower().endswith((".vrm", ".glb")):
            import vrmin
            try:
                _PMX_CACHE[key] = vrmin.load(path, log=log)
            except (struct.error, IndexError, ValueError, AttributeError, KeyError, TypeError, MemoryError) as e:
                raise ConvertError("VRM повреждён или не поддерживается: %s" % e)
        elif path.lower().endswith(".fbx"):
            import fbx
            try:
                _PMX_CACHE[key] = fbx.load(path, log=log)
            except (struct.error, IndexError, ValueError, AttributeError, KeyError, MemoryError, OverflowError) as e:
                raise ConvertError("FBX повреждён или не поддерживается: %s" % e)
        else:
            try:
                _PMX_CACHE[key] = pmx.load(path)
            except (struct.error, IndexError, ValueError, AttributeError, MemoryError, OverflowError) as e:
                raise ConvertError("Файл модели повреждён или не поддерживается: %s" % e)
        sanitize_model(_PMX_CACHE[key])
    return _PMX_CACHE[key]


def sanitize_model(m):
    """NaN/inf in a broken file would make the whole glTF JSON invalid: replace with zeros."""
    for a in ("positions", "normals", "uvs", "weights"):
        v = getattr(m, a, None)
        if v is not None and len(v) and not np.isfinite(v).all():
            setattr(m, a, np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(v.dtype))
    for b in m.bones:
        if not np.isfinite(np.asarray(b.pos, float)).all():
            b.pos = np.nan_to_num(np.asarray(b.pos, float)).astype(np.float32)
    for mo in m.morphs:
        if mo.type == 1 and mo.data is not None and len(mo.data) == 2:
            ix, off = mo.data
            if len(off) and not np.isfinite(off).all():
                mo.data = (ix, np.nan_to_num(off, nan=0.0, posinf=0.0, neginf=0.0))
    for rb in m.rigid_bodies:
        for a in ("pos", "rot", "size"):
            v = getattr(rb, a, None)
            if v is not None and not np.isfinite(np.asarray(v, float)).all():
                setattr(rb, a, tuple(float(x) for x in np.nan_to_num(np.asarray(v, float))))


def tex_hash(raw):
    import hashlib
    return hashlib.sha1(raw).hexdigest()[:20]


def upscaled_raw(raw, upscale_dir):
    """the AI-upscaled copy of a texture (made by app.py with Real-ESRGAN), if there is one"""
    if upscale_dir:
        p = os.path.join(upscale_dir, tex_hash(raw) + ".png")
        if os.path.isfile(p):
            with open(p, "rb") as f:
                return f.read()
    return raw


def model_textures(src, log=lambda s: None):
    """[(hash, raw bytes, name)] of the colour textures a model uses (for the AI upscaler)"""
    m = load_pmx_cached(src, log)
    base_dir = os.path.dirname(os.path.abspath(src))
    out, seen = [], set()
    used = {mt.texture for mt in m.materials if mt.texture is not None and mt.texture >= 0}
    for ti in sorted(used):
        if ti >= len(m.textures):
            continue
        raw = getattr(m, "texture_blobs", {}).get(ti)
        if raw is None:
            p = resolve_path(base_dir, m.textures[ti])
            if p is None:
                continue
            with open(p, "rb") as f:
                raw = f.read()
        hsh = tex_hash(raw)
        if hsh not in seen:
            seen.add(hsh)
            out.append((hsh, raw, m.textures[ti]))
    return out


def bleed_alpha(img):
    """Fill the colour of fully transparent pixels from their opaque neighbours, so filtering and
    mip-maps don't drag a dark / white fringe into hair, lashes and lace."""
    from PIL import ImageFilter
    a = np.asarray(img, np.float32)
    alpha = a[..., 3:4] / 255.0
    hole = alpha[..., 0] < 0.02
    if not hole.any() or hole.all():
        return img
    rgb = a[..., :3]
    out = rgb.copy()
    todo = hole.copy()
    for rad in (2, 6, 18, 54):
        pm = Image.fromarray(np.clip(rgb * alpha, 0, 255).astype(np.uint8), "RGB").filter(ImageFilter.GaussianBlur(rad))
        am = Image.fromarray((alpha[..., 0] * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(rad))
        pm_ = np.asarray(pm, np.float32)
        am_ = np.asarray(am, np.float32)[..., None] / 255.0
        ok = todo & (am_[..., 0] > 0.01)
        out[ok] = (pm_[ok] / np.maximum(am_[ok], 1e-3)).clip(0, 255)
        todo &= ~ok
        if not todo.any():
            break
    res = np.concatenate([out, a[..., 3:4]], axis=-1).astype(np.uint8)
    return Image.fromarray(res, "RGBA")


def sharpen_img(img, amount):
    """edge-aware sharpening of the colour channels (alpha untouched); amount 0..1"""
    from PIL import ImageFilter
    if amount <= 0:
        return img
    has_a = img.mode == "RGBA"
    rgb = img.convert("RGB").filter(ImageFilter.UnsharpMask(radius=1.3, percent=int(40 + 180 * amount), threshold=3))
    if has_a:
        rgb = rgb.convert("RGBA")
        rgb.putalpha(img.getchannel("A"))
    return rgb


def material_skin_color(m, k, base_dir):
    """the colour a material looks like (its texture under its own UVs x its diffuse), sRGB 0..1 —
    used to turn a piece of clothing into skin of the same tone"""
    mt = m.materials[k]
    dif = mt.diffuse
    fac = [min(1.0, dif[c] * 0.6 + mt.ambient[c]) for c in range(3)]
    off = sum(x.index_count for x in m.materials[:k])
    vs = np.unique(np.asarray(m.indices[off:off + mt.index_count]))
    rel = m.textures[mt.texture] if 0 <= mt.texture < len(m.textures) else ""
    p = resolve_path(base_dir, rel) if rel else None
    if not p or len(vs) == 0:
        return fac
    try:
        from PIL import Image
        img = Image.open(p).convert("RGBA")
        w, h = img.size
        a = np.asarray(img)
        uv = np.asarray(m.uvs)[vs]
        xs = (np.clip(np.mod(uv[:, 0], 1.0), 0, 0.9999) * w).astype(int)
        ys = (np.clip(np.mod(uv[:, 1], 1.0), 0, 0.9999) * h).astype(int)
        px = a[ys, xs]
        px = px[px[:, 3] > 128] if (px[:, 3] > 128).any() else px
        med = np.median(px[:, :3], axis=0) / 255.0
        return [float(min(1.0, med[c] * fac[c])) for c in range(3)]
    except Exception:
        return fac


def load_texture(path, max_size=0, raw=None, quality="normal", sharpen=0.0):
    """Returns (bytes, mime, alpha_kind, resized, alpha map). alpha_kind: opaque|cutout|blend.
    quality: "normal" (JPEG for opaque maps) | "best" (lossless PNG everywhere)."""
    if raw is None:
        with open(path, "rb") as f:
            raw = f.read()
    img = Image.open(io.BytesIO(raw))
    img.load()
    fmt = img.format
    has_alpha = img.mode in ("RGBA", "LA", "PA", "RGBa") or (
        img.mode == "P" and "transparency" in img.info)
    kind = "opaque"
    amap = None
    if has_alpha:
        rgba = img.convert("RGBA")
        w, h = rgba.size
        small = rgba.resize((min(w, 256), min(h, 256)), Image.NEAREST)
        a = np.asarray(small)[..., 3]
        amap = a.copy()
        transparent = int((a < 250).sum())
        if transparent > 0:
            partial = int(((a > 5) & (a < 250)).sum())
            kind = "blend" if partial > transparent * 0.5 else "cutout"
        else:
            has_alpha = False
            amap = None
    resized = False
    changed = False
    if has_alpha and kind != "opaque":
        img = bleed_alpha(img.convert("RGBA"))
        changed = True
    if max_size and max(img.size) > max_size:
        from PIL import ImageFilter
        k = max_size / max(img.size)
        img = img.convert("RGBA" if has_alpha else "RGB").resize(
            (max(1, round(img.size[0] * k)), max(1, round(img.size[1] * k))), Image.LANCZOS)
        if k < 0.75 and not sharpen:  # give back the crispness lost by shrinking (line art, eyes, text)
            img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=45, threshold=2))
        resized = True
    if sharpen and sharpen > 0:
        img = sharpen_img(img.convert("RGBA" if has_alpha else "RGB"), float(sharpen))
        changed = True
    if not resized and not changed and (fmt == "JPEG" or (fmt == "PNG" and (has_alpha or len(raw) < 512 * 1024))):
        return raw, "image/png" if fmt == "PNG" else "image/jpeg", kind, False, amap
    out = io.BytesIO()
    if not has_alpha and quality != "best":  # opaque colour map: JPEG is 3-5x smaller than PNG
        img.convert("RGB").save(out, "JPEG", quality=92, subsampling=0)
        return out.getvalue(), "image/jpeg", kind, resized, amap
    if has_alpha:
        img.convert("RGBA").save(out, "PNG", optimize=False)
    else:
        img.convert("RGB").save(out, "PNG", optimize=False)
    return out.getvalue(), "image/png", kind, resized, amap


def toon_shade(path):
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        r, g, b = img.getpixel((w // 2, h - 1))
        return [max(0.35, c / 255.0) for c in (r, g, b)]
    except Exception:
        return None


# --------------------------------------------------------------------------
# glTF binary builder
# --------------------------------------------------------------------------
class GLTFBuilder:
    def __init__(self):
        self.buf = bytearray()
        self.views = []
        self.accessors = []

    def view(self, data, target=None):
        while len(self.buf) % 4:
            self.buf.append(0)
        v = {"buffer": 0, "byteOffset": len(self.buf), "byteLength": len(data)}
        if target:
            v["target"] = target
        self.buf += data
        self.views.append(v)
        return len(self.views) - 1

    def accessor(self, arr, ctype, atype, target=None, minmax=False, normalized=False):
        arr = np.ascontiguousarray(arr)
        vi = self.view(arr.tobytes(), target)
        acc = {"bufferView": vi, "componentType": ctype, "count": int(arr.shape[0]), "type": atype}
        if normalized:
            acc["normalized"] = True
        if minmax:
            a2 = arr.reshape(arr.shape[0], -1)
            acc["min"] = [float(x) for x in a2.min(axis=0)]
            acc["max"] = [float(x) for x in a2.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def sparse_vec3(self, count, idx, vals):
        idx = np.ascontiguousarray(idx, np.uint32)
        vals = np.ascontiguousarray(vals, np.float32)
        iv = self.view(idx.tobytes())
        vv = self.view(vals.tobytes())
        mn, mx = vals.min(axis=0), vals.max(axis=0)
        if len(idx) < count:  # the non-listed elements are zero
            mn, mx = np.minimum(mn, 0), np.maximum(mx, 0)
        acc = {"componentType": 5126, "count": int(count), "type": "VEC3",
               "min": [float(x) for x in mn], "max": [float(x) for x in mx],
               "sparse": {"count": int(len(idx)),
                          "indices": {"bufferView": iv, "componentType": 5125},
                          "values": {"bufferView": vv}}}
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def glb_bytes(self, gltf):
        out = io.BytesIO()
        self._write(gltf, out)
        return out.getvalue()

    def write_glb(self, gltf, path):
        with open(path, "wb") as f:
            self._write(gltf, f)

    def _write(self, gltf, f):
        while len(self.buf) % 4:
            self.buf.append(0)
        gltf["buffers"] = [{"byteLength": len(self.buf)}]
        gltf["bufferViews"] = self.views
        gltf["accessors"] = self.accessors
        js = json.dumps(_finite(gltf), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        while len(js) % 4:
            js += b" "
        total = 12 + 8 + len(js) + 8 + len(self.buf)
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON"))
        f.write(js)
        f.write(struct.pack("<I4s", len(self.buf), b"BIN\x00"))
        f.write(self.buf)


# --------------------------------------------------------------------------
# spring bones
# --------------------------------------------------------------------------
PHYSICS_PRESETS = {
    # stiffness, gravityPower, dragForce
    "anime": (1.0, 0.25, 0.45),
    "realistic": (0.6, 0.6, 0.6),
    "stiff": (2.5, 0.1, 0.8),
    "soft": (1.2, 0.2, 0.62),
    "dance": (0.8, 0.35, 0.35),
    "heavy": (1.1, 0.7, 0.58),
    "mmd": None,  # derived from rigid body damping
}
PHYSICS_NAMES = {"anime": "Аниме", "realistic": "Реалистичный", "stiff": "Жёсткий",
                 "soft": "Плавный (стрим, VTuber)", "dance": "Динамичный (танцы)",
                 "heavy": "Тяжёлый (длинные волосы, плащи)", "mmd": "Как в MMD"}

# ready-made settings for one kind of chain: (id, name, stiffness, gravity, drag)
PART_PRESETS = {
    "hair": [("light", "Лёгкие пряди", 1.0, 0.2, 0.45), ("long", "Длинные тяжёлые волосы", 1.6, 0.6, 0.6),
             ("ponytail", "Хвостик / коса", 1.4, 0.5, 0.5), ("short", "Короткие упругие", 2.2, 0.1, 0.55),
             ("bangs", "Чёлка (почти неподвижна)", 3.0, 0.05, 0.7)],
    "tail": [("fluffy", "Пушистый лёгкий", 1.0, 0.3, 0.4), ("springy", "Упругий с весом", 1.8, 0.5, 0.3),
             ("heavy", "Тяжёлый (дракон, ящерица)", 2.5, 0.8, 0.45), ("stiff", "Жёсткий", 3.5, 0.2, 0.6)],
    "skirt": [("light", "Лёгкая ткань", 1.0, 0.3, 0.45), ("dense", "Плотная ткань", 1.6, 0.5, 0.6),
              ("long", "Длинное платье / плащ", 1.3, 0.7, 0.65), ("pleated", "Плиссе (держит форму)", 2.4, 0.3, 0.6)],
    "breast": [("soft", "Мягкая", 1.6, 0.1, 0.55), ("elastic", "Упругая", 2.8, 0.05, 0.6),
               ("still", "Почти без движения", 4.0, 0.0, 0.8)],
    "ear": [("soft", "Мягкие", 1.6, 0.15, 0.5), ("stiff", "Жёсткие", 3.2, 0.05, 0.7)],
    "cloth": [("light", "Лёгкая ткань (рукав, воротник)", 1.2, 0.3, 0.5), ("coat", "Полы куртки / плаща", 1.6, 0.6, 0.6),
              ("cape", "Длинный плащ", 1.4, 0.8, 0.65), ("firm", "Плотная (почти держит форму)", 3.0, 0.2, 0.7)],
    "ribbon": [("light", "Лёгкая лента", 0.8, 0.3, 0.4), ("bow", "Бант (пружинит)", 2.0, 0.15, 0.55),
               ("lace", "Шнурок / кисточка", 1.0, 0.6, 0.45)],
    "accessory": [("pendant", "Подвеска / брелок", 1.2, 0.8, 0.5), ("strap", "Ремешок / цепочка", 1.0, 0.9, 0.55),
                  ("ornament", "Украшение (почти неподвижно)", 3.5, 0.1, 0.7)],
    "flesh": [("subtle", "Еле заметно", 4.0, 0.0, 0.8), ("soft", "Мягко", 2.6, 0.02, 0.7),
              ("off", "Почти выключено", 6.0, 0.0, 0.9)],
    "other": [("ribbon", "Лента", 0.8, 0.3, 0.4), ("pendant", "Подвеска / брелок", 1.2, 0.8, 0.5),
              ("cloth", "Ткань (пояс, плащ)", 1.4, 0.5, 0.6)],
}

TAIL_WORDS = ("尻尾", "しっぽ", "シッポ", "尾", "tail")
EAR_WORDS = ("耳", "ear")
SKIRT_WORDS = ("スカート", "skirt", "裙")
BREAST_WORDS = ("胸", "おっぱい", "乳", "breast", "bust")
HAIR_WORDS = ("髪", "hair", "发", "髮", "前髪", "後髪", "もみあげ", "アホ毛", "ツインテ")


def _tokens(name):
    import re
    s = nk(name)
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+|[^\W\d_A-Za-z]+", s)
    return [p.lower() for p in parts]


# whole-token (latin / pinyin) and substring (CJK) vocabularies
KIND_TOKENS = [
    ("hair", {"hair", "bangs", "bang", "fringe", "strand", "ponytail", "twintail", "braid", "ahoge",
              "liuhai", "binfa", "bingfa", "jianfa", "toufa", "houfa", "qianfa", "mawei", "bianzi", "fa"}),
    ("tail", {"tail", "weiba", "shippo"}),
    ("skirt", {"skirt", "qun", "qunzi", "qunbai"}),
    ("breast", {"breast", "bust", "boob", "xiong", "oppai"}),
    ("ear", {"ear", "ears", "erduo"}),
    ("other", {"coat", "cape", "cloak", "jacket", "trousers", "pants", "piyi", "cuff", "sleeve", "rope",
               "belt", "bag", "beltbag", "gadget", "acc", "earring", "ribbon", "hat", "hood", "scarf", "tie", "cloth"}),
]
KIND_CJK = [("hair", ("髪", "发", "髮", "もみあげ", "アホ毛", "ツインテ", "おさげ", "辫", "马尾")),
            ("tail", ("尻尾", "しっぽ", "シッポ", "尾巴")),
            ("skirt", ("スカート", "裙")),
            ("breast", ("胸", "おっぱい", "乳")),
            ("ear", ("耳",))]


AUTO_EXCLUDE = {"weapon", "prop", "socket", "attach", "attachment", "point", "fx", "eff", "effect", "ik",
                "twist", "nub", "helper", "dummy", "camera", "light", "root", "armature", "offset", "roll"}


def _kind_strict(name):
    """Kind only from explicit name words (no region fallback) — for auto physics."""
    k, _w = vocab.chain_kind(name, strict=True)
    if k:
        return vocab.LEGACY_KIND.get(k, "other")
    if vocab.is_excluded_bone(name):
        return None
    return _kind_strict_old(name)


def _kind_strict_old(name):
    toks = _tokens(name)
    if any(t in AUTO_EXCLUDE for t in toks):
        return None
    for kind, words in KIND_TOKENS:
        if any(t in words for t in toks):
            return kind
    n = nk(name)
    for kind, words in KIND_CJK:
        if any(w in n for w in words):
            return "other" if kind == "ear" and ("耳飾" in n or "耳环" in n) else kind
    return None


AUTO_HIT = {"hair": 0.02, "skirt": 0.04, "breast": 0.03, "tail": 0.03, "ear": 0.02, "other": 0.02,
            "cloth": 0.03, "ribbon": 0.015, "accessory": 0.015, "flesh": 0.02}


def _kind(name, region):
    k, _w = vocab.chain_kind(name, strict=False)
    if k and k != "other":
        return vocab.LEGACY_KIND.get(k, "other")
    return _kind_old(name, region)


def _kind_old(name, region):
    toks = _tokens(name)
    for kind, words in KIND_TOKENS:
        if any(t in words for t in toks):
            return kind
    n = nk(name)
    for kind, words in KIND_CJK:
        if any(w in n for w in words):
            if kind == "ear" and ("耳飾" in n or "耳环" in n):
                return "other"
            return kind
    if region in ("head", "neck", "leftEye", "rightEye"):
        return "hair"
    if region in ("hips", "leftUpperLeg", "rightUpperLeg", "leftLowerLeg", "rightLowerLeg"):
        return "skirt"
    return "other"


KIND_NAMES = {"hair": "волосы", "tail": "хвост", "skirt": "юбка/подол", "cloth": "одежда (рукава, плащ, воротник)",
              "ribbon": "ленты и банты", "accessory": "украшения и подвески", "breast": "грудь",
              "flesh": "мягкость тела", "ear": "уши", "other": "прочее"}
# multipliers: stiffness, gravity, drag(add)
KIND_MOD = {"hair": (1.0, 1.0, 0.0), "skirt": (1.3, 1.2, 0.1), "breast": (2.0, 0.2, 0.25),
            "tail": (1.8, 1.6, -0.1), "ear": (2.0, 0.3, 0.2), "other": (1.0, 1.0, 0.0),
            "cloth": (1.4, 1.1, 0.1), "ribbon": (0.9, 1.1, -0.05), "accessory": (1.2, 2.0, 0.05),
            "flesh": (3.5, 0.0, 0.3)}


def clean_spring_tree(m, parent, children, cur, jo, we, nb0, log):
    """Remove unweighted side branches (e.g. MMD '錘' helper bones) from physics chains,
    put the weighted continuation first, and give zero-length physics bones a tiny length."""
    nb = len(parent)
    wsum = np.zeros(nb)
    for k in range(jo.shape[1]):
        np.add.at(wsum, jo[:, k], we[:, k])
    dyn = {rb.bone for rb in m.rigid_bodies if rb.mode in (1, 2) and 0 <= rb.bone < nb0}
    if not dyn:
        return set()

    def subtree_w(r):
        tot, st = 0.0, [r]
        while st:
            x = st.pop()
            tot += wsum[x]
            st.extend(children[x])
        return tot
    moved = 0
    excluded = set()
    for b in sorted(dyn):
        kids = list(children[b])
        if len(kids) < 2:
            continue
        wk = [(c, subtree_w(c)) for c in kids]
        if not any(w > 0 for _, w in wk):
            continue
        anchor = parent[b]
        while anchor != -1 and anchor in dyn:
            anchor = parent[anchor]
        for c, w in wk:
            if w <= 0 and anchor != -1:
                st = [c]
                while st:
                    x = st.pop()
                    excluded.add(x)
                    st.extend(children[x])
                children[b].remove(c)
                parent[c] = anchor
                children[anchor].append(c)
                moved += 1
        children[b].sort(key=lambda c: -subtree_w(c))
    nudged = 0
    for b in sorted(dyn):
        p = parent[b]
        if p < 0:
            continue
        d = cur[b] - cur[p]
        if np.linalg.norm(d) < 0.01:  # < ~1 mm in MMD units
            gp = parent[p]
            dirv = cur[p] - cur[gp] if gp >= 0 else np.array([0.0, -1.0, 0.0])
            nrm = np.linalg.norm(dirv)
            dirv = dirv / nrm if nrm > 1e-6 else np.array([0.0, -1.0, 0.0])
            cur[b] = cur[p] + dirv * 0.03
            nudged += 1
    if moved or nudged:
        log("  физика: убрано боковых вспомогательных костей из цепочек: %d, исправлено костей нулевой длины: %d"
            % (moved, nudged))
    return excluded


def build_springs(m, h, parent, children, nb0, M, BP, conv, scale, physics, log,
                  chain_ov=None, names=None, excluded=None, rest=None):
    chain_ov = chain_ov or {}
    excluded = excluded or set()
    rest = rest or {}
    nb = len(parent)
    human_set = set(h.values())
    inv_h = {v: k for k, v in h.items()}
    protected = set()
    for hb in human_set:
        j = hb
        while j != -1:
            protected.add(j)
            j = parent[j]
    rb_of_bone = {}
    for ri, rb in enumerate(m.rigid_bodies):
        if 0 <= rb.bone < nb0:
            rb_of_bone.setdefault(rb.bone, []).append(ri)
    dyn = set()
    for rb in m.rigid_bodies:
        if rb.mode in (1, 2) and 0 <= rb.bone < nb0 and rb.bone not in protected \
                and rb.bone not in excluded:
            dyn.add(rb.bone)

    def subtree(r):
        out, st = [], [r]
        while st:
            x = st.pop()
            out.append(x)
            st.extend(children[x])
        return out

    auto = not any(rb.mode in (1, 2) for rb in m.rigid_bodies)
    if auto and rest:
        jo_, we_ = rest["jo"], rest["we"]
        wsum = np.zeros(nb)
        for k in range(jo_.shape[1]):
            np.add.at(wsum, jo_[:, k], we_[:, k])
        sub_w = {}

        def subw(r):
            if r not in sub_w:
                sub_w[r] = wsum[r] + sum(subw(c) for c in children[r])
            return sub_w[r]
        blocked = set()
        for b in range(nb0):
            if vocab.is_excluded_subtree_root(m.bones[b].name):
                blocked.update(subtree(b))
        for b in range(nb0):
            if b in protected or b in excluded or b in blocked:
                continue
            k_ = _kind_strict(m.bones[b].name)
            if k_ is None or subw(b) <= 0:
                continue
            dyn.add(b)
        # single-bone "chains" (facial ear/cheek bones etc.) don't swing — except breasts
        for b in list(dyn):
            if parent[b] in dyn:
                continue
            chain_ = subtree(b)
            real = [x for x in chain_ if _tokens(m.bones[x].name)[-1:] not in (["end"], ["nub"], ["tip"])]
            ptoks = _tokens(m.bones[parent[b]].name) if parent[b] >= 0 else []
            facial = any(t in ("facial", "face", "facialdef") for t in ptoks)
            if (len(real) < 2 or facial) and _kind_strict(m.bones[b].name) != "breast":
                for x in chain_:
                    dyn.discard(x)
        if dyn:
            log("  физики в модели нет — цепочки найдены по именам костей: %d костей" % len(dyn))

    # ---- colliders: one VRM collider group per static rigid body ----
    colliders = []  # (group, bone, [(center, radius)])
    for rb in m.rigid_bodies:
        if rb.mode != 0 or not (0 <= rb.bone < nb0) or rb.bone in dyn:
            continue
        Mbone = M[rb.bone]
        c = Mbone[:3, :3] @ rb.pos + Mbone[:3, 3]
        R = Mbone[:3, :3] @ euler_mmd(*rb.rot)
        if rb.shape == 0:
            r, pts = rb.size[0], [c]
        elif rb.shape == 2:
            r = rb.size[0]
            ax = R @ np.array([0, 1.0, 0])
            hh = rb.size[1] / 2
            n = max(1, int(math.ceil(hh / max(r, 1e-3))))
            pts = [c + ax * hh * (k / n) for k in range(-n, n + 1)]
        else:
            r, pts = float(np.sort(rb.size)[1]), [c]
        colliders.append((rb.group, rb.bone, [(conv(p), float(r * scale)) for p in pts]))

    # ---- auto colliders from the body shape (when the model has no static rigid bodies) ----
    covered = {rb.bone for rb in m.rigid_bodies if rb.mode == 0 and 0 <= rb.bone < nb0}
    if rest:
        P0, pos0, cur0 = rest["P"], rest["pos"], rest["cur"]
        jo_, we_ = rest["jo"], rest["we"]
        main = jo_[np.arange(len(jo_)), np.argmax(we_, axis=1)]
        mainw = we_.max(axis=1)
        nxt = {"hips": "spine", "spine": "chest", "chest": "neck", "upperChest": "neck", "neck": "head",
               "leftUpperArm": "leftLowerArm", "leftLowerArm": "leftHand",
               "rightUpperArm": "rightLowerArm", "rightLowerArm": "rightHand",
               "leftUpperLeg": "leftLowerLeg", "leftLowerLeg": "leftFoot",
               "rightUpperLeg": "rightLowerLeg", "rightLowerLeg": "rightFoot", "head": None}
        n_auto = 0
        for vb, nb_ in nxt.items():
            if vb not in h:
                continue
            b = h[vb]
            if b in covered:
                continue  # the model already has its own collider there
            sel = (main == b) & (mainw > 0.5)
            if sel.sum() < 8:
                # fall back to a radius from the bone length
                if nb_ and nb_ in h:
                    L_ = float(np.linalg.norm(pos0[h[nb_]] - pos0[b]))
                    if L_ <= 0:
                        continue
                    r = float(np.clip(0.18 * L_ * scale, 0.02, 0.12)) / scale
                    ca, cc = cur0[b], cur0[h[nb_]]
                    colliders.append((-1, b, [(conv(ca + (cc - ca) * f), float(r * scale)) for f in (0.25, 0.75)]))
                    n_auto += 1
                continue
            V = P0[sel]
            a = pos0[b]
            if nb_ and nb_ in h:
                c = pos0[h[nb_]]
                ca, cc = cur0[b], cur0[h[nb_]]
            else:  # head: up to ~80% of the head height
                top = float(np.percentile(V[:, 1], 95))
                c = a + np.array([0, max(top - a[1], 1e-3) * 0.8, 0])
                ca, cc = cur0[b], cur0[b] + (c - a)
            seg = c - a
            L2 = float(seg @ seg) or 1e-9
            t = np.clip(((V - a) @ seg) / L2, 0, 1)
            d = np.linalg.norm(V - (a + t[:, None] * seg), axis=1)
            r = float(np.percentile(d, 60 if vb in ("hips", "spine", "chest", "upperChest") or "Leg" in vb else 45))
            r = float(np.clip(r * scale, 0.02, 0.15)) / scale
            pts = [ca + (cc - ca) * f for f in ((0.5,) if vb in ("hips", "neck") else (0.2, 0.5, 0.8))]
            if vb in ("spine", "chest", "upperChest") and len(V) >= 30:
                # torso: bones run near the back, the chest / breasts stick out in front — fit the
                # spheres to the body's real depth and width so hair lying on it cannot sink in
                zf, zb = np.percentile(V[:, 2], [3, 97])
                xl, xr = np.percentile(V[:, 0], [8, 92])
                depth_, width_ = float(zb - zf), float(xr - xl)
                rr = 0.5 * min(depth_, width_)
                if rr > 1e-4:
                    zc = float((zf + zb) / 2) - a[2]
                    xs = [0.0] if width_ < 1.5 * depth_ else [-(width_ / 2 - rr), 0.0, width_ / 2 - rr]
                    pts = [ca + (cc - ca) * f + np.array([x_, 0.0, zc]) for f in (0.3, 0.75) for x_ in xs]
                    r = float(np.clip(rr * 0.95 * scale, 0.02, 0.18)) / scale
            colliders.append((-1, b, [(conv(p), float(r * scale)) for p in pts]))
            n_auto += 1
        if n_auto:
            log("  коллайдеры тела построены по форме модели: %d" % n_auto)

    # ---- chains ----
    roots = sorted(b for b in dyn if parent[b] not in dyn)
    # tails without MMD physics: detect by name
    auto_tail = set()
    for b in range(nb0):
        if b in dyn or b in protected or b in excluded:
            continue
        n = nk(m.bones[b].name).lower()
        if not any(w in n for w in ("尻尾", "しっぽ", "シッポ", "tail")):
            continue
        p = parent[b]
        if p >= 0 and any(w in nk(m.bones[p].name).lower() for w in ("尻尾", "しっぽ", "シッポ", "tail")):
            continue
        if len(subtree(b)) >= 2 and not any(x in dyn for x in subtree(b)):
            roots.append(b)
            auto_tail.add(b)
    if auto_tail:
        log("  хвост без физики найден по имени: " + ", ".join(m.bones[b].name for b in auto_tail))
    # a branch with a different name growing out of a chain (a pendant / strap hanging from a tail,
    # a ribbon on a braid) is its own part: its own kind, settings and colliders
    import re as _re2

    def _stem(i):
        n_ = nk(m.bones[i].name).lower() if i < nb0 else ""
        n_ = _re2.sub(r"^(bn|bone|b|jnt|j)[_ .]", "", n_)
        n_ = _re2.sub(r"(^|[_ .])(l|r|m|c|left|right|lf|rt)(?=[_ .]|$)", r"\1", n_)
        n_ = _re2.sub(r"[左右]|jnt|joint", "", n_)
        n_ = _re2.sub(r"[\d_ .錘锤先\-]+", "", n_)
        return _re2.sub(r"(?<=[a-z]{3})[a-z]$", "", n_)  # variant letter: hairA / hairB

    branch_of = {}
    for b in list(roots):
        for x in subtree(b):
            ks = [y for y in children[x] if y in dyn and y not in branch_of]
            if len(ks) < 2:
                continue
            sx = _stem(x)
            if not sx or not any(_stem(y) == sx for y in ks):
                continue  # a fan of equal strands (bow loops, hair tufts) stays one part
            kx = _kind_strict(m.bones[x].name) if x < nb0 else None
            for y in ks:
                sy = _stem(y)
                ky = _kind_strict(m.bones[y].name) if y < nb0 else None
                # only a part of another kind: a pendant on a tail, a charm on a skirt, a ribbon on a braid
                if sy and sy != sx and ky and ky != kx and len(subtree(y)) >= 3:
                    branch_of[y] = b
    if branch_of:
        roots = roots + sorted(branch_of)
        log("  отдельные части на цепочках (подвески, ленты): " + ", ".join(m.bones[y].name for y in sorted(branch_of)))

    def chain_nodes(r):
        out, st = [], [r]
        while st:
            x = st.pop()
            out.append(x)
            st.extend(y for y in children[x] if y not in branch_of)
        return out

    def mesh_radius(nodes):
        """typical distance from the chain's bones to the skin they carry (thick tails, fat braids)"""
        if not rest:
            return 0.0
        jo_, we_ = rest["jo"], rest["we"]
        main_ = jo_[np.arange(len(jo_)), np.argmax(we_, axis=1)]
        sel = np.isin(main_, list(nodes)) & (we_.max(axis=1) > 0.5)
        if sel.sum() < 12:
            return 0.0
        V = rest["P"][sel]
        Bp = np.array([rest["pos"][x] for x in nodes if x < len(rest["pos"])])
        if len(Bp) == 0:
            return 0.0
        d = np.min(np.linalg.norm(V[:, None, :] - Bp[None, :, :], axis=2), axis=1)
        return float(np.percentile(d, 50)) * scale

    chains_rec = []
    radii = []
    hip_roots = []
    slim_ids = {}
    n_base = len(colliders)
    for b in roots:
        j = parent[b]
        while j != -1 and j not in inv_h:
            j = parent[j]
        if j != -1 and inv_h[j] in ("hips", "spine", "leftUpperLeg", "rightUpperLeg"):
            hip_roots.append(b)

    def depth_of(r):
        best, st = 0, [(r, 1)]
        while st:
            x, d = st.pop()
            best = max(best, d)
            st.extend((y, d + 1) for y in children[x])
        return best

    def tail_like(b, region):
        """one long chain hanging from behind the pelvis, near the middle (not a ring of skirt chains)"""
        if "hips" not in h or depth_of(b) < 4:
            return False
        hp = BP[h["hips"]]
        p0 = BP[b]
        sub = subtree(b)
        tip = BP[sub[-1]] if sub else p0
        hgt = max(1e-6, float(BP[h["head"], 1] - BP[h["hips"], 1])) if "head" in h else 0.5
        # facing in the converted space: VRM0 faces -Z, VRM1 faces +Z -> "behind" is the other side
        back = 1.0 if conv(np.array([0.0, 0.0, -1.0]))[2] > 0 else -1.0
        behind = (p0[2] - hp[2]) * back < 0 and (tip[2] - hp[2]) * back < 0
        central = abs(p0[0] - hp[0]) < 0.12 * hgt
        ring = sum(1 for r in hip_roots if r != b and abs(BP[r][1] - p0[1]) < 0.1 * hgt)
        return behind and central and ring < 3

    for b in roots:
        j = parent[b]
        region = "other"
        while j != -1:
            if j in inv_h:
                region = inv_h[j]
                break
            j = parent[j]
        kind = "tail" if b in auto_tail else _kind(m.bones[b].name if b < nb0 else "", region)
        bname = m.bones[b].name if b < nb0 else ""
        if b not in auto_tail and vocab.excluded_word(bname):
            # a physics chain named like a helper ("xiuziTwistA" = sleeve): the part word decides
            k_ex, _w = vocab.chain_kind(bname, strict=True, ignore_excluded=True)
            if k_ex:
                kind = vocab.LEGACY_KIND.get(k_ex, k_ex)
        if b in branch_of and kind in ("tail", "hair", "other"):
            kind = "accessory"  # something hanging from a tail / braid, not more tail
        if kind == "skirt" and _kind_strict(bname) == "flesh":
            kind = "flesh"  # thigh / calf / butt jiggle bones: soft body, not a skirt
        if region in ("head", "neck", "leftEye", "rightEye", "jaw") and kind == "tail":
            kind = "hair"  # ポニーテール / 马尾 / 发尾: a "tail" growing from the head is hair
        elif (region in ("hips", "spine", "leftUpperLeg", "rightUpperLeg") and kind in ("skirt", "other", "hair")
              and _kind_strict(bname) in (None, "other") and tail_like(b, region)):
            kind = "tail"
        chain = chain_nodes(b)
        rbs = [m.rigid_bodies[ri] for x in chain if x < nb0 for ri in rb_of_bone.get(x, [])
               if m.rigid_bodies[ri].mode in (1, 2)]
        mask = rbs[0].mask if rbs else 0xFFFF
        hit = float(np.median([r.size[0] for r in rbs])) * scale * 0.5 if rbs else AUTO_HIT.get(kind, 0.02)
        if kind == "tail":
            # MMD tails often carry thin rigid bodies; a thick furry tail then sinks into the legs
            hit = max(hit, 0.75 * mesh_radius(chain))
        hit = float(np.clip(hit, 0.005, 0.05))
        radii.append(hit)
        linear = all(sum(1 for y in children[x] if y in chain) <= 1 for x in chain)

        def pick_colliders(use_mask):
          allowed = []
          excluded_n = 0
          skip = 0
          for ci, (g, cb, spheres) in enumerate(colliders[:n_base]):
              if use_mask and g >= 0 and not (mask >> g) & 1:  # MMD non-collision group (auto colliders: g = -1)
                  continue
              ins = [any(np.linalg.norm(BP[x] - c) < r + hit for c, r in spheres) for x in chain]
              if not any(ins):
                  allowed.append(ci)
                  continue
              k = next((i for i, v in enumerate(ins) if not v), len(ins))
              # only the first 1-3 joints sit inside (e.g. hair roots in the head): keep the collider,
              # and start swinging from the first joint outside it
              if linear and 1 <= k <= 3 and not any(ins[k:]) and len(chain) >= k + 3:
                  allowed.append(ci)
                  skip = max(skip, k)
                  continue
              # the strand lies on the body (long hair on the back / chest, a coat on the hips):
              # dropping this collider lets it pass straight through — use a slimmer copy instead
              # that the resting joints just touch
              js = chain[k:] if (linear and 1 <= k <= 3) else chain
              fac = 1.0
              for x in js:
                  for c, r in spheres:
                      d_ = float(np.linalg.norm(BP[x] - c))
                      if d_ < r + hit:
                          fac = min(fac, max(0.0, (d_ - hit) * 0.97) / r)
              if fac >= 0.45:
                  key = (ci, round(fac, 2))
                  if key not in slim_ids:
                      colliders.append((g, cb, [(c, r * key[1]) for c, r in spheres]))
                      slim_ids[key] = len(colliders) - 1
                  allowed.append(slim_ids[key])
                  if js is not chain:
                      skip = max(skip, k)
              else:
                  excluded_n += 1
          return allowed, excluded_n, skip

        allowed, excluded_n, skip = pick_colliders(True) if kind != "flesh" else ([], 0, 0)
        if not allowed and (kind in ("tail", "hair") or b in branch_of
                            or _kind_strict(bname) in ("hair", "tail", "skirt")):
            # MMD often switches body collisions off for tails / hair to avoid jitter in Bullet;
            # in VRM that just lets them pass through the legs — collide with the body anyway
            allowed, excluded_n, skip = pick_colliders(False)
        elif kind == "tail":
            # MMD switches tail-vs-leg collisions off (Bullet jitter); a VRM tail must not pass the legs
            al2, ex2, sk2 = pick_colliders(False)
            allowed = sorted(set(allowed) | set(al2))
            skip = max(skip, sk2)
        chains_rec.append({"root": b, "kind": kind, "auto_kind": kind, "allowed": allowed, "excluded_n": excluded_n,
                           "branch_of": branch_of.get(b), "has_branch": b in branch_of.values(),
                           "rbs": rbs, "hit": hit, "nodes": chain,
                           "spring_root": chain[skip] if skip < len(chain) else b})

    preset = PHYSICS_PRESETS.get(physics, PHYSICS_PRESETS["anime"])
    for c in chains_rec:
        rid = m.bones[c["root"]].name
        o = chain_ov.get(rid) or {}
        if o.get("kind") in KIND_MOD:
            c["kind"] = o["kind"]
        if preset is None:
            rbs = c["rbs"]
            ld = float(np.mean([r.lin_damp for r in rbs])) if rbs else 0.5
            ad = float(np.mean([r.ang_damp for r in rbs])) if rbs else 0.5
            st, gr, dr = 0.5 + ad * 2.0, 0.4, float(np.clip(ld, 0.2, 0.9))
        else:
            st, gr, dr = preset
        ms, mg, md = KIND_MOD[c["kind"]]
        st_, gr_, dr_ = st * ms, gr * mg, max(0.05, min(1.0, dr + md))
        n_ = (len(c["nodes"]) - c["nodes"].index(c["spring_root"]) if c.get("has_branch") and c["spring_root"] in c["nodes"]
              else depth_of(c["spring_root"]))
        c["length"] = n_
        if n_ > 5 and c["kind"] in ("hair", "tail", "other", "skirt", "cloth", "ribbon", "accessory"):
            # long chains: every joint adds swing, so the tips whip around like loose springs —
            # stiffen, damp and lighten them in proportion to the length
            ex = n_ - 5
            if c["kind"] == "tail":  # a tail keeps its weight and spring, it only must not whip
                st_ *= min(1.5, 1.0 + 0.06 * ex)
                dr_ = min(0.75, dr_ + min(0.12, 0.02 * ex))
            else:
                st_ *= min(1.8, 1.0 + 0.08 * ex)
                dr_ = min(0.8, dr_ + min(0.25, 0.025 * ex))
                gr_ *= max(0.6, 1.0 - 0.04 * ex)
        c["stiffness"] = round(float(o.get("stiffness", st_)), 3)
        c["gravity"] = round(float(o.get("gravity", gr_)), 3)
        c["drag"] = round(float(o.get("drag", dr_)), 3)
        c["hitRadius"] = round(float(o.get("hitRadius", c["hit"])), 4)
        c["enabled"] = bool(o.get("enabled", True))
        vb_ = rest.get("vis_bones")
        if vb_ is not None and "enabled" not in o and not any(x in vb_[0] for x in c["nodes"]) \
                and any(x in vb_[1] for x in c["nodes"]):
            c["enabled"] = False  # nothing visible hangs on this chain: no physics for a hidden part
            c["hidden_part"] = True
        c["useColliders"] = bool(o.get("colliders", True))
        c["id"] = rid
    active = [c for c in chains_rec if c["enabled"]]

    used_colliders = sorted({ci for c in active if c["useColliders"] for ci in c["allowed"]})
    remap = {}
    collider_groups = []
    for ci in used_colliders:
        g, cb, spheres = colliders[ci]
        remap[ci] = len(collider_groups)
        cl = []
        for cc, r in spheres:
            o = cc - BP[cb]
            cl.append({"offset": {"x": float(o[0]), "y": float(o[1]), "z": float(-o[2])},
                       "radius": r})
        collider_groups.append({"node": int(cb), "colliders": cl})

    # VRM 0.x simulates the whole subtree of a root bone: a branch (pendant on a tail) cannot have its own
    # group there — it swings with its host chain, which also collides with what the branch collides with.
    # VRM 1.0 keeps its own settings (see "_branches", used by to_vrm1).
    by_root = {c["root"]: c for c in active}
    extra_al = {}
    branches_v1 = {}
    for c in chains_rec:
        if c.get("branch_of") is not None and not c in active and c["branch_of"] in by_root:
            branches_v1[int(c["root"])] = {"disabled": True}
    for c in active:
        host = c.get("branch_of")
        if host is None:
            continue
        hc = by_root.get(host)
        if hc is not None and c["useColliders"]:
            # only colliders the host chain does not already sit in at rest (a tail root inside the hip
            # collider would be shoved out and the whole tail would stick up)
            hj = [x for x in hc["nodes"] if x not in c["nodes"]]
            for ci in c["allowed"]:
                if ci in hc["allowed"]:
                    continue
                if not any(np.linalg.norm(BP[x] - cc) < r + hc["hit"] for x in hj for cc, r in colliders[ci][2]):
                    extra_al.setdefault(host, set()).add(ci)
        branches_v1[int(c["spring_root"])] = {
            "comment": KIND_NAMES[c["kind"]], "stiffiness": c["stiffness"], "gravityPower": c["gravity"],
            "dragForce": c["drag"], "hitRadius": c["hitRadius"],
            "colliderGroups": [remap[x] for x in (c["allowed"] if c["useColliders"] else []) if x in remap]}
    groups = {}
    for c in active:
        if c.get("branch_of") is not None and c["branch_of"] in by_root:
            continue
        al = set(c["allowed"]) if c["useColliders"] else set()
        if c["useColliders"]:
            al |= extra_al.get(c["root"], set())
        al = tuple(sorted(x for x in al if x in remap))
        key = (c["kind"], al, c["stiffness"], c["gravity"], c["drag"], c["hitRadius"])
        groups.setdefault(key, []).append(c["spring_root"])
    bone_groups = []
    counts = {}
    for (kind, al, st, gr, dr, hr), rts in groups.items():
        bone_groups.append({
            "comment": KIND_NAMES[kind], "stiffiness": st, "gravityPower": gr,
            "gravityDir": {"x": 0, "y": -1, "z": 0}, "dragForce": dr, "center": -1,
            "hitRadius": hr, "bones": [int(x) for x in rts],
            "colliderGroups": [remap[x] for x in al]})
        counts[kind] = counts.get(kind, 0) + len(rts)
    for c in active:
        if c.get("branch_of") is not None and c["branch_of"] in by_root:
            counts[c["kind"]] = counts.get(c["kind"], 0) + 1
    log("  физика (%s): %s; коллайдеров: %d" % (
        PHYSICS_NAMES.get(physics, physics),
        ", ".join("%s %d" % (KIND_NAMES[k], v) for k, v in counts.items()),
        sum(len(c["colliders"]) for c in collider_groups)))
    chain_info = [{"id": c["id"], "kind": c["kind"], "auto_kind": c["auto_kind"], "length": c.get("length", 0),
                   "stiffness": c["stiffness"], "gravity": c["gravity"], "drag": c["drag"],
                   "hitRadius": c["hitRadius"], "enabled": c["enabled"],
                   "colliders": c["useColliders"], "n_colliders": len(c["allowed"]),
                   "n_colliders_off": c.get("excluded_n", 0),
                   "nodes": [int(x) for x in c["nodes"]], "root": int(c["root"]),
                   "branch_of": (m.bones[c["branch_of"]].name if c.get("branch_of") is not None
                                 and c["branch_of"] < nb0 else None)}
                  for c in chains_rec]
    collider_info = [{"node": int(cb), "spheres": [[float(x) for x in cc] + [r] for cc, r in sp]}
                     for (g, cb, sp) in colliders]
    # data for the physics auto-tuner (simulate.py): a linear path per chain + its colliders
    for c, ci in zip(chains_rec, chain_info):
        path_, x = [c["spring_root"]], c["spring_root"]
        while True:
            nx = [y for y in children[x] if y in c["nodes"]]
            if not nx:
                break
            x = nx[0]
            path_.append(x)
        ci["_sim"] = {"pts": [[float(v) for v in BP[x]] for x in path_],
                      "spheres": [[float(v) for v in cc] + [float(r)] for ai in c["allowed"]
                                  for cc, r in colliders[ai][2]],
                      "hit": float(c["hitRadius"])}
    if rest is not None:
        rest["branches_v1"] = branches_v1  # read back by convert() for the VRM 1.0 export
    return bone_groups, collider_groups, chain_info, collider_info


# --------------------------------------------------------------------------
# main conversion
# --------------------------------------------------------------------------
class ConvertError(Exception):
    pass


def convert(src, dst, scale=0.08, tpose=True, spring=True, outline=False,
            custom_blendshapes=True, title=None, author=None, max_texture=2048,
            physics="anime", overrides=None, editor=False, ru_names=False, repair=False, face_from_bones=True, anime_face=True, vrm_version=0,
            target_mb=0, strip_hidden=False, tex_quality="normal", upscale_dir=None, bone_lang=None,
            tex_sharpen=0.0, log=print):
    """Convert PMX -> VRM 0.x.

    dst: output path, or None to return the bytes in result["vrm"].
    overrides: manual edits from the editor (humanoid / materials / chains / presets).
    editor: never raise on missing humanoid bones, return analysis instead.
    """
    ov = _clean_overrides(overrides)
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        scale = 0.08
    if not math.isfinite(scale) or scale <= 0:
        raise ConvertError("Масштаб должен быть больше нуля")
    try:
        max_texture = max(0, int(max_texture or 0))
    except (TypeError, ValueError):
        max_texture = 2048
    try:
        tex_sharpen = float(tex_sharpen or 0)
        tex_sharpen = min(1.0, max(0.0, tex_sharpen)) if math.isfinite(tex_sharpen) else 0.0
    except (TypeError, ValueError):
        tex_sharpen = 0.0
    log("Чтение %s" % os.path.basename(src))
    m = copy.copy(load_pmx_cached(src, log))
    m.morphs = list(m.morphs)  # synthesized morphs must not leak into the cache
    n_pieces = {}
    for k_, mt_ in enumerate(m.materials):  # how many separate pieces each material has (for the editor)
        if editor and mt_.index_count // 3 >= 2:
            try:
                n_pieces[mt_.name] = len(material_pieces(m, k_))
            except Exception:
                pass
    piece_info = apply_splits(m, ov.get("splits"), log)
    if ov.get("wd_keep"):
        try:
            keep_body_parts(m, set(ov["wd_keep"]), log)
        except Exception as e:  # never fatal
            log("  ! хвост: %s" % e)
    body_parts = [x for x in (ov.get("body_fill") or []) if x in ("feet", "legs", "torso", "arms", "head")]
    if body_parts:
        try:
            add_body_fill(m, body_parts, ov, log, base_dir=os.path.dirname(os.path.abspath(src)))
        except Exception as e:  # never fatal
            log("  ! достройка тела: %s" % e)
    if ov.get("wardrobe"):
        try:
            add_wardrobe(m, ov, log)
        except Exception as e:  # never fatal
            log("  ! гардероб: %s" % e)
    base_dir = os.path.dirname(os.path.abspath(src))
    nv = len(m.positions)
    log("  модель: %s | вершин %d, граней %d, материалов %d, костей %d, морфов %d, жёстких тел %d"
        % (m.name, nv, len(m.indices) // 3, len(m.materials), len(m.bones),
           len(m.morphs), len(m.rigid_bodies)))
    if not m.bones:
        raise ConvertError("В модели нет костей")
    warnings = []
    if face_from_bones:
        try:
            synth_face_morphs(m, log)
        except Exception as e:  # never fatal
            log("  ! выражения из костей: %s" % e)

    def warn(s):
        warnings.append(s)
        log("  ! " + s)

    # ---------------- skeleton ----------------
    names = [b.name for b in m.bones]
    pos = [np.array(b.pos, float) for b in m.bones]
    nb0 = len(names)
    parent = [b.parent if 0 <= b.parent < nb0 else -1 for b in m.bones]
    # break cycles
    for i in range(nb0):
        seen = set()
        j = i
        while j != -1:
            if j in seen:
                parent[i] = -1
                break
            seen.add(j)
            j = parent[j]

    # merge Blender-style duplicates "Bone.001" sitting on top of "Bone" (split/merged armatures)
    import re as _re
    alias = list(range(nb0))
    by_name = {n: i for i, n in enumerate(names)}
    _ys = m.positions[:, 1] if len(m.positions) else np.zeros(1)
    dup_tol = max(0.05, 0.03 * float(_ys.max() - _ys.min()))  # ~3% of model height
    n_dup = 0
    # two bones that each carry their own physics body are separate strips (skirt panels "qun_001",
    # "qun_001.001" ... side by side), not a duplicated armature: never merge those
    _dyn_b = {rb.bone for rb in m.rigid_bodies if rb.mode in (1, 2)}
    for i, n in enumerate(names):
        mm_ = _re.fullmatch(r"(.+)\.\d{3}", n or "")
        if not mm_ or mm_.group(1) not in by_name:
            continue
        j = by_name[mm_.group(1)]
        if i in _dyn_b and j in _dyn_b:
            continue
        if j != i and np.linalg.norm(pos[i] - pos[j]) < dup_tol:
            alias[i] = j
            n_dup += 1
    if n_dup:
        for i in range(nb0):
            a_ = alias[i]
            while alias[a_] != a_:
                a_ = alias[a_]
            alias[i] = a_
        for i in range(nb0):
            if alias[i] != i:
                continue
            p_ = parent[i]
            if p_ >= 0 and alias[p_] != p_:
                tgt = alias[p_]
                if tgt != i:
                    parent[i] = tgt
        log("  скелет: %d дублей костей (вида «Кость.001») слиты с оригиналами" % n_dup)
    # again break cycles that merging might create
    for i in range(nb0):
        seen = set()
        j = i
        while j != -1:
            if j in seen:
                parent[i] = -1
                break
            seen.add(j)
            j = parent[j]
    parent_mmd = list(parent)
    order_mmd = []
    _ch = [[] for _ in range(nb0)]
    for i in range(nb0):
        if parent_mmd[i] >= 0:
            _ch[parent_mmd[i]].append(i)
    _st = [i for i in range(nb0) if parent_mmd[i] < 0]
    while _st:
        x = _st.pop()
        order_mmd.append(x)
        _st.extend(_ch[x])

    def is_ancestor(a, b):
        j = parent[b]
        while j != -1:
            if j == a:
                return True
            j = parent[j]
        return False

    find = {}
    for i, n in enumerate(names):
        find.setdefault(nk(n), i)
    h = map_humanoid(find)
    n_jp = len(h)
    h = map_humanoid_en(names, h)
    if len(h) > n_jp:
        log("  %d костей найдено по английским/игровым именам" % (len(h) - n_jp))
    hint = getattr(m, "humanoid_hint", None)
    if hint:  # VRM input: the humanoid map is already known
        for k_, v_ in hint.items():
            if 0 <= v_ < len(names):
                for k2 in [k2 for k2, v2 in h.items() if v2 == v_ and k2 != k_]:
                    del h[k2]
                h[k_] = v_
    if sum(1 for b in REQUIRED if b not in h) >= 4:
        try:
            wsum_ = np.zeros(len(names))
            okj = (m.joints >= 0) & (m.joints < len(names))
            np.add.at(wsum_, m.joints[okj], m.weights[okj])
            hg = map_humanoid_geometry(pos, parent, wsum_, float(np.ptp(m.positions[:, 1])) if len(m.positions) else 0, log)
            used_ = set(h.values())
            for k_, v_ in hg.items():
                if k_ not in h and v_ not in used_:
                    h[k_] = v_
                    used_.add(v_)
        except Exception as e:
            log("  ! распознавание скелета по форме: %s" % e)
    for vb, bname in (ov.get("humanoid") or {}).items():
        if not bname:
            h.pop(vb, None)
            continue
        i = names.index(bname) if bname in names else None
        if i is not None:
            for k in [k for k, v in h.items() if v == i and k != vb]:
                del h[k]
            h[vb] = i
    auto_h = dict(h)

    # reconnect humanoid bones that hang off the wrong place (merged/split armatures in rips)
    EXPECT = [("chest", ["spine"]), ("upperChest", ["chest"]),
              ("neck", ["upperChest", "chest", "spine"]), ("head", ["neck"]),
              ("leftShoulder", ["upperChest", "chest"]), ("rightShoulder", ["upperChest", "chest"]),
              ("leftUpperArm", ["leftShoulder", "upperChest", "chest"]),
              ("rightUpperArm", ["rightShoulder", "upperChest", "chest"]),
              ("leftLowerArm", ["leftUpperArm"]), ("rightLowerArm", ["rightUpperArm"]),
              ("leftHand", ["leftLowerArm"]), ("rightHand", ["rightLowerArm"]),
              ("leftLowerLeg", ["leftUpperLeg"]), ("rightLowerLeg", ["rightUpperLeg"]),
              ("leftFoot", ["leftLowerLeg"]), ("rightFoot", ["rightLowerLeg"]),
              ("leftToes", ["leftFoot"]), ("rightToes", ["rightFoot"]),
              ("leftEye", ["head"]), ("rightEye", ["head"])]
    n_fix = 0
    for vb, exp in EXPECT:
        if vb not in h:
            continue
        b_ = h[vb]
        cand = [h[e] for e in exp if e in h]
        if not cand or any(is_ancestor(c, b_) for c in cand):
            continue
        tgt = cand[0]
        if tgt == b_ or is_ancestor(b_, tgt):
            continue
        parent[b_] = tgt
        n_fix += 1
    if n_fix:
        log("  скелет: %d гуманоидных костей перецеплено к правильным родителям" % n_fix)

    # hips must be an ancestor of spine (in MMD both hang from センター)
    if "hips" in h and "spine" in h and not is_ancestor(h["hips"], h["spine"]):
        if not is_ancestor(h["spine"], h["hips"]):
            parent[h["spine"]] = h["hips"]
            log("  上半身 перецеплен под 下半身 (hips)")
    for side in ("left", "right"):
        ul = h.get(side + "UpperLeg")
        if ul is None or "hips" not in h:
            continue
        under_spine = "spine" in h and (h["spine"] == parent[ul] or is_ancestor(h["spine"], ul))
        if (not is_ancestor(h["hips"], ul) and not is_ancestor(ul, h["hips"])) or under_spine:
            parent[ul] = h["hips"]
            if under_spine:
                log("  бедро %s было под позвоночником (Biped) — перецеплено к тазу" % names[ul])

    # ---- "Восстановить скелет": add missing limb/torso bones and split weights onto them
    splits = []      # (from_bone, new_bone, seg_start, seg_end, t_new, band)
    repaired = []
    P_rest0 = m.positions.astype(np.float64)
    main_bone = None
    if repair:
        mw = m.weights
        mj = m.joints
        main_bone = mj[np.arange(len(mj)), np.argmax(mw, axis=1)]

    def add_bone(name, p, par, vb):
        i_ = len(names)
        names.append(name)
        pos.append(np.array(p, float))
        parent.append(par)
        h[vb] = i_
        repaired.append(vb)
        return i_

    def insert_between(vb, a_vb, c_vb, t, name):
        a_, c_ = h[a_vb], h[c_vb]
        p = pos[a_] + (pos[c_] - pos[a_]) * t
        n_ = add_bone(name, p, a_, vb)
        parent[c_] = n_
        splits.append((a_, n_, pos[a_].copy(), pos[c_].copy(), t, 0.08))

    src_of = {}  # restored bone -> original bone whose vertices it covers

    def extend_end(vb, a_vb, name, frac, axis_hint=None):
        a_ = h[a_vb]
        src = src_of.get(a_, a_)
        sel = (main_bone == src) if main_bone is not None else None
        if sel is not None and sel.sum() < 4:
            # bone without own vertices: take the vertices of its whole subtree
            sub_ = {src}
            for j in range(len(names)):
                if is_ancestor(src, j):
                    sub_.add(j)
            sel = np.isin(main_bone, list(sub_))
        if sel is None or sel.sum() < 4:
            return
        V = P_rest0[sel]
        pa = pos[a_]
        # limb direction = towards the far end of the limb's own vertices
        if src != a_:
            # continuing a restored bone: keep going along the original segment direction
            d = pa - pos[src]
        else:
            dist = np.linalg.norm(V - pa, axis=1)
            far = V[dist >= np.percentile(dist, 85)]
            d = far.mean(axis=0) - pa if len(far) else np.array(axis_hint or (0, -1.0, 0), float)
        if np.linalg.norm(d) < 1e-6:
            d = np.array(axis_hint or (0, -1.0, 0), float)
        d = d / (np.linalg.norm(d) or 1)
        proj = (V - pa) @ d
        proj = proj[proj > 0]
        if len(proj) < 4:
            return
        ext = float(np.percentile(proj, 99))
        if ext <= 0:
            return
        p = pa + d * ext * frac
        n_ = add_bone(name, p, a_, vb)
        for j in range(len(names) - 1):
            if parent[j] == a_ and j != n_:
                parent[j] = n_
        src_of[n_] = src
        splits.append((a_, n_, pa.copy(), pa + d * ext, frac, 0.06))

    if repair:
        for sd, L in (("left", "L"), ("right", "R")):
            if sd + "LowerArm" not in h and sd + "UpperArm" in h and sd + "Hand" in h:
                insert_between(sd + "LowerArm", sd + "UpperArm", sd + "Hand", 0.5, "Elbow_" + L)
            if sd + "Hand" not in h and sd + "LowerArm" in h:
                extend_end(sd + "Hand", sd + "LowerArm", "Wrist_" + L, 0.72)
            if sd + "LowerArm" not in h and sd + "UpperArm" in h and sd + "Hand" not in h:
                extend_end(sd + "LowerArm", sd + "UpperArm", "Elbow_" + L, 0.5)
                if sd + "LowerArm" in h:
                    extend_end(sd + "Hand", sd + "LowerArm", "Wrist_" + L, 0.72)
            if sd + "LowerLeg" not in h and sd + "UpperLeg" in h and sd + "Foot" in h:
                insert_between(sd + "LowerLeg", sd + "UpperLeg", sd + "Foot", 0.5, "Knee_" + L)
            if sd + "Foot" not in h and sd + "LowerLeg" in h:
                extend_end(sd + "Foot", sd + "LowerLeg", "Ankle_" + L, 0.75, axis_hint=(0, -1, 0))
            if sd + "LowerLeg" not in h and sd + "UpperLeg" in h and sd + "Foot" not in h:
                extend_end(sd + "LowerLeg", sd + "UpperLeg", "Knee_" + L, 0.5, axis_hint=(0, -1, 0))
                if sd + "LowerLeg" in h:
                    extend_end(sd + "Foot", sd + "LowerLeg", "Ankle_" + L, 0.75, axis_hint=(0, -1, 0))
        if "neck" not in h and "head" in h and ("upperChest" in h or "chest" in h or "spine" in h):
            torso = next(t for t in ("upperChest", "chest", "spine") if t in h)
            insert_between("neck", torso, "head", 0.55, "Neck")
        if "spine" not in h and "hips" in h and ("chest" in h or "neck" in h):
            insert_between("spine", "hips", "chest" if "chest" in h else "neck", 0.3, "Spine")
        if repaired:
            log("  восстановление скелета: добавлены кости " + ", ".join(repaired))
        else:
            log("  восстановление скелета: недостающих костей конечностей не найдено")

    # synthetic chest if model has only 上半身
    if "chest" not in h and "spine" in h and "neck" in h:
        sp = h["spine"]
        ci = len(names)
        names.append("Chest")
        pos.append(pos[sp] + (pos[h["neck"]] - pos[sp]) * 0.35)
        parent.append(sp)
        for j in range(ci):
            if parent[j] == sp:
                parent[j] = ci
        h["chest"] = ci
        log("  нет 上半身2 — добавлена кость Chest")

    missing = [b for b in REQUIRED if b not in h]
    info = {
        "model": {"name": m.name, "name_en": m.name_en, "comment": m.comment,
                  "vertices": nv, "faces": len(m.indices) // 3, "bones": nb0,
                  "morphs": len(m.morphs), "rigid_bodies": len(m.rigid_bodies)},
        "bones": names[:nb0],
        "bones_ru": {n: names_ru.translate(n) for n in names},
        "bones_en": {n: names_en.translate(n) for n in names},
        "humanoid": {k: names[v] for k, v in h.items()},
        "required": REQUIRED,
        "missing": missing,
        "warnings": warnings,
        "repaired": repaired,
    }
    if missing and editor:
        info["vrm"] = None
        log("  ! не найдены кости: " + ", ".join(missing) + " — укажите вручную")
        return info
    if missing:
        raise ConvertError("Не найдены обязательные кости: " + ", ".join(missing) +
                           "\n(ожидаются стандартные японские имена: 下半身, 上半身, 首, 頭, 左腕, 左ひじ, 左手首, 左足, 左ひざ, 左足首 ...)")

    nb = len(names)
    pos = np.array(pos, float)
    parent = np.array(parent, int)
    children = [[] for _ in range(nb)]
    for i in range(nb):
        if parent[i] >= 0:
            children[parent[i]].append(i)

    def subtree(r):
        out, st = [], [r]
        while st:
            x = st.pop()
            out.append(x)
            st.extend(children[x])
        return out

    # ---------------- T-pose ----------------
    M = np.tile(np.eye(4), (nb, 1, 1))
    cur = pos.copy()

    def rotate_subtree(root, R, pivot):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pivot - R @ pivot
        for j in subtree(root):
            M[j] = T @ M[j]
            cur[j] = T[:3, :3] @ cur[j] + T[:3, 3]

    rest_fix = {}  # MMD-space orientation of the original (A-pose) rest relative to the VRM T-pose rest
    if tpose:
        for side in ("left", "right"):
            ua, la, hd = h[side + "UpperArm"], h[side + "LowerArm"], h[side + "Hand"]
            d = cur[la] - cur[ua]
            target = np.array([1.0 if d[0] >= 0 else -1.0, 0, 0])
            ang = math.degrees(math.acos(max(-1, min(1, abs(d[0]) / (np.linalg.norm(d) + 1e-12)))))
            Rua, Rla = np.eye(3), np.eye(3)
            if ang > 0.5:
                Rua = rot_between(d, target)
                rotate_subtree(ua, Rua, cur[ua].copy())
            d2 = cur[hd] - cur[la]
            ang2 = math.degrees(math.acos(max(-1, min(1, abs(d2[0]) / (np.linalg.norm(d2) + 1e-12)))))
            if ang2 > 0.5:
                Rla = rot_between(d2, target)
                rotate_subtree(la, Rla, cur[la].copy())
            rest_fix[side + "UpperArm"] = mat_quat(Rua.T)
            rest_fix[side + "LowerArm"] = mat_quat(Rua.T @ Rla.T)
            log("  T-поза: %s рука повернута на %.1f°, предплечье на %.1f°"
                % ("левая" if side == "left" else "правая", ang, ang2))

    # ---------------- vertices ----------------
    jo = m.joints.copy()
    if n_dup:
        al = np.array(alias + list(range(nb0, max(nb0, 1))), dtype=np.int64)[:nb0]
        ok_ = (jo >= 0) & (jo < nb0)
        jo[ok_] = al[jo[ok_]]
    we = m.weights.astype(np.float64).copy()
    bad = (jo < 0) | (jo >= nb0)
    we[bad] = 0
    jo[bad] = 0
    s = we.sum(axis=1)
    zero = s <= 1e-8
    if zero.any():
        warn("%d вершин без весов привязаны к корневой кости" % int(zero.sum()))
        we[zero] = [1, 0, 0, 0]
        jo[zero] = 0
        s[zero] = 1
    we /= s[:, None]
    # move weights onto restored bones (smooth band around the new joint)
    for a_, n_, sa, sc, tn, band in splits:
        seg = sc - sa
        L2 = float(seg @ seg) or 1e-9
        rows, cols = np.nonzero(jo == a_)
        if not len(rows):
            continue
        t = ((m.positions[rows].astype(np.float64) - sa) @ seg) / L2
        f = np.clip((t - (tn - band)) / (2 * band), 0, 1)
        f = f * f * (3 - 2 * f)
        moved_w = we[rows, cols] * f
        we[rows, cols] -= moved_w
        for r_, c_, w_ in zip(rows, cols, moved_w):
            if w_ <= 1e-6:
                continue
            free = [k for k in range(4) if we[r_, k] <= 1e-9 and k != c_]
            if we[r_, c_] <= 1e-9:
                jo[r_, c_], we[r_, c_] = n_, w_
            elif free:
                jo[r_, free[0]], we[r_, free[0]] = n_, w_
            else:
                k = int(np.argmin(np.where(np.arange(4) == c_, 9, we[r_])))
                we[r_, k] = 0
                jo[r_, k], we[r_, k] = n_, w_
                we[r_] /= we[r_].sum()
    if splits:
        log("  веса перераспределены на восстановленные кости")

    P = m.positions.astype(np.float64)
    N = m.normals.astype(np.float64)
    if anime_face:
        flatten_face_normals(m, P, N, h, children, jo, we, ov.get("materials") or {}, log)
    Lin = None
    if tpose:
        Mb = np.einsum("nk,nkab->nab", we, M[jo])
        Lin = Mb[:, :3, :3]
        P = np.einsum("nab,nb->na", Lin, P) + Mb[:, :3, 3]
        N = np.einsum("nab,nb->na", Lin, N)

    vrm1 = str(vrm_version).startswith("1")
    # MMD (LH, faces -Z) -> VRM0 glTF (RH, faces -Z)  or  VRM1 glTF (RH, faces +Z)
    flip = np.array([1.0, 1.0, -1.0]) if vrm1 else np.array([-1.0, 1.0, 1.0])

    def conv(v):
        return v * flip * scale

    P = conv(P)
    N = N * flip
    apply_wardrobe_brush(m, P, N, vrm1, log)
    ln = np.linalg.norm(N, axis=1)
    badn = ln < 1e-8
    N[badn] = [0, 1, 0]
    ln[badn] = 1
    N /= ln[:, None]
    if spring:
        spring_excluded = clean_spring_tree(m, parent, children, cur, jo, we, nb0, log)
    else:
        spring_excluded = set()
    BP = conv(cur)

    # order weights, keep 4
    W = we.astype(np.float32)
    J = jo.astype(np.uint16)

    gb = GLTFBuilder()
    P32 = P.astype(np.float32)
    N32 = N.astype(np.float32)
    UV32 = m.uvs.astype(np.float32)

    # ---------------- morph targets ----------------
    morph_target = {}   # pmx morph index -> global target index
    morph_data = []     # global target -> (sorted vertex idx, offsets)
    target_names = []
    P_rest = m.positions.astype(np.float64)
    jo_orig = jo.copy()
    if len(names) > nb0:  # restored bones -> nearest original ancestor (for bone-morph baking)
        mp_ = np.arange(len(names))
        for i in range(nb0, len(names)):
            j = parent[i]
            while j >= nb0:
                j = parent[j]
            mp_[i] = max(j, 0)
        jo_orig = mp_[jo]
    baked = 0
    for mi, mo in enumerate(m.morphs):
        if mo.type not in (1, 2) or not mo.data:
            continue
        if mo.type == 2:
            offs = bake_bone_morph(m.bones, parent_mmd, order_mmd, mo.data, P_rest, jo_orig, we)
            nzv = np.abs(offs).max(axis=1) > 1e-7
            if not nzv.any():
                continue
            idx, off = np.nonzero(nzv)[0], offs[nzv]
            baked += 1
        else:
            idx, off = mo.data
        ok = (idx >= 0) & (idx < nv)
        idx, off = idx[ok], off[ok].astype(np.float64)
        if len(idx):
            uniq, inv = np.unique(idx, return_inverse=True)
            acc = np.zeros((len(uniq), 3))
            np.add.at(acc, inv, off)
            idx, off = uniq, acc
            if Lin is not None:
                off = np.einsum("nab,nb->na", Lin[idx], off)
            off = conv(off)
            nz = np.abs(off).max(axis=1) > 1e-9
            idx, off = idx[nz], off[nz]
        if not len(idx):
            continue
        order = np.argsort(idx)
        morph_target[mi] = len(morph_data)
        morph_data.append((idx[order], off[order]))
        target_names.append(mo.name)
    target_names = unique_names(target_names, "morph")
    if baked:
        log("  %d морфов костей преобразовано в морфы вершин" % baked)
    skipped = sum(1 for mo in m.morphs if mo.type not in (0, 1, 2, 8))
    if skipped:
        log("  %d морфов других типов (UV/кость-импульс) пропущено" % skipped)

    # ---------------- textures & materials ----------------
    images, textures = [], []
    tex_cache = {}
    tex_alpha = {}
    file_cache = {}
    n_resized = [0]

    def get_tex(ti):
        if ti is None or ti < 0 or ti >= len(m.textures):
            return None, "opaque"
        if ti in tex_cache:
            return tex_cache[ti]
        rel = m.textures[ti]
        res = (None, "opaque")
        blob = getattr(m, "texture_blobs", {}).get(ti)
        if blob is not None:
            try:
                raw, mime, kind, rs, amap_ = load_texture(None, max_texture, raw=upscaled_raw(blob, upscale_dir),
                                                          quality=tex_quality, sharpen=tex_sharpen)
                tex_alpha[len(textures)] = amap_
                n_resized[0] += rs
                vi = gb.view(raw)
                images.append({"name": "embedded_%d" % ti, "bufferView": vi, "mimeType": mime})
                textures.append({"source": len(images) - 1, "sampler": 0})
                res = (len(textures) - 1, kind)
                tex_cache[ti] = res
                return res
            except Exception as e:
                warn("встроенная текстура %s не читается: %s" % (rel, e))
        p = resolve_path(base_dir, rel)
        if p is None:
            warn("текстура не найдена: %s" % rel)
        else:
            key = os.path.normcase(os.path.realpath(p))
            if key in file_cache:
                res = file_cache[key]
            else:
                try:
                    raw, mime, kind, rs, amap_ = load_texture_cached(p, max_texture, tex_quality, upscale_dir,
                                                                     tex_sharpen)
                    tex_alpha[len(textures)] = amap_
                    n_resized[0] += rs
                    vi = gb.view(raw)
                    images.append({"name": os.path.splitext(os.path.basename(p))[0] + "_%d" % ti,
                                   "bufferView": vi, "mimeType": mime})
                    textures.append({"source": len(images) - 1, "sampler": 0})
                    res = (len(textures) - 1, kind)
                except Exception as e:
                    warn("не удалось прочитать текстуру %s: %s" % (rel, e))
                file_cache[key] = res
        tex_cache[ti] = res
        return res

    mat_names = unique_names([mt.name for mt in m.materials], "material")
    gl_materials, vrm_materials, primitives = [], [], []
    a_idx_type = 5125 if nv > 65535 else 5123
    idx_dtype = np.uint32 if nv > 65535 else np.uint16
    off_i = 0
    transparent_q = 0
    zwrite_q = 0
    mat_ov = ov.get("materials") or {}
    mat_info = []
    mesh_maps = []
    alpha_morph_mats = set()
    for mo in m.morphs:
        if mo.type == 8 and mo.data:
            for d in mo.data:
                if (d["op"] == 0 and (d["diffuse"][3] < 0.999 or d["tex"][3] < 0.999)) or \
                        (d["op"] == 1 and d["diffuse"][3] < -0.001):
                    if d["mat"] < 0:
                        alpha_morph_mats.update(range(len(m.materials)))
                    else:
                        alpha_morph_mats.add(d["mat"])
    for k, mt in enumerate(m.materials):
        cnt = mt.index_count
        seg = m.indices[off_i:off_i + cnt]
        off_i += cnt
        texname = m.textures[mt.texture] if 0 <= mt.texture < len(m.textures) else ""
        mo_ = mat_ov.get(mat_names[k]) or {}
        auto_cat = "skin" if mt.name == "BodyFill" else ("clothes" if mt.name.startswith("WD ") else
                                                         "other" if mt.name.endswith(TAIL_SUFFIX) else
                                                         guess_category(mt.name, mt.name_en, texname))
        _c, sub_, word_ = vocab.material_category(*[x for x in (mt.name, mt.name_en, texname) if x])
        rec = {"name": mat_names[k], "name_en": mt.name_en, "texture": texname,
               "name_ru": names_ru.translate(mat_names[k]),
               "auto_category": auto_cat, "category": mo_.get("category") or auto_cat,
               "visible": bool(mo_.get("visible", True)), "gltf_index": None,
               "sublabel": sub_ if _c == auto_cat else None, "matched": word_,
               "faces": int(cnt // 3)}
        wd_ = (getattr(m, "_wd_owner", None) or {}).get(mt.name)
        if wd_ is not None:
            rec["wardrobe"] = wd_["uid"]
        if mt.name in piece_info:
            rec["piece_of"], rec["piece_label"] = piece_info[mt.name]
            rec["piece_of_ru"] = names_ru.translate(rec["piece_of"])
        elif n_pieces.get(mt.name, 0) > 1:
            rec["n_pieces"] = n_pieces[mt.name]
        mat_info.append(rec)
        if not rec["visible"]:
            continue
        if cnt <= 0 or len(seg) < 3:
            continue
        seg = seg[:len(seg) // 3 * 3]
        if seg.max() >= nv:
            warn("материал %s ссылается на несуществующие вершины — пропущен" % mt.name)
            continue
        tri = seg.reshape(-1, 3)[:, ::-1]  # flip winding (mirror transform)
        # one mesh per material with its own vertices -> morphs only where they matter
        used, local = np.unique(tri.reshape(-1), return_inverse=True)
        big = len(used) > 65535
        a_ind = gb.accessor(local.astype(np.uint32 if big else np.uint16), 5125 if big else 5123,
                            "SCALAR", 34963)
        a_pos = gb.accessor(P32[used], 5126, "VEC3", 34962, minmax=True)
        a_nor = gb.accessor(N32[used], 5126, "VEC3", 34962)
        a_uv = gb.accessor(UV32[used], 5126, "VEC2", 34962)
        a_j = gb.accessor(J[used], 5123, "VEC4", 34962)
        a_w = gb.accessor(W[used], 5126, "VEC4", 34962)
        p_targets, p_tnames, p_map = [], [], {}
        for gk, (gidx, goff) in enumerate(morph_data):
            pos_ = np.searchsorted(used, gidx)
            ok_ = pos_ < len(used)
            ok_[ok_] = used[pos_[ok_]] == gidx[ok_]
            if not ok_.any():
                continue
            p_map[gk] = len(p_targets)
            p_targets.append({"POSITION": gb.sparse_vec3(len(used), pos_[ok_], goff[ok_])})
            p_tnames.append(target_names[gk])

        ti, kind = get_tex(mt.texture)
        skin_src = mo_.get("as_skin")
        if skin_src:
            # "make it skin": keep the shape of this piece (a swimsuit cup IS the breast surface),
            # paint it in the tone of the chosen material
            sk = mat_names.index(skin_src) if skin_src in mat_names else next(
                (i_ for i_, x_ in enumerate(m.materials)
                 if guess_category(x_.name, x_.name_en, "") == "skin" and i_ != k), None)
            if sk is not None and sk != k:
                rec["as_skin"] = mat_names[sk]
                ti, kind = None, "opaque"
        amap = tex_alpha.get(ti) if ti is not None else None
        if amap is not None:
            # alpha only where THIS material's UVs land (atlas textures are shared)
            uvs_ = UV32[used]
            tri_uv = UV32[tri].mean(axis=1)
            smp = np.concatenate([uvs_, tri_uv])
            hh_, ww_ = amap.shape
            av = amap[(np.clip(np.mod(smp[:, 1], 1.0), 0, 0.9999) * hh_).astype(int),
                      (np.clip(np.mod(smp[:, 0], 1.0), 0, 0.9999) * ww_).astype(int)]
            trans = float((av < 250).mean())
            if trans < 0.01:
                kind = "opaque"
            else:
                partial = float(((av > 5) & (av < 250)).mean())
                kind = "blend" if partial > trans * 0.5 else "cutout"
        dif = mt.diffuse
        color = [min(1.0, dif[c] * 0.6 + mt.ambient[c]) for c in range(3)]
        alpha = float(dif[3])
        if max(color) < 0.05 and ti is not None and alpha >= 0.99:
            color = [1.0, 1.0, 1.0]
        if rec.get("as_skin"):
            color = material_skin_color(m, mat_names.index(rec["as_skin"]), base_dir)
            alpha = 1.0
        if alpha < 0.99:
            kind = "blend"
        if ti is None and rec.get("sublabel") in ("eyeshell", "tear"):
            kind, alpha = "blend", 0.05  # cornea / tear film without texture: must not hide the iris
        if kind == "opaque" and k in alpha_morph_mats:
            kind = "cutout"  # a material morph hides it: Opaque MToon ignores _Color alpha
        rec["auto_blend"] = kind
        if mo_.get("blend") in ("opaque", "cutout", "blend"):
            kind = mo_["blend"]
        rec["blend"] = kind
        # large transparent meshes (hair, clothes) need depth write or they sort wrongly
        zwrite_blend = kind == "blend" and (rec["category"] in ("hair", "clothes", "skin") or rec["faces"] > 3000) \
            and alpha >= 0.99
        double = bool(mt.flags & 0x01)
        if rec["category"] == "eyes" or rec.get("sublabel") in ("sclera", "iris", "pupil", "highlight"):
            double = True  # inner eye layers are often flipped in rips: never cull them away
        rec["auto_double_sided"] = double
        if "double_sided" in mo_:
            double = bool(mo_["double_sided"])
        rec["double_sided"] = double

        shade = None
        if mt.toon_shared == 0 and 0 <= mt.toon < len(m.textures):
            tp = resolve_path(base_dir, m.textures[mt.toon])
            if tp:
                shade = toon_shade(tp)
        if shade is None:
            shade = [0.82, 0.78, 0.82]
        flat_face = anime_face and (rec["category"] in ("face", "skin") or rec.get("as_skin")) and not rec.get("sublabel")
        if flat_face:
            shade = [max(0.88, x) for x in shade]  # soft, light anime shadow on skin
        shade_col = [color[c] * shade[c] for c in range(3)]

        name = mat_names[k]
        pbr = {"baseColorFactor": [srgb_to_linear(c) for c in color] + [alpha],
               "metallicFactor": 0.0, "roughnessFactor": 1.0}
        if ti is not None:
            pbr["baseColorTexture"] = {"index": ti}
        gm = {"name": name, "pbrMetallicRoughness": pbr, "doubleSided": double,
              "extensions": {"KHR_materials_unlit": {}}}
        if kind == "cutout":
            gm["alphaMode"] = "MASK"
            gm["alphaCutoff"] = 0.5
        elif kind == "blend":
            gm["alphaMode"] = "BLEND"
        else:
            gm["alphaMode"] = "OPAQUE"

        use_outline = bool(outline and (mt.flags & 0x10) and mt.edge_size > 0)
        if "outline" in mo_:
            use_outline = bool(mo_["outline"]) and mt.edge_size > 0
        rec["outline"] = use_outline
        fp = {
            "_Cutoff": 0.5, "_BumpScale": 1.0, "_ReceiveShadowRate": 1.0,
            "_ShadingGradeRate": 1.0, "_ShadeShift": -0.25 if flat_face else 0.0,
            "_ShadeToony": 0.97 if flat_face else 0.9,
            "_LightColorAttenuation": 0.0, "_IndirectLightIntensity": 0.1,
            "_RimLightingMix": 0.0, "_RimFresnelPower": 1.0, "_RimLift": 0.0,
            "_OutlineWidth": float(min(1.0, max(0.01, mt.edge_size * 0.1))) if use_outline else 0.0,
            "_OutlineScaledMaxDistance": 1.0, "_OutlineLightingMix": 1.0,
            "_UvAnimScrollX": 0.0, "_UvAnimScrollY": 0.0, "_UvAnimRotation": 0.0,
            "_MToonVersion": 38, "_DebugMode": 0,
            "_OutlineWidthMode": 1 if use_outline else 0, "_OutlineColorMode": 0,
            "_CullMode": 0 if double else 2, "_OutlineCullMode": 1,
        }
        kw = {}
        if kind == "opaque":
            fp.update(_BlendMode=0, _SrcBlend=1, _DstBlend=0, _ZWrite=1, _AlphaToMask=0)
            rq, rtype = 2000, "Opaque"
        elif kind == "cutout":
            fp.update(_BlendMode=1, _SrcBlend=1, _DstBlend=0, _ZWrite=1, _AlphaToMask=1)
            rq, rtype = 2450, "TransparentCutout"
            kw["_ALPHATEST_ON"] = True
        elif zwrite_blend:
            fp.update(_BlendMode=3, _SrcBlend=5, _DstBlend=10, _ZWrite=1, _AlphaToMask=0)
            rq, rtype = 2501 + min(zwrite_q, 48), "Transparent"
            zwrite_q += 1
            kw["_ALPHABLEND_ON"] = True
        else:
            fp.update(_BlendMode=2, _SrcBlend=5, _DstBlend=10, _ZWrite=0, _AlphaToMask=0)
            rq, rtype = 3000 + min(transparent_q, 49), "Transparent"
            transparent_q += 1
            kw["_ALPHABLEND_ON"] = True
        if use_outline:
            kw["MTOON_OUTLINE_WIDTH_WORLD"] = True
            kw["MTOON_OUTLINE_COLOR_FIXED"] = True
        st = [0.0, 0.0, 1.0, 1.0]
        vp = {"_Color": color + [alpha], "_ShadeColor": shade_col + [1.0],
              "_MainTex": st, "_ShadeTexture": st, "_BumpMap": st,
              "_ReceiveShadowTexture": st, "_ShadingGradeTexture": st,
              "_RimTexture": st, "_SphereAdd": st, "_EmissionMap": st,
              "_OutlineWidthTexture": st, "_UvAnimMaskTexture": st,
              "_EmissionColor": [0.0, 0.0, 0.0, 1.0], "_RimColor": [0.0, 0.0, 0.0, 1.0],
              "_OutlineColor": list(mt.edge_color[:3]) + [1.0]}
        tp_ = {}
        if ti is not None:
            tp_ = {"_MainTex": ti, "_ShadeTexture": ti}
        vrm_materials.append({"name": name, "shader": "VRM/MToon", "renderQueue": rq,
                              "floatProperties": fp, "vectorProperties": vp,
                              "textureProperties": tp_, "keywordMap": kw,
                              "tagMap": {"RenderType": rtype}})
        gl_materials.append(gm)
        rec["gltf_index"] = len(gl_materials) - 1
        prim = {"attributes": {"POSITION": a_pos, "NORMAL": a_nor, "TEXCOORD_0": a_uv,
                               "JOINTS_0": a_j, "WEIGHTS_0": a_w},
                "indices": a_ind, "material": len(gl_materials) - 1, "mode": 4}
        mesh_ = {"name": name, "primitives": [prim]}
        if p_targets:
            prim["targets"] = p_targets
            prim["extras"] = {"targetNames": p_tnames}
            mesh_["extras"] = {"targetNames": p_tnames}
        primitives.append(mesh_)
        mesh_maps.append(p_map)
    if not primitives:
        raise ConvertError("В модели нет полигонов")
    log("  материалов: %d, текстур: %d" % (len(gl_materials), len(images)))
    if n_resized[0]:
        log("  %d текстур уменьшено до %d px" % (n_resized[0], max_texture))

    # ---------------- nodes ----------------
    renames = ov.get("renames") or {}
    lang = bone_lang if bone_lang in ("ru", "en") else ("ru" if ru_names else "orig")
    disp = []
    for n in names:
        if renames.get(n):
            disp.append(renames[n])
        elif lang == "ru":
            disp.append(names_ru.translate(n))
        elif lang == "en":
            disp.append(names_en.translate(n))
        else:
            disp.append(n)
    node_names = unique_names(disp, "bone")
    nodes = []
    for i in range(nb):
        p = parent[i]
        t = BP[i] - (BP[p] if p >= 0 else 0)
        nd = {"name": node_names[i], "translation": [float(x) for x in t]}
        if children[i]:
            nd["children"] = list(children[i])
        nodes.append(nd)
    used_nn = set(node_names)
    mesh_nodes = []
    for mi_, mesh_ in enumerate(primitives):
        mname = names_en.translate(mesh_["name"]) if lang == "en" else mesh_["name"]
        nn = "Mesh_" + mname
        k_ = 1
        while nn in used_nn:
            k_ += 1
            nn = "Mesh_%s_%d" % (mname, k_)
        used_nn.add(nn)
        mesh_nodes.append(len(nodes))
        nodes.append({"name": nn, "mesh": mi_, "skin": 0})
    roots = [i for i in range(nb) if parent[i] < 0]
    ibm = np.zeros((nb, 16), np.float32)
    ibm[:, 0] = ibm[:, 5] = ibm[:, 10] = ibm[:, 15] = 1
    ibm[:, 12:15] = -BP
    a_ibm = gb.accessor(ibm, 5126, "MAT4")
    skin = {"inverseBindMatrices": a_ibm, "joints": list(range(nb)), "skeleton": roots[0]}


    # ---------------- blend shapes ----------------
    find_morph = {}
    for i, mo in enumerate(m.morphs):
        find_morph.setdefault(nk(mo.name), i)

    def expand(mi, w, depth=0):
        mo = m.morphs[mi]
        if mi in morph_target:
            return {morph_target[mi]: w}
        out = {}
        if mo.type == 0 and mo.data and depth < 4:
            for ci, cw in mo.data:
                if 0 <= ci < len(m.morphs) and ci != mi:
                    for t, ww in expand(ci, w * cw, depth + 1).items():
                        out[t] = out.get(t, 0) + ww
        return out

    # ---- material morphs (PMX type 8) -> VRM materialValues on _Color
    mat_by_pmx = {k: rec for k, rec in enumerate(mat_info) if rec.get("gltf_index") is not None}
    base_col = {k: (vrm_materials[rec["gltf_index"]]["vectorProperties"]["_Color"], m.materials[k])
                for k, rec in mat_by_pmx.items()}

    def expand_mat(mi, w=1.0, depth=0):
        """-> {pmx material index: (rgba target at full weight, weight)}"""
        mo = m.morphs[mi]
        out = {}
        if mo.type == 8 and mo.data:
            for d in mo.data:
                ks = list(base_col) if d["mat"] < 0 else ([d["mat"]] if d["mat"] in base_col else [])
                for k in ks:
                    col, mt_ = base_col[k]
                    cur = out.get(k, (list(col), 0))[0]
                    if d["op"] == 0:
                        tgt = [cur[c] * d["diffuse"][c] * d["tex"][c] for c in range(3)] + \
                              [cur[3] * d["diffuse"][3] * d["tex"][3]]
                    else:
                        tgt = [cur[c] + d["diffuse"][c] * 0.6 + d["ambient"][c] for c in range(3)] + \
                              [cur[3] + d["diffuse"][3]]
                    tgt = [float(min(1.0, max(0.0, x))) for x in tgt]
                    out[k] = (tgt, w)
        elif mo.type == 0 and mo.data and depth < 4:
            for ci, cw in mo.data:
                if 0 <= ci < len(m.morphs) and ci != mi:
                    _merge_mat(out, expand_mat(ci, w * cw, depth + 1))
        return out

    def _merge_mat(acc, add):
        """sum the effects (deltas from the base colour) of two material-morph sets"""
        for k, (tgt, w) in add.items():
            base = base_col[k][0]
            eff = [base[c] + (tgt[c] - base[c]) * w for c in range(4)]
            if k in acc:
                t0, w0 = acc[k]
                eff = [eff[c] + (t0[c] - base[c]) * w0 for c in range(4)]
            acc[k] = ([float(min(1.0, max(0.0, x))) for x in eff], 1.0)

    def mat_values_of(dm, scale_w=1.0):
        vals = []
        for k, (tgt, w) in sorted(dm.items()):
            col = base_col[k][0]
            ww = max(0.0, min(1.0, w * scale_w))
            t = [col[c] + (tgt[c] - col[c]) * ww for c in range(4)]
            vals.append({"materialName": mat_by_pmx[k]["name"], "propertyName": "_Color",
                         "targetValue": [round(float(x), 4) for x in t]})
        return vals

    def binds_of(d):
        out = []
        for t, w in sorted(d.items()):
            for mi_, mp in enumerate(mesh_maps):
                if t in mp:
                    out.append({"mesh": mi_, "index": int(mp[t]),
                                "weight": float(max(0.0, min(100.0, w * 100)))})
        return out

    groups = []
    used_names = set()
    preset_ov = ov.get("presets") or {}
    preset_info = {}
    for preset, gname, slots in PRESETS:
        if preset in preset_ov:
            slots = [[n] for n in (preset_ov[preset] or [])]
        acc = {}
        macc = {}
        found = []
        for slot in slots:
            for cand in slot:
                mi = find_morph.get(nk(cand))
                if mi is not None:
                    ex = expand(mi, 1.0)
                    exm = expand_mat(mi, 1.0)
                    if ex or exm:
                        for t, w in ex.items():
                            acc[t] = max(acc.get(t, 0), w)
                        _merge_mat(macc, exm)
                        found.append(m.morphs[mi].name)
                        break
        if preset == "joy" and (acc or macc):  # a light blush makes "joy" read much better
            for cand in ("照れ", "赤面", "頬染め", "blush", "てれ"):
                mi = find_morph.get(nk(cand))
                if mi is not None:
                    for k, (tgt, w) in expand_mat(mi, 0.6).items():
                        macc.setdefault(k, (tgt, w))
                    for t, w in expand(mi, 0.6).items():
                        acc.setdefault(t, w)
                    break
        if slots and not acc and not macc:
            continue
        groups.append({"name": gname, "presetName": preset, "binds": binds_of(acc),
                       "materialValues": mat_values_of(macc), "isBinary": False})
        used_names.add(gname)
        preset_info[preset] = found
        if found:
            log("  выражение %-8s <- %s" % (preset, " + ".join(found)))
    if custom_blendshapes:
        for i, mo in enumerate(m.morphs):
            if mo.type not in (0, 1, 2, 8):
                continue
            ex = expand(i, 1.0)
            exm = expand_mat(i, 1.0)
            if not ex and not exm:
                continue
            n = (mo.name or "morph").strip() or "morph"
            base, k = n, 1
            while n in used_names:
                k += 1
                n = "%s_%d" % (base, k)
            used_names.add(n)
            groups.append({"name": n, "presetName": "unknown", "binds": binds_of(ex),
                           "materialValues": mat_values_of(exm), "isBinary": False})

    # ---------------- spring bones ----------------
    bone_groups, collider_groups = [], []
    chain_info, collider_info = [], []
    spring_branches = {}
    vis_bones = None
    if strip_hidden and any(not r["visible"] for r in mat_info):
        vmask = np.zeros(nv, bool)
        o_ = 0
        for r_, mt_ in zip(mat_info, m.materials):
            if r_["visible"]:
                seg_ = m.indices[o_:o_ + mt_.index_count]
                vmask[seg_[seg_ < nv]] = True
            o_ += mt_.index_count
        vis_bones = set(np.unique(jo[vmask][we[vmask] > 1e-4]).tolist())
        vis_bones = (vis_bones, set(np.unique(jo[we > 1e-4]).tolist()))
    if spring and (m.rigid_bodies or True):
        rest_ = {"P": m.positions.astype(np.float64), "pos": pos, "cur": cur, "jo": jo, "we": we,
                 "vis_bones": vis_bones}
        bone_groups, collider_groups, chain_info, collider_info = build_springs(
            m, h, parent, children, nb0, M, BP, conv, scale, physics, log,
            chain_ov=ov.get("chains"), names=names, excluded=spring_excluded, rest=rest_)
        spring_branches = rest_.get("branches_v1") or {}

    # ---------------- VRM extension ----------------
    human_bones = [{"bone": k, "node": int(v), "useDefaultValues": True}
                   for k, v in sorted(h.items(), key=lambda kv: kv[1])]
    curve = [0, 0, 0, 1, 1, 1, 1, 0]
    vrm = {
        "exporterVersion": "mmd2vrm-" + VERSION,
        "specVersion": "0.0",
        "meta": {
            "title": title or m.name or os.path.splitext(os.path.basename(src))[0],
            "version": "1.0",
            "author": author or "Unknown",
            "contactInformation": "",
            "reference": "Converted from MMD model: " + (m.name or ""),
            "allowedUserName": "OnlyAuthor",
            "violentUssageName": "Disallow",
            "sexualUssageName": "Disallow",
            "commercialUssageName": "Disallow",
            "otherPermissionUrl": "",
            "licenseName": "Other",
            "otherLicenseUrl": "",
        },
        "humanoid": {"humanBones": human_bones, "armStretch": 0.05, "legStretch": 0.05,
                     "upperArmTwist": 0.5, "lowerArmTwist": 0.5, "upperLegTwist": 0.5,
                     "lowerLegTwist": 0.5, "feetSpacing": 0, "hasTranslationDoF": False},
        "firstPerson": {
            "firstPersonBone": int(h["head"]),
            "firstPersonBoneOffset": {"x": 0, "y": 0.06, "z": 0},
            "meshAnnotations": [{"mesh": i, "firstPersonFlag": "Auto"} for i in range(len(primitives))],
            "lookAtTypeName": "Bone" if ("leftEye" in h and "rightEye" in h) else "BlendShape",
            "lookAtHorizontalInner": {"curve": curve, "xRange": 90, "yRange": 8},
            "lookAtHorizontalOuter": {"curve": curve, "xRange": 90, "yRange": 12},
            "lookAtVerticalDown": {"curve": curve, "xRange": 90, "yRange": 10},
            "lookAtVerticalUp": {"curve": curve, "xRange": 90, "yRange": 10},
        },
        "blendShapeMaster": {"blendShapeGroups": groups},
        "secondaryAnimation": {"boneGroups": bone_groups, "colliderGroups": collider_groups},
        "materialProperties": vrm_materials,
    }

    gltf = {
        "asset": {"version": "2.0", "generator": "mmd2vrm " + VERSION},
        "extensionsUsed": ["VRM", "KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": mesh_nodes + roots}],
        "nodes": nodes,
        "meshes": primitives,
        "skins": [skin],
        "materials": gl_materials,
        "extensions": {"VRM": vrm},
    }
    if images:
        gltf["images"] = images
        gltf["textures"] = textures
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987,
                             "wrapS": 10497, "wrapT": 10497}]

    if vrm1:
        vrm["secondaryAnimation"]["_branches"] = spring_branches
        to_vrm1(gltf, vrm, mesh_nodes, title, author)
    data = gb.glb_bytes(gltf)
    try:
        target_mb = float(target_mb or 0)
    except (TypeError, ValueError):
        target_mb = 0
    if target_mb > 0 and len(data) > target_mb * 1048576 and images:
        cur_max = max_texture or 4096
        if cur_max > 256:
            nxt = max(256, cur_max // 2)
            log("  файл %.1f МБ больше %g МБ: текстуры уменьшаются до %d px" % (len(data) / 1048576, target_mb, nxt))
            return convert(src, dst, scale=scale, tpose=tpose, spring=spring, outline=outline,
                           custom_blendshapes=custom_blendshapes, title=title, author=author, max_texture=nxt,
                           physics=physics, overrides=overrides, editor=editor, ru_names=ru_names, repair=repair,
                           face_from_bones=face_from_bones, anime_face=anime_face, vrm_version=vrm_version,
                           target_mb=target_mb, strip_hidden=strip_hidden, tex_quality=tex_quality,
                           tex_sharpen=tex_sharpen,
                           upscale_dir=upscale_dir, bone_lang=bone_lang, log=log)
        warn("Не удалось уложиться в %g МБ: текстуры уже 256 px, остальное — геометрия" % target_mb)
    if dst:
        with open(dst, "wb") as f:
            f.write(data)
    size = len(data) / 1024 / 1024
    log("Готово: %s (%.1f МБ)" % (dst, size))
    log("  гуманоидных костей: %d, выражений: %d (пресетов: %d)"
        % (len(h), len(groups), sum(1 for g in groups if g["presetName"] != "unknown")))
    info.update({
        "vrm": None if dst else data,
        "size_mb": round(size, 2),
        "humanoid": {k: names[v] for k, v in h.items()},
        "humanoid_nodes": {k: int(v) for k, v in h.items()},
        "materials": mat_info,
        "chains": chain_info,
        "_pivot": [float(v) for v in BP[h["hips"]]] if "hips" in h else [0.0, 1.0, 0.0],
        "morphs": [mo.name for mo in m.morphs if mo.type in (0, 1, 2)],
        "presets": preset_info,
        "node_names": node_names,
        "rest_fix": rest_fix,
        "vrm_version": 1 if vrm1 else 0,
        "scale": scale,
        "bone_names": {k: names[v] for k, v in h.items()},
        "missing": [],
        "repaired": repaired,
    })
    return info
