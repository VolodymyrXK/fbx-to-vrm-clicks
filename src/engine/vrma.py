"""VRM Animation (.vrma) export of the built-in animations (web/anims.json).

A .vrma is a glTF binary with the VRMC_vrm_animation extension: a humanoid skeleton in its rest pose
(T-pose, +Z forward, identity rotations) and one animation with local rotations of the humanoid bones
plus the translation of the hips.  The rest skeleton is taken from the VRM being saved, so the hips
height and bone lengths match the model.  The poses are sampled at 30 fps with the same easing as the
preview, so the file plays exactly like the preview.
"""
import json
import math
import os
import struct
import sys

import numpy as np

FPS = 30

# VRM humanoid hierarchy (parent of each bone; the nearest existing ancestor is used)
PARENT = {
    "hips": None, "spine": "hips", "chest": "spine", "upperChest": "chest", "neck": "upperChest", "head": "neck",
    "leftEye": "head", "rightEye": "head", "jaw": "head",
    "leftUpperLeg": "hips", "leftLowerLeg": "leftUpperLeg", "leftFoot": "leftLowerLeg", "leftToes": "leftFoot",
    "rightUpperLeg": "hips", "rightLowerLeg": "rightUpperLeg", "rightFoot": "rightLowerLeg", "rightToes": "rightFoot",
    "leftShoulder": "upperChest", "leftUpperArm": "leftShoulder", "leftLowerArm": "leftUpperArm", "leftHand": "leftLowerArm",
    "rightShoulder": "upperChest", "rightUpperArm": "rightShoulder", "rightLowerArm": "rightUpperArm", "rightHand": "rightLowerArm",
}
for _s in ("left", "right"):
    for _f in ("Thumb", "Index", "Middle", "Ring", "Little"):
        _js = ("Metacarpal", "Proximal", "Distal") if _f == "Thumb" else ("Proximal", "Intermediate", "Distal")
        _prev = _s + "Hand"
        for _j in _js:
            PARENT[_s + _f + _j] = _prev
            _prev = _s + _f + _j
# VRM 0.x names that differ in 1.0
V0_TO_V1 = {"leftThumbProximal": "leftThumbMetacarpal", "leftThumbIntermediate": "leftThumbProximal",
            "rightThumbProximal": "rightThumbMetacarpal", "rightThumbIntermediate": "rightThumbProximal"}


def anims_path():
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "web", "anims.json")


def load_anims():
    with open(anims_path(), encoding="utf-8") as f:
        return {a["id"]: a for a in json.load(f)["anims"]}


# ------------------------------------------------------------------ sampling (same as the preview)
def _ease(x):
    return x * x * (3 - 2 * x)


def _seg(keys, t):
    if t <= keys[0][0]:
        return 0, 0.0
    for i in range(1, len(keys)):
        if t <= keys[i][0]:
            t0, t1 = keys[i - 1][0], keys[i][0]
            return i - 1, _ease((t - t0) / max(1e-6, t1 - t0))
    return len(keys) - 1, 0.0


def sample_track(keys, t):
    """position keys [[t, x, y, z], ...] -> eased value at time t"""
    i, u = _seg(keys, t)
    a, b = keys[i], keys[min(i + 1, len(keys) - 1)]
    return [a[k] + (b[k] - a[k]) * u for k in (1, 2, 3)]


def slerp(a, b, u):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + (b - a) * u
    else:
        th = math.acos(d)
        q = (math.sin((1 - u) * th) * a + math.sin(u * th) * b) / math.sin(th)
    return q / np.linalg.norm(q)


def sample_rot(keys, t):
    """rotation keys [[t, qx, qy, qz, qw], ...] -> quaternion, slerp with the preview easing"""
    i, u = _seg(keys, t)
    return slerp(keys[i][1:5], keys[min(i + 1, len(keys) - 1)][1:5], u)


def euler_xyz_to_quat(x, y, z):
    """three.js Euler order 'XYZ' (radians) -> quaternion [x, y, z, w]"""
    c1, c2, c3 = math.cos(x / 2), math.cos(y / 2), math.cos(z / 2)
    s1, s2, s3 = math.sin(x / 2), math.sin(y / 2), math.sin(z / 2)
    return [s1 * c2 * c3 + c1 * s2 * s3, c1 * s2 * c3 - s1 * c2 * s3,
            c1 * c2 * s3 + s1 * s2 * c3, c1 * c2 * c3 - s1 * s2 * s3]


# ------------------------------------------------------------------ rest skeleton from a VRM
def _glb_json(data):
    if data[:4] != b"glTF":
        raise ValueError("не glb")
    n = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + n].decode("utf-8"))


def rest_skeleton(vrm_bytes):
    """{vrm1 humanoid bone name: world position (+Z forward, metres)} from a VRM 0.x / 1.0 file"""
    j = _glb_json(vrm_bytes)
    nodes = j.get("nodes", [])
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent[c] = i

    def world(i):
        M = np.eye(4)
        chain = []
        while i is not None:
            chain.append(i)
            i = parent.get(i)
        for k in reversed(chain):
            n = nodes[k]
            L = np.eye(4)
            if "matrix" in n:
                L = np.array(n["matrix"], float).reshape(4, 4).T
            else:
                t = n.get("translation", [0, 0, 0])
                q = n.get("rotation", [0, 0, 0, 1])
                s = n.get("scale", [1, 1, 1])
                x, y, z, w = q
                R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                              [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                              [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
                L[:3, :3] = R * np.array(s)
                L[:3, 3] = t
            M = M @ L
        return M[:3, 3]

    ext = j.get("extensions", {})
    bones = {}
    if "VRMC_vrm" in ext:
        for name, v in ext["VRMC_vrm"]["humanoid"]["humanBones"].items():
            bones[name] = world(int(v["node"]))
    elif "VRM" in ext:
        for hb in ext["VRM"]["humanoid"]["humanBones"]:
            name = V0_TO_V1.get(hb["bone"], hb["bone"])
            p = world(int(hb["node"]))
            bones[name] = np.array([-p[0], p[1], -p[2]])  # VRM 0.x looks toward -Z
    else:
        raise ValueError("в файле нет VRM-гуманоида")
    return bones


def default_skeleton(hips_y=0.9):
    """generic proportions, for export without a model"""
    k = hips_y / 0.9
    P = {"hips": (0, 0.9, 0), "spine": (0, 1.0, 0), "chest": (0, 1.12, 0), "upperChest": (0, 1.24, 0), "neck": (0, 1.38, 0),
         "head": (0, 1.48, 0), "leftUpperLeg": (0.09, 0.86, 0), "leftLowerLeg": (0.09, 0.48, 0), "leftFoot": (0.09, 0.08, 0),
         "leftShoulder": (0.03, 1.33, 0), "leftUpperArm": (0.16, 1.33, 0), "leftLowerArm": (0.42, 1.33, 0), "leftHand": (0.66, 1.33, 0)}
    out = {}
    for n, p in P.items():
        out[n] = np.array(p, float) * k
        if n.startswith("left"):
            q = np.array(p, float) * k
            q[0] = -q[0]
            out["right" + n[4:]] = q
    return out


# ------------------------------------------------------------------ export
def build_vrma(anim, skel):
    names = [n for n in PARENT if n in skel]
    if "hips" not in names:
        raise ValueError("нет кости hips")

    def par(n):
        p = PARENT.get(n)
        while p is not None and p not in skel:
            p = PARENT.get(p)
        return p

    idx = {n: i for i, n in enumerate(names)}
    nodes = []
    for n in names:
        p = par(n)
        t = skel[n] - (skel[p] if p else 0)
        nodes.append({"name": n, "translation": [float(x) for x in t]})
    for n in names:
        p = par(n)
        if p:
            nodes[idx[p]].setdefault("children", []).append(idx[n])

    dur = float(anim["duration"])
    nf = max(2, int(round(dur * FPS)) + 1)
    times = np.linspace(0, dur, nf).astype(np.float32)
    bin_ = bytearray()
    views, accs = [], []

    def add(arr, typ, minmax=False):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        while len(bin_) % 4:
            bin_.append(0)
        off = len(bin_)
        bin_.extend(arr.tobytes())
        views.append({"buffer": 0, "byteOffset": off, "byteLength": arr.nbytes})
        a = {"bufferView": len(views) - 1, "componentType": 5126, "count": int(arr.shape[0]), "type": typ}
        if minmax:
            a["min"] = [float(arr.min())]
            a["max"] = [float(arr.max())]
        accs.append(a)
        return len(accs) - 1

    t_acc = add(times, "SCALAR", True)
    channels, samplers = [], []
    for bone, keys in anim["tracks"].items():
        b = bone  # tracks already use VRM 1.0 names
        if b not in idx:
            continue
        q = np.array([sample_rot(keys, float(t)) for t in times])
        for i in range(1, len(q)):  # keep the sign continuous (no flips in linear interpolation)
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] = -q[i]
        samplers.append({"input": t_acc, "output": add(q, "VEC4"), "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": idx[b], "path": "rotation"}})
    hp = np.array([skel["hips"] + np.array(sample_track(anim["hips_pos"], float(t))) for t in times])
    samplers.append({"input": t_acc, "output": add(hp, "VEC3"), "interpolation": "LINEAR"})
    channels.append({"sampler": len(samplers) - 1, "target": {"node": idx["hips"], "path": "translation"}})

    gltf = {
        "asset": {"version": "2.0", "generator": "mmd2vrm"},
        "extensionsUsed": ["VRMC_vrm_animation"],
        "extensions": {"VRMC_vrm_animation": {"specVersion": "1.0",
                                              "humanoid": {"humanBones": {n: {"node": idx[n]} for n in names}}}},
        "scene": 0, "scenes": [{"nodes": [idx["hips"]]}],
        "nodes": nodes,
        "animations": [{"name": anim["id"], "channels": channels, "samplers": samplers}],
        "buffers": [{"byteLength": len(bin_)}], "bufferViews": views, "accessors": accs,
    }
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    while len(bin_) % 4:
        bin_.append(0)
    total = 12 + 8 + len(js) + 8 + len(bin_)
    return (b"glTF" + struct.pack("<II", 2, total) + struct.pack("<I", len(js)) + b"JSON" + js +
            struct.pack("<I", len(bin_)) + b"BIN\x00" + bytes(bin_))


def export_for_vrm(vrm_bytes, vrm_path, ids, log=print):
    """write <name>.<anim id>.vrma next to the saved VRM; returns the written paths"""
    anims = load_anims()
    if ids == "all" or ids == ["all"]:
        ids = list(anims)
    try:
        skel = rest_skeleton(vrm_bytes)
    except Exception as e:
        log("  ! анимации: скелет из VRM не прочитан (%s), взяты средние пропорции" % e)
        skel = default_skeleton()
    base = os.path.splitext(vrm_path)[0]
    out = []
    for i in ids:
        a = anims.get(i)
        if not a:
            log("  ! неизвестная анимация: %s" % i)
            continue
        p = "%s.%s.vrma" % (base, i)
        with open(p, "wb") as f:
            f.write(build_vrma(a, skel))
        out.append(p)
    if out:
        log("  анимации .vrma: %d" % len(out))
    return out
