"""FBX (binary 7.x and ASCII) reader -> pmx.Model-compatible structure for converter.py.

Output space matches PMX/MMD: left-handed, Y-up, model faces -Z. Units are FBX scene units
converted to meters / 0.08 so that the converter's default scale (0.08) yields meters;
model.suggested_scale is set to 0.08 for that reason.
"""
import math
import os
import re
import struct
import zlib

import numpy as np

import pmx


class FBXError(Exception):
    pass


class Node:
    __slots__ = ("name", "props", "children")

    def __init__(self, name, props=None, children=None):
        self.name = name
        self.props = props or []
        self.children = children or []

    def find(self, name):
        for c in self.children:
            if c.name == name:
                return c
        return None

    def find_all(self, name):
        return [c for c in self.children if c.name == name]

    def __repr__(self):
        return "Node(%s, %d props, %d children)" % (self.name, len(self.props), len(self.children))


# ---------------------------------------------------------------- binary
_ARR = {"f": ("f", 4), "d": ("d", 8), "l": ("q", 8), "i": ("i", 4), "b": ("b", 1)}


def _parse_binary(data):
    version = struct.unpack_from("<I", data, 23)[0]
    wide = version >= 7500
    hdr = struct.Struct("<QQQB" if wide else "<IIIB")
    null_len = 25 if wide else 13

    def read_prop(p):
        t = chr(data[p])
        p += 1
        if t == "Y":
            return struct.unpack_from("<h", data, p)[0], p + 2
        if t == "C":
            return bool(data[p]), p + 1
        if t == "I":
            return struct.unpack_from("<i", data, p)[0], p + 4
        if t == "F":
            return struct.unpack_from("<f", data, p)[0], p + 4
        if t == "D":
            return struct.unpack_from("<d", data, p)[0], p + 8
        if t == "L":
            return struct.unpack_from("<q", data, p)[0], p + 8
        if t in _ARR:
            n, enc, clen = struct.unpack_from("<III", data, p)
            p += 12
            raw = data[p:p + clen]
            p += clen
            fmt, sz = _ARR[t]
            if enc == 1:  # never inflate past what the header announces (zip-bomb guard)
                d = zlib.decompressobj()
                raw = d.decompress(raw, n * sz)
            if len(raw) < n * sz:
                raise FBXError("Повреждённый массив в FBX")
            arr = np.frombuffer(raw, dtype="<" + fmt, count=n)
            return arr, p
        if t in "SR":
            n = struct.unpack_from("<I", data, p)[0]
            p += 4
            raw = bytes(data[p:p + n])
            p += n
            if t == "S":
                return raw, p
            return raw, p
        raise FBXError("unknown FBX property type %r" % t)

    def read_node(p):
        end, nprops, plen, nlen = hdr.unpack_from(data, p)
        p += hdr.size
        if end == 0:
            return None, p
        name = bytes(data[p:p + nlen]).decode("ascii", "replace")
        p += nlen
        props = []
        for _ in range(nprops):
            v, p = read_prop(p)
            props.append(v)
        children = []
        while p < end:
            if end - p == null_len and not any(bytes(data[p:end])):
                p = end
                break
            ch, p = read_node(p)
            if ch is None:
                break
            children.append(ch)
        return Node(name, props, children), end

    root = Node("root")
    p = 27
    while p < len(data) - null_len:
        n, p2 = read_node(p)
        if n is None:
            break
        root.children.append(n)
        p = p2
    return root, version


# ---------------------------------------------------------------- ASCII
_TOK = re.compile(r'"(?:[^"]|"")*"|;[^\n]*|[A-Za-z_][\w|:\-]*:|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|\*\d+|[{},]|[A-Za-z_][\w|\-]*')


def _parse_ascii(text):
    toks = [t for t in _TOK.findall(text) if not t.startswith(";")]
    pos = 0

    def value(t):
        if t.startswith('"'):
            return t[1:-1].replace('""', '"').encode("utf-8")
        try:
            if re.fullmatch(r"[-+]?\d+", t):
                return int(t)
            return float(t)
        except ValueError:
            return t.encode("utf-8")

    def parse_block(end_brace):
        nonlocal pos
        nodes = []
        while pos < len(toks):
            t = toks[pos]
            if t == "}":
                pos += 1
                return nodes
            if t.endswith(":") and not t.startswith('"'):
                name = t[:-1]
                pos += 1
                props, children = [], []
                while pos < len(toks):
                    t2 = toks[pos]
                    if t2 == "{":
                        pos += 1
                        children = parse_block(True)
                        break
                    if t2.endswith(":") and not t2.startswith('"') or t2 == "}":
                        break
                    if t2 == ",":
                        pos += 1
                        continue
                    if t2.startswith("*"):  # array count, values follow in a { a: ... } block
                        pos += 1
                        continue
                    props.append(value(t2))
                    pos += 1
                if name == "a" and props:
                    pass
                # arrays: Node "Vertices: *N { a: ... }" -> flatten
                if len(children) == 1 and children[0].name == "a":
                    arr = np.array(children[0].props, dtype=np.float64)
                    props = [arr]
                    children = []
                nodes.append(Node(name, props, children))
            else:
                pos += 1
        return nodes

    root = Node("root", [], parse_block(False))
    ver = 7400
    hdr = root.find("FBXHeaderExtension")
    if hdr and hdr.find("FBXVersion"):
        ver = int(hdr.find("FBXVersion").props[0])
    return root, ver


def parse(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:18] == b"Kaydara FBX Binary":
        return _parse_binary(memoryview(data))
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    if "FBXHeaderExtension" not in text[:5000] and "Objects:" not in text:
        raise FBXError("Не похоже на FBX-файл")
    if "FBXVersion: 6" in text[:3000] or "FBXVersion: 5" in text[:3000]:
        raise FBXError("Старый FBX 6.x не поддерживается — пересохраните как FBX 2013+ (7.x)")
    return _parse_ascii(text)


def dump(node, depth=0, maxdepth=3, out=None):
    out = out if out is not None else []
    if depth > maxdepth:
        return out
    pr = []
    for p in node.props[:4]:
        if isinstance(p, np.ndarray):
            pr.append("arr[%d]" % len(p))
        elif isinstance(p, bytes):
            pr.append(repr(p[:40]))
        else:
            pr.append(repr(p))
    out.append("  " * depth + node.name + " " + " ".join(pr))
    for c in node.children[:40]:
        dump(c, depth + 1, maxdepth, out)
    if len(node.children) > 40:
        out.append("  " * (depth + 1) + "... %d more" % (len(node.children) - 40))
    return out


# ================================================================ texture search
ALB_SUFFIX = ("albd", "alba", "alb", "albedo", "basecolor", "base_color", "basemap", "diffuse", "diff",
              "maintex", "color", "col", "d", "bc", "c")
NON_ALB = ("nrm", "nrmr", "normal", "norm", "nor", "n", "msk", "mskm", "mask", "atoc", "rough", "roughness",
           "metal", "metallic", "spec", "specular", "ao", "occlusion", "emi", "emission", "height", "disp",
           "lightmap", "shadow", "ramp", "orm", "rma", "mra", "m", "r", "s", "e", "h")
IMG_EXT = (".png", ".jpg", ".jpeg", ".tga", ".bmp", ".dds", ".tif", ".tiff", ".gif")


def _stem_tokens(stem):
    return [t for t in re.split(r"[_\-\s.]+", stem.lower()) if t]


class TexIndex:
    def __init__(self, base_dir):
        self.files = []
        seen = set()
        for d, depth in ((base_dir, 4), (os.path.dirname(base_dir), 3)):
            if not d or d in seen:
                continue
            seen.add(d)
            base_depth = d.rstrip(os.sep).count(os.sep)
            for dp, dn, fn in os.walk(d):
                if dp.count(os.sep) - base_depth >= depth:
                    dn[:] = []
                for f in fn:
                    if f.lower().endswith(IMG_EXT):
                        self.files.append(os.path.join(dp, f))
                if len(self.files) > 5000:
                    break
        self.by_stem = {}
        for f in self.files:
            st = os.path.splitext(os.path.basename(f))[0].lower()
            # prefer png/jpg over dds/tga with the same stem
            if st not in self.by_stem or f.lower().endswith((".png", ".jpg", ".jpeg")):
                self.by_stem[st] = f

    def find_stem(self, stem):
        return self.by_stem.get(stem.lower())

    def is_albedo(self, stem):
        toks = _stem_tokens(stem)
        return bool(toks) and toks[-1] in ALB_SUFFIX and toks[-1] not in NON_ALB

    def albedo_sibling(self, stem):
        toks = _stem_tokens(stem)
        if not toks:
            return None
        if toks[-1] in NON_ALB:
            toks = toks[:-1]
        base = "_".join(toks)
        for suf in ALB_SUFFIX:
            for cand in (base + "_" + suf, base + suf):
                f = self.find_stem(cand)
                if f:
                    return f
        return None

    def by_material(self, matname):
        mt = [t for t in _stem_tokens(matname) if t not in ("mat", "material", "mtl", "m")]
        if not mt:
            return None
        best, best_extra = None, 99
        for st, f in self.by_stem.items():
            toks = _stem_tokens(st)
            if not toks or toks[-1] not in ALB_SUFFIX:
                continue
            body = toks[:-1]

            def has(t):
                return any(t == b or (len(t.rstrip("s")) >= 3 and t.rstrip("s") == b.rstrip("s")) or (len(t) >= 4 and len(b) >= 4 and (t in b or b in t))
                           for b in body)
            if all(has(t) for t in mt):
                extra = len(body) - len(mt)
                if extra < best_extra:
                    best, best_extra = f, extra
        return best


# ================================================================ scene -> Model
def _s(b):
    return b.decode("utf-8", "replace") if isinstance(b, bytes) else str(b)


def _objname(node):
    n = _s(node.props[1]) if len(node.props) > 1 else ""
    n = n.split("\x00\x01")[0]
    if "::" in n:
        n = n.split("::", 1)[1]
    return n


def _props70(node):
    out = {}
    p70 = node.find("Properties70") or node.find("Properties60")
    if p70:
        for p in p70.children:
            if p.props:
                out[_s(p.props[0])] = p.props[4:] if len(p.props) > 4 else p.props[1:]
    return out


def _vec(props, key, default):
    v = props.get(key)
    if v is None or len(v) < 3:
        return np.array(default, float)
    try:
        return np.array([float(x) for x in v[:3]])
    except (TypeError, ValueError):
        return np.array(default, float)


def _euler(deg, order=0):
    x, y, z = np.radians(deg)
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    # FBX eEulerXYZ=0 means X applied first -> R = Rz*Ry*Rx
    seq = {0: (Rx, Ry, Rz), 1: (Rx, Rz, Ry), 2: (Ry, Rz, Rx), 3: (Ry, Rx, Rz), 4: (Rz, Rx, Ry),
           5: (Rz, Ry, Rx)}.get(int(order), (Rx, Ry, Rz))
    R = np.eye(3)
    for m in seq:
        R = m @ R
    M = np.eye(4)
    M[:3, :3] = R
    return M


def _T(v):
    M = np.eye(4)
    M[:3, 3] = v
    return M


def _S(v):
    M = np.eye(4)
    M[0, 0], M[1, 1], M[2, 2] = v
    return M


def _local_matrix(props):
    t = _vec(props, "Lcl Translation", (0, 0, 0))
    r = _vec(props, "Lcl Rotation", (0, 0, 0))
    s = _vec(props, "Lcl Scaling", (1, 1, 1))
    order = int(props.get("RotationOrder", [0])[0]) if props.get("RotationOrder") else 0
    try:
        active = int(props["RotationActive"][0]) if props.get("RotationActive") else 1
    except (TypeError, ValueError):
        active = 1
    if active:
        pre = _euler(_vec(props, "PreRotation", (0, 0, 0)))
        post = _euler(_vec(props, "PostRotation", (0, 0, 0)))
    else:
        order = 0
        pre = post = np.eye(4)
    roff = _T(_vec(props, "RotationOffset", (0, 0, 0)))
    rp = _T(_vec(props, "RotationPivot", (0, 0, 0)))
    soff = _T(_vec(props, "ScalingOffset", (0, 0, 0)))
    sp = _T(_vec(props, "ScalingPivot", (0, 0, 0)))
    return (_T(t) @ roff @ rp @ pre @ _euler(r, order) @ np.linalg.inv(post) @ np.linalg.inv(rp)
            @ soff @ sp @ _S(s) @ np.linalg.inv(sp))


def _geo_matrix(props):
    return (_T(_vec(props, "GeometricTranslation", (0, 0, 0))) @
            _euler(_vec(props, "GeometricRotation", (0, 0, 0))) @
            _S(_vec(props, "GeometricScaling", (1, 1, 1))))


def _mat16(arr):
    return np.array(arr, float).reshape(4, 4).T  # FBX stores column-major


def _layer(geom, name, data_key, index_key, poly_corner_cp, n_corners, n_polys, poly_of_corner, width):
    le = geom.find(name)
    if le is None:
        return None
    data = le.find(data_key)
    if data is None or not len(data.props):
        return None
    vals = np.asarray(data.props[0], float).reshape(-1, width)
    mapping = _s(le.find("MappingInformationType").props[0]) if le.find("MappingInformationType") else "ByPolygonVertex"
    ref = _s(le.find("ReferenceInformationType").props[0]) if le.find("ReferenceInformationType") else "Direct"
    idx_node = le.find(index_key)
    if mapping in ("ByPolygonVertex",):
        sel = np.arange(n_corners)
    elif mapping in ("ByVertice", "ByVertex", "ByControlPoint"):
        sel = poly_corner_cp
    elif mapping == "ByPolygon":
        sel = poly_of_corner
    elif mapping == "AllSame":
        sel = np.zeros(n_corners, int)
    else:
        sel = np.arange(n_corners)
    if ref == "IndexToDirect" and idx_node is not None and len(idx_node.props):
        ind = np.asarray(idx_node.props[0], np.int64)
        sel = ind[np.clip(sel, 0, len(ind) - 1)]
    sel = np.clip(sel, 0, len(vals) - 1)
    return vals[sel]


def load(path, log=print):
    root, version = parse(path)
    objs = root.find("Objects")
    if objs is None:
        raise FBXError("В FBX нет объектов")
    byid = {}
    for o in objs.children:
        if o.props:
            byid[o.props[0]] = o
    children_of = {}   # parent id -> [(child id, prop)]
    parents_of = {}    # child id -> [(parent id, prop)]
    conns = root.find("Connections")
    for c in (conns.children if conns else []):
        if len(c.props) < 3:
            continue
        kind = _s(c.props[0])
        ch, par = c.props[1], c.props[2]
        prop = _s(c.props[3]) if len(c.props) > 3 else ""
        children_of.setdefault(par, []).append((ch, prop if kind == "OP" else ""))
        parents_of.setdefault(ch, []).append((par, prop if kind == "OP" else ""))

    # ---- global settings: axes & units
    gs = _props70(root.find("GlobalSettings")) if root.find("GlobalSettings") else {}

    def gi(k, d):
        v = gs.get(k)
        try:
            return int(v[0]) if v else d
        except (TypeError, ValueError):
            return d
    up, ups = gi("UpAxis", 1), gi("UpAxisSign", 1)
    fr, frs = gi("FrontAxis", 2), gi("FrontAxisSign", 1)
    co, cos_ = gi("CoordAxis", 0), gi("CoordAxisSign", 1)
    usf = float(gs.get("UnitScaleFactor", [1.0])[0]) if gs.get("UnitScaleFactor") else 1.0
    E = np.eye(3)
    A = np.stack([E[co] * cos_, E[up] * ups, E[fr] * frs])  # file -> canonical (X right, Y up, Z front)
    if abs(np.linalg.det(A)) < 0.5:
        A = np.eye(3)
    to_m = usf / 100.0          # file units -> meters
    k_mmd = to_m / 0.08         # meters -> "MMD units" (converter default scale 0.08)

    def is_type(o, t):
        return len(o.props) > 2 and _s(o.props[2]) == t

    models = {i: o for i, o in byid.items() if o.name == "Model"}
    mprops = {i: _props70(o) for i, o in models.items()}

    def model_parent(i):
        for p, _ in parents_of.get(i, []):
            if p in models:
                return p
        return None

    local = {i: _local_matrix(mprops[i]) for i in models}
    gcache = {}

    def global_m(i, depth=0):
        if i in gcache:
            return gcache[i]
        p = model_parent(i)
        g = local[i] if p is None or depth > 200 else global_m(p, depth + 1) @ local[i]
        gcache[i] = g
        return g

    # ---- bind pose matrices
    bind = {}
    for o in objs.children:
        if o.name == "Pose" and is_type(o, "BindPose"):
            for pn in o.find_all("PoseNode"):
                nid = pn.find("Node").props[0] if pn.find("Node") else None
                mt = pn.find("Matrix")
                if nid is not None and mt is not None and len(mt.props):
                    bind[nid] = _mat16(mt.props[0])
    clusters = {i: o for i, o in byid.items() if o.name == "Deformer" and is_type(o, "Cluster")}
    for ci, cl in clusters.items():
        bone = next((c for c, _ in children_of.get(ci, []) if c in models), None)
        tl = cl.find("TransformLink")
        if bone is not None and tl is not None and len(tl.props):
            bind[bone] = _mat16(tl.props[0])

    # ---- bones: every Model except meshes/cameras/lights
    limb = {i for i, o in models.items() if _s(o.props[2]) in ("LimbNode", "Root")}
    for ci in clusters:  # any node a skin cluster points to is a bone, whatever its type
        for c, _ in children_of.get(ci, []):
            if c in models:
                limb.add(c)
    keep = set(limb)
    for i in limb:
        p = model_parent(i)
        while p is not None and p not in keep:
            keep.add(p)
            p = model_parent(p)
    if not limb:  # no skeleton: keep all non-mesh nodes
        keep = {i for i, o in models.items() if _s(o.props[2]) not in ("Mesh", "Camera", "Light")}
    bone_ids = [i for i in models if i in keep and _s(models[i].props[2]) not in ("Mesh", "Camera", "Light")]
    bone_index = {i: k for k, i in enumerate(bone_ids)}
    bind_g = {}

    def bind_global(i, depth=0):
        if i in bind_g:
            return bind_g[i]
        if i in bind:
            g = bind[i]
        else:
            p = model_parent(i)
            if p is not None and depth < 200:
                g = bind_global(p, depth + 1) @ local[i]
            else:
                g = global_m(i)
        bind_g[i] = g
        return g

    def to_mmd(v):  # (n,3) file space -> MMD space
        c = v @ A.T
        c = c * k_mmd
        c[:, 2] *= -1.0    # canonical RH facing +Z -> MMD LH facing -Z
        return c

    def to_mmd_dir(v):
        c = v @ A.T
        c[:, 2] *= -1.0
        return c

    # ---- geometry
    P_all, N_all, UV_all, J_all, W_all = [], [], [], [], []
    tri_by_mat = {}          # material id -> [triangles (global vertex idx)]
    morph_acc = {}           # name -> [(idx, offsets)]
    mat_order = []
    vbase = 0
    meshes = [(i, o) for i, o in models.items() if _s(o.props[2]) == "Mesh"]
    for mid, mo in meshes:
        geom = next((byid[c] for c, _ in children_of.get(mid, []) if c in byid and byid[c].name == "Geometry"), None)
        if geom is None or geom.find("Vertices") is None:
            continue
        cp = np.asarray(geom.find("Vertices").props[0], float).reshape(-1, 3)
        pvi = np.asarray(geom.find("PolygonVertexIndex").props[0], np.int64)
        ends = pvi < 0
        corner_cp = np.where(ends, ~pvi, pvi)
        n_c = len(corner_cp)
        poly_id = np.concatenate([[0], np.cumsum(ends)[:-1]]).astype(np.int64)
        n_p = int(ends.sum())
        # skin
        skin = None
        for c, _ in children_of.get(geom.props[0], []):
            if c in byid and byid[c].name == "Deformer" and is_type(byid[c], "Skin"):
                skin = c
        infl = [[] for _ in range(len(cp))]
        mesh_bind = None
        if skin is not None:
            for ci, _ in children_of.get(skin, []):
                cl = clusters.get(ci)
                if cl is None:
                    continue
                bone = next((c for c, _ in children_of.get(ci, []) if c in bone_index), None)
                if bone is None or cl.find("Indexes") is None:
                    continue
                ix = np.asarray(cl.find("Indexes").props[0], np.int64)
                wt = np.asarray(cl.find("Weights").props[0], float)
                bi = bone_index[bone]
                for a, w in zip(ix, wt):
                    if 0 <= a < len(cp) and w > 0:
                        infl[a].append((w, bi))
                if mesh_bind is None and cl.find("Transform") is not None and cl.find("TransformLink") is not None:
                    mesh_bind = _mat16(cl.find("TransformLink").props[0]) @ _mat16(cl.find("Transform").props[0])
        mp = mprops[mid]
        if mesh_bind is None:
            mesh_bind = (bind[mid] if mid in bind else global_m(mid)) @ _geo_matrix(mp)
        # rigid attachment: mesh parented to a bone, no skin
        parent_bone = model_parent(mid)
        rigid_bone = bone_index.get(parent_bone, 0) if parent_bone in bone_index else (0 if bone_ids else -1)
        # world-space control points
        cpw = (np.c_[cp, np.ones(len(cp))] @ mesh_bind.T)[:, :3]
        lin = mesh_bind[:3, :3]
        # per corner attributes
        nrm = _layer(geom, "LayerElementNormal", "Normals", "NormalsIndex", corner_cp, n_c, n_p, poly_id, 3)
        uvs = _layer(geom, "LayerElementUV", "UV", "UVIndex", corner_cp, n_c, n_p, poly_id, 2)
        if nrm is None:
            nrm = np.tile([0.0, 1.0, 0.0], (n_c, 1))
        if uvs is None:
            uvs = np.zeros((n_c, 2))
        nrm_w = nrm @ np.linalg.inv(lin) if abs(np.linalg.det(lin)) > 1e-12 else nrm
        # dedupe corners into vertices
        key = np.c_[corner_cp, np.round(uvs * 4096), np.round(nrm * 256)].astype(np.int64)
        uniq, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        vcp = corner_cp[first]
        P_all.append(to_mmd(cpw[vcp]))
        nn = to_mmd_dir(nrm_w[first])
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
        N_all.append(nn)
        uv = uvs[first].copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        UV_all.append(uv)
        J = np.full((len(vcp), 4), -1, np.int32)
        W = np.zeros((len(vcp), 4), np.float32)
        for vi, c in enumerate(vcp):
            lst = infl[c]
            if not lst:
                if rigid_bone >= 0:
                    J[vi, 0], W[vi, 0] = rigid_bone, 1.0
                continue
            lst = sorted(lst, reverse=True)[:4]
            s = sum(w for w, _ in lst)
            for k, (w, b) in enumerate(lst):
                J[vi, k], W[vi, k] = b, w / s
        J_all.append(J)
        W_all.append(W)
        # materials of this mesh
        mats = [c for c, _ in children_of.get(mid, []) if c in byid and byid[c].name == "Material"]
        for mm in mats:
            if mm not in mat_order:
                mat_order.append(mm)
        lem = geom.find("LayerElementMaterial")
        if lem is not None and lem.find("Materials") is not None and mats:
            pm = np.asarray(lem.find("Materials").props[0], np.int64)
            mapping = _s(lem.find("MappingInformationType").props[0]) if lem.find("MappingInformationType") else "ByPolygon"
            if mapping == "AllSame" or len(pm) < n_p:
                poly_mat = np.full(n_p, int(pm[0]) if len(pm) else 0)
            else:
                poly_mat = pm[:n_p]
        else:
            poly_mat = np.zeros(n_p, np.int64)
        if not mats:
            mats = ["__default__"]
            if "__default__" not in mat_order:
                mat_order.append("__default__")
        # triangulate (fan) per polygon
        vidx = inv + vbase
        starts = np.concatenate([[0], np.nonzero(ends)[0][:-1] + 1])
        counts = np.nonzero(ends)[0] - starts + 1
        tris, tmat = [], []
        for nverts in np.unique(counts):
            if nverts < 3:
                continue
            sel = np.nonzero(counts == nverts)[0]
            st = starts[sel]
            for k in range(1, nverts - 1):
                # FBX is right-handed CCW; mirroring Z into MMD space flips the winding
                tris.append(np.stack([vidx[st], vidx[st + k + 1], vidx[st + k]], axis=1))
                tmat.append(poly_mat[sel])
        if tris:
            tris = np.concatenate(tris)
            tmat = np.concatenate(tmat)
            for li in np.unique(tmat):
                gm = mats[int(li)] if 0 <= li < len(mats) else mats[0]
                tri_by_mat.setdefault(gm, []).append(tris[tmat == li])
        # blend shapes
        for bs, _ in children_of.get(geom.props[0], []):
            if bs not in byid or not is_type(byid[bs], "BlendShape"):
                continue
            for ch, _ in children_of.get(bs, []):
                chn = byid.get(ch)
                if chn is None:
                    continue
                cname = _objname(chn)
                if "." in cname:
                    cname = cname.split(".")[-1]  # 'blendShape1.mouth_a' -> 'mouth_a'
                shapes = [byid.get(sh) for sh, _ in children_of.get(ch, [])]
                shapes = [x for x in shapes if x is not None and x.find("Indexes") is not None]
                fw = chn.find("FullWeights")
                if fw is not None and len(fw.props) and len(shapes) > 1:
                    wts = list(np.asarray(fw.props[0], float))
                    if len(wts) == len(shapes):
                        shapes = [shapes[int(np.argmax(wts))]]
                for shp in shapes[-1:]:
                    six = np.asarray(shp.find("Indexes").props[0], np.int64)
                    sv = np.asarray(shp.find("Vertices").props[0], float).reshape(-1, 3)
                    d_cp = np.zeros((len(cp), 3))
                    ok = (six >= 0) & (six < len(cp))
                    d_cp[six[ok]] = sv[ok]
                    d_v = d_cp[vcp] @ lin.T
                    d_v = to_mmd_dir(d_v) * k_mmd
                    nz = np.abs(d_v).max(axis=1) > 1e-7
                    if nz.any():
                        morph_acc.setdefault(cname or _objname(shp), []).append(
                            (np.nonzero(nz)[0] + vbase, d_v[nz]))
                    break  # first (full-weight) shape only
        vbase += len(vcp)

    if not P_all:
        raise FBXError("В FBX нет полигональных сеток")

    m = pmx.Model()
    m.version = 0
    m.name = os.path.splitext(os.path.basename(path))[0]
    m.comment = "FBX %d" % version
    m.positions = np.concatenate(P_all).astype(np.float32)
    m.normals = np.concatenate(N_all).astype(np.float32)
    m.uvs = np.concatenate(UV_all).astype(np.float32)
    m.joints = np.concatenate(J_all)
    m.weights = np.concatenate(W_all)

    # ---- bones
    for i in bone_ids:
        b = pmx.Bone()
        b.name = _objname(models[i]) or "bone_%d" % bone_index[i]
        b.name_en = ""
        g = bind_global(i)
        b.pos = to_mmd(g[:3, 3][None])[0].astype(float)
        p = model_parent(i)
        b.parent = bone_index.get(p, -1)
        b.layer = 0
        b.flags = 0x0002 | 0x0008 | 0x0010
        m.bones.append(b)

    # ---- facing: character's left must end up at +X in MMD space
    from converter import _en_key  # late import, avoids a cycle
    lx, rx = [], []
    for b in m.bones:
        side, _w = _en_key(b.name)
        if b.name.startswith("左"):
            side = "left"
        elif b.name.startswith("右"):
            side = "right"
        if side == "left":
            lx.append(b.pos[0])
        elif side == "right":
            rx.append(b.pos[0])
    if lx and rx and np.mean(lx) < np.mean(rx):
        flip = np.array([-1.0, 1.0, -1.0])
        m.positions *= flip.astype(np.float32)
        m.normals *= flip.astype(np.float32)
        for b in m.bones:
            b.pos = b.pos * flip
        for k in morph_acc:
            morph_acc[k] = [(ix, off * flip) for ix, off in morph_acc[k]]
        log("  FBX: модель повёрнута на 180° (смотрела назад)")

    # ---- sanity: up axis and size (game rips often lie in GlobalSettings)
    P = m.positions
    ext = P.max(0) - P.min(0)
    heads_ = [b.pos for b in m.bones if _en_key(b.name)[1] == "head" or b.name == "頭"]
    feet_ = [b.pos for b in m.bones if _en_key(b.name)[1] in ("foot", "ankle") or b.name.endswith("足首")]
    lying = ext[2] > 1.6 * ext[1]
    if heads_ and feet_:
        up_v = np.mean(heads_, axis=0) - np.mean(feet_, axis=0)
        lying = abs(up_v[2]) > 2.0 * abs(up_v[1])
    if lying:
        # lying down: Z is really up -> rotate so that +Z becomes +Y (try both signs, keep head up)
        def rot(v, sgn):
            out = v.copy()
            out[..., 1], out[..., 2] = sgn * v[..., 2], -sgn * v[..., 1]
            return out
        heads = [b for b in m.bones if _en_key(b.name)[1] in ("head",) or b.name == "頭"]
        sgn = 1.0
        if heads:
            sgn = 1.0 if heads[0].pos[2] > P[:, 2].mean() else -1.0
        m.positions = rot(m.positions, sgn).astype(np.float32)
        m.normals = rot(m.normals, sgn).astype(np.float32)
        for b in m.bones:
            b.pos = rot(b.pos[None], sgn)[0]
        for k in morph_acc:
            morph_acc[k] = [(ix, rot(off, sgn)) for ix, off in morph_acc[k]]
        log("  FBX: ось «вверх» в файле указана неверно — модель поставлена вертикально")
    height_m = float(m.positions[:, 1].max() - m.positions[:, 1].min()) * 0.08
    fix = 1.0
    if height_m > 30:
        fix = 0.01
    elif height_m < 0.05:
        fix = 100.0
    if fix != 1.0:
        m.positions *= fix
        for b in m.bones:
            b.pos = b.pos * fix
        for k in morph_acc:
            morph_acc[k] = [(ix, off * fix) for ix, off in morph_acc[k]]
        log("  FBX: масштаб в файле неверный (рост %.2f м) — исправлен ×%g" % (height_m, fix))
    # put feet on the ground
    dy = -float(m.positions[:, 1].min())
    if abs(dy) > 1e-3:
        m.positions[:, 1] += dy
        for b in m.bones:
            b.pos = b.pos + np.array([0, dy, 0])

    # ---- materials & textures
    tex_index = {}
    m.texture_blobs = {}
    base_dir = os.path.dirname(os.path.abspath(path))

    tindex = TexIndex(base_dir)

    def add_tex_file(f):
        key = ("file", os.path.normcase(f))
        if key in tex_index:
            return tex_index[key]
        ti = len(m.textures)
        m.textures.append(f)
        tex_index[key] = ti
        return ti

    def texture_for(mat_id):
        mname = _objname(byid[mat_id])
        cands = []
        for c, prop in children_of.get(mat_id, []):
            t = byid.get(c)
            if t is not None and t.name == "Texture":
                cands.append((prop, c, t))
        if not cands:
            f = tindex.by_material(mname)
            return add_tex_file(f) if f else None

        def score(x):
            prop, _, t = x
            pl = prop.lower()
            nm = (_s(t.find("RelativeFilename").props[0]) if t.find("RelativeFilename") else "").lower()
            s = 0
            if pl in ("diffusecolor", "diffuse", "maya|basecolor", "base_color", "basecolor", "maya|TEX_color_map".lower()):
                s += 10
            if any(w in nm for w in ("alb", "basecolor", "base_color", "diffuse", "_d.", "_col", "color")):
                s += 5
            if any(w in nm for w in ("nrm", "normal", "_n.", "rough", "metal", "mask", "msk", "spec", "ao")):
                s -= 8
            return s
        prop, tid, t = max(cands, key=score)
        if tid in tex_index:
            return tex_index[tid]
        rel = _s(t.find("RelativeFilename").props[0]) if t.find("RelativeFilename") else ""
        absf = _s(t.find("FileName").props[0]) if t.find("FileName") else ""
        tp = _props70(t)
        if not rel and tp.get("Path"):
            absf = _s(tp["Path"][0])
        stem = os.path.splitext(os.path.basename((rel or absf).replace("\\", "/")))[0]
        has_blob = any(byid.get(v) is not None and byid[v].name == "Video" and byid[v].find("Content") is not None
                       and byid[v].find("Content").props and len(byid[v].find("Content").props[0] or b"") > 16
                       for v, _ in children_of.get(tid, []))
        direct = score((prop, tid, t)) >= 10 and (has_blob or (absf and os.path.isfile(absf))) \
            and (_stem_tokens(stem)[-1:] or [""])[0] not in NON_ALB
        if not direct and (score((prop, tid, t)) < 5 or not tindex.is_albedo(stem)):
            # connected map is a normal/mask map: look for the colour map next to it
            f = tindex.albedo_sibling(stem) or tindex.by_material(mname)
            if f:
                return add_tex_file(f)
            if score((prop, tid, t)) < 0 and not has_blob:
                return None
        if not has_blob:
            f = tindex.find_stem(stem)
            if f and not (absf and os.path.isfile(absf)):
                ti = add_tex_file(f)
                tex_index[tid] = ti
                return ti
        ti = len(m.textures)
        if absf and os.path.isfile(absf):
            m.textures.append(os.path.relpath(absf, base_dir) if os.path.splitdrive(absf)[0] == os.path.splitdrive(base_dir)[0] else absf)
        else:
            m.textures.append(rel or os.path.basename(absf.replace("\\", "/")))
        for v, _ in children_of.get(tid, []):
            vo = byid.get(v)
            if vo is not None and vo.name == "Video" and vo.find("Content") is not None:
                content = vo.find("Content").props[0] if vo.find("Content").props else b""
                if isinstance(content, (bytes, bytearray)) and len(content) > 16:
                    m.texture_blobs[ti] = bytes(content)
        tex_index[tid] = ti
        return ti

    idx_parts = []
    for mm in mat_order:
        tris = tri_by_mat.get(mm)
        if not tris:
            continue
        tri = np.concatenate(tris)
        mt = pmx.Material()
        if mm == "__default__":
            mt.name, props, ti = "default", {}, None
        else:
            mt.name = _objname(byid[mm]) or "material"
            props = _props70(byid[mm])
            ti = texture_for(mm)
        dc = _vec(props, "DiffuseColor", _vec(props, "Diffuse", (0.8, 0.8, 0.8)))
        if ti is not None:
            dc = np.array([1.0, 1.0, 1.0])
        op = 1.0
        if props.get("Opacity") and False:
            try:
                op = float(props["Opacity"][0])
            except (TypeError, ValueError):
                pass
        try:
            if props.get("Opacity"):
                op = float(props["Opacity"][0])
            elif props.get("TransparencyFactor"):
                tf = float(props["TransparencyFactor"][0])
                tc = _vec(props, "TransparentColor", (1, 1, 1))
                op = 1.0 - tf * float(np.mean(tc))
        except (TypeError, ValueError):
            op = 1.0
        if not (0.05 <= op <= 1.0):
            op = 1.0
        mt.name_en = ""
        mt.diffuse = (float(dc[0]), float(dc[1]), float(dc[2]), float(op))
        mt.specular = (0.0, 0.0, 0.0)
        mt.spec_power = 1.0
        mt.ambient = tuple(float(c) * 0.4 for c in dc)
        mt.flags = 0x01  # double sided: game rips often rely on it
        mt.edge_color = (0, 0, 0, 1)
        mt.edge_size = 0.0
        mt.texture = ti if ti is not None else -1
        mt.sphere = -1
        mt.sphere_mode = 0
        mt.toon_shared = 1
        mt.toon = 0
        mt.memo = ""
        mt.index_count = int(tri.size)
        m.materials.append(mt)
        idx_parts.append(tri.reshape(-1))
    m.indices = np.concatenate(idx_parts).astype(np.int64) if idx_parts else np.zeros(0, np.int64)

    for name, parts in morph_acc.items():
        mo = pmx.Morph()
        mo.name, mo.name_en, mo.panel, mo.type = name, "", 4, 1
        ix = np.concatenate([p[0] for p in parts])
        off = np.concatenate([p[1] for p in parts]).astype(np.float32)
        mo.data = (ix, off)
        m.morphs.append(mo)
    m.rigid_bodies = []
    m.suggested_scale = 0.08
    m.source_format = "fbx"
    log("  FBX %d: сеток %d, вершин %d, костей %d, материалов %d, выражений %d, ед.: %.3g м"
        % (version, len(meshes), len(m.positions), len(m.bones), len(m.materials), len(m.morphs), to_m))
    return m
