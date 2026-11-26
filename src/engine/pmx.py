"""PMX 2.0 / 2.1 reader (MikuMikuDance model format)."""
import struct
import numpy as np


class PMXError(Exception):
    pass


class Bone:
    __slots__ = ("name", "name_en", "pos", "parent", "layer", "flags",
                 "tail_bone", "tail_offset", "grant_parent", "grant_weight",
                 "fixed_axis", "local_x", "local_z", "ik")

    def __init__(self):
        self.tail_bone = -1
        self.tail_offset = None
        self.grant_parent = -1
        self.grant_weight = 0.0
        self.fixed_axis = None
        self.local_x = self.local_z = None
        self.ik = None


class Material:
    __slots__ = ("name", "name_en", "diffuse", "specular", "spec_power", "ambient",
                 "flags", "edge_color", "edge_size", "texture", "sphere",
                 "sphere_mode", "toon_shared", "toon", "memo", "index_count")


class Morph:
    __slots__ = ("name", "name_en", "panel", "type", "data")
    # type: 0 group, 1 vertex, 2 bone, 3 uv, 4-7 add uv, 8 material, 9 flip, 10 impulse
    # data for vertex: (indices np.int64, offsets np.float32 (k,3))
    # data for group/flip: list of (morph_index, weight)
    # data for bone: list of (bone_index, (tx,ty,tz), (qx,qy,qz,qw))
    # other types: None (skipped)


class RigidBody:
    __slots__ = ("name", "name_en", "bone", "group", "mask", "shape", "size",
                 "pos", "rot", "mass", "lin_damp", "ang_damp", "restitution",
                 "friction", "mode")


class Model:
    def __init__(self):
        self.version = 2.0
        self.name = ""
        self.name_en = ""
        self.comment = ""
        self.comment_en = ""
        self.positions = None   # (N,3) float32
        self.normals = None     # (N,3)
        self.uvs = None         # (N,2)
        self.joints = None      # (N,4) int32 (-1 = none)
        self.weights = None     # (N,4) float32
        self.indices = None     # (M,) int64
        self.textures = []
        self.materials = []
        self.bones = []
        self.morphs = []
        self.rigid_bodies = []


class _R:
    def __init__(self, data):
        self.d = data
        self.p = 0

    def unpack(self, fmt):
        v = struct.unpack_from(fmt, self.d, self.p)
        self.p += struct.calcsize(fmt)
        return v

    def i32(self):
        v = struct.unpack_from("<i", self.d, self.p)[0]
        self.p += 4
        return v

    def u8(self):
        v = self.d[self.p]
        self.p += 1
        return v

    def f32(self):
        v = struct.unpack_from("<f", self.d, self.p)[0]
        self.p += 4
        return v

    def vec(self, n):
        v = struct.unpack_from("<%df" % n, self.d, self.p)
        self.p += 4 * n
        return v

    def raw(self, n):
        v = self.d[self.p:self.p + n]
        self.p += n
        return v


def _idx_fmt(size, unsigned):
    if size == 1:
        return "<B" if unsigned else "<b"
    if size == 2:
        return "<H" if unsigned else "<h"
    if size == 4:
        return "<i"
    raise PMXError("bad index size %d" % size)


def load(path):
    with open(path, "rb") as f:
        data = f.read()
    return loads(data)


def loads(data):
    r = _R(data)
    if data[:4] != b"PMX ":
        if data[:3] == b"Pmd":
            return loads_pmd(data)
        raise PMXError("Не PMX/PMD-файл")
    r.p = 4
    m = Model()
    m.version = r.f32()
    nglob = r.u8()
    g = list(r.raw(nglob))
    enc = "utf-16-le" if g[0] == 0 else "utf-8"
    add_uv = g[1]
    vsz, tsz, msz, bsz, mosz, rsz = g[2:8]

    vfmt = _idx_fmt(vsz, True)
    tfmt = _idx_fmt(tsz, False)
    mfmt = _idx_fmt(msz, False)
    bfmt = _idx_fmt(bsz, False)
    mofmt = _idx_fmt(mosz, False)
    rfmt = _idx_fmt(rsz, False)
    vS = struct.calcsize(vfmt)
    tS = struct.calcsize(tfmt)
    bS = struct.calcsize(bfmt)
    moS = struct.calcsize(mofmt)
    rS = struct.calcsize(rfmt)
    mS = struct.calcsize(mfmt)

    def text():
        n = r.i32()
        if n <= 0:
            return ""
        return r.raw(n).decode(enc, errors="replace")

    def vidx():
        v = struct.unpack_from(vfmt, data, r.p)[0]
        r.p += vS
        return v

    def tidx():
        v = struct.unpack_from(tfmt, data, r.p)[0]
        r.p += tS
        return v

    def bidx():
        v = struct.unpack_from(bfmt, data, r.p)[0]
        r.p += bS
        return v

    def moidx():
        v = struct.unpack_from(mofmt, data, r.p)[0]
        r.p += moS
        return v

    def ridx():
        v = struct.unpack_from(rfmt, data, r.p)[0]
        r.p += rS
        return v

    def midx():
        v = struct.unpack_from(mfmt, data, r.p)[0]
        r.p += mS
        return v

    m.name = text()
    m.name_en = text()
    m.comment = text()
    m.comment_en = text()

    # ---- vertices ----
    nv = r.i32()
    if nv < 0 or nv * 38 > len(r.d) - r.p:  # a vertex takes at least 38 bytes
        raise PMXError("Повреждённый PMX: неверное число вершин (%d)" % nv)
    pos = np.zeros((nv, 3), np.float32)
    nor = np.zeros((nv, 3), np.float32)
    uv = np.zeros((nv, 2), np.float32)
    jo = np.full((nv, 4), -1, np.int32)
    we = np.zeros((nv, 4), np.float32)
    base = struct.Struct("<8f")
    skip_add = 16 * add_uv
    b2 = struct.Struct("<" + bfmt[1] * 2 + "f")
    b4 = struct.Struct("<" + bfmt[1] * 4 + "4f")
    b1 = struct.Struct("<" + bfmt[1])
    for i in range(nv):
        v = base.unpack_from(data, r.p)
        r.p += 32 + skip_add
        pos[i] = v[0:3]
        nor[i] = v[3:6]
        uv[i] = v[6:8]
        t = data[r.p]
        r.p += 1
        if t == 0:
            jo[i, 0] = b1.unpack_from(data, r.p)[0]
            we[i, 0] = 1.0
            r.p += b1.size
        elif t == 1 or t == 3:
            a, b, w = b2.unpack_from(data, r.p)
            r.p += b2.size
            jo[i, 0] = a
            jo[i, 1] = b
            we[i, 0] = w
            we[i, 1] = 1.0 - w
            if t == 3:
                r.p += 36  # SDEF C, R0, R1 (approximated as BDEF2)
        elif t == 2 or t == 4:
            vals = b4.unpack_from(data, r.p)
            r.p += b4.size
            jo[i] = vals[0:4]
            we[i] = vals[4:8]
        else:
            raise PMXError("Неизвестный тип весов вершины %d" % t)
        r.p += 4  # edge scale
    m.positions, m.normals, m.uvs, m.joints, m.weights = pos, nor, uv, jo, we

    # ---- faces ----
    nf = r.i32()
    dt = {1: np.uint8, 2: np.uint16, 4: np.int32}[vsz]
    m.indices = np.frombuffer(data, dtype=dt, count=nf, offset=r.p).astype(np.int64)
    r.p += nf * vsz

    # ---- textures ----
    nt = r.i32()
    m.textures = [text() for _ in range(nt)]

    # ---- materials ----
    nm = r.i32()
    for _ in range(nm):
        mt = Material()
        mt.name = text()
        mt.name_en = text()
        mt.diffuse = r.vec(4)
        mt.specular = r.vec(3)
        mt.spec_power = r.f32()
        mt.ambient = r.vec(3)
        mt.flags = r.u8()
        mt.edge_color = r.vec(4)
        mt.edge_size = r.f32()
        mt.texture = tidx()
        mt.sphere = tidx()
        mt.sphere_mode = r.u8()
        mt.toon_shared = r.u8()
        mt.toon = r.u8() if mt.toon_shared else tidx()
        mt.memo = text()
        mt.index_count = r.i32()
        m.materials.append(mt)

    # ---- bones ----
    nb = r.i32()
    for _ in range(nb):
        b = Bone()
        b.name = text()
        b.name_en = text()
        b.pos = np.array(r.vec(3), np.float64)
        b.parent = bidx()
        b.layer = r.i32()
        b.flags = r.unpack("<H")[0]
        if b.flags & 0x0001:
            b.tail_bone = bidx()
        else:
            b.tail_offset = r.vec(3)
        if b.flags & (0x0100 | 0x0200):
            b.grant_parent = bidx()
            b.grant_weight = r.f32()
        if b.flags & 0x0400:
            b.fixed_axis = r.vec(3)
        if b.flags & 0x0800:
            b.local_x = r.vec(3)
            b.local_z = r.vec(3)
        if b.flags & 0x2000:
            r.i32()
        if b.flags & 0x0020:
            target = bidx()
            loops = r.i32()
            limit = r.f32()
            links = []
            for _l in range(r.i32()):
                lb = bidx()
                has = r.u8()
                lim = None
                if has:
                    lim = (r.vec(3), r.vec(3))
                links.append((lb, lim))
            b.ik = (target, loops, limit, links)
        m.bones.append(b)

    # ---- morphs ----
    nmo = r.i32()
    for _ in range(nmo):
        mo = Morph()
        mo.name = text()
        mo.name_en = text()
        mo.panel = r.u8()
        mo.type = r.u8()
        cnt = r.i32()
        mo.data = None
        t = mo.type
        if t == 1:
            st = struct.Struct("<" + vfmt[1] + "3f")
            idx = np.empty(cnt, np.int64)
            off = np.empty((cnt, 3), np.float32)
            for k in range(cnt):
                a = st.unpack_from(data, r.p)
                r.p += st.size
                idx[k] = a[0]
                off[k] = a[1:4]
            mo.data = (idx, off)
        elif t == 0 or t == 9:
            lst = []
            for k in range(cnt):
                mi = moidx()
                w = r.f32()
                lst.append((mi, w))
            mo.data = lst
        elif t == 2:
            st = struct.Struct("<" + bfmt[1] + "7f")
            lst = []
            for k in range(cnt):
                a = st.unpack_from(data, r.p)
                r.p += st.size
                lst.append((a[0], a[1:4], a[4:8]))
            mo.data = lst
        elif 3 <= t <= 7:
            r.p += cnt * (vS + 16)
        elif t == 8:
            lst = []
            st = struct.Struct("<" + mfmt[1] + "B28f")
            for k in range(cnt):
                v = st.unpack_from(data, r.p)
                r.p += st.size
                # (material or -1 = all, op 0 mul / 1 add, diffuse4, specular3, power, ambient3,
                #  edge4, edge size, texture tint4, sphere tint4, toon tint4)
                lst.append({"mat": v[0], "op": v[1], "diffuse": v[2:6], "ambient": v[10:13],
                            "tex": v[18:22]})
            mo.data = lst
        elif t == 10:
            r.p += cnt * (rS + 1 + 24)
        else:
            raise PMXError("Неизвестный тип морфа %d" % t)
        m.morphs.append(mo)

    # ---- display frames ----
    try:
        nd = r.i32()
        for _ in range(nd):
            text(); text()
            r.u8()
            for _e in range(r.i32()):
                et = r.u8()
                if et == 0:
                    r.p += bS
                else:
                    r.p += moS
    except struct.error:
        return m

    # ---- rigid bodies ----
    try:
        nr = r.i32()
        for _ in range(nr):
            rb = RigidBody()
            rb.name = text()
            rb.name_en = text()
            rb.bone = bidx()
            rb.group = r.u8()
            rb.mask = r.unpack("<H")[0]
            rb.shape = r.u8()
            rb.size = np.array(r.vec(3), np.float64)
            rb.pos = np.array(r.vec(3), np.float64)
            rb.rot = np.array(r.vec(3), np.float64)
            rb.mass, rb.lin_damp, rb.ang_damp, rb.restitution, rb.friction = r.vec(5)
            rb.mode = r.u8()
            m.rigid_bodies.append(rb)
    except struct.error:
        pass
    # joints are not needed
    return m


# ======================================================================== PMD
def _sj(b):
    b = bytes(b).split(b"\x00", 1)[0]
    for enc in ("shift_jis", "cp932", "gbk"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("shift_jis", "replace")


def loads_pmd(data):
    """PMD (MikuMikuDance 1.x) -> Model with the same fields as a PMX model."""
    p = [3]

    def un(fmt):
        v = struct.unpack_from("<" + fmt, data, p[0])
        p[0] += struct.calcsize("<" + fmt)
        return v

    def raw(n):
        v = data[p[0]:p[0] + n]
        p[0] += n
        return v

    m = Model()
    m.version = un("f")[0]
    m.name = _sj(raw(20))
    m.comment = _sj(raw(256))
    nv = un("I")[0]
    vs = np.frombuffer(data, dtype=np.dtype([("p", "<3f4"), ("n", "<3f4"), ("uv", "<2f4"), ("b", "<2u2"),
                                             ("w", "u1"), ("e", "u1")]), count=nv, offset=p[0])
    p[0] += 38 * nv
    m.positions = vs["p"].astype(np.float32).copy()
    m.normals = vs["n"].astype(np.float32).copy()
    m.uvs = vs["uv"].astype(np.float32).copy()
    jo = np.full((nv, 4), -1, np.int32)
    we = np.zeros((nv, 4), np.float32)
    jo[:, 0] = vs["b"][:, 0]
    jo[:, 1] = vs["b"][:, 1]
    w = np.clip(vs["w"].astype(np.float32) / 100.0, 0, 1)
    we[:, 0] = w
    we[:, 1] = 1.0 - w
    m.joints, m.weights = jo, we
    nf = un("I")[0]
    m.indices = np.frombuffer(data, dtype="<u2", count=nf, offset=p[0]).astype(np.int64)
    p[0] += 2 * nf
    nm = un("I")[0]
    tex_index = {}
    mat_tex = []
    for _ in range(nm):
        dr, dg, db, alpha, spec_pow, sr, sg, sb, ar, ag, ab = un("11f")
        toon, edge, cnt = un("BBI")
        tname = _sj(raw(20))
        mt = Material()
        mt.name = "material_%d" % len(m.materials)
        mt.name_en = ""
        mt.diffuse = (dr, dg, db, alpha)
        mt.specular = (sr, sg, sb)
        mt.spec_power = spec_pow
        mt.ambient = (ar, ag, ab)
        mt.flags = (0x10 if edge else 0) | (0x01 if alpha < 0.999 else 0)
        mt.edge_color = (0, 0, 0, 1)
        mt.edge_size = 1.0
        main = ""
        for part in tname.split("*"):
            if part and not part.lower().endswith((".sph", ".spa")):
                main = part
                break
        mt.texture = -1
        if main:
            if main not in tex_index:
                tex_index[main] = len(m.textures)
                m.textures.append(main)
            mt.texture = tex_index[main]
        mt.sphere = -1
        mt.sphere_mode = 0
        mt.toon_shared = 1
        mt.toon = toon if toon < 10 else 0
        mt.memo = ""
        mt.index_count = cnt
        m.materials.append(mt)
    nb = un("H")[0]
    for _ in range(nb):
        b = Bone()
        b.name = _sj(raw(20))
        b.name_en = ""
        par, tail, typ, ikp = un("HHBH")
        b.pos = np.array(un("3f"), np.float64)
        b.parent = -1 if par == 0xFFFF else par
        b.layer = 0
        b.flags = 0x0002 | 0x0008 | 0x0010 | (0x0004 if typ in (1, 9) else 0)
        b.tail_bone = -1 if tail in (0, 0xFFFF) else tail
        b.grant_parent = -1
        b.grant_weight = 0.0
        if typ == 5 and ikp != 0xFFFF:  # rotation influenced by another bone (e.g. eyes)
            b.flags |= 0x0100
            b.grant_parent = ikp
            b.grant_weight = 1.0
        if typ == 9 and tail != 0xFFFF:  # rotation follow with ratio (in tail field)
            pass
        m.bones.append(b)
    nik = un("H")[0]
    for _ in range(nik):
        ik, tgt, ln, it, wgt = un("HHBHf")
        p[0] += 2 * ln
    nmo = un("H")[0]
    base_idx = None
    morphs = []
    for _ in range(nmo):
        name = _sj(raw(20))
        cnt, typ = un("IB")
        arr = np.frombuffer(data, dtype=np.dtype([("i", "<u4"), ("o", "<3f4")]), count=cnt, offset=p[0])
        p[0] += 16 * cnt
        if typ == 0:
            base_idx = arr["i"].astype(np.int64)
            continue
        morphs.append((name, arr))
    for name, arr in morphs:
        mo = Morph()
        mo.name, mo.name_en, mo.panel, mo.type = name, "", 4, 1
        ri = arr["i"].astype(np.int64)
        off = arr["o"].astype(np.float32).copy()
        if base_idx is not None and len(base_idx):
            ok = ri < len(base_idx)
            ri, off = ri[ok], off[ok]
            ix = base_idx[ri]
        else:
            ix = ri
        mo.data = (ix, off)
        m.morphs.append(mo)
    # optional sections: display lists, English names, toon textures, physics
    try:
        n_skin_disp = un("B")[0]
        p[0] += 2 * n_skin_disp
        n_bone_disp_names = un("B")[0]
        p[0] += 50 * n_bone_disp_names
        n_bone_disp = un("I")[0]
        p[0] += 3 * n_bone_disp
        if p[0] < len(data) and data[p[0]] in (0, 1):
            has_en = un("B")[0]
            if has_en:
                m.name_en = _sj(raw(20))
                m.comment_en = _sj(raw(256))
                for b in m.bones:
                    b.name_en = _sj(raw(20))
                for mo in m.morphs:
                    mo.name_en = _sj(raw(20))
                p[0] += 50 * n_bone_disp_names
        if p[0] + 1000 <= len(data):
            toons = [_sj(raw(100)) for _ in range(10)]
            m.pmd_toons = toons
        if p[0] + 4 <= len(data):
            nr = un("I")[0]
            for _ in range(nr):
                rb = RigidBody()
                rb.name = _sj(raw(20))
                rb.name_en = ""
                bone, rb.group, rb.mask, rb.shape = un("HBHB")
                rb.bone = -1 if bone == 0xFFFF else bone
                rb.size = np.array(un("3f"), np.float64)
                rel = np.array(un("3f"), np.float64)
                rb.rot = np.array(un("3f"), np.float64)
                rb.mass, rb.lin_damp, rb.ang_damp, rb.restitution, rb.friction = un("5f")
                rb.mode = un("B")[0]
                # PMD stores the rigid body position relative to its bone (or bone 0)
                anchor = m.bones[rb.bone].pos if 0 <= rb.bone < len(m.bones) else (
                    m.bones[0].pos if m.bones else np.zeros(3))
                rb.pos = rel + anchor
                if rb.bone < 0:
                    rb.bone = 0
                m.rigid_bodies.append(rb)
    except struct.error:
        pass
    return m
