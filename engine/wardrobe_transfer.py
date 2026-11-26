"""Carrying clothes from one model to another (wardrobe catalog).

Saving: the chosen clothing materials of model A are taken in A's generation pose (arms to the sides,
see wardrobe.Body) and every vertex is written in BODY COORDINATES of A:
    part (torso / left-right leg / left-right arm), position along it (torso: height between the leg and
    shoulder joints; limbs: 0 at the first joint .. 1 at the end), angle around its axis, and the gap to
    the skin along the skin normal.
Putting on model B: the same body coordinates are looked up on B's body (B's own proportions and shape:
a bigger bust, longer legs, wider hips ...), the vertex is placed at B's skin + the gap, the result is
smoothed, pushed out of B's body, skinned to B's bones and taken back to B's pose.

The catalog lives in the user's profile (%APPDATA%/mmd2vrm/wardrobe/<uid>/): garment.npz, meta.json,
textures, thumb.png.  It is never part of the program itself.
"""
import base64
import json
import math
import os
import re
import shutil
import time
import uuid

import numpy as np

import bodyfill
import wardrobe

PARTS = ("торс", "левая нога", "правая нога", "левая рука", "правая рука")


# ----------------------------------------------------------------------------------------------------
# body coordinates: part, position along its axis, angle around it, distance from it
# ----------------------------------------------------------------------------------------------------
UP = np.array([0.0, 1.0, 0.0])


def _polyline(B, r_):
    """axis of a part as a polyline (gen pose) and the matching limb name"""
    if r_ == 0:
        o = np.array([B.c[0], B.yL, B.c[2]])
        return [o - UP * B.T * 1.5, o + UP * B.T * 2.0], None
    nm = {1: "leftLeg", 2: "rightLeg", 3: "leftArm", 4: "rightArm"}[r_]
    if nm not in B.limbs:
        return None, None
    a, b, c = B.limbs[nm]
    return [a, b, c], nm


def _axis_at(B, r_, s):
    """point on the axis and its direction at parameter s"""
    if r_ == 0:
        o = np.array([B.c[0], B.yL, B.c[2]])
        return o[None] + np.outer(s * B.T, UP), np.repeat(UP[None], len(s), 0)
    pl, nm = _polyline(B, r_)
    a, b, c = pl
    s = np.asarray(s, float)
    first = s < 0.5
    t1 = np.where(first, s * 2, 1.0)
    t2 = np.where(first, 0.0, (s - 0.5) * 2)
    P = np.where(first[:, None], a + np.outer(t1, b - a), b + np.outer(t2, c - b))
    d1 = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    d2 = (c - b) / max(np.linalg.norm(c - b), 1e-9)
    D = np.where(first[:, None], d1, d2)
    return P, D


def _ref(B, r_, D):
    """reference directions (angle 0 and 90 degrees) perpendicular to the axis directions D"""
    f = B.FWD if r_ in (0, 1, 2) else UP
    fr = f[None] - np.einsum("ij,j->i", D, f)[:, None] * D
    fr /= np.maximum(np.linalg.norm(fr, axis=1), 1e-9)[:, None]
    sr = np.cross(D, fr)
    if r_ in (2, 4):  # mirror the right side: same angle = same place on the body
        sr = -sr
    return fr, sr


def to_cyl(B, X, reg):
    """points -> (s, angle, distance from the axis)"""
    s = np.zeros(len(X))
    ang = np.zeros(len(X))
    rad = np.zeros(len(X))
    for r_ in range(5):
        m = reg == r_
        if not m.any():
            continue
        pl, nm = _polyline(B, r_)
        if pl is None:
            reg[m] = 0
            r_ = 0
            pl, nm = _polyline(B, 0)
        if r_ == 0:
            ss = (X[m, 1] - B.yL) / B.T
        else:
            ss = B.limb_s(X[m], nm)
        P, D = _axis_at(B, r_, ss)
        q = X[m] - P
        q = q - np.einsum("ij,ij->i", q, D)[:, None] * D
        fr, sr = _ref(B, r_, D)
        s[m] = ss
        ang[m] = np.arctan2(np.einsum("ij,ij->i", q, sr), np.einsum("ij,ij->i", q, fr))
        rad[m] = np.linalg.norm(q, axis=1)
    return s, ang, rad


def from_cyl(B, reg, s, ang, rad):
    X = np.zeros((len(s), 3))
    for r_ in range(5):
        m = reg == r_
        if not m.any():
            continue
        rr = r_ if _polyline(B, r_)[0] is not None else 0
        P, D = _axis_at(B, rr, s[m])
        fr, sr = _ref(B, rr, D)
        X[m] = P + (np.cos(ang[m])[:, None] * fr + np.sin(ang[m])[:, None] * sr) * rad[m][:, None]
    return X


def own_regions(B, X, J, W):
    """part of every garment vertex from its OWN skin weights (a sleeve belongs to the arm even far from it)"""
    reg_b = B.region
    side_b = B.side
    Jc = np.where(J < 0, len(reg_b) - 1, np.minimum(J, len(reg_b) - 1))
    Wn = np.where(J < 0, 0.0, W)
    Wn = Wn / np.maximum(Wn.sum(1, keepdims=True), 1e-9)
    R, Sd = reg_b[Jc], side_b[Jc]
    out = np.zeros(len(X), np.int8)
    for code, kind, sd in ((1, ("leg", "foot"), "left"), (2, ("leg", "foot"), "right"),
                           (3, ("arm", "hand"), "left"), (4, ("arm", "hand"), "right")):
        w = (Wn * (np.isin(R, kind) & (Sd == sd))).sum(1)
        out[w > 0.5] = code
    return out


class BodyIndex:
    """radius of the body of model B at any (part, s, angle)"""

    def __init__(self, B):
        self.B = B
        reg = np.zeros(len(B.P), np.int8)
        side = (B.P - B.c) @ B.LEFT
        reg[(B.w_leg > 0.5) & (side > 0)] = 1
        reg[(B.w_leg > 0.5) & (side <= 0)] = 2
        reg[(B.w_arm > 0.5) & (side > 0)] = 3
        reg[(B.w_arm > 0.5) & (side <= 0)] = 4
        self.reg = reg
        self.s, self.ang, self.rad = to_cyl(B, B.P, reg.copy())
        self.parts = {}
        for r_ in range(5):
            m = np.nonzero(reg == r_)[0]
            if len(m) > 10:
                self.parts[r_] = (m, self._emb(self.s[m], self.ang[m], r_))

    @staticmethod
    def _emb(s, ang, r_):
        k = {0: 0.25, 1: 0.12, 2: 0.12, 3: 0.08, 4: 0.08}[int(r_)]
        L = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}[int(r_)]
        return np.stack([np.cos(ang) * k, s * L, np.sin(ang) * k], 1)

    def radius(self, reg, s, ang):
        """body radius at the given coordinates (median of the few nearest body points)"""
        out = np.zeros(len(s))
        for r_ in range(5):
            sel = reg == r_
            if not sel.any():
                continue
            part = self.parts.get(r_) or self.parts.get(0)
            idx, E = part
            Q = self._emb(s[sel], ang[sel], r_)
            j, d = bodyfill.nearest(Q, E, 0.04)
            miss = j < 0
            if miss.any():
                j2, _ = bodyfill.nearest(Q[miss], E, 0.5)
                j[miss] = j2
            out[sel] = np.where(j >= 0, self.rad[idx[np.maximum(j, 0)]], 0.0)
        return out


# ----------------------------------------------------------------------------------------------------
# saving
# ----------------------------------------------------------------------------------------------------
def _slug(s):
    return re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE).strip("_")[:40] or "item"


def save_item(m, h, ov, mat_names, name, base_dir, thumb_dataurl=None, src_name="", log=print):
    """Extract the given clothing materials of model m into the catalog.  Returns the item's uid."""
    from converter import guess_category, resolve_path
    mo = ov.get("materials") or {}
    cats = {}
    for k, mt in enumerate(m.materials):
        texname = m.textures[mt.texture] if 0 <= mt.texture < len(m.textures) else ""
        cats[k] = (mo.get(mt.name) or {}).get("category") or ("skin" if mt.name == "BodyFill" else
                                                             guess_category(mt.name, mt.name_en, texname))
    skin = {k for k, c in cats.items() if c == "skin"}
    sel = [k for k, mt in enumerate(m.materials) if mt.name in set(mat_names) and cats[k] not in ("skin", "face", "eyes", "hair")]
    if not sel:
        raise ValueError("не выбрано ни одной части одежды (кожа, лицо, глаза и волосы не сохраняются как одежда)")
    if not skin:
        raise ValueError("у модели нет материала кожи — отметьте категорию «Кожа» у тела")
    cloth = {k for k, c in cats.items() if c not in ("skin", "face", "eyes")} | set(sel)
    B = wardrobe.Body(m, h, skin, cloth, log)
    # triangles of the chosen materials
    tris, mids = [], []
    off = 0
    for k, mt in enumerate(m.materials):
        if k in sel:
            t = np.asarray(m.indices[off:off + mt.index_count]).reshape(-1, 3)
            tris.append(t)
            mids.append(np.full(len(t), sel.index(k)))
        off += mt.index_count
    T = np.concatenate(tris)
    mid = np.concatenate(mids)
    vs, inv = np.unique(T, return_inverse=True)
    T = inv.reshape(-1, 3)
    P = np.asarray(m.positions, float)[vs]
    N = np.asarray(m.normals, float)[vs]
    UV = np.asarray(m.uvs, float)[vs]
    J = np.asarray(m.joints)[vs]
    W = np.asarray(m.weights, float)[vs]
    Pg = B.to_gen(P, J, W)
    Ng = B.to_gen(N, J, W, normals=True)
    reg = own_regions(B, Pg, J, W)
    s, ang, rad = to_cyl(B, Pg, reg)
    BI = BodyIndex(B)
    rb = BI.radius(reg, s, ang)
    gap = rad - rb                               # outside the body: > 0
    # pieces that are not clothes on the body (tail, pendant, wings): far from the body -> left out
    lab = wardrobe.components(T, len(P))
    ul, inv2 = np.unique(lab, return_inverse=True)
    from converter import _kind_strict
    kind_b = np.array([(_kind_strict(b_.name) or "") for b_ in m.bones] + [""])
    Jc_ = np.where(J < 0, len(kind_b) - 1, np.minimum(J, len(kind_b) - 1))
    Wn_ = np.where(J < 0, 0.0, W)
    Wn_ = Wn_ / np.maximum(Wn_.sum(1, keepdims=True), 1e-9)
    not_cloth = (Wn_ * np.isin(kind_b[Jc_], ["tail", "hair", "ear", "accessory"])).sum(1)
    on_head = (Wn_ * np.isin(B.region[Jc_], ["head"])).sum(1)
    far = np.zeros(len(ul), bool)
    head = np.zeros(len(ul), bool)
    for k_ in range(len(ul)):
        mm = inv2 == k_
        if np.median(on_head[mm]) > 0.5:
            head[k_] = True
        elif np.median(not_cloth[mm]) > 0.5 or np.median(gap[mm]) > 0.4 * B.T:
            far[k_] = True
    drop = (far | head)[inv2]
    if head.any():
        log("  ! гардероб: головной убор / украшение на голове (%d вершин) не сохранено — перенос шапок пока не поддерживается"
            % int(head[inv2].sum()))
    if far.any():
        log("  ! гардероб: %d вершин далеко от тела (хвост, подвеска?) — не сохранены. Чтобы сохранить только одежду, "
            "разделите материал (✂) и выберите нужные куски." % int(far[inv2].sum()))
        keep_t = ~drop[T].any(1)
        T, mid = T[keep_t], mid[keep_t]
        used = np.unique(T)
        rm = -np.ones(len(P), np.int64)
        rm[used] = np.arange(len(used))
        T = rm[T]
        P, N, UV, J, W, Pg, Ng = P[used], N[used], UV[used], J[used], W[used], Pg[used], Ng[used]
        reg, s, ang, rad, rb, gap = reg[used], s[used], ang[used], rad[used], rb[used], gap[used]
    if not len(T):
        raise ValueError("выбранные части не похожи на одежду на теле")
    uid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    d_ = os.path.join(wardrobe.catalog_dir(), uid)
    os.makedirs(d_, exist_ok=True)
    mats = []
    copied = {}
    for n_, k in enumerate(sel):
        mt = m.materials[k]
        rel = m.textures[mt.texture] if 0 <= mt.texture < len(m.textures) else ""
        tex = None
        tp = resolve_path(base_dir, rel) if rel else None
        if tp and os.path.isfile(tp):
            tex = copied.get(os.path.abspath(tp))
            if tex is None:  # one copy per texture file
                ext = os.path.splitext(tp)[1].lower() or ".png"
                tex = "tex%d%s" % (len(copied), ext)
                shutil.copyfile(tp, os.path.join(d_, tex))
                copied[os.path.abspath(tp)] = tex
        mats.append({"name": mt.name, "texture": tex, "diffuse": list(map(float, mt.diffuse)),
                     "ambient": list(map(float, mt.ambient)), "specular": list(map(float, mt.specular)),
                     "flags": int(mt.flags), "edge_color": list(map(float, mt.edge_color)), "edge_size": float(mt.edge_size),
                     "toon_shared": int(mt.toon_shared), "toon": int(mt.toon) if mt.toon_shared else -1})
    np.savez_compressed(os.path.join(d_, "garment.npz"), reg=reg, s=s.astype(np.float32), ang=ang.astype(np.float32),
                        gap=(gap / B.T).astype(np.float32), rb=(rb / B.T).astype(np.float32), uv=UV.astype(np.float32),
                        T=T.astype(np.int32), mid=mid.astype(np.int16))
    kinds = sorted({PARTS[int(r_)] for r_ in np.unique(reg)})
    meta = {"uid": uid, "name": name or ", ".join(mat_names), "source": src_name, "materials": mats,
            "created": time.strftime("%Y-%m-%d %H:%M"), "verts": int(len(P)), "tris": int(len(T)), "parts": kinds,
            "version": 2}
    with open(os.path.join(d_, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    if isinstance(thumb_dataurl, str) and thumb_dataurl.startswith("data:image/png;base64,") and len(thumb_dataurl) < 3_000_000:
        try:
            raw = base64.b64decode(thumb_dataurl.split(",", 1)[1], validate=True)
            if raw[:8] == b"\x89PNG\r\n\x1a\n":
                with open(os.path.join(d_, "thumb.png"), "wb") as f:
                    f.write(raw)
        except Exception:
            pass
    log("  гардероб: сохранено в каталог «%s» (%d вершин)" % (meta["name"], len(P)))
    return uid


def list_items():
    out = []
    root = wardrobe.catalog_dir()
    for uid in sorted(os.listdir(root), reverse=True):
        mp = os.path.join(root, uid, "meta.json")
        if os.path.isfile(mp):
            try:
                with open(mp, encoding="utf-8") as f:
                    meta = json.load(f)
                meta["thumb"] = os.path.isfile(os.path.join(root, uid, "thumb.png"))
                out.append({k: meta.get(k) for k in ("uid", "name", "source", "created", "parts", "tris", "thumb")})
            except Exception:
                pass
    return out


def item_dir(uid):
    if not re.fullmatch(r"[\w\-]+", uid or ""):
        raise ValueError("bad id")
    return os.path.join(wardrobe.catalog_dir(), uid)


def delete_item(uid):
    shutil.rmtree(item_dir(uid), ignore_errors=True)


def rename_item(uid, name):
    mp = os.path.join(item_dir(uid), "meta.json")
    if not os.path.isfile(mp):
        raise ValueError("нет такой вещи в каталоге")
    with open(mp, encoding="utf-8") as f:
        meta = json.load(f)
    meta["name"] = str(name)[:80]
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


# ----------------------------------------------------------------------------------------------------
# putting on another model
# ----------------------------------------------------------------------------------------------------
_IDX = {}


def apply_item(m, h, body, it, skin_k, log=print):
    """body(spread) -> wardrobe.Body of the target model (cached)"""
    d_ = item_dir(it.get("id") or it.get("uid") or "")
    with open(os.path.join(d_, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    z = np.load(os.path.join(d_, "garment.npz"))
    reg, s, ang = z["reg"].astype(np.int8), z["s"].astype(float), z["ang"].astype(float)
    B = body(9.0 if np.isin(reg, (1, 2)).any() else 0.0)  # trousers: legs apart; tops: legs as they are
    gap, rbA = z["gap"].astype(float) * B.T, z["rb"].astype(float) * B.T
    UV, T, mid = z["uv"], z["T"].astype(np.int64), z["mid"].astype(int)
    size = float(it.get("size", 1.0))
    loose = float(it.get("loose", 0.0))
    shift = float(it.get("shift", 0.0))
    bi = getattr(B, "_index", None)  # kept on the body itself (no stale index for another model)
    if bi is None:
        bi = BodyIndex(B)
        B._index = bi
    s2 = s + np.where(reg == 0, shift, shift * 0.5)
    rbB = bi.radius(reg, s2, ang)
    rbB_s = wardrobe.smooth_scalar(rbB, T, len(rbB), it=4)   # the lookup is point-wise: smooth the body radius
    rad = rbB_s + gap * size + 0.004 * B.H * loose
    rad = np.maximum(rad, rbB_s + 0.0015 * B.H)
    V = from_cyl(B, reg, s2, ang, rad)
    V = bodyfill.smooth_mesh(V, T, it=1, lam=0.3)
    for _ in range(2 + int(loose * 6)):
        if loose > 0:
            V = bodyfill.smooth_mesh(V, T, it=2, lam=0.5)
        jv, dv = bodyfill.nearest(V, B.P, B.H / 30)
        ok = jv >= 0
        nn = B.N[jv[ok]]
        sd = np.einsum("ij,ij->i", V[ok] - B.P[jv[ok]], nn)
        need = sd < 0.0015 * B.H
        idx = np.nonzero(ok)[0][need]
        V[idx] += nn[need] * (0.0015 * B.H - sd[need])[:, None]
    # tails of model B come out through a round opening
    if getattr(B, "tails", None):
        hole = B.tail_hole(V)
        if (hole < 0).any():
            vmid = np.zeros(len(V), int)
            vmid[T.reshape(-1)] = np.repeat(mid, 3)
            V, T, (UV, vmid) = wardrobe.cut_mesh(V, T, hole, (UV, vmid), pick=(1,))
            mid = vmid[T[:, 0]]
    # weights from B's body, back to B's own pose
    jv, _ = bodyfill.nearest(V, B.P, B.H / 25)
    miss = jv < 0
    if miss.any():
        j2, _ = bodyfill.nearest(V[miss], B.P, B.H / 4)
        jv[miss] = j2
    jv = np.maximum(jv, 0)
    J, W = B.J[jv], B.W[jv]
    N = bodyfill.vertex_normals(V, T)
    Vm = B.from_gen(V, J, W)
    Nm = B.from_gen(N, J, W, normals=True)
    out = []
    for k, mat in enumerate(meta["materials"]):
        tk = T[mid == k]
        if not len(tk):
            continue
        vs, inv = np.unique(tk, return_inverse=True)
        tk2 = inv.reshape(-1, 3)
        res = (Vm[vs], Nm[vs], UV[vs], tk2, J[vs], W[vs])
        tex = os.path.join(d_, mat["texture"]) if mat.get("texture") else wardrobe.fabric_texture("solid", "#dddddd")
        name = "WD %s" % ((it.get("label") or meta["name"]) + ("" if len(meta["materials"]) == 1 else " · " + mat["name"]))
        out.append((res, tex, name, mat, B))
    log("  гардероб: «%s» перенесена с модели %s — %d треугольников" % (meta["name"], meta.get("source") or "?", len(T)))
    return out
