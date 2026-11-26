"""Wardrobe: built-in clothes generated to the shape of the model's body, and clothes carried over from
other models (catalog).  Works in model (MMD) space before the T-pose, like body completion (bodyfill.py).

Built-in garment = a smooth shell around the body:
  1. body proxy: the model's skin plus the completed body (bodyfill) where the skin is missing;
  2. signed distance to the proxy on a grid -> surface at `offset` outside the body (Surface Nets);
  3. loose garments are relaxed (smoothed) and pushed back outside the body, so they hang over hollows;
  4. coverage field on the body (height, neckline, armholes / sleeves / legs from the skin weights)
     -> the shell is cut along it (clean hems);
  5. skin weights from the nearest body point, procedural fabric texture, cylindrical UVs.
All built-in clothes and their textures are generated here (own work, free to redistribute).
"""
import colorsys
import hashlib
import re
import json
import math
import os
import tempfile

import numpy as np

import bodyfill

# ----------------------------------------------------------------------------------------------------
# catalog of built-in garments
# kind: top / bottom.  Values are fractions of the torso length T (shoulder joints -> leg joints) or of
# the limb length; see coverage().
# ----------------------------------------------------------------------------------------------------
BUILTIN = [
    # tops
    dict(id="tshirt", name="Футболка", kind="top", neck="crew", depth=0.06, sleeve=0.30, bottom="hip", loose=0.55, thick=1.0,
         color="#f2f2f2", pattern="solid"),
    dict(id="tshirt_v", name="Футболка с V-вырезом", kind="top", neck="v", depth=0.20, sleeve=0.28, bottom="hip", loose=0.5, thick=1.0,
         color="#2d3a5c", pattern="heather"),
    dict(id="tshirt_stripe", name="Тельняшка", kind="top", neck="crew", depth=0.07, sleeve=0.30, bottom="hip", loose=0.45, thick=1.0,
         color="#ffffff", color2="#1d2f6b", pattern="stripes"),
    dict(id="longsleeve", name="Лонгслив", kind="top", neck="crew", depth=0.05, sleeve=0.97, bottom="hip", loose=0.25, thick=1.0,
         color="#c9b8a6", pattern="rib"),
    dict(id="turtleneck", name="Водолазка", kind="top", neck="turtle", depth=0.0, sleeve=0.97, bottom="hip", loose=0.15, thick=1.0,
         color="#1f1f24", pattern="rib"),
    dict(id="tank", name="Майка", kind="top", neck="scoop", depth=0.16, straps=0.30, bottom="hip", loose=0.35, thick=1.0,
         color="#ffffff", pattern="rib"),
    dict(id="camisole", name="Топ на бретелях", kind="top", neck="straps", straps=0.06, top="bust", bottom="hip", loose=0.55, thick=0.8,
         color="#f3c6d3", pattern="satin"),
    dict(id="crop_top", name="Кроп-топ", kind="top", neck="crew", depth=0.06, sleeve=0.22, bottom="crop", loose=0.25, thick=1.0,
         color="#ff8fb3", pattern="solid"),
    dict(id="tube_top", name="Топ-бандо", kind="top", neck="flat", top="bust", bottom="underbust", loose=0.1, thick=0.9,
         color="#1b1b1f", pattern="rib"),
    dict(id="sports_bra", name="Спортивный топ", kind="top", neck="scoop", depth=0.10, straps=0.20, bottom="underbust", loose=0.05, thick=1.0,
         color="#5ad1c3", color2="#20302e", pattern="sport"),
    dict(id="bra", name="Бюстгальтер", kind="top", neck="straps", straps=0.035, top="bra", bottom="underbust", loose=0.0, thick=0.8,
         color="#f6f0ea", pattern="lace"),
    dict(id="bikini_top", name="Купальник: верх", kind="top", neck="straps", straps=0.03, top="bra", bottom="underbust", loose=0.0, thick=0.8,
         color="#ff5a6e", color2="#ffffff", pattern="dots"),
    dict(id="hoodie", name="Толстовка", kind="top", neck="crew", depth=0.04, sleeve=0.98, bottom="hip_low", loose=1.0, thick=2.2,
         color="#7a8796", pattern="heather"),
    dict(id="sweater", name="Свитер", kind="top", neck="crew", depth=0.03, sleeve=0.98, bottom="hip", loose=0.8, thick=2.0,
         color="#8c2f39", pattern="knit"),
    dict(id="cardigan", name="Кофта на пуговицах", kind="top", neck="v", depth=0.30, sleeve=0.98, bottom="hip_low", loose=0.75, thick=1.6,
         open_front=0.015, color="#d9c7a7", pattern="knit"),
    # bottoms
    dict(id="jeans", name="Джинсы", kind="bottom", top="hipline", leg=0.98, loose=0.2, thick=1.3, color="#3b5b8c", pattern="denim"),
    dict(id="sweatpants", name="Спортивные штаны", kind="bottom", top="waist", leg=0.98, loose=0.9, thick=1.6, color="#3a3d45",
         pattern="heather"),
    dict(id="leggings", name="Леггинсы", kind="bottom", top="waist", leg=0.95, loose=0.0, thick=0.8, color="#141418", pattern="solid"),
    dict(id="capri", name="Бриджи", kind="bottom", top="hipline", leg=0.70, loose=0.3, thick=1.2, color="#c8b08a", pattern="twill"),
    dict(id="shorts", name="Шорты", kind="bottom", top="hipline", leg=0.20, loose=0.35, thick=1.3, color="#4d6f9e", pattern="denim"),
    dict(id="bike_shorts", name="Велосипедки", kind="bottom", top="waist", leg=0.30, loose=0.0, thick=0.8, color="#20232a", pattern="solid"),
    dict(id="plaid_pants", name="Брюки в клетку", kind="bottom", top="waist", leg=0.98, loose=0.45, thick=1.3, color="#4a4a52",
         color2="#b33a3a", pattern="plaid"),
    dict(id="briefs", name="Трусы", kind="bottom", top="low", leg=0.02, loose=0.0, thick=0.7, color="#f6f0ea", pattern="lace"),
    dict(id="bikini_bottom", name="Купальник: низ", kind="bottom", top="low", leg=0.0, loose=0.0, thick=0.7, color="#ff5a6e",
         color2="#ffffff", pattern="dots"),
    dict(id="boxers", name="Шорты-боксеры", kind="bottom", top="hipline", leg=0.10, loose=0.15, thick=0.9, color="#2f4a7a",
         pattern="heather"),
]
BUILTIN_BY_ID = {g["id"]: g for g in BUILTIN}
PATTERNS = {"solid": "гладкая", "heather": "меланж", "stripes": "полоска", "rib": "рубчик", "knit": "вязка", "denim": "деним",
            "twill": "твил", "plaid": "клетка", "dots": "горошек", "lace": "кружево", "satin": "атлас", "sport": "спорт"}

_CACHE = {}


def catalog_dir():
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    d = os.path.join(base, "mmd2vrm", "wardrobe")
    os.makedirs(d, exist_ok=True)
    return d


def tex_dir():
    d = os.path.join(tempfile.gettempdir(), "mmd2vrm_wardrobe_tex")
    os.makedirs(d, exist_ok=True)
    return d


# ----------------------------------------------------------------------------------------------------
# procedural fabric textures (512 px, tileable)
# ----------------------------------------------------------------------------------------------------
def _hex(c, default=(240, 240, 240)):
    try:
        c = c.lstrip("#")
        return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], float)
    except Exception:
        return np.array(default, float)


def _noise(S, scale, seed):
    """tileable value noise in [0, 1]"""
    rng = np.random.default_rng(seed)
    g = rng.random((scale, scale))
    x = np.arange(S) * scale / S
    i0 = np.floor(x).astype(int) % scale
    i1 = (i0 + 1) % scale
    f = x - np.floor(x)
    f = f * f * (3 - 2 * f)
    a = g[i0][:, i0] * (1 - f)[None, :] + g[i0][:, i1] * f[None, :]
    b = g[i1][:, i0] * (1 - f)[None, :] + g[i1][:, i1] * f[None, :]
    return a * (1 - f)[:, None] + b * f[:, None]


_HEX = re.compile(r"#[0-9a-fA-F]{6}")


def fabric_texture(pattern, color, color2=None):
    """path of a PNG (cached in the temp folder).  Only known patterns and #rrggbb colours (they end up in the
    file name)."""
    if pattern not in PATTERNS:
        pattern = "solid"
    color = color if isinstance(color, str) and _HEX.fullmatch(color) else "#dddddd"
    color2 = color2 if isinstance(color2, str) and _HEX.fullmatch(color2) else None
    key = hashlib.md5(("%s|%s|%s|3" % (pattern, color, color2)).encode()).hexdigest()[:16]
    path = os.path.join(tex_dir(), "wd_%s_%s.png" % (pattern, key))
    if os.path.isfile(path):
        return path
    from PIL import Image
    S = 512
    c1 = _hex(color)
    c2 = _hex(color2) if color2 else np.clip(c1 * 0.55, 0, 255)
    y, x = np.mgrid[0:S, 0:S].astype(float)
    fine = _noise(S, 128, 1) * 0.6 + _noise(S, 64, 2) * 0.4
    shade = 1.0 + (fine - 0.5) * 0.06
    img = np.repeat(np.ones((S, S, 1)), 3, axis=2) * c1
    if pattern == "heather":
        n = _noise(S, 256, 3)
        shade = 1.0 + (n - 0.5) * 0.22 + (fine - 0.5) * 0.08
    elif pattern == "stripes":
        band = (np.floor(y / (S / 8)) % 2 == 1)
        img[band] = c2
    elif pattern == "rib":
        shade = shade * (1.0 + 0.07 * np.cos(x / S * 2 * math.pi * 48))
    elif pattern == "knit":
        u = (x / S * 32) % 1.0
        v = (y / S * 24 + np.where((np.floor(x / S * 32) % 2) == 0, 0, 0.5)) % 1.0
        vee = np.abs(u - 0.5) * 2
        shade = shade * (0.88 + 0.16 * np.clip(1 - np.abs(v - vee) * 2.2, 0, 1)) * (1 + (_noise(S, 256, 4) - 0.5) * 0.06)
    elif pattern in ("denim", "twill"):
        diag = np.cos((x + y) / S * 2 * math.pi * (64 if pattern == "denim" else 40))
        n = _noise(S, 256, 5)
        k = 0.10 if pattern == "denim" else 0.06
        shade = 1.0 + diag * k + (n - 0.5) * (0.25 if pattern == "denim" else 0.08)
        if pattern == "denim":
            white = (_noise(S, 512, 6) > 0.82) * 0.12
            img = img * (1 - white[..., None]) + 235 * white[..., None]
    elif pattern == "plaid":
        a = (np.floor(x / (S / 4)) % 2 == 0)
        b = (np.floor(y / (S / 4)) % 2 == 0)
        thin = ((x % (S / 4)) < 6) | ((y % (S / 4)) < 6)
        mix = (a.astype(float) + b.astype(float)) / 2
        img = c1[None, None] * (1 - mix[..., None] * 0.35)
        img[thin] = c2
    elif pattern == "dots":
        cx = (x % (S / 8)) - S / 16
        cy = (y % (S / 8)) - S / 16
        img[(cx * cx + cy * cy) < (S / 40) ** 2] = c2
    elif pattern == "lace":
        u = (x % (S / 8)) / (S / 8) - 0.5
        v = (y % (S / 8)) / (S / 8) - 0.5
        r = np.sqrt(u * u + v * v)
        petal = np.abs(np.cos(np.arctan2(v, u) * 3)) * 0.32
        motif = (np.abs(r - petal) < 0.035) | (np.abs(r - 0.42) < 0.02)
        shade = shade * np.where(motif, 0.84, 1.0)
    elif pattern == "satin":
        shade = 1.0 + 0.10 * np.sin(y / S * 2 * math.pi * 3 + _noise(S, 16, 7) * 3) + (fine - 0.5) * 0.03
    elif pattern == "sport":
        stripe = (np.abs((y / S * 4) % 1.0 - 0.5) < 0.06)
        img[stripe] = c2
    img = np.clip(img * np.asarray(shade)[..., None], 0, 255).astype(np.uint8)
    Image.fromarray(img, "RGB").save(path)
    return path


# ----------------------------------------------------------------------------------------------------
# body proxy and landmarks
# ----------------------------------------------------------------------------------------------------
def _side_of_bones(m, h):
    """'left' / 'right' / '' for every bone (from the nearest humanoid ancestor)"""
    hb = {i: n for n, i in h.items() if i is not None}
    out = []
    for i in range(len(m.bones)):
        j, sd, seen = i, "", 0
        while j is not None and 0 <= j < len(m.bones) and seen < 512:
            n = hb.get(j)
            if n:
                sd = "left" if n.startswith("left") else ("right" if n.startswith("right") else "")
                break
            j = m.bones[j].parent
            seen += 1
        out.append(sd)
    return out


def _rot(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def _region_of_bones(m, h):
    """for every bone: 'arm', 'leg', 'head', 'hand', 'foot' or 'torso' (nearest humanoid ancestor)"""
    hb = {i: n for n, i in h.items() if i is not None}
    reg = []
    for i in range(len(m.bones)):
        j, r = i, "torso"
        seen = 0
        while j is not None and 0 <= j < len(m.bones) and seen < 512:
            n = hb.get(j)
            if n:
                if "Hand" in n or any(f in n for f in ("Thumb", "Index", "Middle", "Ring", "Little")):
                    r = "hand"
                elif "Arm" in n:
                    r = "arm"
                elif "Foot" in n or "Toes" in n:
                    r = "foot"
                elif "Leg" in n:
                    r = "leg"
                elif n in ("head", "leftEye", "rightEye", "jaw"):
                    r = "head"
                elif n == "neck":
                    r = "neck"
                else:
                    r = "torso"
                break
            j = m.bones[j].parent
            seen += 1
        reg.append(r)
    return reg


def _mat_tris(m, mats):
    out = []
    off = 0
    for k, mt in enumerate(m.materials):
        if k in mats:
            out.append(np.asarray(m.indices[off:off + mt.index_count]).reshape(-1, 3))
        off += mt.index_count
    return np.concatenate(out) if out else np.zeros((0, 3), np.int64)


class Body:
    """the full body surface (skin + completed body) as dense points with normals, weights and labels"""

    def __init__(self, m, h, skin_mats, cloth_mats, log=print, leg_spread=9.0):
        self.m, self.h = m, h
        self.leg_spread = float(leg_spread)
        P = np.asarray(m.positions, float)
        self.H = float(np.ptp(P[:, 1])) or 20.0
        T = _mat_tris(m, skin_mats)
        vs = np.unique(T) if len(T) else np.zeros(0, np.int64)
        pts, nrm, jo, we = [P[vs]], [np.asarray(m.normals, float)[vs]], [np.asarray(m.joints)[vs]], [np.asarray(m.weights)[vs]]
        if len(T):  # triangle centres make the cloud denser where the mesh is coarse
            c = P[T].mean(1)
            n = np.asarray(m.normals, float)[T].mean(1)
            pts.append(c)
            nrm.append(n)
            jo.append(np.asarray(m.joints)[T[:, 0]])
            we.append(np.asarray(m.weights)[T[:, 0]])
        fill = None
        try:
            fill = bodyfill.build_body(m, h, {"torso", "legs", "arms", "feet"}, set(skin_mats), set(cloth_mats),
                                       log=lambda *a: None, hidden_mats=set(cloth_mats))
        except Exception as e:
            log("  ! гардероб: достройка тела под одеждой не удалась (%s)" % e)
        n_skin = sum(len(x) for x in pts)
        if fill is not None:
            V, N, Tf, J, W = fill
            pts += [V]
            nrm += [N]
            jo += [J]
            we += [W]
        self.P = np.concatenate(pts)
        self.is_fill = np.arange(len(self.P)) >= n_skin
        # "skin" far from the skeleton is not body (a skirt or coat in the same material as the skin)
        segs = []
        chain = [["hips", "spine", "chest", "upperChest", "neck", "head"]]
        for sd in ("left", "right"):
            chain += [["upperChest", sd + "Shoulder", sd + "UpperArm", sd + "LowerArm", sd + "Hand"],
                      ["chest", sd + "Shoulder"], ["hips", sd + "UpperLeg", sd + "LowerLeg", sd + "Foot", sd + "Toes"]]
        for ch in chain:
            ps = [np.asarray(m.bones[h[n]].pos, float) for n in ch if h.get(n) is not None]
            segs += list(zip(ps[:-1], ps[1:]))
        if len(segs) >= 6:
            dist = np.full(len(self.P), np.inf)
            for a_, b_ in segs:
                ab = b_ - a_
                t = np.clip((self.P - a_) @ ab / max(float(ab @ ab), 1e-12), 0, 1)
                dist = np.minimum(dist, np.linalg.norm(self.P - (a_ + t[:, None] * ab), axis=1))
            keep = dist < 0.14 * self.H
            if (~keep).sum() > 50 and keep.sum() > 200:
                log("  гардероб: %d точек «кожи» далеко от скелета (юбка или плащ в материале кожи?) — не считаются телом"
                    % int((~keep).sum()))
                pts_keep = keep
                self.P = self.P[pts_keep]
                self.is_fill = self.is_fill[pts_keep]
                nrm = [np.concatenate(nrm)[pts_keep]]
                jo = [np.concatenate(jo)[pts_keep]]
                we = [np.concatenate(we)[pts_keep]]
        N = np.concatenate(nrm)
        self.N = N / np.maximum(np.linalg.norm(N, axis=1), 1e-9)[:, None]
        self.J = np.concatenate(jo).astype(np.int64)
        self.W = np.concatenate(we).astype(float)
        # thin the cloud to ~one point per voxel of H/260 (speed of the distance queries)
        st = self.H / 260.0
        key = np.floor(self.P / st).astype(np.int64)
        _, keep = np.unique(key, axis=0, return_index=True)
        keep = np.sort(keep)
        self.P, self.N, self.J, self.W = self.P[keep], self.N[keep], self.J[keep], self.W[keep]
        self.is_fill = self.is_fill[keep]
        reg = np.array(_region_of_bones(m, h) + ["torso"])
        side_b = np.array(_side_of_bones(m, h) + [""])
        self._clean_weights(reg, side_b)
        Jc = np.where(self.J < 0, len(reg) - 1, np.minimum(self.J, len(reg) - 1))
        Wn = self.W / np.maximum(self.W.sum(1, keepdims=True), 1e-9)
        R = reg[Jc]
        self.w_arm = (Wn * ((R == "arm") | (R == "hand"))).sum(1)
        self.w_leg = (Wn * ((R == "leg") | (R == "foot"))).sum(1)
        self.w_head = (Wn * (R == "head")).sum(1)
        self.w_hand = (Wn * (R == "hand")).sum(1)
        self.w_foot = (Wn * (R == "foot")).sum(1)
        self.region = reg
        self.side = np.array(_side_of_bones(m, h) + [""])
        self._gen_pose()
        self._tails()
        Mp = self._point_transforms()
        self.P = np.einsum("nab,nb->na", Mp[:, :3, :3], self.P) + Mp[:, :3, 3]
        Nr = np.einsum("nab,nb->na", Mp[:, :3, :3], self.N)
        self.N = Nr / np.maximum(np.linalg.norm(Nr, axis=1), 1e-9)[:, None]
        self._landmarks()

    def _tails(self):
        """roots of tails (bone kind 'tail' whose parent is not a tail): (root, direction, radius) — clothes get a
        neat round opening there so the tail comes out of them"""
        m = self.m
        self.tails = []
        try:
            from converter import _kind_strict
        except Exception:
            return
        kind = [(_kind_strict(b.name) or "") for b in m.bones]
        P = np.asarray(m.positions, float)
        J = np.asarray(m.joints)
        W = np.asarray(m.weights, float)
        for i, b in enumerate(m.bones):
            if kind[i] != "tail" or (0 <= b.parent < len(m.bones) and kind[b.parent] == "tail"):
                continue
            kids = [k for k, c in enumerate(m.bones) if c.parent == i]
            a = np.asarray(b.pos, float)
            if kids:
                c = np.asarray(m.bones[kids[0]].pos, float)
            else:
                continue
            d = c - a
            L = float(np.linalg.norm(d))
            if L < 1e-6:
                continue
            d /= L
            # tail thickness: vertices weighted to the root bone, distance from the tail axis
            sel = ((J == i) & (W > 0.4)).any(1)
            r = 0.035 * self.H
            if sel.sum() > 8:
                q = P[sel] - a
                q = q - np.outer(q @ d, d)
                r = float(np.percentile(np.linalg.norm(q, axis=1), 70))
            r = min(max(r, 0.008 * self.H), 0.06 * self.H)
            M = self.Mb[i]  # tails hang on the torso: no change in the generation pose, but stay exact
            self.tails.append((M[:3, :3] @ a + M[:3, 3], M[:3, :3] @ d, r, L))

    def tail_hole(self, V):
        """> 0 away from the tails, < 0 inside the opening around a tail root (model units)"""
        c = np.full(len(V), self.H)
        for a, d, r, L in getattr(self, "tails", []):
            # the opening goes from a bit inside the body to 1.5 bone lengths out along the tail
            t = np.clip((V - a) @ d, -2.5 * r, 1.5 * L)
            dist = np.linalg.norm(V - (a + np.outer(t, d)), axis=1)
            c = np.minimum(c, dist - r * 1.3)
        return c

    def _clean_weights(self, reg, side_b):
        """Body points can carry weights of bones they do not belong to (a completed body at the hips takes the
        weights of the hands hanging next to it; rips weight skin to hair bones).  Clothes copy these weights,
        so: arm / hand bones count only near that arm, head / hair bones only near the neck and head."""
        m, h = self.m, self.h
        raw = lambda n: (np.asarray(m.bones[h[n]].pos, float) if h.get(n) is not None else None)
        P = self.P
        J = np.where(self.J < 0, len(reg) - 1, np.minimum(self.J, len(reg) - 1))
        W = self.W.copy()
        la, ll = raw("leftUpperArm"), raw("leftUpperLeg")
        rl = raw("rightUpperLeg")
        T_ = abs(la[1] - ll[1]) if la is not None and ll is not None else self.H * 0.3
        for sd in ("left", "right"):
            pts = [raw(sd + n) for n in ("UpperArm", "LowerArm", "Hand")]
            if any(p is None for p in pts):
                continue
            tip = pts[2] + (pts[2] - pts[1]) * 0.45
            segs = [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], tip)]
            dist = np.full(len(P), np.inf)
            radial = np.zeros((len(P), 3))
            for a, b in segs:
                ab = b - a
                t = np.clip((P - a) @ ab / max(float(ab @ ab), 1e-12), 0, 1)
                u = P - (a + t[:, None] * ab)
                dd = np.linalg.norm(u, axis=1)
                better = dd < dist
                dist = np.where(better, dd, dist)
                radial[better] = u[better] / np.maximum(dd[better], 1e-9)[:, None]
            # arm skin faces away from the arm axis; a hip next to a hanging hand faces another way
            facing = np.einsum("ij,ij->i", radial, self.N) > 0.25
            far = (dist > 0.16 * T_) | (~facing & self.is_fill)
            isarm = np.isin(reg[J], ["arm", "hand"]) & (side_b[J] == sd)
            W[far[:, None] & isarm] = 0.0
        neck = raw("neck")
        if neck is not None:
            low = P[:, 1] < neck[1] - 0.12 * T_
            W[low[:, None] & (reg[J] == "head")] = 0.0
        # bones of swinging chains (hair, skirt, tail, ribbons...) must not carry clothes; breast and soft
        # body bones may (the top moves with the bust)
        try:
            from converter import _kind_strict
            swing = np.array([(_kind_strict(b.name) or "") in ("hair", "skirt", "tail", "ear", "ribbon", "accessory", "cloth",
                                                              "clothes", "other") for b in m.bones] + [False])
            W[swing[J]] = 0.0
        except Exception:
            pass
        bad = W.sum(1) < 1e-6
        if bad.any():  # nothing left: take the nearest torso / leg bone of the skeleton
            cand = [i for i in range(len(m.bones)) if reg[i] in ("torso", "leg") and h.get("hips") is not None]
            if cand:
                bpos = np.array([m.bones[i].pos for i in cand], float)
                jj, _ = bodyfill.nearest(P[bad], bpos, self.H)
                J2 = np.full((bad.sum(), J.shape[1]), -1)
                J2[:, 0] = np.array(cand)[np.maximum(jj, 0)]
                W2 = np.zeros((bad.sum(), J.shape[1]))
                W2[:, 0] = 1.0
                self.J[bad] = J2
                W[bad] = W2
        self.W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)

    def bp(self, n):
        """bone position in the generation pose"""
        i = self.h.get(n)
        if i is None:
            return None
        p = np.asarray(self.m.bones[i].pos, float)
        M = self.Mb[i]
        return M[:3, :3] @ p + M[:3, 3]

    def _gen_pose(self):
        """Generation pose: arms lifted to horizontal (T-pose), legs spread a little.  The clothes are built in
        this pose — hands no longer touch the hips and the thighs do not touch — and then taken back to the
        model's own pose with the inverse of the same weighted transform."""
        m, h = self.m, self.h
        raw = lambda n: (np.asarray(m.bones[h[n]].pos, float) if h.get(n) is not None else None)
        nb = len(m.bones)
        self.Mb = np.repeat(np.eye(4)[None], nb + 1, 0)
        la, ra = raw("leftUpperArm"), raw("rightUpperArm")
        if la is None or ra is None:
            return
        left = la - ra
        left[1] = 0
        left /= max(np.linalg.norm(left), 1e-9)
        up = np.array([0.0, 1.0, 0.0])
        for sd, sx in (("left", 1.0), ("right", -1.0)):
            sh, el = raw(sd + "UpperArm"), raw(sd + "LowerArm")
            hp, kn = raw(sd + "UpperLeg"), raw(sd + "LowerLeg")
            if sh is not None and el is not None:
                d = el - sh
                d /= max(np.linalg.norm(d), 1e-9)
                tgt = left * sx
                ax = np.cross(d, tgt)
                ang = math.atan2(np.linalg.norm(ax), float(d @ tgt))
                R = _rot(ax, ang) if np.linalg.norm(ax) > 1e-9 else np.eye(3)
                M = np.eye(4)
                M[:3, :3] = R
                M[:3, 3] = sh - R @ sh
                for i in range(nb):
                    if self.side[i] == sd and self.region[i] in ("arm", "hand"):
                        self.Mb[i] = M
            if hp is not None and kn is not None and self.leg_spread:
                fwd_ax = np.cross(up, left)  # spreading axis (front/back)
                R = _rot(fwd_ax, -sx * math.radians(self.leg_spread))
                M = np.eye(4)
                M[:3, :3] = R
                M[:3, 3] = hp - R @ hp
                for i in range(nb):
                    if self.side[i] == sd and self.region[i] in ("leg", "foot"):
                        self.Mb[i] = M

    def _point_transforms(self):
        """generation-pose transform of every body point: from its weights, but a point lying right on an arm
        (closer to the arm than to the torso or legs) follows that arm whatever bones it is weighted to
        (cuffs or sleeves on their own bones hang off the torso in some rigs)"""
        M = self.blend(self.J, self.W)
        m, h = self.m, self.h
        raw = lambda n: (np.asarray(m.bones[h[n]].pos, float) if h.get(n) is not None else None)

        def seg_dist(chain):
            ps = [raw(n) for n in chain]
            ps = [p for p in ps if p is not None]
            d = np.full(len(self.P), np.inf)
            for a, b in zip(ps[:-1], ps[1:]):
                ab = b - a
                t = np.clip((self.P - a) @ ab / max(float(ab @ ab), 1e-12), 0, 1)
                d = np.minimum(d, np.linalg.norm(self.P - (a + t[:, None] * ab), axis=1))
            return d
        body = np.minimum(seg_dist(["hips", "spine", "chest", "upperChest", "neck", "head"]),
                          np.minimum(seg_dist(["hips", "leftUpperLeg", "leftLowerLeg", "leftFoot"]),
                                     seg_dist(["hips", "rightUpperLeg", "rightLowerLeg", "rightFoot"])))
        for sd in ("left", "right"):
            ua, la_, hd = raw(sd + "UpperArm"), raw(sd + "LowerArm"), raw(sd + "Hand")
            if ua is None or la_ is None or hd is None:
                continue
            tip = hd + (hd - la_) * 0.45
            da = seg_dist([sd + "UpperArm", sd + "LowerArm", sd + "Hand"])
            ab = tip - hd
            t = np.clip((self.P - hd) @ ab / max(float(ab @ ab), 1e-12), 0, 1)
            da = np.minimum(da, np.linalg.norm(self.P - (hd + t[:, None] * ab), axis=1))
            on_arm = (da < 0.06 * self.H) & (da < body * 0.8)
            arm_bone = h.get(sd + "LowerArm")
            if on_arm.any() and arm_bone is not None:
                M[on_arm] = self.Mb[arm_bone]
        return M

    def blend(self, J, W):
        Jc = np.where(J < 0, len(self.Mb) - 1, np.minimum(J, len(self.Mb) - 1))
        Wn = np.where(J < 0, 0.0, W)
        Wn = Wn / np.maximum(Wn.sum(1, keepdims=True), 1e-9)
        return np.einsum("nk,nkab->nab", Wn, self.Mb[Jc])

    def to_gen(self, X, J, W, normals=False):
        M = self.blend(J, W)
        if normals:
            Y = np.einsum("nab,nb->na", M[:, :3, :3], X)
            return Y / np.maximum(np.linalg.norm(Y, axis=1), 1e-9)[:, None]
        return np.einsum("nab,nb->na", M[:, :3, :3], X) + M[:, :3, 3]

    def from_gen(self, X, J, W, normals=False):
        M = self.blend(J, W)
        Mi = np.linalg.inv(M)
        if normals:
            Y = np.einsum("nab,nb->na", np.transpose(M[:, :3, :3], (0, 2, 1)), X)  # rotation part: inverse = transpose
            return Y / np.maximum(np.linalg.norm(Y, axis=1), 1e-9)[:, None]
        return np.einsum("nab,nb->na", Mi[:, :3, :3], X) + Mi[:, :3, 3]

    def _landmarks(self):
        bp = self.bp
        f = bodyfill.Fitter(self.m, self.h, set(), set())
        self.FWD = bodyfill.model_forward(f)
        la, ra = bp("leftUpperArm"), bp("rightUpperArm")
        ll, rl = bp("leftUpperLeg"), bp("rightUpperLeg")
        if la is None or ra is None or ll is None or rl is None:
            raise ValueError("нет костей плеч или бёдер")
        L = la - ra
        L[1] = 0
        self.LEFT = L / max(np.linalg.norm(L), 1e-9)
        self.yS = float((la[1] + ra[1]) / 2)
        self.yL = float((ll[1] + rl[1]) / 2)
        self.T = max(1e-3, self.yS - self.yL)
        self.c = (la + ra) / 2
        self.c[1] = 0
        nk = bp("neck")
        self.neck = nk if nk is not None else np.array([self.c[0], self.yS + 0.1 * self.T, self.c[2]])
        self.halfW = float(np.linalg.norm((la - ra)[[0, 2]])) / 2
        # apex of the chest: most forward body point in the chest band, near the bust lines
        P, T = self.P, self.T
        side = (P - self.c) @ self.LEFT
        fw = (P - self.c) @ self.FWD
        band = (P[:, 1] < self.yS - 0.08 * T) & (P[:, 1] > self.yS - 0.6 * T) & (np.abs(side) < 0.75 * self.halfW) & \
               (self.w_arm < 0.3)
        if band.sum() > 20:
            ys = np.linspace(self.yS - 0.08 * T, self.yS - 0.6 * T, 40)
            best = None
            for y in ys:
                sel = band & (np.abs(P[:, 1] - y) < T * 0.02)
                if sel.sum() > 3:
                    z = float(np.max(fw[sel]))
                    if best is None or z > best[1] + 1e-6:
                        best = (float(y), z)
            self.yA = best[0] if best else self.yS - 0.27 * T
            self.zA = best[1] if best else 0.1 * T
        else:
            self.yA, self.zA = self.yS - 0.27 * T, 0.1 * T
        self.yU = self.yA - 0.13 * T                        # under the bust
        self.yW = self.yL + 0.42 * T                        # natural waist
        self.yHip = self.yL + 0.16 * T                      # low waistband
        self.yNb = self.yS + 0.07 * T                       # base of the neck
        # limbs: polylines (joint -> joint), for the distance along the arm / leg
        self.limbs = {}
        self.r_arm = 0.07 * self.T
        for sd in ("left", "right"):
            a = [bp(sd + n) for n in ("UpperArm", "LowerArm", "Hand")]
            g = [bp(sd + n) for n in ("UpperLeg", "LowerLeg", "Foot")]
            if all(x is not None for x in a):
                self.limbs[sd + "Arm"] = a
            if all(x is not None for x in g):
                self.limbs[sd + "Leg"] = g
        # thickness of the upper arm (gen pose: arms horizontal)
        rs = []
        for sd in ("left", "right"):
            if sd + "Arm" in self.limbs:
                a, b, _ = self.limbs[sd + "Arm"]
                d_ = (b - a) / max(np.linalg.norm(b - a), 1e-9)
                q = self.P - a
                t = q @ d_
                sel = (self.w_arm > 0.8) & (t > 0.2 * np.linalg.norm(b - a)) & (t < 0.8 * np.linalg.norm(b - a))
                if sel.sum() > 20:
                    rs.append(np.median(np.linalg.norm(q[sel] - np.outer(t[sel], d_), axis=1)))
        if rs:
            self.r_arm = float(np.median(rs))

    def arm_axis_dist(self, V, sd):
        """distance from the arm axis (shoulder -> wrist line, gen pose)"""
        a, b, c = self.limbs[sd + "Arm"]
        d_ = (c - a) / max(np.linalg.norm(c - a), 1e-9)
        q = V - a
        return np.linalg.norm(q - np.outer(q @ d_, d_), axis=1)

    def limb_s(self, Q, name):
        """normalised position along a limb (0 at the first joint, 0.5 at the middle joint, 1 at the end)"""
        a, b, c = self.limbs[name]
        def proj(p0, p1):
            d = p1 - p0
            L2 = max(float(d @ d), 1e-12)
            return np.clip((Q - p0) @ d / L2, -2, 2)
        t1 = proj(a, b)
        t2 = proj(b, c)
        return np.where(t1 < 1.0, t1 * 0.5, 0.5 + np.clip(t2, 0, 2) * 0.5)

    def nearest(self, Q, cell=None):
        cell = cell or self.H / 40.0
        j, d = bodyfill.nearest(Q, self.P, cell)
        return j, d


# ----------------------------------------------------------------------------------------------------
# coverage: positive where the garment covers the body (model units)
# ----------------------------------------------------------------------------------------------------
def coverage(B, g, V, j, opt):
    """g: garment spec, V: shell vertices, j: index of the nearest body point of every vertex"""
    T, H = B.T, B.H
    shift = float(opt.get("shift", 0.0)) * T
    length = float(opt.get("length", 1.0))
    y = V[:, 1]
    rel = V - B.c
    side = rel @ B.LEFT
    fw = rel @ B.FWD
    jj = np.clip(j, 0, len(B.P) - 1)
    w_arm, w_leg, w_head = B.w_arm[jj], B.w_leg[jj], B.w_head[jj]
    w_hand, w_foot = B.w_hand[jj], B.w_foot[jj]
    big = H
    c = np.full(len(V), big)
    k = 0.08 * T  # how weights are turned into a distance for the min/max combination

    # tails come out through a round opening
    c = np.minimum(c, B.tail_hole(V))
    if g["kind"] != "top":  # (tops are limited geometrically below)
        c = np.minimum(c, (0.5 - np.maximum(np.maximum(w_hand, w_foot), w_head)) * k)

    if g["kind"] == "top":
        # --- top edge
        neck = g.get("neck", "crew")
        sw = float(g.get("straps") or 0.04) * T
        sx = 0.55 * B.halfW
        strap_band = sw / 2 - np.minimum(np.abs(side - sx), np.abs(side + sx))  # > 0 inside a strap
        yflat = {"bust": B.yA + 0.07 * T, "bra": B.yA + 0.045 * T}.get(g.get("top", "bust"), B.yA + 0.07 * T)
        if neck == "turtle":
            ytop = B.neck[1] + 0.10 * T - y
        elif neck == "flat":
            ytop = yflat - y
        elif neck == "straps":
            ytop = np.maximum(yflat - y, np.minimum(strap_band, (B.yS + 0.14 * T) - y))
        else:
            front = np.clip(fw / (0.35 * T), 0, 1)
            w = 0.36 * T
            u = np.clip(np.abs(side) / w, 0, 1)
            shape = {"crew": 1 - u * u, "scoop": 1 - u ** 2.5, "v": 1 - u}.get(neck, 1 - u * u)
            depth = float(g.get("depth", 0.06)) * T
            ytop = (B.yNb - y) - depth * front * shape - 0.02 * T * front
            if g.get("straps"):  # tank top: above the chest only the (wide) straps go up to the shoulders
                ytop = np.minimum(ytop, np.maximum((B.yA + 0.10 * T) - y, strap_band))
        c = np.minimum(c, ytop)
        # --- bottom edge
        ybot = {"hip": B.yL + 0.02 * T, "hip_low": B.yL - 0.01 * T, "crop": B.yW + 0.05 * T, "underbust": B.yU,
                "waist": B.yW}.get(g.get("bottom", "hip"), B.yL)
        ybot = ybot - shift - (length - 1.0) * 0.3 * T if g.get("bottom") not in ("underbust",) else ybot - shift
        c = np.minimum(c, y - ybot)
        # --- arms (generation pose: arms horizontal): the armhole is the plane at the shoulder joint,
        # the sleeve ends at its length along the arm; points far from the arm axis are torso
        sl = float(g.get("sleeve", 0.0)) * (length if g.get("sleeve", 0) > 0 else 1.0)
        if neck in ("straps", "flat") or g.get("straps"):
            sl = 0.0
        L_arm = 0.9 * T
        for sd in ("left", "right"):
            if sd + "Arm" not in B.limbs:
                continue
            mine = (side > 0) if sd == "left" else (side <= 0)
            s = B.limb_s(V, sd + "Arm")
            far = B.arm_axis_dist(V, sd) - 2.3 * B.r_arm
            end = (sl - s) * L_arm if sl > 0 else (0.02 - s) * L_arm
            c = np.where(mine, np.minimum(c, np.maximum(end, far)), c)
        # hands and the head are never covered
        c = np.minimum(c, (0.5 - np.maximum(w_hand, w_head)) * k)
        if g.get("open_front"):
            gap = float(g["open_front"]) * T
            front = fw > 0
            c = np.where(front & (y < B.yNb), np.minimum(c, np.abs(side) - gap), c)
    else:
        top = {"waist": B.yW, "hipline": B.yHip, "low": B.yL + 0.08 * T}.get(g.get("top", "waist"), B.yW)
        ytop = (top + shift) - y
        if g.get("top") == "low":  # briefs: lower in front, higher at the sides (leg openings)
            front = np.clip(fw / (0.3 * T), -1, 1)
            ytop = ytop - 0.03 * T * np.maximum(front, 0)
        c = np.minimum(c, ytop)
        ln = float(g.get("leg", 0.98)) * length
        s_leg = np.zeros(len(V))
        for sd in ("left", "right"):
            if sd + "Leg" in B.limbs:
                s = B.limb_s(V, sd + "Leg")
                mine = (side > 0) if sd == "left" else (side <= 0)
                s_leg = np.where(mine, s, s_leg)
        L_leg = 1.2 * T
        if ln <= 0.005:
            # bikini: legs fully open, cut along the crotch line of the weights
            leg_term = (0.62 - w_leg) * k
        else:
            leg_term = np.maximum((0.5 - w_leg) * k, (ln - s_leg) * L_leg)
        c = np.minimum(c, leg_term)
        c = np.minimum(c, (0.5 - w_arm) * k)
    return c


# ----------------------------------------------------------------------------------------------------
# mesh helpers
# ----------------------------------------------------------------------------------------------------
def cut_mesh(V, T, c, extra=(), pick=()):
    """keep the part of the mesh where c > 0; triangles crossing the level are split (clean edge).
    extra: per-vertex arrays interpolated to the new vertices.  Returns V, T, extras."""
    pos = c > 0
    keep_all = pos[T].all(1)
    cross = pos[T].any(1) & ~keep_all
    newV = [V]
    newE = [list(extra)]
    edge_id = {}
    nv = [len(V)]
    tris = [T[keep_all]]

    def edge_point(a, b):
        key = (a, b) if a < b else (b, a)
        if key in edge_id:
            return edge_id[key]
        t = c[a] / (c[a] - c[b])
        p = V[a] + (V[b] - V[a]) * t
        newV.append(p[None])
        for e_, arr in zip(newE[0], extra):
            pass
        idx = nv[0]
        nv[0] += 1
        edge_id[key] = (idx, a, b, t)
        return edge_id[key]

    out = []
    for tri in T[cross]:
        ins = [v for v in tri if pos[v]]
        outs = [v for v in tri if not pos[v]]
        # keep winding: rotate so the pattern starts at index 0
        a, b, cc = tri
        if len(ins) == 1:
            i = ins[0]
            k = list(tri).index(i)
            p, q = tri[(k + 1) % 3], tri[(k + 2) % 3]
            e1 = edge_point(i, p)[0]
            e2 = edge_point(i, q)[0]
            out.append((i, e1, e2))
        else:
            o = outs[0]
            k = list(tri).index(o)
            p, q = tri[(k + 1) % 3], tri[(k + 2) % 3]
            e1 = edge_point(o, p)[0]
            e2 = edge_point(o, q)[0]
            out.append((e1, p, q))
            out.append((e1, q, e2))
    if out:
        tris.append(np.array(out, np.int64))
    V2 = np.concatenate(newV) if len(newV) > 1 else V
    ex2 = []
    if edge_id:
        info = sorted(edge_id.values())
        ia = np.array([x[1] for x in info])
        ib = np.array([x[2] for x in info])
        tt = np.array([x[3] for x in info])
        nearer = np.where(tt < 0.5, ia, ib)
        for k_, arr in enumerate(extra):
            arr = np.asarray(arr)
            if k_ in pick:  # bone ids / weights: take them whole from the nearer end (never average ids)
                mid = arr[nearer]
            else:
                sh = (-1,) + (1,) * (arr.ndim - 1)
                mid = arr[ia] * (1 - tt.reshape(sh)) + arr[ib] * tt.reshape(sh)
            ex2.append(np.concatenate([arr, mid.astype(arr.dtype)]))
    else:
        ex2 = [np.asarray(a) for a in extra]
    T2 = np.concatenate(tris) if tris else np.zeros((0, 3), np.int64)
    # drop unused vertices
    used = np.zeros(len(V2), bool)
    used[T2.reshape(-1)] = True
    remap = -np.ones(len(V2), np.int64)
    remap[used] = np.arange(used.sum())
    return V2[used], remap[T2], [a[used] for a in ex2]


def components(T, n):
    """label of the connected component of every vertex"""
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in np.concatenate([T[:, [0, 1]], T[:, [1, 2]]]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n)])


def smooth_scalar(f, T, n, it=8):
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    deg = np.bincount(e[:, 0], minlength=n).astype(float)
    for _ in range(it):
        acc = np.zeros(n)
        np.add.at(acc, e[:, 0], f[e[:, 1]])
        f = 0.5 * f + 0.5 * acc / np.maximum(deg, 1)
    return f


def grid_sdf(B, lo, step, dims, off, band):
    """signed distance to the body (normal sign) on a grid, exact only near the surface:
    a coarse pass finds the band of nodes close to the body, the fine pass queries only those."""
    nx, ny, nz = dims
    cs = max(step * 4, band)
    cnx, cny, cnz = (int(np.ceil(nx * step / cs)) + 1, int(np.ceil(ny * step / cs)) + 1, int(np.ceil(nz * step / cs)) + 1)
    C = np.stack(np.meshgrid(lo[0] + np.arange(cnx) * cs, lo[1] + np.arange(cny) * cs, lo[2] + np.arange(cnz) * cs,
                             indexing="ij"), -1).reshape(-1, 3)
    jc, dc = bodyfill.nearest(C, B.P, cs * 1.5)
    dc = dc.reshape(cnx, cny, cnz)
    # fine node -> its coarse node
    ix = np.minimum(np.round(np.arange(nx) * step / cs).astype(int), cnx - 1)
    iy = np.minimum(np.round(np.arange(ny) * step / cs).astype(int), cny - 1)
    iz = np.minimum(np.round(np.arange(nz) * step / cs).astype(int), cnz - 1)
    near = dc[np.ix_(ix, iy, iz)] < band + cs * 0.9
    F = np.full((nx, ny, nz), band, float)
    idx = np.nonzero(near)
    if len(idx[0]) == 0:
        return F
    Q = np.stack([lo[0] + idx[0] * step, lo[1] + idx[1] * step, lo[2] + idx[2] * step], 1)
    j, d = bodyfill.nearest(Q, B.P, band)
    ok = j >= 0
    sgn = np.sign(np.einsum("ij,ij->i", Q[ok] - B.P[j[ok]], B.N[j[ok]]))
    sgn[sgn == 0] = 1
    vals = np.full(len(Q), band)
    vals[ok] = sgn * d[ok]
    F[idx] = vals
    return F


# ----------------------------------------------------------------------------------------------------
# building one built-in garment
# ----------------------------------------------------------------------------------------------------
def build_garment(B, g, opt, log=print):
    """returns V, N, UV, T, J, W (model space) or None"""
    H, T = B.H, B.T
    size = float(opt.get("size", 1.0))
    loose = float(opt.get("loose", g.get("loose", 0.3)))
    off = (0.0035 + 0.0022 * float(g.get("thick", 1.0))) * H * size + 0.004 * H * max(0.0, loose) * size
    off = max(off, 0.002 * H)
    # region of interest (bounding box) from the coverage on the body points
    jb = np.arange(len(B.P))
    cb = coverage(B, g, B.P, jb, opt)  # body points carry their own labels
    sel = cb > -0.05 * T
    if sel.sum() < 30:
        log("  ! гардероб: «%s» — на теле нет области для этой вещи" % g["name"])
        return None
    pad = off * 3 + 0.03 * T
    lo = B.P[sel].min(0) - pad
    hi = B.P[sel].max(0) + pad
    step = H / 190.0
    nx, ny, nz = (np.ceil((hi - lo) / step).astype(int) + 1)
    if nx * ny * nz > 6_000_000:
        step = ((hi - lo).prod() / 6_000_000) ** (1 / 3)
        nx, ny, nz = (np.ceil((hi - lo) / step).astype(int) + 1)
    gx = lo[0] + np.arange(nx) * step
    gy = lo[1] + np.arange(ny) * step
    gz = lo[2] + np.arange(nz) * step
    band = max(off * 1.6, step * 2.5)
    F = grid_sdf(B, lo, step, (nx, ny, nz), off, band) - off
    V, Tm = bodyfill.surface_nets(F, lo, step)
    if len(Tm) == 0:
        return None
    # keep the outer shell only (drop pieces that lie inside the body)
    jv, dv = bodyfill.nearest(V, B.P, band)
    okv = jv >= 0
    sv = np.zeros(len(V))
    sv[okv] = np.einsum("ij,ij->i", V[okv] - B.P[jv[okv]], B.N[jv[okv]])
    lab = components(Tm, len(V))
    good = np.zeros(len(V), bool)
    for L_ in np.unique(lab):
        mm = lab == L_
        if mm.sum() > 30 and np.median(sv[mm]) > 0:
            good |= mm
    keep = good[Tm].all(1)
    Tm = Tm[keep]
    used = np.unique(Tm)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    V, Tm = V[used], remap[Tm]
    V = bodyfill.smooth_mesh(V, Tm, it=3, lam=0.5)
    # loose garments: relax (hangs over hollows), then push back out of the body
    it = int(round(max(0.0, loose) * 14))
    for r in range(it):
        V = bodyfill.smooth_mesh(V, Tm, it=3, lam=0.6)
        jv, dv = bodyfill.nearest(V, B.P, band * 1.5)
        okv = jv >= 0
        n = B.N[jv[okv]]
        s = np.einsum("ij,ij->i", V[okv] - B.P[jv[okv]], n)
        need = s < off * 0.85
        idx = np.nonzero(okv)[0][need]
        V[idx] += n[need] * (off * 0.85 - s[need])[:, None]
    # coverage on the shell, smoothed along the surface for clean hems
    jv, dv = bodyfill.nearest(V, B.P, band * 2)
    miss = jv < 0
    if miss.any():
        j2, _ = bodyfill.nearest(V[miss], B.P, B.H / 25)
        jv[miss] = j2
    c = coverage(B, g, V, jv, opt)
    c = smooth_scalar(c, Tm, len(V), it=10)
    jvc = np.where(jv >= 0, jv, 0)
    J = B.J[jvc].astype(np.int64)
    W = B.W[jvc].astype(float)
    V, Tm, (J, W, c2) = cut_mesh(V, Tm, c, (J, W, c), pick=(0, 1))
    if len(Tm) < 10:
        log("  ! гардероб: «%s» — пустая вещь после обрезки" % g["name"])
        return None
    # drop crumbs
    lab = components(Tm, len(V))
    cnt = np.bincount(np.unique(lab, return_inverse=True)[1])
    ul, inv = np.unique(lab, return_inverse=True)
    big = cnt[inv] >= max(40, 0.01 * len(V))
    keep = big[Tm].all(1)
    Tm = Tm[keep]
    used = np.unique(Tm)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    V, Tm, J, W = V[used], remap[Tm], J[used], W[used]
    # the fabric hugs the tail at the opening (no flappy gap)
    if getattr(B, "tails", None):
        hole = B.tail_hole(V)
        near = hole < 0.03 * H
        if near.any():
            jv3, _ = bodyfill.nearest(V[near], B.P, B.H / 20)
            ok3 = jv3 >= 0
            idx3 = np.nonzero(near)[0][ok3]
            p3, n3 = B.P[jv3[ok3]], B.N[jv3[ok3]]
            sd3 = np.einsum("ij,ij->i", V[idx3] - p3, n3)
            w3 = np.clip(1 - hole[idx3] / (0.03 * H), 0, 1)[:, None]
            tgt = np.minimum(sd3, 0.003 * H)
            V[idx3] = V[idx3] - n3 * ((sd3 - tgt)[:, None] * w3)
    # cut vertices take the weights of one parent (interpolated weights may mix bone ids): renormalise
    W = np.where(J >= 0, W, 0)
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    # UV islands (in the generation pose): torso around the body axis, each leg around its own axis,
    # each arm around the arm (arms are horizontal here); triangles take the island of most of their corners
    N0 = bodyfill.vertex_normals(V, Tm)  # before the UV split, so the islands share smooth normals
    jv2, _ = bodyfill.nearest(V, B.P, B.H / 25)
    jv2 = np.maximum(jv2, 0)
    rel = V - B.c
    side = rel @ B.LEFT
    region = np.zeros(len(V), int)  # 0 torso, 1 left leg, 2 right leg, 3 left arm, 4 right arm
    region[(B.w_leg[jv2] > 0.5) & (side > 0)] = 1
    region[(B.w_leg[jv2] > 0.5) & (side <= 0)] = 2
    region[(B.w_arm[jv2] > 0.5) & (side > 0)] = 3
    region[(B.w_arm[jv2] > 0.5) & (side <= 0)] = 4
    tile = 0.075 * H
    UVall = np.zeros((5, len(V), 2))
    per = np.zeros(5)
    ang = np.arctan2(side, rel @ B.FWD)
    per[0] = 2 * math.pi * 0.16 * T / tile
    UVall[0] = np.stack([ang / (2 * math.pi) * per[0], V[:, 1] / tile], 1)
    for r_, sd in ((1, "left"), (2, "right")):
        hp = B.bp(sd + "UpperLeg")
        if hp is None:
            UVall[r_] = UVall[0]
            per[r_] = per[0]
            continue
        q = V - hp
        a_ = np.arctan2(q @ B.LEFT, q @ B.FWD)
        per[r_] = 2 * math.pi * 0.09 * T / tile
        UVall[r_] = np.stack([a_ / (2 * math.pi) * per[r_], V[:, 1] / tile], 1)
    for r_, sd in ((3, "left"), (4, "right")):
        sh = B.bp(sd + "UpperArm")
        if sh is None:
            UVall[r_] = UVall[0]
            per[r_] = per[0]
            continue
        q = V - sh
        a_ = np.arctan2(q[:, 1], q @ B.FWD)
        per[r_] = 2 * math.pi * 0.05 * T / tile
        UVall[r_] = np.stack([a_ / (2 * math.pi) * per[r_], np.abs(q @ B.LEFT) / tile], 1)
    # region of each triangle: majority of its corners
    rt = region[Tm]
    tri_r = np.where((rt[:, 0] == rt[:, 1]) | (rt[:, 0] == rt[:, 2]), rt[:, 0], rt[:, 1])
    cor_v = Tm.reshape(-1)
    cor_r = np.repeat(tri_r, 3)
    cor_uv = UVall[cor_r, cor_v].reshape(-1, 3, 2)
    # wrap seam inside a triangle: lift the low corners by one period
    pr = per[tri_r]
    us = cor_uv[:, :, 0]
    wrap = (us.max(1) - us.min(1)) > pr / 2
    lift = wrap[:, None] & (us < (us.max(1)[:, None] - pr[:, None] / 2))
    cor_uv[:, :, 0] = np.where(lift, us + pr[:, None], us)
    cor_uv = cor_uv.reshape(-1, 2)
    # weld corners that share vertex, region and UV
    key = np.stack([cor_v, cor_r, np.round(cor_uv[:, 0] * 1000).astype(np.int64)], 1)
    uk, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    newv = cor_v[first]
    UV = cor_uv[first]
    Tm = inv.reshape(-1, 3)
    V, J, W, N = V[newv], J[newv], W[newv], N0[newv]
    # back from the generation pose to the model's own pose
    V = B.from_gen(V, J, W)
    N = B.from_gen(N, J, W, normals=True)
    return V, N, UV.astype(np.float32), Tm, J, W


# ----------------------------------------------------------------------------------------------------
# entry point from the converter
# ----------------------------------------------------------------------------------------------------
def apply_wardrobe(m, h, items, mat_cats, log=print):
    """items: [{"src": "builtin", "id": ..., "color": ..., "color2": ..., "size", "loose", "shift", "length",
    "brush": [...]}, ...].  mat_cats: category of every material (index -> 'skin' / 'clothes' / ...).
    Appends the garments to the model as new materials "WD <name>".  Returns the list of added material names."""
    if not items:
        return []
    skin = {k for k, cat in mat_cats.items() if cat in ("skin",)}
    cloth = {k for k, cat in mat_cats.items() if cat not in ("skin", "face", "eyes")}
    if not skin:
        log("  ! гардероб: нет материала кожи — отметьте категорию «Кожа» у материала тела")
        return []
    P_ = np.asarray(m.positions)
    sig = (len(P_), round(float(P_[::997].astype(float).sum()), 4), tuple(mt.name for mt in m.materials))
    key = (sig, tuple(sorted(skin)), tuple(sorted(h.items())))
    def body(spread):
        # tops are built with the legs as they are (spreading them would wrinkle the hem at the hips);
        # trousers with the legs a little apart (the legs of the trousers do not stick together)
        B_ = _CACHE.get(("body", spread) + key)
        if B_ is None:
            B_ = Body(m, h, skin, cloth, log, leg_spread=spread)
            if len(_CACHE) > 16:
                _CACHE.clear()
            _CACHE[("body", spread) + key] = B_
        return B_
    added = []
    skin_best = max(skin, key=lambda k: m.materials[k].index_count)
    # layers: underwear first, then the order of the list (later = on top)
    UNDER = {"bra", "bikini_top", "bikini_bottom", "briefs", "boxers"}
    order = sorted(range(len(items)), key=lambda i: (0 if items[i].get("id") in UNDER else 1, i))
    pieces = []  # (item, res, tex, name, mat_params or None, B)
    for n_ in order:
        it = items[n_]
        try:
            if it.get("src", "builtin") == "builtin":
                g = BUILTIN_BY_ID.get(it.get("id"))
                if not g:
                    log("  ! гардероб: неизвестная вещь %s" % it.get("id"))
                    continue
                opt = {k: it[k] for k in ("size", "loose", "shift", "length") if k in it}
                B = body(0.0 if g["kind"] == "top" else 9.0)
                gkey = ("g",) + key + (json.dumps({"g": g, "o": opt}, sort_keys=True),)
                res = _CACHE.get(gkey)
                if res is None:
                    res = build_garment(B, g, opt, log)
                    _CACHE[gkey] = res
                if res is None:
                    continue
                tex = fabric_texture(it.get("pattern") or g["pattern"], it.get("color") or g["color"],
                                     it.get("color2") or g.get("color2"))
                pieces.append((it, res, tex, "WD %s" % (it.get("label") or g["name"]), None, B))
                log("  гардероб: %s — %d треугольников" % (g["name"], len(res[3])))
            else:
                import wardrobe_transfer
                for res, tex, name, mat, B in wardrobe_transfer.apply_item(m, h, body, it, skin_best, log):
                    pieces.append((it, res, tex, name, mat, B))
        except Exception as e:
            log("  ! гардероб: %s — %s" % (it.get("id"), e))
    pieces = _layer(pieces, body(0.0).H, log)
    for it, res, tex, name, mat, B in pieces:
        _append(m, res, skin_best, tex, name, it, B)
        if mat:
            _apply_mat_params(m.materials[-1], mat)
        added.append(m.materials[-1].name)
    return added


def _boundary_mask(V, T):
    """vertices on the open edges of a mesh (welded by position, so UV seams do not count)"""
    key = np.round(V * 4000).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    Tw = inv[T]
    e = np.sort(np.concatenate([Tw[:, [0, 1]], Tw[:, [1, 2]], Tw[:, [2, 0]]]), 1)
    u, c = np.unique(e, axis=0, return_counts=True)
    bw = np.zeros(inv.max() + 1, bool)
    bw[u[c == 1].reshape(-1)] = True
    return bw[inv]


def _layer(pieces, H, log=print):
    """Layers: where an outer garment covers an inner one closely (or the inner one pokes through it), that
    part of the inner garment is cut away — it is hidden anyway — so nothing flickers through the outer one.
    The cut runs under the outer garment, a little inside its hem."""
    gap = 0.006 * H
    res_list = [list(p[1]) for p in pieces]
    for o in range(1, len(pieces)):
        Vo, No, To = res_list[o][0], res_list[o][1], res_list[o][3]
        if not len(To):
            continue
        Non = No / np.maximum(np.linalg.norm(No, axis=1), 1e-9)[:, None]
        edge = _boundary_mask(Vo, To)
        el = float(np.median(np.linalg.norm(Vo[To[:, 0]] - Vo[To[:, 1]], axis=1)))
        # outer vertices well inside the garment (2+ rings away from the hem): only they "cover"
        e = np.concatenate([To[:, [0, 1]], To[:, [1, 2]], To[:, [2, 0]]])
        e = np.concatenate([e, e[:, ::-1]])
        near_edge = edge.copy()
        for _k in range(2):
            ne = near_edge.copy()
            ne[e[near_edge[e[:, 0]], 1]] = True
            near_edge = ne
        for i in range(o):
            V, N, UV, T, J, W = res_list[i]
            if not len(T):
                continue
            j, d = bodyfill.nearest(V, Vo, max(gap * 3, el * 2))
            ok = j >= 0
            c = np.ones(len(V)) * gap
            idx = np.nonzero(ok)[0]
            if len(idx):
                n = Non[j[idx]]
                rel = V[idx] - Vo[j[idx]]
                sd = np.einsum("ij,ij->i", rel, n)
                tang = np.linalg.norm(rel - sd[:, None] * n, axis=1)
                covered = (~near_edge[j[idx]]) & (tang < 1.5 * el) & (sd < gap)
                c[idx[covered]] = -gap
            if (c < 0).sum() < 3:
                continue
            c = smooth_scalar(c, T, len(V), it=3)
            V2, T2, (N2, UV2, J2, W2) = cut_mesh(V, T, c, (N, UV, J, W), pick=(2, 3))
            res_list[i] = [V2, N2, UV2.astype(np.float32), T2, J2, W2]
    return [(p[0], tuple(r), p[2], p[3], p[4], p[5]) for p, r in zip(pieces, res_list)]


def _smooth_masked(V, T, idx, it=2):
    """light smoothing of the moved vertices and their neighbours (no steps at the edge of the pushed area)"""
    n = len(V)
    mask = np.zeros(n, bool)
    mask[idx] = True
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    for _ in range(it):
        mask2 = mask.copy()
        mask2[e[mask[e[:, 0]], 1]] = True
        mask = mask2
    deg = np.bincount(e[:, 0], minlength=n).astype(float)
    for _ in range(it):
        acc = np.zeros((n, 3))
        np.add.at(acc, e[:, 0], V[e[:, 1]])
        avg = acc / np.maximum(deg, 1)[:, None]
        V = np.where(mask[:, None], 0.6 * V + 0.4 * avg, V)
    return V


def _apply_mat_params(mt, mat):
    try:
        mt.diffuse = tuple(mat["diffuse"])
        mt.ambient = tuple(mat["ambient"])
        mt.specular = tuple(mat["specular"])
        mt.flags = int(mat["flags"]) | 0x01
        mt.edge_color = tuple(mat["edge_color"])
        mt.edge_size = float(mat["edge_size"])
        if mat.get("toon", -1) >= 0 and mat.get("toon_shared"):
            mt.toon_shared, mt.toon = 1, int(mat["toon"])
    except Exception:
        pass


def _append(m, res, skin_k, tex_path, name, it, B):
    V, N, UV, T, J, W = res
    V = V.copy()
    names = {mt_.name for mt_ in m.materials}
    if name in names:  # the same garment twice: keep names unique
        k_ = 2
        while "%s (%d)" % (name, k_) in names:
            k_ += 1
        name = "%s (%d)" % (name, k_)
    owner = getattr(m, "_wd_owner", None)
    if owner is None:
        owner = {}
        m._wd_owner = owner
    owner[name] = {"uid": it.get("uid") or ("auto-%s" % it.get("id", "")), "brush": list(it.get("brush") or [])}
    if tex_path not in m.textures:
        m.textures = list(m.textures) + [tex_path]
    ti = list(m.textures).index(tex_path)
    n0 = len(m.positions)
    J4 = np.full((len(V), 4), -1, np.int64)
    W4 = np.zeros((len(V), 4), float)
    J4[:, :min(4, J.shape[1])] = J[:, :4]
    W4[:, :min(4, W.shape[1])] = W[:, :4]
    m.positions = np.concatenate([m.positions, V.astype(m.positions.dtype)])
    m.normals = np.concatenate([m.normals, N.astype(m.normals.dtype)])
    m.uvs = np.concatenate([m.uvs, UV.astype(m.uvs.dtype)])
    m.joints = np.concatenate([m.joints, J4.astype(m.joints.dtype)])
    m.weights = np.concatenate([m.weights, W4.astype(m.weights.dtype)])
    m.indices = np.concatenate([np.asarray(m.indices), (T + n0).reshape(-1).astype(np.asarray(m.indices).dtype)])
    import copy as _copy
    mt = _copy.copy(m.materials[skin_k])
    mt.name = name
    mt.name_en = "Wardrobe " + str(it.get("id", ""))
    mt.index_count = len(T) * 3
    mt.texture = ti
    mt.diffuse = (1.0, 1.0, 1.0, 1.0)
    mt.ambient = (0.55, 0.55, 0.55)
    mt.specular = (0.0, 0.0, 0.0)
    mt.sphere = -1
    mt.flags = (mt.flags | 0x01)  # double-sided: the inside shows at the openings
    m.materials = list(m.materials) + [mt]
