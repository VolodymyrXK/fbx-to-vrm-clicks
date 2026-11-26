"""Body completion: a smooth body built from the skeleton that fills the places where the model has
no skin (under removed shoes, behind a removed hat, under clothing that was the only surface).

The body is an implicit surface (signed distance field) made of rounded cones and ellipsoids around the
humanoid bones, blended smoothly; the parts are sized from the model's own mesh.  It sits a little inside
the existing surfaces, so it shows only where something is missing.  Feet get five toes; heads get a full
skull.  No nipples, no genitals: the torso is a smooth form.

Surface extraction is a small vectorised Surface Nets (no extra dependencies).
"""
import numpy as np

# --------------------------------------------------------------------------
# signed distance primitives (vectorised over points P (N,3))
# --------------------------------------------------------------------------


def sd_round_cone(P, a, b, ra, rb):
    """rounded cone between spheres (a, ra) and (b, rb)"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ba = b - a
    l2 = float(ba @ ba)
    if l2 < 1e-12:
        return np.linalg.norm(P - a, axis=1) - max(ra, rb)
    t = np.clip(((P - a) @ ba) / l2, 0.0, 1.0)
    r = ra + (rb - ra) * t
    return np.linalg.norm(P - (a + t[:, None] * ba), axis=1) - r


def sd_ellipsoid(P, c, rad, R=None):
    """approximate SDF of an ellipsoid (centre c, radii rad, optional rotation R whose columns are axes)"""
    q = P - np.asarray(c, float)
    if R is not None:
        q = q @ R
    rad = np.maximum(np.asarray(rad, float), 1e-4)
    k0 = np.linalg.norm(q / rad, axis=1)
    k1 = np.linalg.norm(q / (rad * rad), axis=1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def smin(a, b, k):
    """smooth union"""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


# --------------------------------------------------------------------------
# surface nets
# --------------------------------------------------------------------------


def surface_nets(F, origin, step):
    """F: (nx,ny,nz) SDF samples (negative inside).  Returns vertices (V,3) and triangles (T,3),
    triangles wound so that normals point outwards (towards positive F)."""
    nx, ny, nz = F.shape
    inside = F < 0
    # cells: (nx-1, ny-1, nz-1); a cell gets a vertex when its 8 corners differ in sign
    c = [inside[i:nx - 1 + i, j:ny - 1 + j, k:nz - 1 + k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    cs = sum(x.astype(np.int8) for x in c)
    active = (cs > 0) & (cs < 8)
    idx = -np.ones(active.shape, np.int64)
    ci = np.argwhere(active)
    idx[active] = np.arange(len(ci))
    if len(ci) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    # vertex = mean of edge crossings of its cell
    acc = np.zeros((len(ci), 3))
    cnt = np.zeros(len(ci))
    corners = np.array([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if np.abs(corners[a] - corners[b]).sum() == 1]
    x0, y0, z0 = ci[:, 0], ci[:, 1], ci[:, 2]
    fv = [F[x0 + o[0], y0 + o[1], z0 + o[2]] for o in corners]
    for a, b in edges:
        fa, fb = fv[a], fv[b]
        m = (fa < 0) != (fb < 0)
        t = fa[m] / (fa[m] - fb[m])
        p = corners[a] + t[:, None] * (corners[b] - corners[a])
        acc[m] += p
        cnt[m] += 1
    V = ci + acc / np.maximum(cnt, 1)[:, None]
    V = np.asarray(origin) + V * step
    # quads: for every grid edge with a sign change, connect the 4 cells around it
    tris = []
    for ax in range(3):
        o1, o2 = [(1, 2), (0, 2), (0, 1)][ax]
        sl = [slice(None)] * 3
        a_ = inside[tuple(sl)]
        sh = [0, 0, 0]
        sh[ax] = 1
        A = inside[:nx - sh[0], :ny - sh[1], :nz - sh[2]]
        B = inside[sh[0]:, sh[1]:, sh[2]:]
        ch = A != B
        # edge position (x,y,z) from point p to p+e_ax; the 4 cells share it: p - e_o1*{0,1} - e_o2*{0,1}
        pts = np.argwhere(ch)
        if len(pts) == 0:
            continue
        lo = np.ones(len(pts), bool)
        for o in (o1, o2):
            lo &= pts[:, o] >= 1
        lim = np.array(idx.shape)
        ok = lo & np.all(pts < lim, axis=1)
        for o in (o1, o2):
            ok &= pts[:, o] - 1 < lim[o]
        pts = pts[ok]
        flip = A[tuple(pts.T)]  # inside at the lower end: orientation
        e1 = np.zeros(3, int)
        e1[o1] = 1
        e2 = np.zeros(3, int)
        e2[o2] = 1
        c00 = idx[tuple((pts - e1 - e2).T)]
        c10 = idx[tuple((pts - e2).T)]
        c11 = idx[tuple(pts.T)]
        c01 = idx[tuple((pts - e1).T)]
        good = (c00 >= 0) & (c10 >= 0) & (c11 >= 0) & (c01 >= 0)
        c00, c10, c11, c01, flip = c00[good], c10[good], c11[good], c01[good], flip[good]
        q1 = np.stack([c00, c10, c11], 1)
        q2 = np.stack([c00, c11, c01], 1)
        r1 = np.stack([c00, c11, c10], 1)
        r2 = np.stack([c00, c01, c11], 1)
        f = flip[:, None]
        t1 = np.where(f, q1, r1)
        t2 = np.where(f, q2, r2)
        if ax == 1:  # keep a consistent handedness over the three axes
            t1, t2 = t1[:, [0, 2, 1]], t2[:, [0, 2, 1]]
        tris += [t1, t2]
    T = np.concatenate(tris) if tris else np.zeros((0, 3), np.int64)
    return V, T


def smooth_mesh(V, T, it=4, lam=0.5):
    """a few rounds of Laplacian smoothing (keeps the voxel steps out of the surface)"""
    n = len(V)
    if n == 0 or len(T) == 0:
        return V
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    deg = np.bincount(e[:, 0], minlength=n).astype(float)
    for _ in range(it):
        acc = np.zeros_like(V)
        np.add.at(acc, e[:, 0], V[e[:, 1]])
        avg = acc / np.maximum(deg, 1)[:, None]
        V = V + lam * (avg - V) * (deg > 0)[:, None]
    return V


def vertex_normals(V, T):
    N = np.zeros_like(V)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    for k in range(3):
        np.add.at(N, T[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1), 1e-12)[:, None]


# --------------------------------------------------------------------------
# fitting the body to the model
# --------------------------------------------------------------------------

LIMBS = {
    "legs": [("UpperLeg", "LowerLeg"), ("LowerLeg", "Foot")],
    "arms": [("UpperArm", "LowerArm"), ("LowerArm", "Hand")],
}
TORSO = ["hips", "spine", "chest", "upperChest", "neck"]
PARTS = ("feet", "legs", "torso", "arms", "head")


class Fitter:
    def __init__(self, m, h, skin_mats, other_mats, hidden_mats=()):
        self.m = m
        self.h = h
        self.P = np.asarray(m.positions, float)
        J = np.asarray(m.joints)
        W = np.asarray(m.weights)
        self.main = J[np.arange(len(J)), np.argmax(W, axis=1)]
        self.mainw = W.max(axis=1)
        self.skin_v = self._verts(skin_mats)
        self.other_v = self._verts(other_mats)
        self.vis_v = (self.skin_v | self.other_v) & ~self._verts(set(hidden_mats) - set(skin_mats))
        ys = self.P[:, 1]
        self.height = float(ys.max() - ys.min()) if len(ys) else 20.0
        self.ground = float(ys.min()) if len(ys) else 0.0

    def _verts(self, mats):
        m = self.m
        sel = np.zeros(len(self.P), bool)
        off = 0
        for k, mt in enumerate(m.materials):
            if k in mats:
                sel[np.asarray(m.indices[off:off + mt.index_count])] = True
            off += mt.index_count
        return sel

    def bp(self, name):
        i = self.h.get(name)
        return None if i is None else np.asarray(self.m.bones[i].pos, float)

    def radius(self, bones, a, b, default):
        """typical distance of the skin around segment a-b (skin first, clothes x0.88 when no skin)"""
        bones = [x for x in bones if x is not None]
        sel_b = np.isin(self.main, bones) & (self.mainw > 0.4)
        for src, fac in ((self.skin_v, 0.96), (self.other_v, 0.86)):
            sel = sel_b & src
            if sel.sum() >= 24:
                d = -sd_round_cone(self.P[sel], a, b, 0.0, 0.0)  # = -distance to the segment
                return float(np.percentile(-d, 50)) * fac
        return default


def model_forward(f):
    """horizontal direction the model looks to (from the eyes, else the toes, else MMD's -Z)"""
    hd = f.bp("head")
    eyes = [f.bp(e) for e in ("leftEye", "rightEye") if f.bp(e) is not None]
    cands = []
    if hd is not None and eyes:
        cands.append(np.mean(eyes, axis=0) - hd)
    for sd in ("left", "right"):
        ft, tt = f.bp(sd + "Foot"), f.bp(sd + "Toes")
        if ft is not None and tt is not None:
            cands.append(tt - ft)
    for v in cands:
        v = np.array([v[0] * 0.0, 0.0, v[2]])  # only the front/back axis
        if np.linalg.norm(v) > 1e-4:
            return v / np.linalg.norm(v)
    return np.array([0.0, 0.0, -1.0])


def build_body(m, h, parts, skin_mats, other_mats, log=print, hidden_mats=(), opaque=None):
    """Returns (V, N, T, bone_ids (V,4), weights (V,4)) in model (MMD) space, or None."""
    f = Fitter(m, h, skin_mats, other_mats, hidden_mats)
    if opaque is not None:  # see-through skin is not skin
        f.skin_v &= opaque
        f.vis_v &= opaque
    H = f.height
    FWD = model_forward(f)
    groups = {"body": [], "head": []}
    hems = {}  # foot group -> height of the trouser / sock edge above it
    skipped = []  # group -> [(kind, args, blend)]; every group is meshed on its own grid
    bone_segs = []  # (bone index, a, b) for skin weights
    k_body = 0.012 * H

    class _P(list):
        def append(self, item, grp="body", k=None):
            groups.setdefault(grp, []).append((item[0], item[1], k_body if k is None else k))
    prims = _P()

    def seg(bname_a, bname_b, ra, rb):
        a, b = f.bp(bname_a), f.bp(bname_b)
        if a is None or b is None:
            return
        prims.append(("cone", (a, b, ra, rb)))
        bone_segs.append((f.h[bname_a], a, b))

    sides = ("left", "right")
    # ---- legs ---------------------------------------------------------------------------
    leg_r = {}
    for sd in sides:
        ul, ll, ft = f.bp(sd + "UpperLeg"), f.bp(sd + "LowerLeg"), f.bp(sd + "Foot")
        if ul is None or ll is None or ft is None:
            continue
        r_th = f.radius([f.h.get(sd + "UpperLeg")], ul, ll, 0.075 * H)
        r_kn = f.radius([f.h.get(sd + "LowerLeg")], ll, ll + (ft - ll) * 0.2, 0.045 * H)
        r_an = f.radius([f.h.get(sd + "LowerLeg")], ll + (ft - ll) * 0.8, ft, 0.03 * H)
        r_kn = min(r_kn, r_th * 0.75)
        r_an = min(r_an, r_kn * 0.75)
        leg_r[sd] = (r_th, r_kn, r_an)
        if "legs" in parts:
            seg(sd + "UpperLeg", sd + "LowerLeg", r_th, r_kn)
            seg(sd + "LowerLeg", sd + "Foot", r_kn * 1.02, r_an)
    # ---- feet with toes -----------------------------------------------------------------
    if "feet" in parts:
        for sd in sides:
            ft = f.bp(sd + "Foot")
            if ft is None:
                continue
            g = "foot_" + sd
            r_an = min(leg_r.get(sd, (0, 0, 0.02 * H))[2], 0.021 * H)  # wide trousers must not make an elephant ankle
            toe = f.bp(sd + "Toes")
            fwd = FWD.copy()
            if toe is not None:
                tv = toe - ft
                tv[1] = 0.0
                if np.linalg.norm(tv) > 1e-4 and tv @ FWD > 0:
                    fwd = tv / np.linalg.norm(tv)
            up = np.array([0.0, 1.0, 0.0])
            side = np.cross(up, fwd)
            side /= np.linalg.norm(side)
            inner = side if (side[0] * -ft[0]) > 0 else -side  # towards the body centre line
            # the model's own bare foot: leave it alone
            hz = f.P - ft
            near = (np.linalg.norm(hz[:, [0, 2]], axis=1) < 0.12 * H) & (hz[:, 1] < 0.01 * H) & (np.sign(f.P[:, 0]) == np.sign(ft[0]))
            own = near & f.skin_v
            side_ru = "левая" if sd == "left" else "правая"
            if own.sum() > 120 and (hz[own] @ fwd).max() > 0.06 * H:
                log("  ступня (%s) у модели своя — не достраивается" % side_ru)
                skipped.append("своя ступня")
                continue
            shoe = near & f.vis_v & ~f.skin_v
            if shoe.sum() > 120 and (hz[shoe] @ fwd).max() > 0.06 * H:
                log("  ступня (%s): обувь видна — не достраивается (скройте обувь, чтобы достроить)" % side_ru)
                skipped.append("обувь видна")
                continue
            # size of the foot from what was around it (the shoe, even hidden): inside of the shoe
            Lf, back, sole_min = 0.135 * H, None, None
            if near.sum() > 60:
                pr = hz[near] @ fwd
                Lf = float(np.clip((np.percentile(pr, 99) - np.percentile(pr, 1)) * 0.86, 0.10 * H, 0.16 * H))
                back = float(np.percentile(pr, 1))
                sole_min = float(f.P[near][:, 1].min())
            sole = ft[1] - 0.052 * H
            if sole_min is not None:
                sole = max(sole, sole_min + 0.006 * H)
            heel_c = np.array([ft[0], sole, ft[2]]) - fwd * 0.10 * Lf + up * 0.10 * Lf
            if back is not None:
                heel_c = heel_c + fwd * ((back + 0.045 * H) - (heel_c - ft) @ fwd - 0.07 * Lf)
            ball_c = heel_c + fwd * 0.66 * Lf + up * (-0.025 * Lf) + inner * 0.03 * Lf
            ll_ = f.bp(sd + "LowerLeg")
            if ll_ is not None:  # lower shin down to the ankle: boots often had no skin there
                r_kn_ = min(leg_r.get(sd, (0, r_an * 1.4, r_an))[1], 0.03 * H)
                prims.append(("cone", (ll_ + (ft - ll_) * 0.78, ft, min(r_kn_ * 0.8, r_an * 1.2), r_an * 0.95)), g, 0.04 * Lf)
            prims.append(("cone", (ft, heel_c, r_an * 0.92, 0.105 * Lf)), g, 0.04 * Lf)          # ankle -> heel
            prims.append(("cone", (heel_c, ball_c - inner * 0.10 * Lf, 0.10 * Lf, 0.075 * Lf)), g, 0.05 * Lf)  # outer sole
            prims.append(("cone", (heel_c + up * 0.04 * Lf, ball_c + inner * 0.08 * Lf + up * 0.03 * Lf,
                                   0.09 * Lf, 0.08 * Lf)), g, 0.05 * Lf)                          # arch / inner side
            prims.append(("cone", (ft + fwd * 0.05 * Lf, ball_c + up * 0.06 * Lf, r_an * 0.85, 0.07 * Lf)), g, 0.05 * Lf)  # instep
            prims.append(("cone", (ball_c - inner * 0.16 * Lf, ball_c + inner * 0.15 * Lf, 0.07 * Lf, 0.085 * Lf)), g, 0.04 * Lf)  # ball
            # toes: big toe on the inner side; lateral position, length, radius (x Lf); small gaps between
            spec = [(0.140, 0.225, 0.050), (0.052, 0.195, 0.036), (-0.025, 0.175, 0.033),
                    (-0.095, 0.150, 0.030), (-0.158, 0.120, 0.027)]
            for lat, ln, rr in spec:
                base = ball_c + inner * lat * Lf + fwd * 0.04 * Lf + up * 0.005 * Lf
                tip = base + fwd * ln * Lf - up * (0.03 * Lf)
                mid = base + (tip - base) * 0.55 + up * 0.012 * Lf
                prims.append(("cone", (base, mid, rr * Lf, rr * Lf * 0.95)), g, 0.012 * Lf)
                prims.append(("cone", (mid, tip, rr * Lf * 0.95, rr * Lf * 0.88)), g, 0.008 * Lf)
            hem = clothing_hem(f, ft, H)
            if hem is not None:
                hems[g] = hem
            tb = f.h.get(sd + "Toes", f.h.get(sd + "Foot"))
            bone_segs.append((f.h[sd + "Foot"], ft, heel_c))
            bone_segs.append((tb, ball_c - fwd * 0.05 * Lf, ball_c + fwd * 0.25 * Lf))
    # ---- torso --------------------------------------------------------------------------
    if "torso" in parts:
        chain = [(n, f.bp(n)) for n in TORSO if f.bp(n) is not None]
        if len(chain) >= 2:
            ul = [f.bp(s + "UpperLeg") for s in sides if f.bp(s + "UpperLeg") is not None]
            pelvis = np.mean(ul, axis=0) if ul else chain[0][1] - np.array([0, 0.05 * H, 0])
            pts = [("hips", pelvis)] + chain
            for (na, a), (nb, b) in zip(pts[:-1], pts[1:]):
                bi = f.h.get(na) if f.h.get(na) is not None else f.h.get("hips")
                rx, rz = torso_radii(f, [f.h.get(na), f.h.get(nb)], a, b, H)
                if nb == "neck":
                    rx, rz = rx * 0.9, rz * 0.95
                c = (a + b) / 2
                prims.append(("ell", (c, (rx, max(np.linalg.norm(b - a) * 0.75, rz * 0.8), rz))))
                bone_segs.append((bi, a, b))
            hb = f.bp("hips")
            if hb is not None:
                hx, hz = torso_radii(f, [f.h.get("hips"), f.h.get("leftUpperLeg"), f.h.get("rightUpperLeg")],
                                     pelvis - np.array([0, 0.03 * H, 0]), hb, H)
                prims.append(("ell", ((pelvis + hb) / 2, (hx, max(abs(hb[1] - pelvis[1]) * 0.9, 0.04 * H), hz))))
            # chest form from the breast bones (a smooth shape, no details)
            for b_i, bn in enumerate(f.m.bones):
                nm = bn.name or ""
                if ("胸" in nm or "breast" in nm.lower() or "bust" in nm.lower()) and "上半身" not in nm:
                    p = np.asarray(bn.pos, float)
                    sel = (f.main == b_i) & (f.mainw > 0.5)
                    if sel.sum() < 20:
                        continue
                    q = f.P[sel]
                    r = float(np.percentile(np.linalg.norm(q - q.mean(0), axis=1), 70)) * 0.92
                    prims.append(("ell", (q.mean(0) - FWD * 0.004 * H, (r, r * 0.95, r * 0.9))))
                    bone_segs.append((b_i, p, q.mean(0)))
    # ---- arms ---------------------------------------------------------------------------
    if "arms" in parts:
        for sd in sides:
            ua, la, hd = f.bp(sd + "UpperArm"), f.bp(sd + "LowerArm"), f.bp(sd + "Hand")
            if ua is None or la is None or hd is None:
                continue
            r_sh = f.radius([f.h.get(sd + "UpperArm")], ua, la, 0.04 * H)
            r_el = min(f.radius([f.h.get(sd + "LowerArm")], la, hd, 0.03 * H), r_sh * 0.85)
            seg(sd + "UpperArm", sd + "LowerArm", r_sh, r_el)
            seg(sd + "LowerArm", sd + "Hand", r_el, r_el * 0.75)
            sh = f.bp(sd + "Shoulder")
            if sh is not None:
                prims.append(("cone", (sh, ua, r_sh * 0.9, r_sh)))
    # ---- head ---------------------------------------------------------------------------
    if "head" in parts:
        hd = f.bp("head")
        if hd is not None:
            sel = (f.main == f.h["head"]) & f.skin_v
            q = f.P[sel] if sel.sum() > 40 else f.P[(f.main == f.h["head"])]
            if len(q) > 40:
                lo, hi = np.percentile(q, 3, axis=0), np.percentile(q, 97, axis=0)
                rx = (hi[0] - lo[0]) / 2 * 0.97
                eyes = [f.bp(e) for e in ("leftEye", "rightEye") if f.bp(e) is not None]
                cy = float(np.mean([e[1] for e in eyes])) if eyes else float((lo[1] + hi[1]) / 2 + 0.01 * H)
                chin = float(lo[1])
                ry = max(cy - chin, 0.05 * H) * 1.12
                rz = rx * 1.12
                # anime skull: about as wide as the face, deeper than wide, the crown well above the eyes
                rx *= 0.95
                ry = max(cy - chin, 0.05 * H) * 1.25
                rz = rx * 1.28
                zf = float(q[:, 2].min()) if FWD[2] < 0 else float(q[:, 2].max())  # the most forward face point
                c = np.array([float((lo[0] + hi[0]) / 2), cy + ry * 0.14, zf - FWD[2] * rz * 1.02])
                # the skull must stay behind the face (anime faces are flat): cut it by a plane just behind
                # the eyes — the face mesh covers the front, the fill shows on the back and top of the head
                ref = np.mean(eyes, axis=0) if eyes else c + FWD * rz * 0.5
                prims.append(("cut", (c, (rx, ry, rz), ref - FWD * 0.012 * H, FWD)), "head", 0.01 * H)
                bone_segs.append((f.h["head"], hd, c))
                nk_ = f.bp("neck")
                if nk_ is not None:
                    rn = f.radius([f.h.get("neck")], nk_, hd, 0.03 * H)
                    prims.append(("cone", (nk_, hd + (c - hd) * 0.3, rn, rn * 0.95)), "head", 0.01 * H)
                    bone_segs.append((f.h["neck"], nk_, hd))
    all_V, all_T, all_N = [], [], []
    nv = 0
    n_cut = [0]
    for gname, gp in groups.items():
        if not gp:
            continue
        fine = gname.startswith("foot")
        V, T, N = mesh_group(gp, H, 0.0016 * H if fine else 0.0036 * H, 2 if fine else 5)
        if V is None:
            continue
        # feet and head: only the real skin decides (a trouser hem or a hood must not cut the heel / skull);
        # the body: skin and the clothes that stay
        sel = f.skin_v if (fine or gname == "head") else f.vis_v
        n0 = len(T)
        T = cull_outside(V, T, f.P[sel], np.asarray(m.normals, float)[sel], H,
                         near_r=0.02 * H if (fine or gname == "head") else 0.035 * H, V_N=(V, N))
        if gname in hems:  # above the edge of a trouser leg / sock the body is hidden anyway: cut it off
            T = T[(V[T][:, :, 1] < hems[gname]).any(axis=1)]
        if fine or gname == "head":  # ...and not through the clothes that stay (trouser legs, a hood)
            cl = f.vis_v & ~f.skin_v
            if gname in hems:  # below the hem the skin is uncovered: only what reaches up into the hem
                top = (V[T][:, :, 1] > hems[gname] - 0.005 * H).all(axis=1)
                keep_low = T[~top]
                T = np.concatenate([keep_low, cull_outside(V, T[top], f.P[cl], np.asarray(m.normals, float)[cl], H,
                                                           near_r=0.012 * H)])
            else:
                T = cull_outside(V, T, f.P[cl], np.asarray(m.normals, float)[cl], H, near_r=0.012 * H)
        n_cut[0] += n0 - len(T)
        if len(T) == 0:
            continue
        used = np.unique(T)
        rm = -np.ones(len(V), np.int64)
        rm[used] = np.arange(len(used))
        V, N, T = V[used], N[used], rm[T]
        all_V.append(V)
        all_N.append(N)
        all_T.append(T + nv)
        nv += len(V)
    if not all_V:
        log("  ! достройка тела: нечего достраивать" + (" (" + ", ".join(sorted(set(skipped))) + ")" if skipped
                                                        else " — не хватает костей скелета или тело уже есть"))
        return None
    V = np.concatenate(all_V)
    N = np.concatenate(all_N)
    T = np.concatenate(all_T)
    if n_cut[0]:
        log("  достройка: убрано %d треугольников, выступавших из кожи или одежды" % n_cut[0])
    # ---- skin weights: the two nearest bone segments ------------------------------------
    bone_segs = [x for x in bone_segs if x[0] is not None]
    if not bone_segs:
        log("  ! достройка тела: не к чему привязать (нет костей)")
        return None
    D = np.stack([sd_round_cone(V, a, b, 0.0, 0.0) for _, a, b in bone_segs], 1)  # distances
    k = min(2, D.shape[1])
    order = np.argsort(D, axis=1)[:, :k]
    d2 = np.take_along_axis(D, order, 1)
    w = 1.0 / np.maximum(d2, 1e-4) ** 4
    w /= w.sum(1, keepdims=True)
    bid = np.array([b for b, _, _ in bone_segs])[order]
    J = np.full((len(V), 4), -1, np.int32)
    Wt = np.zeros((len(V), 4), np.float32)
    J[:, :k] = bid
    Wt[:, :k] = w
    log("  тело достроено (%s): %d вершин, %d треугольников" % (", ".join(sorted(parts)), len(V), len(T)))
    return V, N, T, J, Wt


def clothing_hem(f, ft, H):
    """height of the lowest edge of visible clothing that wraps the leg all around (trouser leg, sock top)
    just above the ankle; None when the leg is bare there"""
    cl = f.vis_v & ~f.skin_v
    q = f.P[cl] - ft
    sel = (np.linalg.norm(q[:, [0, 2]], axis=1) < 0.07 * H) & (q[:, 1] > -0.02 * H) & (q[:, 1] < 0.25 * H)
    q = q[sel]
    if len(q) < 30:
        return None
    ang = ((np.arctan2(q[:, 2], q[:, 0]) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    step = 0.006 * H
    yb = np.floor(q[:, 1] / step).astype(int)
    for b in range(int(yb.min()), int(yb.max()) + 1):
        m = (yb >= b) & (yb < b + 3)
        if len(set(ang[m].tolist())) >= 6:
            return float(ft[1] + (b + 1.5) * step)
    return None


def sd_prim(P, kind, args):
    if kind == "cone":
        return sd_round_cone(P, *args)
    if kind == "ell":
        return sd_ellipsoid(P, *args)
    if kind == "cut":  # ellipsoid minus the half-space in front of a plane
        c, rad, p0, n = args
        return np.maximum(sd_ellipsoid(P, c, rad), (P - p0) @ n)
    raise ValueError(kind)


def mesh_group(gp, H, step, smooth_it):
    lo, hi = [], []
    for kind, args, _k in gp:
        if kind == "cone":
            a, b, ra, rb = args
            r = max(ra, rb)
            lo.append(np.minimum(a, b) - r)
            hi.append(np.maximum(a, b) + r)
        else:
            c, rad = args[0], args[1]
            lo.append(np.asarray(c) - np.max(rad))
            hi.append(np.asarray(c) + np.max(rad))
    lo = np.min(lo, axis=0) - 3 * step
    hi = np.max(hi, axis=0) + 3 * step
    step = max(step, float(np.prod(hi - lo) / 3.0e6) ** (1 / 3))
    shape = np.ceil((hi - lo) / step).astype(int) + 1
    g = [lo[i] + np.arange(shape[i]) * step for i in range(3)]
    G = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
    F = None
    for kind, args, k in gp:
        d = sd_prim(G, kind, args)
        F = d if F is None else smin(F, d, k)
    V, T = surface_nets(F.reshape(shape), lo, step)
    if len(T) == 0:
        return None, None, None
    V = smooth_mesh(V, T, it=smooth_it, lam=0.5)
    N = vertex_normals(V, T)
    probe = V + N * step * 0.7
    dd = None
    for kind, args, k in gp:
        d = sd_prim(probe, kind, args)
        dd = d if dd is None else np.minimum(dd, d)
    if np.mean(dd > 0) < 0.5:
        T = T[:, ::-1]
        N = -N
    return V, T, N


def torso_radii(f, bones, a, b, H):
    """half width / half depth of the body between the heights of a and b, from the real surface:
    only points in that height band and near the centre line (no arms, no thighs); the 75th percentile
    of |x| on an ellipse is ~0.92 of its radius"""
    bones = [x for x in bones if x is not None]
    c = (a + b) / 2
    y0, y1 = min(a[1], b[1]), max(a[1], b[1])
    band = (f.P[:, 1] >= y0) & (f.P[:, 1] <= y1)
    near = (np.abs(f.P[:, 0] - c[0]) < 0.14 * H) & (np.abs(f.P[:, 2] - c[2]) < 0.12 * H)
    sel_b = np.isin(f.main, bones) & (f.mainw > 0.3) & band & near
    for src, fac in ((f.skin_v, 0.95), (f.other_v, 0.85)):
        sel = sel_b & src
        if sel.sum() >= 40:
            q = f.P[sel] - c
            rx = float(np.percentile(np.abs(q[:, 0]), 75)) / 0.92 * fac
            rz = float(np.percentile(np.abs(q[:, 2]), 75)) / 0.92 * fac
            return max(rx, 0.035 * H), max(rz, 0.03 * H)
    return 0.07 * H, 0.05 * H


def nearest(Q, P, cell):
    """index of (and distance to) the nearest point of P for every query in Q, looking only within the
    27 grid cells of size `cell` around it (farther: index -1, distance inf).  numpy only."""
    if len(P) == 0:
        return -np.ones(len(Q), np.int64), np.full(len(Q), np.inf)
    lo = np.minimum(P.min(0), Q.min(0)) - cell
    ip = np.floor((P - lo) / cell).astype(np.int64)
    iq = np.floor((Q - lo) / cell).astype(np.int64)
    # the grid must cover the queries too: a cell index past the end would wrap into another row of cells
    # and return a "neighbour" from the other side of the model
    dims = np.maximum(ip.max(0), iq.max(0)) + 3
    key = (ip[:, 0] * dims[1] + ip[:, 1]) * dims[2] + ip[:, 2]
    order = np.argsort(key)
    ks = key[order]
    best_d = np.full(len(Q), np.inf)
    best_i = -np.ones(len(Q), np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                c = iq + (dx, dy, dz)
                ok = np.all(c >= 0, axis=1) & np.all(c < dims, axis=1)
                kq = (c[:, 0] * dims[1] + c[:, 1]) * dims[2] + c[:, 2]
                st = np.searchsorted(ks, kq, "left")
                en = np.searchsorted(ks, kq, "right")
                n = np.where(ok, en - st, 0)
                for r in range(int(n.max()) if len(n) else 0):
                    a = n > r
                    if not a.any():
                        break
                    qi = np.nonzero(a)[0]
                    pi = order[st[qi] + r]
                    d = np.linalg.norm(Q[qi] - P[pi], axis=1)
                    b = d < best_d[qi]
                    best_d[qi[b]] = d[b]
                    best_i[qi[b]] = pi[b]
    return best_i, best_d


def cull_outside(V, T, P_vis, N_vis, H, max_pts=40000, near_r=None, V_N=None):
    """drop generated triangles that poke out of the real surface (skin or clothes that stay visible),
    and those lying just under it (they would flicker through): the body is only needed where there is
    nothing"""
    if len(P_vis) == 0 or len(T) == 0:
        return T
    if len(P_vis) > max_pts:
        sel = np.linspace(0, len(P_vis) - 1, max_pts).astype(int)
        P_vis, N_vis = P_vis[sel], N_vis[sel]
    near_r, margin = near_r or 0.035 * H, -0.0045 * H
    j, dm = nearest(V, P_vis, near_r)
    hit = j >= 0
    side = np.full(len(V), -1e9)
    side[hit] = np.einsum("ij,ij->i", V[hit] - P_vis[j[hit]], N_vis[j[hit]])
    # poking out of the surface (within near_r): removed
    out = hit & (dm < near_r) & (side > 0.0)
    # lying just under it: sunk a little deeper instead (no flicker through the skin, and no gap along the
    # open edge of a skin patch, e.g. where a removed boot began) — then tested again: sinking along the
    # generated normal can bring a vertex out on the other side of a thin limb
    if V_N is not None:
        under = hit & ~out & (dm < 0.01 * H) & (side > margin)
        if under.any():
            V_N[0][under] -= V_N[1][under] * (0.007 * H)
            ui = np.nonzero(under)[0]
            j2, d2 = nearest(V_N[0][ui], P_vis, near_r)
            ok2 = j2 >= 0
            s2 = np.full(len(ui), -1e9)
            s2[ok2] = np.einsum("ij,ij->i", V_N[0][ui][ok2] - P_vis[j2[ok2]], N_vis[j2[ok2]])
            out[ui[(d2 < near_r) & (s2 > 0.0)]] = True
    bad = out[T].any(axis=1)
    return T[~bad]


def transfer_uv(V, P, UV, skin_sel, H=20.0, plain_sel=None):
    """UV of a real skin vertex: right at the seams the nearest one (the colour continues exactly), farther
    away the nearest one of plain skin tone (an even surface, no copied blush / shadows / lines)"""
    S = np.nonzero(skin_sel)[0]
    out = np.zeros((len(V), 2), np.float32)
    if len(S) == 0:
        return out
    left = np.arange(len(V))
    j, d = nearest(V, P[S], 0.015 * H)
    ok = (j >= 0) & (d < 0.015 * H)
    out[ok] = UV[S[j[ok]]]
    left = left[~ok]
    if plain_sel is not None and plain_sel.any():
        S = np.nonzero(plain_sel)[0]
    if len(left) == 0:
        return out
    if len(S) > 5000:  # far from the seams any plain-skin point will do: a subsample, brute force
        S = S[np.linspace(0, len(S) - 1, 5000).astype(int)]
    SP = P[S]
    for i in range(0, len(left), 2048):
        q = left[i:i + 2048]
        d = ((V[q][:, None, :] - SP[None, :, :]) ** 2).sum(-1)
        out[q] = UV[S[np.argmin(d, axis=1)]]
    return out
