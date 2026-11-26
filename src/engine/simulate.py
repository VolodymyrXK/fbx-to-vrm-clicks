"""Physics auto-tuner: simulates every spring chain the way VRM spring bones move (UniVRM-style
verlet particles with stiffness / gravity / drag and sphere colliders) under a test motion,
for a grid of settings at once, and picks the settings that behave best for each group of chains:

  * no joint sinks into the body (colliders),
  * the tip follows the motion by a natural amount for its kind (hair swings, a tail sways with
    weight, a breast barely moves),
  * after the motion stops it settles instead of bouncing on like a loose spring,
  * it does not sag far from its modelled shape when standing still.

Not a neural network: an honest search on a simplified simulation of the same physics the
avatar program will run.
"""
import math

import numpy as np

DT = 1.0 / 60.0
# lively range of the tip deviation during motion, as a fraction of the chain length
TARGET = {"hair": (0.15, 0.45), "skirt": (0.10, 0.35), "tail": (0.18, 0.50), "breast": (0.03, 0.12),
          "ear": (0.04, 0.18), "other": (0.12, 0.45), "cloth": (0.08, 0.30), "ribbon": (0.15, 0.50),
          "accessory": (0.12, 0.45), "flesh": (0.0, 0.04)}
GRID_S = np.array([0.6, 1.0, 1.5, 2.2, 3.2])
GRID_G = np.array([0.05, 0.2, 0.4, 0.7])
GRID_D = np.array([0.3, 0.45, 0.6, 0.75])


def _motion(n_still=36, n_move=150, n_stop=72):
    """body transform per frame: (yaw radians, translation xyz). Stand, walk-sway + turn, stop."""
    frames = []
    for _ in range(n_still):
        frames.append((0.0, np.zeros(3)))
    for k in range(n_move):
        t = k * DT
        yaw = 0.25 * math.sin(2 * math.pi * 0.8 * t)
        if k > n_move - 30:
            yaw += (k - (n_move - 30)) / 30.0 * 0.8  # a turn at the end
        tr = np.array([0.04 * math.sin(2 * math.pi * 1.8 * t), 0.02 * abs(math.sin(2 * math.pi * 1.8 * t)),
                       0.4 * t])
        frames.append((yaw, tr))
    last = frames[-1]
    for _ in range(n_stop):
        frames.append(last)
    return frames, n_still, n_still + n_move


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def simulate_chain(sim, pivot, S, G, D):
    """sim: {"pts": [[x,y,z]...] rest joints (first = fixed root), "spheres": [[x,y,z,r]], "hit": r}
    S, G, D: arrays (C,) of settings. Returns metrics dict of arrays (C,)."""
    P0 = np.asarray(sim["pts"], float)
    if len(P0) < 2:
        return None
    sph = np.asarray(sim["spheres"], float).reshape(-1, 4)
    hit = float(sim.get("hit", 0.02))
    C = len(S)
    J = len(P0)
    seg = P0[1:] - P0[:-1]
    L = np.linalg.norm(seg, axis=1)
    L = np.maximum(L, 1e-4)
    axis = seg / L[:, None]
    total = float(L.sum())
    frames, t_move, t_stop = _motion()
    x = np.repeat(P0[None], C, 0)  # (C, J, 3)
    xp = x.copy()
    dev, pen, jit = [], np.zeros(C), np.zeros(C)
    prev_tip_v = np.zeros((C, 3))
    for fi, (yaw, tr) in enumerate(frames):
        R = _rot_y(yaw)
        root = (P0[0] - pivot) @ R.T + pivot + tr
        x[:, 0] = root
        cs = (sph[:, :3] - pivot) @ R.T + pivot + tr if len(sph) else sph[:, :3]
        for j in range(1, J):
            inert = (x[:, j] - xp[:, j]) * (1.0 - D)[:, None]
            stiff = (axis[j - 1] @ R.T)[None] * (S * DT)[:, None]
            grav = np.array([0, -1.0, 0])[None] * (G * DT)[:, None]
            nxt = x[:, j] + inert + stiff + grav
            d = nxt - x[:, j - 1]
            nxt = x[:, j - 1] + d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9) * L[j - 1]
            if len(sph):
                v = nxt[:, None, :] - cs[None]                       # (C, K, 3)
                dist = np.linalg.norm(v, axis=2)
                lim = sph[None, :, 3] + hit
                inside = dist < lim
                if inside.any():
                    # count how deep the free motion would have gone (before the push-out)
                    pen += ((lim - dist) * inside).sum(axis=1) / max(hit, 1e-3)
                    push = np.where(inside[..., None], v / np.maximum(dist, 1e-9)[..., None] * lim[..., None] - v, 0)
                    nxt = nxt + push.sum(axis=1)
                    d = nxt - x[:, j - 1]
                    nxt = x[:, j - 1] + d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9) * L[j - 1]
            xp[:, j] = x[:, j]
            x[:, j] = nxt
        target = (P0[-1] - pivot) @ R.T + pivot + tr
        dev.append(np.linalg.norm(x[:, -1] - target, axis=1) / total)
        tip_v = x[:, -1] - xp[:, -1]
        if fi >= t_move:
            jit += np.linalg.norm(tip_v - prev_tip_v, axis=1)
        prev_tip_v = tip_v
    dev = np.array(dev)  # (F, C)
    return {"sag": dev[t_move - 1], "peak": dev[t_move:t_stop].max(axis=0),
            "settle": dev[t_stop + 30:].std(axis=0) + np.abs(np.diff(dev[t_stop + 30:], axis=0)).mean(axis=0) * 10,
            "pen": pen / len(frames), "jitter": jit / max(1, len(frames) - t_move),
            "len": total}


def score(metrics, kind):
    lo, hi = TARGET.get(kind, TARGET["other"])
    m = metrics
    s = (np.maximum(0, lo - m["peak"]) * 4 + np.maximum(0, m["peak"] - hi) * 3   # liveliness in range
         + m["settle"] * 6                                                        # no endless bouncing
         + np.maximum(0, m["sag"] - 0.15) * 3                                     # keeps its shape
         + m["jitter"] / max(m["len"], 1e-3) * 2)                                 # smooth tip
    return s


def family(cid):
    """same grouping as the editor (chainFamily in web/app.js): hair_base_L_a_01 -> 'hair base'"""
    import re
    s_ = re.sub(r"[左右]", " ", cid)
    s_ = re.sub(r"([a-z])([A-Z])", r"\1 \2", s_)
    toks = [t for t in re.split(r"[_.\-\s]+|\d+", s_) if t.strip()]
    toks = [t for t in toks if not re.fullmatch(r"(?i)(l|r|left|right|jnt|joint|bone|bn|end|tip|[a-z])", t)]
    return " ".join(toks[:2]).lower() or cid


def autotune(chains, pivot, groups=None, log=print):
    """chains: chain_info list (with "_sim"); groups: {group key: [chain ids]} (defaults: per kind).
    Returns {chain id: {"stiffness","gravity","drag"}}"""
    SS, GG, DD = np.meshgrid(GRID_S, GRID_G, GRID_D, indexing="ij")
    S, G, D = SS.ravel(), GG.ravel(), DD.ravel()
    by_id = {c["id"]: c for c in chains if c.get("_sim") and c.get("enabled", True)}
    if groups is None:
        groups = {}
        for c in by_id.values():
            groups.setdefault(c["kind"] + "|" + family(c["id"]), []).append(c["id"])
    pivot = np.asarray(pivot, float)
    out = {}
    for key, ids in groups.items():
        ids = [i for i in ids if i in by_id]
        if not ids:
            continue
        total = np.zeros(len(S))
        used = 0
        for cid in ids[:12]:  # a dozen strands describe a hairstyle well enough
            c = by_id[cid]
            mt = simulate_chain(c["_sim"], pivot, S, G, D)
            if mt is None:
                continue
            total += score(mt, c["kind"]) + mt["pen"] * 0.5
            # stay near the current (style) values unless the simulation clearly prefers otherwise
            total += 0.04 * (np.abs(np.log(S / max(c.get("stiffness", 1.0), 0.05)))
                             + np.abs(D - c.get("drag", 0.45)) * 2 + np.abs(G - c.get("gravity", 0.25)))
            used += 1
        if not used:
            continue
        best = int(np.argmin(total))
        res = {"stiffness": float(S[best]), "gravity": float(G[best]), "drag": float(D[best])}
        for cid in ids:
            out[cid] = dict(res)
        log("  %s: жёсткость %.2f, гравитация %.2f, затухание %.2f (%d цепочек)"
            % (key, res["stiffness"], res["gravity"], res["drag"], len(ids)))
    return out
