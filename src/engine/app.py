"""MMD -> VRM editor: local HTTP server + web UI (three.js / three-vrm) in an app window."""
import io
import json
import os
import queue
import shutil
import subprocess
import tempfile
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import converter


def resource_dir():
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, "web")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


WEB = resource_dir()
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8", ".json": "application/json", ".png": "image/png",
        ".svg": "image/svg+xml", ".txt": "text/plain; charset=utf-8"}


class Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.src = None
        self.vrm = None
        self.info = None
        self.version = 0
        self.last_ping = time.time()
        self.log = []


S = Session()
EXTRACT_LOCK = threading.Lock()
DIALOGS = queue.Queue()  # (kind, kwargs, result_queue) executed on the main thread


def ask_dialog(kind, **kw):
    rq = queue.Queue()
    DIALOGS.put((kind, kw, rq))
    return rq.get()


def run_dialog(kind, kw):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if kind == "open":
            return filedialog.askopenfilename(parent=root, title="Открыть модель MMD",
                                              filetypes=[("Модели и архивы", "*.pmx *.pmd *.fbx *.vrm *.zip *.rar *.7z"), ("PMX", "*.pmx"), ("PMD", "*.pmd"), ("FBX", "*.fbx"), ("VRM", "*.vrm"), ("Архивы", "*.zip *.rar *.7z"), ("Все файлы", "*.*")],
                                              initialdir=kw.get("initialdir") or None) or None
        if kind == "save":
            return filedialog.asksaveasfilename(parent=root, title="Сохранить VRM", defaultextension=".vrm",
                                                filetypes=[("VRM", "*.vrm")],
                                                initialdir=kw.get("initialdir") or None,
                                                initialfile=kw.get("initialfile") or None) or None
        if kind == "folder":
            return filedialog.askdirectory(parent=root, title=kw.get("title") or "Папка для VRM") or None
        if kind == "exe":
            return filedialog.askopenfilename(parent=root, title="Программа для просмотра VRM (Astra, VSeeFace, VMagicMirror…)",
                                              filetypes=[("Программы", "*.exe"), ("Все файлы", "*.*")]) or None
        if kind == "template_save":
            return filedialog.asksaveasfilename(parent=root, title="Сохранить шаблон настроек", defaultextension=".json",
                                                filetypes=[("Шаблон mmd2vrm", "*.json")],
                                                initialfile=kw.get("initialfile") or "шаблон.json") or None
        if kind == "template_open":
            return filedialog.askopenfilename(parent=root, title="Применить шаблон настроек",
                                              filetypes=[("Шаблон mmd2vrm", "*.json"), ("Все файлы", "*.*")]) or None
    finally:
        root.destroy()


def _num(v, default, cast=float):
    try:
        x = cast(v)
    except (TypeError, ValueError):
        return default
    return x if x == x and abs(x) != float("inf") else default


def settings_kwargs(st):
    st = st if isinstance(st, dict) else {}
    meta = st.get("meta") if isinstance(st.get("meta"), dict) else {}
    sc = _num(st.get("scale", 0.08), 0.08)
    return dict(
        scale=sc if sc > 0 else 0.08,
        tpose=bool(st.get("tpose", True)),
        spring=bool(st.get("spring", True)),
        outline=bool(st.get("outline", False)),
        custom_blendshapes=bool(st.get("custom_blendshapes", True)),
        max_texture=max(0, _num(st.get("max_texture", 2048), 2048, int)),
        physics=st.get("physics") if isinstance(st.get("physics"), str) else "soft",
        title=str(meta.get("title") or "") or None,
        author=str(meta.get("author") or "") or None,
        ru_names=bool(st.get("ru_names", False)),
        repair=bool(st.get("repair", False)),
        face_from_bones=st.get("face_from_bones", True) is not False,
        anime_face=bool(st.get("anime_face", True)),
        vrm_version=1 if str(st.get("vrm_version", "0")).startswith("1") else 0,
        target_mb=max(0.0, _num(st.get("target_mb", 0), 0.0)),
        strip_hidden=bool(st.get("strip_hidden", False)),
        tex_quality="best" if st.get("tex_quality") == "best" else "normal",
        bone_lang=st.get("bone_lang") if st.get("bone_lang") in ("orig", "ru", "en") else None,
        upscale_dir=upscale_dir(upscale_model(st.get("upscale"))) if st.get("upscale") else None,
        tex_sharpen=min(1.0, max(0.0, _num(st.get("tex_sharpen", 0), 0.0))),
        overrides={k: st.get(k) for k in ("humanoid", "materials", "chains", "presets", "renames", "splits", "body_fill") if st.get(k)},
    )


def public_info(info):
    d = {k: v for k, v in info.items() if k != "vrm" and not k.startswith("_")}
    if d.get("chains"):
        d["chains"] = [{k: v for k, v in c.items() if not k.startswith("_")} for c in d["chains"]]
    d["has_vrm"] = info.get("vrm") is not None
    d["version"] = S.version
    d["src"] = S.src
    d["log"] = S.log[-200:]
    d["categories"] = converter.CATEGORY_NAMES
    d["kinds"] = converter.KIND_NAMES
    d["ru_labels"] = converter.vocab.RU_LABELS
    d["physics_presets"] = converter.PHYSICS_NAMES
    d["part_presets"] = {k: [{"id": a, "name": b, "stiffness": c, "gravity": g, "drag": r} for a, b, c, g, r in v]
                         for k, v in converter.PART_PRESETS.items()}
    d["preset_names"] = [p[0] for p in converter.PRESETS if p[0] != "neutral"]
    d["all_humanoid"] = ALL_HUMAN_BONES
    return d


ALL_HUMAN_BONES = (["hips", "spine", "chest", "upperChest", "neck", "head", "leftEye", "rightEye", "jaw"] +
                   [s + b for s in ("left", "right") for b in
                    ("Shoulder", "UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot", "Toes")] +
                   [s + f + p for s in ("left", "right")
                    for f in ("Thumb", "Index", "Middle", "Ring", "Little")
                    for p in ("Proximal", "Intermediate", "Distal")])


# ---------------------------------------------------------------- folders & archives
MODEL_EXT = (".pmx", ".pmd", ".fbx", ".vrm")
ARCHIVE_EXT = (".rar", ".7z")


def _is_kana(ch):
    return "\u3040" <= ch <= "\u30ff"


def zip_encoding(z):
    """Pick one encoding for all non-UTF-8 names of an archive (Shift-JIS vs GBK vs UTF-8)."""
    raws = []
    for info in z.infolist():
        if info.flag_bits & 0x800:
            continue
        try:
            raws.append(info.filename.encode("cp437"))
        except UnicodeEncodeError:
            pass  # already decoded from the Unicode path extra field
    if not raws or all(r.isascii() for r in raws):
        return None
    ok = {}
    for enc in ("utf-8", "shift_jis", "gbk"):
        try:
            ok[enc] = [r.decode(enc) for r in raws]
        except UnicodeDecodeError:
            continue
    if "utf-8" in ok:
        return "utf-8"
    if "shift_jis" in ok and "gbk" in ok:
        sj = "".join(ok["shift_jis"])
        kana = sum(1 for c in sj if _is_kana(c))
        halfwidth = sum(1 for c in sj if "\uff61" <= c <= "\uff9f")
        return "shift_jis" if kana > halfwidth else "gbk"
    return next(iter(ok), None)


def _zip_name(info, enc=None):
    n = info.filename
    if enc and not (info.flag_bits & 0x800):
        try:
            n = n.encode("cp437").decode(enc)
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return n.replace("\\", "/")


def scan_archive(full, rel=None):
    """models inside one .zip / .rar / .7z"""
    import zipfile
    f = os.path.basename(full)
    rel = rel or f
    out = []
    if full.lower().endswith(ARCHIVE_EXT):
        names = list_archive(full)
        if names is None:
            return [{"name": f, "path": full, "rel": rel, "size": 0,
                     "error": "нужно распаковать вручную (нет инструмента для %s)" % os.path.splitext(f)[1]}]
        for nm in names:
            nm = nm.replace("\\", "/")
            if nm.lower().endswith(MODEL_EXT) and "__macosx" not in nm.lower():
                out.append({"name": os.path.basename(nm), "path": full, "member": nm,
                            "rel": rel + " → " + nm, "size": 0})
        return out
    try:
        with zipfile.ZipFile(full) as z:
            enc = zip_encoding(z)
            for info in z.infolist():
                nm = _zip_name(info, enc)
                if nm.lower().endswith(MODEL_EXT) and "__macosx" not in nm.lower():
                    out.append({"name": os.path.basename(nm), "path": full, "member": nm,
                                "rel": rel + " → " + nm, "size": info.file_size})
    except (zipfile.BadZipFile, OSError):
        pass
    return out


def scan_folder(root, limit=400):
    import zipfile
    found = []
    root_depth = root.rstrip(os.sep).count(os.sep)
    for dp, dn, fn in os.walk(root):
        if dp.count(os.sep) - root_depth >= 8:
            dn[:] = []
        dn[:] = [d for d in dn if not d.startswith(".") and d.lower() not in ("node_modules", "__pycache__")]
        for f in fn:
            full = os.path.join(dp, f)
            low = f.lower()
            if low.endswith(MODEL_EXT):
                try:
                    size = os.path.getsize(full)
                except OSError:
                    size = 0
                found.append({"name": f, "path": full, "rel": os.path.relpath(full, root), "size": size})
            elif low.endswith(ARCHIVE_EXT + (".zip",)):
                found.extend(scan_archive(full, os.path.relpath(full, root)))
            if len(found) >= limit:
                return found
    return found


def _system_tar():
    """Windows 10/11 ships bsdtar (libarchive) as tar.exe: it reads .rar and .7z."""
    if os.name == "nt":
        p = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "tar.exe")
        if os.path.isfile(p):
            return p
    for n in ("bsdtar",):
        p = shutil.which(n)
        if p:
            return p
    return None


def list_archive(path):
    """Names inside a .rar/.7z, or None if no tool can read it."""
    if path.lower().endswith(".7z"):
        try:
            import py7zr
            with py7zr.SevenZipFile(path, "r") as z:
                return z.getnames()
        except ImportError:
            pass
        except Exception:
            return None
    tar = _system_tar()
    if not tar:
        return None
    try:
        r = subprocess.run([tar, "-tf", path], capture_output=True, timeout=120,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    out = r.stdout
    encs = ("utf-8", "mbcs", "cp932", "gbk") if os.name == "nt" else ("utf-8", "cp932", "gbk")
    for enc in encs:
        try:
            return [l for l in out.decode(enc).splitlines() if l.strip()]
        except (UnicodeDecodeError, LookupError):
            continue
    return [l for l in out.decode("utf-8", "replace").splitlines() if l.strip()]


def cache_dir():
    d = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "mmd2vrm", "cache")
    os.makedirs(d, exist_ok=True)
    return d


def config_path():
    return os.path.join(os.path.dirname(cache_dir()), "config.json")


def load_config():
    try:
        with open(config_path(), encoding="utf-8") as f:
            c = json.load(f)
        return c if isinstance(c, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config(c):
    try:
        with open(config_path(), "w", encoding="utf-8") as f:
            json.dump(c, f, ensure_ascii=False, indent=1)
    except OSError:
        pass


# AI models of Real-ESRGAN (all ship in the ncnn-vulkan zip): fast one for anime video frames,
# slower x4 one trained on anime illustrations (cleaner line art, better for game / MMD textures)
ESRGAN_MODELS = {"fast": ("realesr-animevideov3", None), "best": ("realesrgan-x4plus-anime", 4)}


def upscale_model(v):
    return "best" if v == "best" else "fast"


def upscale_dir(model="fast"):
    d = os.path.join(os.path.dirname(cache_dir()), "upscaled" if model == "fast" else "upscaled_" + model)
    os.makedirs(d, exist_ok=True)
    return d


ESRGAN_URL = {
    "nt": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip",
    "posix": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip",
}


def esrgan_exe(download=True, log=print):
    """Real-ESRGAN (ncnn-vulkan build, BSD-3): downloaded once into %LOCALAPPDATA%/mmd2vrm/tools"""
    import zipfile
    tdir = os.path.join(os.path.dirname(cache_dir()), "tools", "realesrgan")
    exe_name = "realesrgan-ncnn-vulkan.exe" if os.name == "nt" else "realesrgan-ncnn-vulkan"
    for dp, dn, fn in os.walk(tdir):
        if exe_name in fn:
            return os.path.join(dp, exe_name)
    if not download:
        return None
    import urllib.request
    os.makedirs(tdir, exist_ok=True)
    url = ESRGAN_URL["nt" if os.name == "nt" else "posix"]
    log("Скачивание Real-ESRGAN (~45 МБ)…")
    tmp = os.path.join(tdir, "download.zip")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    with zipfile.ZipFile(tmp) as z:
        for info in z.infolist():
            parts = _safe_parts(info.filename.replace("\\", "/"))
            if not parts:
                continue
            dst = os.path.join(tdir, *parts)
            if info.is_dir():
                os.makedirs(dst, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(info) as src_, open(dst, "wb") as out:
                shutil.copyfileobj(src_, out)
    os.remove(tmp)
    for dp, dn, fn in os.walk(tdir):
        if exe_name in fn:
            p = os.path.join(dp, exe_name)
            if os.name != "nt":
                os.chmod(p, 0o755)
            return p
    raise RuntimeError("В архиве Real-ESRGAN нет %s" % exe_name)


def upscale_textures(src, max_side=4096, log=print, model="fast"):
    """AI upscale (x2 / x4) of every colour texture of the model smaller than 2048 px.
    Alpha is enlarged by the network too (as a grey picture) and put back: cut-outs (hair tips, lace,
    lashes) get clean, sharp edges instead of a blurry halo."""
    from PIL import Image
    exe = esrgan_exe(log=log)
    model = upscale_model(model)
    net, fixed_sc = ESRGAN_MODELS[model]
    out_dir = upscale_dir(model)
    work = tempfile.mkdtemp(prefix="esr_", dir=cache_dir())
    done, skipped = 0, 0
    try:
        for hsh, raw, name in converter.model_textures(src):
            dst = os.path.join(out_dir, hsh + ".png")
            if os.path.isfile(dst):
                done += 1
                continue
            img = Image.open(io.BytesIO(raw))
            img.load()
            w, h = img.size
            if max(w, h) >= 2048:
                skipped += 1
                continue
            sc = 4 if max(w, h) <= 512 else 2
            if max(w, h) * sc > max_side:
                sc = 2
            has_a = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
            rgba = img.convert("RGBA")
            if has_a and rgba.getchannel("A").getextrema()[0] >= 250:
                has_a = False  # an alpha channel that is fully opaque

            def run_net(pic, tag):
                inp = os.path.join(work, hsh + tag + "_in.png")
                outp = os.path.join(work, hsh + tag + "_out.png")
                pic.save(inp)
                r = subprocess.run([exe, "-i", inp, "-o", outp, "-n", net, "-s", str(fixed_sc or sc)],
                                   cwd=os.path.dirname(exe), capture_output=True, timeout=900,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if r.returncode != 0 or not os.path.isfile(outp):
                    raise RuntimeError("Real-ESRGAN: %s" % (r.stderr.decode("utf-8", "replace")[-300:] or "ошибка"))
                res = Image.open(outp)
                res.load()
                if res.size != (w * sc, h * sc):  # the x4-only network: bring it to the chosen size
                    res = res.resize((w * sc, h * sc), Image.LANCZOS)
                return res

            up = run_net(rgba.convert("RGB"), "").convert("RGB")
            if has_a:
                try:
                    a = run_net(rgba.getchannel("A").convert("RGB"), "_a").convert("L")
                except RuntimeError:
                    a = rgba.getchannel("A").resize(up.size, Image.LANCZOS)
                up = up.convert("RGBA")
                up.putalpha(a)
            up.save(dst + ".tmp", "PNG")
            os.replace(dst + ".tmp", dst)
            done += 1
            log("  %s: %dx%d → %dx%d" % (name, w, h, up.size[0], up.size[1]))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    converter._TEX_CACHE.clear()  # textures loaded before the upscale must not hide the new files
    return done, skipped


def reveal(path):
    """show the file in Explorer (Windows) / the file manager"""
    try:
        if os.name == "nt":
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(path)])
    except OSError:
        pass


# what a settings template may carry over to another model (no model-specific meta / humanoid)
TEMPLATE_KEYS = ("splits", "body_fill", "bone_lang", "upscale", "tex_sharpen", "physics", "materials", "chains", "presets", "renames", "outline", "tpose", "spring", "tex_quality",
                 "custom_blendshapes", "ru_names", "repair", "anime_face", "vrm_version", "max_texture",
                 "scale", "target_mb", "strip_hidden")

MAX_UNZIP = 4 * 1024 ** 3     # 4 GB total
MAX_FILES = 20000


def _safe_parts(name):
    return [p for p in name.split("/") if p not in ("", ".", "..") and ":" not in p and not p.startswith("\\")]


def extract_zip(zpath):
    """Extract a model archive once into the cache (names decoded per archive) and return the folder."""
    import hashlib
    import tempfile
    import zipfile
    st = os.stat(zpath)
    key = hashlib.sha1(("%s|%d|%d" % (os.path.abspath(zpath), st.st_size, st.st_mtime_ns)).encode()).hexdigest()[:20]
    out = os.path.join(cache_dir(), key)
    if os.path.isfile(os.path.join(out, ".done")):
        return out
    with EXTRACT_LOCK:
        if os.path.isfile(os.path.join(out, ".done")):
            return out
        if zpath.lower().endswith(ARCHIVE_EXT):
            tmp = tempfile.mkdtemp(prefix=key + "_", dir=cache_dir())
            try:
                _extract_other(zpath, tmp)
            except BaseException:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
            open(os.path.join(tmp, ".done"), "w").close()
            if os.path.isdir(out):
                shutil.rmtree(out, ignore_errors=True)
            os.replace(tmp, out)
            return out
        with zipfile.ZipFile(zpath) as z:
            infos = [i for i in z.infolist() if not i.is_dir()]
            total = sum(i.file_size for i in infos)
            if len(infos) > MAX_FILES or total > MAX_UNZIP:
                raise ValueError("Архив слишком большой для распаковки (%d файлов, %.1f ГБ)" % (len(infos), total / 1024 ** 3))
            for i in infos:
                if i.compress_size and i.file_size / max(1, i.compress_size) > 200 and i.file_size > 50 * 1024 ** 2:
                    raise ValueError("Подозрительный архив (zip-бомба?): " + i.filename)
            enc = zip_encoding(z)
            tmp = tempfile.mkdtemp(prefix=key + "_", dir=cache_dir())
            real_tmp = os.path.realpath(tmp)
            for info in infos:
                parts = _safe_parts(_zip_name(info, enc))
                if not parts:
                    continue
                dst = os.path.join(tmp, *parts)
                if not os.path.realpath(dst).startswith(real_tmp + os.sep):
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with z.open(info) as f, open(dst, "wb") as g:
                    shutil.copyfileobj(f, g)
        open(os.path.join(tmp, ".done"), "w").close()
        if os.path.isdir(out):
            shutil.rmtree(out, ignore_errors=True)
        os.replace(tmp, out)
    return out


def _extract_other(path, dst):
    """.7z via py7zr if available, otherwise (and for .rar) via the system bsdtar."""
    real = os.path.realpath(dst)
    if path.lower().endswith(".7z"):
        try:
            import py7zr
            with py7zr.SevenZipFile(path, "r") as z:
                infos = z.list()
                total = sum(getattr(i, "uncompressed", 0) or 0 for i in infos)
                if len(infos) > MAX_FILES or total > MAX_UNZIP:
                    raise ValueError("Архив слишком большой для распаковки")
                names = [i.filename for i in infos
                         if not getattr(i, "is_symlink", False) and _strict_name(i.filename)]
                z.extract(path=dst, targets=names)
            _check_inside(real)
            return
        except ImportError:
            pass
    tar = _system_tar()
    if not tar:
        raise ValueError("Не могу распаковать %s: нужен Windows 10/11 (встроенный tar) — или распакуйте архив сами"
                         % os.path.basename(path))
    _tar_size_check(tar, path)
    r = subprocess.run([tar, "-xf", path, "-C", dst], capture_output=True, timeout=1800,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise ValueError("Архив не распаковался: " + r.stderr.decode("utf-8", "replace")[-300:])
    _check_inside(real)


def _strict_name(n):
    n = n.replace("\\", "/")
    if not n or n.startswith("/") or ":" in n:
        return False
    return ".." not in n.split("/")


def _tar_size_check(tar, path):
    """bsdtar -tv lists 'perm links owner group SIZE date... name': sum sizes before extracting."""
    try:
        r = subprocess.run([tar, "-tvf", path], capture_output=True, timeout=300,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        return
    if r.returncode != 0:
        return
    total = files = 0
    for line in r.stdout.decode("utf-8", "replace").splitlines():
        f = line.split()
        if len(f) > 5 and f[4].isdigit():
            total += int(f[4])
            files += 1
    if files > MAX_FILES or total > MAX_UNZIP:
        raise ValueError("Архив слишком большой для распаковки (%d файлов, %.1f ГБ)" % (files, total / 1024 ** 3))


def _check_inside(real):
    """bsdtar refuses '..' and absolute paths by default; double-check nothing escaped via links."""
    for dp, dn, fn in os.walk(real):
        for x in dn + fn:
            full = os.path.join(dp, x)
            if os.path.islink(full) and not os.path.realpath(full).startswith(real + os.sep):
                os.unlink(full)


def resolve_item(item):
    """item: {path, member?} -> real model file path on disk."""
    if item.get("member"):
        base = extract_zip(item["path"])
        p = os.path.join(base, *_safe_parts(item["member"]))
        if not os.path.exists(p):  # names listed in another code page ('?' for unknown chars)
            import fnmatch
            want = os.path.basename(item["member"].replace("\\", "/")).lower()
            cands = []
            for dp, dn, fn in os.walk(base):
                for x in fn:
                    if x.lower().endswith(MODEL_EXT):
                        if x.lower() == want:
                            return os.path.join(dp, x)
                        if fnmatch.fnmatchcase(x.lower(), want.replace("[", "[[]")):
                            cands.append(os.path.join(dp, x))
            if len(cands) == 1:
                return cands[0]
        return p
    return item["path"]


def sidecar(src):
    """edits are kept next to the model; for files dropped into the window (a new temporary copy
    each time) they are kept in the program's folder under the file's name + size"""
    try:
        drop = os.path.join(cache_dir(), "dropped")
        if os.path.normcase(os.path.realpath(src)).startswith(os.path.normcase(os.path.realpath(drop)) + os.sep):
            d = os.path.join(os.path.dirname(cache_dir()), "edits")
            os.makedirs(d, exist_ok=True)
            key = "%s_%d" % (os.path.basename(src), os.path.getsize(src))
            return os.path.join(d, "".join(c for c in key if c not in '<>:"/\\|?*') + ".mmd2vrm.json")
    except OSError:
        pass
    return src + ".mmd2vrm.json"


def load_sidecar(src):
    try:
        with open(sidecar(src), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save_sidecar(src, settings):
    try:
        with open(sidecar(src), "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=1)
    except OSError:
        pass


def do_convert(src, settings):
    logs = []
    info = converter.convert(src, None, editor=True, log=logs.append, **settings_kwargs(settings))
    with S.lock:
        S.settings = settings if isinstance(settings, dict) else None
        S.src = src
        S.info = info
        S.vrm = info.get("vrm")
        S.version += 1
        S.log = logs
    return info


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        try:
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # the browser went away (closed tab / cancelled download)

    def _trusted(self):
        """only the editor page itself may talk to the API (blocks other sites / DNS rebinding)"""
        port = self.server.server_address[1]
        okhost = {"127.0.0.1:%d" % port, "localhost:%d" % port}
        if (self.headers.get("Host") or "") not in okhost:
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {"http://" + h for h in okhost}:
            return False
        return True

    def _upload(self, n):
        """files dropped into the window arrive one by one (browsers do not reveal local paths)"""
        from urllib.parse import urlparse, parse_qs
        if not self._trusted() or (self.headers.get("Content-Type") or "") != "application/octet-stream":
            self.rfile.read(min(n, 1 << 20)) if n > 0 else None
            return self.send(403, {"error": "forbidden"})
        q = parse_qs(urlparse(self.path).query)
        sess = "".join(ch for ch in (q.get("session") or [""])[0] if ch.isalnum())[:32]
        parts = _safe_parts((q.get("rel") or [""])[0].replace("\\", "/"))
        if not sess or not parts or ".." in parts or n < 0 or n > 2 * 1024 ** 3:
            return self.send(400, {"error": "bad upload"})
        base = os.path.realpath(os.path.join(cache_dir(), "dropped", sess))
        dst = os.path.realpath(os.path.join(base, *parts))
        if not dst.startswith(base + os.sep):
            return self.send(400, {"error": "bad path"})
        left = n
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                while left > 0:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
        except OSError as e:
            while left > 0:  # drain the body so the connection stays usable
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                left -= len(chunk)
            return self.send(400, {"error": "Не удалось сохранить файл: %s" % e})
        return self.send(200, {"ok": True, "bytes": n - left})

    def do_GET(self):
        path = self.path.split("?")[0]
        if not self._trusted():
            return self.send(403, {"error": "forbidden"})
        if path == "/":
            path = "/index.html"
        if path == "/api/vrm":
            with S.lock:
                data = S.vrm
            if data is None:
                return self.send(404, {"error": "no model"})
            return self.send(200, data, "application/octet-stream")
        if path == "/api/original":
            # the model "as in MMD": original pose, normals and materials, no fixes (for before/after view)
            if not S.src:
                return self.send(404, {"error": "no model"})
            # the model before the user's edits, with the physics of the source file (MMD rigid bodies /
            # VRM springs), in T-pose so the same test motion plays on both
            try:
                src_m = converter.load_pmx_cached(S.src)
                has_phys = any(rb.mode in (1, 2) for rb in getattr(src_m, "rigid_bodies", []) or [])
                kw = settings_kwargs(None)
                kw.update(tpose=True, anime_face=False, repair=False, spring=has_phys, face_from_bones=False,
                          max_texture=1024, vrm_version=0, custom_blendshapes=False, physics="mmd")
                info = converter.convert(S.src, None, log=lambda s_: None, **kw)
            except Exception as e:
                return self.send(500, {"error": str(e)})
            if info.get("vrm") is None:
                return self.send(404, {"error": "no model"})
            return self.send(200, info["vrm"], "application/octet-stream",
                             extra={"X-Has-Physics": "1" if has_phys else "0"})
        if path in ("/api/wardrobe/thumb", "/api/wardrobe/swatch"):
            from urllib.parse import parse_qs, urlparse
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            try:
                import wardrobe, wardrobe_transfer
                if path.endswith("thumb"):
                    fp_ = os.path.join(wardrobe_transfer.item_dir(q.get("uid", "")), "thumb.png")
                else:
                    g = wardrobe.BUILTIN_BY_ID.get(q.get("id", ""), {})
                    fp_ = wardrobe.fabric_texture(q.get("pattern") or g.get("pattern", "solid"),
                                                  q.get("color") or g.get("color", "#dddddd"), q.get("color2") or g.get("color2"))
                with open(fp_, "rb") as f_:
                    return self.send(200, f_.read(), "image/png")
            except Exception:
                return self.send(404, {"error": "not found"})
        if path == "/api/state":
            with S.lock:
                info = S.info
            if not info:
                return self.send(200, {"empty": True, "startup": STARTUP})
            d = public_info(info)
            d["settings"] = getattr(S, "settings", None)
            return self.send(200, d)
        fp = os.path.normpath(os.path.join(WEB, path.lstrip("/")))
        if not fp.startswith(os.path.normpath(WEB) + os.sep) or not os.path.isfile(fp):
            return self.send(404, "not found", "text/plain")
        with open(fp, "rb") as f:
            data = f.read()
        self.send(200, data, MIME.get(os.path.splitext(fp)[1].lower(), "application/octet-stream"))

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if self.path.startswith("/api/upload?"):
            return self._upload(n)
        body = self.rfile.read(n) if n > 0 else b""
        if not self._trusted() or not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self.send(403, {"error": "forbidden"})
        try:
            req = json.loads(body or b"{}")
        except ValueError:
            req = {}
        if not isinstance(req, dict):
            req = {}
        path = self.path.split("?")[0]
        try:
            if path == "/api/ping":
                S.last_ping = time.time()
                return self.send(200, {"ok": True})
            if path == "/api/browse":
                p = ask_dialog("open", initialdir=os.path.dirname(S.src) if S.src else None)
                if p and p.lower().endswith(ARCHIVE_EXT + (".zip",)):
                    items = scan_archive(p)
                    if not items:
                        return self.send(400, {"error": "В архиве нет моделей .pmx / .pmd / .fbx / .vrm"})
                    S.root = os.path.dirname(p)
                    converter.EXTRA_TEX_ROOTS[:] = [S.root]
                    return self.send(200, {"root": p, "items": items})
                return self.send(200, {"path": p})
            if path == "/api/scan":
                root = req.get("path") or ask_dialog("folder")
                if not root:
                    return self.send(200, {"cancelled": True})
                if not os.path.isdir(root):
                    return self.send(400, {"error": "Папка не найдена: %s" % root})
                S.root = root
                converter.EXTRA_TEX_ROOTS[:] = [root]
                return self.send(200, {"root": root, "items": scan_folder(root)})
            if path == "/api/open":
                src = req.get("path")
                if req.get("member"):
                    src = resolve_item(req)
                if not src or not os.path.isfile(src):
                    return self.send(400, {"error": "Файл не найден: %s" % src})
                settings = req.get("settings")
                saved = load_sidecar(src) if req.get("use_saved") else None
                if saved:
                    settings = saved
                info = do_convert(src, settings)
                out = public_info(info)
                if saved:
                    out["saved_settings"] = saved
                return self.send(200, out)
            if path == "/api/rebuild":
                if not S.src:
                    return self.send(400, {"error": "Модель не открыта"})
                info = do_convert(S.src, req.get("settings"))
                if isinstance(req.get("settings"), dict):
                    save_sidecar(S.src, req["settings"])  # edits survive a crash / closed window
                return self.send(200, public_info(info))
            if path == "/api/wardrobe/list":
                import wardrobe, wardrobe_transfer
                return self.send(200, {"builtin": [{k: g.get(k) for k in ("id", "name", "kind", "color", "color2", "pattern")}
                                                   for g in wardrobe.BUILTIN],
                                       "patterns": wardrobe.PATTERNS, "catalog": wardrobe_transfer.list_items(),
                                       "folder": wardrobe.catalog_dir()})
            if path == "/api/wardrobe/save":
                if not S.src:
                    return self.send(400, {"error": "Модель не открыта"})
                import wardrobe_transfer
                st = req.get("settings") or getattr(S, "settings", None) or {}
                logs = []
                try:
                    m_, h_, ov_ = converter.prepare_source(S.src, st, log=logs.append)
                    uid = wardrobe_transfer.save_item(m_, h_, ov_, [str(x) for x in (req.get("materials") or [])],
                                                      str(req.get("name") or "")[:80], os.path.dirname(os.path.abspath(S.src)),
                                                      thumb_dataurl=req.get("thumb"), src_name=os.path.basename(S.src),
                                                      log=logs.append)
                except Exception as e:
                    return self.send(400, {"error": "Не удалось сохранить одежду: %s" % e})
                return self.send(200, {"uid": uid, "log": logs})
            if path in ("/api/wardrobe/delete", "/api/wardrobe/rename"):
                import wardrobe_transfer
                try:
                    if path.endswith("delete"):
                        wardrobe_transfer.delete_item(str(req.get("uid") or ""))
                    else:
                        wardrobe_transfer.rename_item(str(req.get("uid") or ""), str(req.get("name") or ""))
                except Exception as e:
                    return self.send(400, {"error": str(e)})
                return self.send(200, {"ok": True})
            if path == "/api/settings":  # light autosave (physics sliders): no reconversion
                if not S.src:
                    return self.send(400, {"error": "Модель не открыта"})
                st = req.get("settings")
                if isinstance(st, dict):
                    with S.lock:
                        S.settings = st
                    save_sidecar(S.src, st)
                return self.send(200, {"ok": True})
            if path == "/api/save":
                if not S.src:
                    return self.send(400, {"error": "Модель не открыта"})
                info = do_convert(S.src, req.get("settings"))
                if info.get("vrm") is None:
                    return self.send(400, {"error": "Не хватает обязательных костей: " + ", ".join(info["missing"])})
                dst = req.get("path")
                if not dst:
                    dst = ask_dialog("save", initialdir=os.path.dirname(S.src),
                                     initialfile=os.path.splitext(os.path.basename(S.src))[0] +
                                     ("_mmd2vrm.vrm" if S.src.lower().endswith(".vrm") else ".vrm"))
                if not dst:
                    return self.send(200, {"cancelled": True})
                dst = str(dst)
                if not dst.lower().endswith(".vrm"):
                    dst += ".vrm"
                if os.path.exists(dst) and os.path.exists(S.src) and os.path.samefile(dst, S.src):
                    return self.send(400, {"error": "Нельзя сохранить поверх исходной модели"})
                with open(dst, "wb") as f:
                    f.write(info["vrm"])
                save_sidecar(S.src, req.get("settings") or {})
                extra = []
                ea = (req.get("settings") or {}).get("export_anims") or []
                ids = ["all"] if ea == "all" else [str(x) for x in ea if isinstance(x, str)] if isinstance(ea, list) else []
                if ids:
                    try:
                        import vrma
                        extra = vrma.export_for_vrm(info["vrm"], dst, ids)
                    except Exception as e:
                        return self.send(200, {"saved": dst, "size_mb": info["size_mb"], "anim_error": str(e)})
                return self.send(200, {"saved": dst, "size_mb": info["size_mb"], "anims": extra})
            if path == "/api/batch":
                items = req.get("items") or [{"path": f} for f in (req.get("files") or [])]
                outdir = req.get("outdir") or ask_dialog("folder")
                if not outdir:
                    return self.send(200, {"cancelled": True})
                os.makedirs(outdir, exist_ok=True)
                out = []
                used_dst = set()
                srcs_ = set()
                for it in items:  # every input file of the batch is protected from being overwritten
                    try:
                        srcs_.add(os.path.normcase(os.path.realpath(it.get("path") or "")))
                    except (TypeError, ValueError):
                        pass
                for it in items:
                    try:
                        src = resolve_item(it)
                        st_ = load_sidecar(src) or req.get("settings")
                        name = os.path.splitext(os.path.basename(src))[0]
                        dst = os.path.join(outdir, name + ".vrm")
                        srcs_.add(os.path.normcase(os.path.realpath(src)))
                        prot = lambda d: os.path.normcase(os.path.realpath(d)) in srcs_ or os.path.normcase(d) in used_dst
                        if prot(dst):
                            dst = os.path.join(outdir, name + "_mmd2vrm.vrm")
                        k = 2
                        while prot(dst):
                            dst = os.path.join(outdir, "%s_%d.vrm" % (name, k))
                            k += 1
                        used_dst.add(os.path.normcase(dst))
                        converter.convert(src, dst, log=lambda s: None, **settings_kwargs(st_))
                        out.append({"src": it.get("rel") or src, "dst": dst})
                    except Exception as e:
                        out.append({"src": it.get("rel") or it.get("path"), "error": str(e)})
                return self.send(200, {"results": out, "outdir": outdir})
            if path == "/api/launch":
                if not S.src:
                    return self.send(400, {"error": "Модель не открыта"})
                cfg = load_config()
                exe = cfg.get("viewer")
                if req.get("pick") or not exe or not os.path.isfile(exe):
                    exe = ask_dialog("exe")
                    if not exe:
                        return self.send(200, {"cancelled": True})
                    cfg["viewer"] = exe
                    save_config(cfg)
                info = do_convert(S.src, req.get("settings"))
                if info.get("vrm") is None:
                    return self.send(400, {"error": "Не хватает обязательных костей: " + ", ".join(info["missing"])})
                pdir = os.path.join(os.path.dirname(cache_dir()), "preview")
                os.makedirs(pdir, exist_ok=True)
                name = "".join(ch for ch in os.path.splitext(os.path.basename(S.src))[0] if ch not in '<>:"/\\|?*') or "model"
                out = os.path.join(pdir, name + ".vrm")
                if os.path.normcase(os.path.realpath(out)) == os.path.normcase(os.path.realpath(S.src)):
                    out = os.path.join(pdir, name + "_preview.vrm")
                with open(out, "wb") as f:
                    f.write(info["vrm"])
                try:
                    subprocess.Popen([exe, out], cwd=os.path.dirname(exe) or None)
                except OSError as e:
                    return self.send(400, {"error": "Не удалось запустить %s: %s" % (exe, e)})
                if req.get("reveal", True):
                    reveal(out)
                return self.send(200, {"launched": exe, "vrm": out})
            if path == "/api/upscale":
                if not S.src:
                    return self.send(400, {"error": "Модель не открыта"})
                logs = []
                try:
                    done, skipped = upscale_textures(S.src, log=logs.append, model=req.get("model"))
                except Exception as e:
                    msg = str(e)
                    if "urlopen" in msg or "HTTP" in msg or "timed out" in msg or "getaddrinfo" in msg:
                        msg = "Не удалось скачать Real-ESRGAN (нужен интернет один раз): " + msg
                    elif "vk" in msg.lower() or "vulkan" in msg.lower():
                        msg = "Видеокарта не поддерживает Vulkan: ИИ-улучшение недоступно (" + msg + ")"
                    return self.send(400, {"error": msg})
                return self.send(200, {"done": done, "skipped": skipped, "log": logs})
            if path == "/api/autotune":
                if not S.src:
                    return self.send(400, {"error": "Модель не открыта"})
                import simulate
                info = do_convert(S.src, req.get("settings"))
                logs = []
                res = simulate.autotune(info.get("chains") or [], info.get("_pivot", [0, 1, 0]), log=logs.append)
                return self.send(200, {"chains": res, "log": logs})
            if path == "/api/template_save":
                st = req.get("settings") if isinstance(req.get("settings"), dict) else {}
                dst = ask_dialog("template_save", initialfile=(os.path.splitext(os.path.basename(S.src or "шаблон"))[0] + "_шаблон.json"))
                if not dst:
                    return self.send(200, {"cancelled": True})
                with open(dst, "w", encoding="utf-8") as f:
                    json.dump({"mmd2vrm_template": 1, "settings": {k: st[k] for k in TEMPLATE_KEYS if k in st}},
                              f, ensure_ascii=False, indent=1)
                return self.send(200, {"saved": dst})
            if path == "/api/template_load":
                src = ask_dialog("template_open")
                if not src:
                    return self.send(200, {"cancelled": True})
                try:
                    with open(src, encoding="utf-8") as f:
                        t = json.load(f)
                    st = t.get("settings") if isinstance(t, dict) else None
                    if not isinstance(st, dict):
                        raise ValueError
                except (OSError, ValueError):
                    return self.send(400, {"error": "Это не шаблон mmd2vrm"})
                return self.send(200, {"settings": {k: st[k] for k in TEMPLATE_KEYS if k in st}, "path": src})
            if path == "/api/upload_done":
                sess = "".join(ch for ch in str(req.get("session") or "") if ch.isalnum())[:32]
                d = os.path.join(cache_dir(), "dropped", sess)
                if not sess or not os.path.isdir(d):
                    return self.send(400, {"error": "Нет загруженных файлов"})
                items = scan_folder(d)
                S.root = d
                converter.EXTRA_TEX_ROOTS[:] = [d]
                return self.send(200, {"root": d, "items": items})
            if path == "/api/reveal":
                p_ = req.get("path")
                if p_ and os.path.exists(p_):
                    reveal(p_)
                return self.send(200, {"ok": True})
            if path == "/api/quit":
                self.send(200, {"ok": True})
                DIALOGS.put(("quit", {}, None))
                return
            return self.send(404, {"error": "unknown"})
        except Exception as e:
            traceback.print_exc()
            return self.send(500, {"error": str(e), "trace": traceback.format_exc()})


STARTUP = {}


def find_browser():
    cands = []
    for env in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
        base = os.environ.get(env)
        if base:
            cands += [os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
                      os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")]
    for c in cands:
        if os.path.isfile(c):
            return c
    for n in ("msedge", "google-chrome", "chromium", "chrome"):
        p = shutil.which(n)
        if p:
            return p
    return None


def open_window(url):
    exe = find_browser()
    if exe:
        prof = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "mmd2vrm", "browser")
        os.makedirs(prof, exist_ok=True)
        try:
            subprocess.Popen([exe, "--app=" + url, "--window-size=1500,920",
                              "--user-data-dir=" + prof, "--no-first-run",
                              "--disable-features=Translate"])
            return
        except OSError:
            pass
    webbrowser.open(url)


def main(files=None, port=0, open_ui=True, headless_timeout=None):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = "http://127.0.0.1:%d/" % srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("mmd2vrm editor:", url, flush=True)
    if files:
        STARTUP["files"] = files
    if open_ui:
        open_window(url)
    S.last_ping = time.time() + 60  # grace period while the window starts
    while True:
        try:
            kind, kw, rq = DIALOGS.get(timeout=1.0)
        except queue.Empty:
            # window closed -> no pings
            limit = headless_timeout or 20
            if time.time() - S.last_ping > limit and open_ui:
                break
            continue
        if kind == "quit":
            break
        try:
            rq.put(run_dialog(kind, kw))
        except Exception:
            traceback.print_exc()
            rq.put(None)
    srv.shutdown()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args or None, port=int(os.environ.get("MMD2VRM_PORT", "0")),
         open_ui="--no-window" not in sys.argv)
