"""FBX to VRM in a couple of clicks — Astra plugin.

Wraps the mmd2vrm converter (engine/): MMD (.pmx/.pmd), FBX and VRM models become VRM 0.x / 1.0
avatars. The assistant gets three tools; the full editor (3D preview, physics, materials) opens in the
user's browser like the stand-alone program.
"""

import asyncio
import os
import subprocess
import sys
import threading

from astra_plugin_sdk import Plugin, tool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "engine")
if ENGINE not in sys.path:
    sys.path.insert(0, ENGINE)

MODEL_EXT = (".pmx", ".pmd", ".fbx", ".vrm")
PHYSICS = ("anime", "realistic", "stiff", "soft", "dance", "heavy", "mmd")
_LOCK = threading.Lock()  # the converter keeps module-level caches: one conversion at a time


def _clean_path(p):
    p = (p or "").strip().strip('"').strip("'").strip()
    return os.path.abspath(os.path.expandvars(os.path.expanduser(p))) if p else ""


def _as_bool(v, default):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return default if v is None else bool(v)


def _as_num(v, default):
    try:
        x = float(v)
        return x if x == x else default
    except (TypeError, ValueError):
        return default


def _output_for(src, output, folder):
    """where the VRM goes: the given path, else the settings folder, else next to the model; never over
    the source file itself"""
    if output:
        dst = _clean_path(output)
        if os.path.isdir(dst):
            dst = os.path.join(dst, os.path.splitext(os.path.basename(src))[0] + ".vrm")
        elif not dst.lower().endswith(".vrm"):
            dst += ".vrm"
    else:
        d = _clean_path(folder) if folder else os.path.dirname(src)
        dst = os.path.join(d, os.path.splitext(os.path.basename(src))[0] + ".vrm")
    if os.path.normcase(os.path.realpath(dst)) == os.path.normcase(os.path.realpath(src)):
        b, e = os.path.splitext(dst)
        dst = b + "_converted" + e
    return dst


def _convert_sync(src, dst, cfg, vrm_version, use_editor_edits, animations):
    import converter  # heavy (numpy, PIL): loaded on first use, not at start-up
    lines = []

    def log(s):
        lines.append(str(s))

    with _LOCK:
        kw = None
        edits = None
        if use_editor_edits:
            try:
                import app as _app
                st = _app.load_sidecar(src)
                if st:
                    kw = _app.settings_kwargs(st)
                    edits = _app.sidecar(src)
            except Exception as e:  # a broken edits file must not stop a plain conversion
                log("! правки из редактора не прочитаны: %s" % e)
        if kw is None:
            phys = str(cfg.get("physics") or "soft")
            kw = dict(physics=phys if phys in PHYSICS else "soft",
                      tex_quality="best" if _as_bool(cfg.get("best_textures"), False) else "normal",
                      max_texture=4096 if _as_bool(cfg.get("best_textures"), False) else 2048,
                      target_mb=max(0.0, _as_num(cfg.get("max_mb"), 0.0)),
                      ru_names=_as_bool(cfg.get("ru_names"), False))
        if vrm_version in (0, 1):
            kw["vrm_version"] = vrm_version
        elif "vrm_version" not in kw:
            kw["vrm_version"] = 1 if str(cfg.get("vrm_version", "0")).startswith("1") else 0
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        converter.convert(src, dst, log=log, **kw)
        anim_files = []
        ids = (animations or "").strip()
        if ids:
            import vrma
            want = "all" if ids.lower() == "all" else [x.strip() for x in ids.split(",") if x.strip()]
            known = set(vrma.load_anims())
            if want != "all":
                bad = [x for x in want if x not in known]
                want = [x for x in want if x in known]
                if bad:
                    log("! неизвестные анимации: %s" % ", ".join(bad))
            if want:
                with open(dst, "rb") as f:
                    res = vrma.export_for_vrm(f.read(), dst, want, log=log)
                if isinstance(res, (list, tuple)):
                    anim_files = [str(x) for x in res]
    return lines, kw["vrm_version"], edits, anim_files


class FbxToVrmClicks(Plugin):
    """Astra plugin: FBX to VRM in a couple of clicks."""

    @tool("Convert a 3D character model (MMD .pmx/.pmd, FBX or VRM file on this computer) into a VRM avatar. "
          "Returns the path of the saved .vrm file. Use when the user asks to make a VRM from a model file.")
    async def convert_to_vrm(self, path: str, output: str = "", vrm_version: int = -1,
                             use_editor_edits: bool = True, animations: str = ""):
        """path: the model file. output: optional .vrm path or folder (default: next to the model, or the
        folder from the plugin settings). vrm_version: 0 = VRM 0.x, 1 = VRM 1.0, -1 = from settings.
        use_editor_edits: apply the tweaks saved for this model in the editor. animations: optional
        comma-separated built-in animation ids (or "all") saved next to the VRM as .vrma files."""
        src = _clean_path(path)
        if not src or not os.path.isfile(src):
            raise ValueError("Файл модели не найден: %s" % (path or "(пусто)"))
        if not src.lower().endswith(MODEL_EXT):
            raise ValueError("Нужен файл .pmx, .pmd, .fbx или .vrm, а получен: %s" % os.path.basename(src))
        cfg = self.config if isinstance(self.config, dict) else {}
        dst = _output_for(src, output, cfg.get("output_folder"))
        try:
            vv = int(vrm_version)
        except (TypeError, ValueError):
            vv = -1
        lines, used_v, edits, anim_files = await asyncio.to_thread(
            _convert_sync, src, dst, cfg, vv, _as_bool(use_editor_edits, True), animations)
        size_mb = os.path.getsize(dst) / 1048576.0 if os.path.isfile(dst) else 0.0
        warnings = [l.strip() for l in lines if l.strip().startswith("!")]
        return {
            "ok": True,
            "vrm": dst,
            "vrm_version": "1.0" if used_v == 1 else "0.x",
            "size_mb": round(size_mb, 1),
            "editor_edits_used": edits or "",
            "animations": anim_files,
            "warnings": warnings[:20],
            "log_tail": [l for l in lines if l.strip()][-12:],
            "source_kept": os.path.isfile(src),
            "tip": "For more precise tuning (physics, expressions, materials, skeleton) open the model in the "
                   "editor with open_vrm_editor; its settings are reused by the next conversion. / "
                   "Для более точной настройки откройте модель в редакторе (open_vrm_editor) — "
                   "настройки сохранятся и будут использованы при следующей конвертации.",
        }

    @tool("Open the full mmd2vrm editor (3D preview, physics, materials, animations) in the "
          "browser, optionally with a model file already loaded.")
    async def open_vrm_editor(self, path: str = ""):
        src = _clean_path(path)
        if src and not os.path.isfile(src):
            raise ValueError("Файл модели не найден: %s" % path)
        args = [sys.executable, os.path.join(ENGINE, "app.py")] + ([src] if src else [])
        kw = {"cwd": ENGINE, "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
              "stderr": subprocess.DEVNULL}
        if os.name == "nt":
            kw["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        subprocess.Popen(args, **kw)
        return {"ok": True, "opened": src or "(без модели)"}

    @tool("List the built-in animations that can be saved next to a VRM as .vrma files "
          "(ids for the 'animations' argument of convert_to_vrm).")
    async def list_vrm_animations(self):
        def _list():
            import vrma
            return [{"id": i, "name": a.get("name", i), "group": a.get("group", "")}
                    for i, a in vrma.load_anims().items()]
        return {"animations": await asyncio.to_thread(_list)}


if __name__ == "__main__":
    FbxToVrmClicks().run()
