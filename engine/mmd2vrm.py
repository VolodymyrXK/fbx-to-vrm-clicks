"""MMD -> VRM converter.

  mmd2vrm.exe                        -> editor window with 3D preview
  mmd2vrm.exe model.pmx              -> editor window with this model (drag .pmx onto the exe)
  mmd2vrm.exe --cli model.pmx [...]  -> command-line conversion, writes model.vrm next to it
  mmd2vrm.exe model.pmx -o out.vrm --physics realistic --scale 0.08 ...
"""
import argparse
import math
import os
import sys
import traceback

import converter


def run_cli(argv):
    ap = argparse.ArgumentParser(prog="mmd2vrm", description="MMD (.pmx) -> VRM 0.x")
    ap.add_argument("inputs", nargs="*", help=".pmx / .pmd / .fbx файлы")
    ap.add_argument("-o", "--output", help="выходной .vrm (только для одного файла)")
    ap.add_argument("--scale", type=float, default=0.08, help="метров на единицу MMD (0.08)")
    ap.add_argument("--no-tpose", action="store_true", help="не переводить руки в T-позу")
    ap.add_argument("--no-spring", action="store_true", help="не создавать spring bones из физики")
    ap.add_argument("--outline", action="store_true", help="перенести контур (edge) в MToon outline")
    ap.add_argument("--presets-only", action="store_true", help="только пресетные выражения")
    ap.add_argument("--max-texture", type=int, default=2048,
                    help="макс. размер текстуры в px (0 = не уменьшать), по умолчанию 2048")
    ap.add_argument("--physics", default="anime", choices=["anime", "realistic", "stiff", "soft", "dance", "heavy", "mmd"],
                    help="стиль физики: anime (по умолч.), realistic, stiff, mmd")
    ap.add_argument("--ru-names", action="store_true", help="русские имена костей в VRM")
    ap.add_argument("--no-face-bones", action="store_true", help="не создавать моргание/рот из костей век и челюсти")
    ap.add_argument("--no-anime-face", action="store_true", help="не выравнивать нормали лица (объёмные 3D-тени)")
    ap.add_argument("--en-names", action="store_true", help="записать имена костей по-английски")
    ap.add_argument("--best-textures", action="store_true", help="текстуры без сжатия (PNG), до 4096 px")
    ap.add_argument("--sharpen", type=float, default=0, metavar="0-100", help="резкость текстур, %%")
    ap.add_argument("--fill-body", default="", metavar="ЧАСТИ",
                    help="достроить тело там, где его нет: feet,legs,torso,arms,head (через запятую)")
    ap.add_argument("--strip-hidden", action="store_true", help="отключать физику скрытых частей")
    ap.add_argument("--use-saved", action="store_true", help="применить правки из редактора (файл .mmd2vrm.json рядом с моделью)")
    ap.add_argument("--vrm1", action="store_true", help="сохранить в формате VRM 1.0 (по умолчанию VRM 0.x)")
    ap.add_argument("--max-mb", type=float, default=0, help="уменьшать текстуры, пока файл не станет меньше N МБ")
    ap.add_argument("--repair", action="store_true", help="восстановить недостающие кости (локти, запястья, колени...)")
    ap.add_argument("--anims", default="", metavar="ID,ID|all",
                    help="сохранить встроенные анимации рядом с VRM (.vrma): список id или all; --anims list — показать список")
    ap.add_argument("--title")
    ap.add_argument("--author")
    ap.add_argument("--pause", action="store_true", help="ждать Enter в конце")
    a = ap.parse_args(argv)
    if not a.inputs and (a.anims or "").strip().lower() != "list":
        ap.error("укажите хотя бы один файл модели")
    ok = True
    inputs_ = {os.path.normcase(os.path.realpath(x)) for x in a.inputs}
    ALL_PARTS = ["feet", "legs", "torso", "arms", "head"]
    fill_parts = []
    for x in (a.fill_body or "").split(","):
        x = x.strip().lower()
        if not x:
            continue
        if x == "all":
            fill_parts = list(ALL_PARTS)
        elif x in ALL_PARTS:
            fill_parts.append(x) if x not in fill_parts else None
        else:
            print("! --fill-body: неизвестная часть «%s» (можно: %s, all)" % (x, ", ".join(ALL_PARTS)))
    if a.anims.strip() == "list":
        import vrma
        for i, an in vrma.load_anims().items():
            print("  %-16s %s (%s)" % (i, an["name"], an["group"]))
        return 0

    def anims_after(dst):
        if not a.anims.strip():
            return
        import vrma
        ids = "all" if a.anims.strip() == "all" else [x.strip() for x in a.anims.split(",") if x.strip()]
        with open(dst, "rb") as f_:
            vrma.export_for_vrm(f_.read(), dst, ids)

    for src in a.inputs:
        dst = a.output if (a.output and len(a.inputs) == 1) else os.path.splitext(src)[0] + ".vrm"
        if not (a.output and len(a.inputs) == 1) and os.path.normcase(os.path.realpath(dst)) in inputs_:
            b_, e_ = os.path.splitext(src)
            dst = b_ + "_" + e_.lstrip(".").lower() + ".vrm"  # never overwrite one of the input files
        if os.path.normcase(os.path.realpath(dst)) in inputs_:
            ok = False
            print("ОШИБКА (%s): файл результата совпадает с исходной моделью" % os.path.basename(src))
            continue
        if a.scale <= 0:
            print("ОШИБКА: --scale должен быть больше нуля")
            return 1
        try:
            if a.use_saved:
                import app as _app
                st = _app.load_sidecar(src)
                if st:
                    print("  правки из редактора: %s" % _app.sidecar(src))
                    converter.convert(src, dst, **_app.settings_kwargs(st))
                    anims_after(dst)
                    continue
                print("  сохранённых правок нет — обычная конвертация")
            converter.convert(src, dst, scale=a.scale, tpose=not a.no_tpose,
                              spring=not a.no_spring, outline=a.outline,
                              custom_blendshapes=not a.presets_only,
                              title=a.title, author=a.author, physics=a.physics,
                              max_texture=(4096 if a.best_textures and a.max_texture == 2048 else a.max_texture),
                              ru_names=a.ru_names, repair=a.repair,
                              face_from_bones=not a.no_face_bones,
                              anime_face=not a.no_anime_face,
                              vrm_version=1 if a.vrm1 else 0,
                              target_mb=a.max_mb,
                              bone_lang="en" if a.en_names else None,
                              tex_quality="best" if a.best_textures else "normal",
                              tex_sharpen=(max(0.0, min(1.0, a.sharpen / 100.0)) if math.isfinite(a.sharpen) else 0.0),
                              strip_hidden=a.strip_hidden,
                              overrides={"body_fill": fill_parts} if fill_parts else None)
            anims_after(dst)
        except Exception as e:
            ok = False
            print("ОШИБКА (%s): %s" % (os.path.basename(src), e))
            if not isinstance(e, (converter.ConvertError,)):
                traceback.print_exc()
    if a.pause:
        input("\nНажмите Enter...")
    return 0 if ok else 1


def main():
    argv = sys.argv[1:]
    if argv and argv[0] in ("--cli", "convert"):
        sys.exit(run_cli([a for a in argv[1:] if a != "--no-window"]))
    files = [a for a in argv if not a.startswith("-")]
    opts = [a for a in argv if a.startswith("-") and a != "--no-window"]
    if (opts and files) or any(o in ("-h", "--help") or o.startswith("--anims") for o in opts):
        sys.exit(run_cli([a for a in argv if a != "--no-window"]))  # options given -> command line conversion
    import app
    app.main(files or None, open_ui="--no-window" not in argv)


if __name__ == "__main__":
    main()
