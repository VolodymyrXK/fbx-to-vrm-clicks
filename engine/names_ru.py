"""Offline translation of MMD / game bone & material names (JP / CN / EN) to Russian."""
import re
import unicodedata
import pinyin

# longest keys are matched first
CJK = {
    # skeleton
    "全ての親": "Корень", "操作中心": "Центр управления", "センター": "Центр", "グルーブ": "Грув",
    "上半身": "Верх тела", "下半身": "Низ тела", "腰キャンセル": "Отмена поясницы", "腰": "Поясница",
    "両目": "Оба глаза", "首": "Шея", "頭": "Голова", "头": "Голова", "目": "Глаз", "眼": "Глаз",
    "肩": "Плечо", "腕捩": "Скрут руки", "手捩": "Скрут кисти", "腕": "Рука", "ひじ": "Локоть", "肘": "Локоть",
    "手首": "Запястье", "手腕": "Запястье", "手": "Кисть",
    "親指": "Большой палец", "人差指": "Указательный", "人指": "Указательный", "中指": "Средний",
    "薬指": "Безымянный", "小指": "Мизинец", "拇指": "Большой палец", "食指": "Указательный",
    "无名指": "Безымянный", "足首": "Лодыжка", "足先": "Носок", "つま先": "Носок", "足": "Нога",
    "ひざ": "Колено", "膝": "Колено", "脚": "Нога", "大腿": "Бедро", "小腿": "Голень",
    "捩": "скрут", "皮肤": "Кожа", "皮膚": "Кожа", "錘": "грузик", "锤": "грузик", "補助": "доп.", "调整": "настр.", "調整": "настр.", "先": "кончик", "親": "родитель",
    "ＩＫ": "ИК", "IK": "ИК", "顎": "Челюсть", "あご": "Челюсть", "舌": "Язык", "歯": "Зубы",
    # hair
    "前髪": "Чёлка", "後ろ髪": "Волосы сзади", "後髪": "Волосы сзади", "横髪": "Волосы сбоку",
    "もみあげ": "Височная прядь", "アホ毛": "Хохолок", "ツインテール": "Хвостик", "ツインテ": "Хвостик",
    "おさげ": "Косичка", "三つ編み": "Коса", "ポニーテール": "Хвост (причёска)", "ポニテ": "Хвост (причёска)",
    "髪": "Волосы", "頭髪": "Волосы", "头发": "Волосы", "頭髮": "Волосы", "前发": "Чёлка", "后发": "Волосы сзади",
    "侧发": "Волосы сбоку", "鬓发": "Височная прядь", "辫子": "Коса", "马尾": "Хвост (причёска)",
    "呆毛": "Хохолок", "发": "Волосы", "髮": "Волосы",
    # clothes / parts
    "スカート": "Юбка", "裙子": "Юбка", "前裙": "Юбка спереди", "后裙": "Юбка сзади", "侧裙": "Юбка сбоку",
    "裙": "Юбка", "服": "Одежда", "衣服": "Одежда", "袖": "Рукав", "襟": "Воротник", "领": "Воротник",
    "帽子": "Шляпа", "帽": "Шляпа", "リボン": "Лента", "丝带": "Лента", "紐": "Шнурок", "ネクタイ": "Галстук",
    "上着": "Куртка", "コート": "Пальто", "外套": "Пальто", "マント": "Плащ", "披风": "Плащ", "フード": "Капюшон",
    "胸": "Грудь", "おっぱい": "Грудь", "乳": "Грудь", "尻尾": "Хвост", "しっぽ": "Хвост", "シッポ": "Хвост",
    "尾巴": "Хвост", "尾": "Хвост", "耳": "Ухо", "角": "Рог", "翼": "Крыло", "羽": "Крыло", "翅膀": "Крыло",
    "飾り": "Украшение", "装飾": "Украшение", "饰": "Украшение", "武器": "Оружие", "剣": "Меч", "剑": "Меч",
    "銃": "Пистолет", "枪": "Пистолет", "鞄": "Сумка", "包": "Сумка", "ベルト": "Ремень", "腰带": "Пояс",
    "靴": "Обувь", "鞋": "Обувь", "手袋": "Перчатка", "手套": "Перчатка", "マフラー": "Шарф", "围巾": "Шарф",
    "ネックレス": "Ожерелье", "项链": "Ожерелье", "イヤリング": "Серьга", "耳环": "Серьга", "眼鏡": "Очки",
    "鼻子": "Нос", "鼻": "Нос", "アーム": "Рука (мех.)", "エンジン": "Двигатель", "亲": "родитель",
    "臀": "Ягодица", "尻": "Ягодица", "肚": "Живот", "腹": "Живот", "大": "большой", "小": "малый",
    # materials
    "肌": "Кожа", "体": "Тело", "身体": "Тело", "顔": "Лицо", "脸": "Лицо", "面": "Лицо", "目白": "Белки глаз",
    "白目": "Белки глаз", "目影": "Тень глаз", "睫毛": "Ресницы", "睫": "Ресницы", "まつ毛": "Ресницы",
    "眉毛": "Брови", "眉": "Брови", "口内": "Рот", "口舌": "Рот", "口": "Рот", "表情": "Эмоции",
    "发影": "Тень волос", "影": "тень", "瞳": "Зрачок", "ハイライト": "блик", "轮廓线": "контур",
    # positions
    "前": "перед", "後": "зад", "后": "зад", "横": "бок", "侧": "бок", "上": "верх", "下": "низ",
    "中": "сред.", "根": "корень", "元": "основание", "内": "внутр.", "外": "внешн.",
}
SIDE_CJK = {"左": "Л", "右": "П"}

EN = {
    "hair": "Волосы", "bangs": "Чёлка", "bang": "Чёлка", "fringe": "Чёлка", "strand": "Прядь",
    "ponytail": "Хвост (причёска)", "twintail": "Хвостик", "braid": "Коса", "ahoge": "Хохолок",
    "skirt": "Юбка", "dress": "Платье", "cloth": "Ткань", "clothes": "Одежда", "coat": "Пальто",
    "jacket": "Куртка", "cuff": "Манжета", "sleeve": "Рукав", "collar": "Воротник", "hood": "Капюшон",
    "cape": "Плащ", "cloak": "Плащ", "scarf": "Шарф", "tie": "Галстук", "ribbon": "Лента", "bow": "Бант",
    "rope": "Верёвка", "string": "Шнурок", "belt": "Ремень", "bag": "Сумка", "beltbag": "Поясная сумка",
    "gadget": "Устройство", "acc": "Аксессуар", "accessory": "Аксессуар", "deco": "Украшение",
    "tail": "Хвост", "ear": "Ухо", "ears": "Уши", "horn": "Рог", "wing": "Крыло", "hat": "Шляпа",
    "cap": "Кепка", "weapon": "Оружие", "sword": "Меч", "gun": "Пистолет", "breast": "Грудь",
    "bust": "Грудь", "chest": "Грудь", "spine": "Позвоночник", "hips": "Таз", "hip": "Бедро (таз)",
    "pelvis": "Таз", "waist": "Талия", "neck": "Шея", "head": "Голова", "face": "Лицо", "eye": "Глаз",
    "eyes": "Глаза", "jaw": "Челюсть", "tongue": "Язык", "shoulder": "Плечо", "clavicle": "Ключица",
    "arm": "Рука", "upperarm": "Плечо (рука)", "forearm": "Предплечье", "lowerarm": "Предплечье",
    "elbow": "Локоть", "hand": "Кисть", "wrist": "Запястье", "finger": "Палец", "thumb": "Большой",
    "index": "Указательный", "middle": "Средний", "ring": "Безымянный", "pinky": "Мизинец",
    "little": "Мизинец", "leg": "Нога", "thigh": "Бедро", "upperleg": "Бедро", "calf": "Голень",
    "lowerleg": "Голень", "shin": "Голень", "knee": "Колено", "foot": "Стопа", "ankle": "Лодыжка",
    "toe": "Носок", "toes": "Пальцы ног", "root": "Корень", "base": "основа", "side": "бок",
    "back": "зад", "front": "перед", "top": "верх", "bottom": "низ", "upper": "верх", "lower": "низ",
    "inner": "внутр.", "outer": "внешн.", "end": "кончик", "tip": "кончик", "twist": "скрут",
    "body": "Тело", "skin": "Кожа", "center": "Центр", "mouth": "Рот", "brow": "Брови", "lash": "Ресницы",
    "teeth": "Зубы", "glasses": "Очки", "shoe": "Обувь", "shoes": "Обувь", "boot": "Сапог",
    "boots": "Сапоги", "glove": "Перчатка", "sock": "Носок", "pants": "Брюки", "shirt": "Рубашка",
    "necklace": "Ожерелье", "earring": "Серьга", "chain": "Цепь", "flower": "Цветок", "crown": "Корона",
    "bell": "Колокольчик", "pendant": "Кулон", "badge": "Значок", "strap": "Ремешок", "frill": "Оборка",
    "lace": "Кружево", "apron": "Фартук", "vest": "Жилет", "armor": "Броня", "plate": "Пластина",
    "tooth": "Зуб", "line": "Линия", "nose": "Нос", "cheek": "Щека", "lip": "Губа", "lid": "Веко",
    "eyelid": "Веко", "pupil": "Зрачок", "weiba": "Хвост", "toufa": "Волосы", "qunzi": "Юбка",
    "qun": "Юбка", "xiong": "Грудь", "erduo": "Ухо", "toushi": "Головной убор", "fa": "Волосы",
    "scapula": "Лопатка", "minus": "минус", "plus": "плюс", "corrective": "корректор", "xiuko": "Манжета",
    "paizi": "Табличка", "push": "упор", "butt": "Ягодица", "highlight": "Блик", "iris": "Радужка", "irissd": "Радужка тень",
    "pouch": "Подсумок", "bowknot": "Бант", "hairpin": "Заколка", "band": "Лента", "biaoqian": "Бирка",
    "clotch": "Ткань", "piyi": "Накидка", "jinqiko": "Прядь у шеи", "nightgown": "Ночная рубашка",
    "gown": "Платье", "hem": "Подол", "lf": "Л", "rt": "П", "sd": "тень", "neckband": "Воротник",
    "bust": "Грудь", "flesh": "мягкость", "jiggle": "покачивание", "chain": "Цепь", "tassel": "Кисточка",
    "physics": "физика", "phys": "физика", "spring": "пружина", "dummy": "пустышка",
}
SIDE_EN = {"l": "Л", "left": "Л", "lf": "Л", "r": "П", "right": "П", "rt": "П"}
DROP_EN = {"jnt", "jt", "bone", "def", "joint", "mixamorig", "bip", "bip001", "bip01", "j", "c", "m", "bn",
           "joint", "alpha"}

_KEYS = sorted(CJK, key=len, reverse=True)
_EN_SEG = {w for w in EN if len(w) >= 3}
_HAS_CJK = re.compile(r"[぀-ヿ㐀-鿿ｦ-ﾟ]")


def _en_words(chunk):
    # split camelCase and letter/digit boundaries
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", chunk)
    return parts


def translate(name):
    """Return a Russian reading of a bone/material name, or the name itself if nothing matched."""
    if not name:
        return name
    s = unicodedata.normalize("NFKC", name)
    s = re.sub(r"(?i)(^|[_ .:])bip0*\d*(?=[_ .:])", r"\1", s)  # Bip001_L_Hand -> _L_Hand
    words, side, hit = [], None, False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch in SIDE_CJK and side is None:
            side = SIDE_CJK[ch]
            i += 1
            hit = True
            continue
        m = None
        if _HAS_CJK.match(ch) or ch in "ＩＫ":
            for k in _KEYS:
                if s.startswith(k, i):
                    m = k
                    break
        if m:
            words.append(CJK[m])
            hit = True
            i += len(m)
            continue
        if ch.isascii() and ch.isalnum():
            j = i
            while j < n and s[j].isascii() and s[j].isalnum():
                j += 1
            ws = _en_words(s[i:j])
            merged = []
            for w in ws:  # UpperArm -> upperarm, ForeArm -> forearm
                if merged and (merged[-1] + w).lower() in EN and not merged[-1].isdigit():
                    merged[-1] = merged[-1] + w
                else:
                    merged.append(w)
            for w in merged:
                lw = w.lower()
                if lw.isdigit():
                    words.append(lw)
                elif lw in SIDE_EN and side is None and (len(lw) > 1 or True):
                    side = SIDE_EN[lw]
                    hit = True
                elif lw in DROP_EN:
                    hit = True
                elif lw in EN:
                    words.append(EN[lw])
                    hit = True
                elif lw == "ik":
                    words.append("ИК")
                    hit = True
                elif lw in pinyin.LEX:
                    words.append(pinyin.LEX[lw][0])
                    hit = True
                else:
                    seg = pinyin.segment(lw, _EN_SEG) if len(lw) > 3 else None
                    if seg and len(seg) <= 4:
                        for p in seg:
                            if p in pinyin.SIDE and side is None:
                                side = SIDE_EN[pinyin.SIDE[p].lower()]
                            elif p in pinyin.LEX:
                                words.append(pinyin.LEX[p][0])
                            elif p in EN:
                                words.append(EN[p])
                            else:
                                words.append(p.upper())
                        hit = True
                    else:
                        words.append(w)
            i = j
            continue
        if _HAS_CJK.match(ch):  # unknown CJK / kana: keep the run together (ダミー, not ダ ミ ー)
            j = i + 1
            while j < n and _HAS_CJK.match(s[j]) and s[j] not in SIDE_CJK and not any(s.startswith(k, j) for k in _KEYS):
                j += 1
            words.append(s[i:j])
            i = j
            continue
        if ch.isalpha():  # Cyrillic, accented latin ...: a word as it is
            j = i + 1
            while j < n and s[j].isalnum() and not s[j].isascii() and not _HAS_CJK.match(s[j]):
                j += 1
            words.append(s[i:j])
            i = j
            continue
        i += 1
    if not hit:
        return name
    # merge consecutive numbers: ['Юбка','0','3'] -> 'Юбка 0_3'
    out = []
    for w in words:
        if out and w.isdigit() and out[-1].replace("_", "").isdigit():
            out[-1] = out[-1] + "_" + w
        else:
            out.append(w)
    # drop duplicated consecutive words
    ded = []
    for w in out:
        if not ded or ded[-1].lower() != w.lower():
            ded.append(w)
    text = " ".join(ded).strip()
    if side:
        text = (text + " " + side).strip()
    if not text or text.replace("_", "").replace(" ", "").isdigit():
        return name
    return text[0].upper() + text[1:]


if __name__ == "__main__":
    import sys
    for a in sys.argv[1:]:
        print(a, "->", translate(a))
