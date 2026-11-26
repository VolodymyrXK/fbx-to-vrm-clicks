"""Offline translation of MMD / game bone & material names (JP / CN / pinyin) to English.
Result is a VRM-friendly ASCII name: words in CamelCase joined with "_", side as "_L" / "_R"."""
import re
import unicodedata

import names_ru
import pinyin

CJK = {
    "皮肤": "Skin", "皮膚": "Skin",
    "錘": "Weight", "锤": "Weight",
    # skeleton
    "全ての親": "Root", "操作中心": "ControlCenter", "センター": "Center", "グルーブ": "Groove",
    "上半身": "UpperBody", "下半身": "LowerBody", "腰キャンセル": "WaistCancel", "腰": "Waist",
    "両目": "Eyes", "首": "Neck", "頭": "Head", "头": "Head", "目": "Eye", "眼": "Eye",
    "肩": "Shoulder", "腕捩": "ArmTwist", "手捩": "WristTwist", "腕": "Arm", "ひじ": "Elbow", "肘": "Elbow",
    "手首": "Wrist", "手腕": "Wrist", "手": "Hand",
    "親指": "Thumb", "人差指": "Index", "人指": "Index", "中指": "Middle",
    "薬指": "Ring", "小指": "Little", "拇指": "Thumb", "食指": "Index",
    "无名指": "Ring", "足首": "Ankle", "足先": "ToeTip", "つま先": "Toe", "足": "Leg",
    "ひざ": "Knee", "膝": "Knee", "脚": "Leg", "大腿": "Thigh", "小腿": "Calf",
    "捩": "Twist", "補助": "Helper", "调整": "Adjust", "調整": "Adjust", "先": "Tip", "親": "Parent",
    "ＩＫ": "IK", "顎": "Jaw", "あご": "Jaw", "舌": "Tongue", "歯": "Teeth",
    "ふくらはぎ": "Calf", "太もも": "Thigh",
    # hair
    "前髪": "HairFront", "後ろ髪": "HairBack", "後髪": "HairBack", "横髪": "HairSide",
    "もみあげ": "Sideburn", "アホ毛": "Ahoge", "ツインテール": "Twintail", "ツインテ": "Twintail",
    "おさげ": "Braid", "三つ編み": "Braid", "ポニーテール": "Ponytail", "ポニテ": "Ponytail",
    "髪": "Hair", "頭髪": "Hair", "头发": "Hair", "頭髮": "Hair", "前发": "HairFront", "后发": "HairBack",
    "侧发": "HairSide", "鬓发": "Sideburn", "辫子": "Braid", "马尾": "Ponytail",
    "呆毛": "Ahoge", "发": "Hair", "髮": "Hair",
    # clothes / parts
    "スカート": "Skirt", "裙子": "Skirt", "前裙": "SkirtFront", "后裙": "SkirtBack", "侧裙": "SkirtSide",
    "裙": "Skirt", "服": "Clothes", "衣服": "Clothes", "袖": "Sleeve", "襟": "Collar", "领": "Collar",
    "帽子": "Hat", "帽": "Hat", "リボン": "Ribbon", "丝带": "Ribbon", "紐": "String", "ネクタイ": "Tie",
    "上着": "Jacket", "コート": "Coat", "外套": "Coat", "マント": "Cape", "披风": "Cape", "フード": "Hood",
    "胸": "Breast", "おっぱい": "Breast", "乳": "Breast", "尻尾": "Tail", "しっぽ": "Tail", "シッポ": "Tail",
    "尾巴": "Tail", "尾": "Tail", "耳": "Ear", "角": "Horn", "翼": "Wing", "羽": "Wing", "翅膀": "Wing",
    "飾り": "Deco", "装飾": "Deco", "饰": "Deco", "武器": "Weapon", "剣": "Sword", "剑": "Sword",
    "銃": "Gun", "枪": "Gun", "鞄": "Bag", "包": "Bag", "ベルト": "Belt", "腰带": "Belt",
    "靴": "Shoe", "鞋": "Shoe", "手袋": "Glove", "手套": "Glove", "マフラー": "Scarf", "围巾": "Scarf",
    "ネックレス": "Necklace", "项链": "Necklace", "イヤリング": "Earring", "耳环": "Earring", "眼鏡": "Glasses",
    "鼻子": "Nose", "鼻": "Nose", "アーム": "MechArm", "エンジン": "Engine", "亲": "Parent",
    "臀": "Butt", "尻": "Butt", "肚": "Belly", "腹": "Belly", "大": "Big", "小": "Small",
    # materials
    "肌": "Skin", "体": "Body", "身体": "Body", "顔": "Face", "脸": "Face", "面": "Face", "目白": "EyeWhite",
    "白目": "EyeWhite", "目影": "EyeShadow", "睫毛": "Eyelash", "睫": "Eyelash", "まつ毛": "Eyelash",
    "眉毛": "Eyebrow", "眉": "Eyebrow", "口内": "MouthInner", "口舌": "Mouth", "口": "Mouth", "表情": "Expression",
    "发影": "HairShadow", "影": "Shadow", "瞳": "Pupil", "ハイライト": "Highlight", "轮廓线": "Outline",
    # positions
    "前": "Front", "後": "Back", "后": "Back", "横": "Side", "侧": "Side", "上": "Upper", "下": "Lower",
    "中": "Mid", "根": "Root", "元": "Base", "内": "Inner", "外": "Outer",
}
PINYIN = {"weiba": "Tail", "toufa": "Hair", "qunzi": "Skirt", "qun": "Skirt", "xiong": "Breast",
          "erduo": "Ear", "toushi": "Headwear", "fa": "Hair", "liuhai": "Bangs", "mawei": "Ponytail",
          "bianzi": "Braid", "xiuzi": "Sleeve", "xiudai": "SleeveBand"}
SIDE_CJK = {"左": "L", "右": "R"}
SIDE_EN = {"l": "L", "left": "L", "lf": "L", "r": "R", "right": "R", "rt": "R"}
DROP = names_ru.DROP_EN
_KEYS = sorted(CJK, key=len, reverse=True)
_HAS_CJK = names_ru._HAS_CJK


def _cap(w):
    return w[:1].upper() + w[1:]


_CYR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                ["a", "b", "v", "g", "d", "e", "yo", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s",
                 "t", "u", "f", "kh", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"]))
_KANA_ROWS = ("a i u e o ka ki ku ke ko sa shi su se so ta chi tsu te to na ni nu ne no ha hi fu he ho "
              "ma mi mu me mo ya yu yo ra ri ru re ro wa wo n ga gi gu ge go za ji zu ze zo da di du de do "
              "ba bi bu be bo pa pi pu pe po").split()
_HIRA = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ"
_KANA = {}
for _c, _r in zip(_HIRA, _KANA_ROWS):
    _KANA[_c] = _r
    _KANA[chr(ord(_c) + 0x60)] = _r  # katakana
_KANA.update({"ー": "", "ッ": "", "っ": "", "ャ": "ya", "ュ": "yu", "ョ": "yo", "ゃ": "ya", "ゅ": "yu", "ょ": "yo",
              "ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o", "ヴ": "vu"})


def translit(text):
    """ASCII spelling of letters we cannot translate (Cyrillic, accents, kana); kanji are dropped"""
    out = []
    for ch in text:
        lo = ch.lower()
        if ch.isascii():
            out.append(ch)
        elif lo in _CYR:
            r = _CYR[lo]
            out.append(r.capitalize() if ch != lo and r else r)
        elif ch in _KANA:
            out.append(_KANA[ch])
        else:
            d = unicodedata.normalize("NFKD", ch)
            out.append("".join(c for c in d if c.isascii()))
    return re.sub(r"[^A-Za-z0-9_]+", "_", "".join(out)).strip("_")


def translate(name):
    """English, ASCII reading of a bone/material name (always ASCII; 'Bone' when nothing is readable)."""
    if not name:
        return name
    r = _translate(name)
    if r and r.isascii():
        return r
    t = translit(r or name)
    return t if re.search(r"[A-Za-z]", t or "") else ("Bone_" + t if t else "Bone")


def _translate(name):
    if not name:
        return name
    s = unicodedata.normalize("NFKC", name)
    s = re.sub(r"(?i)(^|[_ .:])bip0*\d*(?=[_ .:])", r"\1", s)  # Bip001_L_Hand -> _L_Hand
    words, side, hit, unknown = [], None, False, False
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch in SIDE_CJK and side is None:
            side = SIDE_CJK[ch]
            hit = True
            i += 1
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
            for w in names_ru._en_words(s[i:j]):
                lw = w.lower()
                if lw.isdigit():
                    words.append(lw)
                elif lw in SIDE_EN and side is None:
                    side = SIDE_EN[lw]
                elif lw in DROP:
                    pass
                elif lw in PINYIN:
                    words.append(PINYIN[lw])
                    hit = True
                elif lw in pinyin.LEX:
                    words.append(pinyin.LEX[lw][1])
                    hit = True
                else:
                    seg = pinyin.segment(lw, names_ru._EN_SEG) if len(lw) > 3 else None
                    if seg and len(seg) <= 4:
                        for p in seg:
                            if p in pinyin.SIDE and side is None:
                                side = pinyin.SIDE[p]
                            elif p in pinyin.LEX:
                                words.append(pinyin.LEX[p][1])
                            else:
                                words.append(_cap(p) if len(p) > 1 else p.upper())
                        hit = True
                    else:
                        words.append(_cap(w))
            i = j
            continue
        if _HAS_CJK.match(ch):
            unknown = True  # a character we cannot translate
            if ch in _KANA:  # kana run: romaji (ダミー -> Dami)
                j = i
                while j < n and s[j] in _KANA:
                    j += 1
                words.append(_cap(translit(s[i:j])) or "Bone")
                i = j
                continue
        elif ch.isalpha():  # Cyrillic / accented word: transliterate
            j = i + 1
            while j < n and s[j].isalnum() and not s[j].isascii() and not _HAS_CJK.match(s[j]):
                j += 1
            t = translit(s[i:j])
            if t:
                words.append(_cap(t))
                hit = True
            i = j
            continue
        i += 1
    if not hit and not words:
        return name
    out = []
    for w in words:  # numbers: Skirt 0 3 -> Skirt_0_3
        if out and w.isdigit() and out[-1].replace("_", "").isdigit():
            out[-1] += "_" + w
        else:
            out.append(w)
    ded = []
    for w in out:
        if not ded or ded[-1].lower() != w.lower():
            ded.append(w)
    text = "_".join(ded)
    if side:
        text += "_" + side
    text = re.sub(r"[^A-Za-z0-9_]", "", text).strip("_")
    if not text or text.replace("_", "").isdigit():
        return name
    return text


if __name__ == "__main__":
    import sys
    for a in sys.argv[1:]:
        print(a, "->", translate(a))
