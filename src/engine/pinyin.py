"""Pinyin words used in bone / material names of Chinese-made models (Bn_l_changbinfa01, BN_weipd03 ...).

LEX maps a pinyin (or short game-rig) word to (russian, english).  segment() splits a glued token
such as "changbinfa" into known words ("chang", "binfa") with the fewest pieces.
"""

LEX = {
    # hair
    "toufa": ("Волосы", "Hair"), "fa": ("Волосы", "Hair"), "liuhai": ("Чёлка", "Bangs"),
    "qianfa": ("Чёлка", "Bangs"), "houfa": ("Волосы сзади", "BackHair"), "cefa": ("Боковые волосы", "SideHair"),
    "binfa": ("Височная прядь", "Sideburn"), "bingfa": ("Височная прядь", "Sideburn"),
    "jianfa": ("Волосы до плеч", "ShoulderHair"), "changfa": ("Длинные волосы", "LongHair"),
    "duanfa": ("Короткие волосы", "ShortHair"), "daimao": ("Хохолок", "Ahoge"), "maomao": ("Хохолок", "Ahoge"),
    "bianzi": ("Коса", "Braid"), "mawei": ("Хвост (причёска)", "Ponytail"), "shuangmawei": ("Хвостики", "Twintails"),
    "fashao": ("Кончик волос", "HairTip"), "fajia": ("Заколка", "HairClip"), "fashi": ("Украшение волос", "HairOrnament"),
    "fadai": ("Лента для волос", "HairBand"),
    # body
    "weiba": ("Хвост", "Tail"), "erduo": ("Ухо", "Ear"), "shouer": ("Звериные уши", "AnimalEars"),
    "xiong": ("Грудь", "Breast"), "xiongbu": ("Грудь", "Breast"), "rufang": ("Грудь", "Breast"),
    "pigu": ("Ягодица", "Butt"), "tunbu": ("Ягодица", "Butt"), "datui": ("Бедро", "Thigh"),
    "xiaotui": ("Голень", "Calf"), "tui": ("Нога", "Leg"), "jiao": ("Рог", "Horn"), "shou": ("Кисть", "Hand"),
    "shoubi": ("Рука", "Arm"), "jianbang": ("Плечо", "Shoulder"), "tou": ("Голова", "Head"),
    "bozi": ("Шея", "Neck"), "yao": ("Поясница", "Waist"), "dupi": ("Живот", "Belly"),
    "chibang": ("Крыло", "Wing"), "yumao": ("Перо", "Feather"), "jiaozi": ("Рог", "Horn"),
    # clothes
    "yifu": ("Одежда", "Clothes"), "qunzi": ("Юбка", "Skirt"), "qun": ("Юбка", "Skirt"),
    "qunbai": ("Подол юбки", "SkirtHem"), "yibai": ("Подол", "Hem"), "xiuzi": ("Рукав", "Sleeve"),
    "xiu": ("Рукав", "Sleeve"), "xiukou": ("Манжета", "Cuff"), "xiuko": ("Манжета", "Cuff"), "paizi": ("Табличка", "Tag"), "biaoqian": ("Бирка", "Tag"), "xiudai": ("Лента рукава", "SleeveRibbon"),
    "shoudai": ("Лента на руке", "WristRibbon"), "xiedai": ("Шнурок обуви", "Shoelace"),
    "xie": ("Обувь", "Shoe"), "xuezi": ("Сапог", "Boot"), "wazi": ("Носок", "Sock"),
    "lingzi": ("Воротник", "Collar"), "ling": ("Воротник", "Collar"), "lingdai": ("Галстук", "Tie"),
    "lingjie": ("Бант на воротнике", "CollarBow"), "yaodai": ("Пояс", "Belt"), "weijin": ("Шарф", "Scarf"),
    "pifeng": ("Плащ", "Cape"), "doupeng": ("Плащ", "Cloak"), "piyi": ("Накидка", "Cape"),
    "waitao": ("Куртка", "Jacket"), "maozi": ("Шляпа", "Hat"), "toushi": ("Головной убор", "Headwear"),
    "toujin": ("Платок", "Headscarf"), "fajin": ("Платок", "Headscarf"), "weiqun": ("Фартук", "Apron"),
    "shuiyi": ("Ночная рубашка", "Nightgown"), "shuiqun": ("Ночная рубашка", "Nightgown"),
    "kuzi": ("Брюки", "Pants"), "duanku": ("Шорты", "Shorts"), "baisi": ("Чулки", "Stockings"),
    "heisi": ("Чулки", "Stockings"), "siwa": ("Чулки", "Stockings"), "huabian": ("Оборка", "Frill"),
    "leisi": ("Кружево", "Lace"), "hudiejie": ("Бант", "Bow"), "hudie": ("Бант", "Bow"),
    "sidai": ("Лента", "Ribbon"), "piaodai": ("Лента", "Ribbon"), "dai": ("Лента", "Band"),
    "shengzi": ("Шнурок", "Cord"),
    # accessories
    "guajian": ("Подвеска", "Pendant"), "diaozhui": ("Кулон", "Pendant"), "pd": ("Подвеска", "Pendant"),
    "peishi": ("Украшение", "Ornament"), "shipin": ("Украшение", "Ornament"), "liusu": ("Кисточка", "Tassel"),
    "lingdang": ("Колокольчик", "Bell"), "xianglian": ("Ожерелье", "Necklace"), "erhuan": ("Серьга", "Earring"),
    "erzhui": ("Серьга", "Earring"), "shoulian": ("Браслет", "Bracelet"), "shouzhuo": ("Браслет", "Bracelet"),
    "bao": ("Сумка", "Bag"), "beibao": ("Рюкзак", "Backpack"), "wuqi": ("Оружие", "Weapon"),
    # positions / modifiers
    "chang": ("длинн.", "Long"), "duan": ("коротк.", "Short"), "qian": ("перед", "Front"),
    "hou": ("зад", "Back"), "ce": ("бок", "Side"), "zuo": ("Л", "L"), "you": ("П", "R"),
    "zhong": ("сред.", "Mid"), "gen": ("корень", "Root"), "shang": ("верх", "Upper"), "xia": ("низ", "Lower"),
    "nei": ("внутр.", "Inner"), "wai": ("внешн.", "Outer"), "jian": ("плечо", "Shoulder"),
    "wei": ("Хвост", "Tail"),
}
SIDE = {"zuo": "L", "you": "R"}


def segment(tok, extra=()):
    """Split a lowercase latin token into known words (LEX plus extra), fewest pieces; a single trailing
    letter (variant A/B/C) is allowed.  Returns a list of words or None."""
    words = set(LEX) | set(extra)
    n = len(tok)
    best = [None] * (n + 1)
    best[0] = []
    for i in range(n):
        if best[i] is None:
            continue
        for j in range(n, i, -1):
            w = tok[i:j]
            ok = (w in words and (len(w) >= 2 or w in LEX)) or (j == n and len(w) == 1 and i > 0 and w in "abcd")
            if ok and (best[j] is None or len(best[i]) + 1 < len(best[j])):
                best[j] = best[i] + [w]
    return best[n]
