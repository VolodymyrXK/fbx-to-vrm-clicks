"""Multilingual vocabulary for recognizing body parts, clothes, hair, accessories and
physics-chain kinds from MMD / game-rip bone names, material names and texture file names.

Pure Python (re + unicodedata).  Deterministic and explainable: every classifier returns
the word that decided the result.

Public API
----------
tokens(name) -> list[str]
material_category(*names) -> (category, sublabel or None, matched_word or None)
chain_kind(name, strict=True) -> (kind or None, matched_word or None)
is_excluded_bone(name) -> bool
excluded_word(name) -> str or None
is_excluded_subtree_root(name) -> bool
is_leaf_marker(name) -> bool
RU_LABELS: dict  (category / sublabel / chain kind -> Russian UI text)
LEGACY_KIND: dict (new chain kind -> kinds understood by converter.KIND_MOD)

Matching model (same for materials and chains)
----------------------------------------------
A name is NFKC-normalized and split into tokens.  Each rule has three word lists:
  * latin  - whole-token matches ("hair" matches "hair_L_01" / "hairBA" but not "chair");
             long pinyin/English compounds are also segmented ("changbinfa" -> chang+binfa,
             "beltBag" -> belt+bag, "trousersDn" -> trousers+dn);
  * phrase - consecutive latin tokens ("eye shadow" matches "Eye_Shadow" / "EyeShadow");
  * cjk    - substrings of the normalized name (JP kanji/kana, CN simplified+traditional, KR).
Rules are tried IN ORDER; the first rule with any hit wins (the longest hit of that rule is
reported).  Conflicts are resolved by placing the specific rule first (see the "overrides"
blocks), e.g. eyeshadow/eyelash/eyebrow -> face before the generic eye rule.
"""
import re
import unicodedata

__all__ = ["tokens", "material_category", "chain_kind", "is_excluded_bone", "excluded_word",
           "is_excluded_subtree_root", "is_leaf_marker", "CATEGORIES", "SUBLABELS", "CHAIN_KINDS_ORDER", "RU_LABELS",
           "LEGACY_KIND", "MATERIAL_CATEGORIES", "CHAIN_KINDS", "EXCLUDE", "TEXTURE_MAP_TOKENS"]


def _nk(s):
    return unicodedata.normalize("NFKC", s or "").strip()


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------
_TOK = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+|[^\W\d_A-Za-z]+")


def tokens(name):
    """NFKC, then split into: CJK/kana/hangul runs, latin words (camelCase aware), digit runs.
    Separators (_ - . space / etc.) are dropped.  Everything lowercased.
    'Bn_l_hairBA_002' -> ['bn','l','hair','ba','002'];  '左后发0_1' -> ['左后发','0','1']"""
    return [p.lower() for p in _TOK.findall(_nk(name))]


# words that never carry meaning but could glue into segmentations
_MODIFIERS = {
    # english position / shape words
    "base", "back", "front", "side", "long", "short", "top", "bottom", "left", "right", "upper",
    "lower", "inner", "outer", "big", "small", "up", "dn", "down", "mid", "middle", "center",
    "line", "shadow", "light", "alpha", "common", "main", "sub", "root", "end", "tip", "sec",
    "jnt", "joint", "bone", "neo", "low", "high", "add", "extra", "ex", "lf", "rt", "md",
    # pinyin modifiers
    "chang", "duan", "qian", "hou", "zuo", "you", "shang", "xia", "zhong", "nei", "wai",
    "da", "xiao", "ce", "bian", "zi", "tou", "wei",
}

# texture-map type tokens stripped from the END of texture file names (not category words)
TEXTURE_MAP_TOKENS = {
    "d", "alb", "albd", "alba", "albm", "albedo", "basecolor", "base", "color", "colour", "col",
    "diffuse", "diff", "dif", "c", "n", "nrm", "nml", "normal", "normals", "nor", "m", "mask",
    "msk", "s", "spec", "specular", "r", "rough", "roughness", "metal", "metallic", "ao", "e",
    "emi", "emission", "emissive", "orm", "mra", "mrao", "id", "lightmap", "ilm", "sdf", "ramp",
    "lut", "tex", "texture", "t", "png", "tga", "dds", "jpg", "jpeg", "bmp", "psd", "tif", "tiff",
    "hdr", "exr", "webp", "spa", "sph", "toon", "atlas", "cm", "nm", "mtl", "mat", "material",
}
_IMG_EXT = re.compile(r"\.(png|tga|dds|jpe?g|bmp|psd|tiff?|hdr|exr|webp|spa|sph)$", re.I)


class _Rule:
    __slots__ = ("cat", "sub", "latin", "phrases", "cjk")

    def __init__(self, cat, sub, words):
        self.cat, self.sub = cat, sub
        self.latin, self.phrases, self.cjk = set(), [], []
        for w in words:
            w = _nk(w)
            if not w:
                continue
            if w.isascii():
                w = w.lower()
                (self.phrases.append(w) if " " in w else self.latin.add(w))
            else:
                self.cjk.append(w)

    def hit(self, cands, phrase_str, text):
        best = None
        for w in self.latin:
            if w in cands and (best is None or len(w) > len(best)):
                best = w
        for p in self.phrases:
            if (" " + p + " ") in phrase_str and (best is None or len(p) > len(best)):
                best = p
        for c in self.cjk:
            if c in text and (best is None or len(c) > len(best)):
                best = c
        return best


def _build(table):
    return [_Rule(cat, sub, words) for cat, sub, words in table]


# ---------------------------------------------------------------------------
# material categories
# ---------------------------------------------------------------------------
CATEGORIES = ["skin", "face", "eyes", "hair", "clothes", "accessory", "other"]

# (category, sublabel, words) in PRIORITY ORDER
MATERIAL_CATEGORIES = [
    # ---- overrides: compounds that contain a word of another category -------------------
    ("other", "fx", ("emissive", "emission", "glow", "reflector", "reflect", "fx", "vfx", "effect",
                     "effects", "particle", "hologram", "発光", "发光", "エフェクト", "特效", "光效")),
    ("face", "eyeshadow", ("eyeshadow", "eye shadow", "eyeshade", "目影", "眼影", "アイシャドウ",
                           "アイシャドー", "아이섀도")),
    ("face", "lashes", ("eyelash", "eyelashes", "lash", "lashes", "eye lash", "matsuge", "jiemao",
                        "睫", "まつげ", "まつ毛", "マツゲ", "속눈썹")),
    ("face", "brows", ("eyebrow", "eyebrows", "brow", "brows", "eye brow", "mayu", "meimao",
                       "眉", "まゆ", "マユ", "눈썹")),
    ("face", None, ("eyelid", "eyelids", "lid", "eye lid", "eyeline", "eyeliner", "眼皮", "二重",
                    "まぶた", "瞼", "睑", "瞼", "アイライン", "눈꺼풀")),
    ("accessory", "earring", ("earring", "earrings", "ear ring", "ear rings", "eardrop", "piercing", "erhuan", "erzhui",
                              "耳飾", "耳饰", "耳环", "耳環", "耳坠", "耳墜", "耳钉", "イヤリング",
                              "ピアス", "귀걸이")),
    ("accessory", "hairpin", ("hairpin", "hairpins", "hairclip", "hairclips", "hairband",
                              "hairtie", "hairornament", "hairacc", "hairdeco", "headband",
                              "scrunchie", "kanzashi", "hair ornament", "hair clip", "hair pin",
                              "hair band", "hair acc", "hair accessory", "hair deco", "fajia",
                              "fashi", "toushi", "faka", "fadai", "髪飾", "髪留", "髪どめ", "ヘアピン",
                              "ヘアバンド", "ヘアゴム", "カチューシャ", "かんざし", "簪", "发夹",
                              "髮夾", "发卡", "髮卡", "发饰", "髮飾", "发带", "髮帶", "发圈", "头饰",
                              "頭飾", "머리핀", "머리띠")),
    ("accessory", "glasses", ("glasses", "eyeglass", "eyeglasses", "spectacles",
                              "goggle", "goggles", "sunglasses", "megane", "眼鏡",
                              "眼镜", "メガネ", "めがね", "ゴーグル", "墨镜", "안경")),
    ("accessory", "mask", ("mask", "visor", "veil", "面具", "仮面", "マスク", "口罩", "面罩",
                           "面紗", "面纱", "가면")),
    ("accessory", "gadget", ("headphone", "headphones", "headset", "earphone", "earphones",
                             "earmuff", "ヘッドフォン", "ヘッドホン", "イヤホン", "耳机", "耳機",
                             "헤드폰")),
    ("accessory", "bag", ("handbag", "手提包", "手包", "挎包")),
    ("accessory", "necklace", ("首飾", "首饰", "首輪", "首环", "项圈", "項圈")),
    ("clothes", "gloves", ("glove", "gloves", "mitten", "mittens", "gauntlet", "gauntlets",
                           "手袋", "手套", "グローブ", "グラブ", "장갑", "shoutao")),
    ("clothes", "socks", ("靴下", "足袋", "socks", "sock")),
    ("clothes", "coat", ("coatskirt", "coat skirt", "coattail", "coat tail", "tailcoat")),
    ("clothes", None, ("袖口", "领口", "領口", "襟口", "衣领", "衣領")),
    ("clothes", "swimsuit", ("swimsuit", "swimwear", "bikini", "swim", "水着", "泳衣", "泳装",
                             "泳裝", "比基尼", "スク水", "수영복")),
    ("clothes", "uniform", ("bodysuit", "leotard", "plugsuit", "catsuit", "ボディスーツ",
                            "レオタード", "连体衣", "連身衣", "紧身衣", "緊身衣")),
    ("clothes", "dress", ("dress", "onepiece", "gown", "nighty", "nightie", "nightgown",
                          "chemise", "kimono", "yukata", "hanfu", "qipao", "cheongsam",
                          "ワンピース", "ドレス", "着物", "浴衣", "连衣裙", "連衣裙", "连身裙",
                          "礼服", "禮服", "旗袍", "汉服", "漢服", "睡裙", "드레스", "원피스",
                          "lianyiqun")),
    ("hair", "ponytail", ("ponytail", "ponytails", "twintail", "twintails", "pigtail",
                          "pigtails", "mawei", "shuangmawei", "马尾", "馬尾", "ポニーテール",
                          "ポニテ", "ツインテ", "ツインテール", "포니테일")),
    ("hair", None, ("hairshadow", "hair shadow", "发影", "髮影", "髪影")),
    ("eyes", "sclera", ("sclera", "eyewhite", "eye white", "eyewhites", "白目", "目白", "眼白",
                        "白眼", "しろめ", "흰자")),
    ("eyes", "eyeshell", ("eyeshell", "eye shell", "eyecover", "cornea", "eyeglossy",
                          "eyegloss", "eye gloss", "眼球外", "角膜")),

    # ---- accessories ------------------------------------------------------------------
    ("accessory", "hat", ("hat", "hats", "cap", "beret", "helmet", "crown", "tiara", "headdress",
                          "headwear", "headgear", "maozi", "帽", "ハット", "キャップ", "ヘルメット",
                          "ティアラ", "王冠", "冠", "头盔", "頭盔", "모자")),
    ("accessory", "ribbon", ("ribbon", "ribbons", "ribon", "bow", "bowknot", "bowtie",
                             "hudiejie", "sidai", "リボン", "丝带", "絲帶", "蝴蝶结", "蝴蝶結",
                             "蝶結", "ちょうちょ", "리본")),
    ("accessory", "necklace", ("necklace", "choker", "pendant", "amulet", "locket", "xianglian",
                               "ネックレス", "チョーカー", "ペンダント", "项链", "項鍊", "項鏈",
                               "吊坠", "吊墜", "颈圈", "목걸이")),
    ("accessory", "bag", ("bag", "bags", "pouch", "backpack", "purse", "satchel", "beltbag",
                          "wallet", "beibao", "鞄", "バッグ", "かばん", "カバン", "ポーチ", "背包",
                          "包", "가방")),
    ("accessory", "belt", ("belt", "belts", "waistband", "buckle", "yaodai", "ベルト", "腰带",
                           "腰帶", "皮带", "皮帶", "벨트")),
    ("accessory", "weapon", ("weapon", "weapons", "wep", "sword", "blade", "gun", "rifle", "pistol",
                             "knife", "dagger", "spear", "lance", "katana", "scythe", "axe",
                             "hammer", "shield", "wand", "staff", "longbow", "crossbow", "arrow",
                             "武器", "剣", "剑", "劍", "刀", "銃", "枪", "槍", "盾", "弓", "杖",
                             "鎌", "镰", "무기")),
    ("accessory", "wings", ("wing", "wings", "chibang", "翼", "羽根", "羽翼", "翅膀", "翅", "ウイング",
                            "ウィング", "날개")),
    ("accessory", "tail", ("tail", "tails", "weiba", "shippo", "尻尾", "しっぽ", "シッポ", "尾巴",
                           "尾", "꼬리")),
    ("accessory", "horns", ("horn", "horns", "jijiao", "ツノ", "つの", "角", "犄角", "뿔")),
    ("accessory", "ears", ("kemomimi", "nekomimi", "animal ear", "animal ears", "cat ear",
                           "cat ears", "bunny ears", "ear", "ears", "erduo", "獣耳", "ケモミミ",
                           "けもみみ", "猫耳", "ネコミミ", "兽耳", "獸耳", "兔耳", "うさ耳", "耳朵",
                           "耳", "귀")),
    ("accessory", "halo", ("halo", "光環", "光环", "ヘイロー", "光輪", "光轮", "헤일로")),
    ("accessory", "gadget", ("gadget", "device", "mecha", "mech", "machine", "drone", "phone",
                             "tool", "engine", "メカ", "機械", "机械", "装置", "端末", "エンジン",
                             "アーム", "道具")),
    ("accessory", None, ("acc", "accs", "acce", "acs", "accessory", "accessories", "ornament",
                         "ornaments", "deco", "decoration", "jewel", "jewelry", "jewellery",
                         "gem", "gems", "badge", "brooch", "charm", "flower", "flowers", "bell",
                         "tag", "tassel", "feather", "peishi", "guajian", "biaoqian",
                         "paizi", "zhuangshi", "アクセ", "アクセサリ", "飾り", "装飾", "装饰",
                         "飾", "饰", "配饰", "配飾", "宝石", "寶石", "徽章", "胸针", "胸針",
                         "ブローチ", "花", "鈴", "铃", "鈴", "挂件", "掛件", "标签", "標籤",
                         "牌子", "名札", "羽毛", "장신구", "액세서리")),

    # ---- face ---------------------------------------------------------------------------
    ("face", "expression", ("expression", "expressions", "emotion", "emotions", "emote",
                            "bq", "biaoqing", "表情", "顔効果", "漫符", "표정")),
    ("face", "blush", ("blush", "blushing", "赤面", "照れ", "頬染", "ほほ染", "腮红", "腮紅",
                       "红晕", "紅暈", "홍조")),
    ("face", "teeth", ("teeth", "tooth", "fang", "fangs", "yachi", "歯", "牙", "齿", "齒", "이빨",
                       "치아")),
    ("face", "tongue", ("tongue", "shetou", "舌", "혀")),
    ("face", "mouth", ("mouth", "mouthline", "lip", "lips", "oral", "kuchi", "zuiba", "口", "くち",
                       "唇", "嘴", "입")),
    ("face", "cheek", ("cheek", "cheeks", "頬", "ほお", "ほっぺ", "脸颊", "臉頰", "볼")),
    ("face", None, ("face", "faces", "head", "facial", "kao", "lian", "mianbu", "nose", "顔", "颜",
                    "顏", "脸", "臉", "面", "かお", "カオ", "頭部", "头部", "鼻", "얼굴")),

    # ---- hair ---------------------------------------------------------------------------
    ("hair", "bangs", ("bang", "bangs", "fringe", "fore hair", "front hair", "fronthair",
                       "forelock", "maegami", "liuhai", "qianfa", "前髪", "前发", "前髮", "刘海",
                       "劉海", "まえがみ", "앞머리")),
    ("hair", "back_hair", ("backhair", "back hair", "hair back", "hairback", "ushirogami", "houfa", "後髪", "後ろ髪",
                           "后发", "後髮", "后髮", "뒷머리")),
    ("hair", "side_hair", ("sidehair", "side hair", "sideburn", "sideburns", "momiage", "binfa",
                           "bingfa", "cefa", "横髪", "もみあげ", "侧发", "側髮", "鬓发", "鬢髮",
                           "鬓", "鬢", "옆머리")),
    ("hair", "braid", ("braid", "braids", "plait", "bianzi", "三つ編み", "みつあみ", "おさげ",
                       "辫", "辮", "땋은")),
    ("hair", "ahoge", ("ahoge", "daimao", "アホ毛", "あほ毛", "呆毛")),
    ("hair", None, ("hair", "hairs", "kami", "toufa", "fa", "dhair", "wig", "髪", "发", "髮",
                    "頭髪", "头发", "頭髮", "かみ", "カミ", "ヘア", "머리카락", "머리", "헤어")),

    # ---- eyes ---------------------------------------------------------------------------
    ("eyes", "highlight", ("highlight", "highlights", "hl", "eyehi", "eye hi", "eye hl", "eyehl", "eyelight",
                           "eyehighlight", "catchlight", "ハイライト", "高光", "目光", "目ハイライト",
                           "하이라이트")),
    ("eyes", "tear", ("tear", "tears", "teardrop", "涙", "泪", "淚", "なみだ", "눈물")),
    ("eyes", "pupil", ("pupil", "pupils", "瞳孔", "동공")),
    ("eyes", "iris", ("iris", "irises", "irissd", "虹彩", "虹膜", "瞳", "ひとみ", "홍채")),
    ("eyes", None, ("eye", "eyes", "eyeball", "eyeballs", "yanjing", "目", "眼", "め", "眼睛", "眼球",
                    "눈")),

    # ---- clothes ------------------------------------------------------------------------
    ("clothes", "underwear", ("underwear", "lingerie", "bra", "panties", "panty", "pantsu",
                              "neiyi", "neiku", "下着", "内衣", "內衣", "内裤", "內褲", "ブラ",
                              "パンツ", "パンティ", "ショーツ", "속옷")),
    ("clothes", "stockings", ("stocking", "stockings", "tights", "pantyhose", "thighhigh",
                              "thighhighs", "kneehigh", "kneesocks", "legwear", "siwa",
                              "ストッキング", "タイツ", "ニーソ", "ニーハイ", "丝袜", "絲襪",
                              "连裤袜", "褲襪", "黑丝", "스타킹")),
    ("clothes", "socks", ("socks", "sock", "wazi", "ソックス", "袜", "襪", "양말")),
    ("clothes", "shoes", ("shoe", "shoes", "boot", "boots", "sneaker", "sneakers", "heel", "heels",
                          "sandal", "sandals", "loafer", "loafers", "footwear", "xiezi", "xue",
                          "靴", "鞋", "ブーツ", "シューズ", "スニーカー", "サンダル", "ヒール", "履",
                          "신발", "구두")),
    ("clothes", "skirt", ("skirt", "skirts", "qun", "qunzi", "qunbai", "スカート", "裙", "치마",
                          "스커트")),
    ("clothes", "shorts", ("shorts", "hotpants", "duanku", "ショートパンツ", "短裤", "短褲",
                           "반바지")),
    ("clothes", "pants", ("pants", "trousers", "trouser", "jeans", "leggings", "slacks", "kuzi",
                          "bottoms", "ズボン", "スラックス", "ジーンズ", "レギンス", "袴", "裤", "褲", "바지")),
    ("clothes", "coat", ("coat", "overcoat", "trenchcoat", "cape", "cloak", "mantle", "poncho",
                         "robe", "haori", "piyi", "pifeng", "dayi", "waitao", "コート", "マント",
                         "ケープ", "ローブ", "羽織", "外套", "大衣", "斗篷", "披风", "披風", "长袍",
                         "코트", "망토")),
    ("clothes", "jacket", ("jacket", "jackets", "blazer", "hoodie", "parka", "hood", "jiake",
                           "上着", "ジャケット", "パーカー", "ブレザー", "フード", "夹克", "夾克",
                           "兜帽", "재킷")),
    ("clothes", "shirt", ("shirt", "shirts", "tshirt", "blouse", "chenshan", "シャツ", "ブラウス",
                          "衬衫", "襯衫", "셔츠", "블라우스")),
    ("clothes", "uniform", ("uniform", "sailor", "suit", "costume", "outfit", "制服", "セーラー",
                            "校服", "军装", "軍裝", "スーツ", "유니폼", "교복")),
    ("clothes", "armor", ("armor", "armour", "pauldron", "greave", "greaves", "breastplate",
                          "plate", "kaijia", "鎧", "盔甲", "铠甲", "鎧甲", "装甲", "裝甲", "護甲",
                          "护甲", "アーマー", "갑옷")),
    ("clothes", "top", ("top", "tops", "vest", "tanktop", "camisole", "sweater", "cardigan",
                        "jumper", "corset", "bustier", "shangyi", "ベスト", "セーター",
                        "カーディガン", "キャミ", "コルセット", "上衣", "背心", "毛衣", "马甲",
                        "상의")),
    ("clothes", None, ("cloth", "clothes", "clothing", "clotch", "cloths", "garment", "wear",
                       "apparel", "apron", "frill", "frills", "lace", "ruffle", "sleeve", "sleeves",
                       "collar", "cuff", "cuffs", "scarf", "muffler", "tie", "necktie", "fabric",
                       "pajama", "pajamas", "pyjama", "fuku", "yifu", "fuzhuang", "xiuzi", "服",
                       "衣", "袖", "襟", "领", "領", "帯", "エプロン", "フリル", "レース", "スカーフ",
                       "マフラー", "ネクタイ", "パジャマ", "睡衣", "围巾", "圍巾", "领带", "領帶",
                       "布", "옷", "의상")),

    # ---- skin ---------------------------------------------------------------------------
    ("skin", None, ("skin", "body", "bodies", "torso", "hand", "hands", "arm", "arms", "leg", "legs",
                    "foot", "feet", "neck", "nail", "nails", "finger", "fingers", "chest", "breast",
                    "belly", "skinmat", "nude", "pifu", "shenti", "肌", "皮肤", "皮膚", "体", "身体",
                    "躯干", "軀幹", "素体", "手", "腕", "脚", "足", "首", "胸", "爪", "指甲", "からだ",
                    "피부", "몸")),

    # ---- leftovers ------------------------------------------------------------------------
    ("other", "outline", ("outline", "edge", "轮廓", "輪郭", "輪廓", "描边", "アウトライン")),
    ("other", "fx", ("shadow", "shade", "影", "光泽", "光澤", "gloss", "glass", "ガラス", "玻璃")),
]
_MAT_RULES = _build(MATERIAL_CATEGORIES)
_WEAK = set(map(id, _MAT_RULES[-2:]))   # leftovers: a later name (texture) may override

SUBLABELS = {}
for _c, _s, _w in MATERIAL_CATEGORIES:
    if _s and _s not in SUBLABELS.setdefault(_c, []):
        SUBLABELS[_c].append(_s)


# ---------------------------------------------------------------------------
# spring-bone chain kinds
# ---------------------------------------------------------------------------
CHAIN_KINDS_ORDER = ["hair", "tail", "skirt", "cloth", "ribbon", "accessory", "breast", "flesh", "ear", "other"]

CHAIN_KINDS = [
    # ---- overrides ------------------------------------------------------------------------
    ("accessory", None, ("earring", "earrings", "ear ring", "eardrop", "piercing", "erhuan", "erzhui", "耳飾",
                         "耳饰", "耳环", "耳環", "耳坠", "耳墜", "イヤリング", "ピアス", "귀걸이")),
    ("accessory", None, ("hairpin", "hairclip", "hairornament", "hairacc", "kanzashi",
                         "hair ornament", "hair clip", "hair pin", "hair acc", "fajia", "fashi",
                         "toushi", "ヘアピン", "ヘアクリップ", "髪飾", "髪留", "かんざし", "簪", "发夹", "髮夾", "发饰", "髮飾",
                         "头饰", "頭飾", "胸飾", "アクセ", "accessory", "accessories", "acc", "acce",
                         "accs")),
    ("hair", None, ("ponytail", "ponytails", "twintail", "twintails", "pigtail", "pigtails",
                    "mawei", "shuangmawei", "hair tail", "马尾", "馬尾", "ポニーテール", "ポニテ", "ツインテ")),
    ("cloth", None, ("coatskirt", "coat skirt", "coattail", "coattails", "coat tail", "tailcoat",
                     "衣摆", "衣擺", "裾")),
    ("ribbon", None, ("ribbon", "ribbons", "ribon", "bowknot", "bow", "bowtie", "hudiejie", "hudie",
                      "piaodai", "sidai", "hairband", "hairtie", "hair band", "hair ribbon", "xiudai",
                      "shoudai", "xiedai", "lingjie", "fadai", "shoelace", "lace", "laces", "string",
                      "strings", "cord", "tassel", "liusu", "sash", "streamer",
                      "リボン", "丝带", "絲帶", "蝴蝶结", "蝴蝶結", "飘带", "飄帶", "发带", "髮帶",
                      "ヘアバンド", "紐", "ひも", "流苏", "流蘇", "タッセル", "鞋带", "鞋帶", "绳", "繩")),
    ("accessory", None, ("weipd", "pendant", "guajian", "diaozhui", "挂件", "掛件", "吊坠", "吊墜",
                         "ペンダント", "charm", "keychain", "bell", "lingdang", "鈴", "铃", "chain", "鎖", "链",
                         "鏈", "チェーン", "xianglian", "necklace", "ネックレス", "项链", "項鍊")),
    # ---- body ---------------------------------------------------------------------------
    ("breast", None, ("breast", "breasts", "bust", "boob", "boobs", "oppai", "xiong", "xiongbu", "rufang",
                      "胸", "乳", "おっぱい", "バスト", "가슴")),
    ("tail", None, ("tail", "tails", "weiba", "shippo", "尻尾", "しっぽ", "シッポ", "尾巴", "尾",
                    "꼬리")),
    ("ear", None, ("ear", "ears", "kemomimi", "nekomimi", "mimi", "erduo", "shouer", "耳朵", "耳", "ケモミミ",
                   "ネコミミ", "けもみみ", "猫耳", "兽耳", "獸耳", "うさ耳", "兔耳", "귀")),
    # soft body: thigh / calf / belly / butt jiggle bones that hang off the leg and hip bones
    ("flesh", None, ("thighflesh", "calfflesh", "butt", "buttock", "glute", "belly", "tummy", "jiggle",
                     "flesh", "pigu", "tunbu", "dupi", "太もも", "ふともも", "ふくらはぎ", "お尻", "尻肉",
                     "お腹", "腹肉", "肉", "臀", "屁股", "肚子", "大腿肉")),
    ("cloth", None, ("スカーフ", "scarf")),  # before "スカ" (skirt)
    ("skirt", None, ("skirt", "skirts", "qun", "qunzi", "qunbai", "スカート", "スカ", "前垂れ", "前垂", "腰布",
                     "loincloth", "裙", "치마")),
    # ---- clothes ------------------------------------------------------------------------
    ("cloth", None, ("cloth", "clothes", "clothing", "clotch", "coat", "cape", "cloak", "mantle",
                     "poncho", "robe", "jacket", "dress", "trousers", "trouser", "pants", "sleeve",
                     "sleeves", "cuff", "cuffs", "hood", "scarf", "muffler", "tie", "collar",
                     "necktie", "rope", "string", "lace", "laces", "strap", "straps", "band",
                     "sash", "belt", "apron", "frill", "hem", "flap", "obi", "shawl", "stole",
                     "piyi", "pifeng", "nightgown", "gown", "shirt", "shuiyi", "shuiqun", "yibai", "yifu",
                     "waitao", "chenshan", "kimono", "yukata", "hakama", "toga", "shuiyi",
                     "xiuzi", "xiukou", "xiuko", "lingdai", "lingzi",
                     "yaodai", "weijin", "hat", "doupeng",
                     "袖", "袖子", "领", "領", "襟", "紐", "ひも", "マント", "フード", "帯", "ネクタイ",
                     "スカーフ", "マフラー", "コート", "上着", "ケープ", "ショール", "ストール", "服", "衣", "披风", "披風", "斗篷", "领带",
                     "睡衣", "睡裙", "衬衫", "襯衫", "シャツ", "ネグリジェ", "パジャマ", "着物", "浴衣", "和服", "外套",
                     "領帶", "绳", "繩", "带", "帶", "帽", "围巾", "圍巾", "外套", "袴", "옷", "망토")),
    # ---- dangling accessories -------------------------------------------------------------
    ("accessory", None, ("bag", "pouch", "beltbag", "backpack", "purse", "gadget", "pendant",
                         "charm", "necklace", "bell", "tag", "badge", "tassel", "keychain",
                         "ornament", "ornaments", "deco", "jewel", "wing", "wings",
                         "feather", "flower", "biaoqian", "paizi", "guajian", "liusu", "diaozhui",
                         "挂件", "掛件", "吊坠", "吊墜", "鈴", "铃", "ペンダント", "ネックレス",
                         "项链", "項鍊", "飾り", "装飾", "饰", "飾", "流苏", "流蘇", "翼", "羽", "翅膀",
                         "鞄", "包", "バッグ", "タッセル")),
    # ---- hair (last: 'hair' prefix is used for everything in some rigs) ------------------
    ("hair", None, ("hair", "hairs", "bang", "bangs", "fringe", "strand", "strands", "braid",
                    "braids", "ahoge", "sidehair", "backhair", "forelock", "sideburn", "liuhai",
                    "binfa", "bingfa", "jianfa", "toufa", "houfa", "qianfa", "cefa", "bianzi",
                    "daimao", "fa", "kami", "maegami", "momiage", "ushirogami", "髪", "发", "髮",
                    "前髪", "後髪", "もみあげ", "アホ毛", "おさげ", "辫", "辮", "刘海", "劉海",
                    "鬓", "鬢", "呆毛", "かみ", "ヘア", "머리")),
]
_CHAIN_RULES = _build(CHAIN_KINDS)

# generic "this is a physics bone" markers (kind unknown) — only used with strict=False
_PHYS_MARKERS = _Rule("other", None, (
    "physics", "phys", "dynamicbone", "dynamic", "spring", "springbone", "jiggle", "swing", "sim",
    "db", "sb", "dyn", "magica", "揺れ", "ゆれ", "物理", "摇摆", "搖擺"))


# ---------------------------------------------------------------------------
# exclusion
# ---------------------------------------------------------------------------
# hard: always excluded from auto physics (helpers, rig controls, attachment sockets)
EXCLUDE = {
    "twist", "ik", "iktarget", "ikt", "pole", "weapon", "weapons", "wep", "prop", "props",
    "socket", "attach", "attachment", "attachpoint", "fx", "vfx", "eff", "effect", "camera", "cam",
    "light", "armature", "offset", "roll", "helper", "dummy", "adj", "adjust", "facialdef",
    "facial def", "face rig", "facerig", "facial rig", "muscle", "muscleoffset", "corrective",
    "ctrl", "control", "constraint", "locator",
    "ダミー", "捩", "捻", "調整", "调整", "補助", "补助", "補正", "补", "武器", "カメラ", "エフェクト",
    "特效", "挂点", "掛點", "IKターゲット", "ＩＫ",
}
# soft: excluded only when the name has no chain-kind word ('Root' yes, 'hair_root' no)
EXCLUDE_SOFT = {"root", "null", "point", "target", "center", "armature", "global", "master",
                "move", "all", "全ての親", "操作中心", "センター", "グルーブ"}
_EXCL_RULE = _Rule("x", None, EXCLUDE)
_SOFT_RULE = _Rule("x", None, EXCLUDE_SOFT)
LEAF_MARKERS = {"end", "nub", "tip", "ending", "先", "先端", "末端"}
# rig parents whose WHOLE subtree is excluded (RE Engine FacialDef_*, face rigs)
EXCLUDE_SUBTREE = {"facialdef", "facial def", "face rig", "facerig", "facial rig", "facialrig",
                   "face_rig", "facialroot", "facial root"}
_SUBTREE_RULE = _Rule("x", None, EXCLUDE_SUBTREE)

RU_LABELS = {
    # categories
    "skin": "Тело (кожа)", "face": "Лицо", "eyes": "Глаза", "hair": "Волосы", "clothes": "Одежда",
    "accessory": "Аксессуары", "other": "Прочее",
    # clothes sublabels
    "top": "Верх", "jacket": "Куртка", "coat": "Пальто/плащ", "shirt": "Рубашка/блузка",
    "dress": "Платье", "skirt": "Юбка", "pants": "Брюки", "shorts": "Шорты", "socks": "Носки",
    "stockings": "Чулки/колготки", "shoes": "Обувь", "gloves": "Перчатки", "underwear": "Бельё",
    "swimsuit": "Купальник", "uniform": "Костюм/форма", "armor": "Броня",
    # accessory sublabels
    "hat": "Головной убор", "hairpin": "Заколка/украшение волос", "ribbon": "Лента/бант",
    "glasses": "Очки", "earring": "Серьги", "necklace": "Ожерелье/кулон", "bag": "Сумка",
    "belt": "Ремень/пояс", "weapon": "Оружие", "wings": "Крылья", "tail": "Хвост", "horns": "Рога",
    "ears": "Уши (звериные)", "halo": "Нимб", "mask": "Маска", "gadget": "Гаджет/механизм",
    # face sublabels
    "mouth": "Рот", "teeth": "Зубы", "tongue": "Язык", "brows": "Брови", "lashes": "Ресницы",
    "eyeshadow": "Тени век", "cheek": "Щёки", "blush": "Румянец", "expression": "Эмоции (наклейки)",
    # eyes sublabels
    "iris": "Радужка", "pupil": "Зрачок", "sclera": "Белки глаз", "highlight": "Блик",
    "eyeshell": "Оболочка глаза", "tear": "Слёзы",
    # hair sublabels
    "bangs": "Чёлка", "back_hair": "Волосы сзади", "side_hair": "Боковые пряди",
    "ponytail": "Хвост (причёска)", "braid": "Коса", "ahoge": "Хохолок",
    # other sublabels
    "fx": "Эффекты/свечение", "outline": "Контур",
    # chain kinds (hair/skirt/tail/accessory/other share the keys above)
    "breast": "Грудь", "ear": "Уши", "cloth": "Одежда (ткань)", "flesh": "Мягкость тела",
}
# map new chain kinds onto converter.KIND_MOD keys
LEGACY_KIND = {"hair": "hair", "skirt": "skirt", "breast": "breast", "tail": "tail", "ear": "ear",
               "cloth": "cloth", "ribbon": "ribbon", "accessory": "accessory", "flesh": "flesh", "other": "other"}


# ---------------------------------------------------------------------------
# matching engine
# ---------------------------------------------------------------------------
_LEX = set(_MODIFIERS)
for _r in _MAT_RULES + _CHAIN_RULES:
    _LEX |= {w for w in _r.latin if len(w) >= 2 and w.isalpha()}
_LEX -= {"hl", "bq", "sb", "db"}           # too short to be safe inside compounds
_MAXW = max(len(w) for w in _LEX)


def _segment(tok):
    """Split a latin token into lexicon words (fewest pieces), or None."""
    n = len(tok)
    best = [None] * (n + 1)
    best[0] = []
    for i in range(n):
        if best[i] is None:
            continue
        for j in range(min(n, i + _MAXW), i + 1, -1):
            w = tok[i:j]
            if w in _LEX and (best[j] is None or len(best[i]) + 1 < len(best[j])):
                best[j] = best[i] + [w]
    return best[n]


def _prepare(name, strip_texture=False):
    s = _nk(name)
    if strip_texture:
        s = _IMG_EXT.sub("", s.replace("\\", "/").rsplit("/", 1)[-1])
    toks = [p.lower() for p in _TOK.findall(s)]
    if strip_texture:
        while toks and (toks[-1] in TEXTURE_MAP_TOKENS or toks[-1].isdigit()):
            toks.pop()
        while toks and toks[0] in ("t", "tex"):
            toks.pop(0)
    lat = [t for t in toks if t.isascii() and t.isalpha()]
    cands = set(lat)
    for t in lat:
        if len(t) >= 5 and t not in _LEX:
            seg = _segment(t)
            if seg and len(seg) > 1:
                cands.update(seg)
            elif t[:-1] in _LEX:            # variant letter suffix: guajianb, hairbA
                cands.add(t[:-1])
            elif t[-1] in "abcdefgh":
                seg = _segment(t[:-1])
                if seg:
                    cands.update(seg)
    text = "".join(t for t in toks if not t.isascii()) if strip_texture else s.lower()
    return cands, " " + " ".join(lat) + " ", text


def _first(rules, prep):
    for r in rules:
        w = r.hit(*prep)
        if w:
            return r, w
    return None, None


def _looks_like_texture(name):
    return bool(_IMG_EXT.search(name or "")) or "/" in (name or "") or "\\" in (name or "")


def material_category(*names):
    """Classify a material from its names, in decreasing order of trust: e.g.
    material_category(mat.name, mat.name_en, texture_path).  Texture paths (with an image
    extension or a directory) have the directory, extension and trailing map-type tokens
    (_D, _ALBD, _BaseColor, _N, _M ...) stripped first.
    The first name that matches anything decides the category; a later name may only refine
    the sublabel within the same category (material '目' + texture 'iris_01_D' -> eyes/iris).
    Returns (category, sublabel or None, matched_word or None)."""
    found, weak = None, False
    for n in names:
        if not n:
            continue
        r, w = _first(_MAT_RULES, _prepare(n, _looks_like_texture(n)))
        if not r:
            continue
        if found is None or (weak and id(r) not in _WEAK):
            weak = id(r) in _WEAK
            found = (r.cat, r.sub, w)
            if r.sub and not weak:
                break
        elif r.cat == found[0] and r.sub:
            found = (r.cat, r.sub, found[2] + "+" + w)
            break
    return found or ("other", None, None)


def excluded_word(name):
    """The word that excludes this bone from auto physics, or None."""
    prep = _prepare(name)
    w = _EXCL_RULE.hit(*prep)
    if w:
        return w
    w = _SOFT_RULE.hit(*prep)
    if w and not _first(_CHAIN_RULES, prep)[0]:
        return w
    return None


def is_excluded_bone(name):
    """True for helper / rig / socket bones that must never get spring physics (twist, IK,
    weapon/prop sockets, FacialDef / face-rig parents, MMD ダミー/捩/調整/補助 ...).
    Note: for FacialDef-style rig PARENTS the whole subtree should be skipped by the caller."""
    return excluded_word(name) is not None


def is_excluded_subtree_root(name):
    """True for rig parents (FacialDef_Face, FaceRig ...) whose descendants must all be skipped."""
    return _SUBTREE_RULE.hit(*_prepare(name)) is not None


def is_leaf_marker(name):
    """True for chain-end markers (…_end, …Nub, …_tip, MMD …先).  These are not excluded:
    they belong to the chain but carry no weights; use them only as chain terminators."""
    s = _nk(name)
    if s.endswith(("先", "先端", "末端")):
        return True
    toks = [t for t in tokens(s) if not t.isdigit()]
    return bool(toks) and toks[-1] in LEAF_MARKERS


def chain_kind(name, strict=True, ignore_excluded=False):
    """Kind of spring chain a bone belongs to: hair, skirt, breast, tail, ear, cloth, accessory.
    strict=True  - explicit vocabulary only (whole tokens, compounds, CJK substrings).
    strict=False - additionally latin substrings (len>=4) inside unknown tokens and generic
                   physics markers (DynamicBone, SB_/DB_, Physics, spring, 揺れ) -> 'other'.
    Excluded bones return (None, '!<word>').  Returns (kind or None, matched_word or None)."""
    ex = None if ignore_excluded else excluded_word(name)
    if ex:
        return None, "!" + ex
    prep = _prepare(name)
    r, w = _first(_CHAIN_RULES, prep)
    if r:
        return r.cat, w
    if strict:
        return None, None
    cands, _, _ = prep
    for r in _CHAIN_RULES:
        for word in sorted(r.latin, key=len, reverse=True):
            if len(word) >= 4 and any(word in t for t in cands if len(t) > len(word)):
                return r.cat, "~" + word
    w = _PHYS_MARKERS.hit(*prep)
    if w:
        return "other", w
    return None, None


if __name__ == "__main__":
    import sys
    for a in sys.argv[1:]:
        print(a, "| mat:", material_category(a), "| chain:", chain_kind(a, strict=False),
              "| excl:", is_excluded_bone(a), "| leaf:", is_leaf_marker(a))
