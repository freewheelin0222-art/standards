#!/usr/bin/env python3
"""
songs.json（ireal_to_json.py の出力）を分析し、Web公開用のデータを作る。

使い方:
    pip install numpy
    python analyze.py songs.json

出力:
    analyzed.json … {"patterns": 進行パターンの定義, "songs": 曲ごとの度数表記・機能注釈・特徴量}
                    各コード: c=コード, d=度数, f=機能, b=拍数
                    マイナーキーの曲のみ: d_rel=平行長調基準の度数, f_rel=平行長調基準の機能
    similar.json  … 曲ごとの類似曲トップ10（共通パターン付き）

分析はすべて決定的（毎回同じ結果）。特徴量はテーマ本体（type == "head"）だけから計算する。
"""
import json
import re
import sys
from collections import Counter, defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# コードの読み取り
# ---------------------------------------------------------------------------
PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
CHORD_RE = re.compile(r"^([A-G][b#]?)(.*?)(?:/([A-G][b#]?))?(?::(\d+))?$")

NUMERAL = {0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III", 5: "IV",
           6: "#IV", 7: "V", 8: "bVI", 9: "VI", 10: "bVII", 11: "VII"}
FAM_SUFFIX = {"maj": "maj", "min": "m", "dom": "7", "hdim": "m7b5",
              "dim": "dim", "aug": "aug", "sus": "7sus", "other": "?", "nc": "N.C."}

DIATONIC = {
    "major": {0, 2, 4, 5, 7, 9, 11},
    "minor": {0, 2, 3, 5, 7, 8, 9, 10, 11},  # 自然・和声・旋律短音階の和集合
}


def pitch(name):
    v = PC[name[0]]
    for a in name[1:]:
        v += -1 if a == "b" else 1
    return v % 12


def family(q):
    """コード品質を大分類にまとめる"""
    if q.startswith("maj"): return "maj"
    if q.startswith("m7b5"): return "hdim"
    if q.startswith("m"): return "min"
    if q.startswith("dim"): return "dim"
    if q.startswith("aug"): return "aug"
    if q.startswith("7sus") or q.startswith("sus"): return "sus"
    if q[:1] in ("7", "9") or q.startswith("13"): return "dom"
    if q in ("", "5") or q.startswith(("6", "add", "(")): return "maj"
    return "other"


def parse_chord(s, bar_beats, n_in_bar):
    if s.startswith("N.C."):
        beats = int(s.split(":")[1]) if ":" in s else bar_beats / n_in_bar
        return {"c": "N.C.", "root": None, "fam": "nc", "q": "", "bass": None, "beats": beats}
    m = CHORD_RE.match(s)
    root, q, bass, b = m.groups()
    return {"c": s.split(":")[0], "root": pitch(root), "fam": family(q), "q": q,
            "bass": pitch(bass) if bass else None,
            "beats": int(b) if b else bar_beats / n_in_bar}


def degree(ch, tonic):
    """曲のキー基準の度数表記（例：IIm7, V7(b13), bVImaj7/I）"""
    if ch["root"] is None:
        return "N.C."
    rel = (ch["root"] - tonic) % 12
    num = NUMERAL[rel]
    if rel == 6 and ch["fam"] == "dom":
        num = "bV"   # 裏コードとしての表記
    d = num + ch["q"]
    if ch["bass"] is not None:
        d += "/" + NUMERAL[(ch["bass"] - tonic) % 12]
    return d


KEY_NAME = {0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F", 6: "Gb", 7: "G",
            8: "Ab", 9: "A", 10: "Bb", 11: "B"}


def parse_key(k):
    minor = k.endswith("m")
    return pitch(k[:-1] if minor else k), ("minor" if minor else "major")


# ---------------------------------------------------------------------------
# パターン検出（ii-V-I など）
# ---------------------------------------------------------------------------
DOM_LIKE = ("dom", "sus")

# 進行パターンの分類（サイトの「進行で探す」に表示する順）
PATTERNS = [
    {"id": "ii_V_I", "name": "ii–V–I（メジャー・マイナー）",
     "desc": "IIm7–V7–I と IIm7♭5–V7–Im を合わせた数"},
    {"id": "ii_V_I_major", "name": "ii–V–I（メジャー）", "desc": "IIm7–V7–Imaj"},
    {"id": "ii_V_i_minor", "name": "ii–V–i（マイナー）", "desc": "IIm7♭5–V7–Im"},
    {"id": "backdoor", "name": "IVm–♭VII7–I（バックドア）", "desc": "IVm7–♭VII7 から、全音上の I へ"},
    {"id": "iv_minor", "name": "IV–IVm（サブドミナントマイナー）", "desc": "IV から同じルートの IVm へ"},
    {"id": "turnaround", "name": "循環（I–VI–II–V / III–VI–II–V）", "desc": "4度進行でIIm–V7へ向かうターンアラウンド"},
    {"id": "tritone", "name": "裏コード", "desc": "半音下へ解決する属七（subV→I）"},
    {"id": "secondary", "name": "セカンダリードミナント", "desc": "トニック以外へ解決する属七"},
    {"id": "dom_chain", "name": "ドミナントの連鎖", "desc": "属七が4度上の属七へ続く（III7–VI7–II7–V7 など）"},
    {"id": "blues", "name": "ブルース（12小節）", "desc": "12小節ブルースの形式"},
]


def merge_events(items):
    """同じルート・同じ大分類が続くコードを1つのイベントにまとめる"""
    events = []
    for idx, it in enumerate(items):
        key = (it["root"], it["fam"])
        if events and events[-1]["key"] == key:
            events[-1]["items"].append(idx)
            events[-1]["beats"] += it["beats"]
        else:
            events.append({"key": key, "root": it["root"], "fam": it["fam"],
                           "items": [idx], "beats": it["beats"]})
    return events


def detect(events, tonic):
    """パターンを数え、各イベントに機能注釈（ii/bIII など）を付ける"""
    cnt = Counter()
    fn = {}
    targets = []
    n = len(events)

    def up4(a, b): return (b["root"] - a["root"]) % 12 == 5
    def down_half(a, b): return (b["root"] - a["root"]) % 12 == 11

    def target_label(t):
        rel = (t["root"] - tonic) % 12
        return None if rel == 0 else NUMERAL[rel]

    def mark(i, role, t):
        tl = target_label(t)
        fn.setdefault(i, role if tl is None else f"{role}/{tl}")

    for i in range(n):
        a = events[i]
        b = events[i + 1] if i + 1 < n else None
        c = events[i + 2] if i + 2 < n else None
        if a["root"] is None or b is None or b["root"] is None:
            continue
        # ii-V（4度上行）
        if a["fam"] in ("min", "hdim") and b["fam"] in DOM_LIKE and up4(a, b):
            cnt["ii_V"] += 1
            if c is not None and c["root"] is not None and up4(b, c) and c["fam"] in ("maj", "min", "dom"):
                minor = c["fam"] == "min"
                cnt["ii_V_i_minor" if minor else "ii_V_I_major"] += 1
                one = "i" if minor else "I"
                mark(i, "ii", c); mark(i + 1, "V", c); mark(i + 2, one, c)
                targets.append(c)
            elif c is not None and c["root"] is not None and (c["root"] - b["root"]) % 12 == 2 \
                    and c["fam"] in ("maj", "dom"):
                cnt["backdoor"] += 1   # IVm7 - bVII7 - I
                mark(i, "IVm", c); mark(i + 1, "bVII7", c); mark(i + 2, "I", c)
            else:
                cnt["ii_V_unresolved"] += 1
                implied = {"root": (b["root"] + 5) % 12}  # 本来解決するはずの音
                mark(i, "ii", implied); mark(i + 1, "V", implied)
        d = events[i + 3] if i + 3 < n else None
        # 循環：I(またはIIIm)–VI–IIm–V7
        if c is not None and d is not None and None not in (c["root"], d["root"]) \
                and b["fam"] in ("dom", "min") and up4(b, c) and c["fam"] in ("min", "hdim") \
                and up4(c, d) and d["fam"] in DOM_LIKE \
                and ((a["fam"] in ("maj", "dom") and (b["root"] - a["root"]) % 12 == 9)
                     or (a["fam"] == "min" and up4(a, b))):
            cnt["turnaround"] += 1
        # ドミナントの連鎖：属七 → 4度上の属七
        if a["fam"] == "dom" and b["fam"] == "dom" and up4(a, b):
            cnt["dom_chain"] += 1
        # IV → IVm（サブドミナントマイナー）
        if (a["root"] - tonic) % 12 == 5 and a["fam"] in ("maj", "dom") \
                and b["root"] == a["root"] and b["fam"] == "min":
            cnt["iv_minor"] += 1
        # ii-裏V-I（ii から半音下の属七 → さらに半音下へ解決）
        if a["fam"] in ("min", "hdim") and b["fam"] == "dom" and down_half(a, b) \
                and c is not None and c["root"] is not None and down_half(b, c) \
                and c["fam"] in ("maj", "min", "dom"):
            cnt["ii_subV_I"] += 1
            mark(i, "ii", c); mark(i + 1, "subV", c); mark(i + 2, "i" if c["fam"] == "min" else "I", c)
            targets.append(c)
        # 属七の解決
        if a["fam"] == "dom" and b["fam"] in ("maj", "min", "dom"):
            if up4(a, b):
                cnt["V_I"] += 1
                if (b["root"] - tonic) % 12 != 0:
                    cnt["secondary_dominant"] += 1
                mark(i, "V", b)
            elif down_half(a, b):
                cnt["tritone_sub"] += 1
                mark(i, "subV", b)

    tonicized = sorted({(NUMERAL[(t["root"] - tonic) % 12], "minor" if t["fam"] == "min" else "major")
                        for t in targets if (t["root"] - tonic) % 12 != 0})
    return cnt, fn, tonicized


# ---------------------------------------------------------------------------
# 1曲の分析
# ---------------------------------------------------------------------------
def flatten(sections, meter, keep):
    items = []
    beats_per_bar = int(meter.split("/")[0])
    for si, sec in enumerate(sections):
        if not keep(sec):
            continue
        bpb = int(sec.get("meter", meter).split("/")[0]) if "/" in sec.get("meter", meter) else beats_per_bar
        for bi, bar in enumerate(sec["bars"]):
            for ci, s in enumerate(bar):
                ch = parse_chord(s, bpb, len(bar))
                ch.update(sec=si, bar=bi, pos=ci)
                items.append(ch)
    return items


def analyze_song(song):
    tonic, mode = parse_key(song["key"])
    meter = song["meter"]
    is_head = lambda s: s["type"] == "head"

    # --- 全セクションに度数と機能注釈を付ける（表示用） ---
    all_items = flatten(song["sections"], meter, lambda s: True)
    ev_all = merge_events(all_items)
    _, fn_all, _ = detect(ev_all, tonic)
    item_fn = {}
    for ei, fnl in fn_all.items():
        for idx in ev_all[ei]["items"]:
            item_fn[idx] = fnl
    # マイナーキーの曲は、平行長調を基準にした度数と機能も付ける（例：Gm → Bb を I とする）
    rel_tonic = (tonic + 3) % 12 if mode == "minor" else None
    item_fn_rel = {}
    if rel_tonic is not None:
        _, fn_rel, _ = detect(ev_all, rel_tonic)
        for ei, fnl in fn_rel.items():
            for idx in ev_all[ei]["items"]:
                item_fn_rel[idx] = fnl
    out_sections = [dict(label=s["label"], type=s["type"], bar_count=s["bar_count"], bars=[])
                    for s in song["sections"]]
    for sec_out, sec in zip(out_sections, song["sections"]):
        sec_out["bars"] = [[] for _ in sec["bars"]]
    for idx, it in enumerate(all_items):
        cell = {"c": it["c"], "d": degree(it, tonic), "b": round(it["beats"], 2)}
        if idx in item_fn:
            cell["f"] = item_fn[idx]
        if rel_tonic is not None:
            cell["d_rel"] = degree(it, rel_tonic)
            if idx in item_fn_rel:
                cell["f_rel"] = item_fn_rel[idx]
        out_sections[it["sec"]]["bars"][it["bar"]].append(cell)

    # --- 特徴量（テーマ本体のみ） ---
    items = flatten(song["sections"], meter, is_head)
    heads = [s for s in song["sections"] if is_head(s)]
    bars = sum(s["bar_count"] for s in heads) or 1
    ev = merge_events(items)
    cnt, _, tonicized = detect(ev, tonic)

    changes = sum(1 for i, it in enumerate(items) if i == 0 or it["c"] != items[i - 1]["c"])
    per32 = lambda x: round(x * 32 / bars, 2)
    form = "".join(re.sub(r"\d+$", "", s["label"]) for s in heads)
    is_blues = (all(s["bar_count"] == 12 for s in heads) and len(ev) > 0
                and ev[0]["root"] == tonic and ev[0]["fam"] in ("dom", "maj", "min"))

    counts = {
        "ii_V_I": cnt["ii_V_I_major"] + cnt["ii_V_i_minor"],
        "ii_V_I_major": cnt["ii_V_I_major"],
        "ii_V_i_minor": cnt["ii_V_i_minor"],
        "backdoor": cnt["backdoor"],
        "iv_minor": cnt["iv_minor"],
        "turnaround": cnt["turnaround"],
        "tritone": cnt["tritone_sub"],
        "secondary": cnt["secondary_dominant"],
        "dom_chain": cnt["dom_chain"],
        "blues": 1 if is_blues else 0,
    }
    features = {
        "bars": bars,
        "form": form,
        "chord_changes_per_bar": round(changes / bars, 2),
        "tonicized_keys": [f"{n}{'m' if m == 'minor' else ''}" for n, m in tonicized],
        # パターンごとの回数と、32小節あたりの回数（曲の長さの影響を除いた比較用）
        "patterns": {k: v for k, v in counts.items() if v},
        "patterns_per32": {k: (1 if k == "blues" else per32(v)) for k, v in counts.items() if v},
    }
    return {
        "id": song["id"], "title": song["title"], "composer": song["composer"],
        "year_composed": song.get("year_composed"), "key": song["key"], "mode": mode,
        "relative_major": KEY_NAME[rel_tonic] if rel_tonic is not None else None,
        "meter": meter, "source": song.get("source"),
        "features": features, "sections": out_sections,
        "_events": ev,  # 類似度計算用（出力前に削除）
    }


# ---------------------------------------------------------------------------
# 類似度（移調に依存しない進行パターンのTF-IDF＋コサイン類似度）
# ---------------------------------------------------------------------------
def transition_tokens(ev):
    """隣り合うコードの「音程＋大分類」を並べたn-gram（キーに依存しない）"""
    ev = [e for e in ev if e["root"] is not None]
    steps = [(ev[i]["fam"], (ev[i + 1]["root"] - ev[i]["root"]) % 12, ev[i + 1]["fam"])
             for i in range(len(ev) - 1)]
    toks = []
    for i, (fa, iv, fb) in enumerate(steps):
        toks.append(f"{fa}>{iv}>{fb}")
        if i + 1 < len(steps):
            _, iv2, fc = steps[i + 1]
            toks.append(f"{fa}>{iv}>{fb}>{iv2}>{fc}")
    return toks


# メジャーキーで自然に現れる（度数, 大分類）の組み合わせ
_FIT = {(0, "maj"), (5, "maj"), (2, "min"), (4, "min"), (9, "min"), (7, "dom"),
        (7, "sus"), (11, "hdim"), (2, "hdim"), (0, "dom"), (5, "dom")}
_DIATONIC_MAJOR = {0, 2, 4, 5, 7, 9, 11}


def render_pattern(tok):
    """ "min>5>dom>5>maj" → "IIm – V7 – Imaj"
    基準の音を12通り試し、最も自然なローマ数字になる読み方を選ぶ"""
    parts = tok.split(">")
    fams, ivs = parts[0::2], [int(x) for x in parts[1::2]]
    roots = [0]
    for iv in ivs:
        roots.append((roots[-1] + iv) % 12)

    def penalty(ref):
        p = 0
        for r, f in zip(roots, fams):
            d = (r - ref) % 12
            p += 0 if (d, f) in _FIT else (1 if d in _DIATONIC_MAJOR else 2)
        return p + (0 if (roots[-1] - ref) % 12 == 0 else 0.5)  # 同点なら最後をIに

    ref = min(range(12), key=penalty)
    return " – ".join(NUMERAL[(r - ref) % 12] + FAM_SUFFIX[f] for r, f in zip(roots, fams))


def build_similarity(songs, top=10):
    docs = [transition_tokens(s["_events"]) for s in songs]
    vocab = {t: i for i, t in enumerate(sorted({t for d in docs for t in d}))}
    tf = np.zeros((len(docs), len(vocab)), np.float32)
    for r, d in enumerate(docs):
        for t, c in Counter(d).items():
            tf[r, vocab[t]] = c
    df = (tf > 0).sum(0)
    idf = np.log((1 + len(docs)) / (1 + df)) + 1
    x = tf * idf
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    sim = x @ x.T
    np.fill_diagonal(sim, -1)
    inv = {i: t for t, i in vocab.items()}

    out = {}
    for i, s in enumerate(songs):
        order = np.argsort(-sim[i])
        sims = []
        for j in order[:top]:
            shared = x[i] * x[j]
            idx = [k for k in np.argsort(-shared)[:6] if shared[k] > 0]
            pats = []
            for k in idx:  # 3コードのパターンを優先して表示
                p = render_pattern(inv[k])
                if p not in pats:
                    pats.append(p)
            pats.sort(key=lambda p: -p.count("–"))
            sims.append({"id": songs[j]["id"], "score": round(float(sim[i, j]), 3), "shared": pats[:3]})
        out[s["id"]] = sims
    return out


# ---------------------------------------------------------------------------
# 黒本（ジャズ・スタンダード・バイブル1・2）の収録曲マーク
# ---------------------------------------------------------------------------
# 黒本とiRealで曲名の表記が違うもの（黒本の表記 → iRealの表記）
KUROHON_ALIASES = {
    "FREDDIE THE FREELOADER": "Freddie Freeloader",
    "LOVE IS HERE TO STAY": "Our Love is Here to Stay",
    "RECADO": "Recado Bossa Nova",
    "EV'RY TIME WE SAY GOODBYE": "Every Time We Say Goodbye",
    "FIVE HUNDRED MILES HIGH": "500 Miles High",
    "GOLDEN EARRINGS": "Golden Earring",
    "I'M GETTIN SENTIMENTAL OVER YOU": "I'm Getting Sentimental Over You",
    "UNIT 7": "Unit Seven",
    "MEDITACAO": "Meditation",
}


def _norm(t):
    t = t.lower().strip()
    t = re.sub(r",\s*(the|a|an)$", "", t)
    t = re.sub(r"^(the|a|an)\s+", "", t)
    return re.sub(r"[^a-z0-9]", "", t.replace("&", "and"))


def _title_keys(t):
    """曲名の照合キー（冠詞・記号を無視。括弧内や「／」区切りの別名も含める）"""
    base = re.sub(r"\([^)]*\)", "", t)
    ks = {_norm(t), _norm(base)} | {_norm(p) for p in re.findall(r"\(([^)]*)\)", t)} \
        | {_norm(p) for p in re.split(r"[/／]", base)}
    ks.discard("")
    return ks


def mark_kurohon(songs, path="kurohon.json"):
    try:
        books = json.load(open(path, encoding="utf-8"))["books"]
    except FileNotFoundError:
        print("kurohon.json が無いため、黒本マークは付けません")
        return
    index = defaultdict(list)
    for s in songs:
        for k in _title_keys(s["title"]):
            index[k].append(s)
    missing = []
    for book, titles in books.items():
        for t in titles:
            hits = {id(x): x for k in _title_keys(KUROHON_ALIASES.get(t, t)) for x in index.get(k, [])}
            if not hits:
                missing.append(f"黒本{book}: {t}")
            for x in hits.values():
                x.setdefault("kurohon", [])
                if int(book) not in x["kurohon"]:
                    x["kurohon"].append(int(book))
    n = sum(1 for s in songs if s.get("kurohon"))
    print(f"黒本の収録曲 {n}曲にマーク。iRealのデータに見つからなかった曲 {len(missing)}曲:")
    for m in missing:
        print("   ", m)


# ---------------------------------------------------------------------------
def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "songs.json"
    songs = [analyze_song(s) for s in json.load(open(src, encoding="utf-8"))]
    mark_kurohon(songs)
    similar = build_similarity(songs)
    for s in songs:
        del s["_events"]
    json.dump({"patterns": PATTERNS, "songs": songs}, open("analyzed.json", "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    json.dump(similar, open("similar.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print(f"{len(songs)}曲を分析 → analyzed.json, similar.json")


if __name__ == "__main__":
    main()
