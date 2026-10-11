# -*- coding: utf-8 -*-
"""核心算法自检脚本（零第三方依赖，仅用标准库）。

校验 sentiment_pipeline.py 中的纯函数与端到端流程是否符合论文 3.5 节
给出的数学性质：有界性、平滑项防除零、占比互补性、最长优先匹配。

运行:
    python selftest.py
退出码 0 表示全部通过。
"""
from __future__ import annotations

import csv
import sys

from config import DATA_DIR, EPSILON
from sentiment_pipeline import (clean_text, compute_sentiment_index, count_hits,
                                load_lines, LEXICON_NEG, LEXICON_POS, segment,
                                STOPWORDS)

results: list[tuple[bool, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    results.append((ok, name))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))


# --------------------------------------------------------------------------
# 1. 文本清洗（论文 2.2 节）
# --------------------------------------------------------------------------
c = clean_text("这个视频太好看了😍😍！！！123 #话题 @某人")
check(c == "这个视频太好看了话题某人", "文本清洗：剔除表情/数字/标点与话题符号", c)

c2 = clean_text("好好好好好看")
check(c2 == "好看", "文本清洗：连续重复字符压缩为 1 个", c2)

check(clean_text(None) == "", "文本清洗：非字符串输入返回空串")

# --------------------------------------------------------------------------
# 2. 最长优先、不重叠的词典匹配（论文 2.3 节）
# --------------------------------------------------------------------------
lex = {"好看", "太绝了", "绝了"}
t1 = count_hits("太绝了这个", lex)
check(t1 == 1, "词典匹配：最长优先命中「太绝了」而非「绝了」", f"hits={t1}")
t2 = count_hits("好看好看", lex)
check(t2 == 2, "词典匹配：重复词条计数为 2", f"hits={t2}")
check(count_hits("", lex) == 0, "词典匹配：空文本返回 0")

# --------------------------------------------------------------------------
# 3. 情感测度公式的数学性质（论文 3.5 节）
# --------------------------------------------------------------------------
z = compute_sentiment_index(0, 0)
check(z["Total_Sent"] == 0.0 and z["Pos_Sent"] == 0.0 and z["Neg_Sent"] == 0.0,
      "公式：Pos=Neg=0 时不发生除零（ε 生效）", str(z))

h = compute_sentiment_index(100, 0)
check(abs(h["Pos_Sent"] - 1.0) < 1e-3 and h["Total_Sent"] > 0.99,
      "公式：全为正向词时 Pos_Sent→1、Total_Sent→1", f"Pos_Sent={h['Pos_Sent']:.6f}")

h2 = compute_sentiment_index(0, 100)
check(abs(h2["Neg_Sent"] - 1.0) < 1e-3 and h2["Total_Sent"] < -0.99,
      "公式：全为负向词时 Neg_Sent→1、Total_Sent→-1", f"Neg_Sent={h2['Neg_Sent']:.6f}")

bad = []
for pos, neg in [(0, 0), (1, 0), (0, 1), (3, 7), (50, 50), (999, 1), (7, 3)]:
    r = compute_sentiment_index(pos, neg)
    if not (-1 < r["Total_Sent"] < 1):
        bad.append((pos, neg, r["Total_Sent"]))
    if not (0 <= r["Pos_Sent"] < 1 and 0 <= r["Neg_Sent"] < 1):
        bad.append((pos, neg, "占比越界"))
    if abs(r["Pos_Sent"] + r["Neg_Sent"] - (pos + neg) / (pos + neg + EPSILON)) > 1e-9:
        bad.append((pos, neg, "互补性不成立"))
check(not bad, "公式：有界性 Total_Sent∈(-1,1)、占比∈[0,1)、互补性成立", str(bad) if bad else "全部通过")

# --------------------------------------------------------------------------
# 4. 端到端：对示例数据跑完整流水线并校验指标分布
# --------------------------------------------------------------------------
pos_lex = load_lines(LEXICON_POS)
neg_lex = load_lines(LEXICON_NEG)
stops = load_lines(STOPWORDS)
check(len(pos_lex) > 0 and len(neg_lex) > 0, "词典加载成功",
      f"积极 {len(pos_lex)} / 消极 {len(neg_lex)} / 停用词 {len(stops)}")

comments_path = DATA_DIR / "sample_comments.csv"
if comments_path.exists():
    agg: dict[str, list[int]] = {}
    kept = cleaned_total = 0
    with open(comments_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            text = clean_text(row["comment_text"])
            if len(text) < 2:
                continue
            toks = segment(text, stops)
            cleaned_total += len(text)
            kept += len("".join(toks))
            slot = agg.setdefault(row["video_id"], [0, 0])
            slot[0] += count_hits(text, pos_lex)
            slot[1] += count_hits(text, neg_lex)

    idx = [compute_sentiment_index(p, n) for p, n in agg.values()]
    n_v = len(idx)
    ok_bounds = all(-1 < r["Total_Sent"] < 1 for r in idx)
    mean_pos = sum(r["Pos_Sent"] for r in idx) / n_v if n_v else 0
    mean_neg = sum(r["Neg_Sent"] for r in idx) / n_v if n_v else 0
    check(n_v > 0, "端到端：全部视频均成功构建情感测度", f"覆盖视频 {n_v} 条")
    check(ok_bounds, "端到端：所有视频的 Total_Sent 均落在 (-1, 1)")
    check(mean_pos > mean_neg, "端到端：示例数据正向情绪占优（构造使然）",
          f"Pos_Sent 均值 {mean_pos:.3f} > Neg_Sent 均值 {mean_neg:.3f}")
    print(f"       有效语义词汇保留率 ≈ {kept / cleaned_total:.2%}"
          f"（论文 4.1 节报告值 82%，示例词典规模不同，仅作参照）")
else:
    check(False, "端到端：未找到示例数据", "请先运行 python make_sample_data.py")

# --------------------------------------------------------------------------
passed = sum(1 for ok, _ in results if ok)
print(f"\n===== {passed}/{len(results)} 项通过 =====")
sys.exit(0 if passed == len(results) else 1)
