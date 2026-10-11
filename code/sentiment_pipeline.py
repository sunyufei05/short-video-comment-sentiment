# -*- coding: utf-8 -*-
"""NLP 五步处理流水线（参考实现）。

对应论文 2.2 / 2.3 / 3.3 / 3.5 节：
    步骤一 原始数据采集      -> 读取 comments / videos 两张表
    步骤二 文本数据清洗      -> 正则剔除表情、符号、数字、乱码
    步骤三 分词与停用词过滤  -> Jieba 精准分词 + 新媒体停用词表
    步骤四 情感词典匹配统计  -> 统计每视频 Pos / Neg
    步骤五 构建情感测度指标  -> Pos_Sent / Neg_Sent / Total_Sent

用法:
    python sentiment_pipeline.py --comments data/sample_comments.csv \
        --videos data/sample_videos.csv --out out
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from config import (COMMENTS_CSV, EPSILON, LEXICON_NEG, LEXICON_POS,
                    MIN_COMMENT_LEN, STOPWORDS, VIDEOS_CSV)

# 说明：核心算法（clean_text / segment / count_hits / compute_sentiment_index）
# 仅依赖标准库，可直接导入使用；pandas 仅在 run() 读写 CSV 时才需要，
# 因此放在函数内部延迟导入。

# --------------------------------------------------------------------------
# 步骤二：文本数据清洗（论文 2.2 节）
# --------------------------------------------------------------------------
# 保留中文、拉丁字母、数字以外的内容视作噪声；表情、乱码、特殊标点一并剔除。
_CLEAN_PATTERN = re.compile(r"[^\u4e00-\u9fffA-Za-z]")
_REPEAT_PATTERN = re.compile(r"(.)\1{2,}")          # 重复 3 次以上的字符压缩为 1 个


def clean_text(text: str) -> str:
    """正则清洗：去表情、去符号、去数字、去乱码，并压缩重复字符。"""
    if not isinstance(text, str):
        return ""
    t = _REPEAT_PATTERN.sub(r"\1", text)
    t = _CLEAN_PATTERN.sub("", t)
    return t


# --------------------------------------------------------------------------
# 步骤三：分词与停用词过滤（论文 2.2 / 3.3 节）
# --------------------------------------------------------------------------
def load_lines(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def segment(text: str, stopwords: set[str]) -> list[str]:
    """Jieba 精准分词 + 停用词过滤。

    未安装 jieba 时退化为「按字切分」，保证脚本在零依赖环境下仍可运行，
    且不会因为切分粒度变化而虚增字符总量。
    """
    if not text:
        return []
    try:
        import jieba
        tokens = jieba.lcut(text, cut_all=False)
    except ImportError:
        tokens = list(text)
    return [w for w in tokens if w.strip() and w not in stopwords]


# --------------------------------------------------------------------------
# 步骤四：情感词典匹配统计（论文 2.3 / 3.3 节）
# --------------------------------------------------------------------------
def count_hits(text: str, lexicon: set[str]) -> int:
    """在清洗后的连续文本上做「最长优先、不重叠」扫描。

    采用最长优先而非逐词匹配，是为了避免分词粒度把「太绝了」「浪费时间」
    这类多字情感词切开而漏检。
    """
    if not text or not lexicon:
        return 0
    terms = sorted(lexicon, key=len, reverse=True)
    hits, i, n = 0, 0, len(text)
    while i < n:
        for term in terms:
            if text.startswith(term, i):
                hits += 1
                i += len(term)
                break
        else:
            i += 1
    return hits


# --------------------------------------------------------------------------
# 步骤五：构建情感测度指标（论文 3.5 节）
# --------------------------------------------------------------------------
def compute_sentiment_index(pos: float, neg: float, epsilon: float = EPSILON) -> dict:
    """三个自建指标：

        Total_Sent = (Pos - Neg) / (Pos + Neg + ε)
        Pos_Sent   =  Pos        / (Pos + Neg + ε)
        Neg_Sent   =  Neg        / (Pos + Neg + ε)
    """
    denom = pos + neg + epsilon
    return {
        "Pos_Sent": pos / denom,
        "Neg_Sent": neg / denom,
        "Total_Sent": (pos - neg) / denom,
    }


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def run(comments_path: Path, videos_path: Path, out_dir: Path):
    import pandas as pd          # 延迟导入：纯算法路径（selftest）无需 pandas

    lexicon_pos = load_lines(LEXICON_POS)
    lexicon_neg = load_lines(LEXICON_NEG)
    stopwords = load_lines(STOPWORDS)
    print(f"词典规模：积极 {len(lexicon_pos)} 条 / 消极 {len(lexicon_neg)} 条；"
          f"停用词 {len(stopwords)} 条")

    comments = pd.read_csv(comments_path)
    videos = pd.read_csv(videos_path)

    kept_chars = cleaned_chars = 0
    rows = []
    for rec in comments.itertuples(index=False):
        cleaned = clean_text(getattr(rec, "comment_text"))
        if len(cleaned) < MIN_COMMENT_LEN:      # 文本筛选
            continue
        tokens = segment(cleaned, stopwords)
        cleaned_chars += len(cleaned)
        kept_chars += len("".join(tokens))
        rows.append({
            "video_id": getattr(rec, "video_id"),
            "Pos": count_hits(cleaned, lexicon_pos),
            "Neg": count_hits(cleaned, lexicon_neg),
        })

    detail = pd.DataFrame(rows)
    agg = detail.groupby("video_id", as_index=False)[["Pos", "Neg"]].sum()

    idx = pd.DataFrame(agg["video_id"])
    calc = agg.apply(lambda r: compute_sentiment_index(r["Pos"], r["Neg"]), axis=1)
    idx["Pos"] = agg["Pos"].values
    idx["Neg"] = agg["Neg"].values
    idx["Pos_Sent"] = [c["Pos_Sent"] for c in calc]
    idx["Neg_Sent"] = [c["Neg_Sent"] for c in calc]
    idx["Total_Sent"] = [c["Total_Sent"] for c in calc]

    merged = videos.merge(idx, on="video_id", how="inner")
    merged = merged.rename(columns={"duration_sec": "Time", "category": "Type"})

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "sentiment_index.csv"
    merged.to_csv(target, index=False, encoding="utf-8-sig")

    retention = kept_chars / cleaned_chars if cleaned_chars else 0.0
    print(f"清洗后有效评论 {len(detail)} 条；有效语义词汇保留率 ≈ {retention:.2%}")
    print(f"覆盖视频 {len(merged)} 条 -> {target}")
    return merged


def main():
    ap = argparse.ArgumentParser(description="NLP 五步处理流水线")
    ap.add_argument("--comments", type=Path, default=COMMENTS_CSV)
    ap.add_argument("--videos", type=Path, default=VIDEOS_CSV)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    args = ap.parse_args()
    run(args.comments, args.videos, args.out)


if __name__ == "__main__":
    main()
