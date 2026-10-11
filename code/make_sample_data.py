# -*- coding: utf-8 -*-
"""生成结构一致的示例数据（固定随机种子，结果可复现）。

零第三方依赖，仅使用标准库；可直接运行：
    python make_sample_data.py

⚠️ 本脚本生成的是【构造数据】。字段结构对齐论文，但数值与论文的 406 条样本
   完全无关，仅用于验证 sentiment_pipeline.py / empirical_analysis.py 能否跑通。
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

from config import CATEGORIES, DATA_DIR

SEED = 20240101
N_VIDEOS = 60
MAX_COMMENTS_PER_VIDEO = 40

POS_WORDS = ["好看", "精彩", "治愈", "喜欢", "推荐", "太绝了", "惊艳", "实用",
             "不错", "很棒", "满意", "开心", "感动", "干货", "宝藏", "值得"]
NEG_WORDS = ["难看", "无聊", "敷衍", "浪费时间", "虚假", "差评", "失望",
             "一般", "拉胯", "无语", "劝退", "广告", "套路", "标题党"]
NEU_WORDS = ["还行", "路过", "第一次刷到", "记录一下", "看完了", "支持"]

TEMPLATES = [
    "这个{s}，{s}！", "真的{s}，看完了", "感觉{s}，怎么说",
    "{s}，推荐给大家", "{s}{s}", "有点{s}", "{s}，但是还行", "第一次看，{s}",
    "刷到就停下来看了，{s}", "{s}，已经收藏",
]


def make_comment(rng: random.Random, affinity: float) -> str:
    """affinity ∈ [0,1]：越大越偏向正向评论。"""
    picked = None
    if rng.random() < affinity:
        picked = rng.choice(POS_WORDS)
    elif rng.random() < (1 - affinity) * 0.8:
        picked = rng.choice(NEG_WORDS)
    if picked is None:
        picked = rng.choice(NEU_WORDS)
    return rng.choice(TEMPLATES).format(s=picked)


def main() -> None:
    rng = random.Random(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    videos, comments = [], []
    for i in range(1, N_VIDEOS + 1):
        vid = f"SV{i:04d}"
        category = CATEGORIES[i % len(CATEGORIES)]
        duration = rng.randint(7, 59)

        # 构造出「正向口碑越强 → 互动量越高」的弱相关关系，仅供流程演示
        affinity = round(rng.uniform(0.25, 0.95), 3)
        likes = max(0, int(300 + affinity * 4200 + rng.gauss(0, 350)))
        n_comments = max(1, int(likes / rng.uniform(28, 60)))
        shares = max(0, int(likes / rng.uniform(9, 26)))

        videos.append([vid, category, duration, likes, n_comments, shares])
        for _ in range(min(n_comments, MAX_COMMENTS_PER_VIDEO)):
            comments.append([vid, make_comment(rng, affinity)])

    with open(DATA_DIR / "sample_videos.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "category", "duration_sec", "likes", "comments", "shares"])
        w.writerows(videos)

    with open(DATA_DIR / "sample_comments.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "comment_text"])
        w.writerows(comments)

    print(f"已生成 {len(videos)} 条视频、{len(comments)} 条评论 -> {DATA_DIR}")


if __name__ == "__main__":
    main()
