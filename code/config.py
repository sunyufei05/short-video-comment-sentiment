# -*- coding: utf-8 -*-
"""全局路径与常量配置。

对应论文 3.5 节「核心情感测度指标构建公式」中的平滑常数与词典约定。
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ---- 输入 ----
COMMENTS_CSV = DATA_DIR / "sample_comments.csv"   # 字段: video_id, comment_text
VIDEOS_CSV = DATA_DIR / "sample_videos.csv"       # 字段: video_id, category, duration_sec, likes, comments, shares

# ---- 词典与停用词 ----
LEXICON_POS = DATA_DIR / "lexicon_positive.txt"
LEXICON_NEG = DATA_DIR / "lexicon_negative.txt"
STOPWORDS = DATA_DIR / "stopwords_zh.txt"

# ---- 平滑常数 ε（论文 3.5 节，规避分母为 0）----
EPSILON = 0.0001

# ---- 文本清洗 / 筛选规则（论文 2.2 节）----
MIN_COMMENT_LEN = 2       # 文本筛选：剔除字数过少的无效评论

# ---- 视频类别（论文 3.2.2 节）----
CATEGORIES = ["生活", "美食", "娱乐"]
