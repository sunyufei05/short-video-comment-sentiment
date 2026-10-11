# 可执行代码（参考实现）

本目录是论文所述 NLP 测度构建与实证流程的**参考实现**，按流水线顺序组织：

```
data/sample_comments.csv ─┐
                          ├─► sentiment_pipeline.py ─► out/sentiment_index.csv
data/sample_videos.csv  ──┘                              │
                                                         ▼
                                       empirical_analysis.py ─► out/*.md, out/*.csv
```

## 文件说明

| 文件 | 作用 | 对应论文 |
| --- | --- | --- |
| `sentiment_pipeline.py` | 文本清洗 → Jieba 分词 → 去停用词 → 情感词典匹配 → 计算三个情感指数 | 2.2 / 3.3 / 3.5 节 |
| `empirical_analysis.py` | 描述性统计 → Pearson 相关 → 多元线性回归 → 稳健性检验（Z-score 替换） | 4.2 / 4.4 / 4.5 / 4.6 节 |
| `selftest.py` | **零依赖**自检：校验清洗规则、最长优先匹配、公式的有界性/互补性/防除零 | 2.2 / 2.3 / 3.5 节 |
| `make_sample_data.py` | 生成结构一致的示例数据（固定随机种子，可复现） | — |
| `config.py` | 路径与常量（含平滑系数 ε） | 3.5 节 |
| `data/lexicon_positive.txt`<br>`data/lexicon_negative.txt` | 积极 / 消极情感词典（示例片段） | 2.3 节 |
| `data/stopwords_zh.txt` | 中文新媒体停用词表（示例片段） | 2.2 节 |

## 运行环境

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 0) 自检：核心算法的数学性质（零依赖，可先跑这一步）
python selftest.py
#    → 14/14 项通过

# 1) 生成示例数据（仓库已附带生成结果，可跳过）
python make_sample_data.py

# 2) 构建情感测度指标：输出 out/sentiment_index.csv
python sentiment_pipeline.py \
    --comments data/sample_comments.csv \
    --videos    data/sample_videos.csv \
    --out       out

# 3) 实证分析：输出 out/table_descriptive.md、out/table_correlation.md、
#    out/table_regression.md、out/table_robustness.md 及对应 CSV
python empirical_analysis.py --input out/sentiment_index.csv --out out
```

## 已验证的运行结果

`selftest.py` 覆盖 14 项断言：清洗规则、最长优先不重叠匹配、`Pos=Neg=0` 不除零、
`Total_Sent ∈ (−1, 1)`、占比 `∈ [0, 1)`、互补性、端到端覆盖全部视频 —— 全部通过。
在示例数据（60 条视频 / 2263 条评论）上跑完整流程，可复现地得到三张结果表。

## 重要声明

> **示例数据不是论文数据。**
> `data/sample_*.csv` 由 `make_sample_data.py` 以固定随机种子构造，仅用于验证代码可运行。其字段结构与论文一致，但**数值与论文的 406 条样本无关**，不可用于复算或引用论文结论。
>
> `data/lexicon_*.txt` 与 `data/stopwords_zh.txt` 收录论文明确列举的示例词条并做必要扩充，规模远小于研究实际使用版本。
>
> 论文可复算的产出（表 4-1 ~ 表 4-3）已完整导出至 [`../tables/`](../tables)。

## 与论文的一处口径差异

论文 3.4.1 节说明视频热度（`Popularity`）由「点赞量、评论量、转发量**标准化**处理」得到，但未给出各分项权重。本参考实现采用**等权 Z-score 均值**：

```
Popularity = mean( z(log1p(likes)), z(log1p(comments)), z(log1p(shares)) )
```

对数变换用于缓解互动量的长尾分布。该口径为复现所需的最小假设，若论文实际使用了其他权重方案，只需修改 `empirical_analysis.py` 中的 `compute_popularity()`。

## 示例数据的已知特征（阅读结果表前请先看这里）

`selftest.py` 与示例数据上的运行结果中有两处**属于构造数据的固有特征**，不代表真实研究结论：

1. **`Pos_Sent` 与 `Neg_Sent` 相关系数为 −1.0。** 示例评论每条只含一种极性词，使两个占比精确互补；真实短视频评论会同时包含正负词与大量中性词，不会出现这种完全共线。
2. **示例数据的调整 R²（约 0.61）远高于论文表 4-3 的 0.198 ~ 0.234。** 因为示例数据是「按口碑强度反推互动量」生成的，相关关系被刻意放大。

正因如此，`code/out/` 下的任何数值都**不可**与论文表 4-1 ~ 表 4-3 相互印证；论文结果的唯一权威来源是 [`../tables/`](../tables)。
