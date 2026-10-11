# 基于 NLP 文本情感分析的短视频用户评论情绪对视频热度的影响研究

> 以抖音公开短视频数据为样本，自主构建 NLP 情感测度指标体系，实证检验用户评论情绪对视频热度的影响机制。

![研究类型](https://img.shields.io/badge/研究类型-实证研究-2b6cb0)
![方法](https://img.shields.io/badge/方法-NLP%20测度构建%20%2B%20多元回归-0f766e)
![样本](https://img.shields.io/badge/有效样本-N%20%3D%20406-b45309)
![语言](https://img.shields.io/badge/语言-中文%20%2F%20English-7c6bb0)
![许可](https://img.shields.io/badge/许可-CC%20BY%204.0-6b7280)

---

## 项目简介

短视频平台的流量分发机制与视频热度影响因素，是当下大数据与新媒体领域的热门研究话题。用户评论作为最重要的**非结构化文本数据**，蕴含用户真实的观看体验、情感态度与行为偏好，是反映视频口碑与传播潜力的核心数据载体。

既有研究多直接采用平台现成的情感标签或简单词频统计，缺少**自主构建 NLP 情感测度指标**的完整流程，存在测算精度低、可复现性差、数据分析不规范等问题。本研究立足大数据管理与应用专业的 NLP 文本测度构建方法，完成从数据采集到实证检验的**全流程自主建模**：

```
数据采集 → 文本预处理 → NLP 测度构建 → 指标检验 → 描述性统计 → 相关性分析 → 回归实证 → 稳健性检验
```

> 本仓库为该论文的**公开材料仓库**，收录论文正文、图表、参考文献、补充材料与可执行代码，并提供 GitHub Pages 项目主页。

## 核心研究结论

| 假设 | 内容 | 检验结果 |
| --- | --- | --- |
| **H1** | 积极情感指数与视频热度显著**正**相关 | ✅ 成立（β = 3.216，t = 6.842，p < 0.01） |
| **H2** | 消极情感指数与视频热度显著**负**相关 | ✅ 成立（β = −2.874，t = −5.921，p < 0.01） |
| **H3** | 中性评论情绪对视频热度无显著影响 | ✅ 成立 |

三点延伸结论：

1. **自建指标信效度达标**——词典分半信度 Pearson `r = 0.852`（p < 0.001）；高/低热度组积极情感指数均值 0.813 / 0.426（p < 0.001）。
2. **口碑情绪是核心隐性因素**——相较于视频时长等外在形式因素，用户观看后的情感体验与口碑评价对传播效果的影响更为关键（`Time` 系数不显著）。
3. **正向循环机制**——优质内容收获大量积极评论，形成「正向口碑 → 高热度 → 持续传播」的良性循环；劣质内容的负面评价则直接限制流量传播。

## 目录结构

```
.
├── README.md                  # 本文件：项目总览
├── LICENSE                    # CC BY 4.0
├── CITATION.cff               # 引用信息（GitHub 可一键引用）
├── paper/                     # 论文正文
│   ├── 论文正文.md             #   全文 Markdown（含公式与表格）
│   └── *.docx                 #   原始 Word 稿件
├── figures/                   # 图表（矢量 SVG）
│   ├── README.md              #   图表索引与说明
│   ├── fig3-1_*.svg           #   图 3-1  NLP 五步处理流水线
│   └── fig4-{1..4}_*.svg      #   图 4-1 ~ 4-4  统计分析图
├── tables/                    # 表格（Markdown + CSV 双格式）
│   ├── table4-1_变量描述性统计.*
│   ├── table4-2_变量相关性矩阵.*
│   └── table4-3_多元回归结果.*
├── references/                # 参考文献
│   ├── README.md
│   ├── references.md          #   可读列表
│   ├── references.bib         #   BibTeX
│   └── references.json        #   JSON
├── supplementary/             # 补充材料
│   ├── README.md
│   ├── 01_研究假设与变量定义.md
│   ├── 02_核心测度公式.md
│   ├── 03_NLP处理流水线.md
│   └── 04_数据与可复现性说明.md
├── code/                      # 可执行代码（参考实现）
│   ├── README.md
│   ├── requirements.txt
│   ├── selftest.py            #   零依赖自检（14 项断言）
│   ├── sentiment_pipeline.py  #   NLP 五步流水线
│   ├── empirical_analysis.py  #   描述统计 / 相关 / 回归 / 稳健性
│   ├── make_sample_data.py    #   示例数据生成（零依赖）
│   ├── config.py
│   └── data/                  #   示例词典、停用词、示例数据
└── docs/                      # GitHub Pages 项目主页
    ├── index.html
    ├── .nojekyll
    └── assets/                #   页面用图表
```

## 研究设计一览

### 样本

| 项目 | 说明 |
| --- | --- |
| 数据来源 | 抖音公开短视频平台（合规采集公开非涉密数据） |
| 抽样范围 | 生活、美食、娱乐三大热门类别 |
| 采集量 → 有效样本 | 420 条 → **406 条** |
| 剔除规则 | 无效短评、零互动视频、违规下架视频、纯广告视频 |

### 变量

| 类型 | 变量 | 符号 |
| --- | --- | --- |
| 被解释变量 | 短视频热度（点赞/评论/转发标准化） | `Popularity` |
| 核心解释变量 | 积极情感指数、消极情感指数、综合情感指数 | `Pos_Sent` `Neg_Sent` `Total_Sent` |
| 控制变量 | 视频时长、视频类别 | `Time` `Type` |

### 核心测度公式（自建，含平滑系数 ε = 0.0001）

$$
Total\_Sent=\frac{Pos-Neg}{Pos+Neg+\varepsilon}\qquad
Pos\_Sent=\frac{Pos}{Pos+Neg+\varepsilon}\qquad
Neg\_Sent=\frac{Neg}{Pos+Neg+\varepsilon}
$$

其中 `Pos`、`Neg` 分别为单条视频全部评论中积极/消极情感词典命中的词汇总频次。公式推导与取值性质见 [`supplementary/02_核心测度公式.md`](supplementary/02_核心测度公式.md)。

### 回归模型

$$
Popularity=\beta_0+\beta_1\,X+\beta_2\,Time+\beta_3\,Type+\epsilon
$$

分别令 `X` 取 `Pos_Sent`、`Neg_Sent`、`Total_Sent`，得到三组模型；控制变量 `Type` 为生活/美食/娱乐虚拟变量。

## 图表速览

| 图 3-1　NLP 五步处理流水线 | 图 4-1　情感指数描述性统计 |
| --- | --- |
| ![图3-1](figures/fig3-1_NLP五步处理流水线.svg) | ![图4-1](figures/fig4-1_情感指数描述性统计.svg) |

| 图 4-2　变量相关性矩阵 | 图 4-3　多元回归系数 |
| --- | --- |
| ![图4-2](figures/fig4-2_变量相关性热力图.svg) | ![图4-3](figures/fig4-3_多元回归系数.svg) |

全部图表说明见 [`figures/README.md`](figures/README.md)。

## 复现

```bash
cd code
pip install -r requirements.txt

python selftest.py                                           # 核心算法自检（零依赖，14/14 通过）
python make_sample_data.py                                   # 生成示例数据
python sentiment_pipeline.py --comments data/sample_comments.csv \
                             --videos   data/sample_videos.csv \
                             --out      out                  # NLP 五步流水线
python empirical_analysis.py --input out/sentiment_index.csv \
                             --out   out                     # 实证分析
```

> **⚠️ 示例数据不是论文数据。**
> `code/data/sample_*.csv` 为固定随机种子构造的**结构演示数据**，其数值与论文的 406 条样本**没有任何关系**。原始评论语料受平台数据政策约束未公开，详见 [`supplementary/04_数据与可复现性说明.md`](supplementary/04_数据与可复现性说明.md)。
>
> 论文可复算的全部结果（表 4-1 ~ 表 4-3）已完整导出至 [`tables/`](tables/)，可直接用于核对与二次分析。

## 引用

```bibtex
@misc{shortvideo_comment_sentiment,
  title  = {基于 NLP 文本情感分析的短视频用户评论情绪对视频热度的影响研究},
  author = {孙雨霏},
  year   = {2024},
  note   = {大数据管理与应用专业},
  url    = {https://github.com/sunyufei05/short-video-comment-sentiment}
}
```

详见 [`CITATION.cff`](CITATION.cff)。

## 数据可得性与声明

| 内容 | 是否公开 |
| --- | --- |
| 论文正文（Markdown + 原始 docx） | ✅ |
| 图表（3 张表格 + 5 张重绘图形） | ✅ |
| 参考文献（md / bib / json） | ✅ |
| 方法、变量、公式、流水线说明 | ✅ |
| 参考实现代码与示例数据 | ✅ |
| **原始短视频评论语料（406 条样本）** | ❌ 受平台数据政策与个人信息保护约束，未公开 |

本仓库为学术研究材料归档，仅供学习与研究参考。

## 许可协议

论文文本、图表与文档采用 **Creative Commons Attribution 4.0 International (CC BY 4.0)** 许可，详见 [`LICENSE`](LICENSE)。
代码部分（`code/`）同样以 CC BY 4.0 提供，可自由用于研究目的，请保留署名。

---

<sub>仓库结构由论文材料自动整理生成 · 图表依据论文表格与正文数值重绘</sub>
