# -*- coding: utf-8 -*-
"""实证分析（参考实现）：描述性统计 → 相关性 → 多元回归 → 稳健性检验。

对应论文 4.2 / 4.3 / 4.4 / 4.5 / 4.6 节。

用法:
    python empirical_analysis.py --input out/sentiment_index.csv --out out
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PREDICTORS = ["Pos_Sent", "Neg_Sent", "Total_Sent"]


# --------------------------------------------------------------------------
# 被解释变量：短视频热度（论文 3.4.1 节）
# --------------------------------------------------------------------------
def compute_popularity(df: pd.DataFrame) -> pd.Series:
    """Popularity = 三个互动量 log1p 后 Z-score 的等权均值。

    论文仅说明「结合点赞量、评论量、转发量进行标准化处理」，未给出权重。
    此处采用等权 Z-score 均值作为复现所需的最小假设。
    """
    parts = []
    for col in ("likes", "comments", "shares"):
        x = np.log1p(df[col].astype(float))
        parts.append((x - x.mean()) / x.std(ddof=0))
    return pd.concat(parts, axis=1).mean(axis=1)


def md_table(df: pd.DataFrame) -> str:
    """把带索引的 DataFrame 渲染为 GitHub 风格 Markdown 表格。

    逐行取值时先 astype(object)，避免 pandas 把整型列（如「样本量」）
    与浮点列混在一行后统一上抛为 float，出现 60.0 这类显示。
    """
    data = df.astype(object)
    head = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join([" --- "] * len(head)) + "|"]
    for i in range(len(data)):
        cells = []
        for v in data.iloc[i].tolist():
            if isinstance(v, float) and v == 0:
                v = 0.0                      # 消除 -0.0
            cells.append(str(v))
        lines.append("| " + " | ".join([str(data.index[i])] + cells) + " |")
    return "\n".join(lines)


def _stars(pv: float) -> str:
    if not np.isfinite(pv):
        return ""
    return "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.10 else ""))


def _fmt_coef(res, var: str) -> str:
    coef = res.params.get(var, np.nan)
    if not np.isfinite(coef):
        return "—"
    return f"{coef:.3f}{_stars(res.pvalues.get(var, np.nan))}({res.tvalues.get(var, np.nan):.3f})"


def build_table_models(sm, y, data: pd.DataFrame, x_base: pd.DataFrame):
    """返回 {模型名: (对应核心解释变量, 拟合结果)}。"""
    out = {}
    for i, p in enumerate(PREDICTORS):
        X = sm.add_constant(pd.concat([data[[p]].astype(float), x_base], axis=1))
        out[f"模型 {i+1} ({p})"] = (p, sm.OLS(y, X).fit())
    return out


def regression_table(models: dict, with_fit_stats: bool = True) -> pd.DataFrame:
    """核心解释变量逐行对齐到「自己所属的那一列」，其余列填「—」。"""
    rows = []
    for p in PREDICTORS:
        rows.append(pd.Series(
            {name: (_fmt_coef(res, p) if owner == p else "—")
             for name, (owner, res) in models.items()}, name=p))
    if with_fit_stats:
        rows.append(pd.Series(
            {name: _fmt_coef(res, "Time") for name, (_, res) in models.items()}, name="Time"))
        rows.append(pd.Series({name: "已控制" for name in models}, name="控制变量"))
        rows.append(pd.Series(
            {name: f"{res.rsquared_adj:.3f}" for name, (_, res) in models.items()}, name="调整 R²"))
        rows.append(pd.Series(
            {name: f"{int(res.nobs)}" for name, (_, res) in models.items()}, name="样本量"))
    t = pd.DataFrame(rows)
    t.index.name = "变量"
    return t


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="实证分析（描述统计 / 相关 / 回归 / 稳健性）")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df["Popularity"] = compute_popularity(df)

    # ---------------- 4.2 描述性统计 ----------------
    vars_ = ["Popularity", "Pos_Sent", "Neg_Sent", "Total_Sent", "Time"]
    desc = pd.DataFrame({
        "样本量": df[vars_].count().astype(int).values,
        "均值": df[vars_].mean().round(3).values,
        "标准差": df[vars_].std(ddof=1).round(3).values,
        "最小值": df[vars_].min().round(3).values,
        "最大值": df[vars_].max().round(3).values,
    }, index=vars_)
    desc.index.name = "变量"
    desc.to_csv(args.out / "table_descriptive.csv", encoding="utf-8-sig")
    (args.out / "table_descriptive.md").write_text(
        "# 变量描述性统计（参考实现输出）\n\n" + md_table(desc) + "\n", encoding="utf-8")

    # ---------------- 4.4 相关性分析 ----------------
    corr = df[vars_].corr(method="pearson").round(3)
    corr.index.name = "变量"
    corr.to_csv(args.out / "table_correlation.csv", encoding="utf-8-sig")
    (args.out / "table_correlation.md").write_text(
        "# 变量 Pearson 相关性矩阵（参考实现输出）\n\n" + md_table(corr) + "\n", encoding="utf-8")

    # ---------------- 4.5 多元回归 ----------------
    try:
        import statsmodels.api as sm
    except ImportError:
        print("[跳过] 未安装 statsmodels，无法执行回归。请 pip install -r requirements.txt")
        return

    dummies = pd.get_dummies(df["Type"], prefix="Type", drop_first=True).astype(float)
    x_base = pd.concat([df[["Time"]].astype(float), dummies], axis=1)
    y = df["Popularity"].astype(float)

    models = build_table_models(sm, y, df, x_base)
    reg = regression_table(models)
    reg.to_csv(args.out / "table_regression.csv", encoding="utf-8-sig")
    (args.out / "table_regression.md").write_text(
        "# 多元线性回归结果（参考实现输出）\n\n" + md_table(reg) +
        "\n\n> 括号内为 t 值；*** p<0.01，** p<0.05，* p<0.10。\n", encoding="utf-8")

    # ---------------- 4.6 稳健性检验：Z-score 标准化替换 ----------------
    z = df.copy()
    for c in PREDICTORS:
        z[c] = (z[c] - z[c].mean()) / z[c].std(ddof=1)
    rob_models = build_table_models(sm, y, z, x_base)
    rob = regression_table(rob_models, with_fit_stats=False)
    rob.to_csv(args.out / "table_robustness.csv", encoding="utf-8-sig")
    (args.out / "table_robustness.md").write_text(
        "# 稳健性检验：核心解释变量 Z-score 标准化后重新估计（参考实现输出）\n\n"
        + md_table(rob) +
        "\n\n> 系数符号与显著性与原模型保持一致即说明结论稳健。\n", encoding="utf-8")

    print("已输出：")
    for f in sorted(args.out.glob("table_*")):
        print("  -", f.name)


if __name__ == "__main__":
    main()
