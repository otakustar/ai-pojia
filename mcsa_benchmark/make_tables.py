#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_LABELS = {
    "no_update": "不更新",
    "unrestricted_retrain": "自由重训",
    "transport_aipw_safe": "Transport-AIPW+门槛",
    "aml_point": "AML/Riesz点更新",
    "best_single_safe": "最优单方向安全更新",
    "qaggregation_safe": "Q-Aggregation+门槛",
    "krr_transfer_safe": "KRR迁移+门槛",
    "mcsa": "MCSA",
}
SCENARIO_LABELS = {
    "direction_favorable": "方向可认证的大偏移",
    "regular_overlap": "常规良好重叠",
    "null_update": "历史模型正确",
    "radius_misspecified": "半径/包络错设压力",
}
ORDER = list(METHOD_LABELS)


def esc(s: str) -> str:
    return (str(s).replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def fmt(x: float, digits: int = 4) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.{digits}f}"


def write_main_table(summary: pd.DataFrame, out: Path, n_show: int) -> None:
    sub = summary[summary.n == n_show].copy()
    lines = [
        r"\begin{table}[ht]",
        r"\centering\scriptsize",
        rf"\caption{{八方法同预算比较（$n={n_show}$）。风险变化越小越好；有害率与接受率为比例。}}",
        r"\label{tab:eight-main}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"场景 & 方法 & 平均风险变化 & 有害率 & 接受率 & 覆盖率\\",
        r"\midrule",
    ]
    for scenario in SCENARIO_LABELS:
        ss = sub[sub.scenario == scenario].set_index("method")
        first = True
        for method in ORDER:
            if method not in ss.index:
                continue
            r = ss.loc[method]
            scen = SCENARIO_LABELS[scenario] if first else ""
            first = False
            lines.append(
                f"{esc(scen)} & {esc(METHOD_LABELS[method])} & "
                f"{fmt(r.mean_risk_change, 5)} & {fmt(r.mean_harm, 3)} & "
                f"{fmt(r.mean_accepted, 3)} & {fmt(r.mean_coverage, 3)}\\\\"
            )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\begin{minipage}{0.96\textwidth}\footnotesize "
              r"自由重训和AML/Riesz点更新是未施加5\%最终安全门槛的性能参照，覆盖率不适用。"
              r"KRR迁移为COKE思想的KRR同级适配实现，不是原论文代码的逐字复现。\end{minipage}",
              r"\end{table}"]
    out.write_text("\n".join(lines), encoding="utf-8")


def write_paired_table(paired: pd.DataFrame, out: Path, scenario: str) -> None:
    sub = paired[paired.scenario == scenario].copy()
    lines = [
        r"\begin{table}[ht]",
        r"\centering\scriptsize",
        rf"\caption{{{esc(SCENARIO_LABELS[scenario])}下各方法风险减去MCSA风险的配对Monte Carlo区间。正值有利于MCSA。}}",
        r"\label{tab:paired-mcsa}",
        r"\begin{tabular}{rlrrr}", r"\toprule",
        r"$n$ & 方法 & 平均差 & 区间下端 & 区间上端\\", r"\midrule",
    ]
    for n in sorted(sub.n.unique()):
        ss = sub[sub.n == n].set_index("method_minus_mcsa")
        for method in ORDER:
            if method == "mcsa" or method not in ss.index:
                continue
            r = ss.loc[method]
            lines.append(
                f"{n} & {esc(METHOD_LABELS[method])} & {fmt(r.mean_difference,5)} & "
                f"{fmt(r.ci_low,5)} & {fmt(r.ci_high,5)}\\\\"
            )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines), encoding="utf-8")


def findings(summary: pd.DataFrame, paired: pd.DataFrame) -> dict:
    result: dict[str, object] = {}
    fav = paired[(paired.scenario == "direction_favorable") &
                 (paired.method_minus_mcsa != "mcsa")]
    result["favorable_ci_support_mcsa"] = int((fav.ci_low > 0).sum())
    result["favorable_ci_against_mcsa"] = int((fav.ci_high < 0).sum())
    result["favorable_total_comparisons"] = int(len(fav))

    safe = summary[summary.method.isin([
        "no_update", "transport_aipw_safe", "best_single_safe",
        "qaggregation_safe", "krr_transfer_safe", "mcsa"
    ])]
    result["max_harm_theory_scenarios"] = float(
        safe[safe.scenario != "radius_misspecified"].mean_harm.max()
    )
    result["max_harm_misspecified"] = float(
        safe[safe.scenario == "radius_misspecified"].mean_harm.max()
    )

    for scenario in SCENARIO_LABELS:
        for n in sorted(summary.n.unique()):
            ss = summary[(summary.scenario == scenario) & (summary.n == n)]
            best = ss.loc[ss.mean_risk_change.idxmin()]
            result[f"best_{scenario}_{n}"] = {
                "method": best.method,
                "mean_risk_change": float(best.mean_risk_change),
            }
    return result


def write_findings_tex(info: dict, out: Path) -> None:
    supported = info["favorable_ci_support_mcsa"]
    against = info["favorable_ci_against_mcsa"]
    total = info["favorable_total_comparisons"]
    text = (
        rf"在方向可认证的大偏移场景中，共有{total}个方法--样本量配对比较；"
        rf"其中{supported}个区间完全支持MCSA风险更低，{against}个区间完全支持MCSA风险更高。"
        rf"理论条件内安全方法观察到的最大有害更新比例为"
        rf"{info['max_harm_theory_scenarios']:.3f}；半径/包络错设压力场景的对应最大值为"
        rf"{info['max_harm_misspecified']:.3f}。"
    )
    out.write_text(text, encoding="utf-8")


def make_plots(summary: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for scenario in ["direction_favorable", "regular_overlap", "null_update", "radius_misspecified"]:
        ss = summary[summary.scenario == scenario]
        plt.figure(figsize=(9, 5.2))
        for method in ORDER:
            mm = ss[ss.method == method].sort_values("n")
            if len(mm):
                plt.plot(mm.n, mm.mean_risk_change, marker="o", label=METHOD_LABELS[method])
        plt.axhline(0, linewidth=1)
        plt.xlabel("来源标签总量 n")
        plt.ylabel("平均目标风险变化")
        plt.title(SCENARIO_LABELS[scenario])
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(outdir / f"risk_{scenario}.png", dpi=180)
        plt.close()

        plt.figure(figsize=(9, 5.2))
        for method in ORDER:
            mm = ss[ss.method == method].sort_values("n")
            if len(mm):
                plt.plot(mm.n, mm.mean_harm, marker="o", label=METHOD_LABELS[method])
        plt.axhline(0.05, linewidth=1, linestyle="--")
        plt.xlabel("来源标签总量 n")
        plt.ylabel("有害更新比例")
        plt.title(SCENARIO_LABELS[scenario])
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(outdir / f"harm_{scenario}.png", dpi=180)
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/eight_method"))
    ap.add_argument("--tables", type=Path, default=Path("paper/generated"))
    ap.add_argument("--figures", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()
    summary = pd.read_csv(args.results / "summary.csv")
    paired = pd.read_csv(args.results / "paired_vs_mcsa.csv")
    args.tables.mkdir(parents=True, exist_ok=True)
    write_main_table(summary, args.tables / "eight_method_main.tex", int(sorted(summary.n.unique())[1]))
    write_paired_table(paired, args.tables / "paired_favorable.tex", "direction_favorable")
    info = findings(summary, paired)
    (args.results / "findings.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    write_findings_tex(info, args.tables / "findings.tex")
    make_plots(summary, args.figures)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
