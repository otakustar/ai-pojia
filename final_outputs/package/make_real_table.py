#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd


def f(x, d=4):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("results/elaia_mcsa/elaia_mcsa_results.csv"))
    ap.add_argument("--output", type=Path, default=Path("paper/generated/elaia_mcsa.tex"))
    args = ap.parse_args()
    x = pd.read_csv(args.input)
    primary = x[(x.R == 1.0) & (x.multiplicity == "familywise")].copy()
    lines = [
        r"\begin{table}[ht]", r"\centering\scriptsize",
        r"\caption{最终MCSA六医院留出审计：主规格为$R=1$且医院族错误率$0.05/6$。风险变化负值表示改善。}",
        r"\label{tab:elaia-final}",
        r"\begin{tabular}{rrrrrrrr}", r"\toprule",
        r"医院 & 审计$n$ & $\theta_1$ & $\theta_2$ & 认证收益下界 & 风险变化 & 区间下端 & 区间上端\\",
        r"\midrule",
    ]
    for _, r in primary.iterrows():
        lines.append(
            f"{r.hospital} & {int(r.n_target_audit)} & {f(r.theta_prediction,4)} & "
            f"{f(r.theta_supported,4)} & {f(r.certified_lower_gain,5)} & "
            f"{f(r.audit_risk_change,5)} & {f(r.audit_ci_low,5)} & {f(r.audit_ci_high,5)}\\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\begin{minipage}{0.95\textwidth}\footnotesize "
              r"该分析是回顾性地理留出，不是历史部署后的前瞻验证；审计区间用于描述目标结局结果，认证决策不使用目标审计标签。\end{minipage}",
              r"\end{table}"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")

    updated = int(primary.updated.sum())
    harmful = int(((primary.audit_risk_change > 0) & (primary.updated > 0)).sum())
    text = (
        f"主规格在六个医院中采取了{updated}次非零更新；其中{harmful}次更新的目标审计风险差点估计为正。"
        "结果必须与$R=0.5$和$R=2$敏感性分析共同解释，不能把零更新等同于无临床异质性。"
    )
    args.output.with_name("elaia_findings.tex").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
