#!/usr/bin/env python3
"""
Compare quality evaluations from two LLM judges (GPT-4o-mini and Claude-Haiku 4.5).

Reports:
  - Condition-level means for each judge and each dimension
  - Replication test: Diversity Prompt > Baseline on creativity and overall
  - Inter-judge reliability (Pearson correlation between paired ratings)

Reproduces the numbers reported in the paper's Results section
("Diversity Prompt does not lower quality in LLM-judge evaluation")
and in Supplementary Section S4.

Usage:
    python analysis/compare_judges.py
"""

import os
import json
import numpy as np
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPT_PATH    = os.path.join(BASE_DIR, "data/quality_eval/gpt4o-mini_judge.json")
CLAUDE_PATH = os.path.join(BASE_DIR, "data/quality_eval/claude-haiku-4-5_judge.json")

DIMENSIONS = ["creativity", "relevance", "practicality", "completeness", "clarity", "overall"]


def main():
    with open(GPT_PATH) as f:
        gpt = json.load(f)
    with open(CLAUDE_PATH) as f:
        claude = json.load(f)

    print("=" * 72)
    print("CONDITION-LEVEL MEANS (n=15 per condition)")
    print("=" * 72)
    print(f"{'Condition':<22} {'Dimension':<14} {'GPT-4o-mini':<14} {'Claude-Haiku':<14}")
    print("-" * 72)
    for cond in gpt.keys():
        for dim in DIMENSIONS:
            g = [e[dim] for e in gpt[cond] if dim in e]
            c = [e[dim] for e in claude[cond] if dim in e]
            if g and c:
                print(f"{cond:<22} {dim:<14} {np.mean(g):.2f}±{np.std(g):.2f}    "
                      f"{np.mean(c):.2f}±{np.std(c):.2f}")
        print()

    print("=" * 72)
    print("KEY REPLICATION — Diversity Prompt vs Baseline (Group)")
    print("=" * 72)
    for dim in ["creativity", "overall"]:
        print(f"\n--- {dim} ---")
        for name, evals in [("GPT-4o-mini", gpt), ("Claude-Haiku 4.5", claude)]:
            base = [e[dim] for e in evals["Baseline (Group)"]]
            div  = [e[dim] for e in evals["Diversity Prompt"]]
            t, p = stats.ttest_ind(div, base, equal_var=False)
            d = (np.mean(div) - np.mean(base)) / np.sqrt(
                (np.var(div, ddof=1) + np.var(base, ddof=1)) / 2
            )
            print(f"  {name}: Div({np.mean(div):.2f}) vs Base({np.mean(base):.2f})  "
                  f"t={t:.2f}, p={p:.4f}, d={d:.2f}")

    print()
    print("=" * 72)
    print("INTER-JUDGE RELIABILITY (Pearson r, paired by task_id+candidate)")
    print("=" * 72)
    for dim in DIMENSIONS:
        gp, cp = [], []
        for cond in gpt.keys():
            g_idx = {(e["task_id"], e["candidate"]): e for e in gpt[cond]}
            c_idx = {(e["task_id"], e["candidate"]): e for e in claude[cond]}
            for k in set(g_idx) & set(c_idx):
                if dim in g_idx[k] and dim in c_idx[k]:
                    gp.append(g_idx[k][dim])
                    cp.append(c_idx[k][dim])
        if len(gp) >= 3:
            if np.std(gp) == 0 or np.std(cp) == 0:
                print(f"  {dim:<14}: undefined (one judge gave constant ratings)")
            else:
                r, p = stats.pearsonr(gp, cp)
                print(f"  {dim:<14}: r = {r:+.3f}, p = {p:.4f}, n = {len(gp)}")


if __name__ == "__main__":
    main()
