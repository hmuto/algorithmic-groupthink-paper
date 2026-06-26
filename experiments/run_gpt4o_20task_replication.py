#!/usr/bin/env python3
"""
GPT-4o replication on the FULL 20-task set (task-parallel).
============================================================
Extends run_gpt4o_replication.py (5 tasks) to all 20 Experiment-1 tasks, to
test whether GPT-4o's directional reference-mode effect reaches significance
with more data. It does not: the effect remains small and non-significant
(individual +7.8% vs. group -0.5%, Welch t = 0.90, p = 0.37, Cohen's d = 0.29,
n = 20). See the paper's cross-model section and Limitations.

Reuses gen_critic_sum / compute_diversity / settings (5 candidates,
5 iterations, gen-critic-sum, temperature 1.0) from run_gpt4o_replication.py.
The 20 tasks are mutually independent, so they run concurrently in a thread
pool; gen_critic_sum / compute_diversity build a fresh OpenAI client per call
and are therefore thread-safe. Each task accumulates its own rows; CSV/JSON
are written once per condition (no file-write races).

Outputs (group == reference_mode "all"):
  results_gpt4o_20task_individual/{results.csv,diversity.json}
  results_gpt4o_20task_all/{results.csv,diversity.json}
  results_gpt4o_20task_summary.json

Requires OPENAI_API_KEY.
"""
import os
import sys
import csv
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_gpt4o_replication as base  # 5-task script; we reuse its building blocks

N_CAND = base.N_CANDIDATES
N_ITER = base.N_ITERATIONS
MAX_WORKERS = 10

# Full 20-task set (4 categories x 5), identical to Experiment 1.
_idea = [
    "Generate 5 novel product ideas that help reduce food waste at home.",
    "Generate 5 new interaction concepts for supporting remote teamwork.",
    "Generate 5 services that use AI to support elderly people living alone.",
    "Generate 5 ideas for playful urban installations using light and sound.",
    "Generate 5 ideas for improving the experience of public transportation.",
]
_reasoning = [
    "Explain why traffic congestion occurs in large cities and propose 3 countermeasures.",
    "Analyze the trade-offs of remote work vs. in-person work for a software team.",
    "Explain the main causes of climate change and propose realistic mitigation steps.",
    "Compare subscription-based and one-time purchase business models.",
    "Analyze risks and benefits of using AI chatbots in customer support.",
]
_summ = [
    "Summarize the key challenges of AI ethics in autonomous driving.",
    "Summarize main usability issues in mobile banking applications.",
    "Summarize the advantages and disadvantages of online education.",
    "Summarize the key properties of human-centered design.",
    "Summarize typical barriers to adopting new technologies in organizations.",
]
_creative = [
    "Write a short story about a city where AI agents and humans co-create art.",
    "Write a dialogue between two AI agents arguing about creativity.",
    "Write a short story about a future classroom using AI tutors.",
    "Write a short story about a day in the life of an AI facilitator.",
    "Write a short story about a researcher studying AI homogenization.",
]
TASKS = []
for i, t in enumerate(_idea):      TASKS.append((f"idea_{i}", "idea", t))
for i, t in enumerate(_reasoning): TASKS.append((f"reasoning_{i}", "reasoning", t))
for i, t in enumerate(_summ):      TASKS.append((f"summ_{i}", "summarization", t))
for i, t in enumerate(_creative):  TASKS.append((f"creative_{i}", "creative_writing", t))


def run_task(task_id, category, task_text, reference_mode):
    """One task: iterations are sequential (iter k depends on k-1). Thread-safe."""
    outputs, diversity, rows = {}, [], []
    for iteration in range(N_ITER):
        outs = []
        for cand in range(N_CAND):
            if iteration == 0:
                previous = []
            elif reference_mode == "individual":
                prev = outputs.get(iteration - 1, [])
                previous = [prev[cand]] if cand < len(prev) else []
            else:  # "all" / group-history
                previous = outputs.get(iteration - 1, [])
            result = base.gen_critic_sum(task_text, previous, cand, iteration)
            outs.append(result)
            rows.append([datetime.now().isoformat(), reference_mode, task_id, category,
                         task_text, iteration, cand, result.replace("\n", "\\n")])
        outputs[iteration] = outs
        if len(outs) == N_CAND:
            div, _ = base.compute_diversity(outs)
            diversity.append(div)
    print(f"  [{reference_mode}] {task_id}: {diversity[0]:.3f} -> {diversity[-1]:.3f}", flush=True)
    return task_id, diversity, rows


def run_condition(reference_mode, outdir):
    os.makedirs(outdir, exist_ok=True)
    div_by_task, all_rows = {}, []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(run_task, tid, cat, txt, reference_mode) for (tid, cat, txt) in TASKS]
        for fut in as_completed(futs):
            tid, diversity, rows = fut.result()
            div_by_task[tid] = diversity
            all_rows.extend(rows)
    with open(os.path.join(outdir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "reference_mode", "task_id", "category", "task",
                    "iteration", "candidate", "final_output"])
        w.writerows(all_rows)
    with open(os.path.join(outdir, "diversity.json"), "w") as f:
        json.dump(div_by_task, f, indent=2)
    return div_by_task


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"GPT-4o ({base.MODEL}) | {len(TASKS)} tasks | {N_CAND} cand | "
          f"{N_ITER} iter | {MAX_WORKERS} workers", flush=True)

    div_ind = run_condition("individual", os.path.join(base_dir, "results_gpt4o_20task_individual"))
    div_grp = run_condition("all", os.path.join(base_dir, "results_gpt4o_20task_all"))

    tasks = sorted(set(div_ind) & set(div_grp))
    fin_i = [div_ind[t][-1] for t in tasks if len(div_ind[t]) >= 2]
    fin_g = [div_grp[t][-1] for t in tasks if len(div_grp[t]) >= 2]

    def agg_rel(d):
        i0 = np.mean([d[t][0] for t in tasks]); iF = np.mean([d[t][-1] for t in tasks])
        return (iF - i0) / i0 * 100

    t_stat, p = stats.ttest_ind(fin_i, fin_g, equal_var=False)
    ps = np.sqrt((np.std(fin_i) ** 2 + np.std(fin_g) ** 2) / 2)
    d = (np.mean(fin_i) - np.mean(fin_g)) / ps if ps > 0 else 0.0
    res = {
        "model": "gpt-4o", "n_tasks": len(tasks),
        "individual_rel_change_pct": round(agg_rel(div_ind), 1),
        "group_rel_change_pct": round(agg_rel(div_grp), 1),
        "welch_t": round(float(t_stat), 3), "p_value": float(p),
        "cohens_d": round(float(d), 3), "significant": bool(p < 0.05),
    }
    print("\n==== GPT-4o 20-TASK RESULT ====", flush=True)
    print(json.dumps(res, indent=2), flush=True)
    json.dump(res, open(os.path.join(base_dir, "results_gpt4o_20task_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
