#!/usr/bin/env python3
"""
Cross-judge replication: re-evaluate quality outputs with Claude-Haiku 4.5.

Loads the GPT-4o-mini evaluations from data/quality_eval/gpt4o-mini_judge.json,
then re-evaluates the same (task_id, candidate) outputs with Claude-Haiku 4.5
as an independent LLM judge from a different model family.

Output: data/quality_eval/claude-haiku-4-5_judge.json

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python analysis/quality_evaluation_claude.py
"""

import os
import csv
import json
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONDITION_TO_CSV = {
    "Baseline (Group)": os.path.join(BASE_DIR, "data/raw_outputs/exp1_20tasks_all/results.csv"),
    "Individual":        os.path.join(BASE_DIR, "data/raw_outputs/exp1_20tasks_individual/results.csv"),
    "Diversity Prompt":  os.path.join(BASE_DIR, "data/raw_outputs/exp3_countermeasures/diversity_prompt.csv"),
}

GPT4O_MINI_RESULTS = os.path.join(BASE_DIR, "data/quality_eval/gpt4o-mini_judge.json")
OUT_PATH = os.path.join(BASE_DIR, "data/quality_eval/claude-haiku-4-5_judge.json")

JUDGE_MODEL = "claude-haiku-4-5-20251001"

EVALUATION_PROMPT = """You are an expert evaluator assessing the quality of creative ideation responses.

Task: {task}

Response to evaluate:
{response}

Please rate this response on the following criteria (1-5 scale):

1. **Creativity** (1-5): How novel and innovative are the ideas? Do they go beyond obvious solutions?
2. **Relevance** (1-5): How well do the ideas address the given task?
3. **Practicality** (1-5): How feasible are the ideas to implement?
4. **Completeness** (1-5): How thorough and well-developed are the ideas?
5. **Clarity** (1-5): How clear and well-articulated is the response?

Return your evaluation as a JSON object with the following format:
{{
    "creativity": <score>,
    "relevance": <score>,
    "practicality": <score>,
    "completeness": <score>,
    "clarity": <score>,
    "overall": <average of all scores>,
    "brief_rationale": "<1-2 sentence explanation>"
}}

Return ONLY the JSON object, no other text."""


def load_originals(csv_path):
    """Index final-iteration outputs by (task_id, candidate)."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["task_id"] = int(row["task_id"])
            row["candidate"] = int(row["candidate"])
            row["iteration"] = int(row["iteration"])
            row["final_output"] = row["final_output"].replace("\\n", "\n")
            rows.append(row)
    max_iter = max(r["iteration"] for r in rows)
    return {(r["task_id"], r["candidate"]): r for r in rows if r["iteration"] == max_iter}


def call_claude(client, task, response):
    prompt = EVALUATION_PROMPT.format(task=task, response=response)
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=500,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    content = msg.content[0].text.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content.strip())


def main():
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    with open(GPT4O_MINI_RESULTS) as f:
        gpt4mini_evals = json.load(f)

    originals = {cond: load_originals(p) for cond, p in CONDITION_TO_CSV.items()}

    claude_evals = {}
    for cond, evals in gpt4mini_evals.items():
        print(f"=== {cond}: {len(evals)} samples ===")
        claude_evals[cond] = []
        for i, e in enumerate(evals):
            key = (e["task_id"], e["candidate"])
            if key not in originals[cond]:
                print(f"  [SKIP] {key} not found in CSV")
                continue
            original = originals[cond][key]
            try:
                result = call_claude(client, original["task"], original["final_output"])
            except Exception as ex:
                print(f"  [ERROR] {ex}")
                continue
            result["condition"] = cond
            result["task_id"] = e["task_id"]
            result["candidate"] = e["candidate"]
            claude_evals[cond].append(result)
            print(f"  [{i+1}/{len(evals)}] creativity={result['creativity']}, overall={result.get('overall', '-')}")
            time.sleep(0.5)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(claude_evals, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
