#!/usr/bin/env python3
"""
Iterate over simulation_questions*.json, ask a local Ollama model to extract
parameters + target metric from each question, run queue_simulation, compare
to ground truth, print metrics, and save plots.

Requires Ollama running locally (default http://127.0.0.1:11434).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from queue_simulation import metric_value, run_simulation

OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"

VALID_METRICS = frozenset(
    {"customers_started", "mean_wait", "median_wait", "max_wait", "utilization"}
)


def load_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["questions"])


def ollama_extract(
    base_url: str,
    model: str,
    introduction: str,
    question: str,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Call Ollama chat API; expect JSON object in assistant message."""
    url = base_url.rstrip("/") + "/api/chat"
    system = (
        "You map word problems to a discrete-event queue simulation. "
        "The model: exponential time between arrivals (mean = MEAN_INTERARRIVAL minutes), "
        "exponential service length (mean = MEAN_SERVICE minutes), "
        "NUM_CLERKS identical parallel servers, simulation stops at SIM_TIME minutes, "
        "RANDOM_SEED fixes Python's random module. "
        "A customer is counted when they START service (acquire a clerk). "
        "Utilization = sum of sampled service durations for those customers / (NUM_CLERKS * SIM_TIME).\n"
        "Reply with ONLY a single JSON object, no markdown fences, with keys: "
        "RANDOM_SEED (int), NUM_CLERKS (int), SIM_TIME (number), "
        "MEAN_INTERARRIVAL (number), MEAN_SERVICE (number), "
        'target_metric (string, one of: "customers_started", "mean_wait", '
        '"median_wait", "max_wait", "utilization") matching what the question asks.'
    )
    user = f"Context:\n{introduction}\n\nQuestion:\n{question}"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    text = raw.get("message", {}).get("content", "").strip()
    return parse_llm_json(text)


def parse_llm_json(text: str) -> dict[str, Any]:
    """Strip optional fences and parse JSON."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return json.loads(t)


def round_like_ground_truth(pred: float, truth: float | int) -> float:
    """Match rounding of truth when it looks like an integer or fixed decimals."""
    if isinstance(truth, int) or (isinstance(truth, float) and truth == int(truth)):
        return round(pred)
    s = f"{truth}"
    if "." in s:
        places = len(s.split(".")[1].rstrip("0")) or 2
        return round(pred, places)
    return round(pred, 4)


def scores_match(pred: float | None, truth: float | int, metric: str) -> bool:
    if pred is None and truth == 0:
        return metric in ("mean_wait", "median_wait", "max_wait")
    if pred is None:
        return False
    rp = round_like_ground_truth(pred, truth)
    if isinstance(truth, int):
        return abs(rp - int(truth)) < 0.5
    return math.isclose(rp, float(truth), rel_tol=0.002, abs_tol=0.02)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ollama + SimPy benchmark over JSON questions")
    ap.add_argument(
        "--json",
        type=Path,
        default=Path("simulation_questions_natural_language.json"),
        help="Questions file with introduction + questions[]",
    )
    ap.add_argument("--model", default="llama3.2", help="Ollama model name")
    ap.add_argument(
        "--ollama-base",
        default="http://127.0.0.1:11434",
        help="Ollama server base URL",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("agent_results"),
        help="Directory for plots and run summary",
    )
    ap.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout per LLM call")
    args = ap.parse_args()

    data = json.loads(args.json.read_text(encoding="utf-8"))
    introduction = data.get("introduction") or data.get("model_notes", "")
    questions = data["questions"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for q in questions:
        qid = q["id"]
        truth = q["answer_numeric"]
        text = q["question"]
        gold_params = q.get("parameters")

        row: dict[str, Any] = {
            "id": qid,
            "truth": truth,
            "gold_parameters": gold_params,
            "error": None,
        }
        try:
            extracted = ollama_extract(
                args.ollama_base,
                args.model,
                introduction,
                text,
                timeout_s=args.timeout,
            )
            row["extracted"] = extracted
            tm = extracted.get("target_metric", "")
            if tm not in VALID_METRICS:
                raise ValueError(f"Invalid target_metric: {tm!r}")

            params = {
                "random_seed": int(extracted["RANDOM_SEED"]),
                "num_clerks": int(extracted["NUM_CLERKS"]),
                "sim_time": float(extracted["SIM_TIME"]),
                "mean_interarrival": float(extracted["MEAN_INTERARRIVAL"]),
                "mean_service": float(extracted["MEAN_SERVICE"]),
            }
            results = run_simulation(**params)
            pred_raw = metric_value(results, tm)
            pred_rounded = (
                round_like_ground_truth(pred_raw, truth)
                if pred_raw is not None
                else None
            )
            row["target_metric"] = tm
            row["predicted_raw"] = pred_raw
            row["predicted_rounded"] = pred_rounded
            row["params_match_gold"] = (
                gold_params is not None
                and int(gold_params["RANDOM_SEED"]) == params["random_seed"]
                and int(gold_params["NUM_CLERKS"]) == params["num_clerks"]
                and math.isclose(
                    float(gold_params["SIM_TIME"]), params["sim_time"], rel_tol=0, abs_tol=1e-6
                )
                and math.isclose(
                    float(gold_params["MEAN_INTERARRIVAL"]),
                    params["mean_interarrival"],
                    rel_tol=0,
                    abs_tol=1e-6,
                )
                and math.isclose(
                    float(gold_params["MEAN_SERVICE"]),
                    params["mean_service"],
                    rel_tol=0,
                    abs_tol=1e-6,
                )
            )
            row["correct"] = scores_match(pred_raw, truth, tm)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            row["error"] = f"network: {e}"
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            row["error"] = str(e)

        rows.append(row)

    # Summary metrics
    ok = [r for r in rows if r.get("correct")]
    failed = [r for r in rows if r.get("error")]
    attempted = [r for r in rows if r.get("error") is None]
    accuracy = len(ok) / len(rows) if rows else 0.0
    param_acc = (
        sum(1 for r in attempted if r.get("params_match_gold")) / len(attempted)
        if attempted
        else 0.0
    )

    errs: list[float] = []
    for r in attempted:
        if r.get("predicted_raw") is None:
            continue
        t = float(r["truth"])
        p = float(r["predicted_raw"])
        if t != 0:
            errs.append(abs(p - t) / abs(t))
        else:
            errs.append(abs(p - t))

    mae = (
        sum(abs(float(r["truth"]) - float(r["predicted_raw"])) for r in attempted if r.get("predicted_raw") is not None)
        / max(1, sum(1 for r in attempted if r.get("predicted_raw") is not None))
    )
    rmse = 0.0
    vals = [
        (float(r["truth"]), float(r["predicted_raw"]))
        for r in attempted
        if r.get("predicted_raw") is not None
    ]
    if vals:
        rmse = math.sqrt(sum((a - b) ** 2 for a, b in vals) / len(vals))

    summary = {
        "questions_total": len(rows),
        "answer_accuracy": accuracy,
        "parameter_extraction_accuracy": param_acc,
        "llm_failures": len(failed),
        "mean_absolute_error_predicted_vs_truth": mae,
        "rmse_predicted_vs_truth": rmse,
        "mean_relative_error_when_truth_nonzero": sum(errs) / len(errs) if errs else None,
    }

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"summary": summary, "per_question": rows}, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))

    # Plots
    ids = [r["id"] for r in rows]
    truths = [float(r["truth"]) for r in rows]
    preds = [
        float(r["predicted_raw"]) if r.get("predicted_raw") is not None else math.nan
        for r in rows
    ]
    colors = ["#2ca02c" if r.get("correct") else "#d62728" for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(ids))
    width = 0.35
    ax.bar([i - width / 2 for i in x], truths, width, label="Ground truth", color="#1f77b4")
    ax.bar([i + width / 2 for i in x], preds, width, label="Simulation (from LLM params)", color=colors, alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Q{i}" for i in ids])
    ax.set_ylabel("Value")
    ax.set_title("Ground truth vs simulation output per question")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "bar_truth_vs_predicted.png", dpi=150)
    plt.close(fig)

    abs_err = [abs(t - p) if not math.isnan(p) else math.nan for t, p in zip(truths, preds)]
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar([f"Q{i}" for i in ids], abs_err, color="#9467bd")
    ax2.set_ylabel("Absolute error")
    ax2.set_title("Absolute error |predicted − truth|")
    fig2.tight_layout()
    fig2.savefig(args.out_dir / "bar_absolute_error.png", dpi=150)
    plt.close(fig2)

    rel_err_pct = []
    for t, p in zip(truths, preds):
        if math.isnan(p):
            rel_err_pct.append(math.nan)
        elif t == 0:
            rel_err_pct.append(0.0 if p == 0 else 100.0)
        else:
            rel_err_pct.append(100.0 * abs(p - t) / abs(t))
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.bar([f"Q{i}" for i in ids], rel_err_pct, color="#8c564b")
    ax3.set_ylabel("Relative error (%)")
    ax3.set_title("Relative error (%)")
    fig3.tight_layout()
    fig3.savefig(args.out_dir / "bar_relative_error_pct.png", dpi=150)
    plt.close(fig3)

    print(f"Wrote {summary_path} and plots under {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
