#!/usr/bin/env python3
"""
Iterate over simulation question JSON, call a local Ollama model, run SimPy,
and score results.

Easy mode: single scenario, numeric answer (simulation_questions_easy.json).
Comparison mode: multiple scenarios A/B/C, pick best; metrics include final-letter
correctness and textual overlap with the reference answer.

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
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from queue_simulation import metric_value, run_simulation

VALID_METRICS = frozenset(
    {"customers_started", "mean_wait", "median_wait", "max_wait", "utilization"}
)
SCENARIO_LABELS = ("A", "B", "C")


def is_comparison_dataset(data: dict[str, Any]) -> bool:
    if data.get("difficulty") == "comparison":
        return True
    qs = data.get("questions") or []
    return bool(qs) and "scenarios" in qs[0]


def parse_llm_json(text: str) -> dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return json.loads(t)


def ollama_chat_json(
    base_url: str,
    model: str,
    system: str,
    user: str,
    timeout_s: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/api/chat"
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


def ollama_extract_easy(
    base_url: str,
    model: str,
    introduction: str,
    question: str,
    timeout_s: float,
) -> dict[str, Any]:
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
    return ollama_chat_json(base_url, model, system, user, timeout_s)


def ollama_extract_comparison(
    base_url: str,
    model: str,
    introduction: str,
    question: str,
    criterion_hint: str,
    timeout_s: float,
) -> dict[str, Any]:
    system = (
        "You analyze multi-scenario queueing word problems. Same simulation as above for EACH scenario "
        "label A, B, and C: exponential interarrival (mean MEAN_INTERARRIVAL), exponential service "
        "(mean MEAN_SERVICE), NUM_CLERKS parallel clerks, stop at SIM_TIME, RNG seed RANDOM_SEED.\n"
        "You must: (1) extract the five parameters for scenarios A, B, and C exactly as stated; "
        "(2) write comparison_text, a short coherent paragraph in plain English comparing the three "
        "runs and stating which scenario best meets the goal; "
        "(3) set best_scenario to exactly one of \"A\", \"B\", or \"C\" per your reasoning.\n"
        f"The stated goal in this item: {criterion_hint}\n"
        "Reply with ONLY one JSON object with keys: "
        'best_scenario ("A"|"B"|"C"), comparison_text (string), '
        'scenarios (object with keys A, B, C each mapping to an object with '
        "RANDOM_SEED, NUM_CLERKS, SIM_TIME, MEAN_INTERARRIVAL, MEAN_SERVICE)."
    )
    user = f"Context:\n{introduction}\n\nQuestion:\n{question}"
    return ollama_chat_json(base_url, model, system, user, timeout_s)


def normalize_scenario_key(k: str) -> str:
    u = str(k).strip().upper()
    if u in SCENARIO_LABELS:
        return u
    raise ValueError(f"Invalid scenario key: {k!r}")


def params_equal_gold(gold: dict[str, Any], extracted: dict[str, Any]) -> bool:
    return (
        int(gold["RANDOM_SEED"]) == int(extracted["RANDOM_SEED"])
        and int(gold["NUM_CLERKS"]) == int(extracted["NUM_CLERKS"])
        and math.isclose(float(gold["SIM_TIME"]), float(extracted["SIM_TIME"]), rel_tol=0, abs_tol=1e-6)
        and math.isclose(
            float(gold["MEAN_INTERARRIVAL"]),
            float(extracted["MEAN_INTERARRIVAL"]),
            rel_tol=0,
            abs_tol=1e-6,
        )
        and math.isclose(
            float(gold["MEAN_SERVICE"]),
            float(extracted["MEAN_SERVICE"]),
            rel_tol=0,
            abs_tol=1e-6,
        )
    )


def pick_winner_from_sim_results(
    scenario_results: dict[str, dict[str, Any]],
    criterion: dict[str, Any],
) -> str:
    metric = criterion["metric"]
    objective = criterion["objective"]
    sub = criterion.get("subject_to")
    feasible = list(scenario_results.keys())
    if sub:
        thr = int(sub["customers_started_gte"])
        feasible = [k for k in feasible if scenario_results[k]["customers_started"] >= thr]
    if not feasible:
        return ""

    def sort_key(k: str) -> float:
        v = scenario_results[k].get(metric)
        if v is None:
            return float("inf") if objective == "minimize" else float("-inf")
        return float(v)

    if objective == "minimize":
        return min(feasible, key=sort_key)
    return max(feasible, key=sort_key)


def token_jaccard(text_a: str, text_b: str) -> float:
    ta = set(re.findall(r"\w+", text_a.lower()))
    tb = set(re.findall(r"\w+", text_b.lower()))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def sequence_similarity(text_a: str, text_b: str) -> float:
    return SequenceMatcher(None, text_a.lower().strip(), text_b.lower().strip()).ratio()


def mentions_gold_scenario(comparison_text: str, gold_letter: str) -> bool:
    """Heuristic: reference letter appears as a scenario label in prose."""
    if not comparison_text or not gold_letter:
        return False
    g = gold_letter.strip().upper()
    if g not in SCENARIO_LABELS:
        return False
    patterns = [
        rf"\bscenario\s+{g}\b",
        rf"\boption\s+{g}\b",
        rf"\bplan\s+{g}\b",
        rf"\bcase\s+{g}\b",
        rf"\({g}\)",
        rf"\[{g}\]",
    ]
    low = comparison_text.lower()
    for p in patterns:
        if re.search(p, low, re.IGNORECASE):
            return True
    return False


def round_like_ground_truth(pred: float, truth: float | int) -> float:
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


def run_easy_mode(
    introduction: str,
    questions: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for q in questions:
        qid = q["id"]
        truth = q["answer_numeric"]
        text = q["question"]
        gold_params = q.get("parameters")

        row: dict[str, Any] = {
            "id": qid,
            "mode": "easy",
            "truth": truth,
            "gold_parameters": gold_params,
            "error": None,
        }
        try:
            extracted = ollama_extract_easy(
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
            row["target_metric"] = tm
            row["predicted_raw"] = pred_raw
            row["predicted_rounded"] = (
                round_like_ground_truth(pred_raw, truth) if pred_raw is not None else None
            )
            row["params_match_gold"] = gold_params is not None and params_equal_gold(
                gold_params, extracted
            )
            row["correct"] = scores_match(pred_raw, truth, tm)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            row["error"] = f"network: {e}"
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            row["error"] = str(e)

        rows.append(row)

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
        errs.append(abs(p - t) / abs(t) if t != 0 else abs(p - t))

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
        "question_mode": "easy",
        "questions_total": len(rows),
        "answer_accuracy": accuracy,
        "numeric_answer_accuracy": accuracy,
        "parameter_extraction_accuracy": param_acc,
        "llm_failures": len(failed),
        "mean_absolute_error_predicted_vs_truth": mae,
        "rmse_predicted_vs_truth": rmse,
        "mean_relative_error_when_truth_nonzero": sum(errs) / len(errs) if errs else None,
    }
    return summary, rows


def run_comparison_mode(
    introduction: str,
    questions: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for q in questions:
        qid = q["id"]
        gold_best = str(q["best_scenario"]).strip().upper()
        ref_answer = q.get("answer", "")
        criterion = q["criterion"]
        gold_scenarios = q["scenarios"]
        crit_hint = json.dumps(criterion)

        row: dict[str, Any] = {
            "id": qid,
            "mode": "comparison",
            "gold_best_scenario": gold_best,
            "error": None,
        }
        try:
            extracted = ollama_extract_comparison(
                args.ollama_base,
                args.model,
                introduction,
                q["question"],
                crit_hint,
                timeout_s=args.timeout,
            )
            row["extracted"] = extracted

            comp_text = str(extracted.get("comparison_text", "") or "")
            row["comparison_text"] = comp_text

            llm_best = str(extracted.get("best_scenario", "")).strip().upper()
            if llm_best not in SCENARIO_LABELS:
                raise ValueError(f"Invalid best_scenario: {extracted.get('best_scenario')!r}")
            row["llm_best_scenario"] = llm_best

            scen_raw = extracted.get("scenarios") or {}
            by_label: dict[str, Any] = {}
            for k, v in scen_raw.items():
                by_label[normalize_scenario_key(k)] = v
            sim_by_label: dict[str, dict[str, Any]] = {}
            param_matches: dict[str, bool] = {}
            for label in SCENARIO_LABELS:
                if label not in by_label:
                    raise KeyError(f"Missing scenario {label} in LLM output")
                block = by_label[label]
                gold = gold_scenarios[label]
                param_matches[label] = params_equal_gold(gold, block)
                sim_by_label[label] = run_simulation(
                    int(block["RANDOM_SEED"]),
                    int(block["NUM_CLERKS"]),
                    float(block["SIM_TIME"]),
                    float(block["MEAN_INTERARRIVAL"]),
                    float(block["MEAN_SERVICE"]),
                )

            row["parameter_match_by_scenario"] = param_matches
            row["parameter_extraction_accuracy"] = sum(param_matches.values()) / len(
                param_matches
            )

            winner_sim = pick_winner_from_sim_results(sim_by_label, criterion)
            row["winner_from_llm_parameter_simulation"] = winner_sim
            row["winner_from_sim_matches_gold"] = winner_sim == gold_best

            row["final_answer_correct"] = llm_best == gold_best

            row["textual_token_jaccard_vs_reference"] = token_jaccard(comp_text, ref_answer)
            row["textual_sequence_similarity_vs_reference"] = sequence_similarity(
                comp_text, ref_answer
            )
            row["comparison_mentions_gold_scenario"] = mentions_gold_scenario(
                comp_text, gold_best
            )

            row["metrics_from_llm_params"] = {
                lab: {
                    "customers_started": sim_by_label[lab]["customers_started"],
                    "mean_wait": None
                    if sim_by_label[lab]["mean_wait"] is None
                    else round(sim_by_label[lab]["mean_wait"], 4),
                    "median_wait": None
                    if sim_by_label[lab]["median_wait"] is None
                    else round(sim_by_label[lab]["median_wait"], 4),
                    "max_wait": None
                    if sim_by_label[lab]["max_wait"] is None
                    else round(sim_by_label[lab]["max_wait"], 4),
                    "utilization": round(sim_by_label[lab]["utilization"], 4),
                }
                for lab in SCENARIO_LABELS
            }
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            row["error"] = f"network: {e}"
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            row["error"] = str(e)

        rows.append(row)

    failed = [r for r in rows if r.get("error")]
    attempted = [r for r in rows if r.get("error") is None]

    final_ok = [r for r in attempted if r.get("final_answer_correct")]
    sim_ok = [r for r in attempted if r.get("winner_from_sim_matches_gold")]

    jaccards = [r["textual_token_jaccard_vs_reference"] for r in attempted]
    seqs = [r["textual_sequence_similarity_vs_reference"] for r in attempted]
    mentions = [r for r in attempted if r.get("comparison_mentions_gold_scenario")]

    param_rates = [r["parameter_extraction_accuracy"] for r in attempted]

    summary = {
        "question_mode": "comparison",
        "questions_total": len(rows),
        "llm_failures": len(failed),
        "final_answer_correctness_rate": len(final_ok) / len(rows) if rows else 0.0,
        "winner_from_llm_params_simulation_matches_gold_rate": len(sim_ok) / len(rows)
        if rows
        else 0.0,
        "mean_parameter_extraction_accuracy_across_scenarios": sum(param_rates)
        / len(param_rates)
        if param_rates
        else None,
        "textual_comparison_mean_token_jaccard_vs_reference_answer": sum(jaccards)
        / len(jaccards)
        if jaccards
        else None,
        "textual_comparison_mean_sequence_similarity_vs_reference_answer": sum(seqs)
        / len(seqs)
        if seqs
        else None,
        "comparison_text_mentions_gold_scenario_rate": len(mentions) / len(rows)
        if rows
        else 0.0,
    }
    return summary, rows


def plot_easy(out_dir: Path, rows: list[dict[str, Any]]) -> None:
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
    ax.bar(
        [i + width / 2 for i in x],
        preds,
        width,
        label="Simulation (from LLM params)",
        color=colors,
        alpha=0.85,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Q{i}" for i in ids])
    ax.set_ylabel("Value")
    ax.set_title("Easy mode: ground truth vs simulation output")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "bar_truth_vs_predicted.png", dpi=150)
    plt.close(fig)

    abs_err = [abs(t - p) if not math.isnan(p) else math.nan for t, p in zip(truths, preds)]
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar([f"Q{i}" for i in ids], abs_err, color="#9467bd")
    ax2.set_ylabel("Absolute error")
    ax2.set_title("Easy mode: |predicted − truth|")
    fig2.tight_layout()
    fig2.savefig(out_dir / "bar_absolute_error.png", dpi=150)
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
    ax3.set_title("Easy mode: relative error (%)")
    fig3.tight_layout()
    fig3.savefig(out_dir / "bar_relative_error_pct.png", dpi=150)
    plt.close(fig3)


_SCENARIO_BG = {"A": "#c6dbef", "B": "#c7e9c0", "C": "#fdd0a2"}
_MATCH_BG = {"yes": "#d4edda", "no": "#f8d7da", "err": "#e2e3e5"}


def plot_comparison(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    n = len(rows)
    llm_ok = sum(1 for r in rows if r.get("error") is None and r.get("final_answer_correct"))
    sim_ok = sum(
        1 for r in rows if r.get("error") is None and r.get("winner_from_sim_matches_gold")
    )
    ok_rows = sum(1 for r in rows if r.get("error") is None)
    llm_rate = llm_ok / n if n else 0.0
    sim_rate = sim_ok / n if n else 0.0

    fig = plt.figure(figsize=(11, max(5.0, 0.55 * n + 2.2)))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 6], hspace=0.22)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_tbl.axis("off")

    y_b = [0, 1]
    bars = ax_bar.barh(
        y_b,
        [llm_rate, sim_rate],
        height=0.55,
        color=["#4c72b0", "#dd8452"],
        edgecolor="white",
        linewidth=1,
    )
    ax_bar.set_yticks(y_b)
    ax_bar.set_yticklabels(
        [
            f'LLM final = gold  ({llm_ok}/{n})',
            f'Sim winner = gold ({sim_ok}/{n})',
        ],
        fontsize=10,
    )
    ax_bar.set_xlim(0, 1.05)
    ax_bar.set_xlabel("Fraction correct")
    ax_bar.set_title("Comparison mode — summary", fontsize=12, fontweight="bold", loc="left")
    ax_bar.axvline(1.0, color="#ccc", linestyle="--", linewidth=0.8)
    for rect, val in zip(bars, [llm_rate, sim_rate]):
        ax_bar.text(
            min(val + 0.02, 0.98),
            rect.get_y() + rect.get_height() / 2,
            f"{val:.0%}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    col_labels = [
        "Question",
        "Gold\n(correct)",
        "LLM\nchoice",
        "Sim winner\n(LLM params)",
        "LLM\nmatch?",
        "Sim\nmatch?",
    ]
    cell_text: list[list[str]] = []
    cell_colours: list[list[str]] = []
    header_bg = "#343a40"
    header_fg = "white"

    for r in rows:
        qid = f"Q{r['id']}"
        if r.get("error"):
            cell_text.append([qid, "—", "—", "—", "error", "error"])
            cell_colours.append(["#e9ecef"] * 6)
            continue
        g = str(r.get("gold_best_scenario", "")).strip().upper()
        llm = str(r.get("llm_best_scenario", "")).strip().upper()
        sim = str(r.get("winner_from_llm_parameter_simulation", "")).strip().upper()
        if g not in _SCENARIO_BG:
            m_llm = m_sim = "—"
        else:
            m_llm = "Yes" if llm == g else ("No" if llm in _SCENARIO_BG else "—")
            m_sim = "Yes" if sim == g else ("No" if sim in _SCENARIO_BG else "—")
        cell_text.append([qid, g, llm, sim, m_llm, m_sim])
        cell_colours.append(
            [
                "#ffffff",
                _SCENARIO_BG.get(g, "#eeeeee"),
                _SCENARIO_BG.get(llm, "#eeeeee"),
                _SCENARIO_BG.get(sim, "#eeeeee"),
                _MATCH_BG["yes" if m_llm == "Yes" else ("no" if m_llm == "No" else "err")],
                _MATCH_BG["yes" if m_sim == "Yes" else ("no" if m_sim == "No" else "err")],
            ]
        )

    table = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colours,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.85)
    for j, _ in enumerate(col_labels):
        table[(0, j)].set_facecolor(header_bg)
        table[(0, j)].get_text().set_color(header_fg)
        table[(0, j)].get_text().set_fontweight("bold")

    legend_ax = fig.add_axes([0.02, 0.01, 0.4, 0.04])
    legend_ax.axis("off")
    patches = [
        plt.Rectangle((0, 0), 1, 1, fc=_SCENARIO_BG["A"], ec="gray"),
        plt.Rectangle((0, 0), 1, 1, fc=_SCENARIO_BG["B"], ec="gray"),
        plt.Rectangle((0, 0), 1, 1, fc=_SCENARIO_BG["C"], ec="gray"),
    ]
    legend_ax.legend(
        patches,
        ["Scenario A", "Scenario B", "Scenario C"],
        ncol=3,
        loc="center",
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        "Gold vs model answers (read across each row)",
        fontsize=13,
        y=0.995,
        fontweight="bold",
    )
    fig.text(
        0.99,
        0.01,
        f"Evaluated rows: {ok_rows}/{n} without LLM parse errors",
        ha="right",
        fontsize=8,
        style="italic",
        color="#555",
    )
    plt.subplots_adjust(top=0.93, bottom=0.06)
    fig.savefig(out_dir / "comparison_results_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Second figure: track chart — one row per question, A/B/C on X, markers for gold vs LLM vs sim
    fig2, ax2 = plt.subplots(figsize=(9, max(3.5, 0.55 * n + 1.2)))
    x_labels = ["A", "B", "C"]
    x_pos = [0, 1, 2]
    y_step = 1.0
    for i, r in enumerate(rows):
        y = i * y_step
        ax2.axhline(y, color="#e9ecef", linewidth=1, zorder=0)
        ax2.set_xlim(-0.5, 2.5)
        if r.get("error"):
            ax2.text(1, y, f"Q{r['id']}: error", ha="center", va="center", color="#6c757d")
            continue
        g = str(r.get("gold_best_scenario", "")).strip().upper()
        llm = str(r.get("llm_best_scenario", "")).strip().upper()
        sim = str(r.get("winner_from_llm_parameter_simulation", "")).strip().upper()
        gx = x_pos[x_labels.index(g)] if g in x_labels else None
        lx = x_pos[x_labels.index(llm)] if llm in x_labels else None
        sx = x_pos[x_labels.index(sim)] if sim in x_labels else None
        ax2.text(-0.48, y, f"Q{r['id']}", ha="right", va="center", fontsize=10, fontweight="bold")
        if gx is not None:
            ax2.scatter(
                gx,
                y,
                s=140,
                c="#2171b5",
                marker="o",
                zorder=3,
                edgecolors="white",
                linewidths=1.2,
            )
        if lx is not None:
            ax2.scatter(
                lx,
                y + 0.08,
                s=120,
                c="#d94801",
                marker="s",
                zorder=3,
                edgecolors="white",
                linewidths=1.2,
            )
        if sx is not None:
            ax2.scatter(
                sx,
                y - 0.08,
                s=120,
                c="#238b45",
                marker="^",
                zorder=3,
                edgecolors="white",
                linewidths=1.2,
            )
        if gx is not None and lx is not None:
            ax2.plot([gx, lx], [y, y + 0.08], color="#adb5bd", linewidth=1, zorder=1, linestyle=":")

    ax2.set_yticks([])
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, fontsize=12, fontweight="bold")
    ax2.set_xlabel("Scenario", fontsize=11)
    ax2.set_title(
        "Comparison mode — alignment chart (○ gold, □ LLM choice, △ sim winner)",
        fontsize=11,
        pad=12,
    )
    leg = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#2171b5",
            markersize=10,
            label="Gold (correct)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#d94801",
            markersize=9,
            label="LLM choice",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="#238b45",
            markersize=10,
            label="Sim winner (LLM params)",
        ),
    ]
    ax2.legend(handles=leg, loc="upper right", fontsize=9)
    ax2.set_ylim(-0.6, (n - 1) * y_step + 0.6)
    fig2.tight_layout()
    fig2.savefig(out_dir / "comparison_alignment_tracks.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    labels = [f"Q{r['id']}" for r in rows]
    jac = [
        r.get("textual_token_jaccard_vs_reference") or 0.0
        if r.get("error") is None
        else math.nan
        for r in rows
    ]
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(labels, jac, color="#2ca02c")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Token Jaccard vs reference answer")
    ax2.set_title("Comparison mode: textual overlap with reference answer")
    fig2.tight_layout()
    fig2.savefig(out_dir / "bar_comparison_text_jaccard.png", dpi=150)
    plt.close(fig2)

    seq = [
        r.get("textual_sequence_similarity_vs_reference") or 0.0
        if r.get("error") is None
        else math.nan
        for r in rows
    ]
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.bar(labels, seq, color="#9467bd")
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Sequence similarity ratio")
    ax3.set_title("Comparison mode: difflib ratio vs reference answer")
    fig3.tight_layout()
    fig3.savefig(out_dir / "bar_comparison_text_sequence_similarity.png", dpi=150)
    plt.close(fig3)

    param_acc = [
        r.get("parameter_extraction_accuracy") or 0.0 if r.get("error") is None else math.nan
        for r in rows
    ]
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.bar(labels, param_acc, color="#8c564b")
    ax4.set_ylim(0, 1)
    ax4.set_ylabel("Fraction of A/B/C params exact")
    ax4.set_title("Comparison mode: parameter extraction (all five fields per scenario)")
    fig4.tight_layout()
    fig4.savefig(out_dir / "bar_comparison_param_accuracy.png", dpi=150)
    plt.close(fig4)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ollama + SimPy benchmark over JSON questions")
    ap.add_argument(
        "--json",
        type=Path,
        default=Path("simulation_questions_easy.json"),
        help="Questions file",
    )
    ap.add_argument(
        "--mode",
        choices=("auto", "easy", "comparison"),
        default="auto",
        help="Question format (default: infer from JSON)",
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

    use_comparison = args.mode == "comparison" or (
        args.mode == "auto" and is_comparison_dataset(data)
    )

    if use_comparison:
        summary, rows = run_comparison_mode(introduction, questions, args)
        plot_comparison(args.out_dir, rows)
    else:
        summary, rows = run_easy_mode(introduction, questions, args)
        plot_easy(args.out_dir, rows)

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"summary": summary, "per_question": rows}, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path} and plots under {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
