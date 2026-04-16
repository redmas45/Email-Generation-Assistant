import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from groq import Groq

from evaluate import DEFAULT_MODEL_1, DEFAULT_MODEL_2, DEFAULT_JUDGE_MODEL, compute_average_scores, run_evaluation


load_dotenv('.env')

DEFAULT_SCENARIOS = "scenarios.json"
DEFAULT_COMPARISON_CSV = "comparison_results.csv"
DEFAULT_SUMMARY_CSV = "comparison_summary.csv"
DEFAULT_SCENARIO_SCORECARD = "scenario_scorecard.csv"
DEFAULT_REPORT = "comparison_report.md"
DEFAULT_SAVED_MAILS = "saved_mails_comparison.json"


def save_comparison_rows(rows: List[Dict[str, Any]], output_csv: str | Path) -> None:
    fieldnames = [
        "scenario_id",
        "model",
        "tone",
        "target_word_limit",
        "generated_word_count",
        "facts_covered",
        "facts_required",
        "fact_recall_score",
        "tone_accuracy_score",
        "clarity_score",
        "average_score",
        "intent",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def save_metric_summary_csv(
    output_csv: str | Path,
    model_1: str,
    model_2: str,
    averages_1: Dict[str, float],
    averages_2: Dict[str, float],
) -> None:
    metric_map = {
        "Fact Recall": "fact_recall_score",
        "Tone Accuracy": "tone_accuracy_score",
        "Clarity": "clarity_score",
        "Overall Average": "average_score",
        "Avg Generated Words": "generated_word_count",
        "Avg Target Words": "target_word_limit",
    }

    rows = []
    for label, key in metric_map.items():
        score_1 = averages_1[key]
        score_2 = averages_2[key]
        if abs(score_1 - score_2) <= 0.01:
            winner = "tie"
        elif score_1 > score_2:
            winner = model_1
        else:
            winner = model_2

        rows.append(
            {
                "metric": label,
                f"{model_1}_score": score_1,
                f"{model_2}_score": score_2,
                "delta_model_2_minus_model_1": round(score_2 - score_1, 4),
                "winner": winner,
            }
        )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                f"{model_1}_score",
                f"{model_2}_score",
                "delta_model_2_minus_model_1",
                "winner",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_scenario_scorecard_csv(
    rows_1: List[Dict[str, Any]],
    rows_2: List[Dict[str, Any]],
    model_1: str,
    model_2: str,
    output_csv: str | Path,
) -> None:
    by_scenario_1 = {str(row["scenario_id"]): row for row in rows_1}
    by_scenario_2 = {str(row["scenario_id"]): row for row in rows_2}
    scenario_ids = sorted(by_scenario_1.keys())

    fieldnames = [
        "scenario_id",
        "tone",
        "target_word_limit",
        f"{model_1}_generated_words",
        f"{model_2}_generated_words",
        f"{model_1}_fact_recall",
        f"{model_2}_fact_recall",
        f"{model_1}_tone_accuracy",
        f"{model_2}_tone_accuracy",
        f"{model_1}_clarity",
        f"{model_2}_clarity",
        f"{model_1}_average",
        f"{model_2}_average",
        "winner_by_average",
    ]

    output_rows = []
    for scenario_id in scenario_ids:
        left = by_scenario_1[scenario_id]
        right = by_scenario_2[scenario_id]
        left_avg = float(left["average_score"])
        right_avg = float(right["average_score"])

        if abs(left_avg - right_avg) <= 0.01:
            winner = "tie"
        elif left_avg > right_avg:
            winner = model_1
        else:
            winner = model_2

        output_rows.append(
            {
                "scenario_id": scenario_id,
                "tone": left["tone"],
                "target_word_limit": left["target_word_limit"],
                f"{model_1}_generated_words": left["generated_word_count"],
                f"{model_2}_generated_words": right["generated_word_count"],
                f"{model_1}_fact_recall": left["fact_recall_score"],
                f"{model_2}_fact_recall": right["fact_recall_score"],
                f"{model_1}_tone_accuracy": left["tone_accuracy_score"],
                f"{model_2}_tone_accuracy": right["tone_accuracy_score"],
                f"{model_1}_clarity": left["clarity_score"],
                f"{model_2}_clarity": right["clarity_score"],
                f"{model_1}_average": left["average_score"],
                f"{model_2}_average": right["average_score"],
                "winner_by_average": winner,
            }
        )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def save_saved_mails_json(rows: List[Dict[str, Any]], output_path: str | Path) -> None:
    payload = [
        {
            "scenario_id": row["scenario_id"],
            "model": row["model"],
            "target_word_limit": row["target_word_limit"],
            "generated_word_count": row["generated_word_count"],
            "email": row["generated_email"],
        }
        for row in rows
    ]

    import json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def winner_line(metric_label: str, score_1: float, model_1: str, score_2: float, model_2: str) -> str:
    if abs(score_1 - score_2) <= 0.01:
        return f"- {metric_label}: Tie ({score_1:.4f} vs {score_2:.4f})."
    if score_1 > score_2:
        return f"- {metric_label}: {model_1} wins ({score_1:.4f} vs {score_2:.4f})."
    return f"- {metric_label}: {model_2} wins ({score_2:.4f} vs {score_1:.4f})."


def ascii_bar(score: float, width: int = 20) -> str:
    filled = int(round(score * width))
    return "#" * filled + "-" * (width - filled)


def write_report(
    report_path: str | Path,
    model_1: str,
    model_2: str,
    averages_1: Dict[str, float],
    averages_2: Dict[str, float],
    scenario_count: int,
    random_word_limits: bool,
    word_min: int,
    word_max: int,
) -> None:
    lines = [
        "# Model Comparison Report",
        "",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Scenario count: {scenario_count}",
        f"- Model 1: `{model_1}`",
        f"- Model 2: `{model_2}`",
        f"- Word limits: {'Randomized' if random_word_limits else 'Scenario-defined'}",
    ]

    if random_word_limits:
        lines.append(f"- Random range used: {word_min} to {word_max} words")

    lines.extend(
        [
            "",
            "## Average Scores",
            "",
            "| Metric | Model 1 | Model 2 |",
            "|---|---:|---:|",
            f"| Fact Recall | {averages_1['fact_recall_score']:.4f} | {averages_2['fact_recall_score']:.4f} |",
            f"| Tone Accuracy | {averages_1['tone_accuracy_score']:.4f} | {averages_2['tone_accuracy_score']:.4f} |",
            f"| Clarity | {averages_1['clarity_score']:.4f} | {averages_2['clarity_score']:.4f} |",
            f"| Overall Average | {averages_1['average_score']:.4f} | {averages_2['average_score']:.4f} |",
            f"| Avg Generated Words | {averages_1['generated_word_count']:.1f} | {averages_2['generated_word_count']:.1f} |",
            f"| Avg Target Words | {averages_1['target_word_limit']:.1f} | {averages_2['target_word_limit']:.1f} |",
            "",
            "## Quick Visual (ASCII)",
            "",
            f"- Fact Recall:   {model_1} [{ascii_bar(averages_1['fact_recall_score'])}] {averages_1['fact_recall_score']:.2f}",
            f"                 {model_2} [{ascii_bar(averages_2['fact_recall_score'])}] {averages_2['fact_recall_score']:.2f}",
            f"- Tone Accuracy: {model_1} [{ascii_bar(averages_1['tone_accuracy_score'])}] {averages_1['tone_accuracy_score']:.2f}",
            f"                 {model_2} [{ascii_bar(averages_2['tone_accuracy_score'])}] {averages_2['tone_accuracy_score']:.2f}",
            f"- Clarity:       {model_1} [{ascii_bar(averages_1['clarity_score'])}] {averages_1['clarity_score']:.2f}",
            f"                 {model_2} [{ascii_bar(averages_2['clarity_score'])}] {averages_2['clarity_score']:.2f}",
            f"- Overall:       {model_1} [{ascii_bar(averages_1['average_score'])}] {averages_1['average_score']:.2f}",
            f"                 {model_2} [{ascii_bar(averages_2['average_score'])}] {averages_2['average_score']:.2f}",
            "",
            "## Metric Winners",
            winner_line("Fact Recall", averages_1["fact_recall_score"], model_1, averages_2["fact_recall_score"], model_2),
            winner_line("Tone Accuracy", averages_1["tone_accuracy_score"], model_1, averages_2["tone_accuracy_score"], model_2),
            winner_line("Clarity", averages_1["clarity_score"], model_1, averages_2["clarity_score"], model_2),
            winner_line("Overall", averages_1["average_score"], model_1, averages_2["average_score"], model_2),
            "",
            "## Recommendation",
        ]
    )

    if averages_1["average_score"] > averages_2["average_score"] + 0.01:
        lines.append(f"- Recommend `{model_1}` for production based on higher overall score.")
    elif averages_2["average_score"] > averages_1["average_score"] + 0.01:
        lines.append(f"- Recommend `{model_2}` for production based on higher overall score.")
    else:
        lines.append("- Both models are close overall; choose based on latency/cost constraints.")

    lines.extend(
        [
            "",
            "## Notes",
            "- Both models used the same scenarios and scoring pipeline.",
            "- Tone was rated by an LLM judge with a 0-1 score prompt.",
            "- For visual charts, open comparison_summary.csv and scenario_scorecard.csv in Excel/Google Sheets.",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_model_comparison(
    scenarios_path: str | Path,
    model_1: str,
    model_2: str,
    judge_model: str,
    output_csv: str | Path,
    summary_csv: str | Path,
    scorecard_csv: str | Path,
    report_path: str | Path,
    saved_mails_path: str | Path,
    random_word_limits: bool,
    word_min: int,
    word_max: int,
    seed: int,
) -> Dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your environment or .env file.")

    client = Groq(api_key=api_key)

    rows_1 = run_evaluation(
        scenarios_path=scenarios_path,
        model_name=model_1,
        groq_client=client,
        judge_model=judge_model,
        random_word_limits=random_word_limits,
        word_min=word_min,
        word_max=word_max,
        seed=seed,
    )
    rows_2 = run_evaluation(
        scenarios_path=scenarios_path,
        model_name=model_2,
        groq_client=client,
        judge_model=judge_model,
        random_word_limits=random_word_limits,
        word_min=word_min,
        word_max=word_max,
        seed=seed,
    )

    combined = rows_1 + rows_2
    save_comparison_rows(combined, output_csv)
    save_saved_mails_json(combined, saved_mails_path)

    averages_1 = compute_average_scores(rows_1)
    averages_2 = compute_average_scores(rows_2)

    save_metric_summary_csv(summary_csv, model_1, model_2, averages_1, averages_2)
    save_scenario_scorecard_csv(rows_1, rows_2, model_1, model_2, scorecard_csv)
    write_report(
        report_path=report_path,
        model_1=model_1,
        model_2=model_2,
        averages_1=averages_1,
        averages_2=averages_2,
        scenario_count=len(rows_1),
        random_word_limits=random_word_limits,
        word_min=word_min,
        word_max=word_max,
    )

    return {
        "rows_model_1": rows_1,
        "rows_model_2": rows_2,
        "averages_model_1": averages_1,
        "averages_model_2": averages_2,
        "output_csv": str(output_csv),
        "summary_csv": str(summary_csv),
        "scorecard_csv": str(scorecard_csv),
        "report_path": str(report_path),
        "saved_mails_path": str(saved_mails_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3 model comparison.")
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS, help="Path to scenarios JSON")
    parser.add_argument("--model-1", default=DEFAULT_MODEL_1, help="Model 1 for comparison")
    parser.add_argument("--model-2", default=DEFAULT_MODEL_2, help="Model 2 for comparison")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Model used for tone judging")
    parser.add_argument("--output-csv", default=DEFAULT_COMPARISON_CSV, help="Detailed comparison CSV")
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV, help="Metric summary CSV")
    parser.add_argument("--scorecard-csv", default=DEFAULT_SCENARIO_SCORECARD, help="Scenario scorecard CSV")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="Comparison report markdown")
    parser.add_argument("--save-mails", default=DEFAULT_SAVED_MAILS, help="Saved mails JSON")
    parser.add_argument(
        "--random-word-limits",
        dest="random_word_limits",
        action="store_true",
        help="Randomize word limits between --word-min and --word-max",
    )
    parser.add_argument(
        "--fixed-word-limits",
        dest="random_word_limits",
        action="store_false",
        help="Use scenario-defined word_limit values",
    )
    parser.set_defaults(random_word_limits=True)
    parser.add_argument("--word-min", type=int, default=100, help="Minimum random word limit")
    parser.add_argument("--word-max", type=int, default=1000, help="Maximum random word limit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.word_min > args.word_max:
        raise ValueError("--word-min cannot be greater than --word-max")

    result = run_model_comparison(
        scenarios_path=args.scenarios,
        model_1=args.model_1,
        model_2=args.model_2,
        judge_model=args.judge_model,
        output_csv=args.output_csv,
        summary_csv=args.summary_csv,
        scorecard_csv=args.scorecard_csv,
        report_path=args.report,
        saved_mails_path=args.save_mails,
        random_word_limits=args.random_word_limits,
        word_min=args.word_min,
        word_max=args.word_max,
        seed=args.seed,
    )

    print(f"Saved detailed comparison CSV to {result['output_csv']}")
    print(f"Saved metric summary CSV to {result['summary_csv']}")
    print(f"Saved scenario scorecard CSV to {result['scorecard_csv']}")
    print(f"Saved comparison report to {result['report_path']}")
    print(f"Saved mails JSON to {result['saved_mails_path']}")


if __name__ == "__main__":
    main()
