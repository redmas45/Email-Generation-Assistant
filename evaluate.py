import argparse
import csv
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from dotenv import load_dotenv
from groq import Groq

from email_assistant import generate_email
from metrics import clarity_score, fact_recall_details, tone_accuracy_score


load_dotenv('.env')

DEFAULT_MODEL_1 = os.getenv("MODEL_1", "llama-3.3-70b-versatile")
DEFAULT_MODEL_2 = os.getenv("MODEL_2", "llama-3.1-8b-instant")
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", DEFAULT_MODEL_2)
DEFAULT_SCENARIOS = "scenarios.json"
DEFAULT_OUTPUT = "results.csv"
DEFAULT_SUMMARY_OUTPUT = "results_summary.csv"
DEFAULT_REPORT_JSON = "evaluation_report.json"
DEFAULT_SAVED_MAILS = "saved_mails.json"

METRIC_DEFINITIONS = {
    "fact_recall_score": "Fraction of required facts present in the generated email (0-1).",
    "tone_accuracy_score": "LLM-judge score for tone match using prompt: 'Does this email sound [tone]? Rate 0-1.'",
    "clarity_score": "Readability heuristic based on word count range, sentence length, and structure.",
    "average_score": "Arithmetic mean of fact_recall_score, tone_accuracy_score, and clarity_score.",
}


def load_scenarios(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        scenarios = json.load(f)

    if not isinstance(scenarios, list):
        raise ValueError("scenarios.json must contain a list of scenarios")

    return scenarios


def validate_test_data(scenarios_path: str | Path, expected_count: int = 10) -> Dict[str, Any]:
    scenarios = load_scenarios(scenarios_path)
    errors: List[str] = []
    warnings: List[str] = []

    if len(scenarios) != expected_count:
        warnings.append(
            f"Expected {expected_count} scenarios as per assessment brief, found {len(scenarios)}."
        )

    required_keys = ["scenario_id", "intent", "facts", "tone", "word_limit", "reference_email"]

    for index, scenario in enumerate(scenarios, start=1):
        label = scenario.get("scenario_id", f"row_{index}")
        for key in required_keys:
            if key not in scenario:
                errors.append(f"{label}: missing required key '{key}'.")

        facts = scenario.get("facts", [])
        if not isinstance(facts, list) or not facts:
            errors.append(f"{label}: 'facts' must be a non-empty list.")

        tone = scenario.get("tone", "")
        if not isinstance(tone, str) or not tone.strip():
            errors.append(f"{label}: 'tone' must be a non-empty string.")

        reference = scenario.get("reference_email", "")
        if not isinstance(reference, str) or not reference.strip():
            errors.append(f"{label}: 'reference_email' must be a non-empty string.")

    return {
        "scenario_count": len(scenarios),
        "expected_count": expected_count,
        "errors": errors,
        "warnings": warnings,
        "is_valid": len(errors) == 0,
    }


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def choose_word_limit(
    scenario: Dict[str, Any],
    rng: random.Random,
    random_word_limits: bool,
    word_min: int,
    word_max: int,
) -> int:
    if random_word_limits:
        return rng.randint(word_min, word_max)
    return int(scenario.get("word_limit", 120))


def generate_for_scenario(scenario: Dict[str, Any], model_name: str, word_limit: int) -> str:
    facts_str = ", ".join(scenario["facts"])
    return generate_email(
        intent=scenario["intent"],
        facts=facts_str,
        tone=scenario["tone"],
        word_limit=word_limit,
        model=model_name,
    )


def run_evaluation(
    scenarios_path: str | Path,
    model_name: str,
    groq_client: Groq,
    judge_model: str,
    random_word_limits: bool = False,
    word_min: int = 100,
    word_max: int = 1000,
    seed: int | None = None,
) -> List[Dict[str, Any]]:
    scenarios = load_scenarios(scenarios_path)
    rows: List[Dict[str, Any]] = []
    rng = random.Random(seed)

    for idx, scenario in enumerate(scenarios, start=1):
        scenario_id = scenario.get("scenario_id", f"S{idx:02d}")
        word_limit = choose_word_limit(
            scenario=scenario,
            rng=rng,
            random_word_limits=random_word_limits,
            word_min=word_min,
            word_max=word_max,
        )

        generated_email = generate_for_scenario(
            scenario=scenario,
            model_name=model_name,
            word_limit=word_limit,
        )

        generated_word_count = count_words(generated_email)
        fact_score, facts_covered, facts_required = fact_recall_details(scenario["facts"], generated_email)
        tone_score = tone_accuracy_score(
            generated_email,
            scenario["tone"],
            groq_client=groq_client,
            judge_model=judge_model,
        )
        clarity = clarity_score(generated_email)
        average = round((fact_score + tone_score + clarity) / 3, 4)

        rows.append(
            {
                "scenario_id": scenario_id,
                "intent": scenario["intent"],
                "tone": scenario["tone"],
                "target_word_limit": word_limit,
                "generated_word_count": generated_word_count,
                "facts_required": facts_required,
                "facts_covered": facts_covered,
                "fact_recall_score": fact_score,
                "tone_accuracy_score": tone_score,
                "clarity_score": clarity,
                "average_score": average,
                "model": model_name,
                "reference_email": scenario.get("reference_email", ""),
                "generated_email": generated_email,
            }
        )

        print(
            f"[{idx}/{len(scenarios)}] Evaluated {scenario_id} on {model_name} "
            f"(target words: {word_limit}, generated: {generated_word_count})"
        )

    return rows


def compute_average_scores(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "fact_recall_score": 0.0,
            "tone_accuracy_score": 0.0,
            "clarity_score": 0.0,
            "average_score": 0.0,
            "target_word_limit": 0.0,
            "generated_word_count": 0.0,
        }

    return {
        "fact_recall_score": round(mean(float(r["fact_recall_score"]) for r in rows), 4),
        "tone_accuracy_score": round(mean(float(r["tone_accuracy_score"]) for r in rows), 4),
        "clarity_score": round(mean(float(r["clarity_score"]) for r in rows), 4),
        "average_score": round(mean(float(r["average_score"]) for r in rows), 4),
        "target_word_limit": round(mean(float(r["target_word_limit"]) for r in rows), 2),
        "generated_word_count": round(mean(float(r["generated_word_count"]) for r in rows), 2),
    }


def save_results_csv(rows: List[Dict[str, Any]], output_path: str | Path) -> None:
    fieldnames = [
        "scenario_id",
        "model",
        "intent",
        "tone",
        "target_word_limit",
        "generated_word_count",
        "facts_required",
        "facts_covered",
        "fact_recall_score",
        "tone_accuracy_score",
        "clarity_score",
        "average_score",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def save_metric_summary_csv(rows: List[Dict[str, Any]], output_path: str | Path) -> None:
    averages = compute_average_scores(rows)
    summary_rows = [
        {"metric": "fact_recall_score", "average": averages["fact_recall_score"]},
        {"metric": "tone_accuracy_score", "average": averages["tone_accuracy_score"]},
        {"metric": "clarity_score", "average": averages["clarity_score"]},
        {"metric": "average_score", "average": averages["average_score"]},
        {"metric": "target_word_limit", "average": averages["target_word_limit"]},
        {"metric": "generated_word_count", "average": averages["generated_word_count"]},
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "average"])
        writer.writeheader()
        writer.writerows(summary_rows)


def save_saved_mails_json(rows: List[Dict[str, Any]], output_path: str | Path) -> None:
    payload = [
        {
            "scenario_id": row["scenario_id"],
            "model": row["model"],
            "intent": row["intent"],
            "tone": row["tone"],
            "target_word_limit": row["target_word_limit"],
            "generated_word_count": row["generated_word_count"],
            "email": row["generated_email"],
        }
        for row in rows
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_evaluation_report_json(
    rows: List[Dict[str, Any]],
    output_path: str | Path,
    model_name: str,
    scenarios_path: str | Path,
    random_word_limits: bool,
    word_min: int,
    word_max: int,
    seed: int | None,
) -> None:
    averages = compute_average_scores(rows)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "scenarios_path": str(scenarios_path),
        "metric_definitions": METRIC_DEFINITIONS,
        "run_config": {
            "random_word_limits": random_word_limits,
            "word_min": word_min,
            "word_max": word_max,
            "seed": seed,
        },
        "averages": averages,
        "rows": rows,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def generate_evaluation_outputs(
    scenarios_path: str | Path,
    model_name: str,
    judge_model: str,
    output_csv: str | Path,
    summary_csv: str | Path,
    report_json: str | Path,
    saved_mails_json: str | Path,
    random_word_limits: bool,
    word_min: int,
    word_max: int,
    seed: int | None,
) -> List[Dict[str, Any]]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your environment or .env file.")

    client = Groq(api_key=api_key)
    rows = run_evaluation(
        scenarios_path=scenarios_path,
        model_name=model_name,
        groq_client=client,
        judge_model=judge_model,
        random_word_limits=random_word_limits,
        word_min=word_min,
        word_max=word_max,
        seed=seed,
    )

    save_results_csv(rows, output_csv)
    save_metric_summary_csv(rows, summary_csv)
    save_saved_mails_json(rows, saved_mails_json)
    save_evaluation_report_json(
        rows=rows,
        output_path=report_json,
        model_name=model_name,
        scenarios_path=scenarios_path,
        random_word_limits=random_word_limits,
        word_min=word_min,
        word_max=word_max,
        seed=seed,
    )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 evaluation for the Email Generation Assistant.")
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS, help="Path to scenarios JSON file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Detailed results CSV path")
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT, help="Summary CSV path")
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON, help="Evaluation report JSON path")
    parser.add_argument("--save-mails", default=DEFAULT_SAVED_MAILS, help="Saved mails JSON path")
    parser.add_argument(
        "--model",
        default=os.getenv("EVAL_MODEL", DEFAULT_MODEL_1),
        help="Model used for email generation",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Model used for tone judging",
    )
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
        help="Use word_limit from scenarios.json",
    )
    parser.set_defaults(random_word_limits=True)
    parser.add_argument("--word-min", type=int, default=100, help="Minimum random word limit")
    parser.add_argument("--word-max", type=int, default=1000, help="Maximum random word limit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate scenario structure (Stage 2A) and exit",
    )
    args = parser.parse_args()

    if args.word_min > args.word_max:
        raise ValueError("--word-min cannot be greater than --word-max")

    validation = validate_test_data(args.scenarios)
    if validation["warnings"]:
        for warning in validation["warnings"]:
            print(f"WARNING: {warning}")

    if validation["errors"]:
        for error in validation["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit("Scenario validation failed.")

    print(f"Scenario validation passed ({validation['scenario_count']} scenarios).")

    if args.validate_only:
        return

    rows = generate_evaluation_outputs(
        scenarios_path=args.scenarios,
        model_name=args.model,
        judge_model=args.judge_model,
        output_csv=args.output,
        summary_csv=args.summary_output,
        report_json=args.report_json,
        saved_mails_json=args.save_mails,
        random_word_limits=args.random_word_limits,
        word_min=args.word_min,
        word_max=args.word_max,
        seed=args.seed,
    )

    averages = compute_average_scores(rows)
    print(f"Saved detailed evaluation CSV to {args.output}")
    print(f"Saved summary CSV to {args.summary_output}")
    print(f"Saved evaluation report JSON to {args.report_json}")
    print(f"Saved mails JSON to {args.save_mails}")
    print(f"Overall average score: {averages['average_score']}")


if __name__ == "__main__":
    main()
