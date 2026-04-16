import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from comparison import run_model_comparison
from email_assistant import generate_email_with_retry
from evaluate import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MODEL_1,
    DEFAULT_MODEL_2,
    compute_average_scores,
    generate_evaluation_outputs,
    run_evaluation,
    validate_test_data,
)
from groq import Groq


load_dotenv('.env')

SCENARIOS_PATH = "scenarios.json"
SAVED_MAILS_PATH = "saved_mails.json"


def read_menu_choice(prompt: str, valid_choices: set[str]) -> str:
    while True:
        value = input(prompt).strip()
        if value in valid_choices:
            return value
        print("Invalid choice. Please try again.")


def read_int(prompt: str, default: int) -> int:
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Invalid number. Using default: {default}")
        return default


def read_yes_no(prompt: str, default: bool) -> bool:
    fallback = "y" if default else "n"
    raw = input(f"{prompt} [y/n] (default {fallback}): ").strip().lower()
    if raw in {"y", "yes"}:
        return True
    if raw in {"n", "no"}:
        return False
    return default


def get_models() -> tuple[str, str, str]:
    model_1 = os.getenv("MODEL_1", DEFAULT_MODEL_1)
    model_2 = os.getenv("MODEL_2", DEFAULT_MODEL_2)
    judge_model = os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    return model_1, model_2, judge_model


def choose_model_interactive() -> str:
    model_1, model_2, _ = get_models()
    print("\nChoose model:")
    print(f"1) Model 1 ({model_1})")
    print(f"2) Model 2 ({model_2})")
    choice = read_menu_choice("Enter choice: ", {"1", "2"})
    return model_1 if choice == "1" else model_2


def append_saved_mail(entry: dict, path: str = SAVED_MAILS_PATH) -> None:
    file_path = Path(path)
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(entry)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def option_create_email() -> None:
    model_name = choose_model_interactive()
    intent = input("Enter intent: ").strip()
    facts = input("Enter key facts (comma-separated): ").strip()
    tone = input("Enter tone: ").strip()
    word_limit = read_int("Enter target word limit (default 150): ", 150)

    email = generate_email_with_retry(
        intent=intent,
        facts=facts,
        tone=tone,
        word_limit=word_limit,
        model=model_name,
    )

    print("\nGenerated Email:\n")
    print(email)

    append_saved_mail(
        {
            "type": "manual_email",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model": model_name,
            "intent": intent,
            "facts": facts,
            "tone": tone,
            "target_word_limit": word_limit,
            "email": email,
        }
    )
    print(f"\nSaved to {SAVED_MAILS_PATH}")


def option_stage2_submenu() -> None:
    while True:
        print("\nStage 2 - Evaluation Strategy")
        print("1) Stage 2A: Validate Test Data (10 scenarios + reference emails)")
        print("2) Stage 2B: Run Custom Metrics")
        print("3) Stage 2C: Generate Evaluation Report Files")
        print("4) Back to main menu")

        choice = read_menu_choice("Enter choice: ", {"1", "2", "3", "4"})

        if choice == "1":
            validation = validate_test_data(SCENARIOS_PATH)
            print(f"\nScenarios found: {validation['scenario_count']} (expected {validation['expected_count']})")
            if validation["warnings"]:
                print("Warnings:")
                for warning in validation["warnings"]:
                    print(f"- {warning}")
            if validation["errors"]:
                print("Errors:")
                for error in validation["errors"]:
                    print(f"- {error}")
            else:
                print("Validation passed. Test data is ready.")

        elif choice == "2":
            model_name = choose_model_interactive()
            _, _, judge_model = get_models()
            random_word_limits = read_yes_no("Randomize word limits from 100 to 1000?", True)
            seed = read_int("Random seed (default 42): ", 42)

            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print("GROQ_API_KEY is missing in .env")
                continue

            client = Groq(api_key=api_key)
            rows = run_evaluation(
                scenarios_path=SCENARIOS_PATH,
                model_name=model_name,
                groq_client=client,
                judge_model=judge_model,
                random_word_limits=random_word_limits,
                word_min=100,
                word_max=1000,
                seed=seed,
            )
            averages = compute_average_scores(rows)
            print("\nCustom Metrics Summary")
            print(f"- Fact Recall: {averages['fact_recall_score']}")
            print(f"- Tone Accuracy: {averages['tone_accuracy_score']}")
            print(f"- Clarity: {averages['clarity_score']}")
            print(f"- Overall Average: {averages['average_score']}")

        elif choice == "3":
            model_name = choose_model_interactive()
            _, _, judge_model = get_models()
            random_word_limits = read_yes_no("Randomize word limits from 100 to 1000?", True)
            seed = read_int("Random seed (default 42): ", 42)

            try:
                rows = generate_evaluation_outputs(
                    scenarios_path=SCENARIOS_PATH,
                    model_name=model_name,
                    judge_model=judge_model,
                    output_csv="results.csv",
                    summary_csv="results_summary.csv",
                    report_json="evaluation_report.json",
                    saved_mails_json="saved_mails_evaluation.json",
                    random_word_limits=random_word_limits,
                    word_min=100,
                    word_max=1000,
                    seed=seed,
                )
                averages = compute_average_scores(rows)
                print("\nGenerated:")
                print("- results.csv")
                print("- results_summary.csv")
                print("- evaluation_report.json")
                print("- saved_mails_evaluation.json")
                print(f"Overall average score: {averages['average_score']}")
            except Exception as exc:
                print(f"Failed to generate evaluation report: {exc}")

        else:
            return


def option_stage3_comparison() -> None:
    model_1, model_2, judge_model = get_models()
    print("\nStage 3 - Model Comparison")
    print(f"Model 1 from .env: {model_1}")
    print(f"Model 2 from .env: {model_2}")

    random_word_limits = read_yes_no("Randomize word limits from 100 to 1000?", True)
    seed = read_int("Random seed (default 42): ", 42)

    try:
        result = run_model_comparison(
            scenarios_path=SCENARIOS_PATH,
            model_1=model_1,
            model_2=model_2,
            judge_model=judge_model,
            output_csv="comparison_results.csv",
            summary_csv="comparison_summary.csv",
            scorecard_csv="scenario_scorecard.csv",
            report_path="comparison_report.md",
            saved_mails_path="saved_mails_comparison.json",
            random_word_limits=random_word_limits,
            word_min=100,
            word_max=1000,
            seed=seed,
        )

        avg_1 = result["averages_model_1"]["average_score"]
        avg_2 = result["averages_model_2"]["average_score"]

        print("\nGenerated:")
        print(f"- {result['output_csv']}")
        print(f"- {result['summary_csv']}")
        print(f"- {result['scorecard_csv']}")
        print(f"- {result['report_path']}")
        print(f"- {result['saved_mails_path']}")
        print(f"\nOverall averages -> Model 1: {avg_1}, Model 2: {avg_2}")

    except Exception as exc:
        print(f"Comparison failed: {exc}")


def main() -> None:
    while True:
        model_1, model_2, _ = get_models()
        print("\nEmail Generation Assistant")
        print(f"Configured Model 1: {model_1}")
        print(f"Configured Model 2: {model_2}")
        print("1) Create New Email")
        print("2) Stage 2 Evaluation")
        print("3) Stage 3 Comparison")
        print("4) Exit")

        choice = read_menu_choice("Enter choice: ", {"1", "2", "3", "4"})

        if choice == "1":
            option_create_email()
        elif choice == "2":
            option_stage2_submenu()
        elif choice == "3":
            option_stage3_comparison()
        else:
            print("Goodbye!")
            return


if __name__ == "__main__":
    main()
