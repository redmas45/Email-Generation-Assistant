# Email Generation Assistant

A terminal-based AI email assistant with a full evaluation and model-comparison pipeline.

This project follows the assessment flow:
1. **Stage 1** - Generate professional emails from Intent, Key Facts, and Tone
2. **Stage 2** - Evaluate with custom metrics
3. **Stage 3** - Compare two models on the same scenarios

## Quick Start

```bash
pip install groq python-dotenv
```

Create/update `.env`:

```env
GROQ_API_KEY=your_groq_api_key
MODEL_1=llama-3.3-70b-versatile
MODEL_2=llama-3.1-8b-instant
JUDGE_MODEL=llama-3.1-8b-instant
```

`MODEL_1` and `MODEL_2` are global and reused everywhere (single generation, evaluation, and comparison).

## Main Entry Point (Recommended)

Run:

```bash
python main.py
```

You will get a guided menu:

- **Option 1: Create New Email**
  - Choose Model 1 or Model 2
  - Enter intent, facts, tone, target word count
  - Generated output is appended to `saved_mails.json`

- **Option 2: Stage 2 Evaluation**
  - **2A** Validate test data (`scenarios.json`) including 10 scenarios and reference emails
  - **2B** Run custom metrics and print score summary
  - **2C** Generate full evaluation outputs

- **Option 3: Stage 3 Comparison**
  - Runs Model 1 vs Model 2 on the exact same scenarios
  - Produces detailed CSVs + markdown report

## Stage 2 Outputs

Option 2C generates:

- `results.csv` - per-scenario metrics
- `results_summary.csv` - average metric table
- `evaluation_report.json` - metric definitions + raw rows + averages
- `saved_mails_evaluation.json` - generated emails from the run

### Custom Metrics

- `fact_recall_score`: required facts covered in generated email (0-1)
- `tone_accuracy_score`: LLM judge score for requested tone (0-1)
- `clarity_score`: readability heuristic (word/sentence/structure quality)
- `average_score`: mean of the three metrics above

## Stage 3 Outputs

Option 3 generates:

- `comparison_results.csv` - full side-by-side row data
- `comparison_summary.csv` - metric winner table (best for charts)
- `scenario_scorecard.csv` - scenario-wise winner table
- `comparison_report.md` - concise recommendation report
- `saved_mails_comparison.json` - all generated emails from comparison run

## Scenarios

`scenarios.json` includes 10 scenarios, each with:

- `scenario_id`
- `intent`
- `facts`
- `tone`
- `word_limit`
- `reference_email`

## Optional Direct Script Usage

Run Stage 2 directly:

```bash
python evaluate.py --model llama-3.3-70b-versatile
```

Run Stage 3 directly:

```bash
python comparison.py --model-1 llama-3.3-70b-versatile --model-2 llama-3.1-8b-instant
```

## Notes

- Word limits can be randomized between 100 and 1000 for stress testing.
- Use the same random seed for reproducible model comparisons.
- For visual dashboards, open `comparison_summary.csv` and `scenario_scorecard.csv` in Excel/Google Sheets.
