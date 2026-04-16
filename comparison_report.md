# Model Comparison Report

- Date: 2026-04-17 01:51:55
- Scenario count: 10
- Model 1: `llama-3.3-70b-versatile`
- Model 2: `llama-3.1-8b-instant`
- Word limits: Randomized
- Random range used: 100 to 1000 words

## Average Scores

| Metric | Model 1 | Model 2 |
|---|---:|---:|
| Fact Recall | 0.8833 | 0.9083 |
| Tone Accuracy | 0.8330 | 0.8470 |
| Clarity | 0.8205 | 0.9290 |
| Overall Average | 0.8456 | 0.8948 |
| Avg Generated Words | 242.6 | 166.7 |
| Avg Target Words | 431.1 | 431.1 |

## Quick Visual (ASCII)

- Fact Recall:   llama-3.3-70b-versatile [##################--] 0.88
                 llama-3.1-8b-instant [##################--] 0.91
- Tone Accuracy: llama-3.3-70b-versatile [#################---] 0.83
                 llama-3.1-8b-instant [#################---] 0.85
- Clarity:       llama-3.3-70b-versatile [################----] 0.82
                 llama-3.1-8b-instant [###################-] 0.93
- Overall:       llama-3.3-70b-versatile [#################---] 0.85
                 llama-3.1-8b-instant [##################--] 0.89

## Metric Winners
- Fact Recall: llama-3.1-8b-instant wins (0.9083 vs 0.8833).
- Tone Accuracy: llama-3.1-8b-instant wins (0.8470 vs 0.8330).
- Clarity: llama-3.1-8b-instant wins (0.9290 vs 0.8205).
- Overall: llama-3.1-8b-instant wins (0.8948 vs 0.8456).

## Recommendation
- Recommend `llama-3.1-8b-instant` for production based on higher overall score.

## Notes
- Both models used the same scenarios and scoring pipeline.
- Tone was rated by an LLM judge with a 0-1 score prompt.
- For visual charts, open comparison_summary.csv and scenario_scorecard.csv in Excel/Google Sheets.