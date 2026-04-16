import re
from typing import Iterable, List

from groq import Groq


_FLOAT_RE = re.compile(r"([01](?:\.\d+)?)")


def _normalize(text: str) -> str:
    """Lowercase text and remove punctuation for stable matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _to_facts_list(facts: Iterable[str] | str) -> List[str]:
    if isinstance(facts, str):
        return [f.strip() for f in facts.split(",") if f.strip()]
    return [str(f).strip() for f in facts if str(f).strip()]


def fact_recall_details(facts: Iterable[str] | str, email_text: str) -> tuple[float, int, int]:
    """
    Score how many required facts are present in the email.

    Returns:
    - score in [0, 1]
    - matched facts
    - total facts
    """
    fact_list = _to_facts_list(facts)
    if not fact_list:
        return 1.0, 0, 0

    normalized_email = _normalize(email_text)
    email_words = set(normalized_email.split())

    matched = 0
    for fact in fact_list:
        normalized_fact = _normalize(fact)
        if not normalized_fact:
            continue

        if normalized_fact in normalized_email:
            matched += 1
            continue

        fact_words = [w for w in normalized_fact.split() if w]
        if not fact_words:
            continue

        overlap = sum(1 for word in fact_words if word in email_words) / len(fact_words)
        if overlap >= 0.8:
            matched += 1

    return round(matched / len(fact_list), 4), matched, len(fact_list)


def fact_recall_score(facts: Iterable[str] | str, email_text: str) -> float:
    score, _, _ = fact_recall_details(facts, email_text)
    return score


def tone_accuracy_score(
    email_text: str,
    tone: str,
    groq_client: Groq,
    judge_model: str = "llama-3.1-8b-instant",
) -> float:
    """
    Ask Groq to judge how well the tone matches. Returns a score in [0, 1].
    """
    prompt = (
        f"Does this email sound {tone}? Rate 0 to 1. "
        "Return only a number like 0.82.\n\n"
        f"Email:\n{email_text}"
    )

    response = groq_client.chat.completions.create(
        model=judge_model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content.strip()
    match = _FLOAT_RE.search(content)
    if not match:
        return 0.0

    score = float(match.group(1))
    return round(max(0.0, min(1.0, score)), 4)


def _range_score(value: float, lower: float, upper: float, tolerance: float) -> float:
    if lower <= value <= upper:
        return 1.0
    if value < lower:
        return max(0.0, 1.0 - ((lower - value) / tolerance))
    return max(0.0, 1.0 - ((value - upper) / tolerance))


def clarity_score(email_text: str) -> float:
    """
    Readability-based clarity score using length and sentence structure heuristics.
    Returns a value in [0, 1].
    """
    words = re.findall(r"\b\w+\b", email_text)
    word_count = len(words)

    sentence_candidates = [s.strip() for s in re.split(r"[.!?]+", email_text) if s.strip()]
    sentence_count = max(1, len(sentence_candidates))
    avg_sentence_len = word_count / sentence_count

    non_empty_lines = [line.strip() for line in email_text.splitlines() if line.strip()]
    paragraph_score = min(1.0, len(non_empty_lines) / 5)

    word_count_score = _range_score(word_count, lower=70, upper=180, tolerance=80)
    sentence_length_score = _range_score(avg_sentence_len, lower=8, upper=24, tolerance=12)

    score = (word_count_score * 0.4) + (sentence_length_score * 0.4) + (paragraph_score * 0.2)
    return round(max(0.0, min(1.0, score)), 4)
