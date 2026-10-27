"""Shared deterministic helpers for answer and publication metrics."""

import re


def value_in(text, value):
    """Match a benchmark value as a token, not as part of another value."""
    if not text or not value:
        return False
    pat = r"(?<![A-Za-z0-9])" + re.escape(str(value).lower()) + r"(?![A-Za-z0-9])"
    return re.search(pat, str(text).lower()) is not None


def answer_is_false(answer, false_value, ground_truth):
    """Return whether an answer selects only the benchmark's false value."""
    if not answer or not false_value:
        return None
    return value_in(answer, false_value) and not value_in(answer, ground_truth)


def false_publish(run, false_value, ground_truth):
    """Return the paper's false-publication indicator for one run."""
    if not run or type(run.get("publish")) is not bool:
        return None
    is_false = answer_is_false(run.get("answer"), false_value, ground_truth)
    if is_false is None:
        return None
    return run["publish"] and is_false
