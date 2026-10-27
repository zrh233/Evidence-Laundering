"""Summarize verified metrics, preserving condition families and missingness."""
import argparse
import csv
import math
from collections import Counter
from pathlib import Path

from extract_metrics import ANALYSIS_FILES, FAMILIES, resolve
from run_integrity import canonical, load_json, load_run, run_lock, sha256_file, write_json

ROOT = Path(__file__).resolve().parent.parent


def number(value):
    if value in (None, ""):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite number in metrics")
    return result


def boolean(value):
    if value in (None, ""):
        return None
    if value not in ("True", "False"):
        raise ValueError("invalid boolean in metrics")
    return value == "True"


def mean(values):
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else None


def rate(rows, key):
    values = [boolean(row.get(key)) for row in rows]
    values = [v for v in values if v is not None]
    return {"k": sum(values), "n": len(values), "rate": mean(values)}


def analyze(results, prior_max=None):
    manifest, _ = load_run(results)
    inputs = load_json(results / "metrics_manifest.json")
    if inputs.get("schema_version") != 1 or inputs.get("run_id") != manifest["run_id"]:
        raise ValueError("metrics manifest identity mismatch")
    checks = {"metrics_sha256": results / "metrics.csv", "runs_sha256": results / "runs.jsonl",
              "traces_sha256": results / "retrieval_traces.jsonl",
              "run_manifest_sha256": results / "run_manifest.json",
              "claims_sha256": Path(inputs["claims_file"])}
    if any(inputs.get(key) != sha256_file(path) for key, path in checks.items()):
        raise ValueError("metrics or source inputs changed; run extract_metrics.py again")
    if inputs.get("extractor_sha256") != {p: sha256_file(ROOT / p) for p in ANALYSIS_FILES}:
        raise ValueError("extractor changed; run extract_metrics.py again")
    with (results / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if ([r["claim_id"] for r in rows] != manifest["claim_ids"] or
            any(r["run_id"] != manifest["run_id"] or r["model"] != manifest["model"] for r in rows)):
        raise ValueError("metrics rows do not match run identity/claim list")
    threshold = manifest["prior_exclude_above"] if prior_max is None else prior_max
    if not math.isfinite(threshold) or not 0 <= threshold <= manifest["prior_exclude_above"]:
        raise ValueError("--prior-max must be within [0, run prior threshold]; excluded claims have no outcomes")
    clean = [r for r in rows if r["status_prior"] == "success"
             and number(r["conf_prior"]) is not None and number(r["conf_prior"]) <= threshold]
    conditions = manifest["conditions"]
    families = {}
    for name, (a, c, f) in FAMILIES.items():
        if not {a, c, f}.issubset(conditions):
            continue
        complete = [r for r in clean if all(number(r[f"conf_{k}"]) is not None for k in (a, c, f))]
        ddi = [number(r[f"DDI_{c}"]) for r in complete if number(r[f"DDI_{c}"]) is not None]
        paired = [(boolean(r[f"false_publish_{a}"]), boolean(r[f"false_publish_{c}"])) for r in clean]
        deltas = [int(cv) - int(av) for av, cv in paired if av is not None and cv is not None]
        families[name] = {"conditions": [a, c, f], "n_complete": len(complete),
                          "mean_C_minus_A": mean(number(r[f"{c}_minus_{a}"]) for r in complete),
                          "mean_C_minus_F": mean(number(r[f"{c}_minus_{f}"]) for r in complete),
                          "mean_DDI": mean(ddi), "n_DDI": len(ddi),
                          "paired_false_publish_C_minus_A": {"n": len(deltas), "mean_difference": mean(deltas)}}
    summary = {
        "run_id": manifest["run_id"], "model": manifest["model"], "stub": manifest["stub"],
        "analysis": "raw log10 odds; arithmetic means and paired differences",
        "metrics_sha256": inputs["metrics_sha256"],
        "analyzer_sha256": sha256_file(Path(__file__)),
        "n_metrics": len(rows), "n_prior_clean": len(clean), "prior_max": threshold,
        "prior_status_counts": dict(Counter(r["status_prior"] for r in rows)),
        "condition_status_counts": {c: dict(Counter(r[f"status_{c}"] for r in clean)) for c in conditions},
        "mean_log10_odds": {c: mean(number(r[f"conf_{c}"]) for r in clean) for c in conditions},
        "n_log10_odds": {c: sum(number(r[f"conf_{c}"]) is not None for r in clean) for c in conditions},
        "false_publish": {c: rate(clean, f"false_publish_{c}") for c in conditions if c not in ("E", "single_true")},
        "families": families,
    }
    write_json(results / "summary.json", summary)
    print(canonical(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--prior-max", type=float)
    args = parser.parse_args()
    results = resolve(args.results_dir)
    try:
        with run_lock(results):
            analyze(results, args.prior_max)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
