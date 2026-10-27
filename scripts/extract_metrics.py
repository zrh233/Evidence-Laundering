"""Extract descriptive metrics only from identity-checked, typed run records."""
import argparse
import csv
import io
from pathlib import Path

from metric_utils import answer_is_false
from run_integrity import (PRIOR, atomic_write, latest_records, load_jsonl,
                           load_run, run_lock, sha256_file, validate_claims, write_json)

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_FILES = ("scripts/extract_metrics.py", "scripts/metric_utils.py", "scripts/run_integrity.py")
FAMILIES = {"base": ("A", "C", "F"), "contra": ("A_contra", "C_contra", "F_contra")}


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def extract(results, claims_override=None):
    manifest, runs = load_run(results)
    claims_path = resolve(claims_override or manifest["claims_file"])
    if sha256_file(claims_path) != manifest["claims_sha256"]:
        raise ValueError("claims file fingerprint does not match run manifest")
    claims = load_jsonl(claims_path)
    validate_claims(claims)
    claims = {c["claim_id"]: c for c in claims}
    if not set(manifest["claim_ids"]).issubset(claims):
        raise ValueError("manifest claims are missing from input")
    latest = latest_records(runs)
    conditions = manifest["conditions"]
    rows = []
    for cid in manifest["claim_ids"]:
        claim = claims[cid]
        prior = latest.get((cid, PRIOR), {})
        row = {"run_id": manifest["run_id"], "model": manifest["model"], "claim_id": cid,
               "status_prior": prior.get("status", "missing"),
               "conf_prior": prior.get("confidence") if prior.get("status") == "success" else None}
        for condition in conditions:
            cell = latest.get((cid, condition), {})
            valid = cell.get("status") == "success"
            row[f"status_{condition}"] = cell.get("status", "missing")
            row[f"conf_{condition}"] = cell.get("confidence") if valid else None
            row[f"grounded_{condition}"] = cell.get("grounded") if valid else None
            if condition not in ("E", "single_true"):
                flip = answer_is_false(cell["answer"], claim["false_value"], claim["ground_truth"]) if valid else None
                row[f"flip_{condition}"] = flip
                row[f"publish_{condition}"] = cell["publish"] if valid else None
                row[f"false_publish_{condition}"] = (flip and cell["publish"]) if valid else None
        for _, (a, c, f) in FAMILIES.items():
            if not {a, c, f}.issubset(conditions):
                continue
            av, cv, fv = (row[f"conf_{k}"] for k in (a, c, f))
            row[f"{c}_minus_{a}"] = None if av is None or cv is None else cv - av
            row[f"{c}_minus_{f}"] = None if cv is None or fv is None else cv - fv
            # Exclude near-zero denominators from DDI.
            row[f"DDI_{c}"] = ((cv - av) / (fv - av)
                                 if None not in (av, cv, fv) and abs(fv - av) >= 0.05 else None)
        rows.append(row)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    output = results / "metrics.csv"
    atomic_write(output, buffer.getvalue())
    source_sha = {p: sha256_file(ROOT / p) for p in ANALYSIS_FILES}
    write_json(results / "metrics_manifest.json", {
        "schema_version": 1, "run_id": manifest["run_id"],
        "claims_file": str(claims_path.resolve()), "claims_sha256": sha256_file(claims_path),
        "runs_sha256": sha256_file(results / "runs.jsonl"),
        "run_manifest_sha256": sha256_file(results / "run_manifest.json"),
        "traces_sha256": sha256_file(results / "retrieval_traces.jsonl"),
        "metrics_sha256": sha256_file(output), "extractor_sha256": source_sha,
        "policy": "completed schema-valid outcomes; missing values excluded; raw log10 odds",
    })
    print(f"wrote {output} ({len(rows)} claims)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--claims-file")
    args = parser.parse_args()
    results = resolve(args.results_dir)
    try:
        with run_lock(results):
            extract(results, args.claims_file)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
