"""Resumable runner for the controlled evidence-laundering experiment."""

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

from run_agent import no_retrieval_prior, run_agent  # noqa: E402


DEFAULT_CLAIMS = ROOT / "data" / "claims.jsonl"
CORE_FILES = (
    ROOT / "config.json",
    ROOT / "llm.py",
    SCRIPTS / "defense.py",
    SCRIPTS / "retriever.py",
    SCRIPTS / "run_agent.py",
    SCRIPTS / "run_experiment.py",
    SCRIPTS / "run_integrity.py",
)


class StubLLM:
    """Deterministic client for offline workflow checks."""

    model = "stub"
    request_errors = ()

    def chat(self, messages, temperature=None, max_tokens=None):
        system = " ".join(
            message["content"] for message in messages
            if message["role"] == "system"
        )
        if "from your own knowledge only" in system:
            return json.dumps({
                "answer": "(stub: uncertain fictional entity)",
                "confidence_0_100": 18,
                "established_fact": "no",
            })

        joined = "\n".join(
            message["content"] for message in messages
            if message["role"] != "system"
        )
        opened = re.findall(r'"doc_id":\s*"([^"]+)"[^}]*"body"', joined)
        if "SEARCH_RESULTS:" not in joined:
            question = messages[1]["content"].replace("Question:", "").strip()
            return json.dumps({
                "thought": "search",
                "action": "search",
                "query": question,
            })

        candidates = []
        for message in reversed(messages):
            if (message["role"] == "user"
                    and message["content"].startswith("SEARCH_RESULTS:")):
                candidates = re.findall(
                    r'"doc_id":\s*"([^"]+)"', message["content"]
                )
                break
        unopened = [doc_id for doc_id in candidates if doc_id not in set(opened)]
        if len(opened) < 4 and unopened:
            return json.dumps({
                "thought": "open",
                "action": "open",
                "doc_id": unopened[0],
            })

        match = re.search(
            r"_(single_true|A_contra|C_contra|F_contra|A|C|F|E)_(?:c|t)\d",
            joined,
        )
        condition = match.group(1) if match else "unknown"
        log_odds = {
            "A": 0.6,
            "C": 1.6,
            "F": 1.7,
            "E": 1.8,
            "A_contra": 0.1,
            "C_contra": 0.4,
            "F_contra": 0.5,
            "single_true": 0.8,
        }.get(condition, 0.3)
        return json.dumps({
            "thought": "finish",
            "action": "final",
            "answer": "(stub answer)",
            "confidence_log10_odds": log_odds,
            "publish": log_odds >= 1.0,
            "cited_doc_ids": opened,
        })


from run_integrity import (  # noqa: E402
    CONDITIONS, PRIOR, canonical, fingerprint,
    latest_records, load_json, load_jsonl, load_run, prepare_append, run_lock,
    sha256_file, validate_claims, validate_manifest, validate_records, write_json,
)
from run_agent import PRIOR_SYSTEM  # noqa: E402
from retriever import load_corpus  # noqa: E402


def _sha256_tree(path):
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*.json")
                       if p.is_file() and not p.name.startswith(".")):
        digest.update(item.relative_to(path).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _relative_or_absolute(path):
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve(value, default):
    path = Path(value) if value else default
    return path if path.is_absolute() else ROOT / path


def _get_client(stub, model):
    if stub:
        return StubLLM()
    from llm import get_llm
    try:
        return get_llm(model_override=model)
    except ModuleNotFoundError as exc:
        if exc.name != "openai":
            raise
        raise ValueError("Install the API client with: python -m pip install -r requirements-api.txt") from exc


def _load_reused_priors(path, prior_identity, selected_claim_ids):
    if path is None:
        return {}, None
    if not path.is_file() or path.name != "runs.jsonl":
        raise ValueError("prior reuse requires a runs.jsonl file")
    with run_lock(path.parent):
        source, rows = load_run(path.parent)
        if source["prior_identity"] != prior_identity:
            raise ValueError("prior source configuration/endpoint/code does not match")
        selected = set(selected_claim_ids)
        priors = {cid: row for (cid, condition), row in latest_records(rows).items()
                  if condition == PRIOR and cid in selected and row["status"] == "success"}
        provenance = {"source_run_id": source["run_id"],
                      "source_runs_sha256": sha256_file(path)}
        return priors, provenance


def _preflight(corpora, claims, conditions):
    for claim in claims:
        for condition in conditions:
            directory = corpora / claim["claim_id"] / f"cond_{condition}"
            docs = load_corpus(directory)
            if not directory.is_dir() or not docs:
                raise ValueError(f"missing or empty corpus: {claim['claim_id']}/{condition}")
            ids = set()
            for doc in docs:
                if (not isinstance(doc, dict) or not isinstance(doc.get("doc_id"), str)
                        or not doc["doc_id"] or doc["doc_id"] in ids):
                    raise ValueError("corpus has missing/duplicate document ids")
                ids.add(doc["doc_id"])
                if any(not isinstance(doc.get(k), str) for k in
                       ("title", "outlet", "author", "date", "url", "body")):
                    raise ValueError("corpus page is missing text fields")


def _error_row(claim_id, condition, exc):
    return {"claim_id": claim_id, "condition": condition, "confidence": None,
            "completed": False, "status": "api_error", "error_type": type(exc).__name__}


def _execute(args, config, claims_path, claims, corpora, conditions, results):
    client = _get_client(args.stub, args.model)
    if not args.stub:
        client.lc.update({k: config["llm"][k] for k in
                          ("temperature", "max_tokens", "request_timeout_s", "max_retries")})
    model = client.model
    llm_config = {k: config["llm"][k] for k in
                  ("temperature", "max_tokens", "request_timeout_s", "max_retries")}
    llm_config["endpoint_fingerprint"] = "stub" if args.stub else client.endpoint_fingerprint
    core_sha = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in CORE_FILES}
    prior_identity = {
        "model": model, "stub": args.stub, "llm": llm_config,
        "claims_sha256": sha256_file(claims_path), "prompt_sha256": fingerprint(PRIOR_SYSTEM),
        "code_sha256": {k: core_sha[k] for k in
                        ("llm.py", "scripts/run_agent.py", "scripts/run_integrity.py")},
    }
    prior_path = _resolve(args.reuse_priors_from, None) if args.reuse_priors_from else None
    if prior_path and prior_path.parent.resolve() == results.resolve():
        raise ValueError("cannot reuse priors from the destination itself")
    reused, reused_provenance = _load_reused_priors(
        prior_path, prior_identity, [c["claim_id"] for c in claims])
    identity = {
        "schema_version": 2, "stub": bool(args.stub), "model": model, "llm": llm_config,
        "claims_requested": len(claims), "claim_ids": [c["claim_id"] for c in claims],
        "claims_sha256": sha256_file(claims_path), "corpora_sha256": _sha256_tree(corpora),
        "conditions": conditions, "prior_exclude_above": config["killswitch"]["prior_exclude_above"],
        "prior_identity": prior_identity, "reused_priors": reused_provenance,
        "retrieval": config["retrieval"], "agent": config["agent"],
        "defense": args.defense, "defense_config": config["defense"], "core_sha256": core_sha,
    }
    run_id = fingerprint(identity)[:16]
    manifest = {**identity, "run_id": run_id,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "claims_file": _relative_or_absolute(claims_path),
                "corpora_dir": _relative_or_absolute(corpora)}
    with run_lock(results):
        mp = results / "run_manifest.json"
        runs_path, trace_path = results / "runs.jsonl", results / "retrieval_traces.jsonl"
        if mp.exists():
            previous = validate_manifest(load_json(mp))
            if previous["run_id"] != run_id:
                raise ValueError("results directory belongs to a different run configuration; choose a new --outdir")
            manifest, rows = load_run(results, recover_tail=args.recover_tail)
        elif runs_path.exists() or trace_path.exists():
            raise ValueError("results exist without a manifest; choose a new output directory")
        else:
            write_json(mp, manifest)
            rows = []
        latest = latest_records(rows)
        prepare_append(runs_path)
        prepare_append(trace_path)
        with runs_path.open("a", encoding="utf-8") as runs_file, trace_path.open("a", encoding="utf-8") as traces_file:
            def emit(handle, value):
                handle.write(canonical(value) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

            def record(row, trace=None):
                row = {**row, "attempt_id": uuid.uuid4().hex, "run_id": run_id, "model": model}
                if trace is not None:
                    row["trace_sha256"] = fingerprint(trace)
                validate_records([row], manifest)
                # The trace is durable before its result commits. An orphan trace
                # after interruption is safe; a committed result without one is not.
                if trace is not None:
                    emit(traces_file, {k: row[k] for k in
                         ("attempt_id", "run_id", "claim_id", "condition")} | {"trace": trace})
                emit(runs_file, row)
                latest[(row["claim_id"], row["condition"])] = row

            for cid, row in reused.items():
                if latest.get((cid, PRIOR), {}).get("status") == "success":
                    continue
                copied = {k: row[k] for k in ("claim_id", "condition", "question", "answer",
                          "confidence", "established_fact", "completed", "status")}
                copied.update(reused_prior=True, source_run_id=reused_provenance["source_run_id"])
                record(copied)
            pending = [c for c in claims if latest.get((c["claim_id"], PRIOR), {}).get("status") != "success"]
            print(f"run_id={run_id} model={model} priors={len(pending)}")
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(no_retrieval_prior, client, c): c for c in pending}
                for future in as_completed(futures):
                    claim = futures[future]
                    try:
                        row = future.result()
                    except client.request_errors as exc:
                        row = _error_row(claim["claim_id"], PRIOR, exc)
                    record(row)
            survivors = [c for c in claims
                         if latest[(c["claim_id"], PRIOR)]["status"] == "success"
                         and latest[(c["claim_id"], PRIOR)]["confidence"] <= identity["prior_exclude_above"]]
            work = [(c, cond) for c in survivors for cond in conditions
                    if latest.get((c["claim_id"], cond), {}).get("status") != "success"]
            if args.prior_only:
                work = []
            print(f"prior-clean claims={len(survivors)}/{len(claims)} experiment cells={len(work)}")
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(run_agent, client, c, cond,
                           corpora / c["claim_id"] / f"cond_{cond}", config,
                           defense=args.defense): (c["claim_id"], cond) for c, cond in work}
                for future in as_completed(futures):
                    cid, cond = futures[future]
                    try:
                        row = future.result()
                    except client.request_errors as exc:
                        row, trace = _error_row(cid, cond, exc), None
                    else:
                        trace = row.pop("trace")
                    record(row, trace)
        expected = [(c["claim_id"], cond) for c in survivors for cond in conditions] if not args.prior_only else []
        failed_priors = sum(latest[(c["claim_id"], PRIOR)]["status"] != "success" for c in claims)
        incomplete = [key for key in expected if latest.get(key, {}).get("status") != "success"]
        status = {"run_id": run_id, "phase": "prior_only" if args.prior_only else "experiment",
                  "claims_requested": len(claims), "prior_clean": len(survivors),
                  "failed_priors": failed_priors,
                  "excluded_priors": len(claims) - len(survivors) - failed_priors,
                  "expected_cells": len(expected), "successful_cells": len(expected) - len(incomplete),
                  "incomplete_cells": [{"claim_id": c, "condition": d} for c, d in incomplete],
                  "complete": not failed_priors and not incomplete}
        write_json(results / "run_status.json", status)
        print(canonical(status))
        return 0 if status["complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--claims", type=int, default=10)
    parser.add_argument("--claims-file")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    parser.add_argument("--prior-only", action="store_true")
    parser.add_argument("--no-single-true", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model")
    for option in ("max-tokens", "max-steps", "max-opens", "min-opens", "max-searches", "top-k"):
        parser.add_argument("--" + option, type=int)
    parser.add_argument("--reuse-priors-from")
    parser.add_argument("--outdir", default="")
    parser.add_argument("--results-dir")
    parser.add_argument("--corpora-dir")
    parser.add_argument("--recover-tail", action="store_true", help="back up and remove only an unterminated malformed JSONL tail")
    parser.add_argument("--defense", choices=("none", "shingle", "skeleton"), default="none")
    args = parser.parse_args()
    try:
        if args.claims < 1 or args.workers < 1:
            raise ValueError("--claims and --workers must be positive")
        if args.results_dir and args.outdir:
            raise ValueError("use either --outdir or --results-dir")
        config = load_json(ROOT / "config.json")
        overrides = {"max_steps": "max_steps", "max_opens": "max_opens",
                     "min_opens": "require_min_opens", "max_searches": "max_searches"}
        for option, key in overrides.items():
            value = getattr(args, option)
            if value is not None:
                if value < (0 if option == "min_opens" else 1):
                    raise ValueError(f"invalid --{option.replace('_', '-')}")
                config["agent"][key] = value
        if config["agent"]["require_min_opens"] > config["agent"]["max_opens"]:
            raise ValueError("--min-opens exceeds --max-opens")
        for option, section, key in (("top_k", "retrieval", "top_k"), ("max_tokens", "llm", "max_tokens")):
            value = getattr(args, option)
            if value is not None:
                if value < 1:
                    raise ValueError(f"invalid {option}")
                config[section][key] = value
        conditions = list(args.conditions or config["killswitch"]["conditions"])
        if len(conditions) != len(set(conditions)) or any(c not in CONDITIONS for c in conditions):
            raise ValueError("conditions must be supported and unique")
        if not args.no_single_true and "single_true" not in conditions:
            conditions.append("single_true")
        claims_path = _resolve(args.claims_file, DEFAULT_CLAIMS)
        all_claims = load_jsonl(claims_path)
        validate_claims(all_claims)
        if len(all_claims) < args.claims:
            raise ValueError(f"requested {args.claims} claims but input contains {len(all_claims)}; generate enough claims first")
        claims = all_claims[:args.claims]
        corpora = _resolve(args.corpora_dir, ROOT / "data/corpora")
        results = _resolve(args.results_dir, ROOT / "results" / args.outdir)
        if not args.prior_only:
            _preflight(corpora, claims, conditions)
        return _execute(args, config, claims_path, claims, corpora, conditions, results)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
