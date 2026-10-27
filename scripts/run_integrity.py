"""Shared schemas, fingerprints and crash-aware local run storage."""
import hashlib
import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

CONDITIONS = ("A", "B", "C", "D", "F", "E", "single_true",
              "A_contra", "C_contra", "F_contra")
PRIOR = "no_retrieval_prior"
IDENTITY_KEYS = (
    "schema_version", "stub", "model", "llm", "claims_requested", "claim_ids",
    "claims_sha256", "corpora_sha256", "conditions", "prior_exclude_above",
    "prior_identity", "reused_priors", "retrieval", "agent", "defense",
    "defense_config", "core_sha256",
)
FAILURES = {"api_error", "invalid_output", "ungrounded"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def valid_prior(value):
    return (isinstance(value, dict) and isinstance(value.get("answer"), str)
            and bool(value["answer"].strip())
            and finite_number(value.get("confidence_0_100"))
            and 0 <= value["confidence_0_100"] <= 100
            and value.get("established_fact") in ("yes", "no"))


def valid_final(value):
    ids = value.get("cited_doc_ids") if isinstance(value, dict) else None
    return (isinstance(value, dict) and isinstance(value.get("answer"), str)
            and bool(value["answer"].strip())
            and finite_number(value.get("confidence_log10_odds"))
            and type(value.get("publish")) is bool and isinstance(ids, list)
            and all(isinstance(x, str) and x for x in ids)
            and len(ids) == len(set(ids)))


def reject_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=reject_constant)


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = data.encode("utf-8") if isinstance(data, str) else data
    fd, temp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False,
                                 allow_nan=False) + "\n")


def load_jsonl(path, recover_tail=False):
    """Read JSONL, optionally backing up and removing a torn final record.

    Recovery requires the caller to hold run_lock. Other corruption raises.
    """
    path = Path(path)
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows, offset = [], 0
    for index, line in enumerate(lines):
        if not line.strip():
            offset += len(line)
            continue
        try:
            row = json.loads(line, parse_constant=reject_constant)
            if not isinstance(row, dict):
                raise ValueError("record is not an object")
        except (ValueError, UnicodeDecodeError) as exc:
            # Schema errors and non-finite values are not torn JSON writes.
            torn = isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError))
            if (recover_tail and torn and index == len(lines) - 1
                    and not line.endswith(b"\n")):
                digest = hashlib.sha256(line).hexdigest()[:16]
                atomic_write(path.parent / "recovery" / f"{path.name}.{digest}.tail", line)
                atomic_write(path, raw[:offset])
                return rows
            raise ValueError(f"invalid JSONL at {path.name}:{index + 1}") from exc
        rows.append(row)
        offset += len(line)
    return rows


def prepare_append(path):
    path = Path(path)
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            atomic_write(path, data + b"\n")


@contextmanager
def run_lock(directory):
    """Process-scoped file lock using a persistent lock inode."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if handle.seek(0, 2) == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError("results directory is locked by another process") from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ValueError("results directory is locked by another process") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def safe_id(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value)


def validate_claims(claims):
    seen = set()
    for claim in claims:
        cid = claim.get("claim_id")
        if not safe_id(cid) or cid in seen:
            raise ValueError("claims require unique safe claim_id values")
        seen.add(cid)
        for key in ("question", "entity", "false_value", "ground_truth", "false_claim"):
            if not isinstance(claim.get(key), str) or not claim[key].strip():
                raise ValueError(f"claim {cid} has invalid {key}")
        control = claim.get("control", {})
        if not isinstance(control, dict) or not isinstance(control.get("question"), str):
            raise ValueError(f"claim {cid} has no control question")


def validate_manifest(manifest):
    if manifest.get("schema_version") != 2:
        raise ValueError("unsupported run schema; choose a new output directory")
    try:
        identity = {key: manifest[key] for key in IDENTITY_KEYS}
    except KeyError as exc:
        raise ValueError("incomplete run manifest") from exc
    if fingerprint(identity)[:16] != manifest.get("run_id"):
        raise ValueError("run manifest fingerprint mismatch")
    return manifest


def validate_records(rows, manifest):
    """Reject identity/schema corruption; retain declared failed attempts."""
    claims = set(manifest["claim_ids"])
    conditions = set(manifest["conditions"]) | {PRIOR}
    attempts, successes = set(), set()
    for row in rows:
        if row.get("run_id") != manifest["run_id"] or row.get("model") != manifest["model"]:
            raise ValueError("result identity does not match manifest")
        key = (row.get("claim_id"), row.get("condition"))
        if key[0] not in claims or key[1] not in conditions:
            raise ValueError("result claim/condition is outside manifest")
        attempt = row.get("attempt_id")
        if not isinstance(attempt, str) or not attempt or attempt in attempts:
            raise ValueError("missing or duplicate result attempt_id")
        attempts.add(attempt)
        status = row.get("status")
        if status == "success":
            if row.get("completed") is not True:
                raise ValueError("successful result must be completed")
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError("successful result must retain its question")
            if key[1] == PRIOR:
                valid = valid_prior({**row, "confidence_0_100": row.get("confidence")})
            else:
                valid = valid_final({**row, "confidence_log10_odds": row.get("confidence")})
                valid = valid and row.get("forced_final") is False
            if not valid or key in successes:
                raise ValueError("invalid or duplicate successful result")
            successes.add(key)
        elif (status not in FAILURES or row.get("completed") is not False
              or row.get("confidence") is not None):
            raise ValueError("invalid failed-result schema")


def load_run(directory, recover_tail=False):
    directory = Path(directory)
    manifest = validate_manifest(load_json(directory / "run_manifest.json"))
    rows = load_jsonl(directory / "runs.jsonl", recover_tail) if (directory / "runs.jsonl").exists() else []
    validate_records(rows, manifest)
    trace_path = directory / "retrieval_traces.jsonl"
    traces = load_jsonl(trace_path, recover_tail) if trace_path.exists() else []
    by_attempt = {}
    for entry in traces:
        aid = entry.get("attempt_id")
        if (entry.get("run_id") != manifest["run_id"] or not isinstance(aid, str)
                or aid in by_attempt or not isinstance(entry.get("trace"), dict)
                or entry.get("claim_id") not in manifest["claim_ids"]
                or entry.get("condition") not in manifest["conditions"]):
            raise ValueError("invalid trace identity/schema")
        by_attempt[aid] = entry
    for row in rows:
        if row["condition"] == PRIOR:
            continue
        if row.get("trace_sha256") is None:
            if row["status"] != "api_error":
                raise ValueError("result is missing trace fingerprint")
            continue
        entry = by_attempt.get(row["attempt_id"])
        if (entry is None or entry["claim_id"] != row["claim_id"]
                or entry["condition"] != row["condition"]
                or fingerprint(entry["trace"]) != row["trace_sha256"]):
            raise ValueError("result trace is missing or does not match its fingerprint")
    return manifest, rows


def latest_records(rows):
    latest = {}
    for row in rows:
        key = (row["claim_id"], row["condition"])
        if key not in latest or latest[key]["status"] != "success":
            latest[key] = row
    return latest
