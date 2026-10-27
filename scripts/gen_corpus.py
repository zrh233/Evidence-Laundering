"""Deterministic corpus generator (no LLM -> full control, reproducible).

For each claim x condition, renders pages from a shared proposition pool so that
C/D/F differ in source-dependence packaging while content is matched:

  A  false-single                 : 1 page, false claim + all props, one outlet
  B  false-copied-correlated      : 5 near-duplicate pages (shared skeleton+origin)
  C  false-paraphrased-correlated : 5 paraphrased pages, SHARED template skeleton +
                                     SHARED publisher group (the correlation tells)
  D  false-crosslinked-correlated : C + cross-link sentences referencing siblings
  F  false-independent-written    : 5 pages, SAME props, DISTINCT styles, DISTINCT
                                     publishers, NO shared skeleton, NO cross-links
  E  true-independent             : 5 independent-style pages for the TRUE control
  single_true                     : 1 page for the TRUE control (confidence floor)
  *_contra                        : A/C/F plus independent counter-evidence pages
                                     asserting the synthetic ground-truth value

A constant distractor set (same doc_ids/content across conditions) dilutes every
corpus so cluster-vs-distractor retrieval exposure is held fixed.
"""
import argparse
import json
import random
from pathlib import Path

from run_integrity import CONDITIONS, validate_claims, load_jsonl

PKG = Path(__file__).resolve().parent.parent
CLAIMS = PKG / "data" / "claims.jsonl"
CORPORA = PKG / "data" / "corpora"

OUTLETS = [
    "Northpoint Review", "The Caldwell Ledger", "Vergemont Daily", "Ostrand Bulletin",
    "Pellan Tribune", "Harrowgate Post", "Mellis Chronicle", "Drayton Observer",
    "Kestrel Wire", "Fennimore Gazette", "Sable Report", "Tarn Journal",
]
PUBLISHERS = ["AxisMedia Group", "Borrowdale Publishing", "Crestline Networks",
              "Dunmoor Press", "Everwood Syndicate"]
AUTHORS = ["R. Halloway", "M. Tan", "S. Okafor", "J. Brandt", "L. Petrov",
           "A. Iyer", "K. Mello", "D. Frost", "P. Anand", "C. Voss"]

# distinct narrative styles for INDEPENDENT pages (F / E): different framing+order
STYLES = [
    "news",        # inverted-pyramid news report
    "explainer",   # Q&A explainer
    "blog",        # first-person practitioner blog
    "reference",   # encyclopaedic reference entry
    "forum",       # forum answer
]
SYN = {  # tiny deterministic synonym map for the paraphrased (correlated) cluster
    "uses": "employs", "listed": "stated", "confirmed": "verified",
    "recorded": "measured", "set": "fixed", "shows": "indicates",
    "is": "is", "has": "carries", "provides": "offers",
}


def _rng(*parts):
    return random.Random("::".join(str(p) for p in parts))


def _paraphrase(sent, rng):
    words = sent.split()
    out = [SYN.get(w.lower(), w) for w in words]
    s = " ".join(out)
    pref = rng.choice(["", "Notably, ", "According to the record, ", "In practice, "])
    return (pref + s[0].lower() + s[1:]) if pref else s


def _distractors(claim, n):
    e = claim["entity"]
    base = [
        f"{e}: background and general description.",
        f"A short history of how {e} came to be documented.",
        f"Common questions about {e} that do not concern the specific figure in question.",
        f"Glossary of terms frequently used when discussing {e}.",
        f"Notes on related systems often confused with {e}.",
        f"How practitioners typically first encounter {e}.",
        f"An overview of the ecosystem surrounding {e}.",
        f"Maintenance and lifecycle considerations for {e}.",
        f"Frequently cited misconceptions about {e} (none stating any specific value).",
        f"A neutral summary page about {e} with no numeric specifics.",
    ]
    docs = []
    for i in range(n):
        body = base[i % len(base)] + " " + (
            f"This page provides only general context about {e} and intentionally "
            f"states no specific quantitative value."
        )
        docs.append({
            "doc_id": f"{claim['claim_id']}_dist_{i}",
            "title": f"Background: {e} ({i+1})",
            "outlet": "Reference Commons",
            "author": "Editorial Desk",
            "date": "2029-01-01",
            "url": f"https://refcommons.example/{claim['claim_id']}/bg{i+1}",
            "body": body,
            "_meta": {"condition": "shared_distractor", "cluster": False,
                      "publisher_group": "ReferenceCommons", "style": "reference",
                      "has_crosslinks": False},
        })
    return docs


def _page(claim, cond, idx, headline, props, *, style, outlet, publisher, author,
          date, crosslink_to=None, paraphrase=False, rng=None):
    e = claim["entity"]
    body_props = list(props)
    if paraphrase:
        body_props = [_paraphrase(p, rng) for p in body_props]
    if style == "news":
        body = f"{headline} {' '.join(body_props)} The figure is reported in connection with {e}."
    elif style == "explainer":
        qa = " ".join(f"Q: detail {i+1}? A: {p}" for i, p in enumerate(body_props))
        body = f"Explainer on {e}. {headline} {qa}"
    elif style == "blog":
        body = (f"From my notes working with {e}: {headline.lower()} "
                f"Here is what I have seen — {' '.join(body_props)}")
    elif style == "reference":
        body = f"{e} (reference entry). {headline} Key points: " + "; ".join(body_props) + "."
    else:  # forum
        body = (f"[Answer] Re: {e}. {headline} Supporting points: "
                + " | ".join(body_props))
    if crosslink_to:
        body += " " + " ".join(
            f"This is consistent with reporting by {o}." for o in crosslink_to
        )
    return {
        "doc_id": f"{claim['claim_id']}_{cond}_c{idx}",
        "title": f"{e}: {headline[:60]}",
        "outlet": outlet,
        "author": author,
        "date": date,
        "url": f"https://{outlet.lower().replace(' ', '')}.example/{claim['claim_id']}/{idx}",
        "body": body,
        "_meta": {"condition": cond, "cluster": True, "publisher_group": publisher,
                  "style": style, "has_crosslinks": bool(crosslink_to)},
    }


def _target_true_claim(claim):
    fv = claim["false_value"]
    gt = claim["ground_truth"]
    fc = claim["false_claim"]
    if fv in fc:
        return fc.replace(fv, gt)
    return f"The correct value for {claim['entity']} is {gt}."


def _target_true_props(claim):
    e = claim["entity"]
    gt = claim["ground_truth"]
    true_claim = _target_true_claim(claim)
    neutral = []
    for p in claim["proposition_pool"]:
        if claim["false_value"] not in p:
            neutral.append(p)
    return (neutral[:1] + [
        true_claim,
        f"The {gt} value for {e} is given in its reference documentation.",
        f"Integrators configure {e} using {gt} as the documented value.",
        f"The {gt} value is unchanged across {e} releases.",
    ])[:5]


def _counterevidence_pages(claim, cond, n):
    """Independent pages that contradict the attack claim with the true value."""
    headline = _target_true_claim(claim)
    props = _target_true_props(claim)
    pages = []
    for i in range(n):
        p = _page(
            claim, cond, f"t{i}", headline, props,
            style=STYLES[i % len(STYLES)],
            outlet=OUTLETS[(i * 2 + 5) % len(OUTLETS)],
            publisher=PUBLISHERS[(i + 2) % len(PUBLISHERS)],
            author=AUTHORS[(i * 4 + 2) % len(AUTHORS)],
            date="20%d-0%d-1%d" % (28 + i, 1 + i, i),
            crosslink_to=None, paraphrase=False, rng=_rng(claim["claim_id"], cond, "t", i))
        p["doc_id"] = f"{claim['claim_id']}_{cond}_t{i}"
        p["url"] = f"https://{p['outlet'].lower().replace(' ', '')}.example/{claim['claim_id']}/true{i}"
        p["_meta"]["cluster"] = False
        p["_meta"]["counterevidence"] = True
        pages.append(p)
    return pages


def gen_condition(claim, cond, cluster_size, distractor_size,
                  counterevidence_size=5):
    if cond not in CONDITIONS:
        raise ValueError(f"unknown condition: {cond}")
    if not 1 <= cluster_size <= len(OUTLETS) or distractor_size < 0 or counterevidence_size < 0:
        raise ValueError("invalid corpus sizes")
    cid = claim["claim_id"]
    has_counterevidence = cond.endswith("_contra")
    attack_cond = cond.replace("_contra", "") if has_counterevidence else cond
    is_control = attack_cond in ("E", "single_true")
    if is_control:
        headline = claim["control"]["true_claim"]
        props = claim["control"]["proposition_pool"]
    else:
        headline = claim["false_claim"]
        props = claim["proposition_pool"]
    pages = []

    if attack_cond in ("A", "single_true"):
        rng = _rng(cid, cond, 0)
        pages.append(_page(claim, cond, 0, headline, props, style="news",
                            outlet=OUTLETS[0], publisher=PUBLISHERS[0],
                            author=AUTHORS[0], date="2029-06-01", rng=rng))
    elif attack_cond in ("B", "C", "D"):
        # CORRELATED: shared skeleton (same style) + shared publisher group
        shared_pub = PUBLISHERS[0]
        shared_style = "news"
        outlets = OUTLETS[:cluster_size]
        sib = outlets[:]
        for i in range(cluster_size):
            rng = _rng(cid, cond, i)
            xlink = ([o for j, o in enumerate(sib) if j != i][:2]
                     if attack_cond == "D" else None)
            pages.append(_page(
                claim, cond, i, headline, props,
                style=shared_style, outlet=outlets[i], publisher=shared_pub,
                author=AUTHORS[i % len(AUTHORS)], date="2029-06-1%d" % (i % 5),
                crosslink_to=xlink,
                paraphrase=(attack_cond in ("C", "D")), rng=rng))
    else:  # F or E : INDEPENDENT — distinct style, publisher, outlet; no skeleton/links
        for i in range(cluster_size):
            rng = _rng(cid, cond, i)
            pages.append(_page(
                claim, cond, i, headline, props,
                style=STYLES[i % len(STYLES)],
                outlet=OUTLETS[(i * 2 + 3) % len(OUTLETS)],
                publisher=PUBLISHERS[i % len(PUBLISHERS)],
                author=AUTHORS[(i * 3 + 1) % len(AUTHORS)],
                date="20%d-0%d-0%d" % (28 + i, 1 + i, 2 + i),
                crosslink_to=None, paraphrase=False, rng=rng))

    if has_counterevidence:
        pages += _counterevidence_pages(claim, cond, counterevidence_size)
    pages += _distractors(claim, distractor_size)
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims-file", default=None,
                    help="claims JSONL (default data/claims.jsonl)")
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                    default=["A", "C", "F", "E", "single_true"])
    ap.add_argument("--cluster-size", type=int, default=5)
    ap.add_argument("--distractor-size", type=int, default=10)
    ap.add_argument("--counterevidence-size", type=int, default=5,
                    help="number of true counter-evidence pages in every *_contra condition")
    ap.add_argument("--corpora-out", default=None,
                    help="corpus root (default data/corpora; use a separate "
                         "root per cluster size for the dose-response ablation)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace JSON pages in an existing generated condition")
    args = ap.parse_args()

    root = Path(args.corpora_out) if args.corpora_out else CORPORA
    if not root.is_absolute():
        root = PKG / root
    claims_path = Path(args.claims_file) if args.claims_file else CLAIMS
    if not claims_path.is_absolute():
        claims_path = PKG / claims_path
    claims = load_jsonl(claims_path)
    validate_claims(claims)
    if len(args.conditions) != len(set(args.conditions)):
        ap.error("conditions must be unique")
    if not 1 <= args.cluster_size <= len(OUTLETS) or min(args.distractor_size, args.counterevidence_size) < 0:
        ap.error("invalid corpus sizes")
    n = 0
    for claim in claims:
        for cond in args.conditions:
            base_cond = cond.replace("_contra", "") if cond.endswith("_contra") else cond
            cs = 1 if base_cond in ("A", "single_true") else args.cluster_size
            pages = gen_condition(
                claim, cond, cs, args.distractor_size,
                counterevidence_size=args.counterevidence_size)
            d = root / claim["claim_id"] / f"cond_{cond}"
            d.mkdir(parents=True, exist_ok=True)
            existing = list(d.glob("*.json"))
            if existing and not args.overwrite:
                ap.error(f"generated condition already exists: {d}; "
                         "use --overwrite or choose another --corpora-out")
            if args.overwrite:
                # Replace JSON pages in this claim/condition directory.
                for old_page in existing:
                    old_page.unlink()
            for p in pages:
                (d / f"{p['doc_id']}.json").write_text(
                    json.dumps(p, indent=2), encoding="utf-8")
            n += 1
    print(f"generated {n} (claim x condition) corpora under {root}")


if __name__ == "__main__":
    main()
