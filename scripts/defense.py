"""Cluster retrieved pages by body-text similarity and keep ranked representatives.

Shingle mode compares word n-grams; skeleton mode masks content tokens first.
Each connected component contributes its highest-ranked page.
"""
import re

_WORD = re.compile(r"[a-z0-9]+")
_TOKP = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")

# Keep function words and punctuation; mask other tokens in the skeleton.
_STOP = set((
    "the a an of to in on for and or but with as is are was were be been by at "
    "from this that these those it its their his her you we i they he she not "
    "no nor so if then than into over under about between within without per "
    "via re q"
).split())


def _shingles(text, n=5):
    toks = _WORD.findall(text.lower())
    if len(toks) < n:
        return frozenset(toks)
    return frozenset(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))


def _skeleton(text):
    out = []
    for tok in _TOKP.findall(text.lower()):
        if tok.isalnum():
            out.append(tok if tok in _STOP else "#")
        else:
            out.append(tok)
    return re.sub(r"(# )+#", "#", " ".join(out))


def _skel_shingles(text, n=4):
    toks = _skeleton(text).split()
    if len(toks) < n:
        return frozenset(toks)
    return frozenset(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))


def _jacc(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _singleton_info(docs, name="none", tau=None):
    components = [[d["doc_id"]] for d in docs]
    return {
        "defense": name,
        "tau": tau,
        "n_in": len(docs),
        "n_kept": len(docs),
        "max_pair_jaccard": 0.0,
        "mean_pair_jaccard": 0.0,
        "components": components,
        "doc_to_component": {did: i for i, (did,) in enumerate(components)},
    }


def _collapse(docs, fp, tau, name):
    """Single-linkage clustering with representatives chosen by retrieval rank."""
    docs = list(docs)
    if len(docs) <= 1:
        return docs, _singleton_info(docs, name, tau)

    sh = [fp(d["body"]) for d in docs]
    m = len(docs)
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    max_j = 0.0
    pair_vals = []
    for i in range(m):
        for j in range(i + 1, m):
            jj = _jacc(sh[i], sh[j])
            pair_vals.append(jj)
            max_j = max(max_j, jj)
            if jj >= tau:
                parent[find(i)] = find(j)

    components_by_root = {}
    for i in range(m):
        r = find(i)
        components_by_root.setdefault(r, []).append(i)

    # Preserve retrieval order across representatives and within components.
    component_indices = sorted(components_by_root.values(), key=lambda xs: min(xs))
    reps = [docs[min(xs)] for xs in component_indices]
    components = [[docs[i]["doc_id"] for i in xs] for xs in component_indices]
    doc_to_component = {
        did: component_id
        for component_id, component in enumerate(components)
        for did in component
    }
    info = {
        "defense": name,
        "tau": tau,
        "n_in": m,
        "n_kept": len(reps),
        "max_pair_jaccard": round(max_j, 4),
        "mean_pair_jaccard": round(sum(pair_vals) / len(pair_vals), 4),
        "components": components,
        "doc_to_component": doc_to_component,
    }
    return reps, info


def apply_defense(docs, mode, cfg):
    """Apply none, word-shingle, or masked-skeleton clustering."""
    docs = list(docs)
    if mode == "none":
        return docs, _singleton_info(docs)
    if mode not in ("shingle", "skeleton"):
        raise ValueError(f"unknown defense mode: {mode}")
    d = cfg["defense"]
    if mode == "shingle":
        return _collapse(docs, lambda b: _shingles(b, d["ngram"]),
                         d["tau_shingle"], "shingle")
    return _collapse(docs, lambda b: _skel_shingles(b, d["skel_ngram"]),
                     d["tau_skeleton"], "skeleton")


def gate_ranked_docs(ranked_docs, mode, cfg, top_k=None):
    """Return up to top_k representatives from a best-first candidate pool."""
    kept, info = apply_defense(ranked_docs, mode, cfg)
    if top_k is not None:
        kept = kept[:top_k]
    info["n_returned"] = len(kept)
    info["returned_doc_ids"] = [d["doc_id"] for d in kept]
    return kept, info


def effective_source_count(info, doc_ids):
    """Count components containing the selected evaluation documents."""
    mapping = info["doc_to_component"]
    return len({mapping[did] for did in doc_ids})
