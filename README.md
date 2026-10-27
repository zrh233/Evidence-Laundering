# Evidence Laundering: Deep Research Agents Mistake Correlated Sources for Independent Evidence

Ruihua Zhou, Hao Huang, Yijie Zheng, Jiatang Luo, Yunpeng Li, Yuling Liu, and Dan Du.

[Paper](https://aclanthology.org/2026.emnlp-main.509/)

Requires **Python 3.10+**. Includes synthetic corpora, retrieval, two text-similarity
defenses, and experiment scripts. 

## Quick start

Run from the repository root. No dependencies or API key are needed:

```bash
python3 scripts/gen_claims.py --n 3 --out data/claims.jsonl
python3 scripts/gen_corpus.py
python3 scripts/run_experiment.py --stub --claims 3 --outdir smoke
python3 scripts/extract_metrics.py --results-dir results/smoke
python3 scripts/analyze_results.py --results-dir results/smoke
```

Outputs are saved in `results/smoke/`. Add `--defense shingle` or
`--defense skeleton` to use a defense, with a new `--outdir` for each setting.
Repeating the same runner command resumes that run.

## API model

After generating the data above, install the client and set your credentials:

```bash
python3 -m pip install -r requirements-api.txt
export OPENAI_API_KEY='your-api-key'
export EVIDENCE_LAUNDERING_MODEL='your-model-name'
python3 scripts/run_experiment.py --claims 3 --outdir api
python3 scripts/extract_metrics.py --results-dir results/api
python3 scripts/analyze_results.py --results-dir results/api
```

Set `OPENAI_BASE_URL` if using an OpenAI-compatible endpoint.

## Cite

```bibtex
@inproceedings{zhou-etal-2026-evidence,
    title = "Evidence Laundering: Deep Research Agents Mistake Correlated Sources for Independent Evidence",
    author = "Zhou, Ruihua  and
      Huang, Hao  and
      Zheng, Yijie  and
      Luo, JiaTang  and
      Li, Yunpeng  and
      Liu, Yuling  and
      Du, Dan",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.509/",
    pages = "11682--11693",
    ISBN = "979-8-89176-459-0",
    abstract = "Deep research agents often rely on agreement among retrieved documents when deciding whether a claim is reliable. This heuristic breaks when the agreeing documents are not independent: several independent reports can support a claim, but several copies of the same false claim should not. We study evidence laundering, a content-only attack that publishes one false claim across paraphrased pages that appear to provide support from multiple sources. We evaluate this failure with deterministic matched corpora that present the same false claim and supporting propositions as a single source, a paraphrased correlated cluster, or an apparent independence control, with matched retrieval exposure. Experiments show that agents often assign laundered clusters credence comparable to the apparent independence control and become more willing to publish the false claim. Within the matched evaluation, a structural provenance gate substantially reduces this failure, suggesting that source independence modeling should precede answer synthesis."
}
```