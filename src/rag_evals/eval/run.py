"""Run the full eval suite over a golden file.

Each golden item is run through retrieve → rerank → generate. Per-query records and
aggregate scores are written to results.json. The CI workflow consumes that file.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv

from ..config import CONFIG
from ..generate import generate
from ..rerank import Reranker
from ..retrieve import HybridRetriever
from .golden import GoldenItem
from .metrics import aggregate


def _faithfulness_score(answer_text: str, retrieved_texts: list[str]) -> float:
    """Cheap lexical-overlap proxy for faithfulness.

    Real systems plug in an LLM-judge here; this proxy is honest about what it measures
    (token overlap) and produces stable numbers for CI gating without an extra LLM call.
    """
    ans_tokens = {t.lower() for t in answer_text.split() if len(t) > 3}
    if not ans_tokens:
        return 1.0
    ctx_tokens: set[str] = set()
    for c in retrieved_texts:
        ctx_tokens |= {t.lower() for t in c.split() if len(t) > 3}
    overlap = len(ans_tokens & ctx_tokens)
    return overlap / len(ans_tokens)


def _load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(GoldenItem(**json.loads(line)))
    return items


def run_evaluation(
    golden_path: Path, index_dir: Path, mock: bool = False
) -> dict:
    items = _load_golden(golden_path)
    retriever = HybridRetriever(index_dir)
    reranker = Reranker()

    records: list[dict] = []
    faithfulness_scores: list[float] = []

    for item in items:
        retrieved = retriever.search(item.question)
        reranked = reranker.rerank(item.question, retrieved)
        rec: dict = {
            "qid": item.qid,
            "question": item.question,
            "expected_doc_ids": item.expected_doc_ids,
            "must_refuse": item.must_refuse,
            "retrieved": [{"doc_id": h.doc_id, "chunk_id": h.chunk_id, "score": h.score} for h in reranked],
        }
        if mock:
            rec["answer"] = "(mock)"
            rec["citations"] = []
            rec["refused"] = False
            faithfulness_scores.append(1.0)
        else:
            try:
                ans = generate(item.question, reranked)
                rec["answer"] = ans.answer
                rec["citations"] = ans.citation_chunk_ids
                rec["refused"] = ans.refused
                faithfulness_scores.append(
                    _faithfulness_score(ans.answer, [h.text for h in reranked])
                )
            except ValueError as e:
                rec["generator_error"] = str(e)
                rec["refused"] = False
                faithfulness_scores.append(0.0)
        records.append(rec)

    scores = aggregate(records)
    scores["faithfulness"] = sum(faithfulness_scores) / max(len(faithfulness_scores), 1)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": CONFIG.as_dict(),
        "n_queries": len(items),
        "scores": scores,
        "records": records,
    }


def _check_thresholds(scores: dict[str, float], thresholds_path: Path) -> list[str]:
    if not thresholds_path.exists():
        return []
    thresholds = yaml.safe_load(thresholds_path.read_text()) or {}
    failures = []
    for name, minimum in thresholds.items():
        actual = scores.get(name)
        if actual is None:
            failures.append(f"{name}: missing from results")
        elif name == "hallucination_rate":
            # lower is better
            if actual > minimum:
                failures.append(f"{name}: {actual:.3f} > {minimum:.3f} (max allowed)")
        elif actual < minimum:
            failures.append(f"{name}: {actual:.3f} < {minimum:.3f} (min required)")
    return failures


@click.command("rag-eval")
@click.option("--golden", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--index", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path("data/index"))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--thresholds", type=click.Path(dir_okay=False, path_type=Path), default=Path("eval_thresholds.yaml"))
@click.option("--mock", is_flag=True, help="Skip the generator call (CI smoke runs without an API key).")
def cli(golden: Path, index: Path, out: Path, thresholds: Path, mock: bool) -> None:
    load_dotenv()
    if not mock and not os.getenv("ANTHROPIC_API_KEY"):
        raise click.ClickException("ANTHROPIC_API_KEY not set; pass --mock to skip generation.")

    result = run_evaluation(golden, index, mock=mock)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    click.echo(f"… ran {result['n_queries']} golden queries")
    for k, v in result["scores"].items():
        click.echo(f"  {k:<20} {v:.3f}")
    click.echo(f"✓ wrote {out}")

    failures = _check_thresholds(result["scores"], thresholds)
    if failures:
        click.echo("\n✗ threshold check failed:")
        for f in failures:
            click.echo(f"  {f}")
        raise SystemExit(1)
    if thresholds.exists():
        click.echo("✓ all thresholds met")


if __name__ == "__main__":
    cli()
