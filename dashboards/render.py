"""Render results.json into a static HTML report. Output goes to dashboards/build/."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>rag-evals report</title>
<style>
  body {{ font-family: ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif; margin: 2rem auto; max-width: 920px; padding: 0 1rem; color: #1c1c1c; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
  .meta {{ color: #6b6b6b; margin-bottom: 1.5rem; font-size: 0.9rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
  th, td {{ text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #eee; }}
  th {{ background: #fafafa; font-weight: 600; }}
  td.num {{ font-variant-numeric: tabular-nums; text-align: right; font-family: ui-monospace, "SF Mono", monospace; }}
  .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.8rem; }}
  .pass {{ background: #e8f5e9; color: #2e7d32; }}
  .fail {{ background: #ffebee; color: #c62828; }}
  details {{ margin-bottom: 0.5rem; }}
  summary {{ cursor: pointer; padding: 0.25rem 0; }}
  pre {{ background: #f6f8fa; padding: 0.75rem; overflow-x: auto; border-radius: 4px; }}
</style>
</head>
<body>
<h1>rag-evals report</h1>
<div class="meta">Generated {generated_at} · {n_queries} queries · model {model}</div>

<h2>Aggregate scores</h2>
<table>
  <thead><tr><th>Metric</th><th class="num">Score</th></tr></thead>
  <tbody>{score_rows}</tbody>
</table>

<h2>Per-query records</h2>
{query_blocks}

</body>
</html>
"""


def _score_row(name: str, value: float) -> str:
    return f'<tr><td>{name}</td><td class="num">{value:.3f}</td></tr>'


def _query_block(rec: dict) -> str:
    retrieved = "\n".join(
        f"  {h['doc_id']:<25} chunk={h['chunk_id']:<35} score={h['score']:.3f}"
        for h in rec.get("retrieved", [])[:5]
    )
    answer = rec.get("answer", "(no answer)")
    cites = ", ".join(rec.get("citations", [])) or "(none)"
    err = rec.get("generator_error")
    err_html = f'<p class="pill fail">generator error: {err}</p>' if err else ""
    return f"""
<details>
<summary><strong>{rec['qid']}</strong> · {rec['question']}</summary>
<pre>expected_doc_ids: {rec['expected_doc_ids']}
must_refuse:      {rec.get('must_refuse', False)}
refused:          {rec.get('refused', False)}

top retrieved:
{retrieved}

answer:
{answer}

citations: {cites}
</pre>
{err_html}
</details>
"""


@click.command()
@click.option("--results", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), required=True)
def main(results: Path, out: Path) -> None:
    data = json.loads(results.read_text())
    out.mkdir(parents=True, exist_ok=True)

    scores = data.get("scores", {})
    score_rows = "".join(_score_row(k, v) for k, v in scores.items())
    query_blocks = "".join(_query_block(r) for r in data.get("records", []))

    html = HTML.format(
        generated_at=data.get("generated_at", datetime.utcnow().isoformat()),
        n_queries=data.get("n_queries", 0),
        model=data.get("config", {}).get("generator_model", "(unknown)"),
        score_rows=score_rows,
        query_blocks=query_blocks,
    )
    (out / "index.html").write_text(html)
    click.echo(f"✓ wrote {out / 'index.html'}")


if __name__ == "__main__":
    main()
