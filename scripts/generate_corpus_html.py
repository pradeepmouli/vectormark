"""Generate an inspectable HTML gallery for the verification SVG corpus.

Run:
    PYTHONPATH=src ./.venv/bin/python scripts/generate_corpus_html.py --output scratch/corpus-html
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import json
import shutil
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vectormark import Options, idealize

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class CorpusEntry:
    name: str
    mode: str
    image_factory: Callable[[], np.ndarray]
    options: Options


def default_entries() -> list[CorpusEntry]:
    """Verification corpus entries rendered by the gallery.

    The committed byte-identical corpus is the source of truth for the baseline
    images/options. Optimizer entries are added for the same non-flattened images
    so the experimental path can be inspected alongside the legacy pipeline.
    """
    from tests.optimizer.test_integration import _asymmetric_cloud_image, _disk_image
    from tests.test_candidate_byte_identical import CASES

    entries = [
        CorpusEntry(name=name, mode="current", image_factory=factory, options=opts)
        for name, factory, opts in CASES
    ]

    seen: set[str] = set()
    for name, factory, opts in CASES:
        base_name = name.removesuffix("_flatten")
        if base_name in seen:
            continue
        seen.add(base_name)
        entries.append(
            CorpusEntry(
                name=base_name,
                mode="optimizer",
                image_factory=factory,
                options=dataclasses.replace(opts, optimizer=True, flatten=False),
            )
        )

    entries.extend(
        [
            CorpusEntry("disk", "optimizer", _disk_image, Options(optimizer=True)),
            CorpusEntry(
                "asymmetric_cloud",
                "optimizer",
                _asymmetric_cloud_image,
                Options(optimizer=True),
            ),
        ]
    )
    return entries


def _diagnostics_json(report: Any) -> str:
    diagnostics = report.diagnostics.to_dict() if report.diagnostics is not None else None
    payload = {
        "strategies": dict(report.strategies),
        "gradients": report.gradients,
        "elements": report.elements,
        "axes": [dataclasses.asdict(axis) for axis in report.axes],
        "symmetry": report.symmetry,
        "diagnostics": diagnostics,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _options_json(options: Options) -> str:
    return json.dumps(dataclasses.asdict(options), indent=2, sort_keys=True, default=repr)


def _safe_filename(entry: CorpusEntry) -> str:
    return f"{entry.mode}-{entry.name}.svg".replace("/", "_")


def _input_filename(entry: CorpusEntry) -> str:
    return f"{entry.mode}-{entry.name}.png".replace("/", "_")


def generate_corpus_html(output: Path, entries: Iterable[CorpusEntry] | None = None) -> Path:
    output = Path(output)
    if output.exists():
        shutil.rmtree(output)
    svg_dir = output / "svg"
    input_dir = output / "input"
    svg_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    cards: list[str] = []
    for entry in sorted(entries or default_entries(), key=lambda item: (item.mode, item.name)):
        image = np.asarray(entry.image_factory(), dtype=np.uint8)
        input_name = _input_filename(entry)
        input_path = input_dir / input_name
        Image.fromarray(image).save(input_path)

        svg, report = idealize(image, options=entry.options, report=True)
        svg_name = _safe_filename(entry)
        svg_path = svg_dir / svg_name
        svg_path.write_text(svg)

        rel_svg = Path("svg") / svg_name
        rel_input = Path("input") / input_name
        cards.append(
            f"""
    <article class="entry">
      <header>
        <h2>{html.escape(entry.mode)} / {html.escape(entry.name)}</h2>
        <a href="{html.escape(str(rel_svg))}">{html.escape(str(rel_svg))}</a>
      </header>
      <section class="renders">
        <figure>
          <figcaption>Input PNG</figcaption>
          <img src="{html.escape(str(rel_input))}" alt="{html.escape(entry.name)} input">
        </figure>
        <figure>
          <figcaption>Generated SVG</figcaption>
          <object data="{html.escape(str(rel_svg))}" type="image/svg+xml"></object>
        </figure>
      </section>
      <details>
        <summary>SVG</summary>
        <pre><code>{html.escape(svg)}</code></pre>
      </details>
      <details>
        <summary>Diagnostics</summary>
        <pre><code>{html.escape(_diagnostics_json(report))}</code></pre>
      </details>
      <details>
        <summary>Options</summary>
        <pre><code>{html.escape(_options_json(entry.options))}</code></pre>
      </details>
    </article>"""
        )

    (output / "index.html").write_text(_html_document(cards))
    return output / "index.html"


def _html_document(cards: list[str]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>vectormark SVG corpus</title>
  <style>
    body {{
      margin: 24px;
      background: #f7f7f5;
      color: #161616;
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    h1 {{ margin: 0 0 16px; font-size: 24px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
      gap: 16px;
      align-items: start;
    }}
    .entry {{
      border: 1px solid #d6d6d2;
      border-radius: 6px;
      background: white;
      padding: 12px;
    }}
    .entry header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 10px;
    }}
    h2 {{ margin: 0; font-size: 14px; }}
    a {{ color: #2457a7; }}
    .renders {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    figure {{ margin: 0; }}
    figcaption {{
      margin-bottom: 4px;
      color: #555;
      font-size: 12px;
      font-weight: 600;
    }}
    object,
    img {{
      display: block;
      width: 100%;
      height: 240px;
      border: 1px solid #e7e7e2;
      background: white;
      object-fit: contain;
    }}
    details {{ margin-top: 10px; }}
    summary {{ cursor: pointer; font-weight: 600; }}
    pre {{
      max-height: 360px;
      overflow: auto;
      padding: 10px;
      border: 1px solid #e5e5df;
      border-radius: 4px;
      background: #fbfbf8;
      font-size: 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
  </style>
</head>
<body>
  <h1>vectormark SVG corpus</h1>
  <main class="grid">{''.join(cards)}
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the vectormark SVG corpus gallery.")
    parser.add_argument("--output", type=Path, default=Path("scratch/corpus-html"))
    args = parser.parse_args()
    index = generate_corpus_html(args.output)
    print(index)


if __name__ == "__main__":
    main()
