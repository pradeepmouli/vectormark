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

    Prefer the untracked real-logo corpus used for acceptance inspection. For
    each logo, render the current default pipeline and the optimizer pipeline.
    Fall back to a tiny synthetic set when the local corpus is unavailable.
    """
    real = _real_logo_entries()
    if real:
        return real

    from tests.optimizer.test_integration import _asymmetric_cloud_image, _disk_image

    entries = [
        CorpusEntry("disk", "current", _disk_image, Options()),
        CorpusEntry("disk", "optimizer", _disk_image, Options(optimizer=True)),
        CorpusEntry("asymmetric_cloud", "current", _asymmetric_cloud_image, Options()),
        CorpusEntry(
            "asymmetric_cloud",
            "optimizer",
            _asymmetric_cloud_image,
            Options(optimizer=True),
        )
    ]
    return entries


def _image_factory(path: Path) -> Callable[[], np.ndarray]:
    def _load() -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

    return _load


def _real_logo_entries() -> list[CorpusEntry]:
    corpus = REPO_ROOT / "scratch" / "real-logos"
    if not corpus.exists():
        return []

    skip_prefixes = (
        "cap",
        "cmp_",
        "contact_sheet",
        "corpus",
        "cubic_demo",
        "lever_",
        "node",
        "out_",
        "poc_",
        "pre",
        "rebuild",
        "s40",
        "settir_cmp",
        "settir_spike",
        "threeway",
        "vbird_",
        "zoom",
    )
    pngs = [
        path for path in sorted(corpus.glob("*.png"))
        if not path.stem.startswith(skip_prefixes)
    ]
    if not any(path.stem == "vbird" for path in pngs):
        vbird = corpus / "vbird.png"
        if vbird.exists():
            pngs.append(vbird)
            pngs.sort()

    entries: list[CorpusEntry] = []
    for path in pngs:
        factory = _image_factory(path)
        entries.append(CorpusEntry(path.stem, "current", factory, Options()))
        entries.append(CorpusEntry(path.stem, "optimizer", factory, Options(optimizer=True)))
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


def _gallery_options(options: Options, working_max_dim: int | None) -> Options:
    if working_max_dim is None or options.working_max_dim is not None:
        return options
    return dataclasses.replace(options, working_max_dim=working_max_dim)


def generate_corpus_html(
    output: Path,
    entries: Iterable[CorpusEntry] | None = None,
    *,
    working_max_dim: int | None = None,
) -> Path:
    output = Path(output)
    if output.exists():
        shutil.rmtree(output)
    svg_dir = output / "svg"
    input_dir = output / "input"
    svg_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    cards: list[str] = []
    mode_order = {"current": 0, "optimizer": 1}
    for entry in sorted(entries or default_entries(), key=lambda item: (item.name, mode_order.get(item.mode, 99))):
        image = np.asarray(entry.image_factory(), dtype=np.uint8)
        input_name = _input_filename(entry)
        input_path = input_dir / input_name
        Image.fromarray(image).save(input_path)

        options = _gallery_options(entry.options, working_max_dim)
        print(f"rendering {entry.mode}/{entry.name}", file=sys.stderr)
        svg, report = idealize(image, options=options, report=True)
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
        <pre><code>{html.escape(_options_json(options))}</code></pre>
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
    parser.add_argument(
        "--working-max-dim",
        type=int,
        default=None,
        help="Opt-in downscale long side for faster gallery generation.",
    )
    args = parser.parse_args()
    index = generate_corpus_html(
        args.output,
        working_max_dim=args.working_max_dim,
    )
    print(index)


if __name__ == "__main__":
    main()
