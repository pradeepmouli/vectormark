"""Generate an inspectable HTML gallery for the verification SVG corpus.

Run:
    PYTHONPATH=src ./.venv/bin/python scripts/generate_corpus_html.py
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import json
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

DEFAULT_CORPUS_INPUT = REPO_ROOT / "corpus" / "input"
DEFAULT_CORPUS_OUTPUT = Path("corpus")
LEGACY_CORPUS_INPUT = REPO_ROOT / "scratch" / "real-logos"


@dataclass(frozen=True)
class CorpusEntry:
    name: str
    mode: str
    image_factory: Callable[[], np.ndarray]
    options: Options


def default_entries() -> list[CorpusEntry]:
    """Verification corpus entries rendered by the gallery.

    Prefer the checked-in/drop-in corpus directory, then the legacy untracked
    real-logo corpus used for acceptance inspection. For each logo, render the
    current default pipeline and the optimizer pipeline.
    Fall back to a tiny synthetic set when the local corpus is unavailable.
    """
    for corpus in (DEFAULT_CORPUS_INPUT, LEGACY_CORPUS_INPUT):
        entries = corpus_entries(corpus)
        if entries:
            return entries

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


def corpus_entries(corpus: Path) -> list[CorpusEntry]:
    """Build paired current/optimizer entries from PNG files in a corpus directory."""
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
    if not corpus.exists():
        return []
    pngs = [
        path for path in sorted(corpus.glob("*.png"))
        if not path.name.startswith(".") and not path.stem.startswith(skip_prefixes)
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


def _entry_id(entry: CorpusEntry) -> str:
    return f"{entry.mode}-{entry.name}".replace("/", "_")


def _diagnostics_filename(entry: CorpusEntry) -> str:
    return f"{_entry_id(entry)}.json"


def _options_filename(entry: CorpusEntry) -> str:
    return f"{_entry_id(entry)}.json"


def _raw_svg_filename(entry: CorpusEntry) -> str:
    return f"{_entry_id(entry)}.svg.txt"


def _gallery_options(
    options: Options,
    working_max_dim: int | None,
    *,
    cubic_paths: bool = False,
) -> Options:
    updates: dict[str, object] = {}
    if working_max_dim is not None and options.working_max_dim is None:
        updates["working_max_dim"] = working_max_dim
    if cubic_paths:
        updates["cubic_paths"] = True
    if not updates:
        return options
    return dataclasses.replace(options, **updates)


def _sorted_entries(entries: Iterable[CorpusEntry]) -> list[CorpusEntry]:
    mode_order = {"current": 0, "optimizer": 1}
    return sorted(entries, key=lambda item: (item.name, mode_order.get(item.mode, 99)))


def _manifest_payload(entries: list[CorpusEntry]) -> dict[str, Any]:
    return {
        "version": 2,
        "entries": [
            {
                "id": _entry_id(entry),
                "name": entry.name,
                "mode": entry.mode,
            }
            for entry in entries
        ],
    }


def _matches_filter(entry: CorpusEntry, selectors: set[str]) -> bool:
    if not selectors:
        return True
    entry_id = _entry_id(entry)
    return bool({
        entry.name,
        entry.mode,
        f"{entry.mode}/{entry.name}",
        entry_id,
        f"{entry.mode}-{entry.name}",
    } & selectors)


def _write_index_if_needed(output: Path, entries: list[CorpusEntry], *, force: bool = False) -> None:
    manifest_path = output / "corpus-manifest.json"
    manifest = _manifest_payload(entries)
    current_manifest = None
    if manifest_path.exists():
        current_manifest = json.loads(manifest_path.read_text())
    if not force and current_manifest == manifest and (output / "index.html").exists():
        return

    cards = [_entry_card(entry) for entry in entries]
    (output / "index.html").write_text(_html_document(cards))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _entry_card(entry: CorpusEntry) -> str:
    svg_name = _safe_filename(entry)
    input_name = _input_filename(entry)
    raw_name = _raw_svg_filename(entry)
    diagnostics_name = _diagnostics_filename(entry)
    options_name = _options_filename(entry)

    rel_svg = Path("svg") / svg_name
    rel_input = Path("rendered-input") / input_name
    rel_raw = Path("raw") / raw_name
    rel_diagnostics = Path("diagnostics") / diagnostics_name
    rel_options = Path("options") / options_name
    return f"""
    <article class="entry" id="{html.escape(_entry_id(entry))}">
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
        <iframe class="code-frame" src="{html.escape(str(rel_raw))}" title="{html.escape(entry.name)} SVG source"></iframe>
      </details>
      <details>
        <summary>Diagnostics</summary>
        <iframe class="code-frame" src="{html.escape(str(rel_diagnostics))}" title="{html.escape(entry.name)} diagnostics"></iframe>
      </details>
      <details>
        <summary>Options</summary>
        <iframe class="code-frame" src="{html.escape(str(rel_options))}" title="{html.escape(entry.name)} options"></iframe>
      </details>
    </article>"""


def generate_corpus_html(
    output: Path,
    entries: Iterable[CorpusEntry] | None = None,
    *,
    working_max_dim: int | None = None,
    cubic_paths: bool = False,
    only: Iterable[str] | None = None,
    rebuild_index: bool = False,
) -> Path:
    output = Path(output)
    svg_dir = output / "svg"
    input_dir = output / "rendered-input"
    raw_dir = output / "raw"
    diagnostics_dir = output / "diagnostics"
    options_dir = output / "options"
    svg_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    options_dir.mkdir(parents=True, exist_ok=True)

    all_entries = _sorted_entries(entries or default_entries())
    _write_index_if_needed(output, all_entries, force=rebuild_index)

    selectors = set(only or ())
    for entry in [entry for entry in all_entries if _matches_filter(entry, selectors)]:
        image = np.asarray(entry.image_factory(), dtype=np.uint8)
        input_name = _input_filename(entry)
        input_path = input_dir / input_name
        Image.fromarray(image).save(input_path)

        options = _gallery_options(entry.options, working_max_dim, cubic_paths=cubic_paths)
        print(f"rendering {entry.mode}/{entry.name}", file=sys.stderr)
        svg, report = idealize(image, options=options, report=True)
        svg_name = _safe_filename(entry)
        svg_path = svg_dir / svg_name
        svg_path.write_text(svg)
        (raw_dir / _raw_svg_filename(entry)).write_text(svg)
        (diagnostics_dir / _diagnostics_filename(entry)).write_text(_diagnostics_json(report))
        (options_dir / _options_filename(entry)).write_text(_options_json(options))
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
    pre,
    .code-frame {{
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
    .code-frame {{
      display: block;
      box-sizing: border-box;
      width: 100%;
      height: 360px;
      white-space: normal;
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
    parser.add_argument("--output", type=Path, default=DEFAULT_CORPUS_OUTPUT)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory of corpus PNGs. Defaults to corpus/input, then scratch/real-logos.",
    )
    parser.add_argument(
        "--working-max-dim",
        type=int,
        default=None,
        help="Opt-in downscale long side for faster gallery generation.",
    )
    parser.add_argument(
        "--cubic-paths",
        action="store_true",
        help="Opt in to cubic Bézier fitting for curved path runs.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Render only matching entries. Accepts name, mode, mode/name, or mode-name. May be repeated.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rewrite index.html even when the corpus manifest is unchanged.",
    )
    args = parser.parse_args()
    entries = corpus_entries(args.input_dir) if args.input_dir is not None else None
    index = generate_corpus_html(
        args.output,
        entries,
        working_max_dim=args.working_max_dim,
        cubic_paths=args.cubic_paths,
        only=args.only,
        rebuild_index=args.rebuild_index,
    )
    print(index)


if __name__ == "__main__":
    main()
