"""Generate an inspectable HTML gallery for the verification SVG corpus.

Run:
    PYTHONPATH=src ./.venv/bin/python scripts/generate_corpus_html.py
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import html
import json
import pickle
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vectormark import Options, idealize
from vectormark.pipeline import _flatten_on_white

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CORPUS_INPUT = REPO_ROOT / "corpus" / "input"
DEFAULT_CORPUS_OUTPUT = Path("corpus")
DEFAULT_CORPUS_MANIFEST = REPO_ROOT / "corpus" / "sources.json"
DEFAULT_CORPUS_CACHE = REPO_ROOT / "corpus" / "cache"
LEGACY_CORPUS_INPUT = REPO_ROOT / "scratch" / "real-logos"


@dataclass(frozen=True)
class CorpusEntry:
    name: str
    mode: str
    image_factory: Callable[[], np.ndarray]
    options: Options


def default_entries() -> list[CorpusEntry]:
    """Verification corpus entries rendered by the gallery.

    Prefer the local ignored URL manifest, then the local ignored drop-in corpus
    directory, then the legacy untracked real-logo corpus used for acceptance
    inspection. For each logo, render the current default pipeline and the
    optimizer pipeline.
    Fall back to a tiny synthetic set when the local corpus is unavailable.
    """
    entries = source_manifest_entries(DEFAULT_CORPUS_MANIFEST, DEFAULT_CORPUS_CACHE)
    if entries:
        return entries

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
        with Image.open(path) as image:
            return _flatten_on_white(image)

    return _load


def _cache_filename(name: str, url: str, filename: str | None = None) -> str:
    if filename:
        suffix = Path(filename).suffix
    else:
        suffix = Path(urllib.parse.urlparse(url).path).suffix
    safe_name = name.replace("/", "_")
    return f"{safe_name}{suffix or '.png'}"


def _download_source(url: str, destination: Path, *, refresh: bool = False) -> Path:
    if destination.exists() and not refresh:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "vectormark-corpus/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        tmp.write_bytes(response.read())
    tmp.replace(destination)
    return destination


def _downloaded_image_factory(
    name: str,
    url: str,
    cache_dir: Path,
    *,
    filename: str | None = None,
    refresh: bool = False,
) -> Callable[[], np.ndarray]:
    cache_path = cache_dir / _cache_filename(name, url, filename)

    def _load() -> np.ndarray:
        return _image_factory(_download_source(url, cache_path, refresh=refresh))()

    return _load


def _source_items(manifest: Any) -> list[dict[str, str]]:
    raw_entries = manifest.get("entries", manifest) if isinstance(manifest, dict) else manifest
    if isinstance(raw_entries, dict):
        return [{"name": str(name), "url": str(url)} for name, url in raw_entries.items()]
    if not isinstance(raw_entries, list):
        raise ValueError("source manifest must be a list, a name-to-url object, or an object with entries")

    items: list[dict[str, str]] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise ValueError("source manifest entries must be objects")
        if "name" not in raw or "url" not in raw:
            raise ValueError("source manifest entries require name and url")
        item = {"name": str(raw["name"]), "url": str(raw["url"])}
        if "filename" in raw:
            item["filename"] = str(raw["filename"])
        items.append(item)
    return items


def source_manifest_entries(
    manifest_path: Path,
    cache_dir: Path,
    *,
    refresh: bool = False,
) -> list[CorpusEntry]:
    """Build paired entries from a local URL manifest, downloading images on demand."""
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    entries: list[CorpusEntry] = []
    for item in sorted(_source_items(manifest), key=lambda source: source["name"]):
        factory = _downloaded_image_factory(
            item["name"],
            item["url"],
            cache_dir,
            filename=item.get("filename"),
            refresh=refresh,
        )
        entries.append(CorpusEntry(item["name"], "current", factory, Options()))
        entries.append(CorpusEntry(item["name"], "optimizer", factory, Options(optimizer=True)))
    return entries


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


def _diagnostics_payload(report: Any) -> dict[str, Any]:
    diagnostics = report.diagnostics.to_dict() if report.diagnostics is not None else None
    return {
        "strategies": dict(report.strategies),
        "gradients": report.gradients,
        "elements": report.elements,
        "axes": [dataclasses.asdict(axis) for axis in report.axes],
        "symmetry": report.symmetry,
        "diagnostics": diagnostics,
    }


def _diagnostics_json(report: Any) -> str:
    payload = _diagnostics_payload(report)
    return json.dumps(payload, indent=2, sort_keys=True)


def _options_json(options: Options) -> str:
    return json.dumps(dataclasses.asdict(options), indent=2, sort_keys=True, default=repr)


def _trace_cache_key(entry: CorpusEntry, image: np.ndarray, options: Options) -> str:
    arr = np.ascontiguousarray(image, dtype=np.uint8)
    payload = {
        "version": 1,
        "entry": entry.name,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "options": dataclasses.asdict(options),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(payload, sort_keys=True, default=repr).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()[:20]


def _trace_cache_path(cache_dir: Path, entry: CorpusEntry, image: np.ndarray, options: Options) -> Path:
    safe_name = entry.name.replace("/", "_")
    return cache_dir / f"trace-{safe_name}-{_trace_cache_key(entry, image, options)}.pkl"


def _load_or_create_trace(cache_path: Path, image: np.ndarray, options: Options):
    from vectormark.optimizer.trace import trace_regions

    if cache_path.exists():
        payload = pickle.loads(cache_path.read_bytes())
        if payload.get("version") == 1:
            return payload["objects"], payload["masks"]

    objects, masks = trace_regions(image, options)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.write_bytes(pickle.dumps({"version": 1, "objects": objects, "masks": masks}))
    tmp.replace(cache_path)
    return objects, masks


def _idealize_optimizer_with_trace_cache(
    image: np.ndarray,
    *,
    options: Options,
    entry: CorpusEntry,
    cache_dir: Path,
):
    from vectormark.emit import render_svg_doc
    from vectormark.optimizer.framework import optimize
    from vectormark.pipeline import (
        _condition_input,
        _optimizer_passes,
        _optimizer_report,
        _prefer_optimizer_svg,
        _render_optimizer_body,
        _set_svg_output_size,
    )

    orig_h, orig_w = image.shape[:2]
    working = _condition_input(np.asarray(image, dtype=np.uint8), options.working_max_dim)
    h0, w0 = working.shape[:2]
    cache_path = _trace_cache_path(cache_dir, entry, working, options)
    objects, masks = _load_or_create_trace(cache_path, working, options)
    optimized = optimize(objects, masks, _optimizer_passes(options))
    trace_body, trace_defs = _render_optimizer_body(objects, flatten=options.flatten)
    trace_svg = render_svg_doc(w0, h0, trace_body, trace_defs)
    optimized_body, optimized_defs = _render_optimizer_body(optimized, flatten=options.flatten)
    optimized_svg = render_svg_doc(w0, h0, optimized_body, optimized_defs)
    if _prefer_optimizer_svg(trace_svg, optimized_svg):
        svg = optimized_svg
        report = _optimizer_report(optimized, options)
    else:
        svg = trace_svg
        report = _optimizer_report(
            objects,
            options,
            fallback_reason="optimized output is structurally larger than trace baseline",
        )
    if (working.shape[1], working.shape[0]) != (orig_w, orig_h):
        svg = _set_svg_output_size(svg, orig_w, orig_h)
    return svg, report


def _safe_filename(entry: CorpusEntry) -> str:
    return f"{entry.mode}-{entry.name}.svg".replace("/", "_")


def _input_filename(entry: CorpusEntry) -> str:
    return f"{entry.mode}-{entry.name}.png".replace("/", "_")


def _entry_id(entry: CorpusEntry) -> str:
    return f"{entry.mode}-{entry.name}".replace("/", "_")


def _diagnostics_filename(entry: CorpusEntry) -> str:
    return f"{_entry_id(entry)}.json"


def _diagnostics_summary_filename(entry: CorpusEntry) -> str:
    return f"{_entry_id(entry)}.html"


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


def _fmt_scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt_scalar(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _summary_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "<p class=\"empty\">none</p>"
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(f"<td>{html.escape(_fmt_scalar(value))}</td>" for value in row)
            + "</tr>"
        )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _diagnostics_summary_html(report: Any) -> str:
    payload = _diagnostics_payload(report)
    diagnostics = payload.get("diagnostics") or {}
    stats = diagnostics.get("stats") or {}
    optimizer_regions = diagnostics.get("optimizer_regions") or []
    optimizer_objects = diagnostics.get("optimizer_objects") or []
    legacy_regions = diagnostics.get("regions") or []

    stat_rows = [[key, value] for key, value in sorted(stats.items())]
    strategy_rows = [[key, value] for key, value in sorted((payload.get("strategies") or {}).items())]
    region_rows = [
        [
            region.get("id"),
            region.get("z"),
            region.get("kind"),
            region.get("children"),
            json.dumps(region.get("diagnostics") or {}, sort_keys=True),
        ]
        for region in optimizer_regions
    ]
    object_rows = [
        [
            obj.get("id"),
            obj.get("z"),
            obj.get("shape"),
            obj.get("fill"),
            obj.get("bounds"),
            obj.get("area"),
        ]
        for obj in optimizer_objects
    ]
    legacy_region_rows = [
        [
            region.get("id"),
            region.get("area"),
            region.get("color_hex"),
            (region.get("strategies") or {}).get("geom"),
            region.get("symmetry"),
        ]
        for region in legacy_regions
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <style>
    body {{
      margin: 0;
      color: #161616;
      font: 12px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    h3 {{ margin: 12px 0 6px; font-size: 13px; }}
    h3:first-child {{ margin-top: 0; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{
      padding: 4px 6px;
      border: 1px solid #e2e2dc;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #f2f2ee; font-weight: 600; }}
    td {{ overflow-wrap: anywhere; }}
    .empty {{ margin: 0; color: #666; }}
  </style>
</head>
<body>
  <h3>Stats</h3>
  {_summary_table(["metric", "value"], stat_rows)}
  <h3>Strategies</h3>
  {_summary_table(["strategy", "count"], strategy_rows)}
  <h3>Optimizer Regions</h3>
  {_summary_table(["id", "z", "kind", "children", "diagnostics"], region_rows)}
  <h3>Optimizer Objects</h3>
  {_summary_table(["id", "z", "shape", "fill", "bounds", "area"], object_rows)}
  <h3>Trace Regions</h3>
  {_summary_table(["id", "area", "color", "shape", "symmetry"], legacy_region_rows)}
</body>
</html>
"""


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
    diagnostics_summary_name = _diagnostics_summary_filename(entry)
    options_name = _options_filename(entry)

    rel_svg = Path("svg") / svg_name
    rel_input = Path("rendered-input") / input_name
    rel_raw = Path("raw") / raw_name
    rel_diagnostics = Path("diagnostics") / diagnostics_name
    rel_diagnostics_summary = Path("diagnostic-summary") / diagnostics_summary_name
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
        <iframe class="summary-frame" src="{html.escape(str(rel_diagnostics_summary))}" title="{html.escape(entry.name)} diagnostics summary"></iframe>
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
    diagnostics_summary_dir = output / "diagnostic-summary"
    options_dir = output / "options"
    trace_cache_dir = output / "cache"
    svg_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_summary_dir.mkdir(parents=True, exist_ok=True)
    options_dir.mkdir(parents=True, exist_ok=True)
    trace_cache_dir.mkdir(parents=True, exist_ok=True)

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
        if options.optimizer:
            svg, report = _idealize_optimizer_with_trace_cache(
                image,
                options=options,
                entry=entry,
                cache_dir=trace_cache_dir,
            )
        else:
            svg, report = idealize(image, options=options, report=True)
        svg_name = _safe_filename(entry)
        svg_path = svg_dir / svg_name
        svg_path.write_text(svg)
        (raw_dir / _raw_svg_filename(entry)).write_text(svg)
        (diagnostics_dir / _diagnostics_filename(entry)).write_text(_diagnostics_json(report))
        (diagnostics_summary_dir / _diagnostics_summary_filename(entry)).write_text(_diagnostics_summary_html(report))
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
    .code-frame,
    .summary-frame {{
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
    .code-frame,
    .summary-frame {{
      display: block;
      box-sizing: border-box;
      width: 100%;
      height: 360px;
      white-space: normal;
    }}
    .summary-frame {{
      height: 260px;
      margin-bottom: 8px;
      background: white;
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
        "--source-manifest",
        type=Path,
        default=None,
        help="JSON manifest of corpus image URLs. Defaults to corpus/sources.json when present.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CORPUS_CACHE,
        help="Directory for downloaded source images.",
    )
    parser.add_argument(
        "--refresh-sources",
        action="store_true",
        help="Redownload source-manifest images even when cached.",
    )
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
    if args.source_manifest is not None:
        entries = source_manifest_entries(
            args.source_manifest,
            args.cache_dir,
            refresh=args.refresh_sources,
        )
    elif args.input_dir is not None:
        entries = corpus_entries(args.input_dir)
    else:
        entries = None
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
