"""`vectormark` CLI."""

from __future__ import annotations

import argparse
import sys

from .pipeline import Options, idealize


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vectormark", description="Idealize a logo raster into SVG.")
    ap.add_argument("input", help="input raster (PNG/JPG)")
    ap.add_argument("-o", "--output", help="output .svg (default: stdout)")
    ap.add_argument("--epsilon", type=float, default=1.5, help="fit tolerance in px")
    ap.add_argument("--max-error", type=float, default=1.0, help="Bézier fit tolerance in px")
    ap.add_argument("--colors", type=int, default=16, help="max palette colours")
    ap.add_argument("--flatten", action="store_true", help="flatten primitives to paths")
    ap.add_argument("--no-symmetry", action="store_true", help="disable symmetry detection")
    args = ap.parse_args(argv)

    svg = idealize(args.input, options=Options(
        epsilon=args.epsilon, max_error=args.max_error, max_colors=args.colors,
        flatten=args.flatten, no_symmetry=args.no_symmetry,
    ))
    if args.output:
        with open(args.output, "w") as f:
            f.write(svg)
    else:
        sys.stdout.write(svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
