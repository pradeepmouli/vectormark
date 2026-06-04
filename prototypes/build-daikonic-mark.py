#!/usr/bin/env python3
"""Deterministic generator for the geometric Daikonic mark.

Proportions are taken from measurements of the source raster (band centers,
half-widths, and y-levels), reduced to a small set of named parameters and
emitted as clean primitives (semi-ellipse dome, convex-sided bands, almond
leaves). Symmetric by construction about CX.
"""

CX = 120                       # axis of symmetry
K = 0.5522847498                # cubic-bezier circle constant

PALETTE = dict(navy="#062336", teal="#3DA89D", orange="#FE8D27", red="#F23326")

# (y_top, y_bottom, hw_top, hw_bottom) per band, crop coords (source y - 45)
DOME_BOT, DOME_RX, DOME_RY = 139, 86, 69
BANDS = [
    ("teal",   153, 198, 88, 85),
    ("orange", 213, 257, 82, 69),
]
RED_TOP, RED_HW, RED_TIP = 270, 62, 339     # red tapers to a point
SIDE_BULGE = 3.5                            # outward convexity of band sides


def f(n: float) -> str:
    return f"{n:.2f}".rstrip("0").rstrip(".")


def dome() -> str:
    """Flat-bottomed semi-ellipse, apex at (CX, DOME_BOT-DOME_RY)."""
    rx, ry, yb = DOME_RX, DOME_RY, DOME_BOT
    return (f"M{f(CX-rx)} {f(yb)} "
            f"C{f(CX-rx)} {f(yb-K*ry)} {f(CX-K*rx)} {f(yb-ry)} {f(CX)} {f(yb-ry)} "
            f"C{f(CX+K*rx)} {f(yb-ry)} {f(CX+rx)} {f(yb-K*ry)} {f(CX+rx)} {f(yb)} Z")


def band(yt, yb, hwt, hwb, b=SIDE_BULGE) -> str:
    """Flat top/bottom, gently convex sides; symmetric about CX."""
    t = (yb - yt) / 3
    return (f"M{f(CX-hwt)} {f(yt)} "
            f"L{f(CX+hwt)} {f(yt)} "
            f"C{f(CX+hwt+b)} {f(yt+t)} {f(CX+hwb+b)} {f(yb-t)} {f(CX+hwb)} {f(yb)} "
            f"L{f(CX-hwb)} {f(yb)} "
            f"C{f(CX-hwb-b)} {f(yb-t)} {f(CX-hwt-b)} {f(yt+t)} {f(CX-hwt)} {f(yt)} Z")


def red_tip() -> str:
    """Flat top, sides sweep to a rounded point at (CX, RED_TIP)."""
    yt, hw, tip = RED_TOP, RED_HW, RED_TIP
    h = tip - yt
    return (f"M{f(CX-hw)} {f(yt)} "
            f"L{f(CX+hw)} {f(yt)} "
            f"C{f(CX+hw)} {f(yt+h*0.45)} {f(CX+hw*0.42)} {f(tip)} {f(CX)} {f(tip)} "
            f"C{f(CX-hw*0.42)} {f(tip)} {f(CX-hw)} {f(yt+h*0.45)} {f(CX-hw)} {f(yt)} Z")


def leaf(length, r) -> str:
    """Fat petal along local +x: pointed at origin, convex sides bulging to half-
    width r, capped by a semicircle of radius r centered at (length, 0)."""
    return (f"M0 0 "
            f"C{f(length*0.16)} {f(-r)} {f(length*0.85)} {f(-r)} {f(length)} {f(-r)} "
            f"A{f(r)} {f(r)} 0 0 1 {f(length)} {f(r)} "
            f"C{f(length*0.85)} {f(r)} {f(length*0.16)} {f(r)} 0 0 Z")


def leaves() -> list[str]:
    # two fat petals sprouting from the dome apex, tips nesting at (CX, ~84),
    # symmetric about vertical (right at -56deg, left at -124deg).
    blade = leaf(55, 17)
    navy = PALETTE["navy"]
    return [
        f'<path d="{blade}" fill="{navy}" transform="translate({CX} 86) rotate(-54)"/>',
        f'<path d="{blade}" fill="{navy}" transform="translate({CX} 86) rotate(-126)"/>',
    ]


def build() -> str:
    paths = [
        f'<path d="{dome()}" fill="{PALETTE["navy"]}"/>',
        *(f'<path d="{band(yt, yb, hwt, hwb)}" fill="{PALETTE[c]}"/>'
          for c, yt, yb, hwt, hwb in BANDS),
        f'<path d="{red_tip()}" fill="{PALETTE["red"]}"/>',
        *leaves(),
    ]
    body = "\n  ".join(paths)
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 350" '
            'width="240" height="350" role="img" aria-label="Daikonic mark">\n'
            '  <title>Daikonic</title>\n  '
            + body + "\n</svg>\n")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "daikonic-mark.svg"
    open(out, "w").write(build())
    print("wrote", out)
