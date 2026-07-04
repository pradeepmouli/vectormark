from .clones import clones_pass
from .compound import split_compound_pass
from .occlusion import occlusion_pass
from .primitives import primitives_pass
from .simplify import simplify_pass
from .symmetry import symmetry_pass

__all__ = ["clones_pass", "occlusion_pass", "primitives_pass", "simplify_pass", "split_compound_pass", "symmetry_pass"]
