import numpy as np

from vectormark.candidate import LinearGradientFill, RadialGradientFill
from vectormark.optimizer.faithful import faithful_objects
from vectormark.pipeline import Options


def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def test_faithful_single_disk_one_object_path():
    img = np.full((120, 120, 3), 255, np.uint8)
    img[_disk(120, 120, 60, 60, 40)] = (200, 30, 30)

    objs, masks = faithful_objects(img, Options())

    assert len(objs) == 1
    o = objs[0]
    assert o.exact.kind == "path"
    assert o.id in masks and masks[o.id].sum() > 0
    assert abs(o.flat.area - np.pi * 40**2) / (np.pi * 40**2) < 0.1


def test_faithful_gradient_strip_merges_to_one_gradient_object():
    h, w = 80, 120
    img = np.full((h, w, 3), 255, np.uint8)
    xs = np.linspace(0.0, 1.0, 80)
    left = np.array([220.0, 40.0, 40.0])
    right = np.array([40.0, 40.0, 220.0])
    strip = np.round(left[None, :] * (1.0 - xs[:, None]) + right[None, :] * xs[:, None]).astype(np.uint8)
    img[20:60, 20:100] = strip[None, :, :]

    objs, masks = faithful_objects(img, Options(max_colors=2))

    assert len(objs) == 1
    assert isinstance(objs[0].fill, (LinearGradientFill, RadialGradientFill))
    assert objs[0].id in masks
    assert masks[objs[0].id].sum() == 80 * 40
