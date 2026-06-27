import base64
import io
import numpy as np
import pytest
from PIL import Image
from vectormark.mcp_image import resolve_image, ImageError, MAX_INPUT_BYTES


def _png_bytes(size=(32, 32), color=(20, 120, 200)):
    buf = io.BytesIO(); Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_resolve_base64_and_data_uri():
    raw = _png_bytes()
    b64 = base64.b64encode(raw).decode()
    r = resolve_image({"base64": b64}, local_trust=False)
    assert r.source_kind == "base64" and r.mime_type == "image/png" and r.bytes == raw
    r2 = resolve_image({"data_uri": f"data:image/png;base64,{b64}"}, local_trust=False)
    assert r2.source_kind == "data_uri" and r2.bytes == raw and len(r2.sha256) == 64


def test_resolve_local_path_requires_local_trust(tmp_path):
    p = tmp_path / "m.png"; p.write_bytes(_png_bytes())
    r = resolve_image({"path": str(p)}, local_trust=True)
    assert r.source_kind == "local_path" and r.bytes == p.read_bytes()
    with pytest.raises(ImageError) as exc:
        resolve_image({"path": str(p)}, local_trust=False)
    assert exc.value.error_code == "IMAGE_UNRESOLVABLE"


def test_resolve_download_url_via_injected_fetch():
    raw = _png_bytes()
    r = resolve_image({"download_url": "https://host/file"}, local_trust=False,
                      fetch=lambda url, **kw: raw)
    assert r.source_kind == "platform_file" and r.bytes == raw


def test_resolve_url_ssrf_blocks_loopback():
    with pytest.raises(ImageError) as exc:
        resolve_image({"url": "http://127.0.0.1/x.png"}, local_trust=False,
                      fetch=lambda url, **kw: _png_bytes())
    assert exc.value.error_code == "URL_NOT_ALLOWED"


def test_resolve_rejects_unsupported_type():
    gif = io.BytesIO(); Image.new("RGB", (8, 8)).save(gif, format="GIF")
    with pytest.raises(ImageError) as exc:
        resolve_image({"base64": base64.b64encode(gif.getvalue()).decode()}, local_trust=False)
    assert exc.value.error_code == "UNSUPPORTED_IMAGE_TYPE"


def test_resolve_rejects_too_large():
    big = b"\x89PNG\r\n\x1a\n" + b"0" * (MAX_INPUT_BYTES + 1)
    with pytest.raises(ImageError) as exc:
        resolve_image({"base64": base64.b64encode(big).decode()}, local_trust=False)
    assert exc.value.error_code == "IMAGE_TOO_LARGE"


def test_resolve_none_is_unresolvable():
    with pytest.raises(ImageError) as exc:
        resolve_image({}, local_trust=True)
    assert exc.value.error_code == "IMAGE_UNRESOLVABLE"
