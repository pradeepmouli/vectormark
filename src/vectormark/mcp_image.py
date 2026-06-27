# SPDX-License-Identifier: MIT
"""Pure helpers for the file-first MCP image handoff: resolve an image reference to
validated bytes, preprocess it, and summarize the emitted SVG. No MCP imports."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_DIMENSION_PX = 4096
DEFAULT_MAX_SIZE_PX = 1024
DEFAULT_COLORS = 16
ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp"}
_PIL_FORMAT_MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
_FETCH_TIMEOUT = 15.0


class ImageError(Exception):
    """A structured, code-carrying error for the MCP image tools."""

    def __init__(self, error_code: str, message: str):
        super().__init__(message)
        self.error_code = error_code
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {"error_code": self.error_code, "message": self.message}


@dataclass(frozen=True)
class ResolvedImage:
    bytes: bytes
    mime_type: str
    sha256: str
    source_kind: str
    filename: str | None = None


def _default_fetch(url: str, *, max_bytes: int = MAX_INPUT_BYTES, timeout: float = _FETCH_TIMEOUT) -> bytes:
    """GET `url` and return up to max_bytes; raises ImageError(IMAGE_UNRESOLVABLE) on failure."""
    import httpx
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.content
    except Exception as exc:  # network / HTTP / decode
        raise ImageError("IMAGE_UNRESOLVABLE", f"could not fetch image url: {exc}") from exc
    if len(data) > max_bytes:
        raise ImageError("IMAGE_TOO_LARGE", "fetched image exceeds the size limit")
    return data


def _assert_url_safe(url: str) -> None:
    """SSRF guard for a caller-supplied `url`: https only, no loopback/private/link-local/
    metadata hosts."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ImageError("URL_NOT_ALLOWED", "url must be an absolute https URL")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ImageError("URL_NOT_ALLOWED", f"could not resolve url host: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ImageError("URL_NOT_ALLOWED", "url resolves to a non-public address")


def _decode_b64(data: str) -> bytes:
    data = data.strip()
    if data.startswith("data:"):
        _, _, data = data.partition(",")
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageError("IMAGE_UNRESOLVABLE", "image is not valid base64") from exc


def _validate(data: bytes) -> str:
    """Size + dimension + mime validation. Returns the sniffed mime."""
    if len(data) > MAX_INPUT_BYTES:
        raise ImageError("IMAGE_TOO_LARGE", "input exceeds the configured size limit")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, (w, h) = im.format, im.size
    except Exception as exc:
        raise ImageError("UNSUPPORTED_IMAGE_TYPE", "could not decode image; expected PNG, JPEG, or WebP") from exc
    mime = _PIL_FORMAT_MIME.get(fmt or "")
    if mime not in ALLOWED_MIME:
        raise ImageError("UNSUPPORTED_IMAGE_TYPE", "expected PNG, JPEG, or WebP")
    if w > MAX_DIMENSION_PX or h > MAX_DIMENSION_PX:
        raise ImageError("IMAGE_TOO_LARGE", f"image dimension exceeds {MAX_DIMENSION_PX}px")
    return mime


def resolve_image(image: dict, *, local_trust: bool, fetch=_default_fetch) -> ResolvedImage:
    """Resolve an image reference object to validated bytes. Resolution order:
    download_url (platform_file) -> path (local_trust only) -> url (SSRF-guarded) ->
    data_uri -> base64. Raises ImageError with a structured code on any failure."""
    image = image or {}
    if image.get("download_url"):
        data, kind, name = fetch(image["download_url"]), "platform_file", image.get("file_name")
    elif image.get("path"):
        if not local_trust:
            raise ImageError("IMAGE_UNRESOLVABLE", "local paths are not accepted over a network transport")
        p = Path(image["path"]).expanduser()
        if not p.is_file():
            raise ImageError("IMAGE_UNRESOLVABLE", f"image path does not exist: {p}")
        data, kind, name = p.read_bytes(), "local_path", p.name
    elif image.get("url"):
        _assert_url_safe(image["url"])
        data, kind, name = fetch(image["url"]), "url", None
    elif image.get("data_uri"):
        data, kind, name = _decode_b64(image["data_uri"]), "data_uri", None
    elif image.get("base64"):
        data, kind, name = _decode_b64(image["base64"]), "base64", None
    else:
        raise ImageError("IMAGE_UNRESOLVABLE", "no resolvable image reference was provided")
    mime = _validate(data)
    return ResolvedImage(bytes=data, mime_type=mime, sha256=hashlib.sha256(data).hexdigest(),
                         source_kind=kind, filename=name)
