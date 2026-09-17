"""Decode untrusted image files into fixed-size RGB arrays."""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from keras_image_classifier import KicError

ALLOWED_FORMATS = ("JPEG", "PNG", "WEBP", "BMP")
MAX_PIXELS = 50_000_000


def load_rgb(path: Path, *, max_pixels: int = MAX_PIXELS) -> Image.Image:
    """Open one file as an upright RGB image, or raise KicError saying why not.

    The format and the pixel count come from the header, so a disguised file is
    refused before any pixel data is decoded. Pillow also runs its own
    decompression-bomb ceiling (Image.MAX_IMAGE_PIXELS) while reading that same
    header, ahead of our own max_pixels check below -- so a big enough image
    never reaches our check at all. That ceiling raises DecompressionBombError
    outright, or only warns via DecompressionBombWarning one size band lower;
    the warning is turned into an error here so both bands are always refused
    the same way, regardless of the caller's ambient warnings filters.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as handle:
                if handle.format not in ALLOWED_FORMATS:
                    raise KicError(f"unsupported format {handle.format}")
                width, height = handle.size
                if width * height > max_pixels:
                    raise KicError(f"{width}x{height} exceeds the limit of {max_pixels} pixels")
                upright = ImageOps.exif_transpose(handle)
                return upright.convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise KicError(f"not a readable image ({type(error).__name__})") from error


def letterbox(image: Image.Image, size: int) -> Image.Image:
    """Scale to fit a size x size square, keep the aspect ratio, pad with black."""
    return ImageOps.pad(image, (size, size), method=Image.Resampling.LANCZOS, color=(0, 0, 0))


def pixel_hash(image: Image.Image) -> str:
    """SHA-256 of the decoded pixels: equal for the same picture in two containers."""
    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.width}x{image.height}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()
