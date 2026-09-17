from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from keras_image_classifier import KicError
from keras_image_classifier.images import letterbox, load_rgb, pixel_hash


def save(tmp_path: Path, name: str, image: Image.Image, **options: object) -> Path:
    path = tmp_path / name
    image.save(path, **options)
    return path


@pytest.mark.parametrize("name", ["a.png", "a.jpg", "a.webp", "a.bmp"])
def test_allowed_formats_load_as_rgb(tmp_path: Path, name: str) -> None:
    path = save(tmp_path, name, Image.new("RGB", (20, 10), (200, 10, 10)))
    image = load_rgb(path)
    assert image.mode == "RGB"
    assert image.size == (20, 10)


def test_palette_and_grey_images_become_rgb(tmp_path: Path) -> None:
    grey = save(tmp_path, "grey.png", Image.new("L", (8, 8), 128))
    palette = save(tmp_path, "palette.png", Image.new("P", (8, 8)))
    assert load_rgb(grey).mode == "RGB"
    assert load_rgb(palette).mode == "RGB"


def test_format_is_read_from_content_not_from_the_extension(tmp_path: Path) -> None:
    disguised = save(tmp_path, "photo.jpg", Image.new("RGB", (8, 8)), format="GIF")
    with pytest.raises(KicError, match="unsupported format GIF"):
        load_rgb(disguised)


@pytest.mark.parametrize("content", [b"", b"not an image", b"\x89PNG\r\n\x1a\n" + b"\x00" * 40])
def test_unreadable_files_are_refused(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(content)
    with pytest.raises(KicError, match="not a readable image"):
        load_rgb(path)


def test_truncated_jpeg_is_refused(tmp_path: Path) -> None:
    whole = save(tmp_path, "whole.jpg", Image.new("RGB", (64, 64), (5, 99, 200)))
    cut = tmp_path / "cut.jpg"
    cut.write_bytes(whole.read_bytes()[:200])
    with pytest.raises(KicError, match="not a readable image"):
        load_rgb(cut)


def test_pixel_limit_is_checked_before_decoding(tmp_path: Path) -> None:
    path = save(tmp_path, "big.png", Image.new("RGB", (300, 300)))
    with pytest.raises(KicError, match="300x300 exceeds the limit of 1000 pixels"):
        load_rgb(path, max_pixels=1000)


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    path = save(tmp_path, "rotated.jpg", Image.new("RGB", (40, 20)), exif=exif)
    assert load_rgb(path).size == (20, 40)


def test_letterbox_keeps_the_aspect_ratio_and_pads_with_black() -> None:
    wide = Image.new("RGB", (100, 50), (255, 255, 255))
    pixels = np.asarray(letterbox(wide, 64))
    assert pixels.shape == (64, 64, 3)
    assert (pixels[:16] == 0).all()
    assert (pixels[-16:] == 0).all()
    assert (pixels[16:48] == 255).all()


def test_letterbox_enlarges_small_images() -> None:
    pixels = np.asarray(letterbox(Image.new("RGB", (10, 10), (9, 9, 9)), 64))
    assert (pixels == 9).all()


def test_pixel_hash_ignores_the_container_and_sees_every_pixel(tmp_path: Path) -> None:
    image = Image.new("RGB", (16, 16), (1, 2, 3))
    as_png = load_rgb(save(tmp_path, "a.png", image))
    as_bmp = load_rgb(save(tmp_path, "a.bmp", image))
    assert pixel_hash(as_png) == pixel_hash(as_bmp)
    changed = image.copy()
    changed.putpixel((15, 15), (1, 2, 4))
    assert pixel_hash(changed) != pixel_hash(image)


def test_pixel_hash_includes_the_shape() -> None:
    assert pixel_hash(Image.new("RGB", (4, 16))) != pixel_hash(Image.new("RGB", (16, 4)))
