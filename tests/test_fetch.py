from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pytest

from keras_image_classifier import KicError, fetch


def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, content in members.items():
            bundle.writestr(name, content)
    return path


def test_extract_keeps_class_and_file_name_only(tmp_path: Path) -> None:
    archive = make_zip(
        tmp_path / "a.zip",
        {
            "EuroSAT_RGB/Forest/Forest_1.jpg": b"one",
            "EuroSAT_RGB/River/River_1.jpg": b"two",
            "EuroSAT_RGB/README.txt": b"skipped: not an image",
            "top_level.jpg": b"skipped: no class directory",
        },
    )
    assert fetch.extract_images(archive, tmp_path / "out") == 2
    written = sorted(
        p.relative_to(tmp_path / "out").as_posix() for p in (tmp_path / "out").rglob("*.*")
    )
    assert written == ["Forest/Forest_1.jpg", "River/River_1.jpg"]


def test_hostile_member_names_cannot_leave_the_destination(tmp_path: Path) -> None:
    archive = make_zip(
        tmp_path / "evil.zip",
        {
            "../../outside/escape.jpg": b"x",
            "/absolute/path/root.jpg": b"x",
            "C:/Windows/system.jpg": b"x",
            "Forest/../../up.jpg": b"x",
            "Forest/..\\..\\backslash.jpg": b"x",
            "Forest/ok.jpg": b"x",
        },
    )
    destination = tmp_path / "deep" / "out"
    fetch.extract_images(archive, destination)
    everything = [p for p in tmp_path.rglob("*") if p.is_file() and p.suffix == ".jpg"]
    assert all(destination in p.parents for p in everything)
    assert (destination / "Forest" / "ok.jpg").is_file()
    assert not any(".." in p.name or "\\" in p.name for p in everything)


def test_a_decompression_bomb_is_stopped_by_bytes_actually_written(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "bomb.zip", {"Forest/zeros.jpg": b"\x00" * 20_000_000})
    assert archive.stat().st_size < 100_000
    with pytest.raises(KicError, match="unpacks to more than 1000000 bytes"):
        fetch.extract_images(archive, tmp_path / "out", max_bytes=1_000_000)


def test_the_number_of_files_is_bounded(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "many.zip", {f"Forest/f{i}.jpg": b"x" for i in range(5)})
    with pytest.raises(KicError, match="more than 3 images"):
        fetch.extract_images(archive, tmp_path / "out", max_files=3)


def test_an_archive_without_images_is_an_error(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "empty.zip", {"docs/readme.txt": b"x"})
    with pytest.raises(KicError, match="contains no"):
        fetch.extract_images(archive, tmp_path / "out")


class FakeResponse(io.BytesIO):
    def __enter__(self) -> FakeResponse:
        return self


def serve(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: FakeResponse(payload))


def test_download_refuses_plain_http(tmp_path: Path) -> None:
    with pytest.raises(KicError, match="only https"):
        fetch.download("http://example.org/a.zip", tmp_path / "a", expected_sha256="", max_bytes=1)


def test_download_verifies_the_checksum_and_removes_a_bad_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve(monkeypatch, b"payload")
    good = hashlib.sha256(b"payload").hexdigest()
    fetch.download("https://example.org/a", tmp_path / "ok", expected_sha256=good, max_bytes=100)
    assert (tmp_path / "ok").read_bytes() == b"payload"
    with pytest.raises(KicError, match="checksum mismatch"):
        fetch.download(
            "https://example.org/a", tmp_path / "bad", expected_sha256="0" * 64, max_bytes=100
        )
    assert not (tmp_path / "bad").exists()


def test_download_stops_when_the_server_sends_too_much(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve(monkeypatch, b"x" * 5000)
    with pytest.raises(KicError, match="larger than the expected 1000 bytes"):
        fetch.download(
            "https://example.org/a", tmp_path / "big", expected_sha256="", max_bytes=1000
        )
    assert not (tmp_path / "big").exists()


def test_fetch_eurosat_downloads_verifies_and_unpacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = make_zip(tmp_path / "src.zip", {"EuroSAT_RGB/Forest/Forest_1.jpg": b"img"})
    payload = archive.read_bytes()
    serve(monkeypatch, payload)
    monkeypatch.setattr(fetch, "EUROSAT_SHA256", hashlib.sha256(payload).hexdigest())
    assert fetch.fetch_eurosat(tmp_path / "eurosat") == 1
    assert (tmp_path / "eurosat" / "Forest" / "Forest_1.jpg").read_bytes() == b"img"
    with pytest.raises(KicError, match="is not empty"):
        fetch.fetch_eurosat(tmp_path / "eurosat")


def test_fetch_eurosat_leaves_nothing_behind_when_the_archive_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve(monkeypatch, b"this is not a zip file")
    monkeypatch.setattr(
        fetch, "EUROSAT_SHA256", hashlib.sha256(b"this is not a zip file").hexdigest()
    )
    with pytest.raises(zipfile.BadZipFile):
        fetch.fetch_eurosat(tmp_path / "eurosat")
    assert not (tmp_path / "eurosat").exists()


@pytest.mark.network
def test_the_published_archive_still_has_the_pinned_size() -> None:
    request = urllib.request.Request(fetch.EUROSAT_URL, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        assert int(response.headers["Content-Length"]) == fetch.EUROSAT_BYTES
