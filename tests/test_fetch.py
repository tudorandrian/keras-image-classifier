from __future__ import annotations

import email.message
import hashlib
import http.client
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


def test_hostile_member_names_are_rejected_not_re_homed(tmp_path: Path) -> None:
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
    assert fetch.extract_images(archive, destination) == 1
    written = sorted(p.relative_to(destination).as_posix() for p in destination.rglob("*.*"))
    assert written == ["Forest/ok.jpg"]


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
    monkeypatch.setattr(fetch._OPENER, "open", lambda url, timeout: FakeResponse(payload))


def test_download_refuses_plain_http(tmp_path: Path) -> None:
    with pytest.raises(KicError, match="only https"):
        fetch.download("http://example.org/a.zip", tmp_path / "a", expected_sha256="", max_bytes=1)


def test_download_refuses_a_redirect_that_leaves_https() -> None:
    handler = fetch._HttpsOnlyRedirectHandler()
    fp = io.BytesIO(b"redirect body")
    with pytest.raises(KicError, match="not https"):
        handler.redirect_request(
            urllib.request.Request("https://example.org/a"),
            fp,
            302,
            "Found",
            {},
            "http://example.org/b",
        )
    assert fp.closed


def test_download_follows_a_redirect_that_stays_on_https() -> None:
    handler = fetch._HttpsOnlyRedirectHandler()
    redirected = handler.redirect_request(
        urllib.request.Request("https://example.org/a"),
        io.BytesIO(b""),
        302,
        "Found",
        {},
        "https://example.org/b",
    )
    assert redirected is not None
    assert redirected.full_url == "https://example.org/b"


class _FakeRedirectResponse:
    """A response object shaped enough like http.client.HTTPResponse for
    urllib's own machinery to process it as a 302, so a test can drive a real
    redirect through fetch._OPENER instead of mocking .open() itself away."""

    def __init__(self, status: int, header_pairs: list[tuple[str, str]]) -> None:
        self.status = status
        self.code = status
        self.reason = "Found"
        self.version = 11
        headers = email.message.Message()
        for key, value in header_pairs:
            headers[key] = value
        self.headers = headers
        self.msg = self.headers
        self.closed = False

    def getheader(self, name: str, default: str | None = None) -> str | None:
        return self.headers.get(name, default)

    def getheaders(self) -> list[tuple[str, str]]:
        return list(self.headers.items())

    def info(self) -> email.message.Message:
        return self.headers

    def read(self, *_args: object, **_kwargs: object) -> bytes:
        return b""

    def close(self) -> None:
        self.closed = True

    def isclosed(self) -> bool:
        return self.closed


class _FakeHTTPSConnection:
    """A minimal stand-in for http.client.HTTPSConnection that always answers
    with a redirect to a plain-http URL, so opening through fetch._OPENER
    exercises the real OpenerDirector/HTTPRedirectHandler machinery."""

    debuglevel = 0
    sock = None
    _http_vsn = 11
    _http_vsn_str = "HTTP/1.1"

    def __init__(self, host: str, *_args: object, **_kwargs: object) -> None:
        self.host = host

    def set_debuglevel(self, _level: int) -> None:
        pass

    def request(self, *_args: object, **_kwargs: object) -> None:
        pass

    def getresponse(self) -> _FakeRedirectResponse:
        return _FakeRedirectResponse(302, [("Location", "http://example.org/downgraded")])

    def close(self) -> None:
        pass


def test_the_module_opener_actually_refuses_a_downgrading_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """serve() replaces _OPENER.open() outright, which proves nothing about
    whether _HttpsOnlyRedirectHandler is really reachable from it. This test
    instead fakes the underlying HTTPS connection, so the request travels
    through the real OpenerDirector and its redirect handling, and confirms
    that a genuine redirect to http is refused with KicError from there."""
    monkeypatch.setattr(http.client, "HTTPSConnection", _FakeHTTPSConnection)
    with pytest.raises(KicError, match="not https"):
        fetch._OPENER.open("https://example.org/a", timeout=5)


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


def _mark_first_member_encrypted(path: Path) -> None:
    """Flip the encryption bit in a real zip's headers, without an encryption
    library, so zipfile refuses to read the member back with a RuntimeError."""
    data = bytearray(path.read_bytes())
    local = data.find(b"PK\x03\x04")
    central = data.find(b"PK\x01\x02")
    data[local + 6] |= 0x01
    data[central + 8] |= 0x01
    path.write_bytes(bytes(data))


def test_fetch_eurosat_cleans_up_after_a_non_kicerror_extraction_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = make_zip(tmp_path / "src.zip", {"Forest/enc.jpg": b"data"})
    _mark_first_member_encrypted(archive)
    payload = archive.read_bytes()
    serve(monkeypatch, payload)
    monkeypatch.setattr(fetch, "EUROSAT_SHA256", hashlib.sha256(payload).hexdigest())
    with pytest.raises(RuntimeError, match="encrypted"):
        fetch.fetch_eurosat(tmp_path / "eurosat")
    assert not (tmp_path / "eurosat").exists()


@pytest.mark.network
def test_the_published_archive_still_has_the_pinned_size() -> None:
    request = urllib.request.Request(fetch.EUROSAT_URL, method="HEAD")
    # Through the module's own opener, so this exercises the https-only redirect handler
    # that every real download goes through, not urllib's stock one.
    with fetch._OPENER.open(request, timeout=30) as response:
        assert int(response.headers["Content-Length"]) == fetch.EUROSAT_BYTES
