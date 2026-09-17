"""Download the EuroSAT RGB benchmark and unpack it without trusting the archive.

EuroSAT (Helber et al., 2019) is 27,000 Sentinel-2 satellite patches of 64 x 64 px in
ten land-use classes, published under the MIT licence at doi:10.5281/zenodo.7711810.
It contains no people, which is why it replaced the face data sets of the coursework.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from keras_image_classifier import KicError
from keras_image_classifier.dataset import CLASS_NAME

EUROSAT_URL = "https://zenodo.org/api/records/7711810/files/EuroSAT_RGB.zip/content"
EUROSAT_SHA256 = "b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90"
EUROSAT_BYTES = 94_658_721
FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\.(jpg|jpeg|png)", re.IGNORECASE)
CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, *, expected_sha256: str, max_bytes: int) -> None:
    """Stream url to target; delete it again unless size and checksum are right."""
    if not url.startswith("https://"):
        raise KicError("only https downloads are allowed")
    received = 0
    try:
        # The scheme was checked two lines up, so file: and ftp: URLs cannot reach urlopen.
        response = urllib.request.urlopen(url, timeout=60)  # noqa: S310  # nosec B310
        with response, target.open("wb") as out:
            while chunk := response.read(CHUNK):
                received += len(chunk)
                if received > max_bytes:
                    raise KicError(f"download is larger than the expected {max_bytes} bytes")
                out.write(chunk)
        if sha256_of(target) != expected_sha256:
            raise KicError("checksum mismatch: the downloaded file is not the published one")
    except (KicError, OSError):
        target.unlink(missing_ok=True)
        raise


def extract_images(
    archive: Path, destination: Path, *, max_files: int = 50_000, max_bytes: int = 500_000_000
) -> int:
    """Unpack <anything>/<class>/<file>.jpg members to destination/<class>/<file>.jpg.

    Member paths are never joined to the destination. Only the last two components
    are used, and both must match a strict pattern, so '..', absolute paths and
    drive letters cannot escape. The byte and file limits stop a decompression bomb;
    bytes are counted while copying because the sizes in a zip header can lie.
    """
    written = 0
    total = 0
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            parts = PurePosixPath(member.filename).parts
            if member.is_dir() or len(parts) < 2:
                continue
            label, name = parts[-2], parts[-1]
            if not CLASS_NAME.fullmatch(label) or not FILE_NAME.fullmatch(name):
                continue
            written += 1
            if written > max_files:
                raise KicError(f"archive holds more than {max_files} images")
            (destination / label).mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, (destination / label / name).open("wb") as out:
                while chunk := source.read(CHUNK):
                    total += len(chunk)
                    if total > max_bytes:
                        raise KicError(f"archive unpacks to more than {max_bytes} bytes")
                    out.write(chunk)
    if written == 0:
        raise KicError("archive contains no <class>/<image> members")
    return written


def fetch_eurosat(destination: Path) -> int:
    """Download, verify and unpack EuroSAT RGB; return the number of images."""
    if destination.exists() and any(destination.iterdir()):
        raise KicError(f"{destination} is not empty")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / "EuroSAT_RGB.zip"
        download(EUROSAT_URL, archive, expected_sha256=EUROSAT_SHA256, max_bytes=EUROSAT_BYTES)
        try:
            return extract_images(archive, destination)
        except (KicError, zipfile.BadZipFile):
            shutil.rmtree(destination, ignore_errors=True)
            raise
