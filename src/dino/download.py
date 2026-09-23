"""Download the official DINOv2 Hub files through a configurable HTTPS endpoint."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm.auto import tqdm


MODEL_ID = "facebook/dinov2-base"
MODEL_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_snapshot(
    endpoint: str, revision: str, cache_dir: Path
) -> tuple[Path, dict[str, object]]:
    """Resolve one commit, verify files, and reuse complete cached downloads."""
    endpoint = endpoint.rstrip("/")
    encoded_revision = quote(revision, safe="")
    api_url = f"{endpoint}/api/models/{MODEL_ID}/revision/{encoded_revision}"
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()
    commit = response.json().get("sha")
    if not isinstance(commit, str) or not COMMIT_PATTERN.fullmatch(commit):
        raise ValueError(f"Could not resolve a valid commit for {MODEL_ID} at {endpoint}")
    snapshot = cache_dir / "snapshots" / commit
    snapshot.mkdir(parents=True, exist_ok=True)
    file_info: dict[str, object] = {}

    for name in MODEL_FILES:
        url = f"{endpoint}/{MODEL_ID}/resolve/{commit}/{name}"
        head = requests.head(url, allow_redirects=False, timeout=30)
        head.raise_for_status()
        reported_commit = head.headers.get("x-repo-commit")
        if reported_commit and reported_commit != commit:
            raise ValueError(f"Revision mismatch for {name}: {reported_commit} != {commit}")
        linked_etag = head.headers.get("x-linked-etag", "").strip('"').lower()
        expected_sha256 = linked_etag if SHA256_PATTERN.fullmatch(linked_etag) else None
        linked_size = head.headers.get("x-linked-size")
        expected_size = int(linked_size) if linked_size else None
        path = snapshot / name
        if path.is_file():
            existing_hash = sha256_file(path)
            if (expected_sha256 is None or existing_hash == expected_sha256) and (
                expected_size is None or path.stat().st_size == expected_size
            ):
                file_info[name] = {
                    "sha256": existing_hash,
                    "bytes": path.stat().st_size,
                    "cached": True,
                }
                continue

        temporary = snapshot / f"{name}.part"
        digest = hashlib.sha256()
        size = 0
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as stream:
                stream.raise_for_status()
                with temporary.open("wb") as handle:
                    with tqdm(
                        total=expected_size,
                        desc=f"Downloading {name}",
                        unit="B",
                        unit_scale=True,
                        disable=expected_size is None or expected_size < 1024 * 1024,
                    ) as progress:
                        for chunk in stream.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                                digest.update(chunk)
                                size += len(chunk)
                                progress.update(len(chunk))
            if expected_size is not None and size != expected_size:
                raise ValueError(f"Size mismatch for {name}: {size} != {expected_size}")
            actual_sha256 = digest.hexdigest()
            if expected_sha256 is not None and actual_sha256 != expected_sha256:
                raise ValueError(f"SHA256 mismatch for {name}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        file_info[name] = {"sha256": actual_sha256, "bytes": size, "cached": False}
        print(f"Downloaded {name}: {size} bytes", flush=True)

    source = {"model_id": MODEL_ID, "revision": commit, "endpoint": endpoint, "files": file_info}
    (snapshot / "download_manifest.json").write_text(
        json.dumps(source, indent=2) + "\n", encoding="utf-8"
    )
    return snapshot, source
