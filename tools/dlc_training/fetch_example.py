"""Download and verify the pinned official DLC fixture without mutating Git."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

EXAMPLE_COMMIT = "4dcb9a7d5a701427ef78ecc8103d38eeef70601a"
EXAMPLE_PREFIX = "examples/openfield-Pranav-2018-10-30/"


def fetch_example(destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    tree_url = (
        "https://api.github.com/repos/DeepLabCut/DeepLabCut/git/trees/"
        f"{EXAMPLE_COMMIT}?recursive=1"
    )
    with urlopen(tree_url, timeout=60) as response:  # noqa: S310 -- fixed HTTPS GitHub endpoint
        tree = json.load(response)
    if tree.get("truncated"):
        raise ValueError("GitHub returned an incomplete example inventory")
    entries = [
        item
        for item in tree["tree"]
        if item["type"] == "blob" and item["path"].startswith(EXAMPLE_PREFIX)
    ]
    if not entries:
        raise ValueError("Pinned example contains no files")

    def fetch_file(entry: dict) -> dict:
        relative = PurePosixPath(entry["path"][len(EXAMPLE_PREFIX) :])
        target = destination.joinpath(*relative.parts).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Invalid example path: {relative}")
        if target.exists():
            payload = target.read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/DeepLabCut/DeepLabCut/{EXAMPLE_COMMIT}/{entry['path']}"
            with urlopen(url, timeout=120) as response:
                payload = response.read()
        blob_digest = hashlib.sha1(
            f"blob {len(payload)}\0".encode() + payload, usedforsecurity=False
        ).hexdigest()
        if blob_digest != entry["sha"]:
            raise ValueError(f"Example content mismatch: {relative}")
        if payload.startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"Example contains an unresolved LFS pointer: {relative}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        return {
            "path": str(relative),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "git_blob": blob_digest,
        }

    with ThreadPoolExecutor(max_workers=4) as executor:
        files = list(executor.map(fetch_file, entries))
    manifest = {
        "commit": EXAMPLE_COMMIT,
        "example": EXAMPLE_PREFIX,
        "files": sorted(files, key=lambda item: item["path"]),
    }
    (destination / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Verified {len(files)} files, {sum(item['bytes'] for item in files)} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    fetch_example(parser.parse_args().destination)
