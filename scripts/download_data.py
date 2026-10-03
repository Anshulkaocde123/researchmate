#!/usr/bin/env python3
"""Download QASPER v0.3 from the official S3 source into data/raw/.

Usage
-----
    python scripts/download_data.py [--data-dir data/raw]

Source
------
Official AllenAI S3 release (v0.3):
  • https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz
  • https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz

Each tarball is extracted, the three JSON files are kept, and each split is
re-serialised as line-delimited JSON (``<split>.jsonl``) for streaming access.
A ``MANIFEST.json`` records the source, download date, row counts, file sizes
and SHA-256 checksums.

Note: the HuggingFace ``load_dataset`` path is attempted first for
future-compatibility (if AllenAI converts the dataset to Parquet). If it
fails, the script falls back to direct S3 download, which is the reliable path
as of v0.3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tarfile
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

S3_URLS = {
    "train_dev": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
    "test": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
}

# Maps (tarball_key, json_filename_inside_tar) → split label used as output name
TAR_TO_SPLIT: list[tuple[str, str, str]] = [
    ("train_dev", "qasper-train-v0.3.json", "train"),
    ("train_dev", "qasper-dev-v0.3.json", "validation"),
    ("test", "qasper-test-v0.3.json", "test"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256_of_file(path: Path) -> str:
    """Return the hex SHA-256 digest of *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65_536), b""):
            h.update(block)
    return h.hexdigest()


def download_url(url: str, dest: Path) -> None:
    """Download *url* to *dest*, logging progress every 5 MB."""
    log.info("GET %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _reporthook(count: int, block: int, total: int) -> None:
        downloaded = count * block
        if total > 0 and downloaded % (5 * 1024 * 1024) < block:
            pct = 100 * downloaded // total
            log.info("  … %d %% (%d / %d bytes)", pct, downloaded, total)

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)  # noqa: S310
    log.info("Saved → %s (%d bytes)", dest, dest.stat().st_size)


def extract_json_from_tar(tar_path: Path, json_name: str, dest_dir: Path) -> Path:
    """Extract a single JSON file from a tarball and return its path."""
    with tarfile.open(tar_path) as tf:
        # Find the member matching json_name (ignores directory prefix)
        member = next(
            (m for m in tf.getmembers() if Path(m.name).name == json_name),
            None,
        )
        if member is None:
            raise FileNotFoundError(f"{json_name!r} not found in {tar_path}")
        tf.extract(member, path=dest_dir)
        extracted = dest_dir / member.name
        # Move to a flat location if nested
        flat = dest_dir / json_name
        if extracted != flat:
            extracted.rename(flat)
        return flat


def jsonobj_to_jsonl(json_path: Path, dest: Path) -> int:
    """Convert a QASPER JSON dict-of-papers to a JSONL file.

    QASPER JSON format: ``{paper_id: {title, abstract, full_text, qas, …}}``.
    Each output line is ``{"paper_id": ..., ...all other fields...}``.

    Returns the number of rows written.
    """
    with json_path.open(encoding="utf-8") as fh:
        data: dict[str, dict] = json.load(fh)

    dest.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with dest.open("w", encoding="utf-8") as out:
        for paper_id, paper in data.items():
            row = {"paper_id": paper_id, **paper}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    log.info("Wrote %d rows → %s", count, dest)
    return count


# ---------------------------------------------------------------------------
# Download strategies
# ---------------------------------------------------------------------------


def try_huggingface(data_dir: Path) -> dict[str, int] | None:
    """Try HuggingFace datasets library (future-proof if dataset migrates).

    Returns a dict {split: row_count} or None on failure.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]

        log.info("Attempting load_dataset('allenai/qasper') …")
        ds = load_dataset("allenai/qasper")
        counts: dict[str, int] = {}
        split_map = {"train": "train", "validation": "validation", "test": "test"}
        for split_label, hf_split in split_map.items():
            rows = list(ds[hf_split])  # type: ignore[index]
            dest = data_dir / f"{split_label}.jsonl"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split_label] = len(rows)
            log.info("  %s: %d rows", split_label, len(rows))
        return counts
    except Exception as exc:
        log.warning("HuggingFace load_dataset failed: %s", exc)
        return None


def download_from_s3(data_dir: Path) -> dict[str, int]:
    """Download directly from AllenAI S3 (reliable as of v0.3).

    Returns a dict {split: row_count}.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)

        # Download both tarballs
        tar_paths: dict[str, Path] = {}
        for key, url in S3_URLS.items():
            dest = tmp / f"{key}.tgz"
            download_url(url, dest)
            tar_paths[key] = dest

        # Extract and convert each split
        counts: dict[str, int] = {}
        for tar_key, json_name, split_label in TAR_TO_SPLIT:
            log.info("Extracting %s → %s …", json_name, split_label)
            json_path = extract_json_from_tar(tar_paths[tar_key], json_name, tmp)
            dest_jsonl = data_dir / f"{split_label}.jsonl"
            counts[split_label] = jsonobj_to_jsonl(json_path, dest_jsonl)

    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(data_dir: Path) -> None:
    # --- Download ----------------------------------------------------------
    counts = try_huggingface(data_dir)
    source: str
    if counts is not None:
        source = "allenai/qasper via HuggingFace datasets library"
        log.info("Used HuggingFace datasets path.")
    else:
        log.info("Falling back to direct S3 download …")
        counts = download_from_s3(data_dir)
        source = (
            "allenai/qasper v0.3 via official S3 release "
            "(qasper-train-dev-v0.3.tgz + qasper-test-and-evaluator-v0.3.tgz)"
        )

    # --- Build MANIFEST.json ----------------------------------------------
    manifest_splits: dict[str, dict] = {}
    for split, row_count in counts.items():
        jsonl = data_dir / f"{split}.jsonl"
        manifest_splits[split] = {
            "file": jsonl.name,
            "rows": row_count,
            "size_bytes": jsonl.stat().st_size,
            "sha256": sha256_of_file(jsonl),
        }

    manifest: dict = {
        "source": source,
        "download_date": datetime.now(UTC).isoformat(),
        "splits": manifest_splits,
    }
    manifest_path = data_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    log.info("MANIFEST written → %s", manifest_path)

    # --- Summary ----------------------------------------------------------
    total = sum(s["rows"] for s in manifest_splits.values())
    log.info("─" * 60)
    for split, meta in manifest_splits.items():
        log.info("  %-12s %d rows  %.1f MB", split, meta["rows"], meta["size_bytes"] / 1e6)
    log.info("  %-12s %d rows total", "", total)
    log.info("─" * 60)
    log.info("Source: %s", source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download QASPER v0.3 into data/raw/")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Destination directory (default: data/raw)",
    )
    args = parser.parse_args()
    main(args.data_dir)
