"""Dataset download.

Spider ONLY until week 9. Spider carries the project through week 8 and nothing else is
needed before then.

The HuggingFace Spider dataset (xlangai/spider) ships questions and gold SQL but NOT the
sqlite database files. The entire reward depends on executing queries, so questions alone
are useless -- the databases have to come from the Google Drive archive.

NEVER download BIRD train_databases: 8 GB, and we never train on BIRD.

Usage:
    python scripts/download_data.py --spider-only
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import zipfile

RAW_DIR = os.path.join("data", "raw")
SPIDER_DIR = os.path.join(RAW_DIR, "spider")
SPIDER_ZIP = os.path.join(RAW_DIR, "spider.zip")

# Spider 1.0 archive on Google Drive. If this id rots, the project page is
# https://yale-lily.github.io/spider and the per-database mirror is
# https://raw.githubusercontent.com/taoyds/spider/master/database/{db}/{db}.sqlite
SPIDER_GDRIVE_ID = "1TqleXec_OykOYFREKKtschzY29dUcVAQ"

EXPECTED_DB_COUNT = 166

REQUIRED_FILES = (
    "train_spider.json",
    "dev.json",
    "tables.json",
)


def _have_spider() -> bool:
    """True if Spider already looks downloaded, so the script is safe to re-run."""
    return os.path.exists(SPIDER_ZIP)


def download_spider(force: bool = False) -> None:
    """Fetch and unpack the Spider archive into data/raw/spider/."""
    if _have_spider() and not force:
        print("spider already present, skipping download")
        return

    try:
        import gdown
    except ImportError:
        raise SystemExit(
            "gdown is required for the Spider download: pip install gdown"
        )

    os.makedirs(RAW_DIR, exist_ok=True)

    print("downloading spider (~1 GB) ...")
    gdown.download(id=SPIDER_GDRIVE_ID, output=SPIDER_ZIP, quiet=False)

    print("extracting ...")
    with zipfile.ZipFile(SPIDER_ZIP) as archive:
        archive.extractall(RAW_DIR)

    # The archive unpacks as "spider_data" or "spider" depending on the release.
    for candidate in ("spider_data", "spider"):
        path = os.path.join(RAW_DIR, candidate)
        if os.path.isdir(path) and path != SPIDER_DIR:
            shutil.move(path, SPIDER_DIR)
            break

    print("extracted to %s" % SPIDER_DIR)


def verify_spider() -> None:
    """Fail loudly if the download is incomplete.

    A half-extracted Spider is much worse than no Spider: the filtering pass would drop
    thousands of examples for "missing db file" and the counts would look plausible.
    """
    missing = [
        name
        for name in REQUIRED_FILES
        if not os.path.exists(os.path.join(SPIDER_DIR, name))
    ]
    if missing:
        raise SystemExit("spider is incomplete, missing: %s" % ", ".join(missing))

    db_root = os.path.join(SPIDER_DIR, "database")
    if not os.path.isdir(db_root):
        raise SystemExit("no database/ directory under %s" % SPIDER_DIR)

    entries = glob.glob(os.path.join(db_root, "*"))
    count = len(entries)

    print("found %d databases (expected ~%d)" % (count, EXPECTED_DB_COUNT))
    assert count == EXPECTED_DB_COUNT, (
        "expected %d database folders, found %d -- extraction probably incomplete"
        % (EXPECTED_DB_COUNT, count)
    )

    print("spider looks complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="download project datasets")
    parser.add_argument(
        "--spider-only",
        action="store_true",
        help="download Spider and nothing else (the default and the only supported mode "
        "before week 9)",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        verify_spider()
        return

    download_spider(force=args.force)
    verify_spider()

    print()
    print("next: python -m src.data.filter --report")


if __name__ == "__main__":
    main()
