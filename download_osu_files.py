"""Download .osu files from Wasabi S3 into a local directory."""
import argparse
import logging
import os
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)


def _load_dotenv(path: Path) -> None:
    """Load key=value pairs from a .env file into os.environ."""
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download .osu files from Wasabi S3")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="osu_files",
        help="Directory to save downloaded .osu files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of .osu files to download",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List files without downloading",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    _load_dotenv(Path(__file__).parent / ".env")

    endpoint_url = os.environ["AWS_ENDPOINT_URL"]
    region = os.environ["AWS_REGION"]
    bucket = os.environ["AWS_BUCKET_NAME"]
    access_key = os.environ["AWS_ACCESS_KEY_ID"]
    secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]

    s3_prefix = "beatmaps/"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    logger.info(
        "Listing .osu files from s3://%s/%s (limit=%d)",
        bucket,
        s3_prefix,
        args.limit,
    )

    downloaded = 0
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            if downloaded >= args.limit:
                break

            key = obj["Key"]
            if not key.endswith(".osu"):
                continue

            # Skip empty files and corrupt keys (e.g. tuple-formatted names)
            if obj["Size"] == 0:
                continue

            filename = key.removeprefix(s3_prefix)
            if not filename or "(" in filename:
                continue

            local_path = output_dir / filename

            if local_path.exists():
                downloaded += 1
                continue

            if args.dry_run:
                size_kb = obj["Size"] / 1024
                logger.info("[dry run] %s (%.1f KB)", key, size_kb)
                downloaded += 1
                continue

            s3.download_file(bucket, key, str(local_path))
            downloaded += 1

            if downloaded % 100 == 0:
                logger.info("Downloaded %d / %d files", downloaded, args.limit)

        if downloaded >= args.limit:
            break

    logger.info("Done. Downloaded %d .osu files to %s", downloaded, output_dir)


if __name__ == "__main__":
    main()
