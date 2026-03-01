"""Download beatmap sets (audio + .osu files) into the training dataset structure.

Fetches .osz archives from mirror sites, extracts audio and .osu files,
and organizes them into: dataset/{beatmapset_id}/audio.{ext} + *.osu

Usage:
    python -m dataset_pipeline.download --dataset_dir dataset --limit 100
"""
import argparse
import asyncio
import io
import logging
import os
import shutil
import zipfile
from pathlib import Path

import boto3
import httpx

logger = logging.getLogger(__name__)

BEATMAPS_API = "https://beatmaps.akatsuki.gg"
OSZ_MIRRORS = [
    "https://catboy.best/d/{set_id}",
    "https://osu.direct/api/d/{set_id}",
    "https://storage.ripple.moe/d/{set_id}",
]
AUDIO_EXTENSIONS = {".mp3", ".ogg", ".wav", ".flac"}
ZIP_MAGIC = b"PK\x03\x04"

MAX_CONCURRENT_RESOLVE = 5
MAX_CONCURRENT_DOWNLOAD = 2
CHUNK_SIZE = 200

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
RETRYABLE_STATUSES = {429, 425, 500, 502, 503}

MINO_RATELIMIT_KEY = "REDACTED"

MIRROR_CONCURRENCY = {
    "catboy.best": 2,
    "osu.direct": 2,
    "storage.ripple.moe": 1,
}

MIRROR_EXTRA_HEADERS: dict[str, dict[str, str]] = {
    "catboy.best": {"x-ratelimit-key": MINO_RATELIMIT_KEY},
}

MIRROR_SEMAPHORES: dict[str, asyncio.Semaphore] = {}
MIRROR_STATS: dict[str, dict[str, int]] = {}


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def list_beatmap_ids_from_s3(limit: int) -> list[int]:
    """List beatmap IDs from the S3 bucket."""
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        region_name=os.environ["AWS_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    ids: list[int] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=os.environ["AWS_BUCKET_NAME"], Prefix="beatmaps/"):
        for obj in page.get("Contents", []):
            if obj["Size"] == 0:
                continue
            key = obj["Key"]
            if not key.endswith(".osu"):
                continue
            filename = key.removeprefix("beatmaps/").removesuffix(".osu")
            if "(" in filename or not filename.isdigit():
                continue
            ids.append(int(filename))
            if len(ids) >= limit * 5:  # fetch extra since many will share sets
                return ids
    return ids


async def resolve_beatmapset_id(
    beatmap_id: int,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> int | None:
    """Resolve beatmap_id -> beatmapset_id via Cheesegull API."""
    async with semaphore:
        try:
            resp = await client.get(f"{BEATMAPS_API}/api/b/{beatmap_id}", timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data.get("ParentSetID")
        except httpx.HTTPError:
            return None


async def resolve_all_set_ids(
    beatmap_ids: list[int],
    limit: int,
) -> list[int]:
    """Resolve beatmap IDs to unique beatmapset IDs concurrently."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_RESOLVE)
    async with httpx.AsyncClient() as client:
        pending = [
            asyncio.create_task(resolve_beatmapset_id(bid, client, semaphore))
            for bid in beatmap_ids
        ]

        seen: set[int] = set()
        result: list[int] = []

        for coro in asyncio.as_completed(pending):
            set_id = await coro
            if set_id is None or set_id in seen:
                continue
            seen.add(set_id)
            result.append(set_id)
            if len(result) >= limit:
                break

        # Cancel remaining tasks before closing the client
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    return result


def _get_mirror_semaphore(mirror_host: str) -> asyncio.Semaphore:
    if mirror_host not in MIRROR_SEMAPHORES:
        limit = MIRROR_CONCURRENCY.get(mirror_host, 1)
        MIRROR_SEMAPHORES[mirror_host] = asyncio.Semaphore(limit)
    return MIRROR_SEMAPHORES[mirror_host]


def _record_mirror_stat(mirror_host: str, result: str) -> None:
    if mirror_host not in MIRROR_STATS:
        MIRROR_STATS[mirror_host] = {"success": 0, "ratelimited": 0, "not_found": 0, "error": 0}
    MIRROR_STATS[mirror_host][result] += 1


def _reset_mirror_stats() -> None:
    MIRROR_STATS.clear()


def _log_mirror_stats() -> None:
    for host, stats in sorted(MIRROR_STATS.items()):
        logger.info(
            "  %-25s  ok=%-4d  ratelimited=%-4d  not_found=%-4d  err=%-4d",
            host, stats["success"], stats["ratelimited"], stats["not_found"], stats["error"],
        )


async def _try_mirror(
    beatmapset_id: int,
    mirror_template: str,
    client: httpx.AsyncClient,
) -> bytes | None:
    """Try downloading from a single mirror with retries."""
    mirror_host = mirror_template.split("/")[2]
    semaphore = _get_mirror_semaphore(mirror_host)
    url = mirror_template.format(set_id=beatmapset_id)
    headers = MIRROR_EXTRA_HEADERS.get(mirror_host, {})

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.get(url, timeout=30, follow_redirects=True, headers=headers)

                if resp.status_code == 200 and len(resp.content) > 0:
                    if not resp.content[:4].startswith(ZIP_MAGIC):
                        # Some mirrors (gatari) return rate limit errors as 200 with JSON body
                        _record_mirror_stat(mirror_host, "ratelimited")
                        if attempt == 0:
                            logger.warning(
                                "Set %d: %s returned non-zip (content-type=%s)",
                                beatmapset_id, mirror_host,
                                resp.headers.get("content-type", "unknown"),
                            )
                        await asyncio.sleep(RETRY_BACKOFF * (attempt + 1) + 3.0)
                        continue
                    _record_mirror_stat(mirror_host, "success")
                    remaining = resp.headers.get("x-ratelimit-remaining")
                    if remaining is not None and int(remaining) < 10:
                        logger.info("Rate limit low on %s (%s remaining), pausing", mirror_host, remaining)
                        await asyncio.sleep(5.0)
                    return resp.content

                if resp.status_code == 404:
                    _record_mirror_stat(mirror_host, "not_found")
                    return None

                if resp.status_code in (429, 425):
                    _record_mirror_stat(mirror_host, "ratelimited")
                    retry_after = resp.headers.get("retry-after")
                    delay = max(float(retry_after), 3.0) if retry_after else RETRY_BACKOFF * (attempt + 1) + 3.0
                    logger.warning(
                        "Set %d: HTTP %d from %s, retry %d/%d in %.0fs",
                        beatmapset_id, resp.status_code, mirror_host,
                        attempt + 1, MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                if resp.status_code in RETRYABLE_STATUSES:
                    _record_mirror_stat(mirror_host, "error")
                    delay = RETRY_BACKOFF * (attempt + 1)
                    logger.warning(
                        "Set %d: HTTP %d from %s, retry %d/%d in %.0fs",
                        beatmapset_id, resp.status_code, mirror_host,
                        attempt + 1, MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                _record_mirror_stat(mirror_host, "error")
                logger.warning(
                    "Set %d: HTTP %d from %s (url=%s)",
                    beatmapset_id, resp.status_code, mirror_host, url,
                )
                return None

            except httpx.TimeoutException:
                _record_mirror_stat(mirror_host, "error")
                logger.warning(
                    "Set %d: timeout from %s (attempt %d/%d, url=%s)",
                    beatmapset_id, mirror_host, attempt + 1, MAX_RETRIES, url,
                )
                continue

            except httpx.HTTPError as e:
                _record_mirror_stat(mirror_host, "error")
                logger.warning(
                    "Set %d: %s from %s (url=%s)",
                    beatmapset_id, type(e).__name__, mirror_host, url,
                )
                return None

    return None


async def _download_osz(
    beatmapset_id: int,
    client: httpx.AsyncClient,
) -> bytes | None:
    """Try downloading .osz from all mirrors concurrently."""
    tasks = [
        asyncio.create_task(_try_mirror(beatmapset_id, template, client))
        for template in OSZ_MIRRORS
    ]

    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result is not None:
            for task in tasks:
                if not task.done():
                    task.cancel()
            return result

    return None


def _extract_osz(content: bytes, song_dir: Path) -> bool:
    """Extract audio + .osu files from .osz bytes into song_dir."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            audio_found = False
            osu_found = False

            song_dir.mkdir(parents=True, exist_ok=True)

            for name in zf.namelist():
                basename = Path(name).name
                if not basename:
                    continue

                suffix = Path(basename).suffix.lower()

                if suffix in AUDIO_EXTENSIONS and not audio_found:
                    audio_dest = song_dir / f"audio{suffix}"
                    audio_dest.write_bytes(zf.read(name))
                    audio_found = True

                elif suffix == ".osu":
                    osu_dest = song_dir / basename
                    osu_dest.write_bytes(zf.read(name))
                    osu_found = True

            if not audio_found or not osu_found:
                logger.warning("Set %s: incomplete osz (audio=%s osu=%s)", song_dir.name, audio_found, osu_found)
                if song_dir.exists():
                    shutil.rmtree(song_dir)
                return False

    except zipfile.BadZipFile:
        logger.warning("Set %s: bad zip file", song_dir.name)
        return False

    return True


async def download_and_extract(
    beatmapset_id: int,
    output_dir: Path,
    client: httpx.AsyncClient,
) -> bool:
    """Download .osz and extract audio + .osu files."""
    song_dir = output_dir / str(beatmapset_id)
    if song_dir.exists() and any(song_dir.glob("*.osu")):
        return True

    content = await _download_osz(beatmapset_id, client)

    if content is None:
        return False

    # Extraction is CPU-bound, run in thread to avoid blocking event loop
    return await asyncio.to_thread(_extract_osz, content, song_dir)


MAX_CONCURRENT_DOWNLOADS_PER_CHUNK = 4


async def download_chunk(
    set_ids: list[int],
    output_dir: Path,
    client: httpx.AsyncClient,
) -> tuple[int, list[int]]:
    """Download a chunk of beatmap sets. Returns (success_count, failed_ids)."""
    download_sem = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS_PER_CHUNK)

    async def _limited_download(sid: int) -> tuple[int, bool]:
        async with download_sem:
            result = await download_and_extract(sid, output_dir, client)
            return sid, result

    tasks = [asyncio.create_task(_limited_download(sid)) for sid in set_ids]

    success = 0
    failed: list[int] = []
    for task in asyncio.as_completed(tasks):
        sid, result = await task
        if result:
            success += 1
        else:
            failed.append(sid)

    return success, failed


async def download_all(
    set_ids: list[int],
    output_dir: Path,
    chunk_size: int,
) -> int:
    """Download all beatmap sets in chunks to avoid overwhelming mirrors."""
    total_success = 0
    all_failed: list[int] = []

    async with httpx.AsyncClient() as client:
        for i in range(0, len(set_ids), chunk_size):
            chunk = set_ids[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            total_chunks = (len(set_ids) + chunk_size - 1) // chunk_size

            # Count how many in this chunk are already cached
            to_download = [
                sid for sid in chunk
                if not (output_dir / str(sid)).exists()
                or not any((output_dir / str(sid)).glob("*.osu"))
            ]
            cached = len(chunk) - len(to_download)

            logger.info(
                "Chunk %d/%d: %d sets (%d cached, %d to download)",
                chunk_num, total_chunks, len(chunk), cached, len(to_download),
            )

            _reset_mirror_stats()
            success, failed = await download_chunk(chunk, output_dir, client)
            total_success += success
            all_failed.extend(failed)

            new_downloads = success - cached
            logger.info(
                "Chunk %d/%d complete: %d success (%d new), %d failed",
                chunk_num, total_chunks, success, new_downloads, len(failed),
            )
            if MIRROR_STATS:
                _log_mirror_stats()

            # Pause between chunks to let rate limits recover
            if i + chunk_size < len(set_ids) and new_downloads > 0:
                logger.info("Pausing 5s between chunks")
                await asyncio.sleep(5.0)

    if all_failed:
        logger.warning("Failed to download %d sets: %s", len(all_failed), all_failed[:20])
        if len(all_failed) > 20:
            logger.warning("... and %d more", len(all_failed) - 20)

    return total_success


async def run(
    dataset_dir: str,
    *,
    set_ids_file: str | None = None,
    limit: int = 100,
    chunk_size: int = CHUNK_SIZE,
    dry_run: bool = False,
) -> None:
    """Download beatmap sets into the dataset directory."""
    _load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    output_dir = Path(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if set_ids_file:
        set_ids_path = Path(set_ids_file)
        sets_to_download = []
        with open(set_ids_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0].isdigit():
                    sets_to_download.append(int(parts[0]))
        sets_to_download = sets_to_download[:limit]
        logger.info("Loaded %d beatmapset IDs from %s", len(sets_to_download), set_ids_path)
    else:
        logger.info("Listing beatmap IDs from S3...")
        beatmap_ids = list_beatmap_ids_from_s3(limit)
        logger.info("Found %d beatmap IDs in S3", len(beatmap_ids))

        logger.info("Resolving beatmap IDs to set IDs...")
        sets_to_download = await resolve_all_set_ids(beatmap_ids, limit)
        logger.info("Will download %d beatmap sets", len(sets_to_download))

    if dry_run:
        for sid in sets_to_download:
            logger.info("[dry run] Would download set %d", sid)
        return

    success = await download_all(sets_to_download, output_dir, chunk_size)
    logger.info("Done. Downloaded %d/%d beatmap sets to %s", success, len(sets_to_download), output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download beatmap sets for training")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--limit", type=int, default=100, help="Max beatmap sets to download")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument(
        "--set_ids_file", type=str, default=None,
        help="TSV file with beatmapset_id in first column (skips S3/Cheesegull)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    args = parse_args()
    asyncio.run(run(
        args.dataset_dir,
        set_ids_file=args.set_ids_file,
        limit=args.limit,
        chunk_size=args.chunk_size,
        dry_run=args.dry_run,
    ))
