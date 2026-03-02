"""Download beatmap sets (audio + .osu files) into the training dataset structure.

Fetches .osz archives from mirror sites, extracts audio and .osu files,
and organizes them into: dataset/{beatmapset_id}/audio.{ext} + *.osu

Usage:
    python -m dataset_pipeline.download --dataset_dir dataset --limit 100
"""

import argparse
import asyncio
import enum
import io
import logging
import os
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import boto3
import httpx

logger = logging.getLogger(__name__)

BEATMAPS_API = "https://beatmaps.akatsuki.gg"
OSZ_MIRRORS = [
    # "https://catboy.best/d/{set_id}",  # down as of 2026-03-01
    "https://osu.direct/api/d/{set_id}",
    "https://api.nerinyan.moe/d/{set_id}",
    "https://storage.ripple.moe/d/{set_id}",
    # "https://dl.sayobot.cn/beatmaps/download/full/{set_id}",  # too slow (p50=17s, p90=64s)
]
ZIP_MAGIC = b"PK\x03\x04"

MAX_CONCURRENT_RESOLVE = 5
CHUNK_SIZE = 200

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
RETRYABLE_STATUSES = {429, 425, 500, 502, 503}
NON_RETRYABLE_MIRROR_STATUSES: dict[str, set[int]] = {
    "storage.ripple.moe": {500},
}

MINO_RATELIMIT_KEY = "REDACTED"

MIRROR_RATE: dict[str, float] = {
    # "catboy.best": 1.5,
    "osu.direct": 1.5,
    "api.nerinyan.moe": 0.75,
    "dl.sayobot.cn": 1.0,
    "storage.ripple.moe": 0.75,
}

# Base allocation weights for distributing work across mirrors.
# Higher weight = more items assigned upfront. Separate from concurrency
# because slow mirrors may support high concurrency only due to high latency.
MIRROR_WEIGHT = {
    "osu.direct": 6,
    "api.nerinyan.moe": 4,
    "storage.ripple.moe": 1,
    "dl.sayobot.cn": 0,  # too slow (~16s avg); fallback-only
}

MIRROR_EXTRA_HEADERS: dict[str, dict[str, str]] = {
    # "catboy.best": {"x-ratelimit-key": MINO_RATELIMIT_KEY},
}


class RateLimiter:
    """Limits request initiation rate (not concurrency)."""

    def __init__(self, requests_per_second: float) -> None:
        self._interval = 1.0 / requests_per_second
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._last_request + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()


MIRROR_RATE_LIMITERS: dict[str, RateLimiter] = {}
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


def list_random_beatmap_ids_from_s3(limit: int) -> list[int]:
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
    for page in paginator.paginate(
        Bucket=os.environ["AWS_BUCKET_NAME"], Prefix="beatmaps/"
    ):
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


def _get_mirror_rate_limiter(mirror_host: str) -> RateLimiter:
    if mirror_host not in MIRROR_RATE_LIMITERS:
        rps = MIRROR_RATE.get(mirror_host, 0.5)
        MIRROR_RATE_LIMITERS[mirror_host] = RateLimiter(rps)
    return MIRROR_RATE_LIMITERS[mirror_host]


def _record_mirror_stat(mirror_host: str, result: str) -> None:
    if mirror_host not in MIRROR_STATS:
        MIRROR_STATS[mirror_host] = {
            "success": 0,
            "ratelimited": 0,
            "not_found": 0,
            "error": 0,
        }
    MIRROR_STATS[mirror_host][result] += 1


def _reset_mirror_stats() -> None:
    MIRROR_STATS.clear()


def _log_mirror_stats() -> None:
    for host, stats in sorted(MIRROR_STATS.items()):
        logger.info(
            "  %-25s  ok=%-4d  ratelimited=%-4d  not_found=%-4d  err=%-4d",
            host,
            stats["success"],
            stats["ratelimited"],
            stats["not_found"],
            stats["error"],
        )


class MirrorFailure(enum.Enum):
    NOT_FOUND = "not_found"
    RATELIMITED = "ratelimited"
    ERROR = "error"


async def _try_mirror(
    beatmapset_id: int,
    mirror_template: str,
    client: httpx.AsyncClient,
) -> bytes | MirrorFailure:
    """Try downloading from a single mirror with retries."""
    mirror_host = mirror_template.split("/")[2]
    rate_limiter = _get_mirror_rate_limiter(mirror_host)
    url = mirror_template.format(set_id=beatmapset_id)
    headers = MIRROR_EXTRA_HEADERS.get(mirror_host, {})

    for attempt in range(MAX_RETRIES):
        t_wait = time.monotonic()
        await rate_limiter.acquire()
        wait_ms = (time.monotonic() - t_wait) * 1000
        if wait_ms > 50:
            logger.debug(
                "Set %d: %s rate limit wait %.0fms",
                beatmapset_id, mirror_host, wait_ms,
            )
        try:
            t_req = time.monotonic()
            resp = await client.get(
                url, timeout=30, follow_redirects=True, headers=headers
            )
            req_ms = (time.monotonic() - t_req) * 1000

            if resp.status_code == 200 and len(resp.content) > 0:
                if not resp.content[:4].startswith(ZIP_MAGIC):
                    _record_mirror_stat(mirror_host, "ratelimited")
                    if attempt == 0:
                        logger.warning(
                            "Set %d: %s returned non-zip (content-type=%s)",
                            beatmapset_id,
                            mirror_host,
                            resp.headers.get("content-type", "unknown"),
                        )
                    await asyncio.sleep(RETRY_BACKOFF * (attempt + 1) + 3.0)
                    continue
                _record_mirror_stat(mirror_host, "success")
                logger.debug(
                    "Set %d: %s ok %.0fms (%d KB)",
                    beatmapset_id, mirror_host, req_ms,
                    len(resp.content) // 1024,
                )
                remaining = resp.headers.get("x-ratelimit-remaining")
                if remaining is not None and int(remaining) < 10:
                    logger.info(
                        "Rate limit low on %s (%s remaining), pausing",
                        mirror_host,
                        remaining,
                    )
                    await asyncio.sleep(5.0)
                return resp.content

            if resp.status_code == 404:
                _record_mirror_stat(mirror_host, "not_found")
                logger.debug(
                    "Set %d: %s 404 %.0fms",
                    beatmapset_id, mirror_host, req_ms,
                )
                return MirrorFailure.NOT_FOUND

            if resp.status_code in (429, 425):
                _record_mirror_stat(mirror_host, "ratelimited")
                retry_after = resp.headers.get("retry-after")
                delay = (
                    max(float(retry_after), 3.0)
                    if retry_after
                    else RETRY_BACKOFF * (attempt + 1) + 3.0
                )
                logger.warning(
                    "Set %d: HTTP %d from %s, retry %d/%d in %.0fs",
                    beatmapset_id,
                    resp.status_code,
                    mirror_host,
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
                continue

            non_retryable = NON_RETRYABLE_MIRROR_STATUSES.get(mirror_host)
            if non_retryable and resp.status_code in non_retryable:
                _record_mirror_stat(mirror_host, "error")
                logger.warning(
                    "Set %d: HTTP %d from %s (non-retryable)",
                    beatmapset_id, resp.status_code, mirror_host,
                )
                return MirrorFailure.ERROR

            if resp.status_code in RETRYABLE_STATUSES:
                _record_mirror_stat(mirror_host, "error")
                delay = RETRY_BACKOFF * (attempt + 1)
                logger.warning(
                    "Set %d: HTTP %d from %s, retry %d/%d in %.0fs",
                    beatmapset_id,
                    resp.status_code,
                    mirror_host,
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
                continue

            _record_mirror_stat(mirror_host, "error")
            logger.warning(
                "Set %d: HTTP %d from %s (url=%s)",
                beatmapset_id,
                resp.status_code,
                mirror_host,
                url,
            )
            return MirrorFailure.ERROR

        except httpx.TimeoutException:
            _record_mirror_stat(mirror_host, "error")
            logger.warning(
                "Set %d: timeout from %s (attempt %d/%d, url=%s)",
                beatmapset_id,
                mirror_host,
                attempt + 1,
                MAX_RETRIES,
                url,
            )
            continue

        except httpx.HTTPError as e:
            _record_mirror_stat(mirror_host, "error")
            logger.warning(
                "Set %d: %s from %s (url=%s)",
                beatmapset_id,
                type(e).__name__,
                mirror_host,
                url,
            )
            return MirrorFailure.ERROR

    # Retries exhausted — check what we were retrying
    # If last failure was ratelimit (429/non-zip), report as ratelimited
    # so the caller can re-queue on the same mirror instead of routing elsewhere
    return MirrorFailure.RATELIMITED


MAX_RATELIMIT_REQUEUES = 2


@dataclass
class DownloadItem:
    set_id: int
    tried_mirrors: set[str] = field(default_factory=set)
    ratelimit_retries: int = 0


def _parse_audio_filename(osu_bytes: bytes) -> str | None:
    """Extract AudioFilename from a .osu file's [General] section."""
    try:
        text = osu_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    in_general = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[General]":
            in_general = True
            continue
        if in_general:
            if stripped.startswith("["):
                break
            if stripped.startswith("AudioFilename:"):
                return stripped.split(":", 1)[1].strip()
    return None


def _extract_osz(content: bytes, song_dir: Path) -> bool:
    """Extract audio + .osu files from .osz bytes into song_dir."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            # Build a case-insensitive lookup for zip entries
            zip_entries = {name.lower(): name for name in zf.namelist()}

            # First pass: extract .osu files and collect referenced audio filenames
            osu_files: list[tuple[str, bytes]] = []
            audio_filenames: set[str] = set()

            for name in zf.namelist():
                basename = Path(name).name
                if basename and Path(basename).suffix.lower() == ".osu":
                    osu_bytes = zf.read(name)
                    osu_files.append((basename, osu_bytes))
                    audio_name = _parse_audio_filename(osu_bytes)
                    if audio_name:
                        audio_filenames.add(audio_name)

            if not osu_files:
                logger.warning("Set %s: no .osu files in archive", song_dir.name)
                return False

            # Find the audio file referenced by the .osu files
            audio_zip_name: str | None = None
            for audio_name in audio_filenames:
                # Try exact match first, then case-insensitive
                if audio_name in zf.namelist():
                    audio_zip_name = audio_name
                    break
                lower = audio_name.lower()
                if lower in zip_entries:
                    audio_zip_name = zip_entries[lower]
                    break

            if audio_zip_name is None:
                logger.warning(
                    "Set %s: audio file not found in archive (referenced: %s)",
                    song_dir.name,
                    audio_filenames,
                )
                return False

            # Extract
            song_dir.mkdir(parents=True, exist_ok=True)

            audio_suffix = Path(audio_zip_name).suffix.lower()
            audio_dest = song_dir / f"audio{audio_suffix}"
            audio_dest.write_bytes(zf.read(audio_zip_name))

            for basename, osu_bytes in osu_files:
                (song_dir / basename).write_bytes(osu_bytes)

    except zipfile.BadZipFile:
        logger.warning("Set %s: bad zip file", song_dir.name)
        return False

    return True


OVERFLOW_FRACTION = 0.5


async def _mirror_worker(
    mirror_template: str,
    base_queue: asyncio.Queue[DownloadItem],
    overflow_queue: asyncio.Queue[DownloadItem | None],
    fallback_queue: asyncio.Queue[DownloadItem],
    output_dir: Path,
    completed: set[int],
    route_failure: Callable[[DownloadItem], None],
    on_resolved: Callable[[], None],
    client: httpx.AsyncClient,
) -> None:
    """Process guaranteed base items, then steal from overflow queue."""
    tasks: list[asyncio.Task[None]] = []

    mirror_host = mirror_template.split("/")[2]

    async def _process(item: DownloadItem, source: str) -> None:
        if item.set_id in completed:
            on_resolved()
            return

        logger.debug(
            "Set %d: %s starting (source=%s)", item.set_id, mirror_host, source,
        )
        t0 = time.monotonic()
        result = await _try_mirror(item.set_id, mirror_template, client)
        dl_ms = (time.monotonic() - t0) * 1000

        if isinstance(result, bytes):
            song_dir = output_dir / str(item.set_id)
            t_ext = time.monotonic()
            ok = await asyncio.to_thread(_extract_osz, content=result, song_dir=song_dir)
            ext_ms = (time.monotonic() - t_ext) * 1000
            if ok:
                logger.debug(
                    "Set %d: %s done dl=%.0fms ext=%.0fms",
                    item.set_id, mirror_host, dl_ms, ext_ms,
                )
                completed.add(item.set_id)
                on_resolved()
                return

        if result is MirrorFailure.RATELIMITED:
            item.ratelimit_retries += 1
            if item.ratelimit_retries <= MAX_RATELIMIT_REQUEUES:
                logger.debug(
                    "Set %d: %s ratelimited %.0fms, re-queuing (%d/%d)",
                    item.set_id, mirror_host, dl_ms,
                    item.ratelimit_retries, MAX_RATELIMIT_REQUEUES,
                )
                base_queue.put_nowait(item)
                return
            # Exhausted re-queues, fall through to route_failure

        logger.debug(
            "Set %d: %s failed %.0fms, routing to next mirror",
            item.set_id, mirror_host, dl_ms,
        )
        item.tried_mirrors.add(mirror_host)
        route_failure(item)

    def _try_get_item() -> tuple[DownloadItem, str] | None:
        """Check base → fallback → None (non-blocking)."""
        for q, label in ((base_queue, "base"), (fallback_queue, "fallback")):
            try:
                return q.get_nowait(), label
            except asyncio.QueueEmpty:
                pass
        return None

    while True:
        result = _try_get_item()

        if result is not None:
            item, source = result
        else:
            try:
                overflow_item = await asyncio.wait_for(
                    overflow_queue.get(), timeout=0.5,
                )
            except asyncio.TimeoutError:
                continue
            if overflow_item is None:
                overflow_queue.put_nowait(None)
                break
            item, source = overflow_item, "overflow"

        tasks.append(asyncio.create_task(_process(item, source)))

    if tasks:
        await asyncio.gather(*tasks)


async def download_chunk(
    set_ids: list[int],
    output_dir: Path,
    client: httpx.AsyncClient,
    *,
    force: bool = False,
) -> tuple[int, list[int]]:
    """Download a chunk of beatmap sets using parallel mirrors."""
    # Filter already-cached items
    if force:
        to_download = set_ids
    else:
        to_download = [
            sid
            for sid in set_ids
            if not (output_dir / str(sid)).exists()
            or not any((output_dir / str(sid)).glob("*.osu"))
        ]

    if not to_download:
        return len(set_ids), []

    items = [DownloadItem(set_id=sid) for sid in to_download]
    total = len(items)
    completed: set[int] = set()
    failed_ids: list[int] = []
    resolved_count = 0
    all_resolved = asyncio.Event()

    # Per-mirror base queues (guaranteed allocation) + fallback queues (failure routing)
    base_queues: dict[str, asyncio.Queue[DownloadItem]] = {
        template: asyncio.Queue() for template in OSZ_MIRRORS
    }
    fallback_queues: dict[str, asyncio.Queue[DownloadItem]] = {
        template: asyncio.Queue() for template in OSZ_MIRRORS
    }
    overflow_queue: asyncio.Queue[DownloadItem | None] = asyncio.Queue()

    def on_resolved() -> None:
        nonlocal resolved_count
        resolved_count += 1
        if resolved_count >= total:
            all_resolved.set()

    def route_failure(item: DownloadItem) -> None:
        for template in OSZ_MIRRORS:
            host = template.split("/")[2]
            if host not in item.tried_mirrors:
                fallback_queues[template].put_nowait(item)
                return
        # All mirrors exhausted
        failed_ids.append(item.set_id)
        on_resolved()

    # Split items: base allocation per mirror + overflow for work-stealing
    weights = [
        (template, MIRROR_WEIGHT.get(template.split("/")[2], 1))
        for template in OSZ_MIRRORS
    ]
    base_count = int(total * (1.0 - OVERFLOW_FRACTION))

    schedule: list[str] = []
    for template, weight in weights:
        schedule.extend([template] * weight)

    for i, item in enumerate(items):
        if i < base_count:
            template = schedule[i % len(schedule)]
            base_queues[template].put_nowait(item)
        else:
            overflow_queue.put_nowait(item)

    workers = [
        asyncio.create_task(
            _mirror_worker(
                template,
                base_queues[template],
                overflow_queue,
                fallback_queues[template],
                output_dir,
                completed,
                route_failure,
                on_resolved,
                client,
            )
        )
        for template in OSZ_MIRRORS
    ]

    await all_resolved.wait()

    # Send a single sentinel; workers propagate it to each other
    overflow_queue.put_nowait(None)

    await asyncio.gather(*workers)

    cached = len(set_ids) - total
    return cached + len(completed), failed_ids


async def download_all(
    set_ids: list[int],
    output_dir: Path,
    chunk_size: int,
    *,
    force: bool = False,
) -> int:
    """Download all beatmap sets in chunks to avoid overwhelming mirrors."""
    total_success = 0
    all_failed: list[int] = []

    async with httpx.AsyncClient() as client:
        for i in range(0, len(set_ids), chunk_size):
            chunk = set_ids[i : i + chunk_size]
            chunk_num = i // chunk_size + 1
            total_chunks = (len(set_ids) + chunk_size - 1) // chunk_size

            # Count how many in this chunk are already cached
            if force:
                cached = 0
            else:
                cached = sum(
                    1
                    for sid in chunk
                    if (output_dir / str(sid)).exists()
                    and any((output_dir / str(sid)).glob("*.osu"))
                )

            logger.info(
                "Chunk %d/%d: %d sets (%d cached, %d to download)",
                chunk_num,
                total_chunks,
                len(chunk),
                cached,
                len(chunk) - cached,
            )

            _reset_mirror_stats()
            success, failed = await download_chunk(
                chunk,
                output_dir,
                client,
                force=force,
            )
            total_success += success
            all_failed.extend(failed)

            new_downloads = success - cached
            logger.info(
                "Chunk %d/%d complete: %d success (%d new), %d failed",
                chunk_num,
                total_chunks,
                success,
                new_downloads,
                len(failed),
            )
            if MIRROR_STATS:
                _log_mirror_stats()

            # Pause between chunks to let rate limits recover
            if i + chunk_size < len(set_ids) and (len(chunk) - cached) > 0:
                logger.info("Pausing 5s between chunks")
                await asyncio.sleep(5.0)

    if all_failed:
        logger.warning(
            "Failed to download %d sets: %s", len(all_failed), all_failed[:20]
        )
        if len(all_failed) > 20:
            logger.warning("... and %d more", len(all_failed) - 20)

    return total_success


async def run(
    dataset_dir: str,
    *,
    set_ids_file: str | None = None,
    offset: int = 0,
    limit: int = 100,
    chunk_size: int = CHUNK_SIZE,
    dry_run: bool = False,
    force: bool = False,
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
        sets_to_download = sets_to_download[offset : offset + limit]
        logger.info(
            "Loaded %d beatmapset IDs from %s (offset=%d, limit=%d)",
            len(sets_to_download),
            set_ids_path,
            offset,
            limit,
        )
    else:
        logger.info("Listing beatmap IDs from S3...")
        beatmap_ids = list_random_beatmap_ids_from_s3(limit)
        logger.info("Found %d beatmap IDs in S3", len(beatmap_ids))

        logger.info("Resolving beatmap IDs to set IDs...")
        sets_to_download = await resolve_all_set_ids(beatmap_ids, limit)
        logger.info("Will download %d beatmap sets", len(sets_to_download))

    if dry_run:
        for sid in sets_to_download:
            logger.info("[dry run] Would download set %d", sid)
        return

    success = await download_all(sets_to_download, output_dir, chunk_size, force=force)
    logger.info(
        "Done. Downloaded %d/%d beatmap sets to %s",
        success,
        len(sets_to_download),
        output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download beatmap sets for training")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument(
        "--offset", type=int, default=0, help="Skip first N beatmap sets"
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Max beatmap sets to download"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if already extracted"
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument(
        "--set-ids-file",
        type=str,
        default=None,
        help="TSV file with beatmapset_id in first column (skips S3/Cheesegull)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    asyncio.run(
        run(
            args.dataset_dir,
            set_ids_file=args.set_ids_file,
            offset=args.offset,
            limit=args.limit,
            chunk_size=args.chunk_size,
            dry_run=args.dry_run,
            force=args.force,
        )
    )
