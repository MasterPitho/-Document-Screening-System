"""Lightweight in-memory rate limiting and concurrency resource protection."""

from __future__ import annotations

import asyncio
import collections
import time
from typing import Optional
from fastapi import HTTPException, Request

from app.config import Settings


class ResourceLimiter:
    """Controls concurrency and rate limits for resource-intensive operations."""

    def __init__(self, settings: Settings) -> None:
        self.max_concurrent = max(1, settings.max_concurrent_screenings)
        self.rate_limit_per_minute = max(1, settings.screening_rate_limit_per_minute)
        self._semaphore = asyncio.BoundedSemaphore(self.max_concurrent)
        self._user_requests: dict[str, collections.deque[float]] = collections.defaultdict(collections.deque)
        self._lock = asyncio.Lock()

    def get_client_identifier(self, request: Request, user_id: Optional[int] = None) -> str:
        if user_id is not None:
            return f"user:{user_id}"
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"

    async def check_rate_limit(self, identifier: str) -> None:
        now = time.monotonic()
        cutoff = now - 60.0
        async with self._lock:
            timestamps = self._user_requests[identifier]
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()
            if len(timestamps) >= self.rate_limit_per_minute:
                retry_after = int(max(1.0, 60.0 - (now - timestamps[0])))
                raise HTTPException(
                    status_code=429,
                    detail="Screening rate limit exceeded. Please wait before submitting more requests.",
                    headers={"Retry-After": str(retry_after)},
                )
            timestamps.append(now)

    async def acquire_slot(self) -> None:
        """Acquire a screening concurrency slot or fail immediately with HTTP 429."""
        if self._semaphore.locked():
            raise HTTPException(
                status_code=429,
                detail="Screening engine capacity reached; please retry shortly.",
                headers={"Retry-After": "5"},
            )
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=0.05)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=429,
                detail="Screening engine capacity reached; please retry shortly.",
                headers={"Retry-After": "5"},
            )

    def release_slot(self) -> None:
        try:
            self._semaphore.release()
        except ValueError:
            pass
