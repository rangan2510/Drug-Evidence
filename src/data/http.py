"""Cached HTTP client factory.

Provides a disk-backed (SQLite) async HTTP client via hishel on top of
httpx.  All data-source clients should create their ``httpx.AsyncClient``
through :func:`cached_async_client` to benefit from transparent caching.

Design decisions
----------------
* **Force-cache everything** -- biomedical API responses rarely change and
  do not set ``Cache-Control`` headers.  ``allow_stale=True`` combined
  with ``default_ttl`` on the storage ensures every response is cached
  regardless of response headers.
* **POST is cached** -- OpenTargets and DGIdb use GraphQL (POST).  The
  ``supported_methods`` option includes POST so that request-body-keyed
  caching works out of the box.
* **Single SQLite file** -- simple, portable, no external services needed.
  Located at ``<project>/.cache/http_cache.db`` by default.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
from hishel import AsyncSqliteStorage, CacheOptions, SpecificationPolicy
from hishel.httpx import AsyncCacheClient

from src.config.settings import Settings

logger = logging.getLogger(__name__)

# Module-level singleton storage / policy so every client shares one DB
# connection pool.  Created lazily on first call.
_storage: AsyncSqliteStorage | None = None
_policy: SpecificationPolicy | None = None


def _ensure_globals(settings: Settings) -> tuple[AsyncSqliteStorage, SpecificationPolicy]:
    """Lazily initialise the module-level storage and policy singletons."""
    global _storage, _policy  # noqa: PLW0603

    if _storage is None:
        cache_dir = Path(settings.http_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        db_path = cache_dir / "http_cache.db"

        _storage = AsyncSqliteStorage(
            database_path=str(db_path),
            default_ttl=float(settings.http_cache_ttl),
            refresh_ttl_on_access=True,
        )
        logger.info(
            "HTTP cache: SQLite at %s  (TTL %d s)",
            db_path,
            settings.http_cache_ttl,
        )

    if _policy is None:
        _policy = SpecificationPolicy(
            cache_options=CacheOptions(
                supported_methods=["GET", "POST"],
                allow_stale=True,
            ),
        )

    return _storage, _policy


def cached_async_client(
    settings: Settings | None = None,
    **kwargs: object,
) -> AsyncCacheClient:
    """Return an ``httpx.AsyncClient`` subclass backed by disk cache.

    Parameters
    ----------
    settings
        Application settings (cache dir, TTL).  Falls back to defaults.
    **kwargs
        Forwarded to the underlying ``httpx.AsyncClient`` constructor
        (e.g. ``timeout``, ``headers``).

    Returns
    -------
    AsyncCacheClient
        Drop-in replacement for ``httpx.AsyncClient`` that transparently
        caches responses in a shared SQLite database.
    """
    settings = settings or Settings()
    storage, policy = _ensure_globals(settings)
    return AsyncCacheClient(storage=storage, policy=policy, **kwargs)


def reset_cache_globals() -> None:
    """Reset module singletons (useful in tests)."""
    global _storage, _policy  # noqa: PLW0603
    _storage = None
    _policy = None
