"""Hash-based result cache for experiment resumption.

Stores ``ArmResult`` JSON files keyed by a deterministic hash of
(drug_name, arm_id, model_id) so interrupted experiments can resume
without re-running completed drugs.

Layout::

    .cache/results/{hex_hash}.json

The cache is append-only -- results are never overwritten once written.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from src.experiment.runner import ArmResult

logger = logging.getLogger(__name__)


def _result_key(drug_name: str, arm_id: str, model_id: str) -> str:
    """Compute a deterministic SHA-256 hex key for a result."""
    raw = f"{drug_name.lower().strip()}|{arm_id}|{model_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ResultCache:
    """Persistent, hash-addressed cache for ``ArmResult`` objects.

    Parameters
    ----------
    cache_dir:
        Root directory for the cache.  Defaults to ``.cache/results``.
    """

    def __init__(self, cache_dir: str | Path = ".cache/results") -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def key(drug_name: str, arm_id: str, model_id: str) -> str:
        """Return the cache key for a given (drug, arm, model) triple."""
        return _result_key(drug_name, arm_id, model_id)

    def _path(self, hex_key: str) -> Path:
        return self._dir / f"{hex_key}.json"

    # ------------------------------------------------------------------
    # Read / write / query
    # ------------------------------------------------------------------

    def has(self, drug_name: str, arm_id: str, model_id: str) -> bool:
        """Return *True* if a cached result exists for this triple."""
        k = _result_key(drug_name, arm_id, model_id)
        return self._path(k).exists()

    def get(self, drug_name: str, arm_id: str, model_id: str) -> ArmResult | None:
        """Load a cached result, or *None* if not present."""
        k = _result_key(drug_name, arm_id, model_id)
        path = self._path(k)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ArmResult.model_validate(data)
        except Exception:
            logger.warning("Corrupt cache entry %s -- ignoring", path.name)
            return None

    def put(self, result: ArmResult) -> str:
        """Persist a result and return its cache key.

        If a result with the same key already exists it is **not**
        overwritten (append-only semantics).
        """
        k = _result_key(result.drug_name, result.arm_id, result.model_id)
        path = self._path(k)
        if path.exists():
            logger.debug("Cache hit (skip write) for key %s", k[:12])
            return k
        path.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.debug("Cached result %s -> %s", k[:12], path.name)
        return k

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def load_all(self) -> list[ArmResult]:
        """Load every cached result from the directory."""
        results: list[ArmResult] = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                results.append(ArmResult.model_validate(data))
            except Exception:
                logger.warning("Skipping corrupt cache entry: %s", path.name)
        return results

    def count(self) -> int:
        """Number of cached result files."""
        return sum(1 for _ in self._dir.glob("*.json"))

    def clear(self) -> int:
        """Delete all cached results.  Returns the number removed."""
        removed = 0
        for path in self._dir.glob("*.json"):
            path.unlink()
            removed += 1
        logger.info("Cleared %d cached results from %s", removed, self._dir)
        return removed
