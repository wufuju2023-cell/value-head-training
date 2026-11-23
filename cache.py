#!/usr/bin/env python3
"""Reuse / caching for mission-level MCTS: registry, compile lock, dedupe.

Three separate wastes this module removes:

1. **Duplicated missions.**  Two sibling decompositions of the same goal in the
   same env are the same mission.  `MissionRegistry` keys on
   `sha256(env, statement_formal)` so the mission id and its workspace path are
   returned again with `reused: true` instead of being created twice.
2. **Duplicated compiles.**  One shared Lean project per env, one
   content-addressed task dir per (file, content), and a lock so two children
   never run `lean` on the same file at the same time.  Every compile reports
   `cache_hit`.
3. **Duplicated siblings.**  Identical missions created inside *one*
   decomposition are dropped before dispatch (`dedupe_siblings`), because "prove
   the same lemma twice in parallel" spends search budget and teaches the value
   head nothing.

No network, no Lean binary required: the compile step takes an injectable
`runner`, and the default runner writes the `.olean` marker only (CPU tests).

    python3 -c "import cache, tempfile; print(cache.mission_id_for('bank','theorem t : True'))"
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

try:  # POSIX only; the registry still works without it
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore

REGISTRY_VERSION = 1
STATUSES = ("pending", "running", "accepted", "failed", "split", "blocked")
MISSION_PREFIX = "m-"


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def normalise_statement(statement_formal: str) -> str:
    """Whitespace-only normalisation: the statement, not its formatting."""
    return re.sub(r"\s+", " ", str(statement_formal or "")).strip()


def statement_hash(env: str, statement_formal: str) -> str:
    """sha256(env, statement_formal) — the registry key."""
    payload = f"{env}\x00{normalise_statement(statement_formal)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def mission_id_for(env: str, statement_formal: str) -> str:
    """Deterministic, human-greppable id: 12 hex chars of the statement hash."""
    return MISSION_PREFIX + statement_hash(env, statement_formal)[:12]


def env_slug(env: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(env or "default")).strip("._")
    return slug or "default"


# ---------------------------------------------------------------------------
# Mission registry
# ---------------------------------------------------------------------------

class MissionRegistry:
    """Content-addressed mission table, persisted as one JSON file.

    key      sha256(env, statement_formal)
    value    {mission_id, env, statement_formal, workspace, lean_project,
              created_at, status, depth, parent_mission_id, hits}
    """

    def __init__(self, root: str | os.PathLike, *, path: str | None = None,
                 autosave: bool = True):
        self.root = Path(root)
        self.path = Path(path) if path else self.root / "mission_registry.json"
        self.autosave = autosave
        self._lock = threading.RLock()
        self._missions: dict[str, dict] = {}
        self.reuse_count = 0
        self.created_count = 0
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            blob = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(blob, dict) and isinstance(blob.get("missions"), dict):
            self._missions = blob["missions"]

    def save(self) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"version": REGISTRY_VERSION,
                                   "missions": self._missions},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)   # atomic: a killed run never leaves half a table
        return str(self.path)

    # -- api ---------------------------------------------------------------
    def get_or_create(self, env: str, statement_formal: str, *,
                      lean_project: str | None = None,
                      workspace_root: str | os.PathLike | None = None,
                      depth: int = 0,
                      parent_mission_id: str | None = None,
                      status: str = "pending") -> dict:
        key = statement_hash(env, statement_formal)
        with self._lock:
            hit = self._missions.get(key)
            if hit is not None:
                hit["hits"] = int(hit.get("hits", 0)) + 1
                self.reuse_count += 1
                if self.autosave:
                    self.save()
                return {**hit, "key": key, "reused": True}

            mission_id = mission_id_for(env, statement_formal)
            ws_root = Path(workspace_root) if workspace_root else self.root / "missions"
            workspace = ws_root / mission_id
            workspace.mkdir(parents=True, exist_ok=True)
            rec = {
                "mission_id": mission_id,
                "env": env,
                "statement_formal": normalise_statement(statement_formal),
                "workspace": str(workspace),
                "lean_project": lean_project or str(self.root / "olean" / env_slug(env) / "project"),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "status": status if status in STATUSES else "pending",
                "depth": int(depth),
                "parent_mission_id": parent_mission_id,
                "hits": 0,
            }
            self._missions[key] = rec
            self.created_count += 1
            if self.autosave:
                self.save()
            return {**rec, "key": key, "reused": False}

    def get(self, mission_id: str) -> dict | None:
        with self._lock:
            for rec in self._missions.values():
                if rec["mission_id"] == mission_id:
                    return dict(rec)
        return None

    def by_key(self, key: str) -> dict | None:
        with self._lock:
            rec = self._missions.get(key)
            return dict(rec) if rec else None

    def set_status(self, mission_id: str, status: str) -> dict:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}; expected one of {STATUSES}")
        with self._lock:
            for rec in self._missions.values():
                if rec["mission_id"] == mission_id:
                    rec["status"] = status
                    if self.autosave:
                        self.save()
                    return dict(rec)
        raise KeyError(f"mission not registered: {mission_id}")

    def stats(self) -> dict:
        with self._lock:
            by_status: dict[str, int] = {}
            for rec in self._missions.values():
                by_status[rec.get("status", "pending")] = by_status.get(rec.get("status", "pending"), 0) + 1
            return {"missions": len(self._missions), "created": self.created_count,
                    "reused": self.reuse_count, "by_status": by_status}

    def __len__(self) -> int:
        return len(self._missions)


# ---------------------------------------------------------------------------
# Sibling dedupe
# ---------------------------------------------------------------------------

def dedupe_siblings(children: list[dict], *, env: str | None = None) -> dict:
    """Drop identical sibling missions (same env, same statement hash).

    Returns {"kept": [...], "dropped": [...], "log": [...]}.  Kept entries get
    `dedupe_key`; dropped entries record which kept index they duplicate.
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    log: list[dict] = []
    seen: dict[str, int] = {}
    for idx, child in enumerate(children):
        child_env = child.get("env") or env or "default"
        statement = child.get("statement_formal") or child.get("goal") or ""
        key = statement_hash(child_env, statement)
        if key in seen:
            dropped.append({**child, "dedupe_key": key, "reused": True,
                            "duplicate_of": seen[key]})
            log.append({"reused": True, "statement_hash": key,
                        "dropped_index": idx, "kept_index": seen[key]})
            continue
        seen[key] = len(kept)
        kept.append({**child, "env": child_env, "dedupe_key": key})
        log.append({"reused": False, "statement_hash": key, "kept_index": seen[key]})
    return {"kept": kept, "dropped": dropped, "log": log}


def register_children(registry: MissionRegistry, env: str,
                      children: list[dict], *,
                      parent_mission_id: str | None = None,
                      depth: int = 0,
                      lean_project: str | None = None,
                      workspace_root: str | os.PathLike | None = None) -> dict:
    """Dedupe, then register: the one path from "children created" to dispatch."""
    deduped = dedupe_siblings(children, env=env)
    missions = []
    for child in deduped["kept"]:
        rec = registry.get_or_create(
            child["env"],
            child.get("statement_formal") or child.get("goal") or "",
            lean_project=lean_project,
            workspace_root=workspace_root,
            depth=int(child.get("depth", depth)),
            parent_mission_id=parent_mission_id or child.get("parent_mission_id"),
        )
        missions.append(rec)
    return {"missions": missions, "dropped": deduped["dropped"], "log": deduped["log"]}


# ---------------------------------------------------------------------------
# Compile cache (one shared Lean project per env, content-addressed task dirs)
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _default_runner(task_dir: Path, file_name: str, olean: Path) -> dict:
    """CPU stand-in for `lake env lean <file>`: writes the .olean marker."""
    olean.parent.mkdir(parents=True, exist_ok=True)
    olean.write_text(f"olean-marker {file_name} {content_hash(file_name)[:8]}\n",
                     encoding="utf-8")
    return {"returncode": 0, "stderr": "", "simulated": True}


class CompileCache:
    """Shared Lean project per env + content-addressed task dirs + a lock.

        <root>/olean/<env>/project/            one project per env
        <root>/olean/<env>/tasks/<hash16>/     one dir per file content
            <name>.lean  <name>.olean  .lock
    """

    def __init__(self, root: str | os.PathLike, *,
                 runner: Callable[[Path, str, Path], dict] | None = None):
        self.root = Path(root)
        self.runner = runner or _default_runner
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self.events: list[dict] = []

    # -- layout ------------------------------------------------------------
    def project_dir(self, env: str) -> Path:
        return self.root / "olean" / env_slug(env) / "project"

    def task_dir(self, env: str, content: str) -> Path:
        return self.project_dir(env) / "tasks" / content_hash(content)[:16]

    def olean_path(self, env: str, file_name: str, content: str) -> Path:
        stem = Path(file_name).with_suffix(".olean").name
        return self.task_dir(env, content) / stem

    # -- api ---------------------------------------------------------------
    def compile(self, env: str, file_name: str, content: str, *,
                force: bool = False) -> dict:
        """Compile (or reuse) one file. Reports `cache_hit` per call."""
        content = str(content)
        key = content_hash(content)
        task = self.task_dir(env, content)
        task.mkdir(parents=True, exist_ok=True)
        src = task / Path(file_name).name
        if not src.is_file() or src.read_text(encoding="utf-8") != content:
            src.write_text(content, encoding="utf-8")
        olean = self.olean_path(env, file_name, content)

        t0 = time.time()
        inproc = self._lock_for(f"{env_slug(env)}/{key}")
        with inproc:
            lock_path = task / ".lock"
            fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)   # cross-process
                cache_hit = olean.is_file() and not force
                if cache_hit:
                    result = {"returncode": 0, "stderr": "", "simulated": False}
                else:
                    result = self.runner(task, Path(file_name).name, olean)
            finally:
                try:
                    if fcntl is not None:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)

        event = {
            "env": env, "file": Path(file_name).name, "content_hash": key,
            "task_dir": str(task), "olean": str(olean),
            "cache_hit": bool(cache_hit),
            "returncode": int(result.get("returncode", 0)),
            "seconds": round(time.time() - t0, 4),
        }
        self.events.append(event)
        return event

    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def stats(self) -> dict:
        compiles = len(self.events)
        hits = sum(1 for e in self.events if e["cache_hit"])
        return {"compiles": compiles, "cache_hits": hits,
                "misses": compiles - hits,
                "hit_rate": round(hits / compiles, 4) if compiles else 0.0,
                "envs": sorted({e["env"] for e in self.events})}


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        reg = MissionRegistry(tmp)
        a = reg.get_or_create("tiny", "theorem t : True")
        b = reg.get_or_create("tiny", "theorem t : True")
        c = reg.get_or_create("tiny", "theorem u : 1 = 1")
        cc = CompileCache(tmp)
        e1 = cc.compile("tiny", "Problem.lean", "theorem t : True := by trivial")
        e2 = cc.compile("tiny", "Problem.lean", "theorem t : True := by trivial")
        print(json.dumps({
            "same_id": a["mission_id"] == b["mission_id"],
            "reused": [a["reused"], b["reused"], c["reused"]],
            "missions": reg.stats(),
            "cache_hits": [e1["cache_hit"], e2["cache_hit"]],
            "compile_stats": cc.stats(),
        }))
