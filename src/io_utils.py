"""
Write a result table together with a record of how it was produced.

``write_results`` saves a DataFrame exactly as ``to_csv`` would and writes a small JSON file beside it
holding:

- the git commit and whether the producing script is byte-identical to its committed version
- the script path, its SHA-256 and the command-line arguments
- the SHA-256 and size of every declared input and of the result itself, so a later edit is detectable
- Python and library versions, CUDA/GPU and BLAS thread counts, which govern bitwise reproducibility

Paths are recorded relative to the repository, or as ``{config key}/...`` for directories configured
outside it, so no absolute location is stored.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIDECAR_SUFFIX = ".provenance.json"
SCHEMA_VERSION = 1
_CHUNK = 1 << 20


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------
def sha256_file(path: str | os.PathLike, chunk: int = _CHUNK) -> str:
    """SHA-256 hex digest of a file, streamed so large files use O(1) memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


EXTERNAL_ROOTS = ("piop2_root", "external_derivatives_dir")


def _external_roots() -> dict[str, Path]:
    """Configured directories outside the project, as {token: directory}."""
    roots = {}
    try:
        from src.data.paths import path as configured
        for key in EXTERNAL_ROOTS:
            roots[key] = Path(configured(key)).resolve()
    except Exception:                                   # no config on this machine: fall back to absolute
        pass
    roots["nilearn_data"] = (Path.home() / "nilearn_data").resolve()
    return roots


def _rel(path: str | os.PathLike) -> str:
    """Portable POSIX path: project-relative inside the project, {config key}/... outside it."""
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        pass
    for token, root in _external_roots().items():
        try:
            return "{" + token + "}/" + p.relative_to(root).as_posix()
        except ValueError:
            continue
    return p.as_posix()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------
def _git(*args: str, root: Path = PROJECT_ROOT) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state(script: str | os.PathLike | None = None,
              root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Commit, branch and cleanliness; optionally whether ``script`` matches HEAD.

    ``dirty`` counts tracked modifications only. Untracked files -- which is what
    a freshly written result is -- do not make the tree dirty.
    """
    inside = _git("rev-parse", "--is-inside-work-tree", root=root) == "true"
    if not inside:
        return {"available": False, "commit": None, "branch": None, "dirty": None,
                "script_matches_commit": None}
    commit = _git("rev-parse", "HEAD", root=root)          # None before first commit
    porcelain = _git("status", "--porcelain", "--untracked-files=no", root=root) or ""
    # Sidecars are metadata about results, not code or results, so writing one
    # must not mark the tree dirty for the next.
    porcelain = "\n".join(l for l in porcelain.splitlines()
                          if not l.rstrip().endswith(SIDECAR_SUFFIX))
    state: dict[str, Any] = {
        "available": True,
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", root=root) if commit else None,
        "dirty": bool(porcelain),
        "dirty_files": porcelain.splitlines()[:50],
        "script_matches_commit": None,
    }
    if script is not None and commit is not None:
        rel = _rel(script)
        tracked = _git("ls-files", "--error-unmatch", rel, root=root) is not None
        if tracked:
            diff = subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD",
                                   "--", rel], capture_output=True)
            state["script_matches_commit"] = diff.returncode == 0
        else:
            state["script_matches_commit"] = False
    return state


# ---------------------------------------------------------------------------
# runtime fingerprint
# ---------------------------------------------------------------------------
def environment_fingerprint() -> dict[str, Any]:
    """Library versions, GPU and BLAS threading -- what governs reproducibility."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "thread_env": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")},
    }
    versions = {}
    for mod in ("numpy", "scipy", "pandas", "sklearn", "torch", "h5py", "nilearn",
                "nibabel", "statsmodels", "neuromaps"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    env["versions"] = versions
    try:
        import torch
        env["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception:
        env["cuda"] = None
    try:
        from threadpoolctl import threadpool_info
        env["blas"] = [{k: d.get(k) for k in ("user_api", "internal_api", "version",
                                              "num_threads", "threading_layer")}
                       for d in threadpool_info()]
    except Exception:
        env["blas"] = None
    return env


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def input_hashes(inputs: Iterable[str | os.PathLike]) -> list[dict[str, Any]]:
    """Size and SHA-256 of each declared input, so a changed input is visible afterwards."""
    out = []
    for p in inputs:
        p = Path(p)
        rel = _rel(p)
        if not p.exists():
            out.append({"path": rel, "exists": False})
            continue
        out.append({"path": rel, "exists": True, "size": p.stat().st_size, "sha256": sha256_file(p)})
    return out


# ---------------------------------------------------------------------------
# sidecars
# ---------------------------------------------------------------------------
def sidecar_path(result_path: str | os.PathLike) -> Path:
    return Path(str(result_path) + SIDECAR_SUFFIX)


def write_sidecar(result_path: str | os.PathLike, *,
                  script: str | os.PathLike | None = None,
                  args: list[str] | None = None,
                  inputs: Iterable[str | os.PathLike] = (),
                  extra: Mapping[str, Any] | None = None) -> Path:
    """Write ``<result_path>.provenance.json`` for an existing result file."""
    result_path = Path(result_path)
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    script = Path(script) if script is not None else Path(sys.argv[0]).resolve()
    record = {
        "schema_version": SCHEMA_VERSION,
        "pre_git": False,
        "result": {"path": _rel(result_path), "size": result_path.stat().st_size,
                   "sha256": sha256_file(result_path)},
        "written_at": _now_iso(),
        "git": git_state(script),
        "script": {"path": _rel(script),
                   "sha256": sha256_file(script) if script.exists() else None,
                   "args": list(sys.argv[1:] if args is None else args)},
        "inputs": input_hashes(inputs),
        "environment": environment_fingerprint(),
        "extra": dict(extra or {}),
    }
    sc = sidecar_path(result_path)
    sc.write_text(json.dumps(record, indent=2, sort_keys=False), encoding="utf-8")
    return sc


def write_results(df, path: str | os.PathLike, *, index: bool = False,
                  script: str | os.PathLike | None = None,
                  args: list[str] | None = None,
                  inputs: Iterable[str | os.PathLike] = (),
                  **extra: Any) -> Path:
    """Write ``df`` to CSV exactly as ``df.to_csv(path, index=index)`` would, plus a sidecar.

    Extra keyword arguments (e.g. ``series="psc", repeats=5, n_subjects=877,
    seed=42``) are stored under ``extra`` in the sidecar.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    write_sidecar(path, script=script, args=args, inputs=inputs, extra=extra)
    return path


if __name__ == "__main__":                       # print the runtime fingerprint of this machine
    print(json.dumps(environment_fingerprint(), indent=2))
