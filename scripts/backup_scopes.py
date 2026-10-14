#!/usr/bin/env python
"""把 data/ 下所有 scope 与 registry 用 VACUUM INTO 做一致性快照到备份目录。

用法: .venv/bin/python scripts/backup_scopes.py <目标目录>
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
import time
from pathlib import Path

DATA = Path("data")
SKIP = ("-wal", "-shm")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        con.execute("VACUUM INTO ?", (str(dest),))
    finally:
        con.close()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    dest_root = Path(sys.argv[1]).resolve()
    files = [DATA / "registry.sqlite3"] + sorted((DATA / "scopes").glob("*.sqlite3"))
    lines: list[str] = []
    started = time.monotonic()
    total = 0
    for src in files:
        if any(str(src).endswith(suffix) for suffix in SKIP):
            continue
        rel = src.relative_to(DATA)
        dest = dest_root / "data" / rel
        snapshot(src, dest)
        size = dest.stat().st_size
        total += size
        lines.append(f"{sha256(dest)}  {size:>12d}  {rel}")
        print(f"ok  {rel}  {size / 1e6:.1f} MB", flush=True)
    (dest_root / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{len(lines)} 个文件，共 {total / 1e9:.2f} GB，用时 {time.monotonic() - started:.1f}s")
    print("manifest:", dest_root / "manifest.sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
