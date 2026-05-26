from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _episode_name_from_path(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("episode_"):
            return part
    return None


def _candidates(raw_frame: str, index_path: Path) -> Iterable[Path]:
    frame = Path(raw_frame)
    yield frame
    if not frame.is_absolute():
        yield Path.cwd() / frame
        yield index_path.parent / frame
    ep = _episode_name_from_path(frame)
    if ep:
        yield index_path.parent / ep / "frames" / frame.name
        yield Path.cwd() / "datasets" / "isaac_franka_many" / ep / "frames" / frame.name
    if index_path.parent.name.startswith("episode_"):
        yield index_path.parent / "frames" / frame.name


def resolve(raw_frame: str, index_path: Path) -> Path | None:
    for cand in _candidates(raw_frame, index_path):
        if cand.exists():
            return cand.resolve()
    return None


def rel_or_abs(path: Path, base: Path, absolute: bool) -> str:
    if absolute:
        return str(path.resolve())
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    p = argparse.ArgumentParser(description="Repair frame paths inside Isaac Franka index JSONL files.")
    p.add_argument("--input", required=True, help="Input index JSONL, e.g. datasets/isaac_franka_many/all_index.jsonl")
    p.add_argument("--output", default=None, help="Output path. Defaults to overwriting --input after writing a .bak backup.")
    p.add_argument("--absolute-paths", action="store_true", help="Write absolute paths instead of paths relative to output parent.")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output) if args.output else inp
    tmp = out.with_suffix(out.suffix + ".tmp")
    backup = inp.with_suffix(inp.suffix + ".bak")

    total = 0
    repaired = 0
    missing = 0
    rows = []
    with inp.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total += 1
            raw = json.loads(line)
            old_frame = str(raw["frame"])
            resolved = resolve(old_frame, inp)
            if resolved is None:
                missing += 1
                print(f"MISSING line {line_no}: {old_frame}")
            else:
                new_frame = rel_or_abs(resolved, out.parent, args.absolute_paths)
                if new_frame != old_frame:
                    repaired += 1
                raw["frame"] = new_frame
            rows.append(raw)

    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    if out == inp:
        if not backup.exists():
            inp.replace(backup)
        else:
            inp.unlink()
        tmp.replace(inp)
        print(f"Backup written to {backup}")
    else:
        tmp.replace(out)

    print(f"Checked {total} rows. Repaired {repaired}. Missing {missing}. Output: {out}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
