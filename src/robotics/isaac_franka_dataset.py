from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import imageio.v2 as imageio
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class IsaacFrankaRecord:
    frame: Path
    joint_positions: List[float]
    cube_position: List[float]
    end_effector_position: List[float]
    target_position: List[float]
    step: int | None = None


def _as_float_list(value: Any, name: str) -> List[float]:
    if not isinstance(value, Sequence):
        raise ValueError(f"Field {name!r} must be a sequence, got {type(value).__name__}")
    return [float(x) for x in value]


def _frame_candidates(frame: Path, index_path: Path) -> List[Path]:
    """Return robust candidate paths for Isaac frame records.

    This intentionally handles old merged all_index.jsonl files that accidentally
    duplicated the dataset prefix, for example:

        .../episode_0000/datasets/isaac_franka_many/episode_0000/frames/rgb.png

    The correct path can usually be reconstructed from the merged index parent,
    the episode_XXXX folder name, and the image filename.
    """
    candidates: List[Path] = [frame]

    if not frame.is_absolute():
        candidates.extend([
            Path.cwd() / frame,
            index_path.parent / frame,
            index_path.parent / "frames" / frame.name,
        ])

    episode_name = None
    for part in frame.parts:
        if part.startswith("episode_"):
            episode_name = part
    if episode_name is not None:
        candidates.extend([
            index_path.parent / episode_name / "frames" / frame.name,
            Path.cwd() / "datasets" / "isaac_franka_many" / episode_name / "frames" / frame.name,
        ])

    # If the index lives inside one episode folder, this is the common relative
    # form written by the single-episode converter.
    if index_path.parent.name.startswith("episode_"):
        candidates.append(index_path.parent / "frames" / frame.name)

    # Last-resort duplicate-prefix repair. If a path contains two episode_XXXX
    # segments, keep the first episode segment and the final frame filename.
    episode_parts = [i for i, part in enumerate(frame.parts) if part.startswith("episode_")]
    if episode_parts:
        first_episode = frame.parts[episode_parts[0]]
        candidates.append(index_path.parent / first_episode / "frames" / frame.name)

    return candidates


def _resolve_frame_path(frame: Path, index_path: Path) -> Path:
    seen = set()
    for candidate in _frame_candidates(frame, index_path):
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()
    return frame


def _load_records(index_path: Path) -> List[IsaacFrankaRecord]:
    records: List[IsaacFrankaRecord] = []
    index_path = Path(index_path)
    with index_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            frame = _resolve_frame_path(Path(str(raw["frame"])), index_path)
            rec = IsaacFrankaRecord(
                frame=frame,
                joint_positions=_as_float_list(raw["joint_positions"], "joint_positions"),
                cube_position=_as_float_list(raw["cube_position"], "cube_position"),
                end_effector_position=_as_float_list(raw["end_effector_position"], "end_effector_position"),
                target_position=_as_float_list(raw["target_position"], "target_position"),
                step=int(raw["step"]) if "step" in raw and raw["step"] is not None else None,
            )
            if not rec.frame.exists():
                raise FileNotFoundError(
                    f"Frame path from {index_path}:{line_number} does not exist: {rec.frame}\n"
                    "Try regenerating the merged index with:\n"
                    "  python scripts\\isaac_sim\\merge_episode_indices.py "
                    "--root-dir datasets\\isaac_franka_many "
                    "--output datasets\\isaac_franka_many\\all_index.jsonl"
                )
            records.append(rec)
    if len(records) < 2:
        raise ValueError(f"Need at least 2 records in {index_path}; found {len(records)}")
    return records


class IsaacFrankaEpisodeDataset(Dataset):
    """Isaac Sim RGB + state dataset for a single Franka episode.

    Each item is a consecutive pair: frame_t, q_t, q_{t+stride}, ee_t,
    ee_{t+stride}. This is intentionally separate from Isaac Sim so training can
    happen in a normal PyTorch environment.
    """

    def __init__(
        self,
        index_path: str | Path,
        image_size: int = 128,
        stride: int = 1,
        split: str = "all",
        val_fraction: float = 0.2,
        normalize_delta: bool = False,
        max_pairs: int | None = None,
    ) -> None:
        super().__init__()
        self.index_path = Path(index_path)
        self.image_size = int(image_size)
        self.stride = int(stride)
        self.normalize_delta = bool(normalize_delta)
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if split not in {"all", "train", "val"}:
            raise ValueError("split must be one of: all, train, val")

        records = _load_records(self.index_path)
        pairs = [(i, i + self.stride) for i in range(0, len(records) - self.stride)]
        if max_pairs is not None:
            pairs = pairs[: int(max_pairs)]

        if split != "all":
            val_count = max(1, int(round(len(pairs) * float(val_fraction))))
            split_index = max(1, len(pairs) - val_count)
            pairs = pairs[:split_index] if split == "train" else pairs[split_index:]

        self.records = records
        self.pairs = pairs
        self.dof = len(records[0].joint_positions)
        if self.dof == 0:
            raise ValueError("joint_positions cannot be empty")
        for rec in records:
            if len(rec.joint_positions) != self.dof:
                raise ValueError("All records must have the same joint dimension")

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_frame(self, path: Path) -> Tensor:
        arr = imageio.imread(path)
        if arr.ndim == 2:
            arr = arr[..., None].repeat(3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        x = torch.as_tensor(arr, dtype=torch.float32).permute(2, 0, 1) / 255.0
        if self.image_size > 0 and (x.shape[-2] != self.image_size or x.shape[-1] != self.image_size):
            x = F.interpolate(x[None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
        return x.clamp(0.0, 1.0)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        i, j = self.pairs[index]
        a = self.records[i]
        b = self.records[j]
        q = torch.tensor(a.joint_positions, dtype=torch.float32)
        q_next = torch.tensor(b.joint_positions, dtype=torch.float32)
        delta = q_next - q
        if self.normalize_delta:
            delta = delta / float(self.stride)
        return {
            "frame": self._load_frame(a.frame),
            "joint_positions": q,
            "next_joint_positions": q_next,
            "delta_q": delta,
            "cube_position": torch.tensor(a.cube_position, dtype=torch.float32),
            "next_cube_position": torch.tensor(b.cube_position, dtype=torch.float32),
            "end_effector_position": torch.tensor(a.end_effector_position, dtype=torch.float32),
            "next_end_effector_position": torch.tensor(b.end_effector_position, dtype=torch.float32),
            "target_position": torch.tensor(a.target_position, dtype=torch.float32),
            "step": torch.tensor(-1 if a.step is None else a.step, dtype=torch.long),
        }
