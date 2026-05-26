from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import imageio.v2 as imageio
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from .isaac_franka_dataset import IsaacFrankaRecord, _load_records


def _episode_key_from_record(record: IsaacFrankaRecord, fallback: str) -> str:
    for part in record.frame.parts:
        if part.startswith("episode_"):
            return part
    return fallback


def _split_records_by_episode(records: List[IsaacFrankaRecord], fallback: str) -> List[List[IsaacFrankaRecord]]:
    """Split a merged all_index.jsonl back into episodes using frame paths.

    Older project versions loaded one all_index.jsonl as a single long episode.
    That allowed windows to cross episode boundaries, creating impossible target
    jumps. This helper preserves record order while separating episode_XXXX
    folders into independent trajectories.
    """
    groups: Dict[str, List[IsaacFrankaRecord]] = {}
    order: List[str] = []
    for rec in records:
        key = _episode_key_from_record(rec, fallback)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(rec)
    return [groups[k] for k in order]


def _resolve_index_paths(index_paths: str | Path | Sequence[str | Path]) -> List[Path]:
    if isinstance(index_paths, (str, Path)):
        text = str(index_paths)
        if ";" in text:
            return [Path(p.strip()) for p in text.split(";") if p.strip()]
        return [Path(text)]
    return [Path(p) for p in index_paths]


class IsaacFrankaSequenceDataset(Dataset):
    """Temporal Isaac Franka dataset with image context and multi-step targets.

    Returns context frames/states from t-k+1..t and future targets t+1..t+H.

    v6 adds two important production features:
    - episode-level train/validation split, so validation episodes are not seen
      during training;
    - discontinuity filtering based on step IDs, end-effector jumps, and joint
      jumps. This prevents windows from crossing scripted teleports/resets inside
      an episode.
    """

    def __init__(
        self,
        index_paths: str | Path | Sequence[str | Path],
        image_size: int = 128,
        context_len: int = 4,
        horizon: int = 8,
        stride: int = 1,
        split: str = "all",
        val_fraction: float = 0.2,
        split_mode: str = "temporal",
        max_windows: int | None = None,
        cache_images: bool = False,
        require_contiguous_steps: bool = False,
        expected_step_delta: int | None = None,
        max_ee_step: float | None = None,
        max_q_step: float | None = None,
        image_preprocess: str = "none",
        augment: bool = False,
        brightness_jitter: float = 0.0,
        contrast_jitter: float = 0.0,
        noise_std: float = 0.0,
        image_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if split not in {"all", "train", "val"}:
            raise ValueError("split must be one of: all, train, val")
        if split_mode not in {"temporal", "random", "episode"}:
            raise ValueError("split_mode must be one of: temporal, random, episode")
        self.index_paths = _resolve_index_paths(index_paths)
        self.image_size = int(image_size)
        self.context_len = int(context_len)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.cache_images = bool(cache_images)
        self.require_contiguous_steps = bool(require_contiguous_steps)
        self.expected_step_delta = None if expected_step_delta is None else int(expected_step_delta)
        self.max_ee_step = None if max_ee_step is None else float(max_ee_step)
        self.max_q_step = None if max_q_step is None else float(max_q_step)
        self.image_preprocess = str(image_preprocess)
        self.augment = bool(augment)
        self.brightness_jitter = float(brightness_jitter)
        self.contrast_jitter = float(contrast_jitter)
        self.noise_std = float(noise_std)
        self.image_dropout_prob = float(image_dropout_prob)
        if self.image_preprocess not in {"none", "percentile", "standardize"}:
            raise ValueError("image_preprocess must be one of: none, percentile, standardize")
        if self.context_len < 1:
            raise ValueError("context_len must be >= 1")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")

        self.episodes: List[List[IsaacFrankaRecord]] = []
        for p in self.index_paths:
            records = _load_records(p)
            self.episodes.extend(_split_records_by_episode(records, fallback=Path(p).stem))
        self.dof = len(self.episodes[0][0].joint_positions)
        for ep in self.episodes:
            if any(len(r.joint_positions) != self.dof for r in ep):
                raise ValueError("All records across all episodes must have the same joint dimension")

        windows: List[Tuple[int, int]] = []
        skipped_step = 0
        skipped_ee = 0
        skipped_q = 0
        needed = self.context_len + self.horizon * self.stride
        for ep_idx, ep in enumerate(self.episodes):
            max_start = len(ep) - needed + 1
            for start in range(max(0, max_start)):
                ok, reason = self._window_is_valid(ep, start)
                if not ok:
                    if reason == "step":
                        skipped_step += 1
                    elif reason == "ee":
                        skipped_ee += 1
                    elif reason == "q":
                        skipped_q += 1
                    continue
                windows.append((ep_idx, start))
        if not windows:
            raise ValueError(
                f"No valid windows. Need at least context_len + horizon*stride = {needed} records per episode "
                "after continuity filtering. Try relaxing max_ee_step/max_q_step or disabling require_contiguous_steps."
            )
        if max_windows is not None:
            windows = windows[: int(max_windows)]

        if split != "all":
            if split_mode == "episode":
                episode_ids = sorted({w[0] for w in windows})
                val_count = max(1, int(round(len(episode_ids) * float(val_fraction)))) if len(episode_ids) > 1 else 1
                # Deterministic validation episodes without global RNG.
                keyed_eps = sorted(episode_ids, key=lambda x: (x * 1000003 + 9176) % 104729)
                val_eps = set(keyed_eps[:val_count])
                windows = [w for w in windows if (w[0] in val_eps) == (split == "val")]
            elif split_mode == "temporal":
                train_windows: List[Tuple[int, int]] = []
                val_windows: List[Tuple[int, int]] = []
                for ep_idx in range(len(self.episodes)):
                    ep_windows = [w for w in windows if w[0] == ep_idx]
                    if not ep_windows:
                        continue
                    val_count = max(1, int(round(len(ep_windows) * float(val_fraction))))
                    cut = max(1, len(ep_windows) - val_count)
                    train_windows.extend(ep_windows[:cut])
                    val_windows.extend(ep_windows[cut:])
                windows = train_windows if split == "train" else val_windows
            else:
                keyed = sorted(windows, key=lambda x: (x[0] * 1000003 + x[1] * 9176) % 104729)
                val_count = max(1, int(round(len(keyed) * float(val_fraction))))
                val_set = set(keyed[:val_count])
                windows = [w for w in windows if (w in val_set) == (split == "val")]

        self.windows = windows
        self.skipped_windows = {"step": skipped_step, "ee": skipped_ee, "q": skipped_q}
        self._image_cache: Dict[str, Tensor] = {}

    def _all_indices(self, start: int) -> List[int]:
        context = list(range(start, start + self.context_len))
        future = [start + self.context_len - 1 + self.stride * i for i in range(1, self.horizon + 1)]
        return context + future

    @staticmethod
    def _as_vec(x: Sequence[float]) -> Tensor:
        return torch.tensor(list(x), dtype=torch.float32)

    def _window_is_valid(self, ep: Sequence[IsaacFrankaRecord], start: int) -> tuple[bool, str | None]:
        indices = self._all_indices(start)
        if self.require_contiguous_steps:
            steps = [ep[i].step for i in indices]
            if not self._steps_are_contiguous(steps):
                return False, "step"
        if self.max_ee_step is not None:
            ees = [self._as_vec(ep[i].end_effector_position) for i in indices]
            max_jump = max(float(torch.linalg.norm(ees[i + 1] - ees[i])) for i in range(len(ees) - 1))
            if max_jump > self.max_ee_step:
                return False, "ee"
        if self.max_q_step is not None:
            qs = [self._as_vec(ep[i].joint_positions) for i in indices]
            max_jump = max(float(torch.linalg.norm(qs[i + 1] - qs[i])) for i in range(len(qs) - 1))
            if max_jump > self.max_q_step:
                return False, "q"
        return True, None

    def _steps_are_contiguous(self, steps: Sequence[int | None]) -> bool:
        if any(s is None for s in steps):
            return True
        diffs = [int(steps[i + 1]) - int(steps[i]) for i in range(len(steps) - 1)]
        if self.expected_step_delta is not None:
            return all(d == self.expected_step_delta for d in diffs)
        return len(set(diffs)) == 1 and diffs[0] > 0

    def __len__(self) -> int:
        return len(self.windows)

    def _preprocess_frame(self, x: Tensor) -> Tensor:
        if self.image_preprocess == "percentile":
            flat = x.flatten()
            lo = torch.quantile(flat, 0.005)
            hi = torch.quantile(flat, 0.995)
            x = (x - lo) / (hi - lo).clamp_min(1e-4)
        elif self.image_preprocess == "standardize":
            mean = x.mean(dim=(-2, -1), keepdim=True)
            std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
            x = (x - mean) / std
            x = x * 0.20 + 0.50
        return x.clamp(0.0, 1.0)

    def _augment_frames(self, frames: Tensor) -> Tensor:
        if not self.augment:
            return frames
        if self.image_dropout_prob > 0 and torch.rand(()) < self.image_dropout_prob:
            return torch.zeros_like(frames)
        if self.contrast_jitter > 0:
            factor = 1.0 + (torch.rand(()) * 2.0 - 1.0) * self.contrast_jitter
            mean = frames.mean(dim=(-3, -2, -1), keepdim=True)
            frames = (frames - mean) * factor + mean
        if self.brightness_jitter > 0:
            shift = (torch.rand(()) * 2.0 - 1.0) * self.brightness_jitter
            frames = frames + shift
        if self.noise_std > 0:
            frames = frames + torch.randn_like(frames) * self.noise_std
        return frames.clamp(0.0, 1.0)

    def _load_frame(self, path: Path) -> Tensor:
        key = f"{path}|{self.image_preprocess}|{self.image_size}"
        if self.cache_images and key in self._image_cache:
            return self._image_cache[key].clone()
        arr = imageio.imread(path)
        if arr.ndim == 2:
            arr = arr[..., None].repeat(3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        x = torch.as_tensor(arr, dtype=torch.float32).permute(2, 0, 1) / 255.0
        x = self._preprocess_frame(x)
        if self.image_size > 0 and (x.shape[-2] != self.image_size or x.shape[-1] != self.image_size):
            x = F.interpolate(x[None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
        x = x.clamp(0.0, 1.0)
        if self.cache_images:
            self._image_cache[key] = x.clone()
        return x

    @staticmethod
    def _tensor_list(values: Iterable[Sequence[float]]) -> Tensor:
        return torch.tensor([list(v) for v in values], dtype=torch.float32)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        ep_idx, start = self.windows[index]
        ep = self.episodes[ep_idx]
        context_indices = list(range(start, start + self.context_len))
        future_indices = [start + self.context_len - 1 + self.stride * i for i in range(1, self.horizon + 1)]
        context = [ep[i] for i in context_indices]
        future = [ep[i] for i in future_indices]

        frames = torch.stack([self._load_frame(r.frame) for r in context], dim=0)
        frames = self._augment_frames(frames)
        q_context = self._tensor_list(r.joint_positions for r in context)
        ee_context = self._tensor_list(r.end_effector_position for r in context)
        cube_context = self._tensor_list(r.cube_position for r in context)
        q_future = self._tensor_list(r.joint_positions for r in future)
        ee_future = self._tensor_list(r.end_effector_position for r in future)
        cube_future = self._tensor_list(r.cube_position for r in future)

        prev_q = torch.cat([q_context[-1:].clone(), q_future[:-1]], dim=0)
        delta_q_seq = q_future - prev_q
        steps = torch.tensor([(-1 if r.step is None else int(r.step)) for r in context + future], dtype=torch.long)

        return {
            "frames": frames,
            "q_context": q_context,
            "ee_context": ee_context,
            "cube_context": cube_context,
            "target_position": torch.tensor(context[-1].target_position, dtype=torch.float32),
            "future_joint_positions": q_future,
            "future_delta_q": delta_q_seq,
            "future_ee_positions": ee_future,
            "future_cube_positions": cube_future,
            "q0": q_context[-1].clone(),
            "ee0": ee_context[-1].clone(),
            "cube0": cube_context[-1].clone(),
            "episode_index": torch.tensor(ep_idx, dtype=torch.long),
            "window_start": torch.tensor(start, dtype=torch.long),
            "steps": steps,
        }


def write_multi_index(index_paths: Sequence[str | Path], output_path: str | Path) -> None:
    """Merge multiple JSONL indices into one file while preserving paths."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as dst:
        for p in index_paths:
            path = Path(p)
            with path.open("r", encoding="utf-8") as src:
                for line in src:
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    frame = Path(str(raw["frame"]))
                    if not frame.is_absolute():
                        raw["frame"] = str((path.parent / frame).resolve())
                    dst.write(json.dumps(raw) + "\n")
