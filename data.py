"""Datasets + collator. Two backends share the same per-sample interface:

1. TrajectoryDataset      reads the flat .npz layout from convert_libero.py /
                          convert_libero_lerobot.py. Fast: mmap'd npz, one
                          frame paged in per sample. Recommended for training.
2. LeRobotTrajectoryDataset reads a LeRobot v2.x dataset directly (parquet +
                          mp4) with no offline conversion. Per-frame random
                          access via a persistent PyAV container cached per
                          mp4 path — same idiom as LeRobot's own
                          VideoDecoderCache (LeRobot uses torchcodec; we use
                          PyAV to avoid a system-FFmpeg dep). LIBERO mp4s are
                          encoded with libsvtav1 g=2, so seek + 1-frame decode
                          is ~O(1).

Both expose `.action_q01`/`.action_q99` so train.py doesn't need stats.json
for the LeRobot path. `make_dataset(data_dir, ...)` auto-detects which
backend to use based on which metadata files are present, and accepts a
comma-separated string (or list) of paths to mix multiple datasets via
`MultiDataset` (per-dataset sampling weighted by total step count).

The Collator is unchanged across backends — it consumes the same per-sample
dict (primary, [wrist], instruction, actions).
"""
import itertools
import json
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# Per-worker LRU of open PyAV decoders. Each open container holds an FD plus
# a libsvtav1 decoder context (a few MB + a thread pool); unbounded growth
# starved libero_90 (~3.9k episodes × 2 cameras) and produced EAGAIN out of
# swscale.reformat once the allocator ran out. 128 keeps the working set hot
# for typical batch_size × num_workers draws.
_DECODER_CACHE_SIZE = 128
_VIDEO_DECODERS: "OrderedDict[str, tuple]" = OrderedDict()


def _decoder_for(path: Path):
    key = str(path)
    cached = _VIDEO_DECODERS.get(key)
    if cached is None:
        import av
        container = av.open(key)
        cached = (container, container.streams.video[0])
        _VIDEO_DECODERS[key] = cached
        if len(_VIDEO_DECODERS) > _DECODER_CACHE_SIZE:
            _, (old, _) = _VIDEO_DECODERS.popitem(last=False)
            old.close()
    else:
        _VIDEO_DECODERS.move_to_end(key)
    return cached


class TrajectoryDataset(Dataset):
    """Each item samples a random window from a random episode.

    Window = (image at t, action chunk a[t:t+K], instruction). Chunks that
    would extend past the episode end are right-padded by repeating the
    final action: standard practice that lets the model emit terminal
    actions cleanly at the chunk boundary instead of seeing a hard cutoff.

    Episode sampling is weighted by length so each transition has equal
    probability per epoch (otherwise short episodes get oversampled).
    """

    def __init__(self, data_dir, chunk_size: int = 8, use_wrist_camera: bool = False,
                 num_samples_per_epoch: int | None = None):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.use_wrist_camera = use_wrist_camera
        self.index = json.loads((self.data_dir / "index.json").read_text())
        self.episodes = list(self.index.keys())
        self.lengths = [self.index[k]["length"] for k in self.episodes]
        self.num_samples_per_epoch = num_samples_per_epoch or sum(self.lengths)
        stats = json.loads((self.data_dir / "stats.json").read_text())
        self.action_q01 = stats["action_q01"]
        self.action_q99 = stats["action_q99"]

    def __len__(self):
        return self.num_samples_per_epoch

    def __getitem__(self, idx):
        # idx is unused: we sample (episode, t) ourselves so batches are
        # uniformly spread over transitions, weighted by episode length.
        ep_name = random.choices(self.episodes, weights=self.lengths, k=1)[0]
        T = self.index[ep_name]["length"]
        # Restrict to t where the full chunk fits — avoids training on
        # post-completion frames paired with "last action × chunk_size" pads.
        t = random.randrange(max(1, T - self.chunk_size + 1))
        path = self.data_dir / self.index[ep_name]["file"]
        # mmap_mode='r' means only the bytes for one frame + the action chunk
        # are paged in — critical because each .npz is ~20MB uncompressed.
        with np.load(path, mmap_mode="r") as z:
            sample = {
                "primary": np.array(z["images_primary"][t]),
                "instruction": str(z["instruction"]),
                "actions": np.array(z["actions"][t : t + self.chunk_size],
                                    dtype=np.float32),
            }
            if self.use_wrist_camera:
                sample["wrist"] = np.array(z["images_wrist"][t])

        if len(sample["actions"]) < self.chunk_size:
            pad = np.repeat(sample["actions"][-1:],
                            self.chunk_size - len(sample["actions"]), axis=0)
            sample["actions"] = np.concatenate([sample["actions"], pad], axis=0)
        return sample


class Collator:
    """Per-batch tokenization + action discretization.

    Produces the dict that NanoVLA.forward() consumes:
        primary           (B, H, W, 3) uint8       — kept uint8; model normalizes on GPU
        wrist             same                     — only if use_wrist_camera
        instruction_ids   (B, T_text) long         — left-padded to max_instruction_tokens
        instruction_mask  (B, T_text) long
        action_token_ids  (B, K * action_dim) long — vocab-tail LM token ids
    """

    def __init__(self, tokenizer, action_tokenizer, prompt_template: str,
                 max_instruction_tokens: int, use_wrist_camera: bool = False):
        # Left-pad to a FIXED max_instruction_tokens so (a) action_query RoPE
        # positions are train/eval-invariant (no batch-dependent shift) and
        # (b) the last real instruction token stays adjacent to action_query[0].
        # padding_side="left" is set globally on the shared tokenizer in
        # NanoVLA.__init__.
        self.tokenizer = tokenizer
        self.action_tokenizer = action_tokenizer
        self.prompt_template = prompt_template
        self.max_instruction_tokens = max_instruction_tokens
        self.use_wrist_camera = use_wrist_camera

    def __call__(self, samples):
        prompts = [self.prompt_template.format(instruction=s["instruction"].strip())
                   for s in samples]
        tok = self.tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=self.max_instruction_tokens, return_tensors="pt",
        )
        actions = np.stack([s["actions"] for s in samples], axis=0)        # (B, K, A)
        action_token_ids = self.action_tokenizer.encode(actions)            # (B, K, A) np
        action_token_ids = torch.from_numpy(action_token_ids).reshape(len(samples), -1)
        primary = torch.from_numpy(np.stack([s["primary"] for s in samples], axis=0))

        batch = {
            "primary": primary,
            "instruction_ids": tok.input_ids,
            "instruction_mask": tok.attention_mask,
            "action_token_ids": action_token_ids,
        }
        if self.use_wrist_camera:
            batch["wrist"] = torch.from_numpy(
                np.stack([s["wrist"] for s in samples], axis=0))
        return batch


class LeRobotTrajectoryDataset(Dataset):
    """Read a LeRobot v2.x dataset directly — no offline conversion.

    Layout expected:
        <data_dir>/meta/info.json           # data_path/video_path templates, chunks_size
        <data_dir>/meta/episodes.jsonl      # per-episode length + tasks
        <data_dir>/data/chunk-XXX/episode_NNNNNN.parquet
        <data_dir>/videos/chunk-XXX/<key>/episode_NNNNNN.mp4

    Per-sample dict matches TrajectoryDataset exactly, so the Collator is shared.

    Performance note: random-frame fetch goes through a persistent PyAV
    container per mp4 path (module-level dict, fork-replicated per worker).
    With LIBERO's libsvtav1 g=2 encoding the keyframe-walk is ≤2 frames, so
    every getitem is ~O(1). Action arrays are tiny (~3MB total), so we just
    pre-load them all in __init__. Compared to the .npz path it's still
    slower (mp4 decode vs. mmap'd uint8), but the gap is bounded; for a
    non-LIBERO LeRobot dataset with sparser keyframes, prefer the converter.

    Resize-only (no [::-1]): LeRobot LIBERO mp4s are already upright, so the
    pixels here match what convert_libero.resize_views produces from HDF5.
    """

    PRIMARY_KEY = "observation.images.image"
    WRIST_KEY = "observation.images.wrist_image"

    def __init__(self, data_dir, chunk_size: int = 8, use_wrist_camera: bool = False,
                 image_size: int = 224, num_samples_per_epoch: int | None = None):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.use_wrist_camera = use_wrist_camera
        self.image_size = image_size
        self.info = json.loads((self.data_dir / "meta" / "info.json").read_text())

        self.episode_meta = []  # list of {ep_idx, length, instruction}
        with (self.data_dir / "meta" / "episodes.jsonl").open() as f:
            for line in f:
                em = json.loads(line)
                self.episode_meta.append({
                    "ep_idx": em["episode_index"],
                    "length": em["length"],
                    "instruction": em["tasks"][0],
                })
        self.lengths = [em["length"] for em in self.episode_meta]
        self.num_samples_per_epoch = num_samples_per_epoch or sum(self.lengths)
        # Actions are tiny (~3MB total); preload every episode and compute
        # quantiles in one pass — no LRU needed.
        self._actions = {em["ep_idx"]: self._read_actions(em["ep_idx"])
                         for em in self.episode_meta}
        all_actions = np.concatenate(list(self._actions.values()), axis=0)
        self.action_q01 = np.quantile(all_actions, 0.01, axis=0).tolist()
        self.action_q99 = np.quantile(all_actions, 0.99, axis=0).tolist()

    def __len__(self):
        return self.num_samples_per_epoch

    def _ep_paths(self, ep_idx: int):
        chunk = ep_idx // self.info["chunks_size"]
        fmt = {"episode_chunk": chunk, "episode_index": ep_idx}
        pq = self.data_dir / self.info["data_path"].format(**fmt)
        prim = self.data_dir / self.info["video_path"].format(video_key=self.PRIMARY_KEY, **fmt)
        wrist = self.data_dir / self.info["video_path"].format(video_key=self.WRIST_KEY, **fmt)
        return pq, prim, wrist

    def _read_actions(self, ep_idx: int) -> np.ndarray:
        import pyarrow.parquet as pq_mod
        pq_path, _, _ = self._ep_paths(ep_idx)
        col = pq_mod.read_table(pq_path, columns=["action"])["action"]
        return np.stack(list(col.to_numpy())).astype(np.float32)

    def _read_frame(self, mp4_path: Path, t: int) -> np.ndarray:
        # Seek to nearest keyframe ≤ t, then decode forward until we hit pts ≥ t.
        # LIBERO mp4s use libsvtav1 g=2, so the keyframe-walk is ≤2 frames.
        container, stream = _decoder_for(mp4_path)
        target_pts = int(t / stream.average_rate / stream.time_base)
        container.seek(target_pts, stream=stream)
        for frame in container.decode(stream):
            if frame.pts is not None and frame.pts >= target_pts:
                # .copy(): PyAV's to_ndarray buffer is owned by the frame and freed
                # when the next decode iteration starts — must own the bytes here.
                arr = frame.to_ndarray(format="rgb24").copy()
                break
        else:
            raise IndexError(f"frame {t} not found in {mp4_path}")
        if arr.shape[:2] != (self.image_size, self.image_size):
            arr = np.asarray(Image.fromarray(arr).resize(
                (self.image_size, self.image_size), Image.BILINEAR))
        return arr

    def __getitem__(self, idx):
        # Ignore idx — sample (episode, t) ourselves, length-weighted.
        em = random.choices(self.episode_meta, weights=self.lengths, k=1)[0]
        ep_idx, T, instr = em["ep_idx"], em["length"], em["instruction"]
        # See TrajectoryDataset: keep chunks fully in-episode.
        t = random.randrange(max(1, T - self.chunk_size + 1))

        actions = self._actions[ep_idx][t : t + self.chunk_size].copy()
        _, prim_path, wrist_path = self._ep_paths(ep_idx)
        sample = {"primary": self._read_frame(prim_path, t),
                  "instruction": instr, "actions": actions}
        if self.use_wrist_camera:
            sample["wrist"] = self._read_frame(wrist_path, t)

        if len(actions) < self.chunk_size:
            pad = np.repeat(actions[-1:], self.chunk_size - len(actions), axis=0)
            sample["actions"] = np.concatenate([actions, pad], axis=0)
        return sample


class MultiDataset(Dataset):
    """Length-weighted union of N constituent datasets.

    Each constituent already does its own length-weighted (episode, t) sampling
    and ignores `idx`; we add a per-dataset weighting on top so batches are
    drawn in proportion to total step count across the whole pool. Action
    quantiles are pooled by min(q01) / max(q99) — slightly wider than the true
    union quantile, but no constituent's actions get clipped.
    """

    def __init__(self, datasets):
        self.datasets = list(datasets)
        # Precomputed so random.choices doesn't re-accumulate weights per call.
        self._cum_weights = list(itertools.accumulate(sum(d.lengths) for d in self.datasets))
        q01 = np.minimum.reduce([np.asarray(d.action_q01) for d in self.datasets])
        q99 = np.maximum.reduce([np.asarray(d.action_q99) for d in self.datasets])
        self.action_q01 = q01.tolist()
        self.action_q99 = q99.tolist()

    def __len__(self):
        return self._cum_weights[-1]

    def __getitem__(self, idx):
        d = random.choices(self.datasets, cum_weights=self._cum_weights, k=1)[0]
        return d[0]


def make_dataset(data_dir, chunk_size: int = 8, use_wrist_camera: bool = False,
                 image_size: int = 224) -> Dataset:
    """Pick the right dataset class based on what's in `data_dir`.

    LeRobot layout has meta/info.json; the flat-npz layout has stats.json + index.json.
    `data_dir` may be a single path, a comma-separated string of paths, or a
    list/tuple — multiple paths are wrapped in a MultiDataset.
    """
    if isinstance(data_dir, (list, tuple)):
        paths = [str(p) for p in data_dir]
    else:
        paths = [p.strip() for p in str(data_dir).split(",") if p.strip()]

    def _one(p):
        p = Path(p)
        if (p / "meta" / "info.json").exists():
            return LeRobotTrajectoryDataset(p, chunk_size=chunk_size,
                                            use_wrist_camera=use_wrist_camera,
                                            image_size=image_size)
        return TrajectoryDataset(p, chunk_size=chunk_size,
                                 use_wrist_camera=use_wrist_camera)

    if len(paths) == 1:
        return _one(paths[0])
    return MultiDataset([_one(p) for p in paths])


def make_dataloader(dataset, collator, batch_size: int, num_workers: int = 4,
                    worker_init_fn=None) -> DataLoader:
    # shuffle=False: TrajectoryDataset samples (episode, t) internally via
    # weighted random.choices, so external shuffling adds nothing. Use
    # worker_init_fn to seed Python's `random` per-worker per-rank.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
