"""Datasets + collator. Two backends share the same per-sample interface:

1. TrajectoryDataset      reads the flat .npz layout from convert_libero.py /
                          convert_libero_lerobot.py. Fast: mmap'd npz, one
                          frame paged in per sample. Recommended for training.
2. LeRobotTrajectoryDataset reads a LeRobot v2.x dataset directly (parquet +
                          mp4) with no offline conversion step. Convenient but
                          slower: every fresh episode triggers a sequential mp4
                          decode (AV1 has sparse keyframes; per-frame seek is
                          expensive). A small per-worker LRU softens the cost.

Both expose `.action_q01`/`.action_q99` so train.py doesn't need stats.json
for the LeRobot path. `make_dataset(data_dir, ...)` auto-detects which
backend to use based on which metadata files are present.

The Collator is unchanged across backends — it consumes the same per-sample
dict (primary, [wrist], instruction, actions).
"""
import functools
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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
        t = random.randrange(T)
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
        instruction_ids   (B, T_text) long         — padded to longest in batch
        instruction_mask  (B, T_text) long
        action_token_ids  (B, K * action_dim) long — vocab-tail LM token ids
    """

    def __init__(self, tokenizer, action_tokenizer, prompt_template: str,
                 max_instruction_tokens: int, use_wrist_camera: bool = False):
        self.tokenizer = tokenizer
        self.action_tokenizer = action_tokenizer
        self.prompt_template = prompt_template
        self.max_instruction_tokens = max_instruction_tokens
        self.use_wrist_camera = use_wrist_camera

    def __call__(self, samples):
        prompts = [self.prompt_template.format(instruction=s["instruction"].strip())
                   for s in samples]
        tok = self.tokenizer(
            prompts, padding="longest", truncation=True,
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

    Performance note: AV1-coded mp4s don't seek cheaply, so we decode whole
    episodes sequentially and cache them per-worker via lru_cache. With
    length-weighted episode sampling and N≈400 episodes, the cache rarely hits;
    if you train for real, run convert_libero_lerobot.py once and use the .npz
    path instead — it's ~10× faster.

    Resize-only (no [::-1]): LeRobot LIBERO mp4s are already upright, so the
    pixels here match what convert_libero.resize_views produces from HDF5.
    """

    PRIMARY_KEY = "observation.images.image"
    WRIST_KEY = "observation.images.wrist_image"

    def __init__(self, data_dir, chunk_size: int = 8, use_wrist_camera: bool = False,
                 image_size: int = 224, num_samples_per_epoch: int | None = None,
                 cache_episodes: int = 4):
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
        self.action_q01, self.action_q99 = self._compute_action_quantiles()

        # Per-instance lru caches — each DataLoader worker forks its own copy,
        # so caches don't share across workers (which is what we want).
        self._decode_view = functools.lru_cache(maxsize=cache_episodes)(self._decode_view_uncached)
        self._read_actions = functools.lru_cache(maxsize=cache_episodes)(self._read_actions_uncached)

    def __len__(self):
        return self.num_samples_per_epoch

    def _ep_paths(self, ep_idx: int):
        chunk = ep_idx // self.info["chunks_size"]
        fmt = {"episode_chunk": chunk, "episode_index": ep_idx}
        pq = self.data_dir / self.info["data_path"].format(**fmt)
        prim = self.data_dir / self.info["video_path"].format(video_key=self.PRIMARY_KEY, **fmt)
        wrist = self.data_dir / self.info["video_path"].format(video_key=self.WRIST_KEY, **fmt)
        return pq, prim, wrist

    def _read_actions_uncached(self, ep_idx: int) -> np.ndarray:
        import pyarrow.parquet as pq_mod
        pq_path, _, _ = self._ep_paths(ep_idx)
        col = pq_mod.read_table(pq_path, columns=["action"])["action"]
        return np.stack(list(col.to_numpy())).astype(np.float32)

    def _decode_view_uncached(self, ep_idx: int, view: str) -> np.ndarray:
        import imageio.v3 as iio
        from PIL import Image as PILImage
        _, prim, wrist = self._ep_paths(ep_idx)
        mp4 = prim if view == "primary" else wrist
        size = self.image_size
        frames = []
        for f in iio.imiter(mp4):
            if f.shape[:2] != (size, size):
                f = np.asarray(PILImage.fromarray(f).resize((size, size), PILImage.BILINEAR))
            frames.append(f)
        return np.stack(frames).astype(np.uint8)

    def _compute_action_quantiles(self):
        """One-time scan over all parquets. Cheap: only reads the action column."""
        A = np.concatenate([self._read_actions_uncached(em["ep_idx"])
                            for em in self.episode_meta], axis=0)
        return np.quantile(A, 0.01, axis=0).tolist(), np.quantile(A, 0.99, axis=0).tolist()

    def __getitem__(self, idx):
        # Ignore idx — sample (episode, t) ourselves, length-weighted.
        em = random.choices(self.episode_meta, weights=self.lengths, k=1)[0]
        ep_idx, T, instr = em["ep_idx"], em["length"], em["instruction"]
        t = random.randrange(T)

        actions = self._read_actions(ep_idx)[t : t + self.chunk_size].copy()
        primary = self._decode_view(ep_idx, "primary")[t].copy()
        sample = {"primary": primary, "instruction": instr, "actions": actions}
        if self.use_wrist_camera:
            sample["wrist"] = self._decode_view(ep_idx, "wrist")[t].copy()

        if len(actions) < self.chunk_size:
            pad = np.repeat(actions[-1:], self.chunk_size - len(actions), axis=0)
            sample["actions"] = np.concatenate([actions, pad], axis=0)
        return sample


def make_dataset(data_dir, chunk_size: int = 8, use_wrist_camera: bool = False,
                 image_size: int = 224) -> Dataset:
    """Pick the right dataset class based on what's in `data_dir`.

    LeRobot layout has meta/info.json; the flat-npz layout has stats.json + index.json.
    """
    p = Path(data_dir)
    if (p / "meta" / "info.json").exists():
        return LeRobotTrajectoryDataset(p, chunk_size=chunk_size,
                                        use_wrist_camera=use_wrist_camera,
                                        image_size=image_size)
    return TrajectoryDataset(p, chunk_size=chunk_size, use_wrist_camera=use_wrist_camera)


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
