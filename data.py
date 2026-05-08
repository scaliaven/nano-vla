"""Dataset + collator for the flat .npz format produced by convert_libero.py.

The dataset returns per-sample uint8 frames + raw float actions; the
Collator handles instruction tokenization (LM tokenizer) and action
discretization (ActionTokenizer's vocab-tail mapping) at batch time, so
NanoVLA.forward() receives the model-shaped tensors it expects.

Splitting the work this way keeps both halves small and lets data.py stay
agnostic of model.py — anything with the same tokenizer / action_tokenizer
duck-typed interface plugs in.
"""
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
