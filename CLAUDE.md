# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

nanoVLA is a **teaching artifact** — a single-file-readable Vision-Language-Action model in the spirit of nanoGPT. The product is *clarity*, not SOTA. The whole project is constrained to ~600 functional lines; features that push past that go in the v2 roadmap (`README.md#v2-roadmap`), not into the codebase. When proposing changes, weigh line-count and readability cost against pedagogical value. Don't introduce abstractions, frameworks (Trainer/Lightning/Accelerate/Hydra), or generality that the task doesn't already require.

## Commands

```bash
# 1. Convert LIBERO HDF5 demos -> flat .npz (one-time per dataset)
python convert_libero.py --src /path/to/libero_spatial_no_noops --out data/libero_spatial
# fast smoke test:
python convert_libero.py --src ... --out ... --max-demos-per-task 1

# 2. Smoke-test the eval harness without a trained model
python eval_libero.py --policy random --suite libero_spatial --num-trials 2

# 3. Train (single GPU)
python train.py --data-dir data/libero_spatial --steps 50000 --batch-size 8
# DDP via torchrun (nothing else to configure):
torchrun --nproc-per-node=4 train.py --data-dir data/libero_spatial --batch-size 8
# tiny smoke run:
python train.py --steps 100 --log-every 10

# 4. Evaluate a checkpoint
python eval_libero.py --policy nano-vla --ckpt out/ckpt_last.pt --suite libero_spatial
```

Every field of `TrainConfig` (top of `train.py`) is auto-exposed as a CLI flag (hyphenated). Booleans become `--foo`/`--no-foo` via `argparse.BooleanOptionalAction`. There is no separate config file.

There is no test suite, linter, or build step.

## Architecture

The pipeline is five files connected by two contracts. Understand the contracts and the rest follows.

**Contract 1 — flat `.npz` data format** (produced by `convert_libero.py`, consumed by `data.py`):
```
data/<dataset>/
  episode_NNNNN.npz   # images_primary, images_wrist (uint8), actions (T,7) float32, instruction (str)
  index.json          # {ep_name: {length, instruction, file, source_*}}
  stats.json          # action_q01, action_q99, num_bins, image_size
```
Files are written **uncompressed** (`np.savez`) so the dataloader can `mmap_mode='r'` and page in only one frame per `__getitem__`. Compressed npz would force whole-episode decompression per sample (~10× slower training). To support a non-LIBERO dataset, write a new conversion script that emits this exact layout — nothing in `model.py`/`data.py`/`train.py` is LIBERO-specific.

**Contract 2 — policy interface** (between model and eval harness):
```python
policy.chunk_size: int
policy.predict(images: dict, instruction: str) -> np.ndarray  # shape (chunk_size, 7), float32
```
`images` has `"primary"` (and optionally `"wrist"`), each `(image_size, image_size, 3)` uint8. `eval_libero.py` calls `predict()`, executes the chunk **open-loop**, then calls again. `RandomPolicy` and `NanoVLA` both conform.

### Two non-obvious moves in `model.py` (load-bearing — keep in mind when editing)

1. **Vocab-tail action tokens.** Actions are per-dim discretized into 256 bins, and bin `b` for any action dim is encoded as LM token id `vocab_size - 256 + b`. The last 256 tokens of Qwen2.5's tokenizer are byte-fallback / reserved tokens never produced by normal text training, so reusing them is harmless. Consequence: **no new embeddings, no vocab resize, the LM head IS the action head.** Inference's argmax is constrained to the last 256 logits (`logits[:, offset:].argmax(...)` in `NanoVLA.predict`) so even uncalibrated text logits can't outvote action bins.

2. **Loss masking via slicing, not `-100` labels.** Sequence layout is `[image_tokens, instruction_tokens, action_tokens]` and CE is computed only on the action positions. Instead of building a full-length labels tensor padded with `-100`, `forward()` slices `logits[:, -T_act-1 : -1, :]` directly. Same math, fewer lines. The off-by-one (`-T_act-1 : -1`) reflects that token at position `p` is predicted by logits at `p-1`.

### Train/eval pixel-pipeline invariant

`convert_libero.resize_views` is imported by `eval_libero.obs_to_images` precisely so the vertical-flip (`f[::-1]`) and resize stay byte-for-byte identical between the trajectories the model trained on and the live sim observations it sees at eval. **Do not duplicate this code.** If you change the flip or resize on either side, you silently break the train/eval correspondence and eval success rate collapses while train accuracy looks fine — this is the most likely cause of the symptom "train action_acc is high but eval succeeds 0%".

The HDF5 key names (`agentview_rgb`, `eye_in_hand_rgb`) and the live robosuite obs key names (`agentview_image`, `robot0_eye_in_hand_image`) differ by suffix but contain identical pixels, so the conversion-time pipeline is reusable as-is.

### Sequence layout in `forward()`

Every batch is laid out as `[N_img vision tokens] + [T_text instruction tokens (padded)] + [T_act action tokens]`. The attention mask is built by `F.pad(instruction_mask, (N_img, T_act), value=1)` — image and action positions are always real, only the instruction can be padded. `T_act = chunk_size * action_dim` (default `8 * 7 = 56`).

### Sampling and DDP

`TrajectoryDataset.__getitem__` ignores its index and samples `(episode, t)` internally via length-weighted `random.choices`. There is **no `DistributedSampler`** — DDP rank disambiguation comes from `make_worker_init(seed, rank)` in `train.py`, which seeds Python's `random` and `numpy.random` per worker per rank. PyTorch's DataLoader auto-seeds `torch`'s RNG in workers but not `random`, so without this every rank would draw the same sequence.

### Other things worth knowing

- Vision tower is **frozen by default** (`freeze_vision=True`); `train.py` calls `base.vision.eval()` to keep its LayerNorms in eval mode even though the rest of the model is `.train()`.
- bf16 is on by default (`--bf16`); only the forward is autocast, the optimizer steps in fp32.
- Inference recomputes from scratch each step (no KV cache). KV cache would be ~5× faster but add complexity; see `NEEDS_REVIEW.md` 1.3.
- Loss/accuracy are accumulated as GPU tensors and only `.item()`d at the log boundary — adding a per-step `.item()` will reintroduce a CUDA sync every iteration.
- `forward()` returns `(loss, action_logits)`; the second tensor exists solely for the train-time accuracy metric. This is an intentional (mild) leaky abstraction — see `NEEDS_REVIEW.md` 4.

## Reference files

- `README.md` is the canonical writeup, including the four-axes design-space chart and per-system comparison table. Items marked with † in that table are unverified; don't cite without checking the source.
- `NEEDS_REVIEW.md` lists open decisions, things to verify on first run (LIBERO API specifics), and changes deliberately not made. Read before making structural changes — it documents *why* certain "improvements" were skipped.
