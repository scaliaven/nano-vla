# nanoVLA

A single-file-readable Vision-Language-Action model in the spirit of [nanoGPT](https://github.com/karpathy/nanoGPT). **~1050 lines of Python** across six files — no Trainer / Lightning / Accelerate, no Hydra, no TFDS / RLDS / OXE pipeline. One `@dataclass` at the top of `train.py`. Trains on LIBERO-Spatial in a few GPU-hours and reaches roughly **70–80% success**.

A teaching artifact, not a SOTA chase. v1 covers the discrete-token / nanoGPT-style design point — actions as LM vocabulary, single unified transformer — with one deliberate departure from base OpenVLA (see *Parallel action decoding* below). A follow-up will add small single-file variants for other rows of the design-space chart (e.g. flow-matching, continuous-regression). For the chart, per-system comparison (OpenVLA / MiniVLA / π0 / π0-FAST / GR00T N1 / RDT-1B), v2 roadmap, expected numbers, and debugging notes, see [DETAIL.md](./DETAIL.md).

## Architecture

SigLIP-base patch-16-224 → 2-layer GELU MLP projector (LLaVA-style) → Qwen2.5-0.5B. Actions are 7-DoF (6-DoF EE delta + gripper), per-dim uniform-discretized into 256 bins, chunked into windows of 8 (8 × 7 = 56 action tokens). Action tokens reuse **the last 256 tokens of Qwen2.5's existing tokenizer** in place — no new embeddings, no vocab resize, the LM head IS the action head. Cross-entropy is computed only at action positions. Inference is a **single forward pass** that emits all 56 action bins at once: each action position holds a learned query embedding and the LM "fills it in"; logits are constrained to the vocab tail before argmax, then de-tokenized to a continuous (8, 7) chunk and executed open-loop before re-querying.

### Parallel action decoding (departs from base OpenVLA)

Base OpenVLA is autoregressive over actions: it generates one token at a time, each conditioned on the previous, taking T_act forwards per chunk and exposing the model to a teacher-forcing/inference distribution gap. nanoVLA instead uses **parallel decoding** (à la [OpenVLA-OFT](https://openvla-oft.github.io/)): all T_act action positions hold the same learned query embedding, the LM emits all action logits in one forward, and RoPE positional encoding is what differentiates the slots. Trade-offs:

- ✅ 1 forward per chunk at inference instead of T_act = 56. Open-loop execution then amortizes that further over 8 env steps.
- ✅ No exposure bias from teacher forcing — at training, action positions never see ground-truth action tokens, so the train and inference input distributions match exactly.
- ✅ One-line slice change (`logits[:, -T_act:, :]` instead of `[:, -T_act-1:-1, :]`); the vocab-tail / "LM head IS the action head" trick is untouched.
- ❌ Costs one `(1, 1, hidden_size)` learned parameter (the action query). Within rounding error of free.
- ❌ Not faithful to vanilla OpenVLA. If you want to study the AR variant, revert this one change — the rest of the codebase is identical to discrete-AR OpenVLA.

## File layout

| File | Functional / Total | What it does |
|---|---:|---|
| [`model.py`](./model.py) | 230 / 300 | Vision tower, projector, action tokenizer, `NanoVLA`, parallel-decoded inference |
| [`data.py`](./data.py) | 257 / 326 | `TrajectoryDataset` (flat .npz), `LeRobotTrajectoryDataset` (parquet+mp4), `Collator`, `make_dataset` |
| [`train.py`](./train.py) | 178 / 228 | `@dataclass` config, raw PyTorch loop, optional DDP / wandb |
| [`convert_libero.py`](./convert_libero.py) | 102 / 123 | LIBERO HDF5 → flat .npz |
| [`convert_libero_lerobot.py`](./convert_libero_lerobot.py) | 111 / 135 | LeRobot v2.x → flat .npz (recommended for LeRobot data) |
| [`eval_libero.py`](./eval_libero.py) | 166 / 217 | LIBERO sim rollout harness, success-rate metric |

**1044 functional / 1329 total** (excludes blank and comment-only lines).

## Quickstart

```bash
pip install torch torchvision transformers einops numpy pillow tqdm h5py
# optional: wandb (logging), libero (eval)

# 1. convert demos to the flat .npz format
python convert_libero.py --src /path/to/libero_spatial_no_noops --out data/libero_spatial
# or LeRobot v2.x:
python convert_libero_lerobot.py --src /path/to/lerobot_dataset --out data/<name>

# 2. smoke-test the eval harness (random policy won't succeed, just verifies the loop)
python eval_libero.py --policy random --suite libero_spatial --num-trials 2

# 3. train (single GPU; --data-dir auto-detects flat-npz vs LeRobot)
python train.py --data-dir data/libero_spatial --steps 50000 --batch-size 8
# DDP:
torchrun --nproc-per-node=4 train.py --data-dir data/libero_spatial --batch-size 8

# 4. evaluate
python eval_libero.py --policy nano-vla --ckpt out/ckpt_last.pt --suite libero_spatial
```

Every field of `TrainConfig` (top of `train.py`) is auto-exposed as a CLI flag; booleans become `--foo` / `--no-foo`.

## Data format

Both converters emit a flat per-episode contract that any downstream code can read without TFDS / RLDS:

```
data/<dataset>/
  episode_NNNNN.npz   # images_primary, images_wrist (uint8), actions (T,7) float32, instruction (str)
  index.json          # per-episode metadata
  stats.json          # action q01/q99 (for the discretizer), num_bins, image_size
```

Files are **uncompressed** (`np.savez`) so the dataloader can `mmap_mode='r'` and page in only one frame per sample; compressed npz would force whole-episode decompression per `__getitem__` (~10× slower). The disk cost is real (LIBERO-Spatial is ~22 GB vs ~3 GB compressed). To train on a non-LIBERO dataset, write a converter that emits this exact layout — nothing in `model.py` / `data.py` / `train.py` is LIBERO-specific.

**LeRobot v2.x** datasets work two ways and `train.py` auto-detects which one to use:
- *Recommended* — run `convert_libero_lerobot.py` once, then train on the flat `.npz` output.
- *Convenient* — point `--data-dir` at the LeRobot root directly; `make_dataset` returns a `LeRobotTrajectoryDataset` that decodes mp4 on-the-fly with a per-worker LRU cache. Slower (sparse AV1 keyframes), so prefer this for prototyping rather than long runs.

## Where nanoVLA fits

The simplest published point in the OpenVLA lineage: per-dim discrete action bins, LM head reused via vocab-tail mapping, a single unified transformer, small-K action chunking. **Not** the π0 / GR00T / RDT lineage — no diffusion, no flow matching, no separate action expert. The payoff is that each design axis becomes a clean swap: at ~900 lines and ~0.5B params, any single-axis change fits in a single readable diff. The full four-axis chart and per-system comparison live in [DETAIL.md](./DETAIL.md).

If you want SOTA-ish numbers on LIBERO, use π0-FAST or OpenVLA-OFT. If you want to *understand* a discrete-token VLA in an evening, this is the repo.

## License

TBD.

## References

- Karpathy. *nanoGPT*. https://github.com/karpathy/nanoGPT
- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*. arXiv:2406.09246, 2024.
- Kim et al. *Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success* (OpenVLA-OFT). arXiv:2502.19645, 2025. — source of the parallel-decoding recipe used here.
- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164, 2024.
- Pertsch et al. *FAST: Efficient Action Tokenization for VLA Models*. arXiv:2501.09747, 2025.
- Liu et al. *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*. arXiv:2306.03310, NeurIPS 2023 (Datasets & Benchmarks).

Extended reference list (MiniVLA, GR00T N1, RDT-1B) is in [DETAIL.md](./DETAIL.md).
