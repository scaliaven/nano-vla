# nanoVLA

**Disclaimer**: This is a toy project I vibe-coded in my spare time. I think it would be interesting to maintain a family of simple VLAs that mirror the design space of the big, complex, SOTA-chasing ones — both as a teaching artifact for understanding the core mechanics of VLAs without the overhead of production codebases, and as a playground for fast comparison and debugging of variants over a baseline. The simple structure and small model size also make it a good fit for **autoresearch**. That said, I don't have the bandwidth to actively maintain this long-term — but if you find it useful, issues and pull requests are very welcome!

## Overview

A single-file-readable Vision-Language-Action model in the spirit of [nanoGPT](https://github.com/karpathy/nanoGPT). **~1090 lines of Python** across six files — no Trainer / Lightning / Accelerate, no Hydra, no TFDS / RLDS / OXE pipeline. One `@dataclass` at the top of `train.py`. Trains on a LIBERO suite in a few GPU-hours; measured per-suite success rates (36–88%) are in [Results](#results).

A teaching artifact, not a SOTA chase. v1 covers the discrete-token / nanoGPT-style design point — actions as LM vocabulary, single unified transformer — with one deliberate departure from base OpenVLA (see *Parallel action decoding* below). A follow-up will add small single-file variants for other rows of the design-space chart (e.g. flow-matching, continuous-regression). For the chart, per-system comparison (OpenVLA / MiniVLA / π0 / π0-FAST / GR00T N1 / RDT-1B), v2 roadmap, expected numbers, and debugging notes, see [DETAIL.md](./DETAIL.md).

## Architecture

SigLIP-base patch-16-224 → 2-layer GELU MLP projector (LLaVA-style) → Qwen2.5-0.5B. Actions are 7-DoF (6-DoF EE delta + gripper), per-dim uniform-discretized into 256 bins, chunked into windows of 8 (8 × 7 = 56 action tokens). Action tokens reuse **the last 256 tokens of Qwen2.5's existing tokenizer** in place — no new embeddings, no vocab resize, the LM head IS the action head. Cross-entropy is computed only at action positions. Inference is a **single forward pass** that emits all 56 action bins at once: each action position holds its own learned query embedding and the LM "fills it in"; logits are constrained to the vocab tail before argmax, then de-tokenized to a continuous (8, 7) chunk and executed open-loop before re-querying.

### Parallel action decoding (departs from base OpenVLA)

Base OpenVLA is autoregressive over actions: it generates one token at a time, each conditioned on the previous, taking T_act forwards per chunk and exposing the model to a teacher-forcing/inference distribution gap. nanoVLA instead uses **parallel decoding** (à la [OpenVLA-OFT](https://openvla-oft.github.io/)). Two specifics matter for correctness — predicting an action *chunk* is a joint prediction over a trajectory, not 56 independent decisions:

- **Per-position learned query embeddings** (`(1, T_act, hidden_size)`). Each action slot carries its own learned identity; RoPE adds a second axis of differentiation on top. (OFT uses zero/empty embeddings + RoPE only; we add a learnable per slot for slightly richer slot identity at near-zero parameter cost.)
- **Bidirectional attention in the action region.** A 4D attention mask keeps the [image | prompt] block causal (Qwen2.5 default) but makes the action block fully bidirectional, so every action slot attends to every other. Without this, action_query[0] commits before seeing any reasoning at action_query[7] — the chunk becomes 56 quasi-AR sibling predictions sharing a backbone, not a joint chunk prediction. With it, the chunk is jointly self-consistent through depth.

Trade-offs:

- ✅ 1 forward per chunk at inference instead of T_act = 56. Open-loop execution then amortizes that further over 8 env steps.
- ✅ No exposure bias from teacher forcing — at training, action positions never see ground-truth action tokens, so the train and inference input distributions match exactly.
- ✅ One-line slice change (`logits[:, -T_act:, :]` instead of `[:, -T_act-1:-1, :]`); the vocab-tail / "LM head IS the action head" trick is untouched.
- ❌ Costs one `(1, T_act, hidden_size)` learned parameter (the action queries). With T_act = 56 and hidden_size = 896, that's ~50K params — within rounding error of free.
- ❌ Not faithful to vanilla OpenVLA. If you want to study the AR variant, revert this one change — the rest of the codebase is identical to discrete-AR OpenVLA.

### Train/eval padding consistency (important)

The instruction prompt is **left-padded to a fixed length** (`max_instruction_tokens=64`) at both training (in `data.py:Collator`) and inference (in `model.py:NanoVLA.predict`). Fixed-length padding is load-bearing: action_query slots sit at RoPE positions `N_img + max_instruction_tokens + j`, and those positions must match between train and eval. Using `padding="longest"` in the Collator makes positions batch-dependent, which silently breaks eval — bin accuracy on training data was ~30% under that scheme vs ~100% after the fix. Left-padding additionally keeps the last real instruction token adjacent to the first action query (offset −1, constant), which preserves the relative RoPE distance the model uses to look at the "discriminating tail" of the instruction.

## File layout

| File | Functional / Total | What it does |
|---|---:|---|
| [`model.py`](./model.py) | 264 / 347 | Vision tower, projector, action tokenizer, `NanoVLA`, bidirectional-mask + parallel-decoded inference |
| [`data.py`](./data.py) | 264 / 343 | `TrajectoryDataset` (flat .npz), `LeRobotTrajectoryDataset` (parquet+mp4), `Collator`, `make_dataset` |
| [`train.py`](./train.py) | 182 / 236 | `@dataclass` config, raw PyTorch loop, optional DDP / wandb |
| [`convert_libero.py`](./convert_libero.py) | 102 / 123 | LIBERO HDF5 → flat .npz |
| [`convert_libero_lerobot.py`](./convert_libero_lerobot.py) | 111 / 135 | LeRobot v2.x → flat .npz (recommended for LeRobot data) |
| [`eval_libero.py`](./eval_libero.py) | 165 / 217 | LIBERO sim rollout harness, success-rate metric |

**1088 functional / 1401 total** (excludes blank and comment-only lines). Plus [`sanity_replay.py`](./sanity_replay.py) (195 / 244) — a replay diagnostic for debugging eval collapse; not part of the core six-file budget.

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

## Results

A separate nanoVLA checkpoint was trained per suite (single agentview camera, parallel decoding, 80k steps) and evaluated with the `eval_libero.py` sim harness at 5 trials per task:

| Suite | Tasks × trials | Success rate (`ckpt_last`) | Best in per-step sweep |
|---|---:|---:|---:|
| LIBERO-Spatial | 10 × 5 | 62% | **77% @ step 70k** (30 trials) |
| LIBERO-Object | 10 × 5 | 88% | — |
| LIBERO-Goal | 10 × 5 | 78% | **80% @ step 30k** (30 trials) |
| LIBERO-10 (long-horizon) | 10 × 5 | 36% | — |
| LIBERO-90 | 90 × 5 | 76% | — |

Per-task breakdowns and the per-step sweep are in [`eval_results/`](./eval_results/); `verify_eval_results.py` recomputes each file's overall rate from its per-task counts. At 5 trials/task these are ballpark numbers, not leaderboard entries — expect ±10% run-to-run.

**Caveat: these are `ckpt_last` (step 80k) numbers, which for the small suites is well past the overfitting peak.** LIBERO-Spatial / Object / Goal / 10 each have only ~500 demo episodes; at 80k steps × batch 64 the model memorizes them — train `act_acc` reaches 1.000 and loss falls to ~1e-3 (for LIBERO-Spatial, by ~step 50k). LIBERO-90 (~4500 episodes) can't be memorized in the same budget: it plateaus at train `act_acc` ≈ 0.70. For the small suites an earlier checkpoint usually eval-beats `ckpt_last`, but *which* one varies per suite — Goal peaks at ~30k, Spatial at ~70k (see `eval_results/sweep/_best_steps.json`). LIBERO-Spatial is hit hardest because all 10 of its tasks share one instruction template over visually identical black bowls — language is the only disambiguator, so a memorizing model that leans on visual priors grabs the wrong bowl at eval. `sanity_replay.py --instruction-mode scramble` measures how much the policy actually uses the instruction; see [DETAIL.md](./DETAIL.md#expected-numbers) for the broader failure-mode checklist.

**Chunk size matters at eval.** A diagnostic sweep over `--chunk-size ∈ {1,2,4,8}` against the best step of each suite (50 trials/run, `eval_results/diag/`) shows chunking helps a lot on temporally extended tasks but can over-commit on visually ambiguous ones:

| Suite (best step) | K=1 | K=2 | K=4 | K=8 |
|---|---:|---:|---:|---:|
| LIBERO-Goal @ 30k | 48% | 58% | 70% | **76%** |
| LIBERO-Spatial @ 70k | 60% | 60% | **70%** | 64% |

K=8 is the training default; lowering K at rollout (no retrain) is a free knob on suites where the model commits to the wrong sub-goal mid-chunk.

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
