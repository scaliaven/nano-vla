# nanoVLA

A single-file-readable Vision-Language-Action model in the spirit of [nanoGPT](https://github.com/karpathy/nanoGPT). **~900 lines of Python** total (six files, see [File layout](#file-layout)) — no Trainer / Lightning / Accelerate, no Hydra, no TFDS / RLDS / OXE pipeline. One `@dataclass` config at the top of `train.py`. Trains on LIBERO-Spatial in a few GPU-hours and gets to roughly **70–80% success** on the canonical eval.

This is a **teaching artifact and research substrate**, not a SOTA chase. The product is clarity. If a feature would push the line count significantly higher, it goes in the [v2 roadmap](#v2-roadmap) instead.

---

## Contents

- [Architecture](#architecture-in-one-paragraph)
- [File layout](#file-layout)
- [Data format](#data-format) — flat `.npz` contract, plus reading LeRobot v2.x directly
- [Quickstart](#quickstart)
- [The four axes of VLA design space](#the-four-axes-of-vla-design-space) — action repr, action head, backbone coupling, temporal structure
- [Per-system placement](#per-system-placement) — nanoVLA vs OpenVLA / MiniVLA / π0 / π0-FAST / GR00T N1 / RDT-1B
- [Where nanoVLA fits](#where-nanovla-fits)
- [What nanoVLA is NOT](#what-nanovla-is-not)
- [v2 roadmap](#v2-roadmap)
- [Expected numbers](#expected-numbers)
- [Common debugging notes](#common-debugging-notes)
- [References](#references)

---

## Architecture in one paragraph

A single SigLIP-base patch-16-224 vision tower → 2-layer GELU MLP projector (LLaVA-style) → Qwen2.5-0.5B language model. Actions are 7-DoF (6-DoF EE delta + gripper), per-dim uniform-discretized into 256 bins, and chunked into windows of 8, so each forward emits 8 × 7 = 56 sequential action tokens. The action tokens are **the last 256 tokens of Qwen2.5's existing tokenizer**, reinterpreted in place — no new embeddings, no vocab resizing, the LM head IS the action head. Cross-entropy is computed only at the action positions; image and instruction positions are masked out. Inference autoregressively samples 56 tokens with argmax constrained to the vocab tail, then de-tokenizes back to a continuous (8, 7) action chunk that the robot executes open-loop.

## File layout

| File | Functional / Total | What it does |
|---|---:|---|
| [`model.py`](./model.py) | 217 / 289 | Vision tower, projector, action tokenizer, `NanoVLA` module, inference |
| [`data.py`](./data.py) | 209 / 260 | `TrajectoryDataset` (flat .npz), `LeRobotTrajectoryDataset` (parquet+mp4), `Collator`, `make_dataset` factory |
| [`train.py`](./train.py) | 178 / 225 | `@dataclass` config, raw PyTorch loop, optional DDP / wandb |
| [`convert_libero.py`](./convert_libero.py) | 98 / 119 | LIBERO HDF5 → flat .npz episodes + `index.json` + `stats.json` |
| [`convert_libero_lerobot.py`](./convert_libero_lerobot.py) | 111 / 135 | LeRobot v2.x dataset → same flat .npz format (recommended for LeRobot data) |
| [`eval_libero.py`](./eval_libero.py) | 108 / 133 | LIBERO sim rollout harness, success rate metric |

"Functional" counts exclude blank lines and comment-only lines. Total project: **921 functional / 1161 total**.

## Data format

`convert_libero.py` writes a flat per-episode format that any downstream code can read without TFDS / RLDS:

```
data/libero_spatial/
  episode_00000.npz   # keys: images_primary (T,224,224,3) uint8,
                              images_wrist   (T,224,224,3) uint8,
                              actions        (T,7) float32,
                              instruction    str
  episode_00001.npz
  ...
  index.json          # per-episode metadata: {length, instruction, file, source_*}
  stats.json          # dataset-wide action q01/q99 (for the discretizer), num_bins, image_size
```

This is the canonical contract. To train on a non-LIBERO dataset, write your own conversion script that produces the same files — nothing in `model.py`, `data.py`, or `train.py` is LIBERO-specific. Files are **uncompressed** (`np.savez`) so the dataloader can `mmap_mode='r'` and page in only one frame per sample; compressed npz would force whole-episode decompression per `__getitem__` (~10× slower). The disk cost is real (LIBERO-Spatial is ~22 GB vs ~3 GB compressed).

### Two ways to use a LeRobot dataset

Many recent datasets (including a port of LIBERO) ship in [LeRobot v2.x](https://github.com/huggingface/lerobot) layout (parquet + mp4). nanoVLA supports it through two paths and `train.py` auto-detects which one to use:

- **Recommended — convert once:** run `python convert_libero_lerobot.py --src <lerobot_root> --out data/<name>` to emit the same flat `.npz` contract. Training then uses the fast mmap'd `TrajectoryDataset`.
- **Convenient — read directly:** point `--data-dir` at the LeRobot root and `make_dataset` returns a `LeRobotTrajectoryDataset` that decodes mp4 frames on-the-fly with a per-worker LRU cache. No conversion step, but slower (AV1 has sparse keyframes), so prefer this for prototyping rather than long runs.

## Quickstart

```bash
# 0. deps
pip install torch torchvision transformers einops numpy pillow tqdm h5py
# optional: wandb (logging), libero (eval)

# 1. convert demos to the flat .npz format
#    LIBERO HDF5:
python convert_libero.py --src /path/to/libero_spatial_no_noops --out data/libero_spatial
#    or LeRobot v2.x:
python convert_libero_lerobot.py --src /path/to/lerobot_dataset --out data/<name>

# 2. smoke-test the eval harness with a random policy (will not succeed,
#    but verifies the rollout loop works before any training time is spent)
python eval_libero.py --policy random --suite libero_spatial --num-trials 2

# 3. train (single GPU). --data-dir can be either a flat-npz dir or a LeRobot root.
python train.py --data-dir data/libero_spatial --steps 50000 --batch-size 8

# 3'. or DDP
torchrun --nproc-per-node=4 train.py --data-dir data/libero_spatial --batch-size 8

# 4. evaluate the trained model
python eval_libero.py --policy nano-vla --ckpt out/ckpt_last.pt --suite libero_spatial
```

Every field in `TrainConfig` (top of `train.py`) is also a CLI flag. Booleans become `--foo` / `--no-foo`.

---

## The four axes of VLA design space

VLA models look superficially similar but differ on several axes at once. The four below cleanly separate the design choices; nanoVLA picks one point on each, chosen so any single axis can be swapped without rewriting the rest of the codebase.

### 1. Action representation

How the continuous robot action signal (typically 6-DoF or 7-DoF EE delta + gripper) is encoded. The main families are **per-dim uniform discrete bins**, **FAST / DCT tokens** (DCT + BPE over a chunk), **continuous via flow matching or diffusion**, and **VQ codebook tokens**. The choice trades off sample efficiency, action smoothness, sequence length, and how cleanly actions slot into a language model's token stream.

### 2. Action head

What module produces the action numbers from the backbone's hidden states: **the LM head reused via a vocab-tail token mapping** (cheapest), a **separate MLP or transformer regressor**, a **diffusion or flow-matching expert** conditioned on backbone features, or a dedicated **"action expert" sub-network** that shares attention with the VLM but has its own parameters. The head's complexity often dominates whether the system can express multi-modal action distributions.

### 3. Backbone coupling

How vision, language, and action representations interact. A **single unified transformer** processes patches, text, and action tokens in one sequence (OpenVLA). A **frozen VLM + adapter** learns only a small bridge to actions. A **VLM with cross-attention into a separate action expert** (π0) lets a heavy VLM feed a smaller, faster action network. An **encoder-decoder split** uses different stacks for observation encoding and action decoding. Coupling determines what's frozen vs trained, and whether action inference requires running the full VLM at every control step.

### 4. Temporal structure

How the model handles time. **Single-step** predicts only the next action. **Action chunking with chunk size K** predicts K future actions per forward pass and executes them open-loop before re-querying, amortizing inference cost and smoothing trajectories. **History-conditioned** input feeds the last H observations as context. Most modern VLAs chunk; K (commonly 8–50) interacts strongly with control frequency and action representation.

## Per-system placement

| System | Action representation | Action head | Backbone coupling | Temporal structure |
|---|---|---|---|---|
| **nanoVLA** *(this repo)* | Per-dim uniform 256-bin discretization | LM head, reusing the **last 256 tokens** of the Qwen2.5 vocab as action tokens | Single unified transformer (SigLIP-base + 2-layer GELU MLP projector + Qwen2.5-0.5B) | Action chunking, K = 8 (8 × 7 = 56 sequential action tokens) |
| **OpenVLA** | Per-dim uniform 256-bin discretization | LM head, overwriting the 256 least-used tokens of the Llama-2 tokenizer | Single unified transformer (Prismatic VLM: SigLIP + DINOv2 + Llama-2-7B) | Single-step (one 7-token action per forward pass) |
| **MiniVLA** *(Stanford)* | Per-dim discrete bins or learned VQ over chunks † | LM head over the action-token vocabulary | Single unified transformer with a smaller (~1B) backbone | Action chunking † |
| **π0** | Continuous, via flow matching | Separate "action expert" sub-network trained with a flow-matching objective | VLM (PaliGemma-class) feeding the action expert via shared attention; the expert has its own parameters but interleaves with VLM blocks | Action chunking (K typically large, ~50) † |
| **π0-FAST** | FAST tokens (DCT over an action chunk + BPE into a short discrete sequence) | LM head, autoregressive over FAST tokens | Same VLM backbone as π0, but actions emitted by the LM rather than by a flow-matching expert | Action chunking |
| **GR00T N1** *(NVIDIA)* | Continuous, via diffusion | Diffusion transformer ("System 1") action head | Two-system split: a slower VLM ("System 2") produces latent conditioning; a faster diffusion transformer ("System 1") consumes it and emits actions † | Action chunking † |
| **RDT-1B** | Continuous, via diffusion | Unified diffusion transformer (the same transformer denoises actions while attending to vision/language conditioning) † | Encoder-decoder-ish split: pretrained vision and text encoders provide conditioning; a ~1B diffusion transformer is the action decoder | Action chunking for bimanual control (K reportedly 64) † |

† *Items marked with a dagger are points where this README's authors are not certain of the published details — verify in the original paper before citing.*

### Notes on each system

- **nanoVLA** *(this repo)*. The smallest readable point in the OpenVLA lineage: Qwen2.5-0.5B + SigLIP-base + 2-layer MLP projector, with OpenVLA-OFT-style chunking (K=8). The last 256 tokens of Qwen2.5's vocab are reinterpreted as action bins — no new embeddings, no vocab resize.
- **OpenVLA** (Kim et al., 2024). The reference point for the discrete-AR family: Prismatic VLM (SigLIP + DINOv2, Llama-2-7B) with 256 rarely-used Llama tokens overwritten as action bins. Single-step; later OpenVLA-OFT adds chunking and parallel decoding.
- **MiniVLA** (Stanford, Belkhale et al., 2024). Small-scale VLA aimed at making OpenVLA-style training tractable on modest hardware. Whether actions are per-dim discrete or VQ-tokenized varies by reference — verify in the paper/repo.
- **π0** (Black et al., 2024). A flow-matching "action expert" shares attention with a PaliGemma-class VLM but has its own weights, so the expensive VLM can run at a lower rate while the lighter expert produces continuous chunks at control rate.
- **π0-FAST** (Pertsch et al., 2025). Replaces π0's flow-matching expert with an autoregressive LM head over **FAST tokens** (DCT + BPE over an action chunk), recovering the "actions are just tokens" simplicity without the long flat sequences of per-dim binning.
- **GR00T N1** (NVIDIA, 2025). Humanoid model with an explicit System-1 / System-2 split: a slower VLM produces latent conditioning, and a faster diffusion transformer denoises continuous actions at control rate. Verify exact head architecture and chunk size in the tech report.
- **RDT-1B** (Liu et al., 2024). A ~1B-parameter diffusion transformer for bimanual manipulation with a unified action space. Verify whether the diffusion module is a separate head or the whole transformer is the denoiser.

## Where nanoVLA fits

On all four axes, nanoVLA picks the simplest published option: per-dim uniform discrete bins, the LM head reused via vocab-tail mapping, a single unified transformer, and small-K action chunking. That puts it squarely in the OpenVLA lineage, **not** the π0 / GR00T / RDT lineage — no diffusion, no flow matching, no separate action expert, no new embeddings. The payoff is that each axis becomes a clean swap: representation, head, coupling, or K can be studied one at a time. At ~900 lines and ~0.5B params, any one of these swaps fits in a single readable diff.

---

## What nanoVLA is NOT

- **Not SOTA.** π0, GR00T N1, RDT, π0-FAST, and OpenVLA-OFT all beat us — we trade capability for clarity.
- **Not a foundation model.** No Open-X / multi-robot pretraining; fine-tuned from a base VLM on one dataset at a time.
- **Not Open-X compatible.** No TFDS / RLDS / OXE pipeline, by design — bring your own data via the flat .npz format.
- **Not a deployment-ready stack.** No calibration, safety envelopes, async control, or hardware drivers; inference is synchronous PyTorch.
- **Not feature-complete.** No multi-camera fusion (dual view is uncoupled), no proprioception input, no OOD-eval harness.

## v2 roadmap

Things we deliberately did NOT do in v1, with a brief note on what they'd cost in lines and what they'd buy:

- **FAST tokenization.** Replace per-dim binning with DCT + BPE action tokens. **+~150 LoC.** Lets us use longer chunks (K = 32–50) without an enormous action-token sequence; better fidelity for fast / fine motions.
- **Multi-camera fusion.** Dual-view exists in the data path but is uncoupled (patches concatenated). v2 would add per-view positional embeddings or cross-attention between views. **+~50 LoC.**
- **Proprioception.** Currently no joint-state input. Add a small MLP encoder for proprio + concat its tokens into the LM input. **+~40 LoC.** Helps on contact-rich tasks where vision alone is ambiguous.
- **Residual VQ action tokens.** Replace per-dim bins with a learned residual VQ codebook over action chunks. **+~200 LoC.** Better fidelity, especially for fine manipulation; more code complexity.
- **Flow-matching action head.** Replace the LM head + per-dim bins with a flow-matching action expert (π0-style). **+~250 LoC.** Smooth continuous actions, multi-modal distributions, but moves us out of the discrete-AR family entirely — it's a different system.

Each of these is a single-axis swap on the design-space chart above; the rest of the pipeline (data format, eval harness, training loop) stays untouched.

## Expected numbers

LIBERO-Spatial success rate after training:

| Setup | Expected SR | Compute budget |
|---|---|---|
| nanoVLA, single camera (agentview), 50k steps, batch 8 | **70–80%** | ~6h on 1× 24GB GPU |
| nanoVLA, dual camera (agentview + wrist), 50k steps | **75–82%** | ~7h on 1× 24GB GPU |

These are honest expectations given the architecture, **not aspirational claims**. If you see **<60%**, something is broken — most likely action-tokenizer round-trip, image flip orientation between train and eval, or instruction tokenization mismatch. **<70%** is plausible if eval init-state seeds differ from training trajectory diversity, or if you trained for fewer than 30k steps. **>80%** would surprise me — π0-FAST and OpenVLA-OFT currently sit in the 80–90% range and they're doing more than nanoVLA does.

If you want SOTA-ish numbers on LIBERO, use π0-FAST or OpenVLA-OFT. If you want to *understand* a discrete-AR VLA in an evening, this is the repo.

## Common debugging notes

- **Action accuracy is low (~1/256 = 0.4%)**: model is still predicting random tokens; either training hasn't started or the loss masking is wrong (logits aren't being sliced to action positions). Check the `forward()` slice in `model.py`.
- **Action accuracy is high (>50%) but eval success is 0**: train/eval pixel pipelines disagree. Check `obs_to_images` in `eval_libero.py` matches `resize_views` in `convert_libero.py` (vertical flip and resize must match).
- **Action accuracy is high and eval succeeds intermittently**: try reducing chunk size (more frequent re-planning). LIBERO trajectories sometimes need closed-loop correction.

## License

TBD.

## References

arXiv IDs given where the README authors are confident; verify before citing in academic work.

- Belkhale et al. *MiniVLA*. 2024. (verify exact title and arXiv id)
- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164, 2024.
- Karpathy. *nanoGPT*. https://github.com/karpathy/nanoGPT (the inspiration for this repo's spirit).
- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*. arXiv:2406.09246, 2024.
- Liu et al. *RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation*. 2024. (verify arXiv id)
- NVIDIA. *GR00T N1: A Foundation Model for Humanoid Robots*. 2025. (see project page)
- Pertsch et al. *FAST: Efficient Action Tokenization for Vision-Language-Action Models*. arXiv:2501.09747, 2025.
- LIBERO benchmark: Liu et al. *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*. 2023.
