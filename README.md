# nanoVLA

A single-file-readable Vision-Language-Action model in the spirit of [nanoGPT](https://github.com/karpathy/nanoGPT). **~600 lines of Python** total — no Trainer / Lightning / Accelerate, no Hydra, no TFDS / RLDS / OXE pipeline. One `@dataclass` config at the top of `train.py`. Trains on LIBERO-Spatial in a few GPU-hours and gets to roughly **70–80% success** on the canonical eval.

This is a **teaching artifact and research substrate**, not a SOTA chase. The product is clarity. If a feature would push us past the line budget, it goes in the [v2 roadmap](#v2-roadmap) instead.

---

## Architecture in one paragraph

A single SigLIP-base patch-16-224 vision tower → 2-layer GELU MLP projector (LLaVA-style) → Qwen2.5-0.5B language model. Actions are 7-DoF (6-DoF EE delta + gripper), per-dim uniform-discretized into 256 bins, and chunked into windows of 8, so each forward emits 8 × 7 = 56 sequential action tokens. The action tokens are **the last 256 tokens of Qwen2.5's existing tokenizer**, reinterpreted in place — no new embeddings, no vocab resizing, the LM head IS the action head. Cross-entropy is computed only at the action positions; image and instruction positions are masked out. Inference autoregressively samples 56 tokens with argmax constrained to the vocab tail, then de-tokenizes back to a continuous (8, 7) action chunk that the robot executes open-loop.

## File layout

| File | Functional / Total | What it does |
|---|---:|---|
| [`model.py`](./model.py) | 169 / 301 | Vision tower, projector, action tokenizer, `NanoVLA` module, inference |
| [`data.py`](./data.py) | 86 / 134 | `TrajectoryDataset` over the flat .npz format, `Collator` |
| [`train.py`](./train.py) | 161 / 217 | `@dataclass` config, raw PyTorch loop, optional DDP / wandb |
| [`convert_libero.py`](./convert_libero.py) | 78 / 121 | LIBERO HDF5 → flat .npz episodes + `index.json` + `stats.json` |
| [`eval_libero.py`](./eval_libero.py) | 99 / 150 | LIBERO sim rollout harness, success rate metric |

"Functional" counts exclude blank lines, comments, and docstrings. Total project: **593 functional / ~1000 total**.

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

This is the canonical contract. To train on a non-LIBERO dataset, write your own conversion script that produces the same files; nothing in `model.py`, `data.py`, or `train.py` is LIBERO-specific.

`.npz` files are **uncompressed** (`np.savez`, not `np.savez_compressed`) so that the dataloader can `mmap_mode='r'` and only page in the bytes for one frame per sample. The training-throughput cost of compressed npz is ~10× because every `__getitem__` would have to decompress the entire episode just to read one timestep. Disk overhead is the price: LIBERO-Spatial is ~22 GB on disk in this format vs ~3 GB compressed.

## Quickstart

```bash
# 0. deps
pip install torch torchvision transformers einops numpy pillow tqdm h5py
# optional: wandb (logging), libero (eval)

# 1. convert LIBERO HDF5 demos to nanoVLA's flat .npz format
python convert_libero.py --src /path/to/libero_spatial_no_noops --out data/libero_spatial

# 2. smoke-test the eval harness with a random policy (will not succeed,
#    but verifies the rollout loop works before any training time is spent)
python eval_libero.py --policy random --suite libero_spatial --num-trials 2

# 3. train (single GPU)
python train.py --data-dir data/libero_spatial --steps 50000 --batch-size 8

# 3'. or DDP
torchrun --nproc-per-node=4 train.py --data-dir data/libero_spatial --batch-size 8

# 4. evaluate the trained model
python eval_libero.py --policy nano-vla --ckpt out/ckpt_last.pt --suite libero_spatial
```

Every field in `TrainConfig` (top of `train.py`) is also a CLI flag. Booleans become `--foo` / `--no-foo`.

---

## The four axes of VLA design space

Vision-Language-Action models look superficially similar — a vision encoder, a language model, something that emits robot actions — but the design space underneath is wide, and most published systems differ on several axes at once. This section names four axes that cleanly separate the design choices, and places seven recent systems on each. nanoVLA itself is one specific point in this space, chosen for **pedagogical clarity rather than performance**: the goal is that you can swap any single axis without rewriting the rest of the codebase.

### 1. Action representation

How the continuous robot action signal (typically 6-DoF or 7-DoF EE delta + gripper) is encoded for the model to consume or emit. The main families are: **per-dim uniform discrete bins** (each action dim independently quantized into N buckets, usually 256); **FAST / DCT tokens** (compress an action chunk via DCT + BPE into a short variable-length token sequence); **continuous via flow matching or diffusion** (no discretization; the model emits real-valued vectors via an iterative denoiser); and **VQ codebook tokens** (a learned vector quantizer over action chunks). The choice trades off sample efficiency, action smoothness, sequence length, and how cleanly actions slot into a language model's token stream.

### 2. Action head

What module actually produces the action numbers from the backbone's hidden states. Options include: **the LM head reused via a vocab-tail token mapping** (cheapest — pick N existing tokens to mean "action bin k", so the LM's softmax already covers them); a **separate small MLP or transformer regressor** on top of the backbone; a **diffusion or flow-matching expert** (a separate denoising network conditioned on backbone features); or a dedicated **"action expert" sub-network** that shares attention with the VLM but has its own parameters. The action head's complexity often dominates whether the system can express multi-modal action distributions.

### 3. Backbone coupling

How vision, language, and action representations interact inside the network. A **single unified transformer** processes image patches, text tokens, and action tokens in one sequence (OpenVLA-style). A **frozen VLM + adapter** keeps a pretrained VLM fixed and learns only a small bridge to actions. A **VLM with cross-attention into a separate action expert** (the π0 pattern) lets a heavy VLM feed conditioning into a smaller, faster action network. An **encoder-decoder split** uses one stack to encode observations and a different stack — often a diffusion transformer — to decode actions. Coupling determines what's frozen vs trained, and whether action inference requires running the full VLM at every control step.

### 4. Temporal structure

How the model handles time. **Single-step** predicts only the next action given the current observation. **Action chunking with chunk size K** predicts K future actions in one forward pass, then executes some or all of them open-loop before re-querying — this amortizes inference cost and tends to smooth trajectories. **History-conditioned** input feeds the model the last H observations or actions as context. Most modern VLAs chunk; the choice of K (commonly 8–50) interacts strongly with control frequency and action representation.

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

- **nanoVLA** *(this repo)*. The smallest readable point in the OpenVLA lineage. Keeps OpenVLA's per-dim 256-bin discretization and "actions are just LM tokens" trick, but (a) shrinks the LM to Qwen2.5-0.5B, (b) uses a single SigLIP-base tower behind a 2-layer GELU MLP projector, and (c) adds OpenVLA-OFT-style action chunking with K=8. Crucially, no embeddings are added and no vocab is resized: the **last** 256 tokens of Qwen2.5's existing vocabulary are reinterpreted as action bins, so the LM head doubles as the action head.
- **OpenVLA** (Kim et al., 2024). The reference point for the discrete-AR family: Prismatic-style VLM (SigLIP + DINOv2, Llama-2-7B) fine-tuned on OXE, with 256 rarely-used Llama tokens overwritten to mean uniform action bins. Predicts a single 7-DoF action per forward pass; later work (OpenVLA-OFT) adds chunking and parallel decoding.
- **MiniVLA** (Stanford, Belkhale et al., 2024). Small-scale VLA aimed at making OpenVLA-style training tractable on modest hardware. There are at least two related Stanford small-VLA efforts; details on whether actions are per-dim discrete or VQ-tokenized vary by reference — verify in the MiniVLA paper/repo before citing specifics.
- **π0** (Black et al., 2024). Introduces a flow-matching "action expert" that shares attention with a PaliGemma-class VLM but has its own weights, so the expensive VLM can run at a lower rate while the lighter expert produces continuous action chunks at control rate. The continuous representation removes quantization artifacts and naturally handles multi-modal action distributions.
- **π0-FAST** (Pertsch et al., 2025). Replaces π0's flow-matching expert with an autoregressive LM head over **FAST tokens**: take an action chunk, apply a DCT to compress its low-frequency structure, then BPE-encode the result into a short variable-length token sequence. This recovers the "actions are just tokens" simplicity of OpenVLA while avoiding the long flat sequences that per-dim binning produces for long chunks.
- **GR00T N1** (NVIDIA, 2025). Humanoid foundation model with an explicit System-1 / System-2 split inspired by dual-process cognition: a slower VLM ("System 2") produces high-level latent conditioning, and a faster diffusion transformer ("System 1") consumes that conditioning and denoises continuous actions at control rate. Verify the exact diffusion-head architecture and chunk size in the GR00T N1 tech report.
- **RDT-1B** (Liu et al., 2024). A ~1B-parameter diffusion transformer specialized for bimanual manipulation, with a unified action space across robots. Different summaries describe RDT differently — verify whether the diffusion module is a separate head conditioned on a frozen VLM, or whether the whole transformer is itself the denoiser with vision/language injected as conditioning tokens.

## Where nanoVLA fits

On all four axes, nanoVLA picks the simplest published option: per-dim uniform discrete bins, the LM head reused via vocab-tail mapping, a single unified transformer, and action chunking with a small K. That puts it squarely in the OpenVLA lineage, **not** the π0 / GR00T / RDT lineage — there is no diffusion network, no flow matching, no separate action expert, and no new embeddings to manage. The payoff is that each axis becomes a clean teaching exercise: swap the discretizer for FAST tokens to study representation; replace the LM head with an MLP regressor or a tiny diffusion head to study action heads; freeze the VLM and add a cross-attended expert to study coupling; vary K from 1 to 32 to study temporal structure. The whole model is small enough (~600 lines, ~0.5B params) that any one of these swaps fits in a single readable diff.

---

## What nanoVLA is NOT

- **Not SOTA.** π0, GR00T N1, RDT, π0-FAST, and the OpenVLA-OFT line all beat us on every benchmark we'd run. We are explicitly trading capability for clarity.
- **Not a foundation model.** We don't pretrain on Open-X-Embodiment or any multi-robot corpus. nanoVLA is fine-tuned from a base VLM on one robot dataset at a time.
- **Not Open-X compatible.** No TFDS / RLDS / OXE pipeline, by design. Bring-your-own-data is the flat .npz format described above; you write your own conversion script per dataset.
- **Not a deployment-ready stack.** Real-robot deployment works in principle, but we don't ship calibration, safety envelopes, async control, or hardware drivers. Inference is synchronous PyTorch.
- **Not feature-complete.** No multi-camera fusion (dual view exists in the data path but is uncoupled — concatenated patches), no proprioception input, no language-conditioned reset behavior, no eval-on-OOD-objects harness.

If you need any of these, see the [v2 roadmap](#v2-roadmap) or use one of the bigger systems in the table above.

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

Listed in alphabetical order. arXiv IDs given where the README authors are confident; verify before citing in academic work.

- Belkhale et al. *MiniVLA*. 2024. (verify exact title and arXiv id)
- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164, 2024.
- Karpathy. *nanoGPT*. https://github.com/karpathy/nanoGPT (the inspiration for this repo's spirit).
- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*. arXiv:2406.09246, 2024.
- Liu et al. *RDT-1B: A Diffusion Foundation Model for Bimanual Manipulation*. 2024. (verify arXiv id)
- NVIDIA. *GR00T N1: A Foundation Model for Humanoid Robots*. 2025. (see project page)
- Pertsch et al. *FAST: Efficient Action Tokenization for Vision-Language-Action Models*. arXiv:2501.09747, 2025.
- LIBERO benchmark: Liu et al. *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*. 2023.
