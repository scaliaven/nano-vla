# NEEDS_REVIEW

Open items the project owner should look at before / on first run. Grouped by what kind of decision is required. Last updated after the simplify pass.

---

## 1. Decisions I made without explicit go-ahead

These are fine if you agree, but they're not your choices yet.

| # | Where | Decision | Alternative |
|---|---|---|---|
| 1.1 | `model.py` `VLAConfig.prompt_template` | Used OpenVLA's `"In: What action should the robot take to {instruction}? Out:"` | Could use a chat template, a bare instruction, or a custom format. One-line change. |
| 1.2 | `convert_libero.py:resize_views` | **Vertical flip only** (`f[::-1]`), matching LIBERO's official eval convention. | OpenVLA-OFT uses `[::-1, ::-1]` (180° rotation). If your reference is OpenVLA-OFT, change one line in both `convert_libero.py` and `eval_libero.obs_to_images`. |
| 1.3 | `model.py:NanoVLA.predict` | **Recomputation, no KV cache.** ~30 ms per chunk on GPU. | KV cache via `past_key_values` would cut that ~5×. Not worth the complexity for chunked eval; reconsider for low-latency real-robot deploy. |
| 1.4 | `train.py:TrainConfig` defaults | bf16 + AdamW + cosine-with-warmup + β=(0.9, 0.95) + lr=2e-5 + warmup 500 + clip 1.0. | OpenVLA-ish defaults; no source of truth. Adjust per your training rig. |
| 1.5 | `convert_libero.py` | **`np.savez` (uncompressed)**, not `np.savez_compressed`. ~22 GB on disk for LIBERO-Spatial vs ~3 GB compressed. | Required so the dataloader can `mmap_mode='r'` and avoid decompressing whole episodes per sample. If disk is the constraint, keep compressed and accept ~10× slower data loading. |
| 1.6 | `README.md` License section | **TBD.** | Pick one before publishing. |

## 2. Things to verify on first run (LIBERO API specifics)

I wrote these from memory of the LIBERO API; verify against your actual install.

| # | Where | What to check |
|---|---|---|
| 2.1 | `eval_libero.py:obs_to_images` | Live robosuite obs uses `"agentview_image"` and `"robot0_eye_in_hand_image"`. If your version differs (e.g. drops the `robot0_` prefix), this is a one-line fix. |
| 2.2 | `convert_libero.py` HDF5 keys | Used `data/demo_*/obs/agentview_rgb` and `data/demo_*/obs/eye_in_hand_rgb`. Standard for `libero_spatial_no_noops`, but verify against the file you're converting. |
| 2.3 | `eval_libero.py:make_env` | Uses `OffScreenRenderEnv` from `libero.libero.envs` with `bddl_file_name`, `camera_heights`, `camera_widths`. Confirm the import path and constructor signature for your LIBERO version. |
| 2.4 | `eval_libero.py` | Uses `bench.get_task(i).problem_folder`, `.bddl_file`, `.language`, `.name` and `bench.get_task_init_states(i)`. All standard, but a sanity print on the first task before a long eval would catch any AttributeError early. |
| 2.5 | `convert_libero.py:parse_instruction` | Filename-based instruction extraction (regex strips `_demo.hdf5`, `_SCENE\d+`, leading ALL-CAPS). After conversion, scan a few `index.json` entries to confirm the instructions look right. |

## 3. README factual claims to verify

The design-space writeup was researched without web access; the table cells and citations marked with † in `README.md` are uncertain. Specifically:

- **MiniVLA** action representation (per-dim discrete vs learned VQ over chunks)
- **π0** chunk size K (claimed "~50")
- **GR00T N1** System-2 freezing status; exact diffusion-head architecture; chunk size
- **RDT-1B** whether the diffusion module is a separate head conditioned on a frozen VLM, or whether the whole transformer is itself the denoiser
- **Exact arXiv IDs** for RDT-1B and MiniVLA — left as "verify" rather than guessed

If you cite this README in academic work, sanity-check each daggered claim against the source paper.

## 4. Refactor pass — what changed (for transparency)

After the initial draft, three review agents flagged 25 issues. **12 were fixed**, **4 explicitly skipped**.

### Fixed (high-impact)

- **`data.py:_load_episode`** read entire compressed episode per sample → switched `convert_libero.py` to uncompressed `np.savez`, dataloader uses `mmap_mode='r'`. Loads only the bytes for one frame + the action chunk.
- **DDP RNG correctness**: `random.choices` inside `__getitem__` was not seeded per-rank, so DDP runs would silently sample the same sequence on every rank. Added `make_worker_init(seed, rank)` in `train.py`. Dropped `DistributedSampler` since the dataset ignores indices.
- **`train.py` per-step `.item()` calls**: forced CUDA syncs every iteration. Now loss/acc are accumulated as GPU scalars; `.item()` only at log boundary.

### Fixed (cleanup)

- Dead `num_image_tokens()` method (`model.py`) — deleted.
- Dead `self.stats` attribute (`data.py`) — deleted.
- Dead `obs = None` init (`eval_libero.py`) — deleted.
- Dead `render_size` kwarg in `make_env` (`eval_libero.py`) — inlined.
- Redundant outer `np.clip` in `_to_bins` (`model.py`) — already guaranteed by inner clip.
- `self.range` shadowing builtin (`model.py`) — renamed to `self.span`.
- Manual ones-cat attention mask (`model.py`) → `F.pad(instruction_mask, (N_img, T_act), value=1)`.
- `trainable_params(model)` rebuilt every step in `clip_grad_norm_` (`train.py`) — hoisted once before the loop.
- Stringly-typed `"wrist" in samples[0]` in `Collator` (`data.py`) — replaced with explicit `use_wrist_camera` kwarg.
- Duplicate `import wandb` inside the log block (`train.py`) — dropped.

### Explicitly skipped (decision: clarity > line savings)

- **`forward()` returning `(loss, action_logits)`** for the train accuracy metric. Computing accuracy inside `forward()` would couple training-only diagnostics to the model code. Mild leaky abstraction, intentional.
- **Manual mean/std → `x*2-1`** in `preprocess_image`. SigLIP uses 0.5/0.5/0.5 so the math collapses, but the explicit constants pedagogically signal "this is SigLIP normalization". Clarity wins.
- **Prompt-format-and-tokenize duplicated between `predict` and `Collator`**. 3 lines of overlap; the contexts differ slightly (single string vs list); extracting a helper buys < 5 lines and adds an indirection.
- **Render at 224 directly in eval to skip the resize**. Would diverge from the 128 → 224 pixel pipeline used at training time. Train/eval consistency wins.

## 5. Untested paths

Nothing has been run yet. Before serious training time:

- [ ] `python convert_libero.py --src ... --out ... --max-demos-per-task 1` against actual LIBERO files (validates HDF5 keys + instruction parsing).
- [ ] `python eval_libero.py --policy random --num-trials 1` against actual LIBERO sim (validates `make_env`, `obs_to_images` keys, `rollout` loop).
- [ ] `python train.py --steps 100 --log-every 10` for a smoke run (validates the full forward/backward loop, action accuracy starts climbing).
- [ ] DDP path is guarded but **untested**: `torchrun --nproc-per-node=2 train.py --steps 100`.

## 6. Future / v2

Documented in `README.md#v2-roadmap`:

- FAST tokenization (replace per-dim binning with DCT + BPE)
- Multi-camera fusion (currently uncoupled — concat of patches)
- Proprioception input
- Residual VQ action tokens
- Flow-matching action head (π0-style)
- KV cache in `predict` for low-latency real-robot deploy (~5× faster inference)

Each is a single-axis swap that should fit in a small PR.
