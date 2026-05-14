# NEEDS_REVIEW

Open items the project owner should look at before / on first run. Grouped by what kind of decision is required. Last updated after the simplify pass.

---

## 1. Decisions I made without explicit go-ahead

These are fine if you agree, but they're not your choices yet.

| # | Where | Decision | Alternative |
|---|---|---|---|
| 1.1 | `model.py` `VLAConfig.prompt_template` | Used OpenVLA's `"In: What action should the robot take to {instruction}? Out:"` | Could use a chat template, a bare instruction, or a custom format. One-line change. |
| 1.2 | `convert_libero.py:resize_views` | **Resolved: 180° rotation** (`f[::-1, ::-1]`). LeRobot LIBERO mp4s were published rotated 180° from raw robosuite, so HDF5 conv + live eval must match. Verified against the GR00T LIBERO eval reference (`libero_scripts/utils.py:get_libero_image`). Old HDF5-converted .npz data needs to be regenerated. | — |
| 1.3 | `train.py:TrainConfig` defaults | bf16 + AdamW + cosine-with-warmup + β=(0.9, 0.95) + lr=2e-5 + warmup 500 + clip 1.0. | OpenVLA-ish defaults; no source of truth. Adjust per your training rig. |
| 1.4 | `convert_libero.py` | **`np.savez` (uncompressed)**, not `np.savez_compressed`. ~22 GB on disk for LIBERO-Spatial vs ~3 GB compressed. | Required so the dataloader can `mmap_mode='r'` and avoid decompressing whole episodes per sample. If disk is the constraint, keep compressed and accept ~10× slower data loading. |
| 1.5 | `README.md` License section | **TBD.** | Pick one before publishing. |

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
- ~~**Prompt-format-and-tokenize duplicated between `predict` and `Collator`**~~ → **Reversed**. The duplication was load-bearing in the wrong direction: `Collator` left-padded with `padding="longest"`, `predict` used no padding at all, and the action_query RoPE positions silently diverged between train and eval. On libero_spatial this dropped TF bin-acc from ~100% to ~30% and was the root cause of the suite's eval collapse. Both paths now left-pad to `max_instruction_tokens` (fixed length). The duplication remains but the schemes are now identical; a future cleanup could extract a helper.
- **Render at 224 directly in eval to skip the resize**. Would diverge from the 128 → 224 pixel pipeline used at training time. Train/eval consistency wins.

## 7. Architecture changes (2026-05): parallel decoding tightening

After diagnosing the libero_spatial eval-collapse via `sanity_replay.py`, three coupled changes landed:

- **Per-position action query embeddings.** `self.action_query` is now `(1, T_act, hidden_size)` (was `(1, 1, hidden_size)` broadcast). Each action slot has its own learned identity in addition to RoPE differentiation. OpenVLA-OFT uses empty/zero embeddings + RoPE only; we add learnables for slightly richer slot identity at ~50K extra params.
- **Bidirectional attention in the action region.** A 4D additive attention mask (`_build_attention_mask`) keeps [image | prompt] causal but makes the action block fully bidirectional. Required for action chunking to be a real joint prediction rather than 56 quasi-AR sibling predictions sharing a backbone.
- **Fixed-length left-padding** in both `data.py:Collator` and `model.py:predict`. Pads to `max_instruction_tokens` (default 64). Action_query RoPE positions are now identical across batches and at eval.

**Old checkpoints will NOT load** — `action_query` shape changed from `(1, 1, H)` to `(1, T_act, H)`. Retrain after pulling these changes. Existing ckpts under the old shape can still be loaded via a small `model.from_checkpoint` shim that broadcasts the old `(1, 1, H)` parameter to `(1, T_act, H)` if backwards-compat with prior runs is needed.

## 5. Untested paths

Nothing has been run yet. Before serious training time:

- [ ] `python convert_libero.py --src ... --out ... --max-demos-per-task 1` against actual LIBERO files (validates HDF5 keys + instruction parsing).
- [ ] `python eval_libero.py --policy random --num-trials 1` against actual LIBERO sim (validates `make_env`, `obs_to_images` keys, `rollout` loop).
- [ ] `python train.py --steps 100 --log-every 10` for a smoke run (validates the full forward/backward loop, action accuracy starts climbing). **Note:** post parallel-decoding, old checkpoints won't load (new `action_query` param + different objective) — train fresh.
- [ ] DDP path is guarded but **untested**: `torchrun --nproc-per-node=2 train.py --steps 100`.

## 6. Future / v2

Documented in `README.md#v2-roadmap`:

- FAST tokenization (replace per-dim binning with DCT + BPE)
- Multi-camera fusion (currently uncoupled — concat of patches)
- Proprioception input
- Residual VQ action tokens
- Flow-matching action head (π0-style)

Each is a single-axis swap that should fit in a small PR.
