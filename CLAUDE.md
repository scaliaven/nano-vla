# CLAUDE.md

## Project goal

nanoVLA is a **teaching artifact** — a single-file-readable Vision-Language-Action model in the spirit of nanoGPT. The product is *clarity*, not SOTA. Line-count is a strong constraint: v1 (HDF5 LIBERO) was ~600 functional lines, adding LeRobot v2.x brought it to ~920, adding OFT-style parallel action decoding brought it to ~1050, and tightening that decoding (bidirectional action mask, per-position queries, fixed-length padding) brought it to ~1090; treat ~1100 as the working budget — features that push much past it go in the v2 roadmap (`DETAIL.md#v2-roadmap`). Weigh line-count and readability cost against pedagogical value. Don't introduce abstractions or frameworks (Trainer/Lightning/Accelerate/Hydra) that the task doesn't already require.

## Commands

```bash
# 1. Convert demos -> flat .npz (one-time per dataset).
# LIBERO HDF5:
python convert_libero.py --src /path/to/libero_spatial_no_noops --out data/libero_spatial
# LeRobot v2.x (parquet + mp4):
python convert_libero_lerobot.py --src /path/to/lerobot_root --out data/<name>
# fast smoke test (HDF5):
python convert_libero.py --src ... --out ... --max-demos-per-task 1

# 2. Smoke-test the eval harness without a trained model
python eval_libero.py --policy random --suite libero_spatial --num-trials 2

# 3. Train (single GPU). --data-dir auto-detects flat-npz vs LeRobot layout.
python train.py --data-dir data/libero_spatial --steps 50000 --batch-size 8
# DDP via torchrun (nothing else to configure):
torchrun --nproc-per-node=4 train.py --data-dir data/libero_spatial --batch-size 8
# tiny smoke run:
python train.py --steps 100 --log-every 10

# 4. Evaluate a checkpoint
python eval_libero.py --policy nano-vla --ckpt out/ckpt_last.pt --suite libero_spatial

# 5. Diagnose an eval collapse (eval ~0% but train metrics looked fine).
#    Replays training samples through the policy; --instruction-mode
#    {real,empty,scramble} measures how much language the policy actually uses.
python sanity_replay.py --ckpt out/ckpt_last.pt --data-dir data/libero_spatial
```

Every field of `TrainConfig` (top of `train.py`) is auto-exposed as a CLI flag (hyphenated). Booleans become `--foo`/`--no-foo` via `argparse.BooleanOptionalAction`. There is no separate config file, no test suite, no linter, and no build step.

## Architecture

Six files, two contracts. **Contract 1 — flat `.npz` data format** (both converters produce, `data.TrajectoryDataset` consumes):
```
data/<dataset>/
  episode_NNNNN.npz   # images_primary, images_wrist (uint8), actions (T,7) float32, instruction (str)
  index.json          # {ep_name: {length, instruction, file, source_*}}
  stats.json          # action_q01, action_q99, num_bins, image_size
```
Files are written **uncompressed** (`np.savez`) so the dataloader can `mmap_mode='r'` and page in only one frame per `__getitem__`. Compressed npz would force whole-episode decompression per sample (~10× slower training). To support a non-LIBERO dataset, write a new conversion script that emits this exact layout — nothing in `model.py`/`data.py`/`train.py` is LIBERO-specific. `data.LeRobotTrajectoryDataset` is a second backend that reads a LeRobot v2.x layout (`meta/info.json` + parquet + mp4) directly with no offline conversion; `data.make_dataset(data_dir)` picks the backend by checking for `meta/info.json`. The direct path is slower (per-episode mp4 decode with a per-worker LRU cache) — for serious training, prefer `convert_libero_lerobot.py` and the flat-npz path.

**Contract 2 — policy interface** (between model and eval harness):
```python
policy.chunk_size: int
policy.predict(images: dict, instruction: str) -> np.ndarray  # shape (chunk_size, 7), float32
```
`images` has `"primary"` (and optionally `"wrist"`), each `(image_size, image_size, 3)` uint8. `eval_libero.py` calls `predict()`, executes the chunk **open-loop**, then calls again. `RandomPolicy` and `NanoVLA` both conform.

### Three non-obvious moves in `model.py` (load-bearing — keep in mind when editing)

1. **Vocab-tail action tokens.** Actions are per-dim discretized into 256 bins, and bin `b` for any action dim is encoded as LM token id `vocab_size - 256 + b`. The last 256 tokens of Qwen2.5's tokenizer are byte-fallback / reserved tokens never produced by normal text training, so reusing them is harmless. Consequence: **no new embeddings, no vocab resize, the LM head IS the action head.** Inference's argmax is constrained to the last 256 logits (`logits[..., offset:].argmax(...)` in `NanoVLA.predict`) so even uncalibrated text logits can't outvote action bins.

2. **Parallel action decoding (OFT-style, deliberately not OpenVLA-AR).** Action positions hold **per-position** learned query embeddings (`self.action_query`, one `(1, T_act, hidden_size)` parameter, T_act = `chunk_size * action_dim` = 56 by default). There is **no teacher forcing**: action_token_ids in the batch are used as TARGETS only. All T_act action bins are predicted in a single forward pass. The action region of the attention mask is **bidirectional** (not causal): every action slot attends to every other action slot, plus all of image+prompt — built by `_build_attention_mask()` as a 4D additive mask that overrides Qwen2.5's default causal pattern. Bidirectional in the action region is what makes the chunk a *joint* prediction; with causal + a single shared query, the chunk degenerates into 56 quasi-AR sibling predictions sharing a backbone (slot 0 commits before slot 7 has been "thought about"). `predict()` is one LM call, not a `for _ in range(T_act)` AR loop. To revert to base-OpenVLA AR for comparison: feed `self._embed_tokens(batch["action_token_ids"])` at action positions, drop the bidirectional block in `_build_attention_mask`, and shift the slice to `logits[:, -T_act-1:-1, :]`.

3. **Loss masking via slicing, not `-100` labels.** Sequence layout is `[image_tokens, instruction_tokens, action_query × T_act]` and CE is computed only on the action positions. Instead of building a full-length labels tensor padded with `-100`, `forward()` slices `logits[:, -T_act:, :]` directly — each action-query position's logits ARE the prediction for its own bin (no off-by-one shift, unlike the AR variant). Same math, fewer lines.

4. **Sequence layout.** `[N_img vision tokens] + [T_text instruction tokens (left-padded to max_instruction_tokens)] + [T_act action-query slots]`, with `T_act = chunk_size * action_dim` (default `8 * 7 = 56`). Both `data.py:Collator` and `model.py:NanoVLA.predict` pad the prompt to `max_instruction_tokens` (default 64) with `padding_side="left"` — this is load-bearing: action_query slots sit at RoPE positions `N_img + max_instruction_tokens + j`, and those positions must be identical between train and eval. Padding to longest-in-batch (the previous scheme) makes positions batch-dependent and silently breaks eval (TF bin-acc collapses from ~100% to ~30%; this was the libero_spatial 14% eval-collapse bug). The attention mask is a 4D additive tensor — causal in the [image|prompt] block, bidirectional in the action block — built by `_build_attention_mask()`.

### Train/eval pixel-pipeline invariant

`convert_libero.resize_views` is imported by `eval_libero.obs_to_images` precisely so the **180° rotation (`f[::-1, ::-1]`)** and resize stay byte-for-byte identical between the trajectories the model trained on and the live sim observations it sees at eval. **Do not duplicate this code.** If you change the flip or resize on either side, you silently break the train/eval correspondence and eval success rate collapses while train accuracy looks fine — this is the most likely cause of the symptom "train action_acc is high but eval succeeds 0%". HDF5 key names (`agentview_rgb`, `eye_in_hand_rgb`) and live robosuite obs keys (`agentview_image`, `robot0_eye_in_hand_image`) differ by suffix but contain identical pixels, so the pipeline is reusable as-is. `convert_libero_lerobot.py` and `LeRobotTrajectoryDataset` deliberately **skip the rotation** because LeRobot LIBERO mp4s were *already* published as `raw[::-1, ::-1]`; for a non-LIBERO LeRobot dataset, verify orientation against the live sim before trusting eval numbers. The rotation is image-only — actions in the LIBERO/LeRobot dataset are in robosuite's world/base frame and need no horizontal sign flip (verified against the GR00T LIBERO eval reference: `libero_scripts/utils.py:get_libero_image` does the same `[::-1, ::-1]` and `env.step` receives untransformed actions).

### Sampling and DDP

Both `TrajectoryDataset.__getitem__` and `LeRobotTrajectoryDataset.__getitem__` ignore their index and sample `(episode, t)` internally via length-weighted `random.choices`. The transition `t` is drawn from `range(max(1, T - chunk_size + 1))` so the full action chunk fits inside the episode — training never sees a chunk padded out with post-completion "last action × chunk_size" frames. There is **no `DistributedSampler`** — DDP rank disambiguation comes from `make_worker_init(seed, rank)` in `train.py`, which seeds Python's `random` and `numpy.random` per worker per rank. PyTorch's DataLoader auto-seeds `torch`'s RNG in workers but not `random`, so without this every rank would draw the same sequence. Under DDP with `grad_accum > 1`, `train.py` wraps every micro-batch except the last in `model.no_sync()` so the gradient all-reduce fires once per optimizer step, not once per micro-batch.

### Other things worth knowing

- Vision tower is **frozen by default** (`freeze_vision=True`); `train.py` calls `base.vision.eval()` to keep its LayerNorms in eval mode even though the rest of the model is `.train()`.
- bf16 is on by default (`--bf16`); only the forward is autocast, the optimizer steps in fp32.
- Inference is a single forward pass per chunk (parallel decoding); no KV cache and no AR loop.
- Loss/accuracy are accumulated as GPU tensors and only `.item()`d at the log boundary — adding a per-step `.item()` will reintroduce a CUDA sync every iteration.
- `forward()` returns `(loss, action_logits)` — the second tensor is for the train-time accuracy metric (mild leaky abstraction, see `NEEDS_REVIEW.md` 4).
- The four small LIBERO suites (~500 demo episodes each: spatial/object/goal/10) **overfit well before the default 80k steps** — train `act_acc` hits 1.000 and loss falls to ~1e-3, and `ckpt_last` evals *below* an earlier checkpoint (~step 25k–40k). libero_90 (~4500 episodes) doesn't memorize in that budget and generalizes best. Scale `--steps` with dataset size; don't read `ckpt_last` on a small suite as the model's ceiling. libero_spatial is the most sensitive — its 10 tasks share one instruction template over visually identical bowls, so a memorizing model that leans on vision picks the wrong bowl at eval.

## Reference files

- `README.md` — canonical writeup: architecture, parallel-decoding rationale, file layout, quickstart, and measured per-suite LIBERO results.
- `DETAIL.md` — extended notes: four-axes design-space chart, per-system comparison table, v2 roadmap, expected numbers, debugging notes. Items marked with † are unverified; don't cite without checking the source.
- `NEEDS_REVIEW.md` — open decisions, things to verify on first run (LIBERO API specifics), and changes deliberately not made. Read before structural changes — it documents *why* certain "improvements" were skipped.
- `sanity_replay.py` — diagnostic, run first when eval success is near zero but train metrics looked fine. Replays training samples through the policy and reports per-dim L1 / bin-accuracy; `--instruction-mode {real,empty,scramble}` measures how much language signal the policy actually uses. Not part of the core six-file budget.
- `convert_libero_lerobot.py` — LeRobot v2.x converter (parquet + mp4 → flat .npz); pair with `data.LeRobotTrajectoryDataset` to skip conversion at the cost of training speed.
