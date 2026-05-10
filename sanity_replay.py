"""Diagnostic: replay training samples through the policy.

Disambiguates three failure modes when LIBERO eval reports near-zero
success but training metrics looked fine, by comparing two decode modes
on the same samples:

  * teacher-forced (TF):  model sees GROUND-TRUTH past action tokens at
                          every position. Same as `act_acc` in train.py.
  * autoregressive (AR):  model sees its OWN argmax outputs as the prefix.
                          Same as `policy.predict()` used in eval_libero.py.

Reading the table:
  * AR good, TF good        -> model is fine. Failure is in the eval rollout
                               contract (image preprocessing, gripper rescale,
                               open-loop chunking). Try --exec-chunk-len 1.
  * AR bad,  TF good        -> exposure bias / compounding error. Model fits
                               under teacher forcing but its own outputs walk
                               off-distribution. This is the case CLAUDE.md
                               flags as "train acc high, eval 0%".
  * AR bad,  TF bad         -> model is genuinely undertrained (or wrong ckpt
                               loaded, or train/eval preprocessing diverged).
                               Don't blame the eval.

Reuses the same `make_dataset` the trainer uses, so flat-npz, LeRobot-direct,
and MultiDataset paths all work the same.

Usage:
    python sanity_replay.py --ckpt out/ckpt_last.pt --data-dir <same as train>
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch

from data import make_dataset
from model import NanoVLA


def _autocast(device):
    return torch.amp.autocast("cuda", dtype=torch.bfloat16,
                              enabled=device.type == "cuda")


@torch.no_grad()
def teacher_forced_predict(policy, sample, device):
    """One forward() pass with GT action tokens as the prefix at every position.

    Returns (pred_tokens (T_act,), pred_actions (chunk_size, action_dim)).
    Mirrors what train.py's `act_acc` measures.
    """
    primary = torch.from_numpy(sample["primary"][None]).to(device)
    actions_np = sample["actions"][None]                                   # (1, K, A)
    action_token_ids = torch.from_numpy(
        policy.action_tokenizer.encode(actions_np).reshape(1, -1)
    ).to(device)

    prompt = policy.config.prompt_template.format(
        instruction=sample["instruction"].strip())
    tok = policy.tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=policy.config.max_instruction_tokens,
    )
    batch = {
        "primary": primary,
        "instruction_ids": tok.input_ids.to(device),
        "instruction_mask": tok.attention_mask.to(device),
        "action_token_ids": action_token_ids,
    }
    if policy.config.use_wrist_camera:
        batch["wrist"] = torch.from_numpy(sample["wrist"][None]).to(device)

    with _autocast(device):
        _, action_logits = policy(batch)                                   # (1, T_act, V)

    offset = policy.vocab_size - policy.config.num_bins
    # Constrain argmax to the vocab tail, same trick as policy.predict().
    pred_tokens = (action_logits[0, :, offset:].argmax(-1) + offset).cpu().numpy()
    grid = pred_tokens.reshape(policy.config.chunk_size, policy.config.action_dim)
    pred_actions = policy.action_tokenizer.decode(grid).astype(np.float32)
    return pred_tokens, pred_actions


class Stats:
    """Per-decode-mode accumulator."""

    def __init__(self, action_dim: int):
        self.chunk_l1 = np.zeros(action_dim, dtype=np.float64)
        self.first_l1 = np.zeros(action_dim, dtype=np.float64)
        self.bin_correct = 0
        self.bin_off1 = 0
        self.bin_total = 0

    def update(self, gt_actions, pred_actions, gt_tokens, pred_tokens):
        self.chunk_l1 += np.abs(pred_actions - gt_actions).mean(axis=0)
        self.first_l1 += np.abs(pred_actions[0] - gt_actions[0])
        diff = np.abs(gt_tokens.astype(np.int64) - pred_tokens.astype(np.int64))
        self.bin_correct += int((diff == 0).sum())
        self.bin_off1 += int((diff <= 1).sum())
        self.bin_total += diff.size

    def summary(self, n: int, rng: np.ndarray):
        return {
            "chunk_l1": self.chunk_l1 / n,
            "first_l1": self.first_l1 / n,
            "rel": (self.first_l1 / n) / rng,
            "bin_acc": self.bin_correct / self.bin_total,
            "bin_off1_acc": self.bin_off1 / self.bin_total,
        }


def _fmt_row(label, ar_v, tf_v, fmt="{:.3f}"):
    return f"  {label:<22} | AR {fmt.format(ar_v):<8} | TF {fmt.format(tf_v):<8}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy = NanoVLA.from_checkpoint(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    policy.eval()

    ds = make_dataset(args.data_dir,
                      chunk_size=policy.config.chunk_size,
                      use_wrist_camera=policy.config.use_wrist_camera)

    A = policy.config.action_dim
    ar = Stats(A)
    tf = Stats(A)

    for i in range(args.num_samples):
        s = ds[0]  # idx ignored — both dataset classes sample (episode, t) internally
        images = {"primary": s["primary"]}
        if policy.config.use_wrist_camera:
            images["wrist"] = s["wrist"]
        gt = s["actions"]                                                  # (K, A)
        gt_tokens = policy.action_tokenizer.encode(gt[None])[0].reshape(-1)

        # Autoregressive decode (matches eval_libero.py).
        ar_actions = policy.predict(images, s["instruction"])              # (K, A)
        ar_tokens = policy.action_tokenizer.encode(ar_actions[None])[0].reshape(-1)
        ar.update(gt, ar_actions, gt_tokens, ar_tokens)

        # Teacher-forced decode (matches train.py act_acc).
        tf_tokens, tf_actions = teacher_forced_predict(policy, s, device)
        tf.update(gt, tf_actions, gt_tokens, tf_tokens)

        if i < 3:
            print(f"\n[sample {i}] {s['instruction']!r}")
            print(f"  gt[0]      : {np.array2string(gt[0], precision=3)}")
            print(f"  AR pred[0] : {np.array2string(ar_actions[0], precision=3)}")
            print(f"  TF pred[0] : {np.array2string(tf_actions[0], precision=3)}")

    n = args.num_samples
    q01 = np.asarray(policy.action_tokenizer.q01)
    q99 = np.asarray(policy.action_tokenizer.q99)
    rng = np.maximum(q99 - q01, 1e-6)
    ar_s = ar.summary(n, rng)
    tf_s = tf.summary(n, rng)

    print(f"\n{'='*72}")
    print(f"replay over {n} training samples (chunk_size={policy.config.chunk_size})")
    print(f"AR = autoregressive (eval-style)   TF = teacher-forced (train-style)")
    print(f"{'='*72}")
    print(f"q01: {np.array2string(q01, precision=3)}")
    print(f"q99: {np.array2string(q99, precision=3)}")
    print()
    print(_fmt_row("first-step L1 / range",
                   ar_s["rel"].mean(), tf_s["rel"].mean()))
    print(_fmt_row("bin exact-match acc",
                   ar_s["bin_acc"], tf_s["bin_acc"], fmt="{:.1%}"))
    print(_fmt_row("bin off-by-<=1 acc",
                   ar_s["bin_off1_acc"], tf_s["bin_off1_acc"], fmt="{:.1%}"))
    print()
    print(f"  AR per-dim L1 first step: {np.array2string(ar_s['first_l1'], precision=4)}")
    print(f"  TF per-dim L1 first step: {np.array2string(tf_s['first_l1'], precision=4)}")
    print()

    # Verdict: read the AR-vs-TF gap, not just AR alone.
    GOOD_BIN = 0.85
    GOOD_REL = 0.03
    UNTRAINED_BIN = 0.40

    ar_good = ar_s["bin_acc"] > GOOD_BIN and ar_s["rel"].mean() < GOOD_REL
    tf_good = tf_s["bin_acc"] > GOOD_BIN and tf_s["rel"].mean() < GOOD_REL
    tf_bad = tf_s["bin_acc"] < UNTRAINED_BIN

    print("verdict:")
    if ar_good and tf_good:
        print("  model is well-trained under both modes. Failure is in the eval contract")
        print("  (preprocessing / gripper rescale / open-loop chunking).")
        print("  Try --exec-chunk-len 1 in eval_libero.py, then --save-video.")
    elif tf_good and not ar_good:
        print("  EXPOSURE BIAS: model fits under teacher forcing but its own outputs walk")
        print("  off-distribution under autoregressive decode. This matches the")
        print("  'train acc high, eval 0%' pattern in CLAUDE.md.")
        print("  Mitigations: shorter --chunk-size, scheduled sampling during training,")
        print("  or --exec-chunk-len 1 at eval (re-plan from real obs every step).")
    elif tf_bad:
        print("  model is essentially untrained on this data — even the train-style")
        print("  teacher-forced bin accuracy is low. Don't blame the eval.")
        print("  Check: was the right ckpt loaded? Did train act_acc actually reach >70%?")
        print("  If both yes, raise --lr (1e-4..5e-4), bigger --batch-size or --grad-accum,")
        print("  and consider --no-freeze-vision.")
    else:
        print("  partially trained under teacher forcing, AR worse than TF.")
        print("  Train longer / higher LR, and consider --exec-chunk-len 1 at eval.")


if __name__ == "__main__":
    main()
