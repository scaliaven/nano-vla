"""Diagnostic: replay training samples through the policy.

When LIBERO eval reports near-zero success but training metrics looked
fine, the failure is in one of three places. This script localizes it by
running the SAME training samples through both code paths and comparing:

  * TRAIN path: model.forward() — what train.py's `act_acc` measures.
  * EVAL  path: model.predict() — what eval_libero.py actually runs.

nanoVLA decodes an action chunk in ONE parallel forward — there is no
autoregressive action loop and no teacher forcing (forward() fills action
positions with learned query embeddings, never ground-truth tokens). So the
two paths SHOULD agree token-for-token on the same input; the diagnostic
value is in the cases where they don't, and in the absolute accuracy level:

  * TRAIN good, EVAL good  -> model is fine. The eval failure is in the
                             rollout CONTRACT, not the model: sim->image
                             preprocessing, gripper rescale, or open-loop
                             chunk length. Try --exec-chunk-len 1 in
                             eval_libero.py, then --save-video.
  * TRAIN good, EVAL bad   -> the eval code path diverged from the train
                             code path: predict() does something forward()
                             doesn't — tokenization, padding, attention
                             mask, or image preprocessing. Diff the two.
  * TRAIN bad,  EVAL bad   -> the model is genuinely undertrained (or the
                             wrong ckpt was loaded, or the data the model
                             trained on was preprocessed wrong). Don't
                             blame the eval.

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
def train_path_predict(policy, sample, device):
    """Run the training forward() on one sample and argmax its action logits.

    This is the TRAIN code path — identical to what train.py does to compute
    `act_acc`. It is NOT teacher forcing: nanoVLA's forward() fills action
    positions with learned query embeddings, never ground-truth tokens. The
    action_token_ids are passed only because forward() computes a CE loss
    against them (which we discard); the prediction is argmax(action_logits).

    Returns (pred_tokens (T_act,), pred_actions (chunk_size, action_dim)).
    """
    primary = torch.from_numpy(sample["primary"][None]).to(device)
    actions_np = sample["actions"][None]                                   # (1, K, A)
    action_token_ids = torch.from_numpy(
        policy.action_tokenizer.encode(actions_np).reshape(1, -1)
    ).to(device)

    # Match the training Collator: left-pad to max_instruction_tokens. The
    # tokenizer is already padding_side="left" from NanoVLA.__init__.
    prompt = policy.config.prompt_template.format(
        instruction=sample["instruction"].strip())
    tok = policy.tokenizer(
        prompt, return_tensors="pt", truncation=True,
        padding="max_length",
        max_length=policy.config.max_instruction_tokens,
    )
    batch = {
        "primary": primary,
        "instruction_ids": tok.input_ids.to(device),
        "instruction_mask": tok.attention_mask.to(device),
        "action_token_ids": action_token_ids,   # forward() needs it for the loss
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
    """Per-code-path accumulator."""

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


def _fmt_row(label, train_v, eval_v, fmt="{:.3f}"):
    return (f"  {label:<22} | TRAIN {fmt.format(train_v):<8} "
            f"| EVAL {fmt.format(eval_v):<8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instruction-mode", choices=["real", "empty", "scramble"],
                    default="real",
                    help="real: ground-truth instruction. empty: ''. scramble: "
                         "swap each sample's instruction for a random DIFFERENT "
                         "one from the dataset. Gap between modes = how much "
                         "language signal the policy actually uses.")
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
    train_stats = Stats(A)
    eval_stats = Stats(A)

    # For --instruction-mode scramble we need the set of unique instructions in
    # the dataset to draw a *different* one for each sample. MultiDataset hides
    # this behind .datasets; otherwise read episode_meta directly.
    constituents = getattr(ds, "datasets", [ds])
    instr_pool = sorted({em["instruction"] for d in constituents
                                            for em in d.episode_meta})

    for i in range(args.num_samples):
        s = ds[0]  # idx ignored — both dataset classes sample (episode, t) internally
        if args.instruction_mode == "empty":
            s["instruction"] = ""
        elif args.instruction_mode == "scramble":
            choices = [x for x in instr_pool if x != s["instruction"]] or instr_pool
            s["instruction"] = random.choice(choices)
        images = {"primary": s["primary"]}
        if policy.config.use_wrist_camera:
            images["wrist"] = s["wrist"]
        gt = s["actions"]                                                  # (K, A)
        gt_tokens = policy.action_tokenizer.encode(gt[None])[0].reshape(-1)

        # EVAL code path: policy.predict(), exactly what eval_libero.py runs.
        eval_actions = policy.predict(images, s["instruction"])            # (K, A)
        eval_tokens = policy.action_tokenizer.encode(eval_actions[None])[0].reshape(-1)
        eval_stats.update(gt, eval_actions, gt_tokens, eval_tokens)

        # TRAIN code path: model.forward(), what train.py act_acc measures.
        train_tokens, train_actions = train_path_predict(policy, s, device)
        train_stats.update(gt, train_actions, gt_tokens, train_tokens)

        if i < 3:
            print(f"\n[sample {i}] {s['instruction']!r}")
            print(f"  gt[0]         : {np.array2string(gt[0], precision=3)}")
            print(f"  TRAIN pred[0] : {np.array2string(train_actions[0], precision=3)}")
            print(f"  EVAL  pred[0] : {np.array2string(eval_actions[0], precision=3)}")

    n = args.num_samples
    q01 = np.asarray(policy.action_tokenizer.q01)
    q99 = np.asarray(policy.action_tokenizer.q99)
    rng = np.maximum(q99 - q01, 1e-6)
    train_s = train_stats.summary(n, rng)
    eval_s = eval_stats.summary(n, rng)

    print(f"\n{'='*72}")
    print(f"replay over {n} training samples (chunk_size={policy.config.chunk_size}, "
          f"instruction-mode={args.instruction_mode})")
    print(f"TRAIN = model.forward() (train-style)   EVAL = model.predict() (eval-style)")
    print(f"both are parallel single-forward decodes — they SHOULD match")
    print(f"{'='*72}")
    print(f"q01: {np.array2string(q01, precision=3)}")
    print(f"q99: {np.array2string(q99, precision=3)}")
    print()
    print(_fmt_row("first-step L1 / range",
                   train_s["rel"].mean(), eval_s["rel"].mean()))
    print(_fmt_row("bin exact-match acc",
                   train_s["bin_acc"], eval_s["bin_acc"], fmt="{:.1%}"))
    print(_fmt_row("bin off-by-<=1 acc",
                   train_s["bin_off1_acc"], eval_s["bin_off1_acc"], fmt="{:.1%}"))
    print()
    print(f"  TRAIN per-dim L1 first step: {np.array2string(train_s['first_l1'], precision=4)}")
    print(f"  EVAL  per-dim L1 first step: {np.array2string(eval_s['first_l1'], precision=4)}")
    print()

    # Verdict: read the TRAIN-vs-EVAL gap, not just one column alone.
    GOOD_BIN = 0.85
    GOOD_REL = 0.03
    UNTRAINED_BIN = 0.40

    train_good = train_s["bin_acc"] > GOOD_BIN and train_s["rel"].mean() < GOOD_REL
    eval_good = eval_s["bin_acc"] > GOOD_BIN and eval_s["rel"].mean() < GOOD_REL
    train_bad = train_s["bin_acc"] < UNTRAINED_BIN

    print("verdict:")
    if train_good and eval_good:
        print("  model is well-trained and both code paths agree. The eval failure is")
        print("  in the rollout CONTRACT, not the model: sim->image preprocessing,")
        print("  gripper rescale, or open-loop chunk length.")
        print("  Try --exec-chunk-len 1 in eval_libero.py, then --save-video.")
    elif train_good and not eval_good:
        print("  CODE-PATH DIVERGENCE: forward() fits the training samples but predict()")
        print("  does not, on the SAME inputs. The eval path does something the train")
        print("  path doesn't — tokenization, padding, attention mask, or image")
        print("  preprocessing. Diff NanoVLA.predict() against NanoVLA.forward().")
    elif train_bad:
        print("  model is essentially untrained on this data — even the train-style")
        print("  forward() bin accuracy is low. Don't blame the eval.")
        print("  Check: was the right ckpt loaded? Did train act_acc actually reach >70%?")
        print("  If both yes, raise --lr (1e-4..5e-4), bigger --batch-size or --grad-accum,")
        print("  and consider --no-freeze-vision.")
    else:
        print("  partial / mixed result — train and eval paths disagree but neither is")
        print("  cleanly good or bad. Re-run with more --num-samples; if it persists,")
        print("  diff predict() against forward() and check the ckpt.")


if __name__ == "__main__":
    main()
