"""nanoVLA training: raw PyTorch loop, single-GPU by default, DDP via torchrun.

Every field of TrainConfig is auto-exposed as a `--field-name` CLI flag;
booleans become `--foo` / `--no-foo`. DDP-specific code is guarded by
`if world_size > 1`.
"""
import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from data import Collator, make_dataloader, make_dataset
from model import NanoVLA, VLAConfig


@dataclass
class TrainConfig:
    # data
    data_dir: str = "data/libero_spatial"
    use_wrist_camera: bool = False
    # model
    vision_model_id: str = "google/siglip-base-patch16-224"
    lm_model_id: str = "Qwen/Qwen2.5-0.5B"
    chunk_size: int = 8
    num_bins: int = 256
    freeze_vision: bool = True
    # optim
    batch_size: int = 8
    grad_accum: int = 1
    steps: int = 50_000
    lr: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    # logging / ckpt
    out_dir: str = "out"
    log_every: int = 10
    ckpt_every: int = 5_000
    # misc
    num_workers: int = 4
    seed: int = 0
    bf16: bool = True
    wandb: bool = False
    wandb_project: str = "nanovla"


def parse_cli() -> TrainConfig:
    """Build an argparse from the dataclass fields. Booleans get --foo / --no-foo."""
    cfg = TrainConfig()
    ap = argparse.ArgumentParser()
    for name in cfg.__dataclass_fields__:
        v = getattr(cfg, name)
        flag = f"--{name.replace('_', '-')}"
        if isinstance(v, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=v)
        else:
            ap.add_argument(flag, type=type(v), default=v)
    return TrainConfig(**vars(ap.parse_args()))


def setup_ddp() -> tuple[int, int, int]:
    """Returns (rank, world_size, local_rank). Single-process if not under torchrun."""
    if "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cosine_with_warmup(step: int, peak_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def make_worker_init(base_seed: int, rank: int):
    """Rank-aware per-worker seed for Python `random` (DataLoader doesn't seed it)."""
    def _init(worker_id: int):
        seed = (base_seed + rank * 1_000_003 + worker_id) % (2 ** 32)
        random.seed(seed)
        np.random.seed(seed)
    return _init


def main():
    cfg = parse_cli()
    rank, world_size, local_rank = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)

    out_dir = Path(cfg.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    # ---- data ----
    # make_dataset auto-picks flat-npz vs. LeRobot based on data_dir contents.
    # Building the dataset first lets us read action quantiles off it without
    # needing stats.json (the LeRobot path computes them from parquet).
    dataset = make_dataset(cfg.data_dir, chunk_size=cfg.chunk_size,
                           use_wrist_camera=cfg.use_wrist_camera)

    # ---- model ----
    vla_config = VLAConfig(
        vision_model_id=cfg.vision_model_id,
        lm_model_id=cfg.lm_model_id,
        use_wrist_camera=cfg.use_wrist_camera,
        chunk_size=cfg.chunk_size,
        num_bins=cfg.num_bins,
        freeze_vision=cfg.freeze_vision,
    )
    model = NanoVLA(vla_config, action_q01=dataset.action_q01,
                    action_q99=dataset.action_q99).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    base = model.module if world_size > 1 else model
    collator = Collator(
        tokenizer=base.tokenizer,
        action_tokenizer=base.action_tokenizer,
        prompt_template=vla_config.prompt_template,
        max_instruction_tokens=vla_config.max_instruction_tokens,
        use_wrist_camera=cfg.use_wrist_camera,
    )
    # No DistributedSampler: TrajectoryDataset ignores indices and samples
    # internally; rank disambiguation comes from worker_init_fn instead.
    loader = make_dataloader(dataset, collator, cfg.batch_size,
                             num_workers=cfg.num_workers,
                             worker_init_fn=make_worker_init(cfg.seed, rank))

    # ---- optimizer ----
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                            betas=(0.9, 0.95))

    if cfg.wandb and is_main:
        import wandb
        wandb.init(project=cfg.wandb_project, config=asdict(cfg))

    # ---- training loop ----
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    model.train()
    if cfg.freeze_vision:
        base.vision.eval()  # keep vision LayerNorms in eval mode

    # Loss/acc are accumulated as GPU tensors and only .item()'d at log time
    # to avoid CUDA syncs every step.
    step, accum = 0, 0
    loss_sum = torch.zeros((), device=device)
    acc_sum = torch.zeros((), device=device)
    n_log = 0
    t0 = time.time()
    data_iter = iter(loader)
    while step < cfg.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.amp.autocast("cuda", dtype=dtype,
                                enabled=cfg.bf16 and torch.cuda.is_available()):
            loss, action_logits = model(batch)
        (loss / cfg.grad_accum).backward()
        accum += 1
        if accum < cfg.grad_accum:
            continue

        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        for g in opt.param_groups:
            g["lr"] = cosine_with_warmup(step, cfg.lr, cfg.warmup_steps, cfg.steps)
        opt.step()
        opt.zero_grad(set_to_none=True)
        accum = 0
        step += 1

        with torch.no_grad():
            preds = action_logits.argmax(dim=-1)
            acc_sum += (preds == batch["action_token_ids"]).float().mean()
        loss_sum += loss.detach()
        n_log += 1

        if step % cfg.log_every == 0 and is_main:
            dt = time.time() - t0
            lr_now = opt.param_groups[0]["lr"]
            mean_loss = (loss_sum / n_log).item()
            mean_acc = (acc_sum / n_log).item()
            print(f"step {step:>6d} | loss {mean_loss:.4f} "
                  f"| act_acc {mean_acc:.3f} | lr {lr_now:.2e} "
                  f"| step/s {n_log/dt:.2f}")
            if cfg.wandb:
                wandb.log({"loss": mean_loss, "act_acc": mean_acc,
                           "lr": lr_now, "step": step})
            loss_sum.zero_()
            acc_sum.zero_()
            n_log = 0
            t0 = time.time()

        if step % cfg.ckpt_every == 0 and is_main:
            base.save_checkpoint(out_dir / f"ckpt_{step:06d}.pt")
            base.save_checkpoint(out_dir / "ckpt_last.pt")

    if is_main:
        base.save_checkpoint(out_dir / "ckpt_last.pt")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
