"""LIBERO sim rollout harness for nanoVLA. Reports success rate.

Built BEFORE training so action-tokenizer / model-interface bugs surface early.
Run with --policy random against the real sim — it will not succeed at tasks,
but it exercises the full pipeline (env reset, image preprocessing, chunk
execution, success detection).

Policy interface (model.py conforms to this):
    policy.chunk_size: int
    policy.predict(images, instruction) -> np.ndarray of shape (chunk_size, 7)
        images is a dict with at least 'primary' (and optionally 'wrist'),
        each (image_size, image_size, 3) uint8.

Eval executes the chunk OPEN-LOOP, then calls predict() again.

Usage:
    python eval_libero.py --policy random  --suite libero_spatial --num-trials 5
    python eval_libero.py --policy nano-vla --ckpt out/ckpt.pt --suite libero_spatial
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from convert_libero import resize_views  # shared so train/eval pixel paths match


class RandomPolicy:
    """Smoke-test policy. Won't solve tasks; verifies the rollout loop runs."""
    chunk_size = 8

    def predict(self, images, instruction):
        a = np.random.uniform(-0.05, 0.05, size=(self.chunk_size, 7)).astype(np.float32)
        a[:, -1] = np.random.choice([-1.0, 1.0], size=self.chunk_size)
        return a


def load_nano_vla(ckpt_path: Path):
    """Late-imported so --policy random works before model.py exists."""
    from model import NanoVLA
    return NanoVLA.from_checkpoint(ckpt_path)


def make_env(task):
    """LIBERO renders at 128 natively; we resize to image_size to match training."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    return OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)


def obs_to_images(obs, image_size: int):
    """Apply the SAME flip + resize used by convert_libero.py.

    Note the key-name asymmetry: HDF5 demos use 'agentview_rgb' /
    'eye_in_hand_rgb' suffixes, while live robosuite obs use '_image' suffixes.
    Pixels are otherwise identical, so the convert-time pipeline is reused.
    """
    primary = obs["agentview_image"][None]
    wrist = obs["robot0_eye_in_hand_image"][None]
    return {
        "primary": resize_views(primary, image_size)[0],
        "wrist": resize_views(wrist, image_size)[0],
    }


def rollout(env, policy, instruction, init_state, max_steps: int, image_size: int) -> bool:
    env.reset()
    env.set_init_state(init_state)
    # Robosuite needs a few zero-action warmup steps for physics to settle.
    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

    step = 0
    while step < max_steps:
        chunk = policy.predict(obs_to_images(obs, image_size), instruction)
        for a in chunk:
            obs, _, done, _ = env.step(np.asarray(a, dtype=np.float32))
            step += 1
            if done:
                return True
            if step >= max_steps:
                return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["random", "nano-vla"], default="random")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--num-trials", type=int, default=10, help="trials per task")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("eval_results.json"))
    args = ap.parse_args()

    np.random.seed(args.seed)
    if args.policy == "random":
        policy = RandomPolicy()
    else:
        assert args.ckpt is not None, "--ckpt required for --policy nano-vla"
        policy = load_nano_vla(args.ckpt)

    from libero.libero import benchmark
    bench = benchmark.get_benchmark_dict()[args.suite]()

    results = {}
    total_succ, total_trials = 0, 0
    for task_id in range(bench.n_tasks):
        task = bench.get_task(task_id)
        env = make_env(task)
        init_states = bench.get_task_init_states(task_id)
        n = min(args.num_trials, len(init_states))
        succ = 0
        pbar = tqdm(range(n), desc=f"[{task_id:02d}] {task.language[:40]}")
        for i in pbar:
            ok = rollout(env, policy, task.language, init_states[i],
                         args.max_steps, args.image_size)
            succ += int(ok)
            pbar.set_postfix(success=f"{succ}/{i+1}")
        env.close()
        results[task.name] = {"success": succ, "trials": n, "rate": succ / n}
        total_succ += succ
        total_trials += n
        print(f"  task {task_id} ({task.name}): {succ}/{n}")

    overall = total_succ / max(total_trials, 1)
    summary = {
        "suite": args.suite,
        "policy": args.policy,
        "ckpt": str(args.ckpt) if args.ckpt else None,
        "overall_success_rate": overall,
        "per_task": results,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\n{args.suite} {args.policy}: overall success = {overall:.1%} "
          f"({total_succ}/{total_trials})")


if __name__ == "__main__":
    main()
