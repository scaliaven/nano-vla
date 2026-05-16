"""LIBERO sim rollout harness. Reports success rate.

Policy interface (Contract 2):
    policy.chunk_size: int
    policy.predict(images, instruction) -> (chunk_size, 7) np.float32
The chunk is executed open-loop, then predict() is called again.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from convert_libero import resize_views  # shared so train/eval pixel paths match

import sys, os, pathlib, importlib.util

# GR00T project root (parent of this file's package)
_GR00T_ROOT = pathlib.Path(__file__).resolve().parents[1]
# LIBERO repo root — override with $LIBERO_ROOT if installed elsewhere.
_LIBERO_ROOT = pathlib.Path(os.environ.get("LIBERO_ROOT", "/scratch/hh3043/LIBERO"))

# Prepend so local code wins over site-packages.
for p in (str(_GR00T_ROOT), str(_LIBERO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# DataLoader workers fork before main() runs, so PYTHONPATH is the only
# channel that reaches them — sys.path edits above don't propagate.
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(_GR00T_ROOT), str(_LIBERO_ROOT), os.environ.get("PYTHONPATH", "")]
)

if importlib.util.find_spec("libero") is None:
    raise ModuleNotFoundError(f"'libero' not found on sys.path. Tried: {_LIBERO_ROOT}")

# LIBERO's get_task_init_states() calls torch.load() on a pickled state file;
# torch>=2.6 defaults to weights_only=True and refuses. Force the legacy default.
import torch as _torch
_torch_load_orig = _torch.load
_torch.load = lambda *a, **kw: _torch_load_orig(*a, **{**kw, "weights_only": False})


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
    model = NanoVLA.from_checkpoint(ckpt_path)
    if _torch.cuda.is_available():
        model = model.to("cuda")
    return model


def make_env(task, render_size: int = 256):
    """Render at `render_size` so live pixels match the training dataset.

    LeRobot LIBERO mp4s are 256x256; the original LIBERO HDF5s are 128. Rendering
    at the same source resolution as training avoids a blurry-upsample mismatch.
    """
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    return OffScreenRenderEnv(bddl_file_name=str(bddl),
                              camera_heights=render_size, camera_widths=render_size)


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


def rollout(env, policy, instruction, init_state, max_steps: int, image_size: int,
            gripper_rescale: bool = False, video_frames: list | None = None,
            exec_chunk_len: int | None = None) -> bool:
    env.reset()
    env.set_init_state(init_state)
    # Robosuite needs a few zero-action warmup steps for physics to settle.
    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
    if video_frames is not None:
        # 180° rotation matches what the policy sees and what training data looks like.
        video_frames.append(obs["agentview_image"][::-1, ::-1].copy())

    step = 0
    while step < max_steps:
        chunk = policy.predict(obs_to_images(obs, image_size), instruction)
        if exec_chunk_len is not None:
            chunk = chunk[:exec_chunk_len]
        if gripper_rescale:
            # LeRobot LIBERO: 1=open, 0=close. Sim: -1=open, +1=close. Rescale
            # [0,1] -> [-1,+1] then binarize via sign — matches the GR00T LIBERO
            # eval reference (`libero_scripts/utils.py:normalize_gripper_action`,
            # `binarize=True`). Defensive: dataset gripper is already binary
            # (stats: q01=0, q99=1) so the model emits near-extremes and `sign`
            # is usually a no-op; it only matters on rare uncertain frames.
            chunk[:, -1] = np.sign(1.0 - 2.0 * chunk[:, -1])
        for a in chunk:
            obs, _, done, _ = env.step(np.asarray(a, dtype=np.float32))
            step += 1
            if video_frames is not None:
                video_frames.append(obs["agentview_image"][::-1, ::-1].copy())
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
    # libero_10 / libero_90 are long-horizon; the short suites finish much sooner.
    ap.add_argument("--max-steps", type=int, default=None,
                    help="default: 300 for short suites, 600 for libero_10/libero_90")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--render-size", type=int, default=256,
                    help="sim camera height/width; match training source resolution")
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed; each rollout is seeded from (seed, task_id, trial)")
    ap.add_argument("--out", type=Path, default=Path("eval_results.json"))
    ap.add_argument("--save-video", type=Path, default=None,
                    help="if set, dump per-rollout mp4s here (one file per trial)")
    ap.add_argument("--video-fps", type=int, default=20)
    ap.add_argument("--exec-chunk-len", type=int, default=None,
                    help="execute only first N actions of each predicted chunk before "
                         "re-predicting; default = full chunk_size (open-loop)")
    args = ap.parse_args()

    if args.max_steps is None:
        args.max_steps = {"libero_10": 600, "libero_90": 600}.get(args.suite, 300)

    if args.policy == "random":
        policy = RandomPolicy()
        gripper_rescale = False
    else:
        assert args.ckpt is not None, "--ckpt required for --policy nano-vla"
        policy = load_nano_vla(args.ckpt)
        # LeRobot LIBERO stores gripper in [0,1]; HDF5-converted data uses [-1,1].
        # Detect by sign: any clearly-negative q01 means the [-1,+1] convention.
        # Tolerant of small noise in q01 (e.g. 0.01) on the LeRobot side.
        q01 = policy.action_tokenizer.q01
        gripper_rescale = bool(q01[-1] > -0.1)
        if gripper_rescale:
            print("[eval] gripper rescale ON: model output [0,1] -> sim {-1,+1} (binarized)")

    if args.save_video is not None:
        args.save_video.mkdir(parents=True, exist_ok=True)
        import imageio

    from libero.libero import benchmark
    bench = benchmark.get_benchmark_dict()[args.suite]()

    results = {}
    total_succ, total_trials = 0, 0
    for task_id in range(bench.n_tasks):
        task = bench.get_task(task_id)
        env = make_env(task, render_size=args.render_size)
        init_states = bench.get_task_init_states(task_id)
        n = min(args.num_trials, len(init_states))
        succ = 0
        pbar = tqdm(range(n), desc=f"[{task_id:02d}] {task.language[:40]}")
        for i in pbar:
            # Per-trial deterministic seed: each (task_id, trial) is reproducible
            # independent of --num-trials, task order, or partial reruns. A single
            # global seed instead chains each rollout onto every prior rollout's
            # RNG draws, so changing --num-trials silently shifts later results.
            np.random.seed(
                int(np.random.SeedSequence([args.seed, task_id, i]).generate_state(1)[0])
            )
            frames = [] if args.save_video is not None else None
            ok = rollout(env, policy, task.language, init_states[i],
                         args.max_steps, args.image_size,
                         gripper_rescale=gripper_rescale, video_frames=frames,
                         exec_chunk_len=args.exec_chunk_len)
            succ += int(ok)
            pbar.set_postfix(success=f"{succ}/{i+1}")
            if frames is not None:
                tag = "succ" if ok else "fail"
                vp = args.save_video / f"task{task_id:02d}_trial{i:02d}_{tag}.mp4"
                imageio.mimsave(vp, np.stack(frames), fps=args.video_fps)
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
