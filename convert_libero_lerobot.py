"""Convert a LeRobot v2.x dataset -> the same flat .npz layout convert_libero.py emits.

Layout this script reads (LeRobot v2.x):
    <src>/meta/info.json
    <src>/meta/episodes.jsonl                 # one row per episode, with `length` + `tasks`
    <src>/data/chunk-XXX/episode_NNNNNN.parquet
    <src>/videos/chunk-XXX/<video_key>/episode_NNNNNN.mp4

Layout this script writes (Contract 1 from CLAUDE.md):
    <out>/episode_NNNNN.npz                   # images_primary, images_wrist, actions, instruction
    <out>/index.json
    <out>/stats.json

Why a separate script (vs. convert_libero.py): LeRobot LIBERO mp4s are *already*
vertically un-flipped — applying the [::-1] flip from convert_libero.resize_views
here would put the robot upside-down. We resize-only, and the resulting pixels are
byte-equivalent to what the HDF5 path produces, so eval_libero.obs_to_images
(which still flips live robosuite obs) keeps working unchanged.

Requires `imageio` (with pyav backend) and `pyarrow`. The lerobot conda env has both.

Usage:
    python convert_libero_lerobot.py --src /scratch/hh3043/libero_10 --out data/libero_10
    python convert_libero_lerobot.py --src ... --out ... --max-episodes 4   # smoke test
"""
import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


PRIMARY_KEY = "observation.images.image"
WRIST_KEY = "observation.images.wrist_image"


def episode_paths(src: Path, info: dict, ep_idx: int) -> tuple[Path, Path, Path]:
    """Resolve (parquet, primary mp4, wrist mp4) for one episode using info.json templates."""
    chunk = ep_idx // info["chunks_size"]
    fmt = {"episode_chunk": chunk, "episode_index": ep_idx}
    pq_path = src / info["data_path"].format(**fmt)
    prim = src / info["video_path"].format(video_key=PRIMARY_KEY, **fmt)
    wrist = src / info["video_path"].format(video_key=WRIST_KEY, **fmt)
    return pq_path, prim, wrist


def decode_and_resize(mp4: Path, size: int, expected_T: int) -> np.ndarray:
    """Decode an mp4 sequentially -> (T, size, size, 3) uint8.

    Sequential decode is much cheaper than per-frame seeks (LeRobot encodes with AV1,
    which has sparse keyframes). No vertical flip — LeRobot frames are already upright.
    """
    out = np.empty((expected_T, size, size, 3), dtype=np.uint8)
    for i, frame in enumerate(iio.imiter(mp4)):
        if i >= expected_T:
            break
        if frame.shape[:2] != (size, size):
            frame = np.asarray(Image.fromarray(frame).resize((size, size), Image.BILINEAR))
        out[i] = frame
    if i + 1 != expected_T:
        raise RuntimeError(f"{mp4}: decoded {i+1} frames, expected {expected_T}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="LeRobot dataset root")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="cap total episodes (for smoke tests)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    info = json.loads((args.src / "meta" / "info.json").read_text())

    episodes_meta = []
    with (args.src / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            episodes_meta.append(json.loads(line))
    if args.max_episodes is not None:
        episodes_meta = episodes_meta[: args.max_episodes]

    index, all_actions = {}, []
    for ep_id, em in enumerate(tqdm(episodes_meta, desc="episodes")):
        ep_idx = em["episode_index"]
        instruction = em["tasks"][0]
        pq_path, prim_mp4, wrist_mp4 = episode_paths(args.src, info, ep_idx)

        actions = np.stack(list(pq.read_table(pq_path, columns=["action"])["action"]
                                .to_numpy())).astype(np.float32)
        T = len(actions)
        if T != em["length"]:
            raise RuntimeError(f"length mismatch ep{ep_idx}: parquet={T} meta={em['length']}")

        primary = decode_and_resize(prim_mp4, args.image_size, T)
        wrist = decode_and_resize(wrist_mp4, args.image_size, T)

        ep_name = f"episode_{ep_id:05d}"
        np.savez(
            args.out / f"{ep_name}.npz",
            images_primary=primary,
            images_wrist=wrist,
            actions=actions,
            instruction=np.array(instruction),
        )
        index[ep_name] = {
            "length": int(T),
            "instruction": instruction,
            "file": f"{ep_name}.npz",
            "source_episode_index": int(ep_idx),
        }
        all_actions.append(actions)

    A = np.concatenate(all_actions, axis=0)
    stats = {
        "action_q01": np.quantile(A, 0.01, axis=0).tolist(),
        "action_q99": np.quantile(A, 0.99, axis=0).tolist(),
        "action_dim": int(A.shape[1]),
        "num_bins": 256,
        "num_episodes": len(index),
        "num_steps": int(len(A)),
        "image_size": args.image_size,
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    (args.out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {len(index)} episodes ({len(A)} steps) to {args.out}")


if __name__ == "__main__":
    main()
