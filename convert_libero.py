"""Convert LIBERO HDF5 demos -> flat per-episode .npz + index.json + stats.json.

Expected input: a directory of LIBERO .hdf5 files (e.g. libero_spatial_no_noops),
where each file is one task (one instruction) with multiple demos under data/demo_*.

Output layout (the canonical nanoVLA data format):
    out/episode_00000.npz   keys: images_primary (T,S,S,3) uint8,
                                  images_wrist   (T,S,S,3) uint8,
                                  actions        (T,7)     float32,
                                  instruction    str (0-d numpy array)
    out/index.json          per-episode metadata
    out/stats.json          dataset-wide action q01/q99 for the discretizer

Both camera views are saved; downstream code picks one or both via config.

Usage:
    python convert_libero.py --src /path/to/libero_spatial_no_noops --out data/libero_spatial
"""
import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_instruction(filename: str) -> str:
    """LIBERO HDF5 filenames encode the task; convert to a natural-language instruction.

    Handles both:
      - libero_spatial_no_noops style: 'pick_up_the_black_bowl_..._demo.hdf5'
      - older KITCHEN_SCENE3_turn_on_the_stove_demo.hdf5 style (drops the SCENE prefix)
    """
    name = filename.replace("_demo.hdf5", "").replace(".hdf5", "")
    name = re.sub(r"_SCENE\d+", "", name)
    name = re.sub(r"^[A-Z][A-Z_]+_", "", name)
    return name.replace("_", " ").strip()


def resize_views(frames: np.ndarray, size: int) -> np.ndarray:
    """(T,H,W,3) uint8 LIBERO render -> (T,size,size,3) uint8.

    LIBERO/robosuite renders are flipped vertically (OpenGL origin); we un-flip here so
    saved frames look right-side-up to a human.  eval_libero.py MUST apply the same
    flip when reading sim observations, otherwise train/eval pixels disagree.
    """
    out = np.empty((len(frames), size, size, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        img = Image.fromarray(f[::-1])
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        out[i] = np.asarray(img)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="dir of LIBERO .hdf5 files")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--max-demos-per-task", type=int, default=None,
                    help="cap demos per .hdf5 file (for quick smoke tests)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    hdf5_files = sorted(args.src.glob("*.hdf5"))
    assert hdf5_files, f"no .hdf5 files in {args.src}"

    index, all_actions, ep_id = {}, [], 0
    for fp in tqdm(hdf5_files, desc="files"):
        instruction = parse_instruction(fp.name)
        with h5py.File(fp, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            if args.max_demos_per_task is not None:
                demo_keys = demo_keys[: args.max_demos_per_task]
            for demo in demo_keys:
                g = f["data"][demo]
                actions = g["actions"][:].astype(np.float32)
                primary = resize_views(g["obs/agentview_rgb"][:], args.image_size)
                wrist = resize_views(g["obs/eye_in_hand_rgb"][:], args.image_size)
                assert len(actions) == len(primary) == len(wrist), \
                    f"length mismatch in {fp.name}/{demo}"

                ep_name = f"episode_{ep_id:05d}"
                # Uncompressed (np.savez, not savez_compressed) so the
                # dataloader can use mmap_mode='r' to load only the bytes for
                # one frame per sample. Disk cost: ~20MB per episode.
                np.savez(
                    args.out / f"{ep_name}.npz",
                    images_primary=primary,
                    images_wrist=wrist,
                    actions=actions,
                    instruction=np.array(instruction),
                )
                index[ep_name] = {
                    "length": int(len(actions)),
                    "instruction": instruction,
                    "file": f"{ep_name}.npz",
                    "source_file": fp.name,
                    "source_demo": demo,
                }
                all_actions.append(actions)
                ep_id += 1

    A = np.concatenate(all_actions, axis=0)
    stats = {
        "action_q01": np.quantile(A, 0.01, axis=0).tolist(),
        "action_q99": np.quantile(A, 0.99, axis=0).tolist(),
        "action_dim": int(A.shape[1]),
        "num_bins": 256,
        "num_episodes": ep_id,
        "num_steps": int(len(A)),
        "image_size": args.image_size,
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    (args.out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {ep_id} episodes ({len(A)} steps) to {args.out}")


if __name__ == "__main__":
    main()
