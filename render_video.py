#
# Render a smooth path through cameras.json and write an mp4.
# cameras.json stores camera-to-world: rotation is the 3x3, position is the center.
#

import argparse
import json
import os
import subprocess
from argparse import Namespace

import numpy as np
import torch
import torchvision
from diff_gaussian_rasterization import ExtendedSettings
from tqdm import tqdm

from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import focal2fov


def rotation_to_quat(R):
    # Returns w, x, y, z.
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_to_rotation(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def c2w_to_colmap(c2w):
    w2c = np.linalg.inv(c2w)
    return w2c[:3, :3].T, w2c[:3, 3]


def load_cameras(path):
    with open(path) as f:
        raw = json.load(f)
    cams = []
    for entry in raw:
        rot = np.array(entry["rotation"], dtype=np.float64)
        center = np.array(entry["position"], dtype=np.float64)
        c2w = np.eye(4)
        c2w[:3, :3] = rot
        c2w[:3, 3] = center
        cams.append({
            "c2w": c2w,
            "width": int(entry["width"]),
            "height": int(entry["height"]),
            "fx": float(entry["fx"]),
            "fy": float(entry["fy"]),
            "name": str(entry.get("img_name", entry.get("id", len(cams)))),
        })
    return cams


def smooth_path(cams, start, end, steps):
    start = max(0, start)
    end = min(len(cams) - 1, end)
    if end <= start:
        raise ValueError(f"need end > start, got {start}..{end}")
    path = []
    for i in range(start, end):
        a, b = cams[i], cams[i + 1]
        qa = rotation_to_quat(a["c2w"][:3, :3])
        qb = rotation_to_quat(b["c2w"][:3, :3])
        for s in range(steps):
            t = s / float(steps)
            c2w = np.eye(4)
            c2w[:3, :3] = quat_to_rotation(slerp(qa, qb, t))
            c2w[:3, 3] = (1 - t) * a["c2w"][:3, 3] + t * b["c2w"][:3, 3]
            path.append((c2w, a))
    path.append((cams[end]["c2w"], cams[end]))
    return path


def make_camera(c2w, src, width, height, uid):
    R, T = c2w_to_colmap(c2w)
    fovx = focal2fov(src["fx"], src["width"])
    fovy = focal2fov(src["fy"], src["height"])
    image = torch.zeros(3, height, width)
    return Camera(uid, R, T, fovx, fovy, image, None, src["name"], uid)


def find_ffmpeg():
    for candidate in ("ffmpeg",):
        if subprocess.call(["which", candidate], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            return candidate
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Render a smooth camera path to mp4")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--start", type=int, default=0, help="first training camera index")
    parser.add_argument("--end", type=int, default=20, help="last training camera index")
    parser.add_argument("--steps", type=int, default=6, help="frames between each pair of cameras")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--config", default="", help="sort config json; default is the checkpoint config.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cams = load_cameras(os.path.join(args.model_path, "cameras.json"))
    path = smooth_path(cams, args.start, args.end, args.steps)
    src0 = path[0][1]
    height = int(round(args.width * src0["height"] / src0["width"]))
    height -= height % 2
    print(f"{len(cams)} cameras, rendering {len(path)} frames at {args.width}x{height} from {args.start} to {args.end}")

    cfg = args.config or os.path.join(args.model_path, "config.json")
    splat_args = ExtendedSettings.from_json(cfg)
    gaussians = GaussianModel(3)
    ply = os.path.join(args.model_path, "point_cloud", "iteration_30000", "point_cloud.ply")
    gaussians.load_ply(ply)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=args.device)
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)

    out = args.output or os.path.join(
        args.model_path, "inference", f"smooth_{args.start:03d}_{args.end:03d}.mp4"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    frame_dir = os.path.splitext(out)[0] + "_frames"
    os.makedirs(frame_dir, exist_ok=True)

    with torch.no_grad():
        for i, (c2w, src) in enumerate(tqdm(path, desc="render")):
            cam = make_camera(c2w, src, args.width, height, i)
            image = render(cam, gaussians, pipe, background, splat_args=splat_args)["render"]
            torchvision.utils.save_image(image, os.path.join(frame_dir, f"{i:05d}.png"))

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        print(f"ffmpeg not found. Frames are in {frame_dir}")
        return
    subprocess.check_call([
        ffmpeg, "-y", "-framerate", str(args.fps),
        "-i", os.path.join(frame_dir, "%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out,
    ])
    print(out)


if __name__ == "__main__":
    main()
