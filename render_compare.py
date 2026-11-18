#
# Scan every training camera for the strongest local disagreement between
# vanilla global sort and StopThePop, then render a slow path through that
# stretch as two videos and a labeled side-by-side.
#

import argparse
import json
import os
import shutil
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from diff_gaussian_rasterization import ExtendedSettings
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from gaussian_renderer import render
from render_video import find_ffmpeg, load_cameras, make_camera, smooth_path
from scene.gaussian_model import GaussianModel
import subprocess


def render_image(c2w, src, width, height, gaussians, pipe, background, settings, timer=None):
    cam = make_camera(c2w, src, width, height, 0)
    if timer is None:
        return render(cam, gaussians, pipe, background, splat_args=settings)["render"]
    start, end = timer
    start.record()
    image = render(cam, gaussians, pipe, background, splat_args=settings)["render"]
    end.record()
    torch.cuda.synchronize()
    return image, start.elapsed_time(end)


def local_score(a, b, kernel=64):
    diff = (a - b).abs().mean(0, keepdim=True).unsqueeze(0)
    pooled = F.avg_pool2d(diff, kernel, stride=kernel // 2)
    return float(pooled.max())


def best_window(scores, steps, length, max_step):
    best_i, best_s = 0, -1.0
    for i in range(0, len(scores) - length + 1):
        if steps[i:i + length - 1].max() > max_step:
            continue
        s = float(scores[i:i + length].mean())
        if s > best_s:
            best_i, best_s = i, s
    return best_i, best_i + length - 1, best_s


def label_panel(image, text, font):
    # image: uint8 HWC RGB
    bar = 42
    canvas = Image.new("RGB", (image.shape[1], image.shape[0] + bar), (0, 0, 0))
    canvas.paste(Image.fromarray(image), (0, bar))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), text, fill=(255, 255, 255), font=font)
    return np.asarray(canvas)


def encode(ffmpeg, frame_dir, fps, out):
    subprocess.check_call([
        ffmpeg, "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out,
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--vanilla", default="configs/vanilla.json")
    parser.add_argument("--stopthepop", default="")
    parser.add_argument("--scan_width", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max_step", type=float, default=1.2)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    args = parser.parse_args()

    cams = load_cameras(os.path.join(args.model_path, "cameras.json"))
    centers = np.stack([c["c2w"][:3, 3] for c in cams])
    cam_steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)

    src0 = cams[0]
    scan_h = int(round(args.scan_width * src0["height"] / src0["width"]))
    scan_h -= scan_h % 2
    out_h = int(round(args.width * src0["height"] / src0["width"]))
    out_h -= out_h % 2

    vanilla = ExtendedSettings.from_json(args.vanilla)
    stp_cfg = args.stopthepop or os.path.join(args.model_path, "config.json")
    stopthepop = ExtendedSettings.from_json(stp_cfg)
    gaussians = GaussianModel(3)
    gaussians.load_ply(os.path.join(args.model_path, "point_cloud", "iteration_30000", "point_cloud.ply"))
    cfg_text = open(os.path.join(args.model_path, "cfg_args")).read()
    white = "white_background=True" in cfg_text
    background = torch.tensor([1, 1, 1] if white else [0, 0, 0], dtype=torch.float32, device="cuda")
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)

    if args.start >= 0 and args.end > args.start:
        start, end = args.start, args.end
        print(f"path cameras {start}-{end} ({cams[start]['name']}..{cams[end]['name']})")
    else:
        scores = np.zeros(len(cams), dtype=np.float64)
        with torch.no_grad():
            for i, cam in enumerate(tqdm(cams, desc="scan")):
                a = render_image(cam["c2w"], cam, args.scan_width, scan_h, gaussians, pipe, background, vanilla)
                b = render_image(cam["c2w"], cam, args.scan_width, scan_h, gaussians, pipe, background, stopthepop)
                scores[i] = local_score(a, b)
        start, end, mean_score = best_window(scores, cam_steps, args.window, args.max_step)
        order = np.argsort(scores)[::-1]
        print("strongest views:")
        for i in order[:8]:
            print(f"  idx {i:3d}  img {cams[i]['name']}  score {scores[i]:.3f}")
        print(f"path cameras {start}-{end} ({cams[start]['name']}..{cams[end]['name']}) mean score {mean_score:.3f}")

    out_dir = args.output_dir or os.path.join(args.model_path, "inference", f"compare_{start:03d}_{end:03d}")
    os.makedirs(out_dir, exist_ok=True)
    if args.start < 0:
        np.save(os.path.join(out_dir, "scores.npy"), scores)

    path = smooth_path(cams, start, end, args.steps)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    dirs = {}
    for name in ("vanilla", "stopthepop", "side"):
        dirs[name] = os.path.join(out_dir, name)
        os.makedirs(dirs[name], exist_ok=True)

    van_ms, stp_ms = [], []
    van_timer = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    stp_timer = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    # One warmup so the first timed frame is not paying for kernel launch.
    with torch.no_grad():
        render_image(path[0][0], path[0][1], args.width, out_h, gaussians, pipe, background, vanilla)
        render_image(path[0][0], path[0][1], args.width, out_h, gaussians, pipe, background, stopthepop)
        torch.cuda.synchronize()
        for i, (c2w, src) in enumerate(tqdm(path, desc="render")):
            van, van_t = render_image(c2w, src, args.width, out_h, gaussians, pipe, background, vanilla, van_timer)
            stp, stp_t = render_image(c2w, src, args.width, out_h, gaussians, pipe, background, stopthepop, stp_timer)
            van_ms.append(van_t)
            stp_ms.append(stp_t)
            van_u8 = (van.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy())
            stp_u8 = (stp.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy())
            diff = (van - stp).abs().mul(8).clamp(0, 1)
            diff_u8 = (diff.mul(255).byte().permute(1, 2, 0).cpu().numpy())
            Image.fromarray(van_u8).save(os.path.join(dirs["vanilla"], f"{i:05d}.png"))
            Image.fromarray(stp_u8).save(os.path.join(dirs["stopthepop"], f"{i:05d}.png"))
            side = np.concatenate([
                label_panel(van_u8, "Vanilla 3DGS (global sort)", font),
                label_panel(stp_u8, "StopThePop (hierarchical sort)", font),
                label_panel(diff_u8, "|vanilla - StopThePop| x 8", font),
            ], axis=1)
            Image.fromarray(side).save(os.path.join(dirs["side"], f"{i:05d}.png"))

    def summarize(samples):
        arr = np.asarray(samples, dtype=np.float64)
        return {
            "frames": int(arr.size),
            "mean_ms": float(arr.mean()),
            "median_ms": float(np.median(arr)),
            "std_ms": float(arr.std()),
            "per_frame_ms": [float(x) for x in arr],
        }

    timing = {
        "width": args.width,
        "height": out_h,
        "cameras": [start, end],
        "images": [cams[start]["name"], cams[end]["name"]],
        "vanilla": summarize(van_ms),
        "stopthepop": summarize(stp_ms),
    }
    timing_path = os.path.join(out_dir, "timing.json")
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)
    print(
        f"inference ms/frame  vanilla {timing['vanilla']['mean_ms']:.2f}  "
        f"stopthepop {timing['stopthepop']['mean_ms']:.2f}  -> {timing_path}"
    )

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        print(f"ffmpeg not found. Frames are in {out_dir}")
        return
    for name in ("vanilla", "stopthepop", "side"):
        encode(ffmpeg, dirs[name], args.fps, os.path.join(out_dir, f"{name}.mp4"))
        shutil.rmtree(dirs[name])
        print(os.path.join(out_dir, f"{name}.mp4"))


if __name__ == "__main__":
    main()
