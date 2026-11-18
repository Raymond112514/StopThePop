#
# Render vanilla and StopThePop, then crop each frame onto the patch
# where they disagree and write only the side-by-side video.
#

import argparse
import json
import os
import shutil
import subprocess
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import ExtendedSettings
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from gaussian_renderer import render
from render_compare import encode
from render_video import find_ffmpeg, load_cameras, make_camera, smooth_path
from scene.gaussian_model import GaussianModel


def render_image(c2w, src, width, height, gaussians, pipe, background, settings, timer):
    cam = make_camera(c2w, src, width, height, 0)
    start, end = timer
    start.record()
    image = render(cam, gaussians, pipe, background, splat_args=settings)["render"]
    end.record()
    torch.cuda.synchronize()
    return image, start.elapsed_time(end)


def disagreement_center(diff, kernel=96):
    pooled = F.avg_pool2d(diff[None, None], kernel, stride=16)
    idx = int(pooled.view(-1).argmax())
    pw = pooled.shape[-1]
    iy, ix = divmod(idx, pw)
    return iy * 16 + kernel // 2, ix * 16 + kernel // 2, float(pooled.view(-1)[idx])


def crop_resize(image, cy, cx, box, out):
    h, w = image.shape[:2]
    box = min(box, h, w)
    y0 = int(np.clip(cy - box // 2, 0, h - box))
    x0 = int(np.clip(cx - box // 2, 0, w - box))
    patch = image[y0:y0 + box, x0:x0 + box]
    return np.asarray(Image.fromarray(patch).resize((out, out), Image.BICUBIC))


def label(image, text, font):
    bar = 42
    canvas = Image.new("RGB", (image.shape[1], image.shape[0] + bar), (0, 0, 0))
    canvas.paste(Image.fromarray(image), (0, bar))
    ImageDraw.Draw(canvas).text((12, 8), text, fill=(255, 255, 255), font=font)
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--vanilla", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--box", type=int, default=360)
    parser.add_argument("--panel", type=int, default=560)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    cams = load_cameras(os.path.join(args.model_path, "cameras.json"))
    src0 = cams[args.start]
    height = int(round(args.width * src0["height"] / src0["width"]))
    height -= height % 2
    vanilla = ExtendedSettings.from_json(args.vanilla)
    stopthepop = ExtendedSettings.from_json(os.path.join(args.model_path, "config.json"))
    gaussians = GaussianModel(3)
    gaussians.load_ply(os.path.join(args.model_path, "point_cloud", "iteration_30000", "point_cloud.ply"))
    white = "white_background=True" in open(os.path.join(args.model_path, "cfg_args")).read()
    background = torch.tensor([1, 1, 1] if white else [0, 0, 0], dtype=torch.float32, device="cuda")
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    path = smooth_path(cams, args.start, args.end, args.steps)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)

    out_dir = os.path.join(args.model_path, "inference", f"zoom_{args.start:03d}_{args.end:03d}")
    frame_dir = os.path.join(out_dir, "side")
    os.makedirs(frame_dir, exist_ok=True)
    van_ms, stp_ms = [], []
    timers = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)),
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)),
    ]
    cy = cx = None
    with torch.no_grad():
        render_image(path[0][0], path[0][1], args.width, height, gaussians, pipe, background, vanilla, timers[0])
        render_image(path[0][0], path[0][1], args.width, height, gaussians, pipe, background, stopthepop, timers[1])
        torch.cuda.synchronize()
        for i, (c2w, src) in enumerate(tqdm(path, desc="render")):
            van, vt = render_image(c2w, src, args.width, height, gaussians, pipe, background, vanilla, timers[0])
            stp, st = render_image(c2w, src, args.width, height, gaussians, pipe, background, stopthepop, timers[1])
            van_ms.append(vt)
            stp_ms.append(st)
            diff = (van - stp).abs().mean(0)
            py, px, score = disagreement_center(diff)
            cy = py if cy is None else 0.8 * cy + 0.2 * py
            cx = px if cx is None else 0.8 * cx + 0.2 * px
            van_u8 = (van.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy())
            stp_u8 = (stp.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy())
            diff_u8 = (diff.mul(8).clamp(0, 1).mul(255).byte().cpu().numpy())
            diff_u8 = np.repeat(diff_u8[:, :, None], 3, axis=2)
            panels = [
                label(crop_resize(van_u8, cy, cx, args.box, args.panel), "Vanilla 3DGS (global sort)", font),
                label(crop_resize(stp_u8, cy, cx, args.box, args.panel), "StopThePop (hierarchical sort)", font),
                label(crop_resize(diff_u8, cy, cx, args.box, args.panel), "|difference| x 8", font),
            ]
            Image.fromarray(np.concatenate(panels, axis=1)).save(os.path.join(frame_dir, f"{i:05d}.png"))
            if i == 0:
                print(f"crop score {score:.3f}")

    def summarize(samples):
        arr = np.asarray(samples, dtype=np.float64)
        return {"frames": int(arr.size), "mean_ms": float(arr.mean()), "median_ms": float(np.median(arr)), "per_frame_ms": [float(x) for x in arr]}

    timing = {
        "width": args.width, "height": height, "crop": args.box,
        "cameras": [args.start, args.end],
        "images": [cams[args.start]["name"], cams[args.end]["name"]],
        "vanilla": summarize(van_ms), "stopthepop": summarize(stp_ms),
    }
    with open(os.path.join(out_dir, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)
    print(f"inference ms/frame  vanilla {timing['vanilla']['mean_ms']:.2f}  stopthepop {timing['stopthepop']['mean_ms']:.2f}")
    ffmpeg = find_ffmpeg()
    video = os.path.join(out_dir, "side.mp4")
    encode(ffmpeg, frame_dir, args.fps, video)
    shutil.rmtree(frame_dir)
    print(video)


if __name__ == "__main__":
    main()
