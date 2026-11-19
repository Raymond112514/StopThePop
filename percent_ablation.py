import json, os, sys, shutil, subprocess, time
ROOT = os.path.dirname(os.path.abspath(__file__))
POP = os.path.join(ROOT, "popping_detection")
CORE = os.path.join(POP, "core")
sys.path.insert(0, POP)
sys.path.insert(0, CORE)
from raft import RAFT
from utils.utils import InputPadder
for name in list(sys.modules):
    if name == "utils" or name.startswith("utils.") or name in {"raft", "corr", "extractor", "update"}:
        del sys.modules[name]
sys.path = [p for p in sys.path if p not in (CORE, POP)]
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from argparse import Namespace
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from diff_gaussian_rasterization import ExtendedSettings
from gaussian_renderer import render
from render_video import find_ffmpeg, load_cameras, make_camera
from scene.gaussian_model import GaussianModel

WIDTH, FPS, N, YAW, PITCH, TILE, PANEL = 960, 24, 64, 12.0, 3.0, 16, 640
PERCENTS = (0.20, 0.40, 0.60, 0.80)
SCENES = [("train", 188), ("playroom", 162)]
pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
BAR = 34
MAGMA = np.array([
    [0.001462, 0.000466, 0.013866],
    [0.232077, 0.059889, 0.437695],
    [0.740075, 0.213610, 0.477581],
    [0.988362, 0.652325, 0.369919],
    [0.987053, 0.991438, 0.749504],
], dtype=np.float32)

def axes(c2w):
    R = c2w[:3, :3]
    return R[:, 0], -R[:, 1], R[:, 2]

def rodrigues(v, axis, ang):
    k = axis / np.linalg.norm(axis)
    return v * np.cos(ang) + np.cross(k, v) * np.sin(ang) + k * np.dot(k, v) * (1 - np.cos(ang))

def look_c2w(eye, look, up):
    forward = look - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w

def subject_dist(xyz, c2w):
    _, _, forward = axes(c2w)
    rel = xyz - c2w[:3, 3]
    depth = rel @ forward
    lateral = np.linalg.norm(rel - depth[:, None] * forward, axis=1)
    mask = (depth > 0.15) & (lateral < depth * np.tan(np.deg2rad(18)))
    if int(mask.sum()) < 50:
        mask = depth > 0.15
    return float(np.quantile(depth[mask], 0.30))

def orbit(c2w, dist):
    right, up, forward = axes(c2w)
    look = c2w[:3, 3] + forward * dist
    poses = []
    for i in range(N):
        t = i / (N - 1)
        s = t * t * t * (t * (t * 6 - 15) + 10)
        offset = rodrigues(rodrigues(-forward * dist, up, np.deg2rad(YAW) * np.sin(2 * np.pi * s)), right, np.deg2rad(PITCH) * np.sin(2 * np.pi * s))
        poses.append(look_c2w(look + offset, look, up))
    return poses

def tile_mean(err):
    h, w = err.shape
    th, tw = h // TILE, w // TILE
    return err[: th * TILE, : tw * TILE].reshape(th, TILE, tw, TILE).mean(axis=(1, 3))

def tile_mean_valid(err, valid):
    h, w = err.shape
    th, tw = h // TILE, w // TILE
    e = err[: th * TILE, : tw * TILE].reshape(th, TILE, tw, TILE)
    v = valid[: th * TILE, : tw * TILE].reshape(th, TILE, tw, TILE)
    count = v.sum(axis=(1, 3))
    total = (e * v).sum(axis=(1, 3))
    out = np.zeros((th, tw), np.float32)
    ok = count > 0
    out[ok] = total[ok] / count[ok]
    return out

def top_tiles(score, frac):
    flat = score.ravel()
    k = max(1, int(round(frac * flat.size)))
    chosen = np.argpartition(flat, -k)[-k:]
    mask = np.zeros(flat.shape, np.bool_)
    mask[chosen] = True
    return mask.reshape(score.shape)

def upsample_mask(tiles, height, width):
    up = np.repeat(np.repeat(tiles.astype(np.float32), TILE, axis=0), TILE, axis=1)
    canvas = np.zeros((height, width), np.float32)
    canvas[: up.shape[0], : up.shape[1]] = up[:height, :width]
    return canvas

def renderer_mask(tiles, height, width):
    ty, tx = (height + TILE - 1) // TILE, (width + TILE - 1) // TILE
    full = np.zeros((ty, tx), np.uint8)
    full[: tiles.shape[0], : tiles.shape[1]] = tiles.astype(np.uint8)
    return torch.from_numpy(full).cuda()

def colorize(field, vmax):
    x = np.clip(field / max(vmax, 1e-8), 0, 1) * (len(MAGMA) - 1)
    i = np.clip(np.floor(x).astype(np.int32), 0, len(MAGMA) - 2)
    f = (x - i)[..., None]
    rgb = MAGMA[i] * (1 - f) + MAGMA[i + 1] * f
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

def mask_image(tiles, height, width):
    up = upsample_mask(tiles, height, width)
    rgb = np.zeros((height, width, 3), np.uint8)
    rgb[up > 0.5] = (255, 196, 64)
    return rgb

def fit(image):
    h, w = image.shape[:2]
    out_h = int(round(PANEL * h / w))
    out_h -= out_h % 2
    return cv2.resize(image, (PANEL, out_h), interpolation=cv2.INTER_AREA)

def label(image, text):
    canvas = Image.new("RGB", (image.shape[1], image.shape[0] + BAR), (0, 0, 0))
    canvas.paste(Image.fromarray(image), (0, BAR))
    ImageDraw.Draw(canvas).text((8, 6), text, fill=(255, 255, 255), font=FONT)
    return np.asarray(canvas)

def flow_back(model, src_u8, dst_u8):
    s = torch.from_numpy(np.ascontiguousarray(src_u8)).permute(2, 0, 1).float()
    d = torch.from_numpy(np.ascontiguousarray(dst_u8)).permute(2, 0, 1).float()
    pad = InputPadder(s.shape)
    sp, dp = pad.pad(s[None].cuda(), d[None].cuda())
    with torch.no_grad():
        _, up = model(dp, sp, iters=20, test_mode=True)
    return pad.unpad(up[0]).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

def warp(src, flow):
    h, w = src.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    absmap = np.stack([xx + flow[..., 0], yy + flow[..., 1]], -1).astype(np.float32)
    warped = cv2.remap(src.astype(np.float32), absmap, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    inside = (absmap[..., 0] >= 0) & (absmap[..., 0] <= w - 1) & (absmap[..., 1] >= 0) & (absmap[..., 1] <= h - 1)
    return warped, inside

def load_raft():
    args = Namespace(small=False, mixed_precision=False)
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(os.path.join(ROOT, "models", "raft-sintel.pth"), map_location="cpu"))
    model.cuda().eval()
    return model.module

def to_u8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def timed_render(cam, g, bg, settings, mask=None):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if mask is None:
        out = render(cam, g, pipe, bg, splat_args=settings)["render"]
    else:
        out = render(cam, g, pipe, bg, splat_args=settings, tile_mask=mask)["render"]
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    return out.clamp(0, 1).detach(), ms

def write_video(frame_dir, video):
    subprocess.check_call([
        find_ffmpeg(), "-y", "-loglevel", "error",
        "-framerate", str(FPS), "-i", os.path.join(frame_dir, "%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", video,
    ])
    shutil.rmtree(frame_dir)

def one_scene(scene, idx, model):
    root = f"pretrained/{scene}"
    src = load_cameras(os.path.join(root, "cameras.json"))[idx]
    white = "white_background=True" in open(os.path.join(root, "cfg_args")).read()
    bg = torch.tensor([1, 1, 1] if white else [0, 0, 0], dtype=torch.float32, device="cuda")
    vanilla = ExtendedSettings.from_json("configs/vanilla.json")
    stop = ExtendedSettings.from_json(os.path.join(root, "config.json"))
    g = GaussianModel(3)
    g.load_ply(os.path.join(root, "point_cloud/iteration_30000/point_cloud.ply"))
    dist = subject_dist(g.get_xyz.detach().cpu().numpy(), src["c2w"])
    poses = orbit(src["c2w"], dist)
    height = int(round(WIDTH * src["height"] / src["width"]))
    height -= height % 2
    cams = [make_camera(c2w, src, WIDTH, height, 0) for c2w in poses]
    with torch.no_grad():
        timed_render(cams[0], g, bg, vanilla)
        timed_render(cams[0], g, bg, stop)
        z = renderer_mask(np.zeros((height // TILE, WIDTH // TILE), np.uint8), height, WIDTH)
        timed_render(cams[0], g, bg, stop, z)
        vans, stps, van_u8 = [], [], []
        van_ms, stp_ms = [], []
        for i, cam in enumerate(cams):
            van, ms_v = timed_render(cam, g, bg, vanilla)
            stp, ms_s = timed_render(cam, g, bg, stop)
            van_np = van.permute(1, 2, 0).float().cpu().numpy()
            stp_np = stp.permute(1, 2, 0).float().cpu().numpy()
            vans.append(van_np.astype(np.float16))
            stps.append(stp_np.astype(np.float16))
            van_u8.append((van.mul(255).byte().permute(1, 2, 0).cpu().numpy()))
            van_ms.append(ms_v)
            stp_ms.append(ms_s)
            if i % 16 == 0:
                print(f"  base {scene} {i}/{N} van {ms_v:.2f} stp {ms_s:.2f}", flush=True)
        flows = []
        for i in range(1, N):
            flows.append(flow_back(model, van_u8[i - 1], van_u8[i]))
            if i % 16 == 0:
                print(f"  flow {scene} {i}/{N}", flush=True)

        err_maps = [np.mean((vans[i].astype(np.float32) - stps[i].astype(np.float32)) ** 2, axis=2) for i in range(N)]
        vmax = float(np.quantile(np.stack(err_maps), 0.99))
        out_dir = os.path.join(root, "inference", f"orbit_{idx:03d}")
        os.makedirs(out_dir, exist_ok=True)
        van_u8_fit = [fit(van_u8[i]) for i in range(N)]
        stp_u8_fit = [fit(to_u8(stps[i].astype(np.float32))) for i in range(N)]
        van_err_fit = [fit(colorize(err_maps[i], vmax)) for i in range(N)]

        percent_rows = []
        for frac in PERCENTS:
            pct = int(round(frac * 100))
            err0 = err_maps[0]
            mask_tiles = top_tiles(tile_mean(err0), frac)
            pixel_mask = upsample_mask(mask_tiles, height, WIDTH)
            hybrids, masks, ad_ms, remain, ad_err = [], [], [], [], []
            for i in range(N):
                if i > 0:
                    warped, inside = warp(pixel_mask, flows[i - 1])
                    score = tile_mean_valid(warped, inside)
                    mask_tiles = score >= 0.5
                    pixel_mask = upsample_mask(mask_tiles, height, WIDTH)
                mask = renderer_mask(mask_tiles.astype(np.uint8), height, WIDTH)
                ad, ms = timed_render(cams[i], g, bg, stop, mask)
                ad_np = ad.permute(1, 2, 0).float().cpu().numpy()
                ad_map = np.mean((ad_np - stps[i].astype(np.float32)) ** 2, axis=2)
                van_m = float(err_maps[i].mean())
                ad_m = float(ad_map.mean())
                hybrids.append(to_u8(ad_np))
                ad_err.append(ad_map.astype(np.float16))
                masks.append(mask_tiles.copy())
                ad_ms.append(ms)
                remain.append(ad_m / max(van_m, 1e-12))
                if i % 16 == 0:
                    print(f"  {scene} {pct}% {i} {ms:.2f} ms tiles {mask_tiles.mean():.3f} remain {remain[-1]:.3f}", flush=True)
            frame_dir = os.path.join(out_dir, f"adaptive_{pct}_frames")
            os.makedirs(frame_dir, exist_ok=True)
            titles = [
                "Vanilla", f"Adaptive {pct}%", "StopThePop",
                "StopThePop tiles", "Adaptive error vs StopThePop", "Vanilla error vs StopThePop",
            ]
            for i in range(N):
                panels = [
                    van_u8_fit[i],
                    fit(hybrids[i]),
                    stp_u8_fit[i],
                    fit(mask_image(masks[i], height, WIDTH)),
                    fit(colorize(ad_err[i].astype(np.float32), vmax)),
                    van_err_fit[i],
                ]
                labeled = [label(p, t) for p, t in zip(panels, titles)]
                frame = np.concatenate([
                    np.concatenate(labeled[:3], axis=1),
                    np.concatenate(labeled[3:], axis=1),
                ], axis=0)
                Image.fromarray(frame).save(os.path.join(frame_dir, f"{i:05d}.png"))
            video = os.path.join(out_dir, f"adaptive_{pct}.mp4")
            write_video(frame_dir, video)
            row = {
                "percent": pct,
                "mean_ms": float(np.mean(ad_ms)),
                "median_ms": float(np.median(ad_ms)),
                "mean_tile_fraction": float(np.mean([m.mean() for m in masks])),
                "mean_error_remaining": float(np.mean(remain)),
                "video": video,
            }
            percent_rows.append(row)
            print(f"{video} {row['mean_ms']:.2f} ms remain {row['mean_error_remaining']:.3f}", flush=True)
            del hybrids, masks
    del g
    torch.cuda.empty_cache()
    return {
        "scene": scene,
        "camera": idx,
        "width": WIDTH,
        "height": height,
        "frames": N,
        "vanilla_mean_ms": float(np.mean(van_ms)),
        "vanilla_median_ms": float(np.median(van_ms)),
        "stopthepop_mean_ms": float(np.mean(stp_ms)),
        "stopthepop_median_ms": float(np.median(stp_ms)),
        "adaptive": percent_rows,
    }

def main():
    model = load_raft()
    rows = []
    for scene, idx in SCENES:
        print("scene", scene, flush=True)
        rows.append(one_scene(scene, idx, model))
        with open("pretrained/percentage_ablation.json", "w") as f:
            json.dump(rows, f, indent=2)
    print("PERCENT_DONE")

if __name__ == "__main__":
    main()
