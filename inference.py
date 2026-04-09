"""
UAV-IRE Inference Script
Menjalankan inferensi model pada folder test, menghasilkan:
- Tabel PSNR/SSIM per gambar (Excel)
- Visualisasi SR vs LR vs HR (3 panel + crop detail gulma)
- Ringkasan metrik keseluruhan
"""

import os
import sys
import argparse
import time
import zipfile
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.metrics import calculate_psnr, calculate_ssim, calculate_niqe


# ──────────────────────────────────────────────────────────────
# Helper: load model
# ──────────────────────────────────────────────────────────────

def load_generator(model_path: str, experiment: str = 'full',
                   num_rrdb: int = 23, scale_factor: int = 4,
                   num_features: int = 64,
                   device: Optional[torch.device] = None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Pilih arsitektur sesuai experiment
    full_exps = {'full', 'no_vsd'}          # masih pakai UAVIRE_Generator
    ire_exps  = {'baseline', 'no_nrdb', 'no_mbcm', 'no_ega'}

    if experiment in full_exps:
        from models.uav_ire_generator import UAVIRE_Generator
        gen = UAVIRE_Generator(num_rrdb=num_rrdb,
                               scale_factor=scale_factor,
                               num_features=num_features)
    else:
        from models.network import IRE_Generator
        gen = IRE_Generator(num_rrdb=num_rrdb, scale_factor=scale_factor)

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict):
        for key in ('generator', 'params_ema', 'params', 'state_dict'):
            if key in state:
                state = state[key]
                break
    gen.load_state_dict(state, strict=False)
    gen = gen.to(device).eval()
    return gen, device


# ──────────────────────────────────────────────────────────────
# Helper: tile inference (untuk gambar besar)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_tile(gen, lr: torch.Tensor, tile: int = 256,
               overlap: int = 32, device=None) -> torch.Tensor:
    """Inferensi dengan tiling untuk gambar yang melebihi VRAM."""
    if device is None:
        device = next(gen.parameters()).device
    squeeze = lr.dim() == 3
    if squeeze:
        lr = lr.unsqueeze(0)
    lr = lr.to(device)

    B, C, H, W = lr.shape
    scale = None  # Akan ditentukan dari first tile

    stride = tile - overlap
    out_buf   = None
    count_buf = None

    for top in range(0, H, stride):
        for left in range(0, W, stride):
            b = min(top + tile, H)
            r = min(left + tile, W)
            if b - top < tile:
                top = max(0, b - tile)
            if r - left < tile:
                left = max(0, r - tile)

            tile_lr = lr[:, :, top:b, left:r]
            tile_sr = gen(tile_lr).clamp(0, 1)

            if scale is None:
                scale = tile_sr.shape[-1] // tile_lr.shape[-1]
                out_buf   = torch.zeros(B, C, H * scale, W * scale, device=device)
                count_buf = torch.zeros(B, 1, H * scale, W * scale, device=device)

            ot, ob = top * scale, b * scale
            ol, or_ = left * scale, r * scale
            out_buf  [:, :, ot:ob, ol:or_] += tile_sr
            count_buf[:, :, ot:ob, ol:or_] += 1

    out = (out_buf / count_buf.clamp(min=1)).clamp(0, 1)
    return out.squeeze(0).cpu() if squeeze else out.cpu()


@torch.no_grad()
def infer_single(gen, lr: torch.Tensor, device=None) -> torch.Tensor:
    if device is None:
        device = next(gen.parameters()).device
    squeeze = lr.dim() == 3
    if squeeze:
        lr = lr.unsqueeze(0)
    lr = lr.to(device)
    sr = gen(lr).clamp(0, 1)
    return sr.squeeze(0).cpu() if squeeze else sr.cpu()


# ──────────────────────────────────────────────────────────────
# Visualisasi 3-panel + crop detail
# ──────────────────────────────────────────────────────────────

def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """[C,H,W] float → [H,W,C] uint8"""
    return (t.cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def visualize_result(lr: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor,
                     save_path: str, psnr: float, ssim: float,
                     img_name: str = '',
                     crop_frac: float = 0.25) -> None:
    """
    Simpan visualisasi 3-panel + crop detail area tengah (simulasi area gulma).

    Layout:
    Row 1: [LR (Bicubic Up)] [SR (Model)] [HR (Ground Truth)]
    Row 2: [LR Crop Detail ] [SR Crop   ] [HR Crop          ]
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    # Upsample LR ke ukuran HR untuk perbandingan
    _, H, W = hr.shape
    lr_up = F.interpolate(lr.unsqueeze(0), size=(H, W),
                          mode='bicubic', antialias=True).squeeze(0).clamp(0, 1)

    # Crop area (tengah, fraksi ukuran gambar)
    ch = max(int(H * crop_frac), 32)
    cw = max(int(W * crop_frac), 32)
    cy = H // 2 - ch // 2
    cx = W // 2 - cw // 2

    def crop(t):
        return t[:, cy:cy + ch, cx:cx + cw]

    panels = [
        ('LR (Bicubic ×4)', lr_up),
        (f'SR ({img_name or "Model"})', sr),
        ('HR (Ground Truth)', hr),
    ]

    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.35, wspace=0.06,
                            top=0.88, bottom=0.04)

    for col, (title, img) in enumerate(panels):
        # Row 0: gambar penuh
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(tensor_to_numpy(img))
        ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
        ax.axis('off')
        # Kotak merah menandai crop area
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((cx, cy), cw, ch,
                                linewidth=1.5, edgecolor='red', facecolor='none'))

        # Row 1: crop detail
        ax2 = fig.add_subplot(gs[1, col])
        ax2.imshow(tensor_to_numpy(crop(img)))
        ax2.set_title(f'{title} — Detail', fontsize=9, pad=3)
        ax2.axis('off')

    fig.suptitle(
        f'{Path(save_path).stem}  |  PSNR: {psnr:.2f} dB   SSIM: {ssim:.4f}',
        fontsize=11, fontweight='bold', y=0.96
    )

    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Plot Loss Curve (train vs val per epoch)
# ──────────────────────────────────────────────────────────────

def plot_train_val_loss(history_json_path: str, save_path: str,
                        exp_name: str = '') -> None:
    """
    Plot grafik Train Loss vs Valid Loss per epoch.
    Sesuai permintaan: kedua kurva dalam satu gambar.
    """
    import json
    with open(history_json_path) as f:
        data = json.load(f)

    epochs     = data['epochs']
    train_dict = data.get('train', {})
    val_dict   = data.get('val', {})

    train_loss = train_dict.get('total', [])
    val_loss   = val_dict.get('val_loss', [])
    psnr_vals  = val_dict.get('psnr', [])
    ssim_vals  = val_dict.get('ssim', [])

    # Selaraskan panjang (val mungkin tidak setiap epoch)
    val_epochs = epochs[:len(val_loss)] if val_loss else []
    psnr_epochs = epochs[:len(psnr_vals)] if psnr_vals else []

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Training Progress — {exp_name}', fontsize=13, fontweight='bold')

    # ── Panel 1: Train Loss vs Valid Loss ──────────────────
    ax = axes[0]
    if train_loss:
        ax.plot(epochs[:len(train_loss)], train_loss,
                'b-o', markersize=3, linewidth=1.5, label='Train Loss (G Total)')
    if val_loss:
        ax.plot(val_epochs, val_loss,
                'r-s', markersize=4, linewidth=1.5, label='Valid Loss (SmoothL1)')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Train Loss vs Valid Loss', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)

    # ── Panel 2: PSNR per epoch ────────────────────────────
    ax = axes[1]
    if psnr_vals:
        ax.plot(psnr_epochs, psnr_vals,
                'g-D', markersize=4, linewidth=1.5, label='PSNR (dB)', color='green')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('PSNR (dB)', fontsize=11)
    ax.set_title('Validation PSNR', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)

    # ── Panel 3: SSIM per epoch ────────────────────────────
    ax = axes[2]
    if ssim_vals:
        ax.plot(psnr_epochs, ssim_vals,
                '-^', markersize=4, linewidth=1.5, label='SSIM', color='darkorange')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('SSIM', fontsize=11)
    ax.set_title('Validation SSIM', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Loss curve saved: {save_path}")


# ──────────────────────────────────────────────────────────────
# Inferensi utama
# ──────────────────────────────────────────────────────────────

SUPPORTED_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}


def run_inference(
    model_path: str,
    test_hr_dir: str,
    output_dir: str,
    experiment: str = 'full',
    num_rrdb: int = 23,
    scale_factor: int = 4,
    num_features: int = 64,
    tile_size: Optional[int] = None,
    file_list: Optional[str] = None,    # path ke test_list.txt (filter gambar)
    compute_niqe: bool = False,
    save_visuals: bool = True,
):
    """
    Jalankan inferensi pada semua gambar di test_hr_dir.

    Alur:
    1. Load model
    2. Untuk setiap HR image: buat LR via bicubic → run model → hitung metrik
    3. Simpan visual 3-panel + crop detail
    4. Export tabel Excel
    5. Buat ZIP hasil (tanpa model)
    """
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading model: {model_path}")
    gen, device = load_generator(model_path, experiment, num_rrdb,
                                  scale_factor, num_features, device)
    n_params = sum(p.numel() for p in gen.parameters())
    print(f"Model params: {n_params:,}")

    # Kumpulkan gambar test
    if file_list and os.path.isfile(file_list):
        # Baca dari list file (test_list.txt) — hanya gambar yang terdaftar
        with open(file_list) as f:
            names = [l.strip() for l in f if l.strip()]
        hr_paths = []
        for name in names:
            p = Path(test_hr_dir) / name
            if p.exists():
                hr_paths.append(p)
            else:
                print(f"  [WARN] Skip (tidak ditemukan): {name}")
        print(f"Test images dari {Path(file_list).name}: {len(hr_paths)}")
    else:
        hr_paths = sorted([
            p for p in Path(test_hr_dir).rglob('*')
            if p.suffix.lower() in SUPPORTED_EXT
        ])
    if not hr_paths:
        raise FileNotFoundError(f"Tidak ada gambar di: {test_hr_dir}")
    print(f"Test images: {len(hr_paths)}")

    # ── Loop inferensi ─────────────────────────────────────
    records = []
    total_psnr, total_ssim = 0.0, 0.0

    print("\n" + "─" * 65)
    print(f"{'#':>4}  {'Filename':<30}  {'PSNR(dB)':>9}  {'SSIM':>7}  {'Time(ms)':>9}")
    print("─" * 65)

    for idx, hr_path in enumerate(hr_paths, 1):
        # Muat HR
        hr_pil = Image.open(hr_path).convert('RGB')
        hr = TF.to_tensor(hr_pil)  # [3, H, W]

        # Buat LR via bicubic downsampling
        _, H, W = hr.shape
        lh, lw = H // scale_factor, W // scale_factor
        lr = F.interpolate(hr.unsqueeze(0), size=(lh, lw),
                           mode='bicubic', antialias=True).squeeze(0).clamp(0, 1)

        # Inferensi
        t0 = time.time()
        if tile_size:
            sr = infer_tile(gen, lr, tile=tile_size, device=device)
        else:
            sr = infer_single(gen, lr, device=device)
        elapsed_ms = (time.time() - t0) * 1000

        # Metrik
        psnr_val = calculate_psnr(sr, hr, crop_border=scale_factor)
        ssim_val = calculate_ssim(sr, hr, crop_border=scale_factor)
        niqe_val = calculate_niqe(sr) if compute_niqe else None

        total_psnr += psnr_val
        total_ssim += ssim_val

        record = {
            'index': idx,
            'filename': hr_path.name,
            'width': W, 'height': H,
            'psnr_db': round(psnr_val, 4),
            'ssim': round(ssim_val, 6),
            'time_ms': round(elapsed_ms, 1),
        }
        if niqe_val is not None:
            record['niqe'] = round(niqe_val, 4)
        records.append(record)

        print(f"{idx:>4}  {hr_path.name:<30}  {psnr_val:>9.4f}  "
              f"{ssim_val:>7.4f}  {elapsed_ms:>9.1f}")

        # Simpan SR image
        sr_out = os.path.join(output_dir, f'SR_{hr_path.stem}.png')
        TF.to_pil_image(sr.clamp(0, 1)).save(sr_out)

        # Visualisasi 3-panel + crop
        if save_visuals:
            vis_path = os.path.join(vis_dir, f'{hr_path.stem}_comparison.png')
            visualize_result(lr, sr, hr, vis_path, psnr_val, ssim_val,
                             img_name=experiment)

    # ── Summary ───────────────────────────────────────────
    avg_psnr = total_psnr / len(hr_paths)
    avg_ssim = total_ssim / len(hr_paths)
    print("─" * 65)
    print(f"{'AVERAGE':<36}  {avg_psnr:>9.4f}  {avg_ssim:>7.4f}")
    print("─" * 65)

    summary = {
        'experiment': experiment,
        'model_path': str(model_path),
        'num_test_images': len(hr_paths),
        'avg_psnr_db': round(avg_psnr, 4),
        'avg_ssim': round(avg_ssim, 6),
        'scale_factor': scale_factor,
    }

    # ── Export Excel ──────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame(records)
    df_summary = pd.DataFrame([summary])

    excel_path = os.path.join(output_dir, 'inference_results.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Per Image Metrics', index=False)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

    print(f"\nExcel saved: {excel_path}")

    # ── ZIP hasil (TANPA model .pth) ──────────────────────
    zip_path = os.path.join(output_dir, f'results_{experiment}.zip')
    _create_results_zip(output_dir, zip_path, exclude_ext={'.pth'})
    print(f"ZIP saved:   {zip_path}")

    return summary, records


def _create_results_zip(source_dir: str, zip_path: str,
                         exclude_ext: set = None):
    """Buat ZIP dari output_dir, kecuali file dengan ekstensi tertentu."""
    if exclude_ext is None:
        exclude_ext = set()
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in Path(source_dir).rglob('*'):
            if p.is_file() and p.suffix not in exclude_ext:
                if p == Path(zip_path):   # jangan zip dirinya sendiri
                    continue
                zf.write(p, p.relative_to(source_dir))


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='UAV-IRE Inference')
    p.add_argument('--model',       required=True, help='Path ke generator_final.pth')
    p.add_argument('--test_hr_dir', required=True, help='Folder gambar HR test')
    p.add_argument('--output_dir',  required=True, help='Folder output')
    p.add_argument('--experiment',  default='full',
                   choices=['full','baseline','no_nrdb','no_mbcm','no_ega','no_vsd'])
    p.add_argument('--num_rrdb',    type=int, default=23)
    p.add_argument('--scale',       type=int, default=4)
    p.add_argument('--tile_size',   type=int, default=None,
                   help='Aktifkan tile inference (untuk gambar besar/VRAM kecil)')
    p.add_argument('--niqe',        action='store_true', help='Hitung NIQE (lebih lambat)')
    p.add_argument('--no_visual',   action='store_true', help='Skip simpan visualisasi')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_inference(
        model_path=args.model,
        test_hr_dir=args.test_hr_dir,
        output_dir=args.output_dir,
        experiment=args.experiment,
        num_rrdb=args.num_rrdb,
        scale_factor=args.scale,
        tile_size=args.tile_size,
        compute_niqe=args.niqe,
        save_visuals=not args.no_visual,
    )
