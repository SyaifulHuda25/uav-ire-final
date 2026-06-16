"""
Visualisasi untuk IRE
- Menyimpan comparison grid (LR | SR | HR)
- Plot loss curves
- Visualisasi degradation pipeline
"""

import os
import math
import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional, Dict


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Konversi tensor [C,H,W] float [0,1] ke numpy [H,W,C] uint8."""
    if t.dim() == 4:
        t = t[0]
    img = t.detach().cpu().float().clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


def save_comparison_grid(
    lr: torch.Tensor,
    sr: torch.Tensor,
    hr: torch.Tensor,
    save_path: str,
    scale_factor: int = 2,
    nrow: int = 4,
    add_labels: bool = True,
):
    """
    Simpan comparison grid: LR (upsampled) | SR | HR.

    Args:
        lr: LR images [B, C, H, W]
        sr: SR images [B, C, H, W]
        hr: HR images [B, C, H, W]
        save_path: Path untuk menyimpan gambar
        scale_factor: Faktor upscaling (untuk resize LR ke ukuran HR)
        nrow: Jumlah gambar per baris
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    B = min(lr.shape[0], 8)  # Maksimum 8 gambar
    lr = lr[:B].cpu().clamp(0, 1)
    sr = sr[:B].cpu().clamp(0, 1)
    hr = hr[:B].cpu().clamp(0, 1)

    # Upsample LR ke ukuran HR untuk perbandingan visual
    _, _, h, w = hr.shape
    lr_up = torch.nn.functional.interpolate(
        lr, size=(h, w), mode='nearest'
    )

    # Buat grid: [LR_up, SR, HR] berulang
    imgs = []
    for i in range(B):
        imgs.extend([lr_up[i], sr[i], hr[i]])

    grid = vutils.make_grid(
        torch.stack(imgs),
        nrow=3,  # 3 gambar per baris (LR, SR, HR) per sampel
        padding=4,
        normalize=False,
        pad_value=0.5,
    )

    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    if add_labels:
        # Tambahkan label teks menggunakan matplotlib
        fig, ax = plt.subplots(1, 1, figsize=(grid_np.shape[1] / 100, grid_np.shape[0] / 100 + 0.5))
        ax.imshow(grid_np)
        ax.axis('off')

        # Label kolom
        col_w = grid_np.shape[1] // 3
        for j, label in enumerate(['LR (Bicubic Up)', 'SR (IRE)', 'HR (Ground Truth)']):
            ax.text(
                col_w * j + col_w // 2, -5,
                label,
                ha='center', va='bottom',
                fontsize=8, fontweight='bold',
                transform=ax.get_xaxis_transform()
            )

        plt.tight_layout(pad=0.1)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        Image.fromarray(grid_np).save(save_path)


def plot_loss_curves(
    loss_history: Dict[str, list],
    save_path: str,
    title: str = "IRE Training Loss Curves",
):
    """
    Plot kurva loss selama training.

    Args:
        loss_history: Dictionary berisi list values untuk setiap metrik
        save_path: Path untuk menyimpan plot
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    iterations = loss_history.get('iterations', list(range(len(loss_history.get('g_total', [])))))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Plot 1: Generator Loss Total
    ax = axes[0, 0]
    if 'g_total' in loss_history and loss_history['g_total']:
        ax.plot(iterations[:len(loss_history['g_total'])],
                loss_history['g_total'], 'b-', linewidth=0.8, label='G Total')
    ax.set_title('Generator Total Loss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Generator Loss Komponen
    ax = axes[0, 1]
    colors = {'g_pixel': 'blue', 'g_perceptual': 'green', 'g_adversarial': 'red'}
    labels = {'g_pixel': 'SmoothL1 Pixel', 'g_perceptual': 'Perceptual', 'g_adversarial': 'Adversarial (×0.1)'}
    for key, color in colors.items():
        if key in loss_history and loss_history[key]:
            ax.plot(iterations[:len(loss_history[key])],
                    loss_history[key], color=color, linewidth=0.8, label=labels[key])
    ax.set_title('Generator Loss Components')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Discriminator Loss
    ax = axes[1, 0]
    if 'd_loss' in loss_history and loss_history['d_loss']:
        ax.plot(iterations[:len(loss_history['d_loss'])],
                loss_history['d_loss'], 'r-', linewidth=0.8, label='D Loss (RaGAN)')
    ax.set_title('Discriminator Loss (RaGAN)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: PSNR / SSIM (validation)
    ax = axes[1, 1]
    if 'psnr' in loss_history and loss_history['psnr']:
        ax2 = ax.twinx()
        n_val = len(loss_history['psnr'])
        val_iters = np.linspace(0, iterations[-1] if iterations else n_val, n_val)
        l1, = ax.plot(val_iters, loss_history['psnr'], 'b-o', markersize=4, label='PSNR (dB)')
        if 'ssim' in loss_history and loss_history['ssim']:
            l2, = ax2.plot(val_iters, loss_history['ssim'], 'g-s', markersize=4, label='SSIM')
        ax.set_ylabel('PSNR (dB)', color='b')
        ax2.set_ylabel('SSIM', color='g')
        ax.set_xlabel('Iteration')
        ax.set_title('Validation Metrics')
        lines = [l1]
        if 'ssim' in loss_history and loss_history['ssim']:
            lines.append(l2)
        ax.legend(lines, [l.get_label() for l in lines])
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title('Validation Metrics (not yet available)')
        ax.text(0.5, 0.5, 'Validation data not available',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved: {save_path}")


def visualize_degradation_pipeline(
    hr: torch.Tensor,
    save_path: str,
):
    """
    Visualisasikan tahapan degradation pipeline IRE (second-order).

    Menampilkan: HR -> Blur -> Resize -> Noise -> JPEG -> LR_final
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.degradation import (
        random_blur, downsample_to_size, add_gaussian_noise,
        add_jpeg_compression, IRE_DegradationPipeline
    )

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    if hr.dim() == 4:
        hr = hr[0]

    H, W = hr.shape[-2], hr.shape[-1]
    scale = 4
    target_h, target_w = H // scale, W // scale

    stages = [('HR (Input)', hr.clone())]

    # Stage 1: Blur
    blurred = random_blur(hr, sigma_range=(1.0, 2.0))
    stages.append(('After Blur', blurred))

    # Stage 2: Resize
    resized = downsample_to_size(blurred, target_h, target_w, mode='bicubic')
    stages.append(('After Resize ↓4', resized))

    # Stage 3: Noise
    noisy = add_gaussian_noise(resized, sigma_range=(5, 20))
    stages.append(('After Noise', noisy))

    # Stage 4: JPEG
    jpeg = add_jpeg_compression(noisy, quality_range=(50, 80))
    stages.append(('After JPEG (LR)', jpeg))

    # Visualisasi
    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    fig.suptitle('IRE Second-Order Degradation Pipeline', fontsize=12, fontweight='bold')

    for i, (title, img) in enumerate(stages):
        axes[i].imshow(tensor_to_numpy(img))
        h_disp, w_disp = img.shape[-2], img.shape[-1]
        axes[i].set_title(f'{title}\n({w_disp}×{h_disp})', fontsize=9)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Degradation visualization saved: {save_path}")


def visualize_sr_results(
    lr: torch.Tensor,
    sr: torch.Tensor,
    hr: torch.Tensor,
    save_path: str,
    psnr: Optional[float] = None,
    ssim: Optional[float] = None,
):
    """
    Visualisasikan hasil SR satu gambar dengan perbandingan detail.
    Menampilkan thumbnail keseluruhan + crop detail.
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    if lr.dim() == 4: lr = lr[0]
    if sr.dim() == 4: sr = sr[0]
    if hr.dim() == 4: hr = hr[0]

    lr = lr.cpu().clamp(0, 1)
    sr = sr.cpu().clamp(0, 1)
    hr = hr.cpu().clamp(0, 1)

    H, W = hr.shape[-2], hr.shape[-1]
    lr_up = torch.nn.functional.interpolate(
        lr.unsqueeze(0), size=(H, W), mode='bicubic', antialias=True
    ).squeeze(0).clamp(0, 1)

    # Crop detail (tengah gambar, 1/4 ukuran)
    ch, cw = H // 4, W // 4
    cy, cx = H // 2 - ch // 2, W // 2 - cw // 2

    def crop(img): return img[:, cy:cy+ch, cx:cx+cw]

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.1)

    # Row 1: Full images
    for j, (title, img) in enumerate([
        ('LR (Bicubic ×4)', lr_up),
        ('SR (IRE)', sr),
        ('HR (Ground Truth)', hr),
    ]):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(tensor_to_numpy(img))
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        # Tandai crop area
        rect = plt.Rectangle((cx, cy), cw, ch,
                               linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    # Row 2: Crop detail
    for j, (title, img) in enumerate([
        ('LR Detail (Crop)', lr_up),
        ('SR Detail (Crop)', sr),
        ('HR Detail (Crop)', hr),
    ]):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(tensor_to_numpy(crop(img)))
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    # Tambahkan metrics di judul
    metric_str = ''
    if psnr is not None:
        metric_str += f'PSNR: {psnr:.2f} dB'
    if ssim is not None:
        metric_str += f'  |  SSIM: {ssim:.4f}'

    fig.suptitle(f'IRE Super-Resolution Result  {metric_str}',
                 fontsize=11, fontweight='bold')

    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Testing visualization...")

    hr = torch.rand(1, 3, 256, 256)
    lr = torch.nn.functional.interpolate(hr, scale_factor=0.25, mode='bicubic', antialias=True)
    sr = lr.clone()  # Simulasi SR = LR (untuk test)

    os.makedirs('/tmp/ire_test', exist_ok=True)

    # Test comparison grid
    save_comparison_grid(lr, sr, hr, '/tmp/ire_test/comparison.png')
    print("Comparison grid saved!")

    # Test degradation visualization
    visualize_degradation_pipeline(hr[0], '/tmp/ire_test/degradation.png')
    print("Degradation visualization saved!")

    # Test SR result visualization
    visualize_sr_results(lr[0], sr[0], hr[0], '/tmp/ire_test/sr_result.png', psnr=28.5, ssim=0.85)
    print("SR result visualization saved!")

    print("Visualization test PASSED!")
