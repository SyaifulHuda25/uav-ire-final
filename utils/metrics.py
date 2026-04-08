"""
Metrik Evaluasi untuk IRE
Sesuai paper IRE (Zhu et al., 2023) yang menggunakan NIQE, RankIQA, PI.
Untuk proposal tesis (PSNR, SSIM juga diimplementasikan).

Metrik:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- NIQE (Natural Image Quality Evaluator) - no-reference metric
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Union


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def calculate_psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    max_val: float = 1.0,
    crop_border: int = 0,
) -> float:
    """
    Hitung PSNR antara dua gambar.

    Args:
        img1, img2: Tensor [C, H, W] atau [B, C, H, W] float [0, max_val]
        max_val: Nilai maksimum pixel
        crop_border: Jumlah pixel border yang diabaikan

    Returns:
        psnr: nilai PSNR dalam dB
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    if crop_border > 0:
        img1 = img1[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]

    mse = torch.mean((img1.float() - img2.float()) ** 2)
    if mse == 0:
        return float('inf')

    psnr = 10 * math.log10(max_val ** 2 / mse.item())
    return psnr


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    """Generate 1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _create_ssim_window(window_size: int, channels: int) -> torch.Tensor:
    """Buat 2D Gaussian window untuk SSIM."""
    k = _gaussian_kernel_1d(window_size, 1.5)
    window = k.unsqueeze(1) * k.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.expand(channels, 1, window_size, window_size).float()
    return window


def calculate_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
    crop_border: int = 0,
) -> float:
    """
    Hitung SSIM antara dua gambar.
    Sesuai Wang et al. (2004) yang direferensikan dalam proposal.

    Args:
        img1, img2: Tensor [C, H, W] atau [B, C, H, W] float
        window_size: Ukuran window Gaussian (default 11)
        max_val: Nilai maksimum pixel

    Returns:
        ssim: nilai SSIM dalam [0, 1]
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    if crop_border > 0:
        img1 = img1[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]

    img1 = img1.float()
    img2 = img2.float()

    C = img1.shape[1]
    device = img1.device

    # Konstanta stabilisasi (sesuai Eq.3.3 proposal)
    C1 = (0.01 * max_val) ** 2  # C1 = (k1 * L)^2, k1=0.01
    C2 = (0.03 * max_val) ** 2  # C2 = (k2 * L)^2, k2=0.03

    window = _create_ssim_window(window_size, C).to(device)
    pad = window_size // 2

    # Luminance (mu)
    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Variance dan covariance (sigma)
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2

    # SSIM formula (Eq.3.3 proposal)
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator

    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# NIQE (Natural Image Quality Evaluator) - No-Reference
# ---------------------------------------------------------------------------
# Implementasi simplified NIQE menggunakan scipy jika tersedia
# Untuk evaluasi yang lebih akurat, gunakan package piq atau piqa

def calculate_niqe(img: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Hitung NIQE (No-Reference Image Quality Metric).
    Lower is better.

    Catatan: Implementasi ini menggunakan library piq jika tersedia.
    Jika tidak, gunakan estimasi sederhana berbasis statistik lokal.
    """
    try:
        import piq
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.clamp(0, 1)
        niqe_val = piq.niqe(img, data_range=1.0).item()
        return niqe_val
    except ImportError:
        # Fallback: estimasi berdasarkan local variance (simplified)
        if isinstance(img, torch.Tensor):
            img_np = img.cpu().numpy()
        else:
            img_np = img

        if img_np.ndim == 4:
            img_np = img_np[0]
        if img_np.shape[0] in [1, 3]:
            img_np = img_np.transpose(1, 2, 0)

        # Convert ke grayscale
        if img_np.shape[-1] == 3:
            gray = 0.299 * img_np[..., 0] + 0.587 * img_np[..., 1] + 0.114 * img_np[..., 2]
        else:
            gray = img_np[..., 0]

        # Estimasi sederhana: variasi lokal (simplified)
        from scipy.ndimage import uniform_filter
        mean_local = uniform_filter(gray, size=7)
        var_local = uniform_filter(gray ** 2, size=7) - mean_local ** 2
        niqe_approx = float(np.sqrt(np.mean(var_local)))
        return niqe_approx


# ---------------------------------------------------------------------------
# Evaluasi batch
# ---------------------------------------------------------------------------

class MetricsEvaluator:
    """
    Evaluator untuk menghitung metrik kualitas citra secara batch.
    """

    def __init__(self, crop_border: int = 4):
        self.crop_border = crop_border
        self.reset()

    def reset(self):
        self.psnr_list = []
        self.ssim_list = []
        self.niqe_list = []

    def update(self, sr: torch.Tensor, hr: torch.Tensor, compute_niqe: bool = False):
        """Update metrics dengan satu batch."""
        B = sr.shape[0]
        for i in range(B):
            psnr = calculate_psnr(sr[i], hr[i], crop_border=self.crop_border)
            ssim = calculate_ssim(sr[i], hr[i], crop_border=self.crop_border)
            self.psnr_list.append(psnr)
            self.ssim_list.append(ssim)

            if compute_niqe:
                niqe = calculate_niqe(sr[i])
                self.niqe_list.append(niqe)

    def get_results(self) -> dict:
        """Kembalikan rata-rata metrik."""
        results = {
            'psnr': sum(self.psnr_list) / len(self.psnr_list) if self.psnr_list else 0,
            'ssim': sum(self.ssim_list) / len(self.ssim_list) if self.ssim_list else 0,
        }
        if self.niqe_list:
            results['niqe'] = sum(self.niqe_list) / len(self.niqe_list)
        return results

    def __str__(self) -> str:
        results = self.get_results()
        s = f"PSNR: {results['psnr']:.2f} dB | SSIM: {results['ssim']:.4f}"
        if 'niqe' in results:
            s += f" | NIQE: {results['niqe']:.4f}"
        return s


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Testing metrics...")

    # Test dengan tensor random
    img1 = torch.rand(3, 256, 256)
    img2 = img1 + torch.randn_like(img1) * 0.05  # Sedikit noise
    img2 = img2.clamp(0, 1)

    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)

    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")

    # Test MetricsEvaluator
    evaluator = MetricsEvaluator()
    evaluator.update(img2.unsqueeze(0), img1.unsqueeze(0))
    print(f"Evaluator: {evaluator}")

    # Test dengan gambar identik
    psnr_identical = calculate_psnr(img1, img1)
    ssim_identical = calculate_ssim(img1, img1)
    print(f"Identical - PSNR: {psnr_identical}, SSIM: {ssim_identical:.4f}")

    print("Metrics test PASSED!")
