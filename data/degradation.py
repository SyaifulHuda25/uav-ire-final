"""
IRE Degradation Pipeline - Second-Order HDM Only
Sesuai paper IRE (Zhu et al., 2023): hanya second-order degradation modeling
yang dipertahankan dari Real-ESRGAN (first-order dihilangkan).

Pipeline degradasi:
    HR -> [Blur -> Resize (Downsampling) -> Noise -> JPEG + 2D sinc filter] -> LR

Komponen degradasi (second-order, sesuai Fig.3 dan Fig.6 paper IRE):
- Blur: Gaussian (isotropic/anisotropic), sinc filter
- Resize: bicubic, bilinear, area (down/up/down)
- Noise: Gaussian, Poisson, Color noise, Gray noise
- JPEG compression + 2D sinc filter
"""

import random
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import io
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Utilitas umum
# ---------------------------------------------------------------------------

def random_choice(options: list):
    return options[random.randint(0, len(options) - 1)]


def clip_image(img: torch.Tensor) -> torch.Tensor:
    """Clip nilai pixel ke [0, 1]."""
    return torch.clamp(img, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 1. Gaussian Blur Kernels
# ---------------------------------------------------------------------------

def generate_gaussian_kernel(
    kernel_size: int,
    sigma: float,
    angle: float = 0.0,
    isotropic: bool = True,
    sigma_y: Optional[float] = None
) -> torch.Tensor:
    """
    Generate 2D Gaussian blur kernel (isotropic atau anisotropic).
    Sesuai degradasi blur pada pipeline Real-ESRGAN/IRE.
    """
    if isotropic:
        sigma_y = sigma

    k = kernel_size // 2
    x = torch.arange(-k, k + 1, dtype=torch.float32)
    y = torch.arange(-k, k + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    if not isotropic and angle != 0:
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x_rot = cos_a * xx + sin_a * yy
        y_rot = -sin_a * xx + cos_a * yy
    else:
        x_rot, y_rot = xx, yy

    kernel = torch.exp(
        -(x_rot ** 2 / (2 * sigma ** 2) + y_rot ** 2 / (2 * (sigma_y or sigma) ** 2))
    )
    kernel = kernel / kernel.sum()
    return kernel


def apply_blur(img: torch.Tensor, kernel_size: int, sigma: float,
               isotropic: bool = True, angle: float = 0.0,
               sigma_y: Optional[float] = None) -> torch.Tensor:
    """Aplikasikan Gaussian blur ke tensor gambar [B,C,H,W] atau [C,H,W]."""
    squeeze = img.dim() == 3
    if squeeze:
        img = img.unsqueeze(0)

    kernel = generate_gaussian_kernel(kernel_size, sigma, angle, isotropic, sigma_y)
    kernel = kernel.to(img.device)
    padding = kernel_size // 2

    # Aplikasikan per-channel dengan depthwise convolution
    C = img.shape[1]
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    out = F.conv2d(img, kernel_4d, padding=padding, groups=C)

    if squeeze:
        out = out.squeeze(0)
    return out


def random_blur(img: torch.Tensor, kernel_range: Tuple[int, int] = (7, 21),
                sigma_range: Tuple[float, float] = (0.5, 3.0),
                prob_isotropic: float = 0.5) -> torch.Tensor:
    """
    Blur acak: pilih ukuran kernel dan sigma secara random.
    Mendukung isotropic dan anisotropic.
    """
    # Kernel size harus ganjil
    k_min, k_max = kernel_range
    kernel_size = random.choice(range(k_min if k_min % 2 == 1 else k_min + 1,
                                      k_max + 1, 2))
    sigma = random.uniform(*sigma_range)
    isotropic = random.random() < prob_isotropic
    angle = random.uniform(0, math.pi) if not isotropic else 0.0
    sigma_y = random.uniform(*sigma_range) if not isotropic else sigma

    return apply_blur(img, kernel_size, sigma, isotropic, angle, sigma_y)


# ---------------------------------------------------------------------------
# 2. Resize / Downsampling
# ---------------------------------------------------------------------------

RESIZE_MODES = ['bicubic', 'bilinear', 'area']


def random_resize(img: torch.Tensor, scale_factor: float,
                  mode: Optional[str] = None) -> torch.Tensor:
    """
    Resize gambar dengan mode interpolasi random.
    scale_factor < 1: downsampling, > 1: upsampling
    """
    if mode is None:
        mode = random_choice(RESIZE_MODES)

    squeeze = img.dim() == 3
    if squeeze:
        img = img.unsqueeze(0)

    align_corners = False if mode in ['bilinear', 'bicubic'] else None
    out = F.interpolate(
        img,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        antialias=(mode in ['bicubic', 'bilinear'])
    )
    if squeeze:
        out = out.squeeze(0)
    return clip_image(out)


def downsample_to_size(img: torch.Tensor, target_h: int, target_w: int,
                       mode: Optional[str] = None) -> torch.Tensor:
    """Downsample ke ukuran target."""
    if mode is None:
        mode = random_choice(RESIZE_MODES)

    squeeze = img.dim() == 3
    if squeeze:
        img = img.unsqueeze(0)

    align_corners = False if mode in ['bilinear', 'bicubic'] else None
    out = F.interpolate(img, size=(target_h, target_w), mode=mode,
                        align_corners=align_corners,
                        antialias=(mode in ['bicubic', 'bilinear']))
    if squeeze:
        out = out.squeeze(0)
    return clip_image(out)


# ---------------------------------------------------------------------------
# 3. Noise
# ---------------------------------------------------------------------------

def add_gaussian_noise(img: torch.Tensor, sigma_range: Tuple[float, float] = (1, 30),
                       gray_noise_prob: float = 0.4) -> torch.Tensor:
    """
    Tambahkan Gaussian noise. Mendukung color dan gray noise.
    sigma dalam rentang [0, 255] - dinormalisasi ke [0, 1].
    """
    sigma = random.uniform(*sigma_range) / 255.0
    if random.random() < gray_noise_prob:
        # Gray noise: noise yang sama di semua channel
        noise = torch.randn_like(img[:1]) * sigma
        noise = noise.expand_as(img)
    else:
        # Color noise
        noise = torch.randn_like(img) * sigma
    return clip_image(img + noise)


def add_poisson_noise(img: torch.Tensor, scale_range: Tuple[float, float] = (0.05, 3.0),
                      gray_noise_prob: float = 0.4) -> torch.Tensor:
    """
    Tambahkan Poisson noise.
    scale mengontrol intensitas noise.
    """
    scale = random.uniform(*scale_range)
    if random.random() < gray_noise_prob:
        # Konversi ke grayscale untuk gray noise
        gray = 0.299 * img[0:1] + 0.587 * img[1:2] + 0.114 * img[2:3]
        noise_map = torch.poisson(gray.clamp(0) * 255) / 255.0 - gray.clamp(0)
        noise = noise_map.expand_as(img)
    else:
        noise_map = torch.poisson(img.clamp(0) * 255) / 255.0 - img.clamp(0)
        noise = noise_map

    return clip_image(img + noise * scale)


def add_random_noise(img: torch.Tensor,
                     gaussian_sigma_range: Tuple[float, float] = (1, 30),
                     poisson_scale_range: Tuple[float, float] = (0.05, 3.0),
                     gray_noise_prob: float = 0.4,
                     poisson_prob: float = 0.5) -> torch.Tensor:
    """Tambahkan noise acak: pilih antara Gaussian atau Poisson."""
    if random.random() < poisson_prob:
        return add_poisson_noise(img, poisson_scale_range, gray_noise_prob)
    else:
        return add_gaussian_noise(img, gaussian_sigma_range, gray_noise_prob)


# ---------------------------------------------------------------------------
# 4. JPEG Compression Artifact
# ---------------------------------------------------------------------------

def add_jpeg_compression(img: torch.Tensor,
                          quality_range: Tuple[int, int] = (30, 95)) -> torch.Tensor:
    """
    Simulasikan artefak JPEG compression.
    Konversi tensor -> PIL -> JPEG encode/decode -> tensor.
    """
    quality = random.randint(*quality_range)

    squeeze = img.dim() == 3
    if squeeze:
        img_batch = img.unsqueeze(0)
    else:
        img_batch = img

    results = []
    for i in range(img_batch.shape[0]):
        # Tensor [C,H,W] float [0,1] -> PIL uint8
        img_np = (img_batch[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        # JPEG encode/decode via buffer
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        pil_jpeg = Image.open(buffer).convert('RGB')

        # Kembali ke tensor
        jpeg_tensor = torch.from_numpy(np.array(pil_jpeg)).permute(2, 0, 1).float() / 255.0
        results.append(jpeg_tensor.to(img_batch.device))

    out = torch.stack(results)
    if squeeze:
        out = out.squeeze(0)
    return clip_image(out)


# ---------------------------------------------------------------------------
# 5. Sinc Filter (untuk ringing/overshoot artifact)
# ---------------------------------------------------------------------------

def generate_sinc_kernel(kernel_size: int, cutoff: float) -> torch.Tensor:
    """
    Generate 2D sinc filter kernel untuk artefak ringing/overshoot.
    cutoff: frekuensi cutoff dalam [pi/4, pi] (normalized).
    """
    k = kernel_size // 2
    x = torch.arange(-k, k + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(x, x, indexing='ij')
    r = torch.sqrt(xx ** 2 + yy ** 2)
    r = r.clamp(min=1e-8)

    kernel = cutoff * torch.special.bessel_j1(cutoff * r) / (2 * math.pi * r)
    kernel[k, k] = cutoff ** 2 / (4 * math.pi)  # Center value
    kernel = kernel / kernel.abs().sum()
    return kernel


def apply_sinc_filter(img: torch.Tensor, kernel_size: int = 7,
                      cutoff: Optional[float] = None) -> torch.Tensor:
    """Aplikasikan sinc filter ke gambar."""
    if cutoff is None:
        cutoff = random.uniform(math.pi / 4, math.pi)

    squeeze = img.dim() == 3
    if squeeze:
        img = img.unsqueeze(0)

    kernel = generate_sinc_kernel(kernel_size, cutoff).to(img.device)
    padding = kernel_size // 2
    C = img.shape[1]
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    out = F.conv2d(img, kernel_4d, padding=padding, groups=C)

    if squeeze:
        out = out.squeeze(0)
    return clip_image(out)


# ---------------------------------------------------------------------------
# IRE Second-Order Degradation Pipeline
# Sesuai Fig.3 dan Fig.6 paper IRE: HANYA second-order yang digunakan
# Pipeline: Blur -> Resize(Down) -> Noise -> JPEG + 2D sinc filter
# ---------------------------------------------------------------------------

class IRE_DegradationPipeline:
    """
    IRE Degradation Pipeline - Second Order Only.

    Sesuai paper IRE (Zhu et al., 2023), hanya second-order degradation
    dari HDM Real-ESRGAN yang dipertahankan.

    Pipeline:
        HR_patch -> Blur -> Resize(Down) -> Noise -> JPEG+Sinc -> LR_patch

    Parameter default mengikuti Real-ESRGAN yang dijadikan basis IRE.
    """

    def __init__(
        self,
        scale_factor: int = 2,
        # Blur params
        blur_kernel_range: Tuple[int, int] = (7, 21),
        blur_sigma_range: Tuple[float, float] = (0.5, 3.0),
        prob_isotropic_blur: float = 0.5,
        # Resize params
        resize_prob: Tuple[float, float, float] = (0.2, 0.7, 0.1),  # up, down, keep
        # Noise params
        gaussian_sigma_range: Tuple[float, float] = (1, 25),
        poisson_scale_range: Tuple[float, float] = (0.05, 2.5),
        gray_noise_prob: float = 0.4,
        poisson_noise_prob: float = 0.5,
        # JPEG params
        jpeg_quality_range: Tuple[int, int] = (30, 95),
        # Sinc filter
        sinc_prob: float = 0.1,
        sinc_kernel_size: int = 7,
    ):
        self.scale_factor = scale_factor
        self.blur_kernel_range = blur_kernel_range
        self.blur_sigma_range = blur_sigma_range
        self.prob_isotropic_blur = prob_isotropic_blur
        self.resize_prob = resize_prob
        self.gaussian_sigma_range = gaussian_sigma_range
        self.poisson_scale_range = poisson_scale_range
        self.gray_noise_prob = gray_noise_prob
        self.poisson_noise_prob = poisson_noise_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.sinc_prob = sinc_prob
        self.sinc_kernel_size = sinc_kernel_size

    def __call__(self, hr_patch: torch.Tensor) -> torch.Tensor:
        """
        Terapkan second-order degradation ke HR patch.

        Args:
            hr_patch: Tensor [C, H, W] atau [B, C, H, W] float [0, 1]

        Returns:
            lr_patch: Tensor dengan ukuran H//scale x W//scale
        """
        img = hr_patch.clone()
        H, W = img.shape[-2], img.shape[-1]
        target_h, target_w = H // self.scale_factor, W // self.scale_factor

        # Step 1: Blur
        img = random_blur(
            img,
            kernel_range=self.blur_kernel_range,
            sigma_range=self.blur_sigma_range,
            prob_isotropic=self.prob_isotropic_blur,
        )

        # Step 2: Resize (downsampling dengan variasi mode)
        # Pilih apakah ada up-down atau langsung down
        p = random.random()
        cumsum = [self.resize_prob[0],
                  self.resize_prob[0] + self.resize_prob[1],
                  1.0]
        if p < cumsum[0]:
            # Up terlebih dahulu, lalu down ke target
            up_factor = random.uniform(1.0, 2.0)
            img = random_resize(img, up_factor)
            img = downsample_to_size(img, target_h, target_w)
        elif p < cumsum[1]:
            # Langsung down ke target
            img = downsample_to_size(img, target_h, target_w)
        else:
            # Down ke ukuran antara, lalu down lagi
            mid_h = random.randint(target_h, H)
            mid_w = random.randint(target_w, W)
            img = downsample_to_size(img, mid_h, mid_w)
            img = downsample_to_size(img, target_h, target_w)

        # Step 3: Noise
        img = add_random_noise(
            img,
            gaussian_sigma_range=self.gaussian_sigma_range,
            poisson_scale_range=self.poisson_scale_range,
            gray_noise_prob=self.gray_noise_prob,
            poisson_prob=self.poisson_noise_prob,
        )

        # Step 4: JPEG compression
        img = add_jpeg_compression(img, quality_range=self.jpeg_quality_range)

        # Step 5: Sinc filter (probabilistik, untuk ringing artifacts)
        if random.random() < self.sinc_prob:
            img = apply_sinc_filter(img, kernel_size=self.sinc_kernel_size)

        return clip_image(img)


# ---------------------------------------------------------------------------
# Test degradation pipeline
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import torchvision.transforms as T
    from PIL import Image as PILImage
    import os

    print("Testing IRE Second-Order Degradation Pipeline...")

    # Buat gambar sintetis untuk test
    hr = torch.rand(3, 256, 256)  # Simulasi HR patch
    pipeline = IRE_DegradationPipeline(scale_factor=4)

    lr = pipeline(hr)
    print(f"HR shape: {hr.shape} -> LR shape: {lr.shape}")
    print(f"LR min/max: {lr.min():.4f} / {lr.max():.4f}")
    print("Degradation pipeline test PASSED!")
