"""
UAV-Specific Degradation Model
Sesuai proposal tesis Section 2.11.5 dan Eq.(2.30)-(2.36)

Memperluas pipeline degradasi IRE (second-order) dengan degradasi khas UAV:
1. Motion Blur Non-uniform (Eq.2.31): rotasi mikro UAV, getaran gimbal
2. Exposure Flicker (Eq.2.32): perubahan intensitas pencahayaan alami
3. Downsampling + UAV Noise (Eq.2.33)(Eq.2.34)
4. Rotasi Kamera Kecil (Eq.2.35): perubahan sudut gimbal
5. Kompresi JPEG (Eq.2.36): artefak penyimpanan/transmisi

Pipeline lengkap (Eq.2.30):
    ILR = C_JPEG(Rθ(((IHR ⊗ k_motion) · α) ↓s + n_UAV))

Degradasi tambahan yang tidak ada di Real-ESRGAN/IRE:
- Haze ringan: kelembaban dan partikel udara area sawah
- Exposure flicker: perubahan pencahayaan akibat awan, matahari, refleksi air
- Rotasi kamera kecil akibat perubahan sudut gimbal
"""

import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import io
from typing import Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.degradation import (
    IRE_DegradationPipeline,
    add_jpeg_compression,
    clip_image,
    RESIZE_MODES,
    random_choice,
    downsample_to_size,
)


# ---------------------------------------------------------------------------
# 1. UAV Motion Blur Non-uniform
# Sesuai Eq.(2.31): I_blur = IHR ⊗ k_motion
# ---------------------------------------------------------------------------

def generate_motion_kernel(
    kernel_size: int,
    angle: float,
    length: int,
) -> torch.Tensor:
    """
    Generate motion blur kernel linear (sesuai degradasi UAV).
    Kernel non-isotropik dengan arah dan panjang acak.

    Sesuai Eq.(2.31): k_motion = kernel non-isotropik acak
    """
    kernel = torch.zeros(kernel_size, kernel_size)
    center = kernel_size // 2

    # Gambar garis blur dengan panjang dan sudut tertentu
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    half_len = min(length // 2, center)
    for i in range(-half_len, half_len + 1):
        x = int(center + round(i * cos_a))
        y = int(center + round(i * sin_a))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    # Fallback jika kernel kosong
    if kernel.sum() == 0:
        kernel[center, center] = 1.0

    kernel = kernel / kernel.sum()
    return kernel


def apply_uav_motion_blur(
    img: torch.Tensor,
    kernel_size_range: Tuple[int, int] = (7, 17),
    angle_range: Tuple[float, float] = (0, math.pi),
    length_range: Tuple[int, int] = (3, 8),
) -> torch.Tensor:
    """
    Terapkan UAV motion blur non-uniform.
    Sesuai Eq.(2.31): I_blur = IHR ⊗ k_motion

    Motion blur UAV bersifat non-uniform karena:
    - Rotasi mikro wahana (berbeda di tiap frame)
    - Getaran gimbal (arah dan amplitudo bervariasi)
    """
    # Pilih kernel size (ganjil)
    k_min, k_max = kernel_size_range
    kernel_size = random.choice(range(
        k_min if k_min % 2 == 1 else k_min + 1,
        k_max + 1, 2
    ))
    angle = random.uniform(*angle_range)
    length = random.randint(*length_range)

    kernel = generate_motion_kernel(kernel_size, angle, length)

    squeeze = img.dim() == 3
    if squeeze:
        img = img.unsqueeze(0)

    kernel = kernel.to(img.device)
    C = img.shape[1]
    padding = kernel_size // 2
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    out = F.conv2d(img, kernel_4d, padding=padding, groups=C)

    if squeeze:
        out = out.squeeze(0)
    return clip_image(out)


# ---------------------------------------------------------------------------
# 2. Exposure Flicker
# Sesuai Eq.(2.32): I_exp = α · I_blur
# ---------------------------------------------------------------------------

def apply_exposure_flicker(
    img: torch.Tensor,
    alpha_range: Tuple[float, float] = (0.7, 1.3),
) -> torch.Tensor:
    """
    Simulasi exposure flicker akibat perubahan pencahayaan.
    Sesuai Eq.(2.32): I_exp = α · I_blur
    α ~ U(α_min, α_max)

    Penyebab pada UAV:
    - Pergerakan awan (bayangan tiba-tiba)
    - Perubahan sudut matahari
    - Refleksi cahaya dari permukaan air sawah
    """
    alpha = random.uniform(*alpha_range)  # α ~ U(α_min, α_max)
    return clip_image(img * alpha)


# ---------------------------------------------------------------------------
# 3. UAV Noise (Gaussian + Poisson)
# Sesuai Eq.(2.33)(2.34): n_UAV = n_Gauss + n_Poisson
# ---------------------------------------------------------------------------

def add_uav_noise(
    img: torch.Tensor,
    gaussian_sigma_range: Tuple[float, float] = (1, 20),
    poisson_scale_range: Tuple[float, float] = (0.05, 2.0),
    gray_noise_prob: float = 0.4,
) -> torch.Tensor:
    """
    Tambahkan noise UAV: campuran Gaussian + Poisson.
    Sesuai Eq.(2.34): n_UAV = n_Gauss + n_Poisson

    Sumber noise pada UAV:
    - n_Gauss: noise sensor kamera, noise elektronik
    - n_Poisson: noise akibat jumlah foton yang terbatas (shot noise)
    """
    # n_Gaussian
    sigma_g = random.uniform(*gaussian_sigma_range) / 255.0
    if random.random() < gray_noise_prob:
        n_gauss = torch.randn_like(img[:1]) * sigma_g
        n_gauss = n_gauss.expand_as(img)
    else:
        n_gauss = torch.randn_like(img) * sigma_g

    # n_Poisson (shot noise)
    scale_p = random.uniform(*poisson_scale_range)
    n_poisson = (torch.poisson(img.clamp(0) * 255) / 255.0 - img.clamp(0)) * scale_p

    return clip_image(img + n_gauss + n_poisson)


# ---------------------------------------------------------------------------
# 4. Rotasi Kamera Kecil
# Sesuai Eq.(2.35): I_rot = R_θ(I_ds)
# θ ~ U(-θ_max, θ_max)
# ---------------------------------------------------------------------------

def apply_small_rotation(
    img: torch.Tensor,
    theta_max: float = 5.0,  # derajat, sesuai "rotasi kecil" pada UAV
) -> torch.Tensor:
    """
    Terapkan rotasi kamera kecil.
    Sesuai Eq.(2.35): I_rot = R_θ(I_ds)
    θ ~ U(-θ_max, θ_max)

    Penyebab: perubahan sudut gimbal selama penerbangan.
    """
    theta = random.uniform(-theta_max, theta_max)  # dalam derajat

    squeeze = img.dim() == 3
    if squeeze:
        img = img.unsqueeze(0)

    # Konversi ke PIL untuk rotasi, lalu kembali
    B = img.shape[0]
    rotated = []
    for i in range(B):
        img_np = (img[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        pil_rot = pil_img.rotate(theta, resample=Image.BILINEAR, expand=False)
        t = torch.from_numpy(np.array(pil_rot)).permute(2, 0, 1).float() / 255.0
        rotated.append(t.to(img.device))

    out = torch.stack(rotated)
    if squeeze:
        out = out.squeeze(0)
    return clip_image(out)


# ---------------------------------------------------------------------------
# 5. Haze Ringan
# Simulasi kelembaban dan partikel udara area persawahan
# ---------------------------------------------------------------------------

def apply_haze(
    img: torch.Tensor,
    haze_intensity_range: Tuple[float, float] = (0.05, 0.2),
    atmospheric_light_range: Tuple[float, float] = (0.8, 1.0),
) -> torch.Tensor:
    """
    Simulasi haze ringan (efek kabut ringan dari kelembaban sawah).
    Model: I_haze = img * t + A * (1 - t)
    di mana t = transmission map, A = atmospheric light.

    Sesuai deskripsi di proposal Section 2.4 (Gambar 2.5).
    """
    beta = random.uniform(*haze_intensity_range)  # Koefisien scattering
    A = random.uniform(*atmospheric_light_range)   # Atmospheric light

    # Transmission map yang uniform (haze ringan)
    t = math.exp(-beta)  # Penyederhanaan: depth uniform

    # Model haze: I = img * t + A * (1 - t)
    hazy = img * t + A * (1 - t)
    return clip_image(hazy)


# ---------------------------------------------------------------------------
# UAV-Specific Degradation Pipeline
# Sesuai Eq.(2.30) proposal
# ---------------------------------------------------------------------------

class UAVSpecificDegradationPipeline:
    """
    UAV-Specific Degradation Model.

    Sesuai proposal Section 2.11.5 dan Eq.(2.30):
    ILR = C_JPEG(Rθ(((IHR ⊗ k_motion) · α) ↓s + n_UAV))

    Memperluas IRE degradation pipeline dengan:
    1. Motion blur non-uniform (sesuai karakteristik UAV)
    2. Exposure flicker
    3. Noise UAV (Gaussian + Poisson)
    4. Downsampling
    5. Rotasi kamera kecil
    6. JPEG compression

    Tambahan opsional:
    - Haze ringan (lingkungan sawah)
    - Sinc filter (ringing artifacts)

    Args:
        scale_factor: Faktor downsampling
        prob_motion_blur: Probabilitas motion blur diterapkan
        prob_exposure: Probabilitas exposure flicker
        prob_haze: Probabilitas haze ringan
        prob_rotation: Probabilitas rotasi kamera kecil
        use_second_order_base: Juga terapkan IRE second-order pipeline
    """

    def __init__(
        self,
        scale_factor: int = 4,
        # Motion blur
        prob_motion_blur: float = 0.8,
        motion_kernel_range: Tuple[int, int] = (7, 15),
        motion_angle_range: Tuple[float, float] = (0, math.pi),
        motion_length_range: Tuple[int, int] = (2, 6),
        # Exposure
        prob_exposure: float = 0.7,
        alpha_range: Tuple[float, float] = (0.75, 1.25),
        # Noise
        gaussian_sigma_range: Tuple[float, float] = (1, 20),
        poisson_scale_range: Tuple[float, float] = (0.05, 2.0),
        gray_noise_prob: float = 0.4,
        # Haze
        prob_haze: float = 0.3,
        haze_intensity_range: Tuple[float, float] = (0.05, 0.15),
        # Rotation
        prob_rotation: float = 0.5,
        theta_max: float = 3.0,
        # JPEG
        jpeg_quality_range: Tuple[int, int] = (40, 90),
        # General
        resize_prob: Tuple[float, float, float] = (0.2, 0.7, 0.1),
    ):
        self.scale_factor = scale_factor

        self.prob_motion_blur = prob_motion_blur
        self.motion_kernel_range = motion_kernel_range
        self.motion_angle_range = motion_angle_range
        self.motion_length_range = motion_length_range

        self.prob_exposure = prob_exposure
        self.alpha_range = alpha_range

        self.gaussian_sigma_range = gaussian_sigma_range
        self.poisson_scale_range = poisson_scale_range
        self.gray_noise_prob = gray_noise_prob

        self.prob_haze = prob_haze
        self.haze_intensity_range = haze_intensity_range

        self.prob_rotation = prob_rotation
        self.theta_max = theta_max

        self.jpeg_quality_range = jpeg_quality_range
        self.resize_prob = resize_prob

    def __call__(self, hr_patch: torch.Tensor) -> torch.Tensor:
        """
        Terapkan UAV-specific degradation pipeline.

        Sesuai Eq.(2.30):
        ILR = C_JPEG(Rθ(((IHR ⊗ k_motion) · α) ↓s + n_UAV))

        Args:
            hr_patch: Tensor [C, H, W] float [0, 1]

        Returns:
            lr_patch: Tensor dengan ukuran H//scale × W//scale
        """
        img = hr_patch.clone()
        H, W = img.shape[-2], img.shape[-1]
        target_h, target_w = H // self.scale_factor, W // self.scale_factor

        # ---- Step 1: Motion Blur (Eq.2.31: I_blur = IHR ⊗ k_motion) ----
        if random.random() < self.prob_motion_blur:
            img = apply_uav_motion_blur(
                img,
                kernel_size_range=self.motion_kernel_range,
                angle_range=self.motion_angle_range,
                length_range=self.motion_length_range,
            )

        # ---- Step 2: Exposure Flicker (Eq.2.32: I_exp = α · I_blur) ----
        if random.random() < self.prob_exposure:
            img = apply_exposure_flicker(img, alpha_range=self.alpha_range)

        # ---- Step 3: Haze ringan (opsional, sesuai Section 2.4) ----
        if random.random() < self.prob_haze:
            img = apply_haze(img, haze_intensity_range=self.haze_intensity_range)

        # ---- Step 4: Downsampling (Eq.2.33: I_ds = I_exp ↓s) ----
        p = random.random()
        cumsum = [self.resize_prob[0],
                  self.resize_prob[0] + self.resize_prob[1], 1.0]
        if p < cumsum[0]:
            # Up kemudian down
            up_factor = random.uniform(1.0, 2.0)
            mode = random_choice(RESIZE_MODES)
            img_up = F.interpolate(
                img.unsqueeze(0) if img.dim() == 3 else img,
                scale_factor=up_factor, mode=mode,
                align_corners=False if mode in ['bilinear', 'bicubic'] else None,
                antialias=(mode in ['bicubic', 'bilinear'])
            )
            img_up = img_up.squeeze(0) if img.dim() == 3 else img_up
            img = downsample_to_size(img_up if img.dim() == 3 else img_up, target_h, target_w)
        else:
            img = downsample_to_size(img, target_h, target_w)

        # ---- Step 5: UAV Noise (Eq.2.33-2.34: I_ds + n_UAV) ----
        img = add_uav_noise(
            img,
            gaussian_sigma_range=self.gaussian_sigma_range,
            poisson_scale_range=self.poisson_scale_range,
            gray_noise_prob=self.gray_noise_prob,
        )

        # ---- Step 6: Rotasi Kamera Kecil (Eq.2.35: I_rot = Rθ(I_ds)) ----
        if random.random() < self.prob_rotation:
            img = apply_small_rotation(img, theta_max=self.theta_max)

        # ---- Step 7: JPEG Compression (Eq.2.36: ILR = C_JPEG(I_rot)) ----
        img = add_jpeg_compression(img, quality_range=self.jpeg_quality_range)

        return clip_image(img)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Testing UAV-Specific Degradation Pipeline...")

    hr = torch.rand(3, 256, 256)

    # Test individual komponen
    print("Testing components:")

    blurred = apply_uav_motion_blur(hr)
    print(f"  Motion blur: {hr.shape} -> {blurred.shape}, "
          f"diff={abs(blurred - hr).mean():.4f} ✓")

    exposed = apply_exposure_flicker(hr, alpha_range=(0.8, 1.2))
    print(f"  Exposure flicker: {hr.shape} -> {exposed.shape} ✓")

    noisy = add_uav_noise(hr)
    print(f"  UAV noise: {hr.shape} -> {noisy.shape}, "
          f"diff={abs(noisy - hr).mean():.4f} ✓")

    hazy = apply_haze(hr)
    print(f"  Haze: {hr.shape} -> {hazy.shape} ✓")

    rotated = apply_small_rotation(hr, theta_max=3.0)
    print(f"  Rotation: {hr.shape} -> {rotated.shape} ✓")

    # Test full pipeline
    pipeline = UAVSpecificDegradationPipeline(scale_factor=4)
    lr = pipeline(hr)
    print(f"\nFull UAV pipeline: {hr.shape} -> {lr.shape}")
    print(f"LR range: [{lr.min():.4f}, {lr.max():.4f}]")
    assert lr.shape == (3, 64, 64), f"Expected (3,64,64), got {lr.shape}"

    # Run multiple times untuk test variasi
    print("Testing pipeline variability (5 runs):")
    lrs = [pipeline(hr) for _ in range(5)]
    diffs = [abs(lrs[i] - lrs[i+1]).mean().item() for i in range(4)]
    print(f"  Mean diff between runs: {sum(diffs)/len(diffs):.4f} (harus > 0)")
    assert sum(diffs) > 0, "Pipeline harus menghasilkan variasi!"

    print("\nUAV Degradation Pipeline test PASSED!")
