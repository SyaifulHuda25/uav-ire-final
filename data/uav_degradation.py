"""
UAV-IRE UAV-Specific Degradation Pipeline
==========================================
Sesuai proposal Eq.(2.31)–(2.36):
  motion blur → exposure flicker → downsample → noise → rotasi → JPEG

Tersedia 4 level intensitas degradasi:
  'mild'     : degradasi ringan  — untuk eksperimen awal / pre-train
  'moderate' : degradasi sedang  — mendekati kondisi UAV nyata (REKOMENDASI)
  'strong'   : degradasi kuat    — kondisi UAV dengan gangguan signifikan
  'severe'   : degradasi ekstrem — stress test (tidak direkomendasikan untuk tesis)

Cara pakai di notebook SEL 1:
  DEGRADATION_LEVEL = 'mild'      # ganti sesuai skenario

Cara pakai di epoch_trainer.py:
  pipeline = UAVSpecificDegradationPipeline(scale_factor=4, level='mild')
"""

import random
import io
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter


# ─────────────────────────────────────────────────────────────────
# Parameter preset per level
# ─────────────────────────────────────────────────────────────────

LEVEL_PARAMS = {
    'mild': dict(
        # Motion blur (Eq.2.31): simulasi getaran gimbal ringan
        blur_sigma_range   = (0.3, 0.8),
        # Exposure flicker (Eq.2.32): variasi pencahayaan kecil
        exposure_range     = (0.95, 1.05),
        # Noise (Eq.2.34): noise sensor rendah
        gaussian_std_range = (0.005, 0.010),
        poisson_scale      = 0.10,           # bobot noise Poisson
        # Rotasi kamera (Eq.2.35): perubahan sudut gimbal sangat kecil
        rotation_range     = (-0.5, 0.5),
        # JPEG compression (Eq.2.36): kualitas tinggi
        jpeg_quality_range = (85, 95),
    ),
    'moderate': dict(
        blur_sigma_range   = (0.8, 1.5),
        exposure_range     = (0.90, 1.10),
        gaussian_std_range = (0.010, 0.020),
        poisson_scale      = 0.20,
        rotation_range     = (-1.0, 1.0),
        jpeg_quality_range = (75, 85),
    ),
    'strong': dict(
        blur_sigma_range   = (1.5, 2.5),
        exposure_range     = (0.85, 1.15),
        gaussian_std_range = (0.020, 0.040),
        poisson_scale      = 0.30,
        rotation_range     = (-2.0, 2.0),
        jpeg_quality_range = (65, 80),
    ),
    'severe': dict(
        blur_sigma_range   = (2.5, 4.0),
        exposure_range     = (0.80, 1.20),
        gaussian_std_range = (0.030, 0.060),
        poisson_scale      = 0.40,
        rotation_range     = (-3.0, 3.0),
        jpeg_quality_range = (50, 70),
    ),
}


class UAVSpecificDegradationPipeline:
    """
    Pipeline degradasi khas citra UAV persawahan.
    Mengikuti urutan Eq.(2.31)–(2.36) dari proposal.

    Args:
        scale_factor: Faktor downsampling (default 4 untuk SR×4)
        level: Level intensitas degradasi ('mild'|'moderate'|'strong'|'severe')

    Input : torch.Tensor [C, H, W] float32 range [0,1]  (HR asli)
    Output: torch.Tensor [C, H//scale, W//scale] float32 range [0,1]  (LR terdegradasi)
    """

    LEVELS = ('mild', 'moderate', 'strong', 'severe')

    def __init__(self, scale_factor: int = 4, level: str = 'moderate'):
        assert level in self.LEVELS, \
            f"level harus salah satu dari {self.LEVELS}, bukan '{level}'"
        self.scale_factor = scale_factor
        self.level        = level
        self.params       = LEVEL_PARAMS[level]

    def __repr__(self):
        p = self.params
        return (
            f"UAVSpecificDegradationPipeline("
            f"level='{self.level}', scale={self.scale_factor}x, "
            f"blur={p['blur_sigma_range']}, "
            f"noise_std={p['gaussian_std_range']}, "
            f"jpeg={p['jpeg_quality_range']})"
        )

    def __call__(self, hr_tensor: torch.Tensor) -> torch.Tensor:
        """
        Terapkan pipeline degradasi ke satu gambar HR.

        Args:
            hr_tensor: [C, H, W] float32 [0,1]
        Returns:
            lr_tensor: [C, H//scale, W//scale] float32 [0,1]
        """
        p = self.params
        C, H, W = hr_tensor.shape

        # ── Step 1: Motion blur (Eq.2.31) ────────────────────────
        # Simulasi getaran gimbal UAV → Gaussian blur dengan sigma acak
        sigma   = random.uniform(*p['blur_sigma_range'])
        hr_pil  = TF.to_pil_image(hr_tensor.clamp(0, 1))
        hr_pil  = hr_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
        hr_t    = TF.to_tensor(hr_pil)

        # ── Step 2: Exposure flicker (Eq.2.32) ───────────────────
        # Simulasi fluktuasi pencahayaan selama penerbangan
        alpha   = random.uniform(*p['exposure_range'])
        hr_t    = (hr_t * alpha).clamp(0, 1)

        # ── Step 3: Downsampling (Eq.2.33) ───────────────────────
        # Turunkan resolusi sesuai scale_factor
        lH, lW  = H // self.scale_factor, W // self.scale_factor
        lr_t    = F.interpolate(
            hr_t.unsqueeze(0), size=(lH, lW),
            mode='bicubic', antialias=True
        ).squeeze(0).clamp(0, 1)

        # ── Step 4: UAV Noise (Eq.2.34) ──────────────────────────
        # Gaussian + Poisson — simulasi noise sensor dan angin
        std     = random.uniform(*p['gaussian_std_range'])
        noise_g = torch.randn_like(lr_t) * std
        noise_p = (torch.poisson(lr_t * 255.0) / 255.0 - lr_t) * p['poisson_scale']
        lr_t    = (lr_t + noise_g + noise_p).clamp(0, 1)

        # ── Step 5: Rotasi kamera kecil (Eq.2.35) ────────────────
        # Simulasi perubahan sudut gimbal
        angle   = random.uniform(*p['rotation_range'])
        lr_pil  = TF.to_pil_image(lr_t)
        lr_pil  = lr_pil.rotate(angle, expand=False)
        lr_t    = TF.to_tensor(lr_pil)

        # ── Step 6: JPEG compression (Eq.2.36) ───────────────────
        # Simulasi artefak penyimpanan dan transmisi
        quality = random.randint(*p['jpeg_quality_range'])
        buf     = io.BytesIO()
        TF.to_pil_image(lr_t.clamp(0, 1)).save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        lr_t    = TF.to_tensor(Image.open(buf).copy())

        return lr_t.clamp(0, 1)

    def apply_batch(self, hr_batch: torch.Tensor) -> torch.Tensor:
        """Terapkan pipeline ke batch [B, C, H, W]."""
        return torch.stack(
            [self.__call__(hr_batch[i]) for i in range(hr_batch.shape[0])],
            dim=0
        )

    def describe(self) -> str:
        """Deskripsi parameter aktif untuk logging."""
        p = self.params
        lines = [
            f"  Level          : {self.level}",
            f"  Scale factor   : {self.scale_factor}x",
            f"  Blur σ         : {p['blur_sigma_range']}",
            f"  Exposure α     : {p['exposure_range']}",
            f"  Gaussian σ     : {p['gaussian_std_range']}",
            f"  Poisson scale  : {p['poisson_scale']}",
            f"  Rotation       : {p['rotation_range']} deg",
            f"  JPEG quality   : {p['jpeg_quality_range']}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import math

    print("=" * 55)
    print("UAVSpecificDegradationPipeline — self-test semua level")
    print("=" * 55)

    hr = torch.rand(3, 256, 256)

    for level in UAVSpecificDegradationPipeline.LEVELS:
        pipe = UAVSpecificDegradationPipeline(scale_factor=4, level=level)
        lr   = pipe(hr)

        assert lr.shape == (3, 64, 64), f"Shape salah: {lr.shape}"
        assert lr.min() >= 0.0 and lr.max() <= 1.0, "Nilai di luar [0,1]"

        # Hitung PSNR HR (bicubic upscale LR) vs HR asli sebagai indikator
        lr_up  = F.interpolate(lr.unsqueeze(0), size=(256, 256),
                               mode='bicubic', antialias=True).squeeze(0).clamp(0, 1)
        mse    = ((hr - lr_up) ** 2).mean().item()
        psnr   = 10 * math.log10(1.0 / (mse + 1e-8))

        print(f"\n[{level.upper():8s}] LR shape: {list(lr.shape)} | "
              f"Bicubic PSNR: {psnr:.2f} dB")
        print(pipe.describe())

    print("\nSemua level PASSED ✓")
    print()
    print("Panduan pemilihan level untuk tesis:")
    print("  mild     → mulai dari sini, pastikan model bisa konvergen dulu")
    print("  moderate → level rekomendasi untuk training utama tesis")
    print("  strong   → jika moderate sudah bagus, coba ini untuk robustness")
    print("  severe   → tidak direkomendasikan (terlalu jauh dari kondisi nyata)")
