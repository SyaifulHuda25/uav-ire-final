"""
UAV-IRE Generator
Sesuai proposal tesis Section 2.11 dan Gambar 3.1 (Diagram Alir)

Mengintegrasikan semua modul arsitektur baru ke dalam generator IRE:
- NRDB: Noise-Aware Residual Denoising Block (Section 2.11.1)
- MBCM: Motion-Blur Compensation Module (Section 2.11.2)
- EGA: Edge-Guided Attention (Section 2.11.3)
- VSD: Vegetation Similarity Discriminator (digunakan saat training, Section 2.11.4)

Arsitektur UAV-IRE Generator (sesuai Fig. generator dalam Gambar 3.1):

    Input LR
        │
        ▼
    Conv (feature extraction)
        │
        ▼
    ┌───────────────────────────────────────┐
    │  NRDB Block (denoising)               │  ← Branch 1: noise suppression
    │  MBCM Block (deblurring)              │  ← Branch 2: blur compensation
    └───────────────────────────────────────┘
        │                   │
        ▼                   ▼
    Basic Block       Basic Block
    (dengan Att)      (MBCM path)
        │                   │
        └─────── Fusion ────┘
                    │
                    ▼
               EGA Module
                    │
                    ▼
            Upsampling ×4
                    │
                    ▼
              Conv Output
                    │
                    ▼
               SR Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.network import RRDB, ChannelAttention, PatchGAN_Discriminator
from models.nrdb import NRDB_Block
from models.mbcm import MBCM
from models.ega import EGA


class FusionBlock(nn.Module):
    """
    Fusi fitur dari jalur NRDB dan MBCM.
    Cross-Scale Fusion: menggabungkan informasi dari dua jalur berbeda.
    Sesuai pendekatan CS-IRE (Jin-li et al., 2025) yang diadopsi dalam proposal.
    """

    def __init__(self, num_features: int = 64):
        super().__init__()
        # Fusi via concatenation + 1x1 conv
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1, bias=True),
        )

    def forward(self, feat_nrdb: torch.Tensor, feat_mbcm: torch.Tensor) -> torch.Tensor:
        # Gabungkan fitur dari dua jalur
        fused = torch.cat([feat_nrdb, feat_mbcm], dim=1)
        return self.fusion_conv(fused)


class UAVIRE_Generator(nn.Module):
    """
    UAV-IRE Generator.

    Sesuai proposal tesis, mengintegrasikan komponen berikut
    ke dalam IRE Generator:

    1. NRDB (Noise-Aware Residual Denoising Block)
       - Disisipkan di awal generator
       - Menekan noise sensor, noise angin, artefak kompresi

    2. MBCM (Motion-Blur Compensation Module)
       - Menangani blur non-uniform akibat getaran UAV
       - Direction-aware convolution untuk berbagai arah blur

    3. EGA (Edge-Guided Attention)
       - Disisipkan sebelum upsampling
       - Mempertahankan struktur tepi dan detail objek kecil

    4. Dual-branch: jalur NRDB dan MBCM berjalan paralel lalu difusi

    Args:
        in_channels: Channel input (default 3 untuk RGB)
        out_channels: Channel output
        num_features: Jumlah channel internal
        num_rrdb: Jumlah RRDB blocks
        growth_rate: Growth rate untuk dense blocks
        scale_factor: Faktor upscaling (default 4x)
        num_nrdb_layers: Jumlah layer di dalam NRDB
        num_directions: Jumlah arah blur yang dimodelkan di MBCM
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_rrdb: int = 23,
        growth_rate: int = 32,
        scale_factor: int = 4,
        # NRDB config
        num_nrdb_layers: int = 3,
        # MBCM config
        num_directions: int = 4,
        num_blur_layers: int = 3,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.num_features = num_features

        # ============================================================
        # Stage 1: Feature Extraction (Conv awal)
        # ============================================================
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, padding=1, bias=True)

        # ============================================================
        # Stage 2: Dual-Branch Preprocessing
        # Sesuai Gambar 3.1 (Generator UAV-IRE)
        # Branch 1: NRDB -> Basic Blocks (Att)
        # Branch 2: MBCM -> Basic Blocks
        # ============================================================

        # Branch 1: Noise suppression path
        self.nrdb = NRDB_Block(num_features, num_nrdb_layers)
        # RRDB blocks untuk jalur noise-cleaned features
        n_rrdb_branch = max(num_rrdb // 2, 3)   # Bagi RRDB antara dua jalur
        self.rrdb_nrdb_path = nn.Sequential(
            *[RRDB(num_features, growth_rate) for _ in range(n_rrdb_branch)]
        )

        # Branch 2: Blur compensation path
        self.mbcm = MBCM(num_features, num_directions, num_blur_layers)
        # RRDB blocks untuk jalur blur-compensated features
        self.rrdb_mbcm_path = nn.Sequential(
            *[RRDB(num_features, growth_rate) for _ in range(n_rrdb_branch)]
        )

        # ============================================================
        # Stage 3: Cross-Scale Fusion
        # Menggabungkan informasi dari kedua jalur
        # ============================================================
        self.fusion = FusionBlock(num_features)
        self.conv_body = nn.Conv2d(num_features, num_features, 3, padding=1, bias=True)

        # ============================================================
        # Stage 4: Edge-Guided Attention (sebelum upsampling)
        # Sesuai Gambar 3.1: EGA di antara Fusion dan Upsampling
        # ============================================================
        self.ega = EGA(num_features)

        # ============================================================
        # Stage 5: Upsampling ×4 (via Pixel Shuffle)
        # ============================================================
        upsample_layers = []
        if scale_factor == 4:
            upsample_layers += [
                nn.Conv2d(num_features, num_features * 4, 3, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(num_features, num_features * 4, 3, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif scale_factor == 2:
            upsample_layers += [
                nn.Conv2d(num_features, num_features * 4, 3, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.upsample = nn.Sequential(*upsample_layers)

        # ============================================================
        # Stage 6: Output Convolutions
        # ============================================================
        self.conv_hr = nn.Conv2d(num_features, num_features, 3, padding=1, bias=True)
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass UAV-IRE Generator.

        Args:
            x: LR image [B, C, H, W]

        Returns:
            SR image [B, C, H*scale, W*scale]
        """
        # ---- Stage 1: Feature extraction ----
        feat = self.conv_first(x)           # [B, F, H, W]

        # ---- Stage 2a: Branch 1 - NRDB path ----
        # Menekan noise sebelum memasuki RRDB blocks
        feat_nrdb = self.nrdb(feat)         # [B, F, H, W] - denoised features
        feat_nrdb = self.rrdb_nrdb_path(feat_nrdb)

        # ---- Stage 2b: Branch 2 - MBCM path ----
        # Kompensasi blur sebelum memasuki RRDB blocks
        feat_mbcm = self.mbcm(feat)         # [B, F, H, W] - deblurred features
        feat_mbcm = self.rrdb_mbcm_path(feat_mbcm)

        # ---- Stage 3: Cross-Scale Fusion ----
        feat_fused = self.fusion(feat_nrdb, feat_mbcm)
        body_feat = self.conv_body(feat_fused)
        feat = feat + body_feat             # Global residual connection

        # ---- Stage 4: Edge-Guided Attention ----
        feat = self.ega(feat)               # [B, F, H, W] - edge-enhanced

        # ---- Stage 5: Upsampling ×4 ----
        feat = self.upsample(feat)          # [B, F, H*4, W*4]

        # ---- Stage 6: Output ----
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# UAV-IRE Complete Model (Generator + VSD + PatchGAN)
# ---------------------------------------------------------------------------

class UAVIRE_Model(nn.Module):
    """
    Model UAV-IRE lengkap untuk training.

    Berisi:
    - Generator (UAV-IRE Generator)
    - Discriminator (PatchGAN, dari IRE baseline)
    - VSD (Vegetation Similarity Discriminator)
    """

    def __init__(
        self,
        num_rrdb: int = 23,
        scale_factor: int = 4,
        num_features: int = 64,
        # VSD lambdas
        vsd_lambda_g: float = 1.0,
        vsd_lambda_w: float = 0.5,
        vsd_lambda_v: float = 0.1,
    ):
        super().__init__()
        from models.vsd import VSD

        self.generator = UAVIRE_Generator(
            num_rrdb=num_rrdb,
            scale_factor=scale_factor,
            num_features=num_features,
        )
        self.discriminator = PatchGAN_Discriminator()  # Dari IRE baseline
        self.vsd = VSD(
            lambda_g=vsd_lambda_g,
            lambda_w=vsd_lambda_w,
            lambda_v=vsd_lambda_v,
        )

    def generate(self, lr: torch.Tensor) -> torch.Tensor:
        """Super-resolve LR image."""
        return self.generator(lr)

    def discriminate(self, img: torch.Tensor) -> torch.Tensor:
        """PatchGAN discriminator output."""
        return self.discriminator(img)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing UAV-IRE Generator on {device}...")

    # Test dengan num_rrdb kecil untuk kecepatan
    gen = UAVIRE_Generator(
        num_rrdb=4,
        scale_factor=4,
        num_features=32,
    ).to(device)

    lr = torch.randn(1, 3, 32, 32).to(device)

    with torch.no_grad():
        sr = gen(lr)

    print(f"Input LR:  {lr.shape}")
    print(f"Output SR: {sr.shape}")
    assert sr.shape == (1, 3, 128, 128), f"Expected (1,3,128,128), got {sr.shape}"

    # Parameter count
    n_params = gen.count_parameters()
    print(f"UAV-IRE Generator parameters: {n_params:,}")

    # Breakdown per modul
    components = {
        'conv_first': gen.conv_first,
        'nrdb': gen.nrdb,
        'rrdb_nrdb_path': gen.rrdb_nrdb_path,
        'mbcm': gen.mbcm,
        'rrdb_mbcm_path': gen.rrdb_mbcm_path,
        'fusion': gen.fusion,
        'ega': gen.ega,
        'upsample': gen.upsample,
    }
    print("\nParameter breakdown:")
    for name, module in components.items():
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name}: {n:,}")

    # Test backward
    lr_grad = torch.randn(1, 3, 32, 32, requires_grad=True).to(device)
    sr_grad = gen(lr_grad)
    loss = sr_grad.mean()
    loss.backward()
    assert lr_grad.grad is not None
    print("\nBackward pass: OK ✓")

    print("\nUAV-IRE Generator test PASSED!")
