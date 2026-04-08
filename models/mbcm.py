"""
Motion-Blur Compensation Module (MBCM)
Sesuai proposal tesis Section 2.11.2 dan Eq.(2.14)-(2.19)

Pendekatan: kernel-free deblurring pada domain fitur (tanpa estimasi kernel blur
secara langsung), terinspirasi dari:
- Chakrabarti (2016): Neural Approach to Blind Motion Deblurring
- Nah et al. (2017): Deep Multi-Scale CNN for Dynamic Scene Deblurring
- Sun et al. (2015): Learning CNN for Non-Uniform Motion Blur Removal
- Woo et al. (2018): CBAM - Convolutional Block Attention Module

Alur MBCM (Eq.2.14-2.19):
    F0 = φ(Iin)                    -- Eq.(2.14): ekstrasi fitur awal
    F_dir = Σ w_i * F0             -- Eq.(2.15): direction-aware conv
    B_hat = R(F_dir)               -- Eq.(2.16): estimasi blur residual
    F_deb = F0 - B_hat             -- Eq.(2.17): kompensasi blur
    A = σ(C(F_deb))                -- Eq.(2.18): motion-aware attention
    F_out = A ⊙ F_deb             -- Eq.(2.19): output dengan attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DirectionAwareConv(nn.Module):
    """
    Direction-Aware Convolution untuk menangkap pola blur berarah.
    Sesuai Eq.(2.15): F_dir = Σ w_i * F0

    Menggunakan beberapa kernel dengan orientasi berbeda (horizontal,
    vertikal, diagonal 45°, diagonal 135°) untuk menangkap arah blur
    akibat pergerakan UAV dari berbagai arah.

    Terinspirasi dari Sun et al. (2015) yang membuktikan pola motion blur
    bisa dipelajari via representasi fitur berarah tanpa estimasi kernel eksplisit.
    """

    def __init__(self, num_features: int = 64, num_directions: int = 4):
        super().__init__()
        self.num_directions = num_directions
        self.num_features = num_features

        # N conv berarah - setiap conv menangkap satu arah blur (Eq.2.15: w_i)
        self.direction_convs = nn.ModuleList([
            nn.Conv2d(num_features, num_features, 3, padding=1, bias=True)
            for _ in range(num_directions)
        ])

        # Fusion conv untuk menggabungkan N arah
        self.fusion = nn.Conv2d(num_features * num_directions, num_features, 1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # Inisialisasi kernel dengan orientasi berbeda
        self._initialize_directional_kernels()

    def _initialize_directional_kernels(self):
        """
        Inisialisasi kernel dengan bias arah yang berbeda.
        Ini memberikan titik awal yang lebih baik untuk pembelajaran arah blur.
        """
        # Definisi arah: horizontal, vertikal, diagonal-kanan, diagonal-kiri
        directions = [
            # Horizontal blur kernel
            torch.tensor([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=torch.float32) / 3,
            # Vertikal blur kernel
            torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=torch.float32) / 3,
            # Diagonal 45°
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32) / 3,
            # Diagonal 135°
            torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float32) / 3,
        ]

        for i, conv in enumerate(self.direction_convs):
            if i < len(directions):
                # Inisialisasi dengan kernel terarah, broadcast ke semua channel
                kernel = directions[i]
                # Shape: [out_ch, in_ch, kH, kW]
                with torch.no_grad():
                    C = conv.weight.shape[0]
                    for c in range(C):
                        conv.weight[c, c % conv.weight.shape[1]] = kernel
            else:
                nn.init.kaiming_normal_(conv.weight, a=0.2, mode='fan_in')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Eq.(2.15): F_dir = Σ w_i * F0, i=1..N
        Implementasi: konkatenasi hasil tiap arah, lalu fusion 1×1 conv
        """
        dir_feats = [self.lrelu(conv(x)) for conv in self.direction_convs]
        # Concat semua arah dan fuse
        out = self.fusion(torch.cat(dir_feats, dim=1))
        return out


class ResidualDeblurBlock(nn.Module):
    """
    Blok residual untuk estimasi komponen blur.
    Sesuai Eq.(2.16): B_hat = R(F_dir)

    Terinspirasi dari residual learning (He et al., 2016) yang terbukti
    efektif untuk restorasi citra dan deblurring tanpa estimasi kernel.
    """

    def __init__(self, num_features: int = 64, num_layers: int = 3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(num_features, num_features, 3, padding=1, bias=True))
            if i < num_layers - 1:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)
        self.res_scale = 0.2

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * self.res_scale


class MotionAwareAttention(nn.Module):
    """
    Motion-Aware Attention untuk menekankan area terdampak blur.
    Sesuai Eq.(2.18): A = σ(C(F_deb))

    Terinspirasi dari CBAM (Woo et al., 2018) - spatial attention module.
    Memungkinkan jaringan menekankan wilayah dengan degradasi blur lebih parah.
    """

    def __init__(self, num_features: int = 64):
        super().__init__()
        # C(.) = blok konvolusi attention (Eq.2.18)
        self.attention_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features // 4, 3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features // 4, 1, 3, padding=1, bias=True),
            nn.Sigmoid(),   # σ(.) -> peta attention [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eq.(2.18): A = σ(C(F_deb))
        return self.attention_conv(x)  # [B, 1, H, W]


class MBCM(nn.Module):
    """
    Motion-Blur Compensation Module.

    Sesuai proposal Section 2.11.2, mengimplementasikan Eq.(2.14)-(2.19):

    1. F0 = φ(Iin)              - Ekstraksi fitur awal (conv awal pada generator)
    2. F_dir = Σ w_i * F0       - Direction-aware convolution
    3. B_hat = R(F_dir)          - Estimasi blur residual
    4. F_deb = F0 - B_hat        - Kompensasi blur
    5. A = σ(C(F_deb))           - Motion-aware attention
    6. F_out = A ⊙ F_deb        - Output dengan attention weighting

    Args:
        num_features: Jumlah channel
        num_directions: Jumlah arah blur yang dimodelkan (N dalam Eq.2.15)
        num_blur_layers: Jumlah layer dalam residual deblur block
    """

    def __init__(
        self,
        num_features: int = 64,
        num_directions: int = 4,
        num_blur_layers: int = 3,
    ):
        super().__init__()
        self.num_features = num_features

        # Step 2: Direction-aware convolution (Eq.2.15)
        self.dir_conv = DirectionAwareConv(num_features, num_directions)

        # Step 3: Residual deblur block (Eq.2.16)
        self.deblur_block = ResidualDeblurBlock(num_features, num_blur_layers)

        # Step 5: Motion-aware attention (Eq.2.18)
        self.attention = MotionAwareAttention(num_features)

        # Fusion: gabungkan F0 dan F_out via learnable weighting
        self.out_conv = nn.Conv2d(num_features, num_features, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Fitur input F0 [B, C, H, W]
               Dalam konteks UAV-IRE, F0 adalah output dari NRDB
               (fitur sudah tereduksi noise, tapi masih mengandung blur)

        Returns:
            F_out: Fitur terkompensasi blur [B, C, H, W]
        """
        # Catatan: F0 = x (fitur dari NRDB atau conv pertama generator)
        # Step 2: Eq.(2.15) - direction-aware conv
        f_dir = self.dir_conv(x)

        # Step 3: Eq.(2.16) - estimasi komponen blur
        b_hat = self.deblur_block(f_dir)

        # Step 4: Eq.(2.17) - kompensasi blur
        f_deb = x - b_hat

        # Step 5: Eq.(2.18) - motion-aware attention
        A = self.attention(f_deb)  # [B, 1, H, W]

        # Step 6: Eq.(2.19) - F_out = A ⊙ F_deb
        f_out = A * f_deb

        # Final refinement + residual dengan input asli
        f_out = self.out_conv(f_out) + x

        return f_out


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Testing MBCM components...")

    # Test DirectionAwareConv
    dac = DirectionAwareConv(64, num_directions=4)
    x = torch.randn(2, 64, 32, 32)
    out = dac(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"DirectionAwareConv: {x.shape} -> {out.shape} ✓")

    # Test ResidualDeblurBlock
    rdb = ResidualDeblurBlock(64)
    out = rdb(x)
    assert out.shape == x.shape
    print(f"ResidualDeblurBlock: {x.shape} -> {out.shape} ✓")

    # Test MotionAwareAttention
    maa = MotionAwareAttention(64)
    A = maa(x)
    assert A.shape == (2, 1, 32, 32)
    assert A.min() >= 0 and A.max() <= 1  # Sigmoid output
    print(f"MotionAwareAttention: {x.shape} -> {A.shape}, range [{A.min():.3f}, {A.max():.3f}] ✓")

    # Test MBCM penuh
    mbcm = MBCM(num_features=64, num_directions=4)
    out = mbcm(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"MBCM full: {x.shape} -> {out.shape} ✓")

    # Test gradient flow
    x_grad = torch.randn(2, 64, 32, 32, requires_grad=True)
    out_grad = mbcm(x_grad)
    out_grad.mean().backward()
    assert x_grad.grad is not None
    print("Gradient flow: OK ✓")

    # Parameter count
    n_params = sum(p.numel() for p in mbcm.parameters())
    print(f"MBCM parameters: {n_params:,}")

    print("\nMBCM test PASSED!")
