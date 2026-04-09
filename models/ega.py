"""
Edge-Guided Attention Mechanism (EGA)
Sesuai proposal tesis Section 2.11.3 dan Eq.(2.20)(2.21)(2.22)(2.23)

Motivasi: Citra UAV persawahan mengandung gulma kecil dengan batas morfologi
halus dan kontras rendah terhadap padi. Detail tepi sangat rentan terdegradasi
akibat noise, blur, dan downsampling.

EGA menggunakan Sobel operator pada domain fitur untuk:
1. Ekstraksi gradien spasial (Eq.2.20)
2. Komputasi magnitudo gradien = peta tepi (Eq.2.21)
3. Pembentukan peta atensi dari peta tepi (Eq.2.22)
4. Penguatan fitur via residual attention (Eq.2.23)

Referensi:
- Dong et al. (2016), Ledig et al. (2017): edge preservation krusial untuk ISR
- Xu et al. (2014): deep edge-aware image restoration
- Woo et al. (2018): CBAM spatial attention
- He et al. (2016): residual learning
- Zhang et al. (2018): RCAN - attention for SR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SobelEdgeExtractor(nn.Module):
    """
    Ekstraksi gradien spasial menggunakan Sobel operator (depth-wise).
    Sesuai Eq.(2.20): Gx = F * Kx,  Gy = F * Ky

    Diterapkan secara depth-wise (per-channel) untuk mempertahankan
    informasi edge pada setiap channel fitur.
    """

    def __init__(self, num_features: int = 64):
        super().__init__()
        self.num_features = num_features

        # Kernel Sobel horizontal dan vertikal
        # Kx: deteksi tepi horizontal
        kx = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32)
        # Ky: deteksi tepi vertikal
        ky = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32)

        # Expand ke [C, 1, 3, 3] untuk depthwise conv
        self.register_buffer(
            'kx',
            kx.unsqueeze(0).unsqueeze(0).repeat(num_features, 1, 1, 1)
        )
        self.register_buffer(
            'ky',
            ky.unsqueeze(0).unsqueeze(0).repeat(num_features, 1, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Fitur input [B, C, H, W]

        Returns:
            G: Peta magnitudo gradien [B, C, H, W] (Eq.2.21)
        """
        # Eq.(2.20): Gx = F * Kx, Gy = F * Ky (depthwise convolution)
        Gx = F.conv2d(x, self.kx, padding=1, groups=self.num_features)
        Gy = F.conv2d(x, self.ky, padding=1, groups=self.num_features)

        # Eq.(2.21): G = sqrt(Gx^2 + Gy^2) - peta tepi
        G = torch.sqrt(Gx ** 2 + Gy ** 2 + 1e-8)

        return G


class EGA(nn.Module):
    """
    Edge-Guided Attention Mechanism.

    Sesuai proposal Section 2.11.3, mengimplementasikan Eq.(2.20)-(2.23):

    1. Gx = F * Kx,  Gy = F * Ky        (Eq.2.20) - gradien Sobel
    2. G = sqrt(Gx² + Gy²)              (Eq.2.21) - peta tepi
    3. A = σ(Conv1×1(Conv3×3(G)))        (Eq.2.22) - peta atensi dari tepi
    4. F_EGA = F ⊙ (1 + A)              (Eq.2.23) - residual attention

    Properti Eq.(2.23): F ⊙ (1 + A)
    - Jika A → 0: F_EGA ≈ F  (tidak mengubah area tanpa tepi)
    - Jika A → 1: F_EGA = 2F (memperkuat 2x area dengan tepi kuat)
    Ini memastikan area homogen/noise tidak didominasi attention berlebihan.

    Args:
        num_features: Jumlah channel fitur input
    """

    def __init__(self, num_features: int = 64):
        super().__init__()
        self.num_features = num_features

        # Step 1-2: Sobel edge extractor (Eq.2.20-2.21)
        self.sobel = SobelEdgeExtractor(num_features)

        # Step 3: Pembentukan peta atensi dari peta tepi (Eq.2.22)
        # A = σ(Conv1×1(Conv3×3(G)))
        # Menangkap konteks lokal (Conv3×3) lalu reduksi dimensi (Conv1×1)
        self.attention_net = nn.Sequential(
            # Conv3×3: menangkap konteks lokal dari peta tepi
            nn.Conv2d(num_features, num_features // 2, 3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Conv1×1: reduksi dimensi -> peta atensi spasial [B, 1, H, W]
            nn.Conv2d(num_features // 2, 1, 1, bias=True),
            # Sigmoid: A ∈ [0, 1] (Eq.2.22: A ∈ R^{1×H×W})
            nn.Sigmoid(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.attention_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Fitur input F [B, C, H, W]
               Dalam UAV-IRE: fitur hasil fusi dari jalur NRDB dan MBCM

        Returns:
            F_EGA: Fitur terkalibasi berbasis tepi [B, C, H, W]
        """
        # Step 1-2: Ekstraksi peta tepi (Eq.2.20-2.21)
        G = self.sobel(x)   # [B, C, H, W] - peta magnitudo gradien

        # Step 3: Pembentukan peta atensi (Eq.2.22)
        # A = σ(Conv1×1(Conv3×3(G)))
        A = self.attention_net(G)   # [B, 1, H, W]

        # Step 4: Residual attention (Eq.2.23): F_EGA = F ⊙ (1 + A)
        # Broadcast: A [B, 1, H, W] * x [B, C, H, W]
        F_EGA = x * (1 + A)

        return F_EGA


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Testing EGA components...")

    # Test SobelEdgeExtractor
    sobel = SobelEdgeExtractor(64)
    x = torch.randn(2, 64, 32, 32)
    G = sobel(x)
    assert G.shape == x.shape
    assert (G >= 0).all(), "Magnitude harus non-negatif"
    print(f"SobelEdgeExtractor: {x.shape} -> G{G.shape}, range [{G.min():.3f}, {G.max():.3f}] ✓")

    # Test dengan gambar sintetis yang punya edge jelas
    x_edge = torch.zeros(1, 1, 8, 8)
    x_edge[:, :, :, 4:] = 1.0   # Hard edge di tengah
    sobel_1ch = SobelEdgeExtractor(1)
    G_edge = sobel_1ch(x_edge)
    # Edge harus terdeteksi di kolom 4
    assert G_edge[:, :, :, 3:5].mean() > G_edge[:, :, :, :3].mean(), \
        "Sobel harus mendeteksi edge dengan baik"
    print(f"Sobel edge detection test: edge correctly detected ✓")

    # Test EGA penuh
    ega = EGA(num_features=64)
    out = ega(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"EGA full: {x.shape} -> {out.shape} ✓")

    # Verifikasi Eq.(2.23): F_EGA = F ⊙ (1 + A)
    # Output harus >= input karena (1 + A) >= 1 (A ∈ [0,1])
    # Ini memperkuat, tidak melemahkan
    # Untuk input positif, output harus lebih besar
    x_pos = torch.abs(x)
    out_pos = ega(x_pos)
    # Rata-rata absolute output harus >= input (karena diperkuat oleh A)
    assert out_pos.abs().mean() >= x_pos.abs().mean() * 0.9, \
        "EGA harus memperkuat fitur tepi"
    print(f"Eq.(2.23) residual attention verification ✓")

    # Test gradient flow
    x_grad = torch.randn(2, 64, 32, 32, requires_grad=True)
    out_grad = ega(x_grad)
    out_grad.mean().backward()
    assert x_grad.grad is not None
    print("Gradient flow: OK ✓")

    # Parameter count
    n_params = sum(p.numel() for p in ega.parameters())
    print(f"EGA parameters: {n_params:,}")

    print("\nEGA test PASSED!")
