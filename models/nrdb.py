"""
Noise-Aware Residual Denoising Block (NRDB)
Sesuai proposal tesis Section 2.11.1 dan Eq.(2.10)(2.11)(2.12)(2.13)

Terinspirasi dari CBDNet (Guo et al., CVPR 2019) namun lebih ringan.
NRDB mempelajari noise secara implisit pada level fitur (tanpa noise estimation
subnetwork eksplisit), sehingga efisien untuk integrasi ke dalam generator GAN.

Formulasi:
    x = x_clean + n                          -- Eq.(2.10): model noise
    n_hat = F_NRDB(x; θ)                     -- Eq.(2.11): estimasi noise
    y = x - n_hat                            -- Eq.(2.12): subtraksi noise
    n_hat = W2 * σ(W1*x + b1) + b2          -- Eq.(2.13): struktur internal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NRDB(nn.Module):
    """
    Noise-Aware Residual Denoising Block.

    Sesuai proposal Section 2.11.1:
    - Diletakkan di awal generator sebagai modul praproses fitur
    - Menekan noise sensor non-Gaussian, artefak kompresi, dan gangguan UAV
    - Dioptimalkan end-to-end bersama generator (tidak ada loss denoising tersendiri)

    Struktur internal (Eq.2.13):
        n_hat = W2 * σ(W1*x + b1) + b2
    
    di mana W1, W2 = kernel conv 3×3, σ = LeakyReLU.
    Residual subtraction: y = x - n_hat (Eq.2.12)

    Args:
        num_features: Jumlah channel input/output
        num_inner: Jumlah conv layers di dalam NRDB
        bias: Gunakan bias pada conv
    """

    def __init__(
        self,
        num_features: int = 64,
        num_inner: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.num_features = num_features

        # Rangkaian conv untuk estimasi noise (Eq.2.13)
        # Setiap conv: W_i = kernel 3×3
        layers = []
        in_ch = num_features
        for i in range(num_inner):
            out_ch = num_features
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias))
            if i < num_inner - 1:
                # LeakyReLU (σ) di antara layer kecuali layer terakhir
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch

        self.noise_estimator = nn.Sequential(*layers)

        # Inisialisasi: weight kecil agar di awal training tidak merusak fitur
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Fitur input [B, C, H, W]
               Merepresentasikan superposisi fitur bersih + noise (Eq.2.10)

        Returns:
            y: Fitur tereduksi-noise [B, C, H, W] (Eq.2.12: y = x - n_hat)
        """
        # Eq.(2.11): estimasi komponen noise residual
        n_hat = self.noise_estimator(x)

        # Eq.(2.12): residual subtraction
        # y = x - n_hat  =>  fitur bersih tanpa noise
        y = x - n_hat

        return y


class NRDB_Block(nn.Module):
    """
    NRDB yang dibungkus dengan skip connection untuk stabilitas training.
    Versi yang sedikit lebih robust: y = x + alpha * (x - n_hat)
    dimana alpha adalah parameter yang dapat dipelajari (learnable scalar).
    """

    def __init__(self, num_features: int = 64, num_inner: int = 3):
        super().__init__()
        self.nrdb = NRDB(num_features, num_inner)
        # Learnable scale untuk stabilitas awal training
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        denoised = self.nrdb(x)
        # Soft integration: mulai dari hampir identitas, belajar seberapa besar denoising
        return x + self.scale * (denoised - x)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Testing NRDB...")

    nrdb = NRDB(num_features=64, num_inner=3)
    x = torch.randn(2, 64, 32, 32)
    y = nrdb(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == x.shape, "Shape mismatch!"

    # Verifikasi Eq.(2.12): y = x - n_hat => y berbeda dari x
    diff = (y - x).abs().mean().item()
    print(f"Mean |y - x| = {diff:.6f}  (harus > 0, berarti noise diestimasi)")
    assert diff > 0, "NRDB tidak melakukan denoising!"

    # Test gradient flow
    x_grad = torch.randn(2, 64, 32, 32, requires_grad=True)
    y_grad = nrdb(x_grad)
    loss = y_grad.mean()
    loss.backward()
    assert x_grad.grad is not None
    print("Gradient flow: OK")

    # Test NRDB_Block
    nrdb_block = NRDB_Block(64)
    y_block = nrdb_block(x)
    print(f"NRDB_Block output: {y_block.shape}")

    print("NRDB test PASSED!")
