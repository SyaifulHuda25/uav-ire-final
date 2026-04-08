"""
UAV-IRE Loss Functions
Sesuai proposal tesis Section 2.11.6 dan Eq.(2.37)-(2.43)

Total loss (Eq.2.43):
L_total = λ1*L_rec + λ2*L_perc + λ3*L_adv + λ4*L_edge + λ5*L_VSD

Komponen:
1. L_rec: Smooth-L1 reconstruction loss (Eq.2.37)(2.38) - primary loss
2. L_perc: Perceptual loss VGG (Eq.2.39) - konsistensi visual struktur
3. L_adv: Adversarial loss GAN (Eq.2.40) - realisme visual
4. L_edge: Edge-guided loss (Eq.2.41) - ketajaman batas objek kecil (auxiliary)
5. L_VSD: Vegetation Similarity Discriminator loss (Eq.2.42) - supervisi vegetasi (auxiliary)

Catatan: λ4 dan λ5 diberikan nilai lebih kecil (sesuai proposal).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Smooth-L1 Reconstruction Loss
# Sesuai Eq.(2.37)(2.38)
# ---------------------------------------------------------------------------

class SmoothL1ReconstructionLoss(nn.Module):
    """
    Smooth-L1 loss untuk rekonstruksi pixel-wise.
    Sesuai Eq.(2.37)(2.38):
    L_SmoothL1(x,y) = 0.5*(x-y)² jika |x-y|<1, else |x-y|-0.5
    L_rec = (1/N) Σ L_SmoothL1(G(ILR)_i, IHR_i)
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss(reduction='mean', beta=1.0)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return self.loss(sr, hr)


# ---------------------------------------------------------------------------
# Perceptual Loss (VGG-based)
# Sesuai Eq.(2.39)
# ---------------------------------------------------------------------------

class PerceptualLossVGG(nn.Module):
    """
    Perceptual loss berbasis fitur VGG.
    Sesuai Eq.(2.39): L_perc = (1/ClHlWl) * ||φl(G(ILR)) - φl(IHR)||₁

    Menggunakan norma L1 (seperti IRE/ESRGAN).
    Feature layer: VGG relu3_4 (index 26) secara default.
    """

    def __init__(self, feature_layer: int = 34):
        super().__init__()
        from models.network import VGGFeatureExtractor
        self.vgg = VGGFeatureExtractor(feature_layer=feature_layer)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_feat = self.vgg(sr)
        with torch.no_grad():
            hr_feat = self.vgg(hr)

        # Eq.(2.39): L1 distance pada ruang fitur VGG
        return F.l1_loss(sr_feat, hr_feat)


# ---------------------------------------------------------------------------
# Adversarial Loss (GAN)
# Sesuai Eq.(2.40)
# ---------------------------------------------------------------------------

class AdversarialLossGAN(nn.Module):
    """
    Standard adversarial loss untuk generator.
    Sesuai Eq.(2.40): L_adv = -E_ILR[log D(G(ILR))]
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        fake_preds: torch.Tensor,
        real_preds: Optional[torch.Tensor] = None,
        use_ragan: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            fake_preds: D(G(ILR)) - discriminator output pada SR
            real_preds: D(IHR) - discriminator output pada HR (untuk RaGAN)
            use_ragan: Gunakan Relativistic average GAN (sesuai IRE/ESRGAN)
        """
        if use_ragan and real_preds is not None:
            # RaGAN version (konsisten dengan IRE baseline)
            real_mean = real_preds.mean()
            loss = F.binary_cross_entropy_with_logits(
                fake_preds - real_mean, torch.ones_like(fake_preds)
            )
            return loss
        else:
            # Standard GAN (Eq.2.40)
            return F.binary_cross_entropy_with_logits(
                fake_preds, torch.ones_like(fake_preds)
            )


class AdversarialLossDiscriminator(nn.Module):
    """Discriminator loss (RaGAN)."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        real_preds: torch.Tensor,
        fake_preds: torch.Tensor,
    ) -> torch.Tensor:
        real_mean = real_preds.mean()
        fake_mean = fake_preds.mean()
        loss = (
            F.binary_cross_entropy_with_logits(
                real_preds - fake_mean, torch.ones_like(real_preds)
            ) +
            F.binary_cross_entropy_with_logits(
                fake_preds - real_mean, torch.zeros_like(fake_preds)
            )
        ) / 2
        return loss


# ---------------------------------------------------------------------------
# Edge-Guided Loss (auxiliary)
# Sesuai Eq.(2.41)
# ---------------------------------------------------------------------------

class EdgeGuidedLoss(nn.Module):
    """
    Edge-guided loss untuk mempertahankan ketajaman batas objek.
    Sesuai Eq.(2.41): L_edge = ||∇G(ILR) - ∇IHR||₁

    Menggunakan Sobel operator sebagai ∇ (gradient operator).
    Bersifat auxiliary dengan bobot kecil untuk menghindari artefak berlebih.
    """

    def __init__(self):
        super().__init__()
        # Register Sobel kernels sebagai buffer (tidak ditraining)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('kx', kx.unsqueeze(0).unsqueeze(0))  # [1,1,3,3]
        self.register_buffer('ky', ky.unsqueeze(0).unsqueeze(0))

    def _gradient_map(self, img: torch.Tensor) -> torch.Tensor:
        """Hitung magnitude gradien Sobel dari gambar [B, C, H, W]."""
        B, C, H, W = img.shape
        # Konversi ke grayscale untuk efisiensi
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

        Gx = F.conv2d(gray, self.kx, padding=1)
        Gy = F.conv2d(gray, self.ky, padding=1)
        return torch.sqrt(Gx ** 2 + Gy ** 2 + 1e-8)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # Eq.(2.41): L_edge = ||∇G(ILR) - ∇IHR||₁
        grad_sr = self._gradient_map(sr)
        with torch.no_grad():
            grad_hr = self._gradient_map(hr)
        return F.l1_loss(grad_sr, grad_hr)


# ---------------------------------------------------------------------------
# VSD Loss Wrapper
# Sesuai Eq.(2.42)
# ---------------------------------------------------------------------------

class VSDLoss(nn.Module):
    """
    Wrapper untuk Vegetation Similarity Discriminator loss.
    Sesuai Eq.(2.42): L_VSD = ||D_veg(G(ILR)) - D_veg(IHR)||₁
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        vsd_module=None,
    ) -> torch.Tensor:
        """
        Hitung VSD loss menggunakan VSD module yang sudah di-training.
        """
        if vsd_module is None:
            return torch.tensor(0.0, device=sr.device)

        loss, _ = vsd_module(sr, hr, mask, mode='generator')
        return loss


# ---------------------------------------------------------------------------
# UAVIRE Total Generator Loss (Eq.2.43)
# ---------------------------------------------------------------------------

class UAVIRE_GeneratorLoss(nn.Module):
    """
    Total generator loss untuk UAV-IRE.

    Sesuai Eq.(2.43):
    L_total = λ1*L_rec + λ2*L_perc + λ3*L_adv + λ4*L_edge + λ5*L_VSD

    Default weights (sesuai proposal):
    - λ1 = 1.0 (rekonstruksi utama)
    - λ2 = 1.0 (konsistensi visual)
    - λ3 = 0.1 (realisme - konsisten dengan IRE Eq.6)
    - λ4 = 0.05 (edge, auxiliary - kecil sesuai proposal)
    - λ5 = 0.1 (VSD, auxiliary - kecil sesuai proposal)

    Args:
        lambda_rec: Bobot reconstruction loss
        lambda_perc: Bobot perceptual loss
        lambda_adv: Bobot adversarial loss
        lambda_edge: Bobot edge loss (auxiliary)
        lambda_vsd: Bobot VSD loss (auxiliary)
        use_edge_loss: Aktifkan edge-guided loss
        use_vsd_loss: Aktifkan VSD loss
    """

    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_perc: float = 1.0,
        lambda_adv: float = 0.1,
        lambda_edge: float = 0.05,
        lambda_vsd: float = 0.1,
        use_edge_loss: bool = True,
        use_vsd_loss: bool = True,
        vgg_feature_layer: int = 34,
    ):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_perc = lambda_perc
        self.lambda_adv = lambda_adv
        self.lambda_edge = lambda_edge
        self.lambda_vsd = lambda_vsd
        self.use_edge_loss = use_edge_loss
        self.use_vsd_loss = use_vsd_loss

        # Loss functions
        self.rec_loss = SmoothL1ReconstructionLoss()
        self.perc_loss = PerceptualLossVGG(feature_layer=vgg_feature_layer)
        self.adv_loss = AdversarialLossGAN()
        self.edge_loss = EdgeGuidedLoss() if use_edge_loss else None
        self.vsd_loss_fn = VSDLoss() if use_vsd_loss else None

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        fake_preds: torch.Tensor,
        real_preds: Optional[torch.Tensor] = None,
        weed_mask: Optional[torch.Tensor] = None,
        vsd_module=None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Hitung total loss UAV-IRE.

        Args:
            sr: SR image dari generator
            hr: HR image (ground truth)
            fake_preds: D(SR) dari discriminator
            real_preds: D(HR) dari discriminator (untuk RaGAN)
            weed_mask: Mask area gulma [B, 1, H, W] (opsional)
            vsd_module: VSD module yang sudah di-training

        Returns:
            (total_loss, loss_dict)
        """
        losses = {}

        # ---- L_rec: Smooth-L1 reconstruction (Eq.2.38) ----
        l_rec = self.rec_loss(sr, hr)
        losses['rec'] = l_rec.item()

        # ---- L_perc: Perceptual loss (Eq.2.39) ----
        l_perc = self.perc_loss(sr, hr)
        losses['perc'] = l_perc.item()

        # ---- L_adv: Adversarial loss (Eq.2.40) ----
        l_adv = self.adv_loss(fake_preds, real_preds, use_ragan=(real_preds is not None))
        losses['adv'] = l_adv.item()

        # ---- L_edge: Edge-guided loss auxiliary (Eq.2.41) ----
        if self.use_edge_loss and self.edge_loss is not None:
            l_edge = self.edge_loss(sr, hr)
            losses['edge'] = l_edge.item()
        else:
            l_edge = torch.tensor(0.0, device=sr.device)
            losses['edge'] = 0.0

        # ---- L_VSD: VSD loss auxiliary (Eq.2.42) ----
        if self.use_vsd_loss and self.vsd_loss_fn is not None and vsd_module is not None:
            l_vsd = self.vsd_loss_fn(sr, hr, weed_mask, vsd_module)
            losses['vsd'] = l_vsd.item()
        else:
            l_vsd = torch.tensor(0.0, device=sr.device)
            losses['vsd'] = 0.0

        # ---- Total loss (Eq.2.43) ----
        total = (
            self.lambda_rec * l_rec
            + self.lambda_perc * l_perc
            + self.lambda_adv * l_adv
            + self.lambda_edge * l_edge
            + self.lambda_vsd * l_vsd
        )
        losses['total'] = total.item()

        return total, losses


class UAVIRE_DiscriminatorLoss(nn.Module):
    """
    Discriminator loss untuk UAV-IRE (RaGAN + VSD discriminator).
    """

    def __init__(self, lambda_vsd_d: float = 0.5):
        super().__init__()
        self.lambda_vsd_d = lambda_vsd_d
        self.adv_loss = AdversarialLossDiscriminator()

    def forward(
        self,
        real_preds: torch.Tensor,
        fake_preds: torch.Tensor,
        sr: Optional[torch.Tensor] = None,
        hr: Optional[torch.Tensor] = None,
        weed_mask: Optional[torch.Tensor] = None,
        vsd_module=None,
    ) -> Tuple[torch.Tensor, Dict]:
        losses = {}

        # RaGAN discriminator loss (sesuai IRE baseline)
        l_d = self.adv_loss(real_preds, fake_preds)
        losses['d_adv'] = l_d.item()

        # VSD discriminator loss (jika tersedia)
        if vsd_module is not None and sr is not None and hr is not None:
            l_vsd_d, vsd_d_dict = vsd_module(sr.detach(), hr, weed_mask, mode='discriminator')
            total = l_d + self.lambda_vsd_d * l_vsd_d
            losses['d_vsd'] = l_vsd_d.item()
        else:
            total = l_d

        losses['d_total'] = total.item()
        return total, losses


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Testing UAV-IRE Loss Functions...")

    device = torch.device('cpu')

    sr = torch.rand(2, 3, 128, 128, requires_grad=True)
    hr = torch.rand(2, 3, 128, 128)
    fake_preds = torch.randn(2, 1, 6, 6, requires_grad=True)
    real_preds = torch.randn(2, 1, 6, 6)
    mask = (torch.rand(2, 1, 128, 128) > 0.7).float()

    # Test individual loss components
    rec = SmoothL1ReconstructionLoss()
    l_rec = rec(sr, hr)
    print(f"L_rec (SmoothL1): {l_rec.item():.4f} ✓")

    edge = EdgeGuidedLoss()
    l_edge = edge(sr, hr)
    print(f"L_edge (Sobel): {l_edge.item():.4f} ✓")

    adv = AdversarialLossGAN()
    l_adv = adv(fake_preds, real_preds, use_ragan=True)
    print(f"L_adv (RaGAN): {l_adv.item():.4f} ✓")

    # Test total loss (tanpa VSD dan Perceptual untuk kecepatan)
    gen_loss = UAVIRE_GeneratorLoss(
        use_edge_loss=True,
        use_vsd_loss=False,  # Skip VSD untuk test cepat
    )
    # Temporarily disable perceptual loss for test speed
    gen_loss.lambda_perc = 0.0

    total, loss_dict = gen_loss(sr, hr, fake_preds, real_preds, mask)
    print(f"\nTotal G Loss: {total.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    # Test backward
    total.backward()
    assert sr.grad is not None
    assert fake_preds.grad is not None
    print("\nBackward pass: OK ✓")

    # Test Discriminator loss
    disc_loss = UAVIRE_DiscriminatorLoss()
    d_total, d_dict = disc_loss(real_preds, fake_preds.detach())
    print(f"\nTotal D Loss: {d_total.item():.4f}")

    print("\nUAV-IRE Loss Functions test PASSED!")
