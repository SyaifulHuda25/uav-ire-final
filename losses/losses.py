"""
IRE Loss Functions
Sesuai paper IRE (Zhu et al., 2023) Section III.C dan Eq.(6-17)

Loss yang digunakan:
- SmoothL1 Loss (menggantikan L1 dari Real-ESRGAN) - Eq.(7)(10)
- Perceptual Loss berbasis VGG dengan SmoothL1 - Eq.(12)
- Adversarial Loss (RaGAN) - Eq.(15)(16)(17)

Total loss: L_IRE = L_SmoothL1 + L_VGG/i,j + 0.1 * L_GAN  -- Eq.(6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# SmoothL1 Loss (Girshick, Fast R-CNN 2015)
# Sesuai Eq.(7)(8) paper IRE
# ---------------------------------------------------------------------------

class SmoothL1Loss(nn.Module):
    """
    SmoothL1 Loss (Huber Loss):
        SmoothL1(x) = 0.5 * x^2       if |x| < 1
                    = |x| - 0.5        otherwise

    Sesuai Eq.(7) paper IRE.
    Lebih stabil dari L1: gradient terbatas untuk outlier besar,
    dan gradient halus untuk perbedaan kecil.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(reduction=reduction, beta=1.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)


# ---------------------------------------------------------------------------
# Perceptual Loss berbasis VGG dengan SmoothL1
# Sesuai Eq.(11)(12) paper IRE
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """
    Perceptual Loss dengan VGG19 + SmoothL1.

    Menggunakan fitur VGG SEBELUM aktivasi (pre-activation),
    sesuai ESRGAN dan IRE (berbeda dengan SRGAN yang post-activation).

    Sesuai Eq.(12) paper IRE:
    L_SmoothL1_VGG = 1/(W*H) * Σ SmoothL1(φ(IHR) - φ(G(ILR)))
    """

    def __init__(self, feature_layer: int = 34):
        super().__init__()
        # Import dari network.py
        from models.network import VGGFeatureExtractor
        self.vgg = VGGFeatureExtractor(feature_layer=feature_layer)
        self.smooth_l1 = SmoothL1Loss(reduction='mean')

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # Ekstraksi fitur VGG
        sr_features = self.vgg(sr)
        with torch.no_grad():
            hr_features = self.vgg(hr)

        # SmoothL1 pada ruang fitur
        return self.smooth_l1(sr_features, hr_features)


# ---------------------------------------------------------------------------
# Adversarial Loss (RaGAN - Relativistic average GAN)
# Sesuai Eq.(15)(16)(17) paper IRE
# ---------------------------------------------------------------------------

class RaGAN_GeneratorLoss(nn.Module):
    """
    Generator adversarial loss menggunakan Relativistic average GAN (RaGAN).

    Sesuai Eq.(16) paper IRE:
    L_G_Ra = -E_IHR[log(1 - DRa(IHR, ISR))]
           - E_ISR[log(DRa(ISR, IHR))]

    DRa(a, b) = σ(D(a) - E[D(b)])
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        real_preds: torch.Tensor,   # D(IHR)
        fake_preds: torch.Tensor,   # D(ISR)
    ) -> torch.Tensor:
        # DRa(IHR, ISR) = σ(D(IHR) - E[D(ISR)])
        real_mean = real_preds.mean()
        fake_mean = fake_preds.mean()

        # DRa_real = σ(D(IHR) - E[D(ISR)])
        # DRa_fake = σ(D(ISR) - E[D(IHR)])
        loss_real = F.binary_cross_entropy_with_logits(
            real_preds - fake_mean,
            torch.zeros_like(real_preds)   # Target 0: DRa(IHR,ISR) -> 0 (generator view)
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            fake_preds - real_mean,
            torch.ones_like(fake_preds)    # Target 1: DRa(ISR,IHR) -> 1
        )

        return (loss_real + loss_fake) / 2


class RaGAN_DiscriminatorLoss(nn.Module):
    """
    Discriminator adversarial loss menggunakan RaGAN.

    Sesuai Eq.(17) paper IRE:
    L_D_Ra = -E_IHR[log(DRa(IHR, ISR))]
           - E_ISR[log(1 - DRa(ISR, IHR))]
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        real_preds: torch.Tensor,   # D(IHR)
        fake_preds: torch.Tensor,   # D(ISR)
    ) -> torch.Tensor:
        real_mean = real_preds.mean()
        fake_mean = fake_preds.mean()

        loss_real = F.binary_cross_entropy_with_logits(
            real_preds - fake_mean,
            torch.ones_like(real_preds)    # Target 1: real harus dianggap real
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            fake_preds - real_mean,
            torch.zeros_like(fake_preds)   # Target 0: fake harus dianggap fake
        )

        return (loss_real + loss_fake) / 2


# ---------------------------------------------------------------------------
# IRE Total Loss - mengintegrasikan semua komponen
# Sesuai Eq.(6): L_IRE = L_SmoothL1 + L_VGG/i,j + 0.1 * L_GAN
# ---------------------------------------------------------------------------

class IRE_GeneratorLoss(nn.Module):
    """
    Total generator loss untuk IRE.

    Sesuai Eq.(6):
    L_IRE = L_SmoothL1 + L_VGG/i,j + 0.1 * L_GAN

    Komponen:
    - pixel_loss: SmoothL1 pada pixel space
    - perceptual_loss: SmoothL1 pada VGG feature space
    - adversarial_loss: RaGAN generator loss
    """

    def __init__(
        self,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        adversarial_weight: float = 0.1,
        vgg_feature_layer: int = 34,
    ):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight

        self.pixel_loss = SmoothL1Loss()
        self.perceptual_loss = PerceptualLoss(feature_layer=vgg_feature_layer)
        self.adversarial_loss = RaGAN_GeneratorLoss()

    def forward(
        self,
        sr: torch.Tensor,           # Gambar SR dari generator
        hr: torch.Tensor,           # Gambar HR (ground truth)
        real_preds: torch.Tensor,   # D(HR) dari discriminator
        fake_preds: torch.Tensor,   # D(SR) dari discriminator
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: total loss tensor
            loss_dict: dictionary berisi komponen loss untuk logging
        """
        # Pixel loss (SmoothL1)
        l_pixel = self.pixel_loss(sr, hr)

        # Perceptual loss (VGG + SmoothL1)
        l_percep = self.perceptual_loss(sr, hr)

        # Adversarial loss (RaGAN)
        l_adv = self.adversarial_loss(real_preds, fake_preds)

        # Total loss sesuai Eq.(6)
        total = (
            self.pixel_weight * l_pixel
            + self.perceptual_weight * l_percep
            + self.adversarial_weight * l_adv
        )

        loss_dict = {
            'total': total.item(),
            'pixel': l_pixel.item(),
            'perceptual': l_percep.item(),
            'adversarial': l_adv.item(),
        }

        return total, loss_dict


class IRE_DiscriminatorLoss(nn.Module):
    """Discriminator loss untuk IRE (RaGAN)."""

    def __init__(self):
        super().__init__()
        self.disc_loss = RaGAN_DiscriminatorLoss()

    def forward(
        self,
        real_preds: torch.Tensor,
        fake_preds: torch.Tensor,
    ) -> torch.Tensor:
        return self.disc_loss(real_preds, fake_preds)


# ---------------------------------------------------------------------------
# Test loss functions
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test SmoothL1
    smooth_l1 = SmoothL1Loss()
    a = torch.randn(2, 3, 64, 64)
    b = torch.randn(2, 3, 64, 64)
    loss = smooth_l1(a, b)
    print(f"SmoothL1 Loss: {loss.item():.4f}")

    # Test RaGAN losses
    gen_loss_fn = RaGAN_GeneratorLoss()
    disc_loss_fn = RaGAN_DiscriminatorLoss()

    real_preds = torch.randn(2, 1, 8, 8)
    fake_preds = torch.randn(2, 1, 8, 8)

    g_loss = gen_loss_fn(real_preds, fake_preds)
    d_loss = disc_loss_fn(real_preds, fake_preds)
    print(f"Generator RaGAN Loss: {g_loss.item():.4f}")
    print(f"Discriminator RaGAN Loss: {d_loss.item():.4f}")

    print("Loss functions test PASSED!")
