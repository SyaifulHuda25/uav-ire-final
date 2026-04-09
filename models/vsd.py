"""
Vegetation Similarity Discriminator (VSD)
Sesuai proposal tesis Section 2.11.4 dan Eq.(2.24)-(2.29)

Motivasi: Padi budidaya dan gulma (weedy rice) memiliki kemiripan visual tinggi.
Discriminator global PatchGAN tidak memiliki kesadaran domain vegetasi, sehingga
kontribusi area gulma yang kecil dalam loss adversarial global menjadi lemah.

VSD terdiri dari dua komponen:
1. Global Discriminator (D_global): PatchGAN untuk menilai realisme global (Eq.2.24-2.25)
2. Weed-Specific Discriminator (D_VSD-W): Fokus pada area gulma via mask (Eq.2.26-2.27)
3. Vegetation Structure Consistency Loss (Eq.2.28)
4. Total VSD Loss (Eq.2.29): λg*L_global + λw*L_weed + λv*L_veg

Referensi:
- Zhu et al. (2023): PatchGAN discriminator dari IRE (D_global)
- Park et al. (2019): SPADE - region-aware GAN untuk deteksi objek spesifik
- Liu et al. (2020): plant phenotyping CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Global Discriminator (D_global) - PatchGAN diperkuat
# Sesuai Eq.(2.24)(2.25)
# ---------------------------------------------------------------------------

class GlobalDiscriminator(nn.Module):
    """
    Global Discriminator berbasis PatchGAN (dari IRE) yang diperkuat.
    Sesuai Eq.(2.24): D_global: I -> P_rf ∈ R^{h×w}
    Sesuai Eq.(2.25): L_global_adv = E[log D_global(IHR)] + E[log(1 - D_global(ISR))]

    Mengadopsi arsitektur PatchGAN dari IRE dengan lapisan konvolusi dan max-pooling.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        def conv_block(in_ch, out_ch, stride=1, use_bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1,
                                bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.block1 = conv_block(in_channels, 64, stride=2, use_bn=False)
        self.block2 = conv_block(64, 128, stride=2, use_bn=True)
        self.block3 = conv_block(128, 256, stride=2, use_bn=True)
        self.block4 = conv_block(256, 512, stride=2, use_bn=True)
        self.block5 = conv_block(512, 512, stride=1, use_bn=True)
        self.block6 = conv_block(512, 512, stride=1, use_bn=True)
        self.output = nn.Conv2d(512, 1, 4, stride=1, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return self.output(x)


# ---------------------------------------------------------------------------
# Weed-Specific Discriminator (D_VSD-W)
# Sesuai Eq.(2.26)(2.27)
# ---------------------------------------------------------------------------

class WeedSpecificDiscriminator(nn.Module):
    """
    Discriminator khusus vegetasi gulma.
    Sesuai Eq.(2.26): I_SR_weed = ISR ⊙ M_weed
    Sesuai Eq.(2.27): L_weed_adv = E[log D_VSD-W(I_HR_weed)] + E[log(1-D_VSD-W(I_SR_weed))]

    Terinspirasi dari konsep region-aware multi-discriminator GAN (Park et al., 2019).
    Hanya memproses area citra yang ditandai mask gulma, sehingga fokus
    pembelajaran terarah pada struktur gulma (area minoritas).

    Arsitektur lebih sederhana dari D_global karena input sudah dipangkas ke ROI.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        def conv_block(in_ch, out_ch, stride=2, use_bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1,
                                bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            conv_block(in_channels, 64, use_bn=False),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_masked: Citra yang sudah dimasking dengan M_weed (Eq.2.26)
        """
        return self.net(x_masked)


# ---------------------------------------------------------------------------
# Vegetation Feature Extractor untuk Structure Consistency Loss
# Sesuai Eq.(2.28)
# ---------------------------------------------------------------------------

class VegetationFeatureExtractor(nn.Module):
    """
    CNN untuk mengekstraksi fitur vegetasi.
    Sesuai Eq.(2.28): L_veg = ||φ(I_SR_weed) - φ(I_HR_weed)||²₂

    Digunakan untuk mengukur kesesuaian struktur daun dan pola kanopi.
    Terinspirasi dari plant phenotyping CNN (Liu et al., 2020).

    Implementasi menggunakan VGG-like lightweight feature extractor
    yang spesifik untuk menangkap perbedaan tekstur vegetasi.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: Low-level features (edges, textures)
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: Mid-level features (struktur daun, serat)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: High-level features (pola kanopi, densitas rumpun)
            nn.Conv2d(128, feature_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Freeze setelah training awal (opsional)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# ---------------------------------------------------------------------------
# VSD - Vegetation Similarity Discriminator (lengkap)
# Sesuai Eq.(2.24)-(2.29)
# ---------------------------------------------------------------------------

class VSD(nn.Module):
    """
    Vegetation Similarity Discriminator (VSD).

    Mengintegrasikan:
    - D_global: PatchGAN untuk realisme global
    - D_VSD-W: Discriminator khusus area gulma
    - L_veg: Vegetation structure consistency loss
    
    Total loss (Eq.2.29):
    L_VSD = λg * L_global_adv + λw * L_weed_adv + λv * L_veg

    Args:
        lambda_g: Bobot global adversarial loss
        lambda_w: Bobot weed-specific adversarial loss
        lambda_v: Bobot vegetation structure consistency loss
    """

    def __init__(
        self,
        in_channels: int = 3,
        lambda_g: float = 1.0,
        lambda_w: float = 0.5,
        lambda_v: float = 0.1,
    ):
        super().__init__()
        self.lambda_g = lambda_g
        self.lambda_w = lambda_w
        self.lambda_v = lambda_v

        # Komponen discriminator
        self.d_global = GlobalDiscriminator(in_channels)
        self.d_weed = WeedSpecificDiscriminator(in_channels)
        self.veg_extractor = VegetationFeatureExtractor(in_channels)

    def apply_mask(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Terapkan mask gulma ke gambar (Eq.2.26).
        M_weed ∈ {0,1}^{H×W}

        Jika mask None, gunakan semua piksel (fallback untuk kasus tanpa anotasi).
        """
        if mask is None:
            return img
        # Pastikan mask shape [B, 1, H, W]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] != 1:
            mask = mask[:, :1]
        return img * mask.float()

    def compute_global_adv_loss(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hitung global adversarial loss untuk Generator dan Discriminator.
        Sesuai Eq.(2.25) menggunakan standard GAN loss.

        Returns:
            (loss_g, loss_d): Generator loss dan Discriminator loss
        """
        real_preds = self.d_global(hr)
        fake_preds = self.d_global(sr)

        # Generator loss: ingin D_global(SR) = 1
        loss_g = F.binary_cross_entropy_with_logits(
            fake_preds, torch.ones_like(fake_preds)
        )
        # Discriminator loss
        loss_d = (
            F.binary_cross_entropy_with_logits(real_preds, torch.ones_like(real_preds))
            + F.binary_cross_entropy_with_logits(fake_preds.detach(), torch.zeros_like(fake_preds))
        ) / 2

        return loss_g, loss_d

    def compute_weed_adv_loss(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hitung weed-specific adversarial loss.
        Sesuai Eq.(2.26)(2.27).

        Returns:
            (loss_g, loss_d)
        """
        # Eq.(2.26): masking area gulma
        sr_weed = self.apply_mask(sr, mask)
        hr_weed = self.apply_mask(hr, mask)

        real_preds = self.d_weed(hr_weed)
        fake_preds = self.d_weed(sr_weed)

        # Eq.(2.27): L_weed_adv
        loss_g = F.binary_cross_entropy_with_logits(
            fake_preds, torch.ones_like(fake_preds)
        )
        loss_d = (
            F.binary_cross_entropy_with_logits(real_preds, torch.ones_like(real_preds))
            + F.binary_cross_entropy_with_logits(fake_preds.detach(), torch.zeros_like(fake_preds))
        ) / 2

        return loss_g, loss_d

    def compute_veg_consistency_loss(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Vegetation structure consistency loss.
        Sesuai Eq.(2.28): L_veg = ||φ(I_SR_weed) - φ(I_HR_weed)||²₂
        """
        sr_weed = self.apply_mask(sr, mask)
        hr_weed = self.apply_mask(hr, mask)

        sr_feat = self.veg_extractor(sr_weed)
        with torch.no_grad():
            hr_feat = self.veg_extractor(hr_weed)

        return F.mse_loss(sr_feat, hr_feat)

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mode: str = 'generator',   # 'generator' atau 'discriminator'
    ) -> Tuple[torch.Tensor, dict]:
        """
        Hitung total VSD loss.
        Sesuai Eq.(2.29): L_VSD = λg*L_global + λw*L_weed + λv*L_veg

        Args:
            sr: SR gambar [B, C, H, W]
            hr: HR gambar [B, C, H, W]
            mask: Mask gulma [B, 1/H/W, H, W] (opsional)
            mode: 'generator' atau 'discriminator'

        Returns:
            (total_loss, loss_dict)
        """
        # Global adversarial loss (Eq.2.25)
        loss_g_global, loss_d_global = self.compute_global_adv_loss(sr, hr)

        # Weed-specific loss (Eq.2.27)
        loss_g_weed, loss_d_weed = self.compute_weed_adv_loss(sr, hr, mask)

        # Vegetation structure consistency (Eq.2.28)
        loss_veg = self.compute_veg_consistency_loss(sr, hr, mask)

        if mode == 'generator':
            # Eq.(2.29): L_VSD = λg*L_global + λw*L_weed + λv*L_veg
            total = (
                self.lambda_g * loss_g_global
                + self.lambda_w * loss_g_weed
                + self.lambda_v * loss_veg
            )
            loss_dict = {
                'vsd_total': total.item(),
                'vsd_global_g': loss_g_global.item(),
                'vsd_weed_g': loss_g_weed.item(),
                'vsd_veg': loss_veg.item(),
            }
        else:  # discriminator
            total = (
                self.lambda_g * loss_d_global
                + self.lambda_w * loss_d_weed
            )
            loss_dict = {
                'vsd_d_total': total.item(),
                'vsd_d_global': loss_d_global.item(),
                'vsd_d_weed': loss_d_weed.item(),
            }

        return total, loss_dict


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Testing VSD components...")

    device = torch.device('cpu')
    B, C, H, W = 2, 3, 128, 128

    sr = torch.rand(B, C, H, W)
    hr = torch.rand(B, C, H, W)
    # Mask gulma: area acak ~30% piksel
    mask = (torch.rand(B, 1, H, W) > 0.7).float()

    # Test GlobalDiscriminator
    d_global = GlobalDiscriminator()
    preds = d_global(hr)
    print(f"GlobalDiscriminator: {hr.shape} -> {preds.shape} ✓")

    # Test WeedSpecificDiscriminator
    d_weed = WeedSpecificDiscriminator()
    hr_masked = hr * mask
    preds_weed = d_weed(hr_masked)
    print(f"WeedSpecificDiscriminator: {hr_masked.shape} -> {preds_weed.shape} ✓")

    # Test VegetationFeatureExtractor
    vfe = VegetationFeatureExtractor()
    feats = vfe(hr)
    print(f"VegetationFeatureExtractor: {hr.shape} -> {feats.shape} ✓")

    # Test VSD penuh - generator mode
    vsd = VSD(lambda_g=1.0, lambda_w=0.5, lambda_v=0.1)

    sr_req = sr.requires_grad_(True)
    loss_g, loss_dict_g = vsd(sr_req, hr, mask, mode='generator')
    print(f"VSD Generator Loss: {loss_g.item():.4f}")
    print(f"  Components: {loss_dict_g}")

    loss_g.backward()
    assert sr_req.grad is not None
    print("VSD generator backward: OK ✓")

    # Test discriminator mode
    loss_d, loss_dict_d = vsd(sr.detach(), hr, mask, mode='discriminator')
    print(f"VSD Discriminator Loss: {loss_d.item():.4f}")
    print(f"  Components: {loss_dict_d}")

    # Test tanpa mask (fallback)
    loss_no_mask, _ = vsd(sr, hr, mask=None, mode='generator')
    print(f"VSD (no mask): {loss_no_mask.item():.4f} ✓")

    n_params = sum(p.numel() for p in vsd.parameters())
    print(f"VSD total parameters: {n_params:,}")

    print("\nVSD test PASSED!")
