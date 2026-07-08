"""
IRE: Improved Image Super-Resolution Based on Real-ESRGAN
Arsitektur Generator dan Discriminator sesuai paper Zhu et al. (2023)

Komponen utama:
- ChannelAttention: mekanisme channel attention (Zhang et al., RCAN)
- DenseBlock: dense block dengan channel attention menggantikan 2 conv terakhir
- RRDB: Residual-in-Residual Dense Block (dari ESRGAN)
- IRE_Generator: Generator berbasis RRDB + Channel Attention
- PatchGAN_Discriminator: Discriminator berbasis PatchGAN (Isola et al.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Channel Attention Mechanism (Zhang et al., ECCV 2018 - RCAN)
# Sesuai Fig.4 dan Eq.(1)(2)(3) paper IRE
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """
    Channel Attention Block.
    Menggantikan dua conv terakhir pada dense block (lihat Fig.8 paper IRE).

    Struktur: AdaptiveAvgPool -> Conv(descend) -> ReLU -> Conv(ascend) -> Sigmoid
    """

    def __init__(self, num_features: int, reduction: int = 16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                            # Global Average Pooling (Eq.1)
            nn.Conv2d(num_features, num_features // reduction, 1, bias=True),  # WD (descend)
            nn.ReLU(inplace=True),                              # R(.)
            nn.Conv2d(num_features // reduction, num_features, 1, bias=True),  # WU (ascend)
            nn.Sigmoid(),                                       # S(.) -> FC weights
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eq.(3): Ioutput = FC * xc
        return x * self.attention(x)


# ---------------------------------------------------------------------------
# Dense Block dengan Channel Attention (Fig.8 paper IRE - right side)
# ---------------------------------------------------------------------------

class DenseLayer(nn.Module):
    """Satu layer dalam dense block: Conv2d -> LeakyReLU."""

    def __init__(self, in_channels: int, growth_rate: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, 3, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lrelu(self.conv(x))


class ImprovedDenseBlock(nn.Module):
    """
    Improved Dense Block dengan Channel Attention (Fig.8 IRE, sisi kanan).

    Modifikasi dari dense block Real-ESRGAN:
    - 3 conv layers pertama tetap dengan dense connection
    - 2 conv layers terakhir DIGANTI dengan Channel Attention block
    
    Input: num_features channel
    Growth rate: 32 (default, seperti Real-ESRGAN)
    """

    def __init__(self, num_features: int = 64, growth_rate: int = 32):
        super().__init__()
        # 3 conv layers dengan dense connections
        self.layer1 = DenseLayer(num_features, growth_rate)
        self.layer2 = DenseLayer(num_features + growth_rate, growth_rate)
        self.layer3 = DenseLayer(num_features + 2 * growth_rate, growth_rate)
        # Final conv untuk mereduksi kembali ke num_features
        self.final_conv = nn.Conv2d(num_features + 3 * growth_rate, num_features, 1, bias=True)
        # Channel Attention menggantikan 2 conv terakhir
        self.channel_attention = ChannelAttention(num_features)
        # Residual scaling (beta = 0.2, dari ESRGAN)
        self.res_scale = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dense connections
        d1 = self.layer1(x)
        d2 = self.layer2(torch.cat([x, d1], dim=1))
        d3 = self.layer3(torch.cat([x, d1, d2], dim=1))
        # Gabungkan dan reduksi channel
        out = self.final_conv(torch.cat([x, d1, d2, d3], dim=1))
        # Terapkan channel attention
        out = self.channel_attention(out)
        # Residual connection dengan scaling
        return x + out * self.res_scale


# ---------------------------------------------------------------------------
# RRDB: Residual-in-Residual Dense Block (Wang et al., ESRGAN 2018)
# Sesuai Eq.(2.5) proposal: RRDB(x) = x + β * F_dense-res(x)
# ---------------------------------------------------------------------------

class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block.
    Berisi 3 ImprovedDenseBlock dengan outer residual connection.
    Sesuai paper ESRGAN (Wang et al.) yang digunakan di IRE.
    """

    def __init__(self, num_features: int = 64, growth_rate: int = 32):
        super().__init__()
        self.dense1 = ImprovedDenseBlock(num_features, growth_rate)
        self.dense2 = ImprovedDenseBlock(num_features, growth_rate)
        self.dense3 = ImprovedDenseBlock(num_features, growth_rate)
        self.res_scale = 0.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dense1(x)
        out = self.dense2(out)
        out = self.dense3(out)
        return x + out * self.res_scale


# ---------------------------------------------------------------------------
# IRE Generator (Fig.7 Real-ESRGAN generative network, dimodifikasi IRE)
# Struktur: Conv -> N x RRDB -> Conv -> Upsample x4 -> Conv -> Conv
# ---------------------------------------------------------------------------

class IRE_Generator(nn.Module):
    """
    Generator IRE (Improved Real-ESRGAN Generator).

    Struktur berbasis SRResNet (sub-pixel conv + residual depth module):
    - Feature extraction: Conv awal
    - Residual depth module: N buah RRDB (default 23, seperti Real-ESRGAN)
    - Sub-pixel convolution module: upsampling x4 via pixel shuffle

    Scale factor: 4x (sesuai paper IRE yang fokus pada 4x upscaling)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_rrdb: int = 23,
        growth_rate: int = 32,
        scale_factor: int = 2,
    ):
        super().__init__()
        self.scale_factor = scale_factor

        # Conv pertama: ekstraksi fitur awal
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, padding=1, bias=True)

        # Residual depth module: N buah RRDB
        self.rrdb_blocks = nn.Sequential(
            *[RRDB(num_features, growth_rate) for _ in range(num_rrdb)]
        )

        # Conv setelah RRDB (sebelum upsampling)
        self.conv_body = nn.Conv2d(num_features, num_features, 3, padding=1, bias=True)

        # Sub-pixel convolution (upsampling)
        # Untuk scale 4x: dua kali upsample 2x
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
            self._fractional_scale = False
        elif scale_factor == 2:
            upsample_layers += [
                nn.Conv2d(num_features, num_features * 4, 3, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            self._fractional_scale = False
        else:
            # CATATAN METODOLOGIS: cabang ini KHUSUS eksperimen kalibrasi Lanczos
            # scale pecahan (3/4, 4/5, 5/6, 9/10 dst -- kampanye K7), BUKAN bagian
            # arsitektur SR1-SR6 utama yang tetap pakai PixelShuffle scale=2.
            # PixelShuffle tidak bisa menangani scale non-integer (4/3, 5/4, dst),
            # jadi dipakai interpolation-based upsample (bicubic) + conv refine
            # sebagai gantinya. Ini deviasi dari sub-pixel conv asli Real-ESRGAN --
            # wajib dicatat sebagai batasan pada eksperimen kalibrasi kalau ditanya.
            upsample_layers += [
                nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False),
                nn.Conv2d(num_features, num_features, 3, padding=1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            self._fractional_scale = True
        self.upsample = nn.Sequential(*upsample_layers)
        self._target_size = None  # di-set eksternal sebelum forward() kalau fractional

        # Conv output
        self.conv_hr = nn.Conv2d(num_features, num_features, 3, padding=1, bias=True)
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # Inisialisasi bobot
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.rrdb_blocks(feat))
        feat = feat + body_feat  # Residual connection global
        feat = self.upsample(feat)
        if self._fractional_scale and self._target_size is not None:
            # Snap ke ukuran HR pasangan yang eksak (menghindari off-by-1 px
            # akibat pembulatan ganda saat downsample Lanczos lalu upsample bicubic)
            feat = torch.nn.functional.interpolate(
                feat, size=self._target_size, mode='bicubic', align_corners=False
            )
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ---------------------------------------------------------------------------
# PatchGAN Discriminator (Isola et al., 2017 - pix2pix)
# Sesuai Fig.9 paper IRE:
# Conv(K4n64s2) -> Conv(K4n128s2)+MaxPool+BN -> ... -> Conv(K4n512s1)+BN
# -> Conv(K4n512s1)+BN -> Conv(K1n2s2)
# ---------------------------------------------------------------------------

class PatchGAN_Discriminator(nn.Module):
    """
    PatchGAN Discriminator sesuai Fig.9 paper IRE.

    Arsitektur: interleaved Conv + MaxPool + BatchNorm + LeakyReLU
    Output: N*N matrix di mana setiap titik merepresentasikan
            probabilitas real/fake pada area patch lokal.

    Struktur berdasarkan diagram Fig.9:
    K4n64s2 -> K4n128s2+MaxPool+BN -> K4n256s2+MaxPool+BN 
    -> K4n512s2+MaxPool+BN -> K4n512s1+BN -> K4n512s1+BN -> K4n1s2
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        def conv_block(in_ch, out_ch, stride=1, use_bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # Sesuai diagram Fig.9 paper IRE
        # FIX: hapus MaxPool agar spatial size tidak terlalu kecil sebelum block5/block6
        self.block1 = conv_block(in_channels, 64, stride=2, use_bn=False)   # K4n64s2
        self.block2 = conv_block(64, 128, stride=2, use_bn=True)            # K4n128s2+BN
        self.block3 = conv_block(128, 256, stride=2, use_bn=True)           # K4n256s2+BN
        self.block4 = conv_block(256, 512, stride=2, use_bn=True)           # K4n512s2+BN
        self.block5 = conv_block(512, 512, stride=1, use_bn=True)           # K4n512s1+BN
        self.block6 = conv_block(512, 512, stride=1, use_bn=True)           # K4n512s1+BN

        # Output layer: K4n1s2 - menghasilkan peta probabilitas patch
        self.output = nn.Conv2d(512, 1, 4, stride=1, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
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
# VGG Feature Extractor untuk Perceptual Loss
# ---------------------------------------------------------------------------

class VGGFeatureExtractor(nn.Module):
    """
    Ekstraksi fitur VGG19 untuk perceptual loss.
    Menggunakan fitur SEBELUM aktivasi (pre-activation), sesuai ESRGAN/IRE.
    Layer default: features setelah conv3_4 (layer index 26).
    """

    def __init__(self, feature_layer: int = 34, use_input_norm: bool = True):
        super().__init__()
        from torchvision import models
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Ambil features hingga layer yang ditentukan
        self.features = nn.Sequential(*list(vgg.features.children())[:feature_layer])
        self.use_input_norm = use_input_norm

        if use_input_norm:
            # Normalisasi ImageNet
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        return self.features(x)


# ---------------------------------------------------------------------------
# Ringkasan model (untuk debugging)
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test Generator
    gen = IRE_Generator(num_rrdb=6).to(device)  # Reduced RRDB for quick test
    lr_img = torch.randn(1, 3, 64, 64).to(device)
    with torch.no_grad():
        sr_img = gen(lr_img)
    print(f"Generator - Input: {lr_img.shape} -> Output: {sr_img.shape}")
    print(f"Generator parameters: {count_parameters(gen):,}")

    # Test Discriminator
    disc = PatchGAN_Discriminator().to(device)
    hr_img = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        disc_out = disc(hr_img)
    print(f"Discriminator - Input: {hr_img.shape} -> Output: {disc_out.shape}")
    print(f"Discriminator parameters: {count_parameters(disc):,}")

    # Test VGG
    vgg = VGGFeatureExtractor().to(device)
    with torch.no_grad():
        feat = vgg(hr_img)
    print(f"VGG Features - Input: {hr_img.shape} -> Features: {feat.shape}")