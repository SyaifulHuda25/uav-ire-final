"""
UAV-IRE Generator (Ablation-Ready)
Sesuai proposal tesis Section 2.11 dan Gambar 3.1 (Diagram Alir)

CATATAN ABLASI:
- use_nrdb, use_mbcm, use_ega  -> flag ARSITEKTUR GENERATOR (mempengaruhi forward pass & inferensi)
- VSD BUKAN bagian generator ini. VSD adalah discriminator tambahan yang hanya dipakai
  saat training (loss). Untuk ablasi VSD, ganti discriminator/loss di training script
  (gunakan PatchGAN_Discriminator biasa vs VSD), generator di file ini TIDAK berubah.

Perilaku dual-branch:
- use_nrdb=True & use_mbcm=True   -> dual-branch penuh (NRDB path + MBCM path) + Fusion
                                     (arsitektur original, RRDB dibagi 2 antar cabang)
- Hanya salah satu True           -> single-branch (NRDB saja / MBCM saja),
                                     RRDB TIDAK dibagi 2 (dapat kapasitas penuh num_rrdb)
                                     supaya perbandingan adil terhadap baseline
- Keduanya False                  -> baseline IRE murni: conv_first -> RRDB -> conv_body
                                     (tidak ada fusion, tidak ada noise/blur branch)

Mapping 6 skenario ablasi yang direncanakan:

  1. IRE (baseline)        : use_nrdb=False, use_mbcm=False, use_ega=False | training: use_vsd=False
  2. IRE+NRDB               : use_nrdb=True,  use_mbcm=False, use_ega=False | training: use_vsd=False
  3. IRE+NRDB+MBCM          : use_nrdb=True,  use_mbcm=True,  use_ega=False | training: use_vsd=False
  4. IRE+EGA (standalone)   : use_nrdb=False, use_mbcm=False, use_ega=True  | training: use_vsd=False
  5. IRE+VSD (standalone)   : use_nrdb=False, use_mbcm=False, use_ega=False | training: use_vsd=True
  6. IRE full               : use_nrdb=True,  use_mbcm=True,  use_ega=True  | training: use_vsd=True
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.network import RRDB, ChannelAttention, PatchGAN_Discriminator
from models.nrdb import NRDB
from models.mbcm import MBCM
from models.ega import EGA


class FusionBlock(nn.Module):
    """
    Fusi fitur dari jalur NRDB dan MBCM.
    Cross-Scale Fusion: menggabungkan informasi dari dua jalur berbeda.
    Sesuai pendekatan CS-IRE (Jin-li et al., 2025) yang diadopsi dalam proposal.
    Hanya dipakai ketika KEDUA cabang (NRDB & MBCM) aktif.
    """

    def __init__(self, num_features: int = 64):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1, bias=True),
        )

    def forward(self, feat_nrdb: torch.Tensor, feat_mbcm: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([feat_nrdb, feat_mbcm], dim=1)
        return self.fusion_conv(fused)


class UAVIRE_Generator(nn.Module):
    """
    UAV-IRE Generator dengan flag ablasi.

    Args tambahan (baru):
        use_nrdb: aktifkan cabang NRDB (denoising)
        use_mbcm: aktifkan cabang MBCM (deblurring)
        use_ega:  aktifkan Edge-Guided Attention sebelum upsampling

    Semua kombinasi flag tetap menghasilkan generator yang valid dan
    fully-convolutional (bisa dipakai untuk inferensi full-size / sliding window).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_rrdb: int = 23,
        growth_rate: int = 32,
        scale_factor: int = 2,
        num_nrdb_layers: int = 3,
        num_directions: int = 4,
        num_blur_layers: int = 3,
        use_nrdb: bool = True,
        use_mbcm: bool = True,
        use_ega: bool = True,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.num_features = num_features
        self.use_nrdb = use_nrdb
        self.use_mbcm = use_mbcm
        self.use_ega = use_ega
        self.num_active_branches = int(use_nrdb) + int(use_mbcm)

        # ============================================================
        # Stage 1: Feature Extraction (Conv awal) — selalu ada
        # ============================================================
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, padding=1, bias=True)

        # ============================================================
        # Stage 2: Preprocessing branch(es) — jumlah & bentuk tergantung flag
        # ============================================================
        if self.num_active_branches == 2:
            # --- Dual-branch penuh (arsitektur original) ---
            self.nrdb = NRDB(num_features, num_nrdb_layers)
            self.mbcm = MBCM(num_features, num_directions, num_blur_layers)
            n_rrdb_branch = max(num_rrdb // 2, 3)
            self.rrdb_nrdb_path = nn.Sequential(
                *[RRDB(num_features, growth_rate) for _ in range(n_rrdb_branch)]
            )
            self.rrdb_mbcm_path = nn.Sequential(
                *[RRDB(num_features, growth_rate) for _ in range(n_rrdb_branch)]
            )
            self.fusion = FusionBlock(num_features)

        elif self.num_active_branches == 1:
            # --- Single-branch: NRDB saja ATAU MBCM saja ---
            if use_nrdb:
                self.nrdb = NRDB(num_features, num_nrdb_layers)
            if use_mbcm:
                self.mbcm = MBCM(num_features, num_directions, num_blur_layers)
            # RRDB TIDAK dibagi 2 -> kapasitas penuh, adil dibanding baseline
            self.rrdb_single_path = nn.Sequential(
                *[RRDB(num_features, growth_rate) for _ in range(num_rrdb)]
            )

        else:
            # --- Baseline IRE murni: tanpa NRDB, tanpa MBCM ---
            self.rrdb_single_path = nn.Sequential(
                *[RRDB(num_features, growth_rate) for _ in range(num_rrdb)]
            )

        self.conv_body = nn.Conv2d(num_features, num_features, 3, padding=1, bias=True)

        # ============================================================
        # Stage 3: Edge-Guided Attention (opsional)
        # ============================================================
        if self.use_ega:
            self.ega = EGA(num_features)

        # ============================================================
        # Stage 4: Upsampling ×scale_factor (via Pixel Shuffle) — selalu ada
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
        # Stage 5: Output Convolutions — selalu ada
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
        Forward pass UAV-IRE Generator. Path yang dijalankan tergantung
        flag use_nrdb / use_mbcm yang di-set saat __init__.
        """
        feat = self.conv_first(x)  # [B, F, H, W]

        if self.num_active_branches == 2:
            # Dual-branch: NRDB path + MBCM path -> Fusion
            feat_nrdb = self.nrdb(feat)
            feat_nrdb = self.rrdb_nrdb_path(feat_nrdb)

            feat_mbcm = self.mbcm(feat)
            feat_mbcm = self.rrdb_mbcm_path(feat_mbcm)

            feat_processed = self.fusion(feat_nrdb, feat_mbcm)

        elif self.num_active_branches == 1:
            # Single-branch: NRDB saja atau MBCM saja
            if self.use_nrdb:
                branch_feat = self.nrdb(feat)
            else:
                branch_feat = self.mbcm(feat)
            feat_processed = self.rrdb_single_path(branch_feat)

        else:
            # Baseline IRE: langsung RRDB tanpa preprocessing branch
            feat_processed = self.rrdb_single_path(feat)

        body_feat = self.conv_body(feat_processed)
        feat = feat + body_feat  # Global residual connection

        if self.use_ega:
            feat = self.ega(feat)  # Edge-enhanced

        feat = self.upsample(feat)  # [B, F, H*scale, W*scale]

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def active_modules(self) -> str:
        """Ringkasan modul aktif, berguna untuk logging tiap eksperimen ablasi."""
        mods = []
        if self.use_nrdb:
            mods.append("NRDB")
        if self.use_mbcm:
            mods.append("MBCM")
        if self.use_ega:
            mods.append("EGA")
        return "IRE" + ("+" + "+".join(mods) if mods else "")


# ---------------------------------------------------------------------------
# UAV-IRE Complete Model (Generator + VSD + PatchGAN)
# ---------------------------------------------------------------------------

class UAVIRE_Model(nn.Module):
    """
    Model UAV-IRE lengkap untuk training.

    use_vsd mengontrol discriminator mana yang dipakai saat training:
      - use_vsd=False -> PatchGAN_Discriminator biasa (setara skenario tanpa VSD)
      - use_vsd=True  -> VSD (Global + Weed-specific + Vegetation feature)

    generator_kwargs diteruskan langsung ke UAVIRE_Generator, misal:
      generator_kwargs=dict(use_nrdb=True, use_mbcm=False, use_ega=False)
    """

    def __init__(
        self,
        num_rrdb: int = 23,
        scale_factor: int = 2,
        num_features: int = 64,
        use_vsd: bool = True,
        generator_kwargs: dict = None,
        # VSD lambdas (hanya dipakai jika use_vsd=True)
        vsd_lambda_g: float = 1.0,
        vsd_lambda_w: float = 0.5,
        vsd_lambda_v: float = 0.1,
    ):
        super().__init__()
        generator_kwargs = generator_kwargs or {}

        self.use_vsd = use_vsd

        self.generator = UAVIRE_Generator(
            num_rrdb=num_rrdb,
            scale_factor=scale_factor,
            num_features=num_features,
            **generator_kwargs,
        )

        if use_vsd:
            from models.vsd import VSD
            self.discriminator = VSD(
                lambda_g=vsd_lambda_g,
                lambda_w=vsd_lambda_w,
                lambda_v=vsd_lambda_v,
            )
        else:
            self.discriminator = PatchGAN_Discriminator()

    def generate(self, lr: torch.Tensor) -> torch.Tensor:
        """Super-resolve LR image."""
        return self.generator(lr)

    def discriminate(self, *args, **kwargs):
        """
        PatchGAN: discriminate(img) -> skor realness
        VSD:      discriminate(sr, hr, mask=None) -> loss gabungan (lihat models/vsd.py)
        Signature berbeda tergantung use_vsd; sesuaikan pemanggilan di training loop.
        """
        return self.discriminator(*args, **kwargs)


# ---------------------------------------------------------------------------
# Konfigurasi Ablasi — BEBAS ON/OFF, tidak terikat skema SR1-SR6
# ---------------------------------------------------------------------------
#
# use_nrdb, use_mbcm, use_ega, use_vsd masing-masing independen (True/False).
# Total ada 2x2x2x2 = 16 kombinasi valid. Tidak ada dependency antar modul:
# kamu bisa nyalakan VSD saja tanpa NRDB/MBCM/EGA, atau EGA+VSD saja, dsb.
#
# Cara pakai bebas (tanpa dictionary preset apapun):
#
#   model = UAVIRE_Model(
#       generator_kwargs=dict(use_nrdb=True, use_mbcm=False, use_ega=True),
#       use_vsd=False,
#   )
#
# Fungsi di bawah ini cuma helper opsional untuk (1) bikin nama run otomatis,
# dan (2) generate semua/sebagian kombinasi untuk full-factorial study kalau mau.

import itertools


def build_config(use_nrdb: bool, use_mbcm: bool, use_ega: bool, use_vsd: bool) -> dict:
    """Bikin config lengkap (generator_kwargs + use_vsd) dari 4 flag bebas."""
    return {
        "generator_kwargs": dict(use_nrdb=use_nrdb, use_mbcm=use_mbcm, use_ega=use_ega),
        "use_vsd": use_vsd,
    }


def run_name(use_nrdb: bool, use_mbcm: bool, use_ega: bool, use_vsd: bool) -> str:
    """Nama run otomatis buat logging/folder checkpoint, misal: IRE_NRDB_EGA_VSD"""
    mods = []
    if use_nrdb:
        mods.append("NRDB")
    if use_mbcm:
        mods.append("MBCM")
    if use_ega:
        mods.append("EGA")
    if use_vsd:
        mods.append("VSD")
    return "IRE" + ("_" + "_".join(mods) if mods else "_baseline")


def all_16_combinations() -> dict:
    """
    Generate seluruh 16 kombinasi (full factorial 2^4) sebagai dict
    {nama_run: config}. Berguna kalau suatu saat mau ablasi menyeluruh,
    bukan cuma 6 titik SR1-SR6.
    """
    combos = {}
    for use_nrdb, use_mbcm, use_ega, use_vsd in itertools.product([False, True], repeat=4):
        name = run_name(use_nrdb, use_mbcm, use_ega, use_vsd)
        combos[name] = build_config(use_nrdb, use_mbcm, use_ega, use_vsd)
    return combos


# --- Preset SR1-SR6 (opsional, cuma shortcut kalau masih mau ikut skema tesis) ---
# Silakan pilih subset mana pun dari all_16_combinations() sesuai kebutuhan --
# preset ini TIDAK wajib dipakai.
ABLATION_CONFIGS_SR1_SR6 = {
    "SR1_baseline": build_config(False, False, False, False),
    "SR2_no_nrdb": build_config(False, True, True, True),
    "SR3_no_mbcm": build_config(True, False, True, True),
    "SR4_no_ega": build_config(True, True, False, True),
    "SR5_no_vsd": build_config(True, True, True, False),
    "SR6_full": build_config(True, True, True, True),
}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing UAV-IRE Generator - full factorial 16 kombinasi on/off, on {device}...\n")

    all_combos = all_16_combinations()
    print(f"Total kombinasi yang diuji: {len(all_combos)} (2^4: NRDB x MBCM x EGA x VSD)\n")

    for name, cfg in all_combos.items():
        gen_cfg = cfg["generator_kwargs"]

        gen = UAVIRE_Generator(
            num_rrdb=4,       # kecil untuk test cepat
            scale_factor=2,
            num_features=32,
            **gen_cfg,
        ).to(device)

        lr = torch.randn(1, 3, 32, 32).to(device)
        with torch.no_grad():
            sr = gen(lr)

        assert sr.shape == (1, 3, 64, 64), f"[{name}] Expected (1,3,64,64), got {sr.shape}"

        lr_grad = torch.randn(1, 3, 32, 32, requires_grad=True).to(device)
        sr_grad = gen(lr_grad)
        sr_grad.mean().backward()
        assert lr_grad.grad is not None

        n_params = gen.count_parameters()
        vsd_flag = "VSD=ON " if cfg["use_vsd"] else "VSD=off"
        print(f"[{name:22s}] {vsd_flag} | out={tuple(sr.shape)} "
              f"params={n_params:,} | forward+backward OK")

    print(f"\nSemua {len(all_combos)} kombinasi ablasi generator PASSED (forward + backward).")
    print("Bebas pilih kombinasi mana pun untuk eksperimen -- tidak terikat SR1-SR6.")
    print("Contoh pakai kombinasi custom apapun:")
    print('  cfg = build_config(use_nrdb=False, use_mbcm=True, use_ega=False, use_vsd=True)')
    print('  model = UAVIRE_Model(generator_kwargs=cfg["generator_kwargs"], use_vsd=cfg["use_vsd"])')