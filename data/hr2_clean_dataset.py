"""
HR2-Clean Dataset (×2)  —  pipeline PSNR-oriented
==================================================
Pipeline ini dirancang untuk MEMAKSIMALKAN PSNR (sesuai diskusi dosen):

    citra asli (HR_orig)
        │  ↓2 (area)  →  ↑2 (NEAREST)
        ▼
      HR2  ............................  TARGET training & referensi PSNR
        │  ↓2 (bicubic, antialias)
        ▼
      LR   ............................  INPUT ke generator
        │  generator(scale=2)
        ▼
      SR   ≈  HR2

Berbeda dari `uav_degradation.py` yang menambah blur + noise + JPEG +
ROTASI acak (rotasi membuat LR tidak sejajar dengan HR sehingga PSNR
mentok rendah), pipeline ini bersifat DETERMINISTIK dan TANPA noise.
Karena tidak ada error yang tak-tereduksi, model dapat mencapai PSNR
tinggi (≈ 35–40 dB) terhadap HR2.

CATATAN PENTING (untuk laporan tesis):
- HR2 ≠ citra asli. HR2 adalah versi blok hasil nearest-upscale.
  PSNR yang dilaporkan adalah faithfulness terhadap HR2, BUKAN terhadap
  citra mentah. Ini disengaja agar referensi deterministik & reproducible.
- Mask dikembalikan pada ukuran HR2 (sama dengan SR) untuk VSD & mIoU.

Return:
    Train : (lr, hr2, mask_or_None)
    Val   : (lr, hr2, mask_or_None, filename)
"""

import os
import sys
import random
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# Pastikan root repo ada di path (untuk eksekusi standalone / self-test)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pakai ulang helper yang sudah ada agar konsisten
from data.weedyrice_dataset import (
    read_list_file, jpg_name_to_mask_name, load_rgb, load_mask,
)


# ──────────────────────────────────────────────────────────
# Transform inti: HR2 dan LR
# ──────────────────────────────────────────────────────────

def make_hr2(hr: torch.Tensor, down_factor: int = 2) -> torch.Tensor:
    """
    HR2 = HR_orig ↓down (area)  →  ↑down (nearest).
    Ukuran HR2 = ukuran HR_orig (tapi berstruktur blok).
    """
    _, H, W = hr.shape
    dh, dw = H // down_factor, W // down_factor
    small = F.interpolate(
        hr.unsqueeze(0), size=(dh, dw), mode='area'
    ).squeeze(0)
    hr2 = F.interpolate(
        small.unsqueeze(0), size=(dh * down_factor, dw * down_factor),
        mode='nearest'
    ).squeeze(0)
    return hr2.clamp(0, 1)


def make_lr(hr2: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """LR = HR2 ↓scale (bicubic + antialias). Deterministik, tanpa noise."""
    _, H, W = hr2.shape
    lr = F.interpolate(
        hr2.unsqueeze(0), size=(H // scale, W // scale),
        mode='bicubic', antialias=True
    ).squeeze(0)
    return lr.clamp(0, 1)


def _align_to_multiple(t: torch.Tensor, m: int) -> torch.Tensor:
    """Potong tensor [C,H,W] agar H dan W habis dibagi m (center)."""
    _, H, W = t.shape
    nH, nW = (H // m) * m, (W // m) * m
    top, left = (H - nH) // 2, (W - nW) // 2
    return t[:, top:top + nH, left:left + nW]


# ──────────────────────────────────────────────────────────
# Training Dataset
# ──────────────────────────────────────────────────────────

class HR2CleanTrainDataset(Dataset):
    """
    Args:
        dataset_root  : folder WeedyRice-RGBMS-DB/
        list_file     : train_list.txt
        gt_patch_size : ukuran patch HR2 (target). Harus kelipatan scale.
        scale_factor  : faktor SR (2).
        use_aug       : augmentasi flip/rotate90.
        use_mask      : load mask gulma (untuk VSD).
    """

    def __init__(
        self,
        dataset_root: str,
        list_file: str,
        gt_patch_size: int = 256,
        scale_factor: int = 2,
        use_aug: bool = True,
        use_mask: bool = True,
    ):
        assert gt_patch_size % scale_factor == 0, \
            f"gt_patch_size ({gt_patch_size}) harus kelipatan scale ({scale_factor})"
        self.rgb_dir  = os.path.join(dataset_root, 'RGB')
        self.mask_dir = os.path.join(dataset_root, 'Masks')
        self.gt_patch = gt_patch_size
        self.scale    = scale_factor
        self.use_aug  = use_aug
        self.use_mask = use_mask

        self.file_names = [
            n for n in read_list_file(list_file)
            if os.path.isfile(os.path.join(self.rgb_dir, n))
        ]
        if not self.file_names:
            raise RuntimeError(f"Tidak ada file valid di {self.rgb_dir} sesuai {list_file}")

        print(f"[HR2CleanTrainDataset] {len(self.file_names)} gambar | "
              f"patch={gt_patch_size} | scale={scale_factor}x | "
              f"mask={'ON' if use_mask else 'OFF'} | pipeline=CLEAN-HR2 (PSNR-oriented)")

    def __len__(self) -> int:
        return len(self.file_names)

    def _sync_crop(self, hr, mask, patch):
        _, H, W = hr.shape
        if H < patch or W < patch:
            s = max(patch / H, patch / W) * 1.05
            nH, nW = int(H * s), int(W * s)
            hr = TF.resize(hr, (nH, nW), antialias=True)
            if mask is not None:
                mask = TF.resize(mask, (nH, nW),
                                 interpolation=TF.InterpolationMode.NEAREST)
            _, H, W = hr.shape
        top  = random.randint(0, H - patch)
        left = random.randint(0, W - patch)
        hr_c = hr[:, top:top + patch, left:left + patch]
        mask_c = mask[:, top:top + patch, left:left + patch] if mask is not None else None
        return hr_c, mask_c

    def _augment(self, hr, mask):
        if random.random() > 0.5:
            hr = TF.hflip(hr); mask = TF.hflip(mask) if mask is not None else None
        if random.random() > 0.5:
            hr = TF.vflip(hr); mask = TF.vflip(mask) if mask is not None else None
        rot = random.choice([0, 90, 180, 270])
        if rot:
            hr = TF.rotate(hr, rot)
            if mask is not None:
                mask = TF.rotate(mask, rot)
        return hr, mask

    def __getitem__(self, idx):
        name = self.file_names[idx]
        hr = load_rgb(os.path.join(self.rgb_dir, name))

        mask = None
        if self.use_mask:
            mask = load_mask(os.path.join(self.mask_dir, jpg_name_to_mask_name(name)))

        # crop di citra asli, lalu augment
        hr, mask = self._sync_crop(hr, mask, self.gt_patch)
        if self.use_aug:
            hr, mask = self._augment(hr, mask)

        # HR2 (target) dan LR (input) — deterministik
        hr2 = make_hr2(hr, self.scale)
        lr  = make_lr(hr2, self.scale)
        return lr, hr2, mask


# ──────────────────────────────────────────────────────────
# Validation Dataset
# ──────────────────────────────────────────────────────────

class HR2CleanValDataset(Dataset):
    """
    Dataset validasi.

    PENTING — pemisahan target vs metrik (sesuai desain tesis):
      • Training tetap meniru HR2 (lihat HR2CleanTrainDataset).
      • Validasi/metrik diukur terhadap HR ASLI (orig), BUKAN HR2.
        Tujuannya: menguji apakah SR bisa "melampaui" HR2, yakni lebih
        dekat ke citra asli daripada HR2 yang blocky.

    Alur:
      orig (crop)
        │  ↓scale ↑scale(nearest)  → HR2  (hanya untuk membuat LR)
        │  HR2 ↓scale (bicubic)    → LR   (input model)
      referensi metrik = orig  (jika metric_vs_orig=True)

    Catatan: SR = generator(LR) berukuran sama dengan orig (eval patch),
    sehingga PSNR/SSIM(SR, orig) valid dan sejajar piksel.
    """

    def __init__(
        self,
        dataset_root: str,
        list_file: str,
        eval_patch_size: int = 512,
        scale_factor: int = 2,
        use_mask: bool = True,
        max_samples: Optional[int] = None,
        metric_vs_orig: bool = True,   # True → referensi = HR asli; False → HR2
    ):
        self.rgb_dir  = os.path.join(dataset_root, 'RGB')
        self.mask_dir = os.path.join(dataset_root, 'Masks')
        # pastikan kelipatan scale
        self.patch = (eval_patch_size // scale_factor) * scale_factor
        self.scale = scale_factor
        self.use_mask = use_mask
        self.metric_vs_orig = metric_vs_orig

        self.file_names = [
            n for n in read_list_file(list_file)
            if os.path.isfile(os.path.join(self.rgb_dir, n))
        ]
        if max_samples:
            self.file_names = self.file_names[:max_samples]

        ref = 'HR asli (orig)' if metric_vs_orig else 'HR2'
        print(f"[HR2CleanValDataset] {len(self.file_names)} gambar | "
              f"eval_patch={self.patch} | scale={scale_factor}x | "
              f"referensi metrik={ref}")

    def __len__(self) -> int:
        return len(self.file_names)

    def _center_crop(self, img, size):
        _, H, W = img.shape
        top  = max((H - size) // 2, 0)
        left = max((W - size) // 2, 0)
        return img[:, top:top + min(size, H), left:left + min(size, W)]

    def __getitem__(self, idx):
        name = self.file_names[idx]
        hr = load_rgb(os.path.join(self.rgb_dir, name))   # citra asli (orig)
        mask = None
        if self.use_mask:
            mask = load_mask(os.path.join(self.mask_dir, jpg_name_to_mask_name(name)))

        hr = self._center_crop(hr, self.patch)
        hr = _align_to_multiple(hr, self.scale)
        if mask is not None:
            mask = self._center_crop(mask, self.patch)
            mask = _align_to_multiple(mask, self.scale)

        hr2 = make_hr2(hr, self.scale)     # hanya untuk membuat LR
        lr  = make_lr(hr2, self.scale)     # input model
        ref = hr if self.metric_vs_orig else hr2   # referensi metrik
        return lr, ref, mask, name


# ──────────────────────────────────────────────────────────
# Collate
# ──────────────────────────────────────────────────────────

def collate_train(batch):
    lrs, hrs, masks = zip(*batch)
    lr = torch.stack(lrs); hr = torch.stack(hrs)
    mask = torch.stack(masks) if all(m is not None for m in masks) else None
    return lr, hr, mask


def collate_val(batch):
    lrs, hrs, masks, names = zip(*batch)
    lr = torch.stack(lrs); hr = torch.stack(hrs)
    mask = torch.stack(masks) if all(m is not None for m in masks) else None
    return lr, hr, mask, list(names)


# ──────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import math
    hr = torch.rand(3, 256, 256)
    hr2 = make_hr2(hr, 2)
    lr  = make_lr(hr2, 2)
    assert hr2.shape == (3, 256, 256)
    assert lr.shape == (3, 128, 128)

    lr_up = F.interpolate(lr.unsqueeze(0), size=(256, 256),
                          mode='nearest').squeeze(0).clamp(0, 1)
    mse = ((hr2 - lr_up) ** 2).mean().item()
    print(f"[OK] HR2 {list(hr2.shape)} | LR {list(lr.shape)} | "
          f"nearest-baseline PSNR(LR↑ vs HR2) = {10*math.log10(1/(mse+1e-10)):.2f} dB")
