"""
WeedyRice-RGBMS-DB Dataset
Dataset khusus untuk WeedyRice-RGBMS-DB (Nguyen et al., 2025)

Struktur dataset di Kaggle:
/kaggle/input/.../WeedyRice-RGBMS-DB/
├── RGB/          ← gambar UAV (.JPG, 5280×3956)
├── Masks/        ← binary mask gulma (.PNG, sama ukuran dengan RGB)
├── Multispectral/ ← tidak digunakan untuk SR
├── Overlay/       ← tidak digunakan
├── Metadata/
├── train_list.txt ← daftar nama file untuk training
├── val_list.txt
└── test_list.txt

Desain Dataset:
- Split dibaca dari train/val/test_list.txt (bukan scan folder)
- RGB (.JPG) dipakai sebagai HR; LR dibangkitkan on-the-fly
- Mask (.PNG) dipakai sebagai weed_mask untuk VSD loss
- Resolusi asli 5280×3956 → wajib random patch crop saat training
- Tile inference saat test untuk menangani resolusi besar
"""

import os
import random
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────
# Utilitas
# ──────────────────────────────────────────────────────────

def read_list_file(list_path: str) -> List[str]:
    """
    Baca file list (train/val/test_list.txt).
    Setiap baris adalah nama file RGB, contoh:
      DJI_DateTime_2024_06_02_13_42_0035_...m.JPG
    """
    with open(list_path, 'r') as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def jpg_name_to_mask_name(jpg_name: str) -> str:
    """
    Konversi nama file RGB (.JPG) ke nama file Mask (.png).
    Contoh:
      DJI_DateTime_...m.JPG  →  DJI_DateTime_...m.png
    """
    stem = Path(jpg_name).stem        # tanpa ekstensi
    return stem + '.png'


def load_rgb(path: str) -> torch.Tensor:
    """Muat gambar RGB sebagai tensor float [3, H, W] ∈ [0, 1]."""
    img = Image.open(path).convert('RGB')
    return TF.to_tensor(img)          # [3, H, W] float32


def load_mask(path: str) -> Optional[torch.Tensor]:
    """
    Muat mask binary sebagai tensor float [1, H, W] ∈ {0, 1}.
    Putih (255) = gulma = 1, Hitam (0) = background = 0.
    Kembalikan None jika file tidak ada.
    """
    if not os.path.isfile(path):
        return None
    mask = Image.open(path).convert('L')   # grayscale
    t = TF.to_tensor(mask)                 # [1, H, W] float ∈ [0, 1]
    return (t > 0.5).float()               # binarisasi


# ──────────────────────────────────────────────────────────
# Training Dataset
# ──────────────────────────────────────────────────────────

class WeedyRiceTrainDataset(Dataset):
    """
    Training dataset WeedyRice-RGBMS-DB.

    Alur per sample:
    1. Baca nama file dari train_list.txt
    2. Load RGB (5280×3956) sebagai HR
    3. Load Mask (binary gulma) — dipakai untuk VSD
    4. Random crop ke gt_patch_size (misal 256×256)
    5. Sinkronisasi crop antara RGB dan Mask (same random seed)
    6. Augmentasi: horizontal flip, vertical flip, rotasi 90°
    7. Bangkitkan LR via degradation pipeline (on-the-fly)
    8. Return (lr, hr, mask)

    Args:
        dataset_root: Path ke folder WeedyRice-RGBMS-DB/
        list_file:    Path ke train_list.txt
        gt_patch_size: Ukuran patch HR (default 256)
        scale_factor:  Faktor SR (default 4)
        degradation_pipeline: Pipeline degradasi (UAV atau IRE)
        use_aug:       Gunakan augmentasi data
        use_mask:      Load dan kembalikan mask (untuk VSD)
    """

    def __init__(
        self,
        dataset_root: str,
        list_file: str,
        gt_patch_size: int = 256,
        scale_factor: int = 4,
        degradation_pipeline=None,
        use_aug: bool = True,
        use_mask: bool = True,
    ):
        self.rgb_dir   = os.path.join(dataset_root, 'RGB')
        self.mask_dir  = os.path.join(dataset_root, 'Masks')
        self.gt_patch  = gt_patch_size
        self.scale     = scale_factor
        self.use_aug   = use_aug
        self.use_mask  = use_mask

        # Baca list file
        self.file_names = read_list_file(list_file)

        # Validasi: pastikan file RGB ada
        valid = []
        for name in self.file_names:
            rgb_path = os.path.join(self.rgb_dir, name)
            if os.path.isfile(rgb_path):
                valid.append(name)
            else:
                print(f"  [WARN] File tidak ditemukan, skip: {name}")
        self.file_names = valid

        if not self.file_names:
            raise RuntimeError(f"Tidak ada file valid di {self.rgb_dir} "
                               f"sesuai {list_file}")

        # Degradation pipeline (default: UAV-specific)
        if degradation_pipeline is None:
            from data.uav_degradation import UAVSpecificDegradationPipeline
            degradation_pipeline = UAVSpecificDegradationPipeline(scale_factor)
        self.pipeline = degradation_pipeline

        print(f"[WeedyRiceTrainDataset] {len(self.file_names)} gambar "
              f"| patch={gt_patch_size} | scale={scale_factor}x "
              f"| mask={'ON' if use_mask else 'OFF'}")

    def __len__(self) -> int:
        return len(self.file_names)

    def _sync_crop(
        self, hr: torch.Tensor,
        mask: Optional[torch.Tensor],
        patch: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Random crop yang sinkron antara HR dan Mask.
        Keduanya dipotong di koordinat yang sama.
        """
        _, H, W = hr.shape
        # Pastikan gambar cukup besar
        if H < patch or W < patch:
            scale_up = max(patch / H, patch / W) * 1.05
            new_H = int(H * scale_up)
            new_W = int(W * scale_up)
            hr = TF.resize(hr, (new_H, new_W), antialias=True)
            if mask is not None:
                mask = TF.resize(mask, (new_H, new_W),
                                 interpolation=TF.InterpolationMode.NEAREST)
            _, H, W = hr.shape

        top  = random.randint(0, H - patch)
        left = random.randint(0, W - patch)

        hr_crop = hr[:, top:top+patch, left:left+patch]
        mask_crop = None
        if mask is not None:
            mask_crop = mask[:, top:top+patch, left:left+patch]

        return hr_crop, mask_crop

    def _augment(
        self, hr: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Augmentasi sinkron: flip horizontal, flip vertikal, rotasi 90°."""
        if random.random() > 0.5:
            hr = TF.hflip(hr)
            if mask is not None:
                mask = TF.hflip(mask)
        if random.random() > 0.5:
            hr = TF.vflip(hr)
            if mask is not None:
                mask = TF.vflip(mask)
        rot = random.choice([0, 90, 180, 270])
        if rot > 0:
            hr = TF.rotate(hr, rot)
            if mask is not None:
                mask = TF.rotate(mask, rot)
        return hr, mask

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        name = self.file_names[idx]

        # ── Load RGB ────────────────────────────────────────
        rgb_path = os.path.join(self.rgb_dir, name)
        hr = load_rgb(rgb_path)           # [3, 5280, 3956] float

        # ── Load Mask ───────────────────────────────────────
        mask = None
        if self.use_mask:
            mask_name = jpg_name_to_mask_name(name)
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = load_mask(mask_path)   # [1, H, W] binary atau None

        # ── Random patch crop (sinkron) ──────────────────────
        hr, mask = self._sync_crop(hr, mask, self.gt_patch)

        # ── Augmentasi ──────────────────────────────────────
        if self.use_aug:
            hr, mask = self._augment(hr, mask)

        # ── Bangkitkan LR via degradation ───────────────────
        lr = self.pipeline(hr)            # [3, 64, 64] untuk scale=4

        return lr, hr, mask


# ──────────────────────────────────────────────────────────
# Validation / Test Dataset
# ──────────────────────────────────────────────────────────

class WeedyRiceValDataset(Dataset):
    """
    Validasi/Test dataset WeedyRice-RGBMS-DB.

    Untuk efisiensi evaluasi pada gambar 5280×3956:
    - Ambil patch tengah ukuran eval_patch_size dari setiap gambar
    - LR dibangkitkan via degradation pipeline yang SAMA dengan training
      (deterministik per-gambar dengan seed tetap agar reproducible)

    CATATAN: Sebelumnya LR dibuat via bicubic saja — ini menyebabkan
    domain gap antara training dan validasi sehingga PSNR turun di bawah
    baseline bicubic (~15 dB). Setelah fix ini PSNR validasi naik ke
    rentang yang benar (~20-25 dB).
    - Return (lr, hr, mask, filename)

    Untuk full-image inference: gunakan tile inference di inference.py

    Args:
        dataset_root:   Path ke WeedyRice-RGBMS-DB/
        list_file:      Path ke val_list.txt atau test_list.txt
        eval_patch_size: Ukuran patch untuk evaluasi (default 512)
        scale_factor:   Faktor SR
        use_mask:       Load mask untuk evaluasi VSD
        max_samples:    Batasi jumlah sampel (None = semua)
    """

    def __init__(
        self,
        dataset_root: str,
        list_file: str,
        eval_patch_size: int = 512,
        scale_factor: int = 4,
        use_mask: bool = True,
        max_samples: Optional[int] = None,
        degradation_pipeline=None,   # [FIX] pipeline degradasi konsisten dengan training
    ):
        self.rgb_dir  = os.path.join(dataset_root, 'RGB')
        self.mask_dir = os.path.join(dataset_root, 'Masks')
        self.patch    = eval_patch_size
        self.scale    = scale_factor
        self.use_mask = use_mask

        self.file_names = read_list_file(list_file)

        # Filter file yang ada
        self.file_names = [
            n for n in self.file_names
            if os.path.isfile(os.path.join(self.rgb_dir, n))
        ]

        if max_samples:
            self.file_names = self.file_names[:max_samples]

        # [FIX] Degradation pipeline untuk validasi — default UAV-Specific,
        # sama dengan training, agar domain LR konsisten dan PSNR tidak drop.
        # Seed per-gambar dipakai di __getitem__ agar deterministik/reproducible.
        if degradation_pipeline is None:
            from data.uav_degradation import UAVSpecificDegradationPipeline
            degradation_pipeline = UAVSpecificDegradationPipeline(scale_factor)
        self.pipeline = degradation_pipeline

        print(f"[WeedyRiceValDataset] {len(self.file_names)} gambar "
              f"| eval_patch={eval_patch_size} | scale={scale_factor}x "
              f"| degradation=pipeline [FIX]")

    def __len__(self) -> int:
        return len(self.file_names)

    def _center_crop(
        self, img: torch.Tensor, size: int
    ) -> torch.Tensor:
        """Ambil center crop dari tensor [C, H, W]."""
        _, H, W = img.shape
        top  = max((H - size) // 2, 0)
        left = max((W - size) // 2, 0)
        h = min(size, H)
        w = min(size, W)
        return img[:, top:top+h, left:left+w]

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], str]:
        name = self.file_names[idx]

        # Load HR
        hr = load_rgb(os.path.join(self.rgb_dir, name))

        # Load Mask
        mask = None
        if self.use_mask:
            mask_name = jpg_name_to_mask_name(name)
            mask = load_mask(os.path.join(self.mask_dir, mask_name))

        # Center crop untuk evaluasi konsisten
        hr = self._center_crop(hr, self.patch)
        if mask is not None:
            mask = self._center_crop(mask, self.patch)

        # [FIX] Gunakan degradation pipeline yang sama dengan training.
        # Seed deterministik per-gambar (idx) agar hasil validasi reproducible
        # dan konsisten antar epoch. Ini menghilangkan domain gap yang
        # sebelumnya menyebabkan PSNR validasi ~15 dB (di bawah bicubic).
        random.seed(idx)
        import torch
        torch.manual_seed(idx)
        lr = self.pipeline(hr)   # [3, patch//scale, patch//scale]

        return lr, hr, mask, name


# ──────────────────────────────────────────────────────────
# Test Dataset untuk Full-Image Inference
# ──────────────────────────────────────────────────────────

class WeedyRiceTestDataset(Dataset):
    """
    Test dataset untuk inferensi full-image (tanpa crop).
    Gambar asli 5280×3956 dikembalikan utuh — gunakan tile inference.

    Return: (hr_tensor, mask_tensor_or_None, filename)
    """

    def __init__(
        self,
        dataset_root: str,
        list_file: str,
        use_mask: bool = True,
    ):
        self.rgb_dir  = os.path.join(dataset_root, 'RGB')
        self.mask_dir = os.path.join(dataset_root, 'Masks')
        self.use_mask = use_mask

        self.file_names = [
            n for n in read_list_file(list_file)
            if os.path.isfile(os.path.join(self.rgb_dir, n))
        ]
        print(f"[WeedyRiceTestDataset] {len(self.file_names)} gambar "
              f"(full resolution, gunakan tile inference)")

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
        name = self.file_names[idx]
        hr   = load_rgb(os.path.join(self.rgb_dir, name))
        mask = None
        if self.use_mask:
            mask_name = jpg_name_to_mask_name(name)
            mask = load_mask(os.path.join(self.mask_dir, mask_name))
        return hr, mask, name


# ──────────────────────────────────────────────────────────
# Factory functions untuk DataLoader
# ──────────────────────────────────────────────────────────

KAGGLE_ROOT = (
    '/kaggle/input/datasets/muhamadsyaifulhuda/'
    'weedy-rice-uav-segmentation/'
    'A Dataset of Aligned RGB and Multispectral UAV Ima/'
    'WeedyRice-RGBMS-DB/WeedyRice-RGBMS-DB'
)


def make_train_loader(
    dataset_root: str = KAGGLE_ROOT,
    list_file: Optional[str] = None,
    gt_patch_size: int = 256,
    scale_factor: int = 4,
    batch_size: int = 4,
    num_workers: int = 2,
    use_mask: bool = True,
    degradation_pipeline=None,
) -> DataLoader:
    """Buat DataLoader training WeedyRice."""
    if list_file is None:
        list_file = os.path.join(dataset_root, '..', 'train_list.txt')
        # Coba beberapa lokasi
        candidates = [
            list_file,
            os.path.join(dataset_root, 'train_list.txt'),
            os.path.join(os.path.dirname(dataset_root), 'train_list.txt'),
        ]
        for c in candidates:
            if os.path.isfile(c):
                list_file = c
                break

    ds = WeedyRiceTrainDataset(
        dataset_root=dataset_root,
        list_file=list_file,
        gt_patch_size=gt_patch_size,
        scale_factor=scale_factor,
        degradation_pipeline=degradation_pipeline,
        use_mask=use_mask,
    )
    return DataLoader(
        ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
        collate_fn=_collate_with_mask,
    )


def make_val_loader(
    dataset_root: str = KAGGLE_ROOT,
    list_file: Optional[str] = None,
    eval_patch_size: int = 512,
    scale_factor: int = 4,
    batch_size: int = 1,
    num_workers: int = 0,
    use_mask: bool = True,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Buat DataLoader validasi WeedyRice."""
    if list_file is None:
        list_file = _find_list_file(dataset_root, 'val_list.txt')

    # [FIX] Buat pipeline yang sama dengan training agar LR val konsisten
    from data.uav_degradation import UAVSpecificDegradationPipeline
    val_pipeline = UAVSpecificDegradationPipeline(scale_factor=scale_factor)

    ds = WeedyRiceValDataset(
        dataset_root=dataset_root,
        list_file=list_file,
        eval_patch_size=eval_patch_size,
        scale_factor=scale_factor,
        use_mask=use_mask,
        max_samples=max_samples,
        degradation_pipeline=val_pipeline,  # [FIX]
    )
    return DataLoader(
        ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=False,
        collate_fn=_collate_val,
    )


def make_test_loader(
    dataset_root: str = KAGGLE_ROOT,
    list_file: Optional[str] = None,
    use_mask: bool = True,
) -> DataLoader:
    """Buat DataLoader test WeedyRice (full resolution, batch=1)."""
    if list_file is None:
        list_file = _find_list_file(dataset_root, 'test_list.txt')

    ds = WeedyRiceTestDataset(
        dataset_root=dataset_root,
        list_file=list_file,
        use_mask=use_mask,
    )
    return DataLoader(
        ds, batch_size=1,
        shuffle=False, num_workers=0,
    )


def _find_list_file(dataset_root: str, filename: str) -> str:
    """Cari file list di beberapa lokasi yang mungkin."""
    candidates = [
        os.path.join(dataset_root, filename),
        os.path.join(dataset_root, '..', filename),
        os.path.join(os.path.dirname(dataset_root), filename),
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"Tidak menemukan {filename}. Dicari di: {candidates}"
    )


def _collate_with_mask(batch):
    """
    Custom collate untuk training: handle mask=None.
    Return (lr, hr, mask_or_None)
    """
    lrs, hrs, masks = zip(*batch)
    lr = torch.stack(lrs)
    hr = torch.stack(hrs)
    # Stack mask hanya jika semua tidak None
    if all(m is not None for m in masks):
        mask = torch.stack(masks)
    else:
        mask = None
    return lr, hr, mask


def _collate_val(batch):
    """
    Custom collate untuk validasi: handle mask=None + filename.
    Return (lr, hr, mask_or_None, filenames)
    """
    lrs, hrs, masks, names = zip(*batch)
    lr = torch.stack(lrs)
    hr = torch.stack(hrs)
    mask = torch.stack(masks) if all(m is not None for m in masks) else None
    return lr, hr, mask, list(names)
