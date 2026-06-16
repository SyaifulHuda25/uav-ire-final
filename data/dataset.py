"""
Dataset untuk IRE Training
Mendukung dataset citra HR tunggal (tanpa pasangan LR),
karena LR dibangkitkan secara on-the-fly via degradation pipeline.

Juga mendukung dataset paired (HR + LR) untuk evaluasi.
"""

import os
import random
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.degradation import IRE_DegradationPipeline


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def scan_image_files(directory: str) -> List[str]:
    """Scan semua file gambar dalam direktori (rekursif)."""
    paths = []
    for p in Path(directory).rglob('*'):
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(str(p))
    return sorted(paths)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Konversi PIL Image ke tensor float [0, 1] dengan shape [C, H, W]."""
    return TF.to_tensor(img)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Konversi tensor float [0,1] [C, H, W] ke PIL Image."""
    return TF.to_pil_image(t.clamp(0, 1))


# ---------------------------------------------------------------------------
# Dataset untuk Training (HR only, LR dibangkitkan on-the-fly)
# ---------------------------------------------------------------------------

class IRE_TrainDataset(Dataset):
    """
    Dataset training IRE.

    Alur:
    1. Muat gambar HR
    2. Crop random HR patch (ukuran: gt_patch_size)
    3. Terapkan augmentasi (flip, rotate)
    4. Bangkitkan LR via IRE degradation pipeline
    5. Return pasangan (LR, HR)

    Args:
        hr_dir: Direktori berisi gambar HR
        gt_patch_size: Ukuran patch HR (default 256, sesuai paper IRE)
        scale_factor: Faktor upscaling (default 4)
        degradation_pipeline: Instance IRE_DegradationPipeline
        use_aug: Gunakan augmentasi (flip, rotate)
    """

    def __init__(
        self,
        hr_dir: str,
        gt_patch_size: int = 256,
        scale_factor: int = 2,
        degradation_pipeline: Optional[IRE_DegradationPipeline] = None,
        use_aug: bool = True,
    ):
        super().__init__()
        self.hr_paths = scan_image_files(hr_dir)
        if len(self.hr_paths) == 0:
            raise ValueError(f"Tidak ada gambar ditemukan di: {hr_dir}")

        self.gt_patch_size = gt_patch_size
        self.scale_factor = scale_factor
        self.use_aug = use_aug

        # Gunakan default pipeline jika tidak disediakan
        self.pipeline = degradation_pipeline or IRE_DegradationPipeline(
            scale_factor=scale_factor
        )

        print(f"[TrainDataset] Found {len(self.hr_paths)} images in {hr_dir}")

    def __len__(self) -> int:
        return len(self.hr_paths)

    def _random_crop(self, img: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Ambil random crop dari tensor [C, H, W]."""
        _, h, w = img.shape
        if h < patch_size or w < patch_size:
            # Resize jika terlalu kecil
            scale = max(patch_size / h, patch_size / w) + 0.1
            new_h, new_w = int(h * scale), int(w * scale)
            img = TF.resize(img, (new_h, new_w), antialias=True)
            _, h, w = img.shape

        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)
        return img[:, top:top + patch_size, left:left + patch_size]

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        """Augmentasi: horizontal flip dan rotasi 90 derajat."""
        if random.random() > 0.5:
            img = TF.hflip(img)
        if random.random() > 0.5:
            img = TF.vflip(img)
        rot = random.choice([0, 90, 180, 270])
        if rot > 0:
            img = TF.rotate(img, rot)
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Muat HR
        hr_path = self.hr_paths[idx]
        hr_pil = Image.open(hr_path).convert('RGB')
        hr = pil_to_tensor(hr_pil)  # [C, H, W] float [0,1]

        # Random crop ke gt_patch_size
        hr = self._random_crop(hr, self.gt_patch_size)

        # Augmentasi
        if self.use_aug:
            hr = self._augment(hr)

        # Bangkitkan LR via degradation pipeline
        lr = self.pipeline(hr)

        return lr, hr


# ---------------------------------------------------------------------------
# Dataset untuk Evaluasi/Validasi (paired HR-LR atau HR only)
# ---------------------------------------------------------------------------

class IRE_ValDataset(Dataset):
    """
    Dataset validasi/evaluasi IRE.

    Mode:
    1. Paired: Direktori terpisah untuk HR dan LR (nama file harus sama)
    2. HR only: Hanya HR, LR dibangkitkan dengan degradasi deterministik
       (bicubic downsampling) untuk evaluasi yang konsisten

    Args:
        hr_dir: Direktori gambar HR
        lr_dir: Direktori gambar LR (opsional, jika None gunakan bicubic)
        scale_factor: Faktor upscaling
        max_size: Maksimum jumlah gambar (None = semua)
    """

    def __init__(
        self,
        hr_dir: str,
        lr_dir: Optional[str] = None,
        scale_factor: int = 2,
        max_size: Optional[int] = None,
    ):
        super().__init__()
        self.hr_paths = scan_image_files(hr_dir)
        self.lr_dir = lr_dir
        self.scale_factor = scale_factor

        if max_size:
            self.hr_paths = self.hr_paths[:max_size]

        if lr_dir:
            self.lr_paths = scan_image_files(lr_dir)
            assert len(self.hr_paths) == len(self.lr_paths), \
                "Jumlah gambar HR dan LR harus sama"
        else:
            self.lr_paths = None

        print(f"[ValDataset] Found {len(self.hr_paths)} images in {hr_dir}")

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Muat HR
        hr = pil_to_tensor(Image.open(self.hr_paths[idx]).convert('RGB'))

        if self.lr_paths:
            # Paired: muat LR dari direktori
            lr = pil_to_tensor(Image.open(self.lr_paths[idx]).convert('RGB'))
        else:
            # Bicubic downsampling untuk evaluasi konsisten
            _, h, w = hr.shape
            lh, lw = h // self.scale_factor, w // self.scale_factor
            lr = TF.resize(hr, (lh, lw), interpolation=T.InterpolationMode.BICUBIC,
                           antialias=True)
            lr = torch.clamp(lr, 0, 1)

        return lr, hr, self.hr_paths[idx]


# ---------------------------------------------------------------------------
# Dataset untuk Inferensi (LR only)
# ---------------------------------------------------------------------------

class IRE_InferenceDataset(Dataset):
    """Dataset untuk inferensi: hanya LR images."""

    def __init__(self, lr_dir: str):
        super().__init__()
        self.lr_paths = scan_image_files(lr_dir)
        print(f"[InferenceDataset] Found {len(self.lr_paths)} images in {lr_dir}")

    def __len__(self) -> int:
        return len(self.lr_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        lr = pil_to_tensor(Image.open(self.lr_paths[idx]).convert('RGB'))
        return lr, self.lr_paths[idx]


# ---------------------------------------------------------------------------
# Fungsi helper untuk membuat DataLoader
# ---------------------------------------------------------------------------

def create_train_dataloader(
    hr_dir: str,
    batch_size: int = 16,
    gt_patch_size: int = 256,
    scale_factor: int = 2,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Buat DataLoader untuk training."""
    dataset = IRE_TrainDataset(
        hr_dir=hr_dir,
        gt_patch_size=gt_patch_size,
        scale_factor=scale_factor,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def create_val_dataloader(
    hr_dir: str,
    lr_dir: Optional[str] = None,
    scale_factor: int = 2,
    batch_size: int = 1,
    num_workers: int = 2,
    max_size: Optional[int] = None,
) -> DataLoader:
    """Buat DataLoader untuk validasi."""
    dataset = IRE_ValDataset(
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        scale_factor=scale_factor,
        max_size=max_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Test dataset
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import tempfile
    import torchvision

    print("Testing IRE Datasets...")

    # Buat dataset sintetis sementara
    with tempfile.TemporaryDirectory() as tmpdir:
        hr_dir = os.path.join(tmpdir, 'hr')
        os.makedirs(hr_dir)

        # Buat beberapa gambar test
        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            )
            img.save(os.path.join(hr_dir, f'test_{i:03d}.png'))

        # Test TrainDataset
        train_ds = IRE_TrainDataset(hr_dir=hr_dir, gt_patch_size=256)
        lr, hr = train_ds[0]
        print(f"Train - LR: {lr.shape}, HR: {hr.shape}")
        assert lr.shape == (3, 64, 64), f"Expected (3,64,64), got {lr.shape}"
        assert hr.shape == (3, 256, 256), f"Expected (3,256,256), got {hr.shape}"

        # Test ValDataset
        val_ds = IRE_ValDataset(hr_dir=hr_dir)
        lr, hr, path = val_ds[0]
        print(f"Val - LR: {lr.shape}, HR: {hr.shape}")

        print("Dataset test PASSED!")
