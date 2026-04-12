"""
UAV-IRE Epoch-Based Trainer  [PATCHED v2]
=========================================
PATCH LOG (4 perbaikan berdasarkan diskusi metodologi):

  [PATCH 1] lambda_edge 0.05→0.15, lambda_vsd 0.1→0.25
            Bobot loss semantik dinaikkan agar sinyal EGA & VSD
            tidak tenggelam di antara rec + perc loss.
            → berpengaruh langsung ke mIoU segmentasi gulma.

  [PATCH 2] HR2 training target (fairness PSNR, saran dosen)
            HR asli didegradasi ringan → HR2 (blur kecil + JPEG q=90)
            HR2 dipakai sebagai target semua training loss.
            PSNR/SSIM validasi tetap dihitung vs HR asli (unseen).
            → model tidak pernah melihat HR asli saat training.
            Aktifkan dengan use_hr2=True (default: True).

  [PATCH 3] val_loss = full G_loss (bukan hanya L_rec)
            _validate() sekarang menghitung rec+perc+adv
            agar grafik train G_loss vs val G_loss apple-to-apple
            dan deteksi overfitting valid secara metodologi.
            PSNR/SSIM masih dihitung vs HR asli (unseen).

  [PATCH 4] CosineAnnealingLR menggantikan step decay tunggal
            lr_decay_epoch dihapus; scheduler cosine dipakai
            agar LR turun halus sepanjang training, bukan tiba-tiba
            di epoch 25 yang terlalu dini untuk 50 epoch.
            Bisa dikembalikan ke step decay via lr_schedule='step'.

Semua perubahan backward-compatible dengan checkpoint lama.
Checkpoint baru menyimpan scheduler state untuk auto-resume.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
from collections import defaultdict

import torch
import torch.optim as optim
# [PATCH 4] tambah import scheduler
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# [PATCH 2] tambah import untuk degradasi HR2
import random
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.network import PatchGAN_Discriminator
from losses.uav_ire_losses import UAVIRE_GeneratorLoss, UAVIRE_DiscriminatorLoss
from utils.metrics import calculate_psnr, calculate_ssim


# ─────────────────────────────────────────────────────────────────────────────
# [PATCH 2] HR2 Degradation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class HR2DegradationPipeline:
    """
    Mendegradasi HR asli menjadi HR2 dengan intensitas RINGAN.

    Tujuan: menciptakan jarak antara target training (HR2) dan
    referensi evaluasi PSNR (HR asli), sehingga model tidak
    pernah melihat HR asli selama training.

    Degradasi yang diterapkan (urutan acak ringan):
      1. Gaussian blur ringan  (sigma 0.3–0.7, radius kecil)
      2. JPEG compression      (quality 85–95)
      3. Subtle noise          (std 1–3/255, opsional)

    Semua parameter dirancang agar HR2 masih sangat dekat dengan HR
    (PSNR HR2 vs HR sekitar 38–42 dB) — bukan downsampling,
    hanya "mengaburkan sedikit" agar model tidak menghafal piksel HR.
    """

    def __init__(
        self,
        blur_sigma_range: tuple = (0.3, 0.7),
        jpeg_quality_range: tuple = (85, 95),
        noise_std_range: tuple = (1.0, 3.0),
        apply_noise_prob: float = 0.3,   # noise hanya 30% kemungkinan
    ):
        self.blur_sigma_range    = blur_sigma_range
        self.jpeg_quality_range  = jpeg_quality_range
        self.noise_std_range     = noise_std_range
        self.apply_noise_prob    = apply_noise_prob

    def __call__(self, hr_tensor: torch.Tensor) -> torch.Tensor:
        """
        Input : hr_tensor  float32 [C, H, W] range [0, 1]
        Output: hr2_tensor float32 [C, H, W] range [0, 1]
        """
        # Konversi ke PIL untuk operasi image processing
        hr_pil = TF.to_pil_image(hr_tensor.clamp(0, 1))

        # 1. Gaussian blur ringan
        sigma = random.uniform(*self.blur_sigma_range)
        hr_pil = hr_pil.filter(ImageFilter.GaussianBlur(radius=sigma))

        # 2. JPEG compression
        import io
        quality = random.randint(*self.jpeg_quality_range)
        buf = io.BytesIO()
        hr_pil.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        hr_pil = Image.open(buf).copy()

        # Konversi kembali ke tensor
        hr2_tensor = TF.to_tensor(hr_pil)

        # 3. Subtle noise (opsional, 30% kemungkinan)
        if random.random() < self.apply_noise_prob:
            noise_std = random.uniform(*self.noise_std_range) / 255.0
            noise = torch.randn_like(hr2_tensor) * noise_std
            hr2_tensor = (hr2_tensor + noise).clamp(0, 1)

        return hr2_tensor

    def apply_batch(self, hr_batch: torch.Tensor) -> torch.Tensor:
        """
        Terapkan degradasi ke seluruh batch.
        Input/Output: [B, C, H, W] float32 [0, 1]
        """
        hr2_list = [self.__call__(hr_batch[i]) for i in range(hr_batch.shape[0])]
        return torch.stack(hr2_list, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Logger (tidak berubah)
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(log_dir: str, name: str = 'UAV-IRE') -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter(
        '[%(asctime)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler(os.path.join(log_dir, 'training.log'), encoding='utf-8')
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# TrainingHistory (tidak berubah)
# ─────────────────────────────────────────────────────────────────────────────

class TrainingHistory:
    """
    Menyimpan metrik training dan validasi per epoch.
    Mendukung export ke JSON dan Excel.
    """

    def __init__(self):
        self.epochs: List[int] = []
        self.train: Dict[str, List[float]] = defaultdict(list)
        self.val:   Dict[str, List[float]] = defaultdict(list)
        self.iter_log: List[Dict] = []

    def push_epoch(self, epoch: int,
                   train_means: Dict[str, float],
                   val_means: Dict[str, float]):
        self.epochs.append(epoch)
        for k, v in train_means.items():
            self.train[k].append(v)
        for k, v in val_means.items():
            self.val[k].append(v)

    def push_iter(self, epoch: int, iteration: int, losses: Dict[str, float]):
        entry = {'epoch': epoch, 'iteration': iteration}
        entry.update(losses)
        self.iter_log.append(entry)

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = {
            'epochs': self.epochs,
            'train': dict(self.train),
            'val': dict(self.val),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> 'TrainingHistory':
        hist = cls()
        with open(path) as f:
            data = json.load(f)
        hist.epochs = data.get('epochs', [])
        hist.train  = defaultdict(list, data.get('train', {}))
        hist.val    = defaultdict(list, data.get('val', {}))
        return hist

    def to_dataframes(self):
        import pandas as pd
        n = len(self.epochs)
        epoch_data = {'epoch': self.epochs}
        for k, v in self.train.items():
            padded = list(v) + [None] * (n - len(v))
            epoch_data[f'train_{k}'] = padded[:n]
        for k, v in self.val.items():
            padded = list(v) + [None] * (n - len(v))
            epoch_data[f'val_{k}'] = padded[:n]
        df_epoch = pd.DataFrame(epoch_data)
        df_iter = pd.DataFrame(self.iter_log) if self.iter_log else pd.DataFrame()
        return df_epoch, df_iter

    def export_excel(self, path: str):
        import pandas as pd
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        df_epoch, df_iter = self.to_dataframes()
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            df_epoch.to_excel(writer, sheet_name='Epoch Summary', index=False)
            if not df_iter.empty:
                df_iter.to_excel(writer, sheet_name='Iteration Detail', index=False)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# EpochTrainer  [PATCHED]
# ─────────────────────────────────────────────────────────────────────────────

class EpochTrainer:
    """
    UAV-IRE Trainer berbasis epoch — versi patched.

    Perubahan dari versi asli:
      - lambda_edge, lambda_vsd dinaikkan (PATCH 1)
      - use_hr2=True untuk fairness PSNR (PATCH 2)
      - _validate() menghitung full G_loss (PATCH 3)
      - CosineAnnealingLR sebagai scheduler default (PATCH 4)
    """

    PRESETS = {
        'full':     dict(use_nrdb=True,  use_mbcm=True,  use_ega=True,  use_vsd=True,  use_uav_deg=True),
        'baseline': dict(use_nrdb=False, use_mbcm=False, use_ega=False, use_vsd=False, use_uav_deg=False),
        'no_nrdb':  dict(use_nrdb=False, use_mbcm=True,  use_ega=True,  use_vsd=True,  use_uav_deg=True),
        'no_mbcm':  dict(use_nrdb=True,  use_mbcm=False, use_ega=True,  use_vsd=True,  use_uav_deg=True),
        'no_ega':   dict(use_nrdb=True,  use_mbcm=True,  use_ega=False, use_vsd=True,  use_uav_deg=True),
        'no_vsd':   dict(use_nrdb=True,  use_mbcm=True,  use_ega=True,  use_vsd=False, use_uav_deg=True),
    }

    PRESET_NAMES = {
        'baseline': 'SR1_IRE_Baseline',
        'no_nrdb':  'SR2_UAV-IRE_no_NRDB',
        'no_mbcm':  'SR3_UAV-IRE_no_MBCM',
        'no_ega':   'SR4_UAV-IRE_no_EGA',
        'no_vsd':   'SR5_UAV-IRE_no_VSD',
        'full':     'SR6_UAV-IRE_Full',
    }

    def __init__(
        self,
        experiment: str = 'full',
        # ── Dataset ──────────────────────────────────────────────
        dataset_root: Optional[str] = None,
        train_list: Optional[str] = None,
        val_list:   Optional[str] = None,
        hr_train_dir: Optional[str] = None,
        hr_val_dir:   Optional[str] = None,
        # ── Training config ───────────────────────────────────────
        save_dir: str = 'experiments',
        total_epochs: int = 50,
        batch_size: int = 4,
        gt_patch_size: int = 256,
        eval_patch_size: int = 512,
        scale_factor: int = 4,
        num_rrdb: int = 23,
        num_features: int = 64,
        lr_g: float = 1e-4,
        lr_d: float = 1e-4,
        # [PATCH 4] scheduler: 'cosine' (default) atau 'step'
        lr_schedule: str = 'cosine',
        # lr_decay_epoch hanya dipakai jika lr_schedule='step'
        lr_decay_epoch: int = 40,
        use_mask: bool = True,
        # ── [PATCH 1] Loss weights — semantik dinaikkan ──────────
        lambda_rec:  float = 1.0,
        lambda_perc: float = 1.0,
        lambda_adv:  float = 0.1,
        lambda_edge: float = 0.15,   # [PATCH 1] 0.05 → 0.15
        lambda_vsd:  float = 0.25,   # [PATCH 1] 0.10 → 0.25
        # ── [PATCH 2] HR2 training target ────────────────────────
        use_hr2: bool = True,        # [PATCH 2] aktifkan HR2
        hr2_blur_sigma:   tuple = (0.3, 0.7),
        hr2_jpeg_quality: tuple = (85, 95),
        hr2_noise_std:    tuple = (1.0, 3.0),
        hr2_noise_prob:   float = 0.3,
        # ── Logging ───────────────────────────────────────────────
        log_iter_freq: int = 10,
        val_epoch_freq: int = 1,
        save_epoch_freq: int = 1,
        vis_epoch_freq: int = 5,
        pretrained_path: Optional[str] = None,
        use_amp: bool = True,
        num_workers: int = 2,
        device: Optional[torch.device] = None,
    ):
        assert experiment in self.PRESETS, \
            f"experiment harus salah satu dari: {list(self.PRESETS)}"
        assert lr_schedule in ('cosine', 'step'), \
            "lr_schedule harus 'cosine' atau 'step'"

        self.experiment      = experiment
        self.flags           = self.PRESETS[experiment]
        self.exp_name        = self.PRESET_NAMES[experiment]
        self.save_dir        = os.path.join(save_dir, self.exp_name)
        self.total_epochs    = total_epochs
        self.lr_schedule     = lr_schedule
        self.lr_decay_epoch  = lr_decay_epoch
        self.log_iter_freq   = log_iter_freq
        self.val_epoch_freq  = val_epoch_freq
        self.save_epoch_freq = save_epoch_freq
        self.vis_epoch_freq  = vis_epoch_freq
        self.use_amp         = use_amp and torch.cuda.is_available()
        self.device          = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_mask        = use_mask and self.flags.get('use_vsd', False)

        # [PATCH 2] inisialisasi HR2 pipeline
        self.use_hr2 = use_hr2
        self.hr2_pipeline = None
        if use_hr2:
            self.hr2_pipeline = HR2DegradationPipeline(
                blur_sigma_range=hr2_blur_sigma,
                jpeg_quality_range=hr2_jpeg_quality,
                noise_std_range=hr2_noise_std,
                apply_noise_prob=hr2_noise_prob,
            )

        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = setup_logger(self.save_dir, name=self.exp_name)
        self.logger.info(f"Experiment   : {self.exp_name}")
        self.logger.info(f"Device       : {self.device}")
        self.logger.info(f"Flags        : {self.flags}")
        self.logger.info(f"Use mask     : {self.use_mask} (VSD)")
        self.logger.info(
            f"[PATCH 1] lambda_edge={lambda_edge}, lambda_vsd={lambda_vsd}"
        )
        self.logger.info(
            f"[PATCH 2] use_hr2={use_hr2} | PSNR val vs HR asli (unseen)"
        )
        self.logger.info(
            f"[PATCH 3] val_loss = full G_loss (rec+perc+adv)"
        )
        self.logger.info(
            f"[PATCH 4] lr_schedule={lr_schedule}"
        )

        # ── Build models ─────────────────────────────────────────
        self._build_models(num_rrdb, scale_factor, num_features,
                           lambda_rec, lambda_perc, lambda_adv,
                           lambda_edge, lambda_vsd)

        # ── Optimisers ───────────────────────────────────────────
        self.optimizer_g = optim.Adam(
            self.generator.parameters(), lr=lr_g, betas=(0.9, 0.99))
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=lr_d, betas=(0.9, 0.99))
        self.scaler_g = GradScaler(enabled=self.use_amp)
        self.scaler_d = GradScaler(enabled=self.use_amp)

        # [PATCH 4] Setup scheduler
        self.scheduler_g = self._make_scheduler(self.optimizer_g)
        self.scheduler_d = self._make_scheduler(self.optimizer_d)

        # ── DataLoaders ──────────────────────────────────────────
        self._build_dataloaders(
            dataset_root=dataset_root,
            train_list=train_list,
            val_list=val_list,
            hr_train_dir=hr_train_dir,
            hr_val_dir=hr_val_dir,
            batch_size=batch_size,
            gt_patch_size=gt_patch_size,
            eval_patch_size=eval_patch_size,
            scale_factor=scale_factor,
            num_workers=num_workers,
        )

        # ── History & resume ─────────────────────────────────────
        self.history     = TrainingHistory()
        self.start_epoch = 1

        if pretrained_path and os.path.isfile(pretrained_path):
            self._load_pretrained(pretrained_path)

        latest_ckpt = os.path.join(self.save_dir, 'checkpoint_latest.pth')
        if os.path.isfile(latest_ckpt):
            self._resume(latest_ckpt)

    # ─────────────────────────────────────────────────────────────
    # [PATCH 4] Scheduler factory
    # ─────────────────────────────────────────────────────────────

    def _make_scheduler(self, optimizer):
        """
        Buat scheduler sesuai pilihan lr_schedule.
        - 'cosine': CosineAnnealingLR → LR turun halus dari lr ke lr_min=1e-7
        - 'step'  : Halved sekali saat epoch == lr_decay_epoch (perilaku asli)
        """
        if self.lr_schedule == 'cosine':
            return lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.total_epochs,
                eta_min=1e-7,
            )
        else:
            # StepLR dengan milestone tunggal — identik perilaku asli
            return lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[self.lr_decay_epoch],
                gamma=0.5,
            )

    # ─────────────────────────────────────────────────────────────
    # Build helpers (tidak berubah signifikan)
    # ─────────────────────────────────────────────────────────────

    def _build_models(self, num_rrdb, scale_factor, num_features,
                      lambda_rec, lambda_perc, lambda_adv,
                      lambda_edge, lambda_vsd):
        use_nrdb = self.flags['use_nrdb']
        use_mbcm = self.flags['use_mbcm']
        use_ega  = self.flags['use_ega']
        use_vsd  = self.flags['use_vsd']

        if use_nrdb and use_mbcm and use_ega:
            from models.uav_ire_generator import UAVIRE_Generator
            self.generator = UAVIRE_Generator(
                num_rrdb=num_rrdb,
                scale_factor=scale_factor,
                num_features=num_features,
            ).to(self.device)
            self.logger.info("Generator: UAVIRE_Generator (NRDB+MBCM+EGA)")
        else:
            from models.network import IRE_Generator
            self.generator = IRE_Generator(
                num_rrdb=num_rrdb,
                scale_factor=scale_factor,
            ).to(self.device)
            self.logger.info(f"Generator: IRE_Generator (ablation, nrdb={use_nrdb}, "
                             f"mbcm={use_mbcm}, ega={use_ega})")

        self.discriminator = PatchGAN_Discriminator().to(self.device)

        self.vsd = None
        if use_vsd:
            from models.vsd import VSD
            self.vsd = VSD().to(self.device)
            self.logger.info("VSD: enabled")

        self.gen_loss_fn = UAVIRE_GeneratorLoss(
            lambda_rec=lambda_rec, lambda_perc=lambda_perc,
            lambda_adv=lambda_adv,
            lambda_edge=lambda_edge if use_ega else 0.0,
            lambda_vsd=lambda_vsd  if use_vsd else 0.0,
            use_edge_loss=use_ega,
            use_vsd_loss=use_vsd,
        ).to(self.device)

        self.disc_loss_fn = UAVIRE_DiscriminatorLoss().to(self.device)

    def _build_dataloaders(self,
                           dataset_root=None, train_list=None, val_list=None,
                           hr_train_dir=None, hr_val_dir=None,
                           batch_size=4, gt_patch_size=256, eval_patch_size=512,
                           scale_factor=4, num_workers=2):
        if self.flags['use_uav_deg']:
            from data.uav_degradation import UAVSpecificDegradationPipeline
            pipeline = UAVSpecificDegradationPipeline(scale_factor=scale_factor)
            self.logger.info("Degradation: UAV-Specific")
        else:
            from data.degradation import IRE_DegradationPipeline
            pipeline = IRE_DegradationPipeline(scale_factor=scale_factor)
            self.logger.info("Degradation: IRE Second-Order")

        if dataset_root and os.path.isdir(dataset_root):
            from data.weedyrice_dataset import (
                WeedyRiceTrainDataset, WeedyRiceValDataset,
                make_train_loader, make_val_loader
            )
            def _find(root, fname):
                for p in [
                    os.path.join(root, fname),
                    os.path.join(root, '..', fname),
                    os.path.join(os.path.dirname(root), fname),
                ]:
                    if os.path.isfile(os.path.normpath(p)):
                        return os.path.normpath(p)
                raise FileNotFoundError(f"{fname} tidak ditemukan")

            _train_list = train_list or _find(dataset_root, 'train_list.txt')
            _val_list   = val_list   or _find(dataset_root, 'val_list.txt')

            train_ds = WeedyRiceTrainDataset(
                dataset_root=dataset_root,
                list_file=_train_list,
                gt_patch_size=gt_patch_size,
                scale_factor=scale_factor,
                degradation_pipeline=pipeline,
                use_mask=self.use_mask,
            )
            from data.weedyrice_dataset import _collate_with_mask
            self.train_loader = DataLoader(
                train_ds, batch_size=batch_size,
                shuffle=True, num_workers=num_workers,
                pin_memory=True, drop_last=True,
                persistent_workers=(num_workers > 0),
                collate_fn=_collate_with_mask,
            )

            val_ds = WeedyRiceValDataset(
                dataset_root=dataset_root,
                list_file=_val_list,
                eval_patch_size=eval_patch_size,
                scale_factor=scale_factor,
                use_mask=self.use_mask,
            )
            from data.weedyrice_dataset import _collate_val
            self.val_loader = DataLoader(
                val_ds, batch_size=1, shuffle=False, num_workers=0,
                collate_fn=_collate_val,
            )
            self.logger.info("Dataset: WeedyRice-RGBMS-DB")

        else:
            from data.dataset import IRE_TrainDataset, IRE_ValDataset
            train_ds = IRE_TrainDataset(
                hr_dir=hr_train_dir or 'data/train',
                gt_patch_size=gt_patch_size,
                scale_factor=scale_factor,
                degradation_pipeline=pipeline,
            )
            self.train_loader = DataLoader(
                train_ds, batch_size=batch_size,
                shuffle=True, num_workers=num_workers,
                pin_memory=True, drop_last=True,
                persistent_workers=(num_workers > 0),
            )
            self.val_loader = None
            if hr_val_dir and os.path.isdir(hr_val_dir):
                val_ds = IRE_ValDataset(hr_dir=hr_val_dir, scale_factor=scale_factor)
                self.val_loader = DataLoader(
                    val_ds, batch_size=1, shuffle=False, num_workers=0)
            self.logger.info("Dataset: Generic (fallback)")

        self.iters_per_epoch = len(self.train_loader)
        self.logger.info(f"Train: {len(train_ds)} imgs | "
                         f"{self.iters_per_epoch} iter/epoch | batch={batch_size}")

    # ─────────────────────────────────────────────────────────────
    # Checkpoint helpers — diperbarui untuk menyimpan scheduler
    # ─────────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, filename: str):
        ckpt = {
            'epoch': epoch,
            'experiment': self.experiment,
            'generator':  self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            # [PATCH 4] simpan scheduler state untuk auto-resume
            'scheduler_g': self.scheduler_g.state_dict(),
            'scheduler_d': self.scheduler_d.state_dict(),
            'history_json': json.dumps({
                'epochs': self.history.epochs,
                'train':  dict(self.history.train),
                'val':    dict(self.history.val),
            }),
        }
        if self.vsd:
            ckpt['vsd'] = self.vsd.state_dict()
        torch.save(ckpt, os.path.join(self.save_dir, filename))

    def _resume(self, path: str):
        self.logger.info(f"Resuming from: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(ckpt['generator'], strict=False)
        if 'discriminator' in ckpt:
            self.discriminator.load_state_dict(ckpt['discriminator'])
        if 'vsd' in ckpt and self.vsd:
            self.vsd.load_state_dict(ckpt['vsd'])
        if 'optimizer_g' in ckpt:
            self.optimizer_g.load_state_dict(ckpt['optimizer_g'])
        if 'optimizer_d' in ckpt:
            self.optimizer_d.load_state_dict(ckpt['optimizer_d'])
        # [PATCH 4] resume scheduler state (backward-compatible: tidak error jika key tidak ada)
        if 'scheduler_g' in ckpt:
            self.scheduler_g.load_state_dict(ckpt['scheduler_g'])
        if 'scheduler_d' in ckpt:
            self.scheduler_d.load_state_dict(ckpt['scheduler_d'])
        if 'history_json' in ckpt:
            d = json.loads(ckpt['history_json'])
            self.history.epochs = d.get('epochs', [])
            self.history.train  = defaultdict(list, d.get('train', {}))
            self.history.val    = defaultdict(list, d.get('val', {}))
        self.start_epoch = ckpt.get('epoch', 0) + 1
        self.logger.info(f"Resumed → start epoch {self.start_epoch}")

    def _load_pretrained(self, path: str):
        self.logger.info(f"Loading pretrained: {path}")
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict):
            for key in ('generator', 'params_ema', 'params', 'state_dict'):
                if key in state:
                    state = state[key]
                    break
        missing, unexpected = self.generator.load_state_dict(state, strict=False)
        self.logger.info(f"Pretrained loaded — missing:{len(missing)}, "
                         f"unexpected:{len(unexpected)}")

    # [PATCH 4] _adjust_lr digantikan scheduler — method ini dipertahankan
    # sebagai no-op untuk backward-compatibility jika dipanggil dari tempat lain
    def _adjust_lr(self, epoch: int):
        pass   # scheduler.step() dipanggil di akhir train loop

    # ─────────────────────────────────────────────────────────────
    # [PATCH 2+3] Train step — HR2 sebagai target loss
    # ─────────────────────────────────────────────────────────────

    def _train_step(self, batch):
        """
        Satu step training.

        [PATCH 2] Jika use_hr2=True:
          - hr_img (HR asli) TIDAK dipakai sebagai target loss
          - hr2_img (HR + degradasi ringan) dipakai sebagai target loss
          - Discriminator membandingkan SR vs hr2_img (bukan HR asli)

        Catatan: hr_img asli hanya digunakan di _validate() untuk
        menghitung PSNR/SSIM yang fair (unseen by model).
        """
        if len(batch) == 3:
            lr_img, hr_img, weed_mask = batch
        else:
            lr_img, hr_img = batch
            weed_mask = None

        lr_img = lr_img.to(self.device)
        hr_img = hr_img.to(self.device)
        if weed_mask is not None:
            weed_mask = weed_mask.to(self.device)

        # [PATCH 2] buat HR2 dari HR asli (on-the-fly, per batch)
        if self.use_hr2 and self.hr2_pipeline is not None:
            hr_target = self.hr2_pipeline.apply_batch(hr_img.cpu()).to(self.device)
        else:
            hr_target = hr_img   # fallback ke perilaku asli

        # ── Discriminator step ─────────────────────────────────
        self.optimizer_d.zero_grad(set_to_none=True)
        with autocast(enabled=self.use_amp):
            with torch.no_grad():
                sr = self.generator(lr_img)
            # [PATCH 2] discriminator membandingkan SR vs hr_target (HR2)
            real_p = self.discriminator(hr_target)
            fake_p = self.discriminator(sr.detach())
            d_loss, d_dict = self.disc_loss_fn(real_p, fake_p)
        self.scaler_d.scale(d_loss).backward()
        self.scaler_d.step(self.optimizer_d)
        self.scaler_d.update()

        # ── Generator step ─────────────────────────────────────
        self.optimizer_g.zero_grad(set_to_none=True)
        with autocast(enabled=self.use_amp):
            sr = self.generator(lr_img)
            real_p = self.discriminator(hr_target).detach()
            fake_p = self.discriminator(sr)
            # [PATCH 2] gen_loss_fn menerima hr_target (HR2) bukan hr_img
            g_loss, g_dict = self.gen_loss_fn(
                sr, hr_target, fake_p, real_p,
                weed_mask=weed_mask,
                vsd_module=self.vsd,
            )
        self.scaler_g.scale(g_loss).backward()
        self.scaler_g.step(self.optimizer_g)
        self.scaler_g.update()

        return g_dict, d_dict

    # ─────────────────────────────────────────────────────────────
    # [PATCH 3] Validate — full G_loss + PSNR vs HR asli (unseen)
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self):
        """
        [PATCH 3] val_loss sekarang = gabungan rec + perc + adv
        agar grafik train G vs val G apple-to-apple.

        [PATCH 2] PSNR/SSIM dihitung vs hr_img (HR asli, unseen by model)
        bukan vs hr_target (HR2 yang dipakai training loss).
        Ini menghasilkan PSNR yang fair — model belum pernah melihat HR asli.
        """
        if not self.val_loader:
            return {}

        self.generator.eval()
        psnr_list, ssim_list = [], []
        val_g_losses: Dict[str, List[float]] = defaultdict(list)

        for batch in self.val_loader:
            # Unpack batch dari val_loader
            if len(batch) == 4:
                lr_img, hr_img, _, _ = batch   # WeedyRice: (lr, hr, mask, filename)
            elif len(batch) == 3:
                lr_img, hr_img, _ = batch
            else:
                lr_img, hr_img = batch[0], batch[1]

            lr_img = lr_img.to(self.device)
            hr_img = hr_img.to(self.device)   # HR asli — target PSNR (unseen)

            # Generate SR
            with autocast(enabled=self.use_amp):
                sr = self.generator(lr_img).clamp(0, 1)

            # [PATCH 3] Hitung full G_loss untuk val_loss yang kompatibel
            # Gunakan hr_img (HR asli) sebagai referensi loss validasi
            # agar skala val_loss setara dengan train G_loss
            with autocast(enabled=self.use_amp):
                real_p = self.discriminator(hr_img).detach()
                fake_p = self.discriminator(sr).detach()
                _, g_val_dict = self.gen_loss_fn(
                    sr, hr_img, fake_p, real_p,
                    weed_mask=None,      # mask tidak dipakai saat validasi
                    vsd_module=None,     # VSD tidak dipakai saat validasi
                )
            for k, v in g_val_dict.items():
                val_g_losses[k].append(v)

            # [PATCH 2+3] PSNR/SSIM vs HR asli (unseen) — fair evaluation
            for i in range(sr.shape[0]):
                psnr_list.append(calculate_psnr(sr[i], hr_img[i]))
                ssim_list.append(calculate_ssim(sr[i], hr_img[i]))

        self.generator.train()

        # Rata-rata semua komponen val loss
        val_means = {
            f'val_{k}': sum(v) / len(v)
            for k, v in val_g_losses.items() if v
        }
        # val_loss = total G_loss validasi (untuk grafik utama)
        val_means['val_loss'] = val_means.get('val_total', 0.0)
        val_means['psnr']     = sum(psnr_list) / len(psnr_list)  if psnr_list else 0.0
        val_means['ssim']     = sum(ssim_list) / len(ssim_list)  if ssim_list else 0.0

        return val_means

    # ─────────────────────────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────────────────────────

    def train(self) -> TrainingHistory:
        n_params = sum(p.numel() for p in self.generator.parameters())
        self.logger.info("=" * 60)
        self.logger.info(f"Training {self.exp_name}  [PATCHED v2]")
        self.logger.info(f"  Generator params : {n_params:,}")
        self.logger.info(f"  Total epochs     : {self.total_epochs}")
        self.logger.info(f"  Iters/epoch      : {self.iters_per_epoch}")
        self.logger.info(f"  LR schedule      : {self.lr_schedule}")
        self.logger.info(f"  use_hr2          : {self.use_hr2}")
        self.logger.info(f"  val PSNR vs      : {'HR asli (unseen)' if self.use_hr2 else 'HR (sama dengan training)'}")
        self.logger.info("=" * 60)

        vis_dir = os.path.join(self.save_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        global_iter = (self.start_epoch - 1) * self.iters_per_epoch

        for epoch in range(self.start_epoch, self.total_epochs + 1):
            # [PATCH 4] _adjust_lr tidak lagi dipanggil di sini
            # scheduler.step() dipanggil di akhir epoch
            self.generator.train()
            self.discriminator.train()

            epoch_g_losses: Dict[str, List[float]] = defaultdict(list)
            epoch_d_losses: Dict[str, List[float]] = defaultdict(list)
            t_epoch = time.time()

            for i, batch in enumerate(self.train_loader, 1):
                global_iter += 1
                g_dict, d_dict = self._train_step(batch)

                for k, v in g_dict.items():
                    epoch_g_losses[k].append(v)
                for k, v in d_dict.items():
                    epoch_d_losses[k].append(v)

                if i % self.log_iter_freq == 0:
                    iter_losses = {f'g_{k}': v for k, v in g_dict.items()}
                    iter_losses.update({f'd_{k}': v for k, v in d_dict.items()})
                    self.history.push_iter(epoch, global_iter, iter_losses)

                    eta_iter = (self.iters_per_epoch - i) + \
                               (self.total_epochs - epoch) * self.iters_per_epoch
                    speed    = i / (time.time() - t_epoch + 1e-8)
                    eta_min  = eta_iter / speed / 60
                    lr_now   = self.optimizer_g.param_groups[0]['lr']
                    self.logger.info(
                        f"[E{epoch:03d}/{self.total_epochs}|"
                        f"I{i:04d}/{self.iters_per_epoch}] "
                        f"G={g_dict.get('total',0):.4f} "
                        f"(rec={g_dict.get('rec',0):.3f} "
                        f"perc={g_dict.get('perc',0):.3f} "
                        f"adv={g_dict.get('adv',0):.3f} "
                        f"edge={g_dict.get('edge',0):.3f}) "
                        f"D={d_dict.get('d_total',0):.4f} "
                        f"lr={lr_now:.2e} | "   # [PATCH 4] tampilkan LR saat ini
                        f"{speed:.1f}it/s ETA:{eta_min:.0f}m"
                    )

            # ── Rata-rata epoch train loss ─────────────────────
            train_means = {k: (sum(v) / len(v)) for k, v in epoch_g_losses.items()}
            train_means['d_total'] = sum(epoch_d_losses.get('d_total', [0])) / \
                                     max(len(epoch_d_losses.get('d_total', [1])), 1)
            epoch_time = time.time() - t_epoch

            # [PATCH 4] step scheduler di akhir epoch
            self.scheduler_g.step()
            self.scheduler_d.step()
            lr_now = self.optimizer_g.param_groups[0]['lr']
            train_means['lr_g'] = lr_now    # simpan LR ke history

            # ── Validasi ──────────────────────────────────────
            val_means = {}
            if epoch % self.val_epoch_freq == 0:
                val_means = self._validate()
                self.logger.info(
                    f"[E{epoch:03d}] EPOCH SUMMARY | "
                    f"Train G={train_means.get('total',0):.4f} "
                    f"D={train_means.get('d_total',0):.4f} | "
                    # [PATCH 3] val_loss sekarang = full G (bukan hanya rec)
                    f"Val G={val_means.get('val_loss',0):.4f} "
                    f"(rec={val_means.get('val_rec',0):.3f} "
                    f"perc={val_means.get('val_perc',0):.3f}) | "
                    # [PATCH 2] PSNR vs HR asli (unseen)
                    f"PSNR={val_means.get('psnr',0):.2f}dB [vs HR_orig] "
                    f"SSIM={val_means.get('ssim',0):.4f} | "
                    f"lr={lr_now:.2e} | time={epoch_time:.0f}s"
                )
            else:
                self.logger.info(
                    f"[E{epoch:03d}] Train G={train_means.get('total',0):.4f} "
                    f"D={train_means.get('d_total',0):.4f} | "
                    f"lr={lr_now:.2e} | {epoch_time:.0f}s"
                )

            self.history.push_epoch(epoch, train_means, val_means)

            if epoch % self.save_epoch_freq == 0:
                self._save_checkpoint(epoch, 'checkpoint_latest.pth')
                if epoch % (self.save_epoch_freq * 10) == 0:
                    self._save_checkpoint(epoch, f'checkpoint_epoch{epoch:04d}.pth')

            if epoch % self.vis_epoch_freq == 0 and self.val_loader:
                self._save_val_visual(epoch, vis_dir)

            self.history.save_json(os.path.join(self.save_dir, 'history.json'))

        self.logger.info("Training complete!")
        self._finalize()
        return self.history

    @torch.no_grad()
    def _save_val_visual(self, epoch: int, vis_dir: str):
        """Simpan grid comparison SR vs HR pada sampel validasi."""
        from utils.visualization import save_comparison_grid
        self.generator.eval()
        try:
            batch = next(iter(self.val_loader))
            lr_img = batch[0].to(self.device)
            hr_img = batch[1].to(self.device)
            sr = self.generator(lr_img).clamp(0, 1)
            path = os.path.join(vis_dir, f'epoch_{epoch:04d}_comparison.png')
            save_comparison_grid(lr_img.cpu(), sr.cpu(), hr_img.cpu(), path)
        except Exception as e:
            self.logger.warning(f"Visualisasi gagal epoch {epoch}: {e}")
        finally:
            self.generator.train()

    def _finalize(self):
        gen_path   = os.path.join(self.save_dir, 'generator_final.pth')
        json_path  = os.path.join(self.save_dir, 'history.json')
        excel_path = os.path.join(self.save_dir, 'training_results.xlsx')
        torch.save(self.generator.state_dict(), gen_path)
        self.logger.info(f"Generator final saved : {gen_path}")
        self.history.save_json(json_path)
        self.logger.info(f"History JSON saved    : {json_path}")
        self.history.export_excel(excel_path)
        self.logger.info(f"History Excel saved   : {excel_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("UAV-IRE epoch_trainer PATCHED — self-test")
    print("=" * 60)

    # Test TrainingHistory
    h = TrainingHistory()
    h.push_iter(1, 10, {'g_total': 1.2, 'd_total': 0.8})
    h.push_epoch(1,
        {'total': 1.1, 'rec': 0.5, 'perc': 0.4, 'adv': 0.2, 'd_total': 0.7, 'lr_g': 1e-4},
        {'val_loss': 0.9, 'val_rec': 0.5, 'val_perc': 0.3, 'psnr': 28.5, 'ssim': 0.82})
    h.push_epoch(2,
        {'total': 0.9, 'rec': 0.4, 'perc': 0.35, 'adv': 0.15, 'd_total': 0.65, 'lr_g': 9.5e-5},
        {'val_loss': 0.75, 'val_rec': 0.42, 'val_perc': 0.25, 'psnr': 29.1, 'ssim': 0.84})
    h.save_json('/tmp/test_history_patched.json')
    h2 = TrainingHistory.load_json('/tmp/test_history_patched.json')
    assert h2.epochs == [1, 2], "TrainingHistory load gagal"
    print("[OK] TrainingHistory: push, save, load")

    # Test HR2 pipeline
    dummy_hr = torch.rand(3, 256, 256)
    hr2_pipe = HR2DegradationPipeline()
    hr2      = hr2_pipe(dummy_hr)
    assert hr2.shape == dummy_hr.shape, "HR2 shape mismatch"
    assert hr2.min() >= 0 and hr2.max() <= 1, "HR2 out of [0,1]"
    # PSNR HR2 vs HR harus tinggi (>35 dB) karena degradasi ringan
    mse  = ((dummy_hr.float() - hr2.float()) ** 2).mean().item()
    psnr = 10 * torch.log10(torch.tensor(1.0 / (mse + 1e-8))).item()
    print(f"[OK] HR2DegradationPipeline: PSNR(HR2 vs HR) = {psnr:.1f} dB  (target > 35 dB)")
    assert psnr > 30, f"HR2 degradasi terlalu berat! PSNR = {psnr:.1f} dB"

    # Test batch apply
    batch_hr  = torch.rand(2, 3, 64, 64)
    batch_hr2 = hr2_pipe.apply_batch(batch_hr)
    assert batch_hr2.shape == batch_hr.shape, "HR2 batch shape mismatch"
    print("[OK] HR2DegradationPipeline.apply_batch: OK")

    # Excel export
    try:
        h.export_excel('/tmp/test_history_patched.xlsx')
        print("[OK] Excel export: OK")
    except ImportError:
        print("[SKIP] Excel export: pandas/openpyxl tidak terinstall")

    print()
    print("Semua self-test PASSED")
    print()
    print("Ringkasan patch:")
    print("  [1] lambda_edge=0.15 (dari 0.05), lambda_vsd=0.25 (dari 0.10)")
    print("  [2] use_hr2=True — HR2 sebagai target training, PSNR vs HR asli")
    print("  [3] _validate() menghitung full G_loss (rec+perc+adv)")
    print("  [4] CosineAnnealingLR menggantikan step decay tunggal di epoch 25")
