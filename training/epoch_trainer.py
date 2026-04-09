"""
UAV-IRE Epoch-Based Trainer
Training loop berbasis EPOCH (bukan iterasi) untuk:
- Grafik train loss vs valid loss per epoch (untuk laporan tesis)
- Auto-save checkpoint tiap epoch (untuk resume sesi Kaggle)
- Auto-resume dari checkpoint terakhir
- Export history ke Excel (.xlsx)
- Support semua skenario ablation SR1-SR6 (Tabel 3.1)

Rekomendasi: 200 epoch, batch 4, T4 single GPU
WeedyRice 438 train images → ~110 iter/epoch @ batch 4
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
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.network import PatchGAN_Discriminator
from losses.uav_ire_losses import UAVIRE_GeneratorLoss, UAVIRE_DiscriminatorLoss
from utils.metrics import calculate_psnr, calculate_ssim


# ─────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# History — train & val per epoch
# ─────────────────────────────────────────────

class TrainingHistory:
    """
    Menyimpan metrik training dan validasi per epoch.
    Mendukung export ke JSON dan Excel.
    """

    def __init__(self):
        # Per-epoch averages
        self.epochs: List[int] = []
        self.train: Dict[str, List[float]] = defaultdict(list)
        self.val:   Dict[str, List[float]] = defaultdict(list)

        # Per-iteration detail (untuk monitoring intra-epoch)
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
        """Kembalikan dua DataFrame: epoch summary dan iteration detail."""
        import pandas as pd

        # Epoch summary — pad semua kolom ke panjang yang sama
        n = len(self.epochs)
        epoch_data = {'epoch': self.epochs}
        for k, v in self.train.items():
            padded = list(v) + [None] * (n - len(v))
            epoch_data[f'train_{k}'] = padded[:n]
        for k, v in self.val.items():
            padded = list(v) + [None] * (n - len(v))
            epoch_data[f'val_{k}'] = padded[:n]
        df_epoch = pd.DataFrame(epoch_data)

        # Iteration detail
        df_iter = pd.DataFrame(self.iter_log) if self.iter_log else pd.DataFrame()

        return df_epoch, df_iter

    def export_excel(self, path: str):
        """Ekspor ke Excel dengan dua sheet: epoch summary dan iterasi detail."""
        import pandas as pd
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        df_epoch, df_iter = self.to_dataframes()
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            df_epoch.to_excel(writer, sheet_name='Epoch Summary', index=False)
            if not df_iter.empty:
                df_iter.to_excel(writer, sheet_name='Iteration Detail', index=False)
        return path


# ─────────────────────────────────────────────
# Epoch-Based Trainer
# ─────────────────────────────────────────────

class EpochTrainer:
    """
    UAV-IRE Trainer berbasis epoch.

    Fitur utama:
    - Loop: for epoch in range(total_epochs)
    - Catat train loss rata-rata per epoch
    - Catat val loss (PSNR, SSIM) per epoch
    - Auto-save checkpoint tiap epoch
    - Auto-resume dari checkpoint_latest.pth
    - Export history ke JSON + Excel
    - Support semua ablation SR1–SR6

    Args
    ----
    experiment : str
        Preset ablation ('full'|'baseline'|'no_nrdb'|'no_mbcm'|'no_ega'|'no_vsd')
    hr_train_dir : str
    hr_val_dir   : str
    save_dir     : str
    total_epochs : int        default 200
    batch_size   : int        default 4  (T4 single, model besar)
    gt_patch_size: int        default 256
    lr_g / lr_d  : float      1e-4
    lr_decay_epoch: int       epoch di mana LR dikurangi ½ (default 100)
    pretrained_path: str      path ke Real-ESRGAN / IRE pretrained weights
    """

    # Mapping preset ke ablation flags
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
        # ── WeedyRice dataset ─────────────────────────────
        dataset_root: Optional[str] = None,   # path ke WeedyRice-RGBMS-DB/
        train_list: Optional[str] = None,     # path ke train_list.txt
        val_list:   Optional[str] = None,     # path ke val_list.txt
        # ── Fallback untuk dataset generik ────────────────
        hr_train_dir: Optional[str] = None,
        hr_val_dir:   Optional[str] = None,
        # ──────────────────────────────────────────────────
        save_dir: str = 'experiments',
        total_epochs: int = 50,
        batch_size: int = 4,
        gt_patch_size: int = 256,
        eval_patch_size: int = 512,          # patch untuk validasi
        scale_factor: int = 4,
        num_rrdb: int = 23,
        num_features: int = 64,
        lr_g: float = 1e-4,
        lr_d: float = 1e-4,
        lr_decay_epoch: int = 25,            # ~epoch 25 untuk 50 epoch
        use_mask: bool = True,               # gunakan mask untuk VSD
        # Loss weights (Eq.2.43)
        lambda_rec: float = 1.0,
        lambda_perc: float = 1.0,
        lambda_adv: float = 0.1,
        lambda_edge: float = 0.05,
        lambda_vsd: float = 0.1,
        # Logging
        log_iter_freq: int = 10,
        val_epoch_freq: int = 1,
        save_epoch_freq: int = 1,
        vis_epoch_freq: int = 5,
        # Pretrained
        pretrained_path: Optional[str] = None,
        # AMP
        use_amp: bool = True,
        num_workers: int = 2,
        device: Optional[torch.device] = None,
    ):
        assert experiment in self.PRESETS, \
            f"experiment harus salah satu dari: {list(self.PRESETS)}"

        self.experiment  = experiment
        self.flags       = self.PRESETS[experiment]
        self.exp_name    = self.PRESET_NAMES[experiment]
        self.save_dir    = os.path.join(save_dir, self.exp_name)
        self.total_epochs = total_epochs
        self.lr_decay_epoch = lr_decay_epoch
        self.log_iter_freq  = log_iter_freq
        self.val_epoch_freq = val_epoch_freq
        self.save_epoch_freq = save_epoch_freq
        self.vis_epoch_freq  = vis_epoch_freq
        self.use_amp   = use_amp and torch.cuda.is_available()
        self.device    = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_mask  = use_mask and self.flags.get('use_vsd', False)

        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = setup_logger(self.save_dir, name=self.exp_name)
        self.logger.info(f"Experiment : {self.exp_name}")
        self.logger.info(f"Device     : {self.device}")
        self.logger.info(f"Flags      : {self.flags}")
        self.logger.info(f"Use mask   : {self.use_mask} (VSD)")

        # ── Build models ──────────────────────────────────────
        self._build_models(num_rrdb, scale_factor, num_features,
                           lambda_rec, lambda_perc, lambda_adv,
                           lambda_edge, lambda_vsd)

        # ── Optimisers ────────────────────────────────────────
        self.optimizer_g = optim.Adam(
            self.generator.parameters(), lr=lr_g, betas=(0.9, 0.99))
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=lr_d, betas=(0.9, 0.99))
        self.scaler_g = GradScaler(enabled=self.use_amp)
        self.scaler_d = GradScaler(enabled=self.use_amp)

        # ── DataLoaders ───────────────────────────────────────
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

        # ── History & resume state ────────────────────────────
        self.history    = TrainingHistory()
        self.start_epoch = 1

        # Load pretrained jika ada (sebelum cek resume)
        if pretrained_path and os.path.isfile(pretrained_path):
            self._load_pretrained(pretrained_path)

        # Auto-resume dari checkpoint terbaru
        latest_ckpt = os.path.join(self.save_dir, 'checkpoint_latest.pth')
        if os.path.isfile(latest_ckpt):
            self._resume(latest_ckpt)

    # ──────────────────────────────────────────────────────────
    # Build helpers
    # ──────────────────────────────────────────────────────────

    def _build_models(self, num_rrdb, scale_factor, num_features,
                      lambda_rec, lambda_perc, lambda_adv,
                      lambda_edge, lambda_vsd):
        """Bangun generator sesuai flag ablation."""
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

    # ── FIXED: signature sekarang menerima semua parameter yang diperlukan ──
    def _build_dataloaders(self,
                           dataset_root=None,
                           train_list=None,
                           val_list=None,
                           hr_train_dir=None,
                           hr_val_dir=None,
                           batch_size=4,
                           gt_patch_size=256,
                           eval_patch_size=512,
                           scale_factor=4,
                           num_workers=2):
        # ── Pilih dataset: WeedyRice (prioritas) atau generik ──
        if self.flags['use_uav_deg']:
            from data.uav_degradation import UAVSpecificDegradationPipeline
            pipeline = UAVSpecificDegradationPipeline(scale_factor=scale_factor)
            self.logger.info("Degradation: UAV-Specific")
        else:
            from data.degradation import IRE_DegradationPipeline
            pipeline = IRE_DegradationPipeline(scale_factor=scale_factor)
            self.logger.info("Degradation: IRE Second-Order")

        if dataset_root and os.path.isdir(dataset_root):
            # ── WeedyRice-RGBMS-DB ────────────────────────────
            from data.weedyrice_dataset import (
                WeedyRiceTrainDataset, WeedyRiceValDataset,
                make_train_loader, make_val_loader
            )
            # Cari list files
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
            self.logger.info(f"Dataset: WeedyRice-RGBMS-DB")

        else:
            # ── Generik (fallback) ─────────────────────────────
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

    # ──────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ──────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, filename: str):
        ckpt = {
            'epoch': epoch,
            'experiment': self.experiment,
            'generator':  self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
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
        if 'history_json' in ckpt:
            d = json.loads(ckpt['history_json'])
            from collections import defaultdict
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

    def _adjust_lr(self, epoch: int):
        if epoch == self.lr_decay_epoch:
            for pg in self.optimizer_g.param_groups:
                pg['lr'] *= 0.5
            for pg in self.optimizer_d.param_groups:
                pg['lr'] *= 0.5
            lr_now = self.optimizer_g.param_groups[0]['lr']
            self.logger.info(f"LR halved at epoch {epoch} → {lr_now:.2e}")

    # ──────────────────────────────────────────────────────────
    # Train / Val steps
    # ──────────────────────────────────────────────────────────

    def _train_step(self, batch):
        """
        Satu step training. batch bisa berupa:
        - (lr, hr, mask)  dari WeedyRiceTrainDataset
        - (lr, hr)        dari dataset generik
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

        # ── Discriminator ──
        self.optimizer_d.zero_grad(set_to_none=True)
        with autocast(enabled=self.use_amp):
            with torch.no_grad():
                sr = self.generator(lr_img)
            real_p = self.discriminator(hr_img)
            fake_p = self.discriminator(sr.detach())
            d_loss, d_dict = self.disc_loss_fn(real_p, fake_p)
        self.scaler_d.scale(d_loss).backward()
        self.scaler_d.step(self.optimizer_d)
        self.scaler_d.update()

        # ── Generator ──
        self.optimizer_g.zero_grad(set_to_none=True)
        with autocast(enabled=self.use_amp):
            sr = self.generator(lr_img)
            real_p = self.discriminator(hr_img).detach()
            fake_p = self.discriminator(sr)
            g_loss, g_dict = self.gen_loss_fn(
                sr, hr_img, fake_p, real_p,
                weed_mask=weed_mask,       # ← diteruskan ke VSD loss
                vsd_module=self.vsd,
            )
        self.scaler_g.scale(g_loss).backward()
        self.scaler_g.step(self.optimizer_g)
        self.scaler_g.update()

        return g_dict, d_dict

    @torch.no_grad()
    def _validate(self):
        if not self.val_loader:
            return {}
        self.generator.eval()
        psnr_list, ssim_list, rec_losses = [], [], []

        from losses.uav_ire_losses import SmoothL1ReconstructionLoss
        rec_fn = SmoothL1ReconstructionLoss()

        for batch in self.val_loader:
            # WeedyRice: (lr, hr, mask, filenames)
            # Generic:   (lr, hr, [mask], filename)
            if len(batch) == 4:
                lr_img, hr_img, _, _ = batch
            elif len(batch) == 3:
                lr_img, hr_img, _ = batch
            else:
                lr_img, hr_img = batch[0], batch[1]

            lr_img = lr_img.to(self.device)
            hr_img = hr_img.to(self.device)
            sr     = self.generator(lr_img).clamp(0, 1)

            with autocast(enabled=self.use_amp):
                l_rec = rec_fn(sr, hr_img).item()
            rec_losses.append(l_rec)

            for i in range(sr.shape[0]):
                psnr_list.append(calculate_psnr(sr[i], hr_img[i]))
                ssim_list.append(calculate_ssim(sr[i], hr_img[i]))

        self.generator.train()
        return {
            'val_loss': sum(rec_losses) / len(rec_losses),
            'psnr':     sum(psnr_list)  / len(psnr_list),
            'ssim':     sum(ssim_list)  / len(ssim_list),
        }

    # ──────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────

    def train(self) -> TrainingHistory:
        n_params = sum(p.numel() for p in self.generator.parameters())
        self.logger.info("=" * 60)
        self.logger.info(f"Training {self.exp_name}")
        self.logger.info(f"  Generator params : {n_params:,}")
        self.logger.info(f"  Total epochs     : {self.total_epochs}")
        self.logger.info(f"  Iters/epoch      : {self.iters_per_epoch}")
        self.logger.info(f"  Estimated total  : "
                         f"{self.total_epochs * self.iters_per_epoch:,} iterations")
        self.logger.info("=" * 60)

        vis_dir = os.path.join(self.save_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        global_iter = (self.start_epoch - 1) * self.iters_per_epoch

        for epoch in range(self.start_epoch, self.total_epochs + 1):
            self._adjust_lr(epoch)
            self.generator.train()
            self.discriminator.train()

            # ── Accumulators untuk rata-rata epoch ──────────
            epoch_g_losses: Dict[str, List[float]] = defaultdict(list)
            epoch_d_losses: Dict[str, List[float]] = defaultdict(list)
            t_epoch = time.time()

            for i, batch in enumerate(self.train_loader, 1):
                global_iter += 1
                g_dict, d_dict = self._train_step(batch)

                # Kumpulkan untuk rata-rata epoch
                for k, v in g_dict.items():
                    epoch_g_losses[k].append(v)
                for k, v in d_dict.items():
                    epoch_d_losses[k].append(v)

                # Log per-iterasi
                if i % self.log_iter_freq == 0:
                    iter_losses = {f'g_{k}': v for k, v in g_dict.items()}
                    iter_losses.update({f'd_{k}': v for k, v in d_dict.items()})
                    self.history.push_iter(epoch, global_iter, iter_losses)

                    eta_iter = (self.iters_per_epoch - i) + \
                               (self.total_epochs - epoch) * self.iters_per_epoch
                    speed = i / (time.time() - t_epoch + 1e-8)
                    eta_min = eta_iter / speed / 60
                    self.logger.info(
                        f"[E{epoch:03d}/{self.total_epochs}|"
                        f"I{i:04d}/{self.iters_per_epoch}] "
                        f"G={g_dict.get('total',0):.4f} "
                        f"(rec={g_dict.get('rec',0):.3f} "
                        f"perc={g_dict.get('perc',0):.3f} "
                        f"adv={g_dict.get('adv',0):.3f} "
                        f"edge={g_dict.get('edge',0):.3f}) "
                        f"D={d_dict.get('d_total',0):.4f} | "
                        f"{speed:.1f}it/s ETA:{eta_min:.0f}m"
                    )

            # ── Rata-rata epoch train loss ───────────────────
            train_means = {k: (sum(v) / len(v)) for k, v in epoch_g_losses.items()}
            train_means['d_total'] = sum(epoch_d_losses.get('d_total', [0])) / \
                                     max(len(epoch_d_losses.get('d_total', [1])), 1)
            epoch_time = time.time() - t_epoch

            # ── Validasi ─────────────────────────────────────
            val_means = {}
            if epoch % self.val_epoch_freq == 0:
                val_means = self._validate()
                self.logger.info(
                    f"[E{epoch:03d}] EPOCH SUMMARY | "
                    f"Train G={train_means.get('total',0):.4f} "
                    f"D={train_means.get('d_total',0):.4f} | "
                    f"Val loss={val_means.get('val_loss',0):.4f} "
                    f"PSNR={val_means.get('psnr',0):.2f}dB "
                    f"SSIM={val_means.get('ssim',0):.4f} | "
                    f"time={epoch_time:.0f}s"
                )
            else:
                self.logger.info(
                    f"[E{epoch:03d}] Train G={train_means.get('total',0):.4f} "
                    f"D={train_means.get('d_total',0):.4f} | {epoch_time:.0f}s"
                )

            # ── Push ke history ───────────────────────────────
            self.history.push_epoch(epoch, train_means, val_means)

            # ── Simpan checkpoint ─────────────────────────────
            if epoch % self.save_epoch_freq == 0:
                self._save_checkpoint(epoch, 'checkpoint_latest.pth')
                if epoch % (self.save_epoch_freq * 10) == 0:
                    self._save_checkpoint(epoch, f'checkpoint_epoch{epoch:04d}.pth')

            # ── Visualisasi ───────────────────────────────────
            if epoch % self.vis_epoch_freq == 0 and self.val_loader:
                self._save_val_visual(epoch, vis_dir)

            # ── Auto-save JSON history ────────────────────────
            self.history.save_json(os.path.join(self.save_dir, 'history.json'))

        # ── Selesai ───────────────────────────────────────────
        self.logger.info("Training complete!")
        self._finalize()
        return self.history

    @torch.no_grad()
    def _save_val_visual(self, epoch: int, vis_dir: str):
        """Simpan grid comparison SR vs HR pada sampel validasi."""
        from utils.visualization import save_comparison_grid
        self.generator.eval()
        try:
            lr_img, hr_img, _ = next(iter(self.val_loader))
            lr_img = lr_img.to(self.device)
            hr_img = hr_img.to(self.device)
            sr = self.generator(lr_img).clamp(0, 1)
            path = os.path.join(vis_dir, f'epoch_{epoch:04d}_comparison.png')
            save_comparison_grid(lr_img.cpu(), sr.cpu(), hr_img.cpu(), path)
        except Exception as e:
            self.logger.warning(f"Visualisasi gagal epoch {epoch}: {e}")
        finally:
            self.generator.train()

    def _finalize(self):
        """Simpan model final, history JSON, dan Excel."""
        # Generator final
        gen_path = os.path.join(self.save_dir, 'generator_final.pth')
        torch.save(self.generator.state_dict(), gen_path)
        self.logger.info(f"Generator final saved: {gen_path}")

        # History JSON
        json_path = os.path.join(self.save_dir, 'history.json')
        self.history.save_json(json_path)
        self.logger.info(f"History JSON saved: {json_path}")

        # History Excel
        excel_path = os.path.join(self.save_dir, 'training_results.xlsx')
        self.history.export_excel(excel_path)
        self.logger.info(f"History Excel saved: {excel_path}")


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("TrainingHistory smoke test...")
    h = TrainingHistory()
    h.push_iter(1, 10, {'g_total': 1.2, 'd_total': 0.8})
    h.push_epoch(1, {'total': 1.1, 'rec': 0.5, 'perc': 0.4, 'adv': 0.2, 'd_total': 0.7},
                    {'val_loss': 0.9, 'psnr': 28.5, 'ssim': 0.82})
    h.push_epoch(2, {'total': 0.9, 'rec': 0.4, 'perc': 0.35, 'adv': 0.15, 'd_total': 0.65},
                    {'val_loss': 0.75, 'psnr': 29.1, 'ssim': 0.84})
    h.save_json('/tmp/test_history.json')
    h2 = TrainingHistory.load_json('/tmp/test_history.json')
    assert h2.epochs == [1, 2]
    print("TrainingHistory: OK")

    # Excel export (butuh pandas + openpyxl)
    try:
        h.export_excel('/tmp/test_history.xlsx')
        print("Excel export: OK")
    except ImportError:
        print("Excel export: skip (pandas/openpyxl tidak terinstall)")

    print("EpochTrainer smoke test PASSED")
