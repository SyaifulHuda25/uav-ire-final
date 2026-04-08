"""
UAV-IRE Training Entry Point
Mendukung semua skenario eksperimen sesuai Tabel 3.1 dan 3.2 proposal:

Training Skenario (Tabel 3.1):
  SR1: IRE Baseline            --experiment baseline
  SR2: UAV-IRE tanpa NRDB      --no_nrdb
  SR3: UAV-IRE tanpa MBCM      --no_mbcm
  SR4: UAV-IRE tanpa EGA       --no_ega
  SR5: UAV-IRE tanpa VSD       --no_vsd
  SR6: UAV-IRE Lengkap         --experiment full (default)

Contoh penggunaan:
  python train_uav_ire.py --hr_train /data/WeedyRice/train --hr_val /data/WeedyRice/val
  python train_uav_ire.py --hr_train /data/DIV2K/train --experiment baseline
  python train_uav_ire.py --hr_train /data/train --no_nrdb --exp_name ablation_no_nrdb
  python train_uav_ire.py --hr_train /data/train --quick_test
"""

import os
import sys
import argparse
import yaml
from dataclasses import dataclass, asdict, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class UAVIREConfig:
    """Konfigurasi lengkap UAV-IRE training."""

    # Data
    hr_train_dir: str = 'data/WeedyRice/train_HR'
    hr_val_dir: Optional[str] = 'data/WeedyRice/val_HR'

    # Experiment
    exp_name: str = 'uav_ire_full'
    save_dir: str = 'experiments'

    # Model
    num_rrdb: int = 23
    scale_factor: int = 4
    num_features: int = 64

    # Ablation flags (Tabel 3.1)
    use_nrdb: bool = True
    use_mbcm: bool = True
    use_ega: bool = True
    use_vsd: bool = True
    use_uav_degradation: bool = True

    # Training
    total_iter: int = 100_000
    batch_size: int = 16
    gt_patch_size: int = 256
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    lr_decay_iter: int = 50_000

    # Loss weights (Eq.2.43)
    lambda_rec: float = 1.0
    lambda_perc: float = 1.0
    lambda_adv: float = 0.1
    lambda_edge: float = 0.05
    lambda_vsd: float = 0.1

    # VSD lambda
    vsd_lambda_g: float = 1.0
    vsd_lambda_w: float = 0.5
    vsd_lambda_v: float = 0.1

    # Misc
    use_amp: bool = True
    num_workers: int = 4
    val_freq: int = 5000
    save_freq: int = 10_000
    log_freq: int = 100
    vis_freq: int = 2000

    # Init
    pretrained_ire: Optional[str] = None
    resume_checkpoint: Optional[str] = None

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> 'UAVIREConfig':
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train UAV-IRE - Super-Resolution untuk Citra UAV'
    )

    # Config
    parser.add_argument('--config', type=str, default=None)

    # Data
    parser.add_argument('--hr_train_dir', type=str, default=None)
    parser.add_argument('--hr_val_dir', type=str, default=None)

    # Experiment
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)

    # Model
    parser.add_argument('--num_rrdb', type=int, default=None)
    parser.add_argument('--scale_factor', type=int, default=None)
    parser.add_argument('--num_features', type=int, default=None)

    # Ablation (Tabel 3.1)
    parser.add_argument('--no_nrdb', action='store_true', help='Nonaktifkan NRDB (SR2)')
    parser.add_argument('--no_mbcm', action='store_true', help='Nonaktifkan MBCM (SR3)')
    parser.add_argument('--no_ega', action='store_true', help='Nonaktifkan EGA (SR4)')
    parser.add_argument('--no_vsd', action='store_true', help='Nonaktifkan VSD (SR5)')
    parser.add_argument('--no_uav_degradation', action='store_true',
                        help='Gunakan degradasi IRE standar')

    # Preset experiments
    parser.add_argument('--experiment', type=str,
                        choices=['full', 'baseline', 'no_nrdb', 'no_mbcm',
                                 'no_ega', 'no_vsd'],
                        default=None,
                        help='Preset eksperimen sesuai Tabel 3.1')

    # Training
    parser.add_argument('--total_iter', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--gt_patch_size', type=int, default=None)
    parser.add_argument('--lr_g', type=float, default=None)
    parser.add_argument('--lr_d', type=float, default=None)

    # Loss weights
    parser.add_argument('--lambda_rec', type=float, default=None)
    parser.add_argument('--lambda_perc', type=float, default=None)
    parser.add_argument('--lambda_adv', type=float, default=None)
    parser.add_argument('--lambda_edge', type=float, default=None)
    parser.add_argument('--lambda_vsd', type=float, default=None)

    # Special modes
    parser.add_argument('--quick_test', action='store_true',
                        help='Mode test cepat (num_rrdb=4, iter=100)')
    parser.add_argument('--pretrained_ire', type=str, default=None,
                        help='Path ke pretrained IRE weights untuk inisialisasi')
    parser.add_argument('--resume', type=str, default=None,
                        dest='resume_checkpoint')

    return parser.parse_args()


# Preset konfigurasi untuk setiap eksperimen (Tabel 3.1)
EXPERIMENT_PRESETS = {
    'full': {
        'use_nrdb': True, 'use_mbcm': True, 'use_ega': True,
        'use_vsd': True, 'use_uav_degradation': True,
        'exp_name': 'SR6_uav_ire_full',
    },
    'baseline': {
        'use_nrdb': False, 'use_mbcm': False, 'use_ega': False,
        'use_vsd': False, 'use_uav_degradation': False,
        'exp_name': 'SR1_ire_baseline',
    },
    'no_nrdb': {
        'use_nrdb': False, 'use_mbcm': True, 'use_ega': True,
        'use_vsd': True, 'use_uav_degradation': True,
        'exp_name': 'SR2_uav_ire_no_nrdb',
    },
    'no_mbcm': {
        'use_nrdb': True, 'use_mbcm': False, 'use_ega': True,
        'use_vsd': True, 'use_uav_degradation': True,
        'exp_name': 'SR3_uav_ire_no_mbcm',
    },
    'no_ega': {
        'use_nrdb': True, 'use_mbcm': True, 'use_ega': False,
        'use_vsd': True, 'use_uav_degradation': True,
        'exp_name': 'SR4_uav_ire_no_ega',
    },
    'no_vsd': {
        'use_nrdb': True, 'use_mbcm': True, 'use_ega': True,
        'use_vsd': False, 'use_uav_degradation': True,
        'exp_name': 'SR5_uav_ire_no_vsd',
    },
}


def main():
    args = parse_args()

    # Build config
    if args.config:
        cfg = UAVIREConfig.from_yaml(args.config)
    else:
        cfg = UAVIREConfig()

    # Apply preset experiment
    if args.experiment and args.experiment in EXPERIMENT_PRESETS:
        preset = EXPERIMENT_PRESETS[args.experiment]
        for k, v in preset.items():
            setattr(cfg, k, v)

    # Apply explicit flags
    if args.no_nrdb:
        cfg.use_nrdb = False
    if args.no_mbcm:
        cfg.use_mbcm = False
    if args.no_ega:
        cfg.use_ega = False
    if args.no_vsd:
        cfg.use_vsd = False
    if args.no_uav_degradation:
        cfg.use_uav_degradation = False

    # Apply CLI overrides
    for attr in ['hr_train_dir', 'hr_val_dir', 'exp_name', 'save_dir',
                 'num_rrdb', 'scale_factor', 'num_features',
                 'total_iter', 'batch_size', 'gt_patch_size',
                 'lr_g', 'lr_d', 'lambda_rec', 'lambda_perc',
                 'lambda_adv', 'lambda_edge', 'lambda_vsd',
                 'pretrained_ire', 'resume_checkpoint']:
        val = getattr(args, attr, None)
        if val is not None:
            setattr(cfg, attr, val)

    # Quick test mode
    if args.quick_test:
        cfg.num_rrdb = 4
        cfg.num_features = 32
        cfg.total_iter = 100
        cfg.val_freq = 50
        cfg.save_freq = 100
        cfg.log_freq = 10
        cfg.vis_freq = 50
        cfg.batch_size = 2
        cfg.exp_name = cfg.exp_name + '_quicktest'
        print("[Mode] Quick Test")

    # Setup save directory
    exp_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    cfg_path = os.path.join(exp_dir, 'config.yaml')
    cfg.save(cfg_path)

    # Print config
    print("\n" + "=" * 65)
    print(f"UAV-IRE Training: {cfg.exp_name}")
    print("=" * 65)
    print(f"  Data train : {cfg.hr_train_dir}")
    print(f"  Data val   : {cfg.hr_val_dir}")
    print(f"  Save dir   : {exp_dir}")
    print(f"\n  Model:")
    print(f"    num_rrdb    : {cfg.num_rrdb}")
    print(f"    scale_factor: {cfg.scale_factor}x")
    print(f"    num_features: {cfg.num_features}")
    print(f"\n  Komponen aktif:")
    print(f"    NRDB      : {'✓' if cfg.use_nrdb else '✗'}")
    print(f"    MBCM      : {'✓' if cfg.use_mbcm else '✗'}")
    print(f"    EGA       : {'✓' if cfg.use_ega else '✗'}")
    print(f"    VSD       : {'✓' if cfg.use_vsd else '✗'}")
    print(f"    UAV Degrad: {'✓' if cfg.use_uav_degradation else '✗'}")
    print(f"\n  Training:")
    print(f"    Total iter  : {cfg.total_iter:,}")
    print(f"    Batch size  : {cfg.batch_size}")
    print(f"    GT patch    : {cfg.gt_patch_size}×{cfg.gt_patch_size}")
    print(f"\n  Loss weights (Eq.2.43):")
    print(f"    λ1 rec  : {cfg.lambda_rec}")
    print(f"    λ2 perc : {cfg.lambda_perc}")
    print(f"    λ3 adv  : {cfg.lambda_adv}")
    print(f"    λ4 edge : {cfg.lambda_edge} (auxiliary)")
    print(f"    λ5 vsd  : {cfg.lambda_vsd} (auxiliary)")
    print("=" * 65)

    from training.uav_ire_trainer import UAVIRE_Trainer

    trainer = UAVIRE_Trainer(
        hr_train_dir=cfg.hr_train_dir,
        hr_val_dir=cfg.hr_val_dir,
        save_dir=exp_dir,
        num_rrdb=cfg.num_rrdb,
        scale_factor=cfg.scale_factor,
        num_features=cfg.num_features,
        use_nrdb=cfg.use_nrdb,
        use_mbcm=cfg.use_mbcm,
        use_ega=cfg.use_ega,
        use_vsd=cfg.use_vsd,
        use_uav_degradation=cfg.use_uav_degradation,
        total_iter=cfg.total_iter,
        batch_size=cfg.batch_size,
        gt_patch_size=cfg.gt_patch_size,
        lr_g=cfg.lr_g,
        lr_d=cfg.lr_d,
        lr_decay_iter=cfg.lr_decay_iter,
        lambda_rec=cfg.lambda_rec,
        lambda_perc=cfg.lambda_perc,
        lambda_adv=cfg.lambda_adv,
        lambda_edge=cfg.lambda_edge,
        lambda_vsd=cfg.lambda_vsd,
        vsd_lambda_g=cfg.vsd_lambda_g,
        vsd_lambda_w=cfg.vsd_lambda_w,
        vsd_lambda_v=cfg.vsd_lambda_v,
        use_amp=cfg.use_amp,
        num_workers=cfg.num_workers,
        val_freq=cfg.val_freq,
        save_freq=cfg.save_freq,
        log_freq=cfg.log_freq,
        vis_freq=cfg.vis_freq,
        pretrained_ire=cfg.pretrained_ire,
        resume_checkpoint=cfg.resume_checkpoint,
    )

    trainer.train()
    print(f"\nTraining selesai! Hasil disimpan di: {exp_dir}")


if __name__ == '__main__':
    main()
