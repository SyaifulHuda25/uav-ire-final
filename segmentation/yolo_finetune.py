"""
YOLOv8n-seg Fine-Tuning — WeedyRice Weed Segmentation

Dua mode freeze (lihat arg `freeze_mode`):
  'frozen'  : backbone (layer 0-9) FROZEN — untuk fair comparison antar
              skenario SEG1-SEG5. LR rendah, epoch sedang.
  'full'    : seluruh jaringan trainable — untuk mIoU maksimal. Pakai
              warmup + cosine decay + early stopping.

Perbaikan vs versi lama:
  - LR schedule benar: lr0 disesuaikan per-mode, cosine decay (cos_lr=True),
    warmup eksplisit. Versi lama lr0=1e-3 + 100 epoch + lrf=0.1 membuat
    head berosilasi sehingga mIoU stagnan.
  - Early stopping nyata (patience), bukan patience=epochs.
  - Optimizer AdamW (lebih stabil untuk head pada data kecil).
  - close_mosaic 10 epoch terakhir agar tepi mask di-fine-tune tanpa mosaic.
"""

import os
import shutil
from pathlib import Path


def finetune_yolov8n_seg(
    data_yaml: str,
    output_dir: str,
    epochs: int = 80,
    img_size: int = 640,
    batch_size: int = 8,
    device: str = '',
    freeze_mode: str = 'frozen',     # 'frozen' | 'full'
    lr0: float = None,               # None = default per-mode
    patience: int = 20,
    run_name: str = None,
) -> str:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics belum terinstall. Jalankan: pip install ultralytics")

    if freeze_mode not in ('frozen', 'full'):
        raise ValueError("freeze_mode harus 'frozen' atau 'full'")

    os.makedirs(output_dir, exist_ok=True)

    if freeze_mode == 'frozen':
        freeze_layers = 10           # backbone YOLOv8n = layer 0-9
        default_lr0   = 1e-3
        warmup        = 3.0
    else:  # 'full'
        freeze_layers = 0
        default_lr0   = 1e-4         # backbone ikut dilatih -> LR kecil
        warmup        = 5.0

    lr0 = lr0 if lr0 is not None else default_lr0
    run_name = run_name or f'yolov8n_weed_{freeze_mode}'

    print("Loading yolov8n-seg.pt (pretrained COCO)...")
    model = YOLO('yolov8n-seg.pt')

    print(f"\nFine-tuning YOLOv8n-seg - mode='{freeze_mode}'")
    print(f"  Freeze layers : {freeze_layers} (0=full trainable)")
    print(f"  Epochs        : {epochs} (early stop patience={patience})")
    print(f"  lr0           : {lr0:.1e} | cosine decay | warmup={warmup}ep")
    print(f"  Batch / imgsz : {batch_size} / {img_size}")
    print(f"  Data          : {data_yaml}\n")

    model.train(
        data         = data_yaml,
        epochs       = epochs,
        imgsz        = img_size,
        batch        = batch_size,
        freeze       = freeze_layers,
        device       = device,
        task         = 'segment',

        # Optimizer & LR schedule (inti perbaikan)
        optimizer    = 'AdamW',
        lr0          = lr0,
        lrf          = 0.01,
        cos_lr       = True,
        warmup_epochs= warmup,
        weight_decay = 5e-4,
        momentum     = 0.937,

        patience     = patience,

        # Augmentasi sedang
        hsv_h        = 0.015,
        hsv_s        = 0.6,
        hsv_v        = 0.4,
        degrees      = 5.0,
        translate    = 0.1,
        scale        = 0.3,
        fliplr       = 0.5,
        flipud       = 0.2,
        mosaic       = 0.5,
        close_mosaic = 10,

        project      = output_dir,
        name         = run_name,
        exist_ok     = True,
        save         = True,
        val          = True,
        plots        = True,
        verbose      = True,
        workers      = 2,
    )

    run_dir = Path(output_dir) / run_name
    best_pt = run_dir / 'weights' / 'best.pt'
    last_pt = run_dir / 'weights' / 'last.pt'
    chosen  = best_pt if best_pt.is_file() else last_pt

    final_pt = os.path.join(output_dir, 'yolov8n_weed_best.pt')
    if chosen.is_file():
        shutil.copy2(str(chosen), final_pt)
        print(f"\nModel saved: {final_pt}  (from {chosen.name})")
        return final_pt

    print(f"\n[WARN] Checkpoint tidak ditemukan di {run_dir}")
    return str(chosen)