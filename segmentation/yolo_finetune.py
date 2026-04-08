"""
YOLOv8n-seg Fine-Tuning (Sangat Minimal)
Model  : YOLOv8n-seg (nano) — paling ringan dan cepat
Strategi: backbone FROZEN, hanya train segmentation head
Durasi : 1–3 epoch saja — cukup agar model mengenal kelas "weed"

Justifikasi untuk tesis:
- COCO pretrained tidak memiliki kelas "weed/gulma"
- Training 1–3 epoch dengan backbone frozen = dampak minimal
  pada bobot jaringan, sehingga tidak mengaburkan perbandingan
  dampak SR antar skenario SEG1–SEG5
- Model yang sama (satu best.pt) dipakai untuk SEMUA skenario
  → fair comparison

Referensi: Yang et al. (2025) — YOLOv8-seg sebagai model evaluatif
"""

import os
import shutil
from pathlib import Path
from typing import Optional


def finetune_yolov8n_seg(
    data_yaml: str,
    output_dir: str,
    epochs: int = 3,             # 1–3 epoch cukup untuk mengenal kelas weed
    img_size: int = 640,
    batch_size: int = 8,
    lr0: float = 1e-3,           # LR lebih tinggi karena epoch sedikit
    device: str = '',
) -> str:
    """
    Fine-tune YOLOv8n-seg pada WeedyRice dengan backbone sepenuhnya frozen.
    Hanya segmentation head yang dilatih.

    Args:
        data_yaml   : Path ke data.yaml dataset YOLO
        output_dir  : Folder simpan model
        epochs      : Jumlah epoch (1–3, default 3)
        img_size    : Ukuran input YOLO (640)
        batch_size  : Batch size (8 di T4)
        lr0         : Learning rate
        device      : '' = auto-detect GPU/CPU

    Returns:
        Path ke best.pt model
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics belum terinstall.\n"
            "Jalankan: pip install ultralytics"
        )

    os.makedirs(output_dir, exist_ok=True)

    # Load YOLOv8n-seg pretrained COCO
    print("Loading yolov8n-seg.pt (pretrained COCO)...")
    model = YOLO('yolov8n-seg.pt')

    # Freeze: layer 0–9 = backbone YOLOv8n
    # Neck (layer 10–18) + Head (layer 19–22) tetap trainable
    # Dengan 3 epoch, neck+head belajar mengenal pola gulma UAV
    FREEZE_N_LAYERS = 10

    print(f"\nFine-tuning YOLOv8n-seg — WeedyRice Weed Segmentation")
    print(f"  Backbone (0-9)  : FROZEN sepenuhnya")
    print(f"  Neck + Head     : Trainable")
    print(f"  Epochs          : {epochs}")
    print(f"  Batch size      : {batch_size}")
    print(f"  Learning rate   : {lr0}")
    print(f"  Data            : {data_yaml}")
    print()

    results = model.train(
        data         = data_yaml,
        epochs       = epochs,
        imgsz        = img_size,
        batch        = batch_size,
        lr0          = lr0,
        lrf          = 0.1,           # LR final = lr0 * lrf
        freeze       = FREEZE_N_LAYERS,
        patience     = epochs,        # jangan early stop — epoch sudah sangat sedikit
        project      = output_dir,
        name         = 'yolov8n_weed',
        exist_ok     = True,
        device       = device,
        task         = 'segment',
        # Augmentasi minimal — kita evaluator, bukan model utama
        hsv_h        = 0.01,
        hsv_s        = 0.5,
        hsv_v        = 0.3,
        degrees      = 3.0,
        translate    = 0.05,
        scale        = 0.2,
        fliplr       = 0.5,
        flipud       = 0.1,
        mosaic       = 0.3,
        close_mosaic = 1,
        # Checkpoint
        save         = True,
        save_period  = 1,
        val          = True,
        plots        = True,
        verbose      = True,
        # Tidak pakai DDP / kompleksitas tambahan
        workers      = 2,
    )

    # Cari best.pt
    run_dir = Path(str(results.save_dir)) \
              if hasattr(results, 'save_dir') \
              else Path(output_dir) / 'yolov8n_weed'

    best_pt  = run_dir / 'weights' / 'best.pt'
    last_pt  = run_dir / 'weights' / 'last.pt'

    # Fallback ke last.pt jika best.pt tidak ada
    chosen = best_pt if best_pt.is_file() else last_pt

    final_pt = os.path.join(output_dir, 'yolov8n_weed_best.pt')
    if chosen.is_file():
        shutil.copy2(str(chosen), final_pt)
        print(f"\nModel saved: {final_pt}")
        return final_pt
    else:
        print(f"\n[WARN] Checkpoint tidak ditemukan di {run_dir}")
        return str(chosen)
