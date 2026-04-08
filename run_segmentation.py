"""
run_segmentation.py — Evaluasi Segmentasi Gulma (Tabel 3.2)
YOLOv8n-seg | Fine-tune 1-3 epoch backbone frozen | WeedyRice-RGBMS-DB

Alur:
  1. Konversi mask PNG → YOLO polygon format
  2. Fine-tune YOLOv8n-seg: 3 epoch, backbone frozen
  3. Evaluasi SEG1–SEG5: mIoU + overlay visual per gambar
  4. Export Excel + bar chart + ZIP

Penggunaan:
  # Default (Kaggle)
  python run_segmentation.py

  # Lokal
  python run_segmentation.py \\
    --dataset_root /path/WeedyRice-RGBMS-DB \\
    --sr_results   experiments/ \\
    --output_dir   seg_results/

  # Pakai model yang sudah ada (skip fine-tune)
  python run_segmentation.py \\
    --yolo_model   yolov8n_weed_best.pt
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KAGGLE_BASE = (
    '/kaggle/input/datasets/muhamadsyaifulhuda/'
    'weedy-rice-uav-segmentation/'
    'A Dataset of Aligned RGB and Multispectral UAV Ima/'
    'WeedyRice-RGBMS-DB'
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_root', default=f'{KAGGLE_BASE}/WeedyRice-RGBMS-DB')
    p.add_argument('--list_base',    default=KAGGLE_BASE)
    p.add_argument('--sr_results',   default='/kaggle/working/experiments')
    p.add_argument('--output_dir',   default='/kaggle/working/segmentation')
    p.add_argument('--yolo_model',   default=None,
                   help='Path ke model .pt yang sudah ada (skip fine-tune)')
    p.add_argument('--epochs',       type=int, default=3)
    p.add_argument('--batch',        type=int, default=8)
    p.add_argument('--imgsz',        type=int, default=640)
    p.add_argument('--device',       default='')
    p.add_argument('--no_visuals',   action='store_true')
    p.add_argument('--max_images',   type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    ds   = args.dataset_root
    lb   = args.list_base
    out  = args.output_dir
    os.makedirs(out, exist_ok=True)

    train_list = os.path.join(lb, 'train_list.txt')
    val_list   = os.path.join(lb, 'val_list.txt')
    test_list  = os.path.join(lb, 'test_list.txt')

    print("=" * 60)
    print("UAV-IRE Segmentation Evaluation (Tabel 3.2)")
    print("YOLOv8n-seg | Fine-tune 1-3 epoch | backbone frozen")
    print("=" * 60)

    # ── 1. YOLO dataset ────────────────────────────────────
    yolo_ds = os.path.join(out, 'yolo_dataset')
    print(f"\n[Step 1/3] Menyiapkan YOLO dataset...")
    from segmentation.yolo_dataset_builder import build_yolo_dataset
    data_yaml = build_yolo_dataset(
        dataset_root    = ds,
        output_dir      = yolo_ds,
        train_list_path = train_list,
        val_list_path   = val_list,
        test_list_path  = test_list,
        copy_images     = False,
        min_area_px     = 200,
    )

    # ── 2. Fine-tune / load model ──────────────────────────
    ft_dir  = os.path.join(out, 'yolo_finetune')
    ft_ckpt = os.path.join(ft_dir, 'yolov8n_weed_best.pt')

    if args.yolo_model and os.path.isfile(args.yolo_model):
        model_path = args.yolo_model
        print(f"\n[Step 2/3] Menggunakan model: {model_path}")

    elif os.path.isfile(ft_ckpt):
        model_path = ft_ckpt
        print(f"\n[Step 2/3] Model sudah ada (skip fine-tune): {ft_ckpt}")

    else:
        print(f"\n[Step 2/3] Fine-tuning YOLOv8n-seg ({args.epochs} epoch)...")
        from segmentation.yolo_finetune import finetune_yolov8n_seg
        model_path = finetune_yolov8n_seg(
            data_yaml  = data_yaml,
            output_dir = ft_dir,
            epochs     = args.epochs,
            img_size   = args.imgsz,
            batch_size = args.batch,
            device     = args.device,
        )

    print(f"  Model: {model_path}")

    # ── 3. Evaluasi SEG1–SEG5 ─────────────────────────────
    print(f"\n[Step 3/3] Evaluasi segmentasi SEG1–SEG5...")
    from segmentation.seg_eval import run_all_scenarios
    summaries = run_all_scenarios(
        dataset_root    = ds,
        sr_results_base = args.sr_results,
        yolo_model_path = model_path,
        output_dir      = os.path.join(out, 'evaluation'),
        test_list       = test_list,
        device          = args.device,
        save_visuals    = not args.no_visuals,
        max_per_scenario = args.max_images,
    )

    # ── Ringkasan akhir ────────────────────────────────────
    print("\n" + "=" * 60)
    print("HASIL AKHIR EVALUASI SEGMENTASI (Tabel 3.2)")
    print("=" * 60)
    print(f"{'Skenario':<32}  {'mIoU':>6}  {'IoU_w':>6}  {'F1':>6}")
    print("─" * 56)
    for sc, m in summaries.items():
        if m:
            print(f"{sc:<32}  {m.get('mIoU',0):>6.4f}  "
                  f"{m.get('iou_weed',0):>6.4f}  {m.get('F1',0):>6.4f}")
    print("=" * 60)
    print(f"\nOutput: {os.path.join(out, 'evaluation')}")


if __name__ == '__main__':
    main()
