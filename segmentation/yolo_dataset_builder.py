"""
WeedyRice → YOLO Segmentation Format Converter
Mengkonversi mask binary PNG WeedyRice-RGBMS-DB ke format label YOLO-seg.

Format YOLO-seg per baris:
  class_id x1 y1 x2 y2 ... xn yn
  (koordinat normalized [0,1], polygon kontur tiap objek)

Struktur output YOLO:
  yolo_dataset/
  ├── images/
  │   ├── train/  ← symlink atau copy dari RGB/
  │   ├── val/
  │   └── test/
  ├── labels/
  │   ├── train/  ← file .txt hasil konversi
  │   ├── val/
  │   └── test/
  └── data.yaml   ← config dataset YOLO

Class:
  0 = weed (gulma / weedy rice)  ← putih pada mask
  (background = tidak dianotasi, YOLO implicit)
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────
# Konversi mask → YOLO polygon labels
# ─────────────────────────────────────────────

def mask_to_yolo_polygons(
    mask_path: str,
    min_area_px: int = 100,
    epsilon_factor: float = 0.005,
) -> List[str]:
    """
    Konversi mask binary PNG ke daftar string YOLO-seg.

    Args:
        mask_path:      Path ke file mask .png (putih=gulma)
        min_area_px:    Abaikan kontur kecil (noise) di bawah N piksel
        epsilon_factor: Faktor aproksimasi polygon (semakin besar = semakin kasar)

    Returns:
        List string YOLO label, satu baris per objek gulma:
        "0 x1 y1 x2 y2 ..."
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    H, W = mask.shape

    # Binarisasi
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Temukan kontur
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    yolo_lines = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue

        # Aproksimasi polygon (kurangi titik berlebihan)
        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Minimal 3 titik untuk polygon valid
        if len(approx) < 3:
            continue

        # Normalisasi koordinat ke [0, 1]
        points = approx.reshape(-1, 2)
        coords = []
        for x, y in points:
            coords.extend([
                round(float(x) / W, 6),
                round(float(y) / H, 6),
            ])

        # Format YOLO: "class_id x1 y1 x2 y2 ..."
        line = '0 ' + ' '.join(map(str, coords))
        yolo_lines.append(line)

    return yolo_lines


def has_weed(mask_path: str) -> bool:
    """Cek apakah mask memiliki area gulma (tidak semua hitam)."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    return mask.max() > 0


# ─────────────────────────────────────────────
# Build YOLO dataset dari WeedyRice
# ─────────────────────────────────────────────

def build_yolo_dataset(
    dataset_root: str,
    output_dir: str,
    train_list_path: str,
    val_list_path: str,
    test_list_path: str,
    img_size: int = 640,
    copy_images: bool = False,   # False = symlink (hemat disk Kaggle)
    min_area_px: int = 200,
) -> str:
    """
    Bangun dataset YOLO-seg dari WeedyRice-RGBMS-DB.

    Args:
        dataset_root:   Path ke WeedyRice-RGBMS-DB/
        output_dir:     Output folder yolo_dataset/
        train/val/test_list_path: path ke list file
        img_size:       Ukuran training YOLO (default 640)
        copy_images:    True=copy, False=symlink (hemat disk)
        min_area_px:    Minimum area kontur gulma (abaikan noise kecil)

    Returns:
        Path ke data.yaml
    """
    rgb_dir  = os.path.join(dataset_root, 'RGB')
    mask_dir = os.path.join(dataset_root, 'Masks')

    splits = {
        'train': train_list_path,
        'val':   val_list_path,
        'test':  test_list_path,
    }

    stats = {}

    for split, list_path in splits.items():
        img_out_dir   = os.path.join(output_dir, 'images', split)
        label_out_dir = os.path.join(output_dir, 'labels', split)
        os.makedirs(img_out_dir,   exist_ok=True)
        os.makedirs(label_out_dir, exist_ok=True)

        with open(list_path) as f:
            filenames = [l.strip() for l in f if l.strip()]

        n_total = n_weed = n_empty = n_missing = 0
        for fname in filenames:
            rgb_path  = os.path.join(rgb_dir, fname)
            stem      = Path(fname).stem
            mask_name = stem + '.png'
            mask_path = os.path.join(mask_dir, mask_name)

            if not os.path.isfile(rgb_path):
                n_missing += 1
                continue

            n_total += 1

            # ── Link/copy gambar ─────────────────────────
            img_dest = os.path.join(img_out_dir, fname)
            if not os.path.exists(img_dest):
                if copy_images:
                    shutil.copy2(rgb_path, img_dest)
                else:
                    # Symlink relatif
                    try:
                        os.symlink(os.path.abspath(rgb_path), img_dest)
                    except FileExistsError:
                        pass

            # ── Konversi mask → YOLO label ───────────────
            label_dest = os.path.join(label_out_dir, stem + '.txt')
            if os.path.isfile(mask_path):
                yolo_lines = mask_to_yolo_polygons(mask_path, min_area_px)
                with open(label_dest, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                if yolo_lines:
                    n_weed += 1
                else:
                    n_empty += 1
            else:
                # Tidak ada mask → file label kosong (semua background)
                open(label_dest, 'w').close()
                n_empty += 1

        stats[split] = {
            'total': n_total, 'weed': n_weed,
            'empty': n_empty, 'missing': n_missing
        }
        print(f"[{split:5s}] total={n_total} | weed={n_weed} | "
              f"empty={n_empty} | missing={n_missing}")

    # ── Buat data.yaml ──────────────────────────────────
    yaml_path = os.path.join(output_dir, 'data.yaml')
    yaml_content = f"""# WeedyRice-RGBMS-DB YOLO Segmentation Dataset
# Generated from WeedyRice-RGBMS-DB (Nguyen et al., 2025)
# Task: Weed (weedy rice) instance segmentation

path: {os.path.abspath(output_dir)}
train: images/train
val:   images/val
test:  images/test

nc: 1
names:
  0: weed

# Dataset stats
# train: {stats.get('train', {}).get('total', 0)} images
# val:   {stats.get('val',   {}).get('total', 0)} images
# test:  {stats.get('test',  {}).get('total', 0)} images
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nYOLO dataset ready: {output_dir}")
    print(f"data.yaml: {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────
# Buat dataset SR untuk evaluasi segmentasi
# Sesuai Tabel 3.2: SEG1(LR) SEG2(IRE) SEG3-5(ablation/full/HR)
# ─────────────────────────────────────────────

def build_sr_eval_dataset(
    sr_image_dir: str,
    original_label_dir: str,
    output_dir: str,
    split: str = 'test',
) -> str:
    """
    Buat dataset evaluasi YOLO dengan gambar SR tapi label dari ground-truth.

    Untuk SEG2–SEG4: gambar = SR hasil model, label = GT mask (tidak berubah).
    Label TIDAK di-generate ulang — kita evaluasi apakah SR membantu model
    mendeteksi area yang sama dengan yang ada di GT mask.

    Args:
        sr_image_dir:     Folder berisi gambar SR (dari run_inference)
        original_label_dir: Folder label YOLO dari build_yolo_dataset
        output_dir:       Output folder
        split:            'test'

    Returns:
        Path ke data.yaml
    """
    img_out = os.path.join(output_dir, 'images', split)
    lbl_out = os.path.join(output_dir, 'labels', split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    # Link gambar SR
    sr_files = sorted(Path(sr_image_dir).glob('SR_*.png'))
    n_linked = 0
    for sr_path in sr_files:
        dest = os.path.join(img_out, sr_path.name)
        if not os.path.exists(dest):
            try:
                os.symlink(os.path.abspath(str(sr_path)), dest)
            except FileExistsError:
                pass

        # Cari label yang sesuai: SR_stem.png → stem.txt
        # SR_{original_stem}.png → {original_stem}.txt
        original_stem = sr_path.stem[3:]  # buang prefix 'SR_'
        src_label = os.path.join(original_label_dir, original_stem + '.txt')
        dst_label = os.path.join(lbl_out, sr_path.stem + '.txt')
        if os.path.isfile(src_label) and not os.path.exists(dst_label):
            try:
                os.symlink(os.path.abspath(src_label), dst_label)
            except FileExistsError:
                pass
        elif not os.path.isfile(src_label):
            # Buat label kosong jika tidak ditemukan
            open(dst_label, 'w').close()
        n_linked += 1

    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"""path: {os.path.abspath(output_dir)}
train: images/train
val:   images/val
test:  images/{split}
nc: 1
names:
  0: weed
""")

    print(f"SR eval dataset: {n_linked} images → {output_dir}")
    return yaml_path


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import tempfile, numpy as np

    print("Testing mask_to_yolo_polygons...")
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        # Buat mask sintetis dengan 2 area putih
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)
        cv2.circle(mask, (150, 150), 30, 255, -1)
        cv2.imwrite(tmp.name, mask)
        tmp_path = tmp.name

    lines = mask_to_yolo_polygons(tmp_path, min_area_px=50)
    print(f"Polygons found: {len(lines)}")
    for l in lines:
        parts = l.split()
        print(f"  class={parts[0]}, points={len(parts[1:])//2}")
    assert len(lines) == 2, f"Expected 2 polygons, got {len(lines)}"
    os.unlink(tmp_path)
    print("mask_to_yolo_polygons: PASSED")
