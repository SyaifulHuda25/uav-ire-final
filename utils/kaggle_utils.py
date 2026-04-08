"""
Kaggle Output Utilities
Membantu memisahkan file output di Kaggle:
- Model (.pth) → download terpisah jika perlu inferensi lokal
- Hasil training (loss, grafik, Excel) → ZIP kecil untuk laporan
- Hasil inferensi (visual, tabel) → ZIP kecil untuk analisis

Cara pakai (di Kaggle notebook):
    from utils.kaggle_utils import package_outputs
    package_outputs(exp_dir, output_base)
"""

import os
import zipfile
import json
import shutil
from pathlib import Path
from typing import Optional, List


def package_outputs(
    exp_dir: str,
    output_base: str,
    exp_name: str = '',
    include_checkpoints: bool = False,
) -> dict:
    """
    Buat tiga paket output terstruktur dari exp_dir:
    1. ZIP hasil training (loss, grafik, log, Excel) — TANPA .pth
    2. Model saja (.pth) — opsional, untuk download terpisah
    3. Manifest JSON (daftar file + ukuran)

    Returns:
        dict berisi path masing-masing paket
    """
    os.makedirs(output_base, exist_ok=True)
    name = exp_name or Path(exp_dir).name

    # ── 1. ZIP hasil training (tanpa model) ───────────────
    results_zip = os.path.join(output_base, f'{name}_training_results.zip')
    training_files = []
    skip_ext  = {'.pth'} if not include_checkpoints else set()
    skip_names = {'checkpoint_latest.pth'}

    with zipfile.ZipFile(results_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(Path(exp_dir).rglob('*')):
            if not p.is_file():
                continue
            if p.suffix in skip_ext:
                continue
            if p.name in skip_names:
                continue
            if 'checkpoint_epoch' in p.name and not include_checkpoints:
                continue
            rel = str(p.relative_to(exp_dir))
            zf.write(p, rel)
            training_files.append({'path': rel, 'size_kb': p.stat().st_size // 1024})

    results_zip_mb = os.path.getsize(results_zip) / 1e6

    # ── 2. Identifikasi file model ─────────────────────────
    model_path = os.path.join(exp_dir, 'generator_final.pth')
    model_mb = os.path.getsize(model_path) / 1e6 if os.path.isfile(model_path) else 0

    # ── 3. Manifest ────────────────────────────────────────
    manifest = {
        'experiment': name,
        'files': {
            'training_results_zip': {
                'path': results_zip,
                'size_mb': round(results_zip_mb, 2),
                'description': 'Hasil training: loss Excel, grafik, log (TANPA model)',
                'contents': training_files,
            },
            'model_pth': {
                'path': model_path,
                'size_mb': round(model_mb, 2),
                'description': 'Generator final — hanya download jika perlu inferensi lokal',
            },
        },
    }

    manifest_path = os.path.join(output_base, f'{name}_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # ── Print summary ──────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  OUTPUT PACKAGES — {name}")
    print(f"{'═'*55}")
    print(f"  [ZIP KECIL] Training results:")
    print(f"    {Path(results_zip).name}  ({results_zip_mb:.1f} MB)")
    print(f"    Isi: Excel loss, grafik, log, visualisasi training")
    print()
    print(f"  [MODEL PTH] Generator:")
    print(f"    generator_final.pth  ({model_mb:.1f} MB)")
    print(f"    Download hanya jika perlu inferensi lokal")
    print()
    print(f"  Cara download di Kaggle:")
    print(f"    → Tab 'Output' → klik file → Download")
    print(f"{'═'*55}\n")

    return manifest


def merge_ablation_results(
    experiments: List[str],
    exp_base_dir: str,
    output_path: str,
) -> str:
    """
    Gabungkan hasil semua eksperimen ablation (SR1–SR6) ke satu Excel.
    Berguna setelah semua 6 skenario selesai dijalankan.

    Args:
        experiments: list nama folder eksperimen
        exp_base_dir: direktori dasar yang berisi semua folder eksperimen
        output_path: path output Excel gabungan
    """
    import pandas as pd

    all_epoch_dfs = []
    summary_rows  = []

    for exp in experiments:
        history_path = os.path.join(exp_base_dir, exp, 'history.json')
        if not os.path.isfile(history_path):
            print(f"  [Skip] {exp}: history.json tidak ditemukan")
            continue

        with open(history_path) as f:
            data = json.load(f)

        epochs     = data['epochs']
        train_dict = data.get('train', {})
        val_dict   = data.get('val', {})

        # Epoch summary
        row = {'experiment': exp}
        for k, v in train_dict.items():
            row[f'train_{k}'] = v[-1] if v else None   # nilai epoch terakhir
        for k, v in val_dict.items():
            row[f'val_{k}'] = v[-1] if v else None
        summary_rows.append(row)

        # Per-epoch data
        df = pd.DataFrame({'epoch': epochs})
        for k, v in train_dict.items():
            df[f'train_{k}'] = v[:len(epochs)]
        for k, v in val_dict.items():
            # val mungkin lebih jarang → pad dengan NaN
            v_padded = [None] * len(epochs)
            for i, ep in enumerate(epochs):
                val_idx = (ep - 1) // max(1, len(epochs) // max(len(v), 1))
                if val_idx < len(v):
                    v_padded[i] = v[val_idx]
            df[f'val_{k}'] = v_padded
        df['experiment'] = exp
        all_epoch_dfs.append(df)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Ringkasan semua eksperimen (nilai akhir)
        pd.DataFrame(summary_rows).to_excel(
            writer, sheet_name='Summary All Experiments', index=False)

        # Sheet per eksperimen
        for df in all_epoch_dfs:
            exp = df['experiment'].iloc[0]
            sheet = exp[:30]  # Excel max 31 chars
            df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"Merged ablation results saved: {output_path}")
    return output_path


def print_kaggle_instructions(exp_name: str, output_base: str):
    """Print panduan lengkap cara download dan menggunakan output di Kaggle."""
    print(f"""
╔══════════════════════════════════════════════════════════╗
║         PANDUAN DOWNLOAD OUTPUT KAGGLE                    ║
╠══════════════════════════════════════════════════════════╣
║  Eksperimen: {exp_name:<43} ║
╠══════════════════════════════════════════════════════════╣
║                                                           ║
║  Di tab OUTPUT Kaggle (kanan atas notebook):              ║
║                                                           ║
║  1. HASIL TRAINING (download ini dulu — kecil):           ║
║     {exp_name}_training_results.zip                ║
║     Isi: Excel loss, grafik train/val, log, visual        ║
║                                                           ║
║  2. HASIL INFERENSI (opsional — visual + metrik):         ║
║     {exp_name}_inference_results.zip               ║
║     Isi: SR images, 3-panel visual, Excel PSNR/SSIM       ║
║                                                           ║
║  3. MODEL (hanya jika perlu inferensi lokal):             ║
║     {exp_name}/generator_final.pth (~200 MB)       ║
║                                                           ║
║  INFERENSI LOKAL setelah download model:                  ║
║  python inference.py                                      ║
║    --model generator_final.pth                            ║
║    --test_hr_dir data/test                                ║
║    --output_dir  results/                                 ║
║    --experiment  {exp_name:<35}║
║                                                           ║
╚══════════════════════════════════════════════════════════╝
""")
