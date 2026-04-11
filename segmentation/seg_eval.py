"""
Segmentation Inference & Evaluation -- Tabel 3.2 Proposal Tesis
YOLOv8n-seg | fine-tune 1-3 epoch | backbone frozen

Output:
  - mIoU per gambar per skenario
  - Overlay 3-panel: gambar + TP/FP/FN + peta segmen
  - Collage grid: SEG1 vs SEG4 vs SEG5 side-by-side per gambar
  - Excel ringkasan + bar chart + ZIP
"""

import os
import time
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Label pendek skenario -- didefinisikan di level modul
_SC_LABELS = {
    'SEG1_LR_Baseline':          'SEG1\n(LR Bicubic x4)',
    'SEG2_IRE_SR':               'SEG2\n(IRE SR)',
    'SEG3_SR2_UAV-IRE_no_NRDB': 'SEG3\n(no NRDB)',
    'SEG3_SR3_UAV-IRE_no_MBCM': 'SEG3\n(no MBCM)',
    'SEG3_SR4_UAV-IRE_no_EGA':  'SEG3\n(no EGA)',
    'SEG3_SR5_UAV-IRE_no_VSD':  'SEG3\n(no VSD)',
    'SEG4_UAV_IRE_Full':         'SEG4\n(UAV-IRE Full)',
    'SEG5_HR_UpperBound':        'SEG5\n(HR Asli)',
}


# =================================================================
# METRIK
# =================================================================

def compute_miou(pred_mask: np.ndarray,
                 gt_mask:   np.ndarray) -> Dict[str, float]:
    """
    mIoU untuk 2 kelas (background=0, weed=1).
    Kedua mask: H x W, nilai 0 atau 255.
    """
    pred = (pred_mask > 127).astype(np.uint8)
    gt   = (gt_mask   > 127).astype(np.uint8)
    if pred.shape != gt.shape:
        H, W = gt.shape
        pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)

    ious = []
    for cls in [0, 1]:
        p = (pred == cls)
        g = (gt   == cls)
        inter = int((p & g).sum())
        union = int((p | g).sum())
        ious.append(inter / union if union > 0 else 1.0)

    pw = (pred == 1)
    gw = (gt   == 1)
    TP = int((pw &  gw).sum())
    FP = int((pw & ~gw).sum())
    FN = int((~pw & gw).sum())
    prec  = TP / (TP + FP + 1e-8)
    rec   = TP / (TP + FN + 1e-8)
    f1    = 2 * prec * rec / (prec + rec + 1e-8)
    iou_w = TP / (TP + FP + FN + 1e-8)

    return {
        'mIoU':      round(float(np.mean(ious)), 4),
        'iou_weed':  round(iou_w, 4),
        'iou_bg':    round(ious[0], 4),
        'precision': round(prec, 4),
        'recall':    round(rec,  4),
        'F1':        round(f1,   4),
        'TP': TP, 'FP': FP, 'FN': FN,
    }


# =================================================================
# PREDIKSI
# =================================================================

def predict_weed_mask(model, img_path: str,
                      conf: float = 0.25, iou_thr: float = 0.45,
                      imgsz: int = 640, device: str = '') -> np.ndarray:
    """YOLOv8n-seg -> binary mask weed (class 0). Return H x W uint8."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        from PIL import Image
        pil = Image.open(img_path).convert('RGB')
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]
    out  = np.zeros((H, W), dtype=np.uint8)

    results = model.predict(source=img_path, conf=conf, iou=iou_thr,
                            imgsz=imgsz, device=device,
                            verbose=False, save=False)
    for r in results:
        if r.masks is None:
            continue
        for mask_t, cls_t in zip(r.masks.data, r.boxes.cls):
            if int(cls_t.item()) != 0:
                continue
            m = cv2.resize(mask_t.cpu().numpy(), (W, H),
                           interpolation=cv2.INTER_LINEAR)
            out = np.maximum(out, (m > 0.5).astype(np.uint8) * 255)
    return out


# =================================================================
# OVERLAY VISUAL PER GAMBAR
# =================================================================

def save_weed_overlay(img_path: str, pred_mask: np.ndarray,
                      gt_mask: np.ndarray, save_path: str,
                      metrics: Dict[str, float],
                      scenario: str = '', alpha: float = 0.50,
                      display_size: int = 1024) -> None:
    """
    Simpan overlay 3-panel:
      Panel 1: Gambar input (SR/LR/HR) — ditampilkan PENUH, di-resize ke display_size
      Panel 2: Overlay TP/FP/FN + kontur GT putih
      Panel 3: Peta segmen berwarna

    [REVISI] Gambar input sekarang di-resize ke display_size (default 1024px sisi terpanjang)
    sebelum ditampilkan, sehingga collage menampilkan gambar representatif yang besar
    (bukan patch 256×256). Semua mask disesuaikan ke ukuran yang sama.

    Konvensi warna:
      Kuning = TP   Merah = FP   Hijau = FN
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        from PIL import Image
        pil = Image.open(img_path).convert('RGB')
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    Ho, Wo = img_rgb_orig.shape[:2]

    # [REVISI] Resize ke display_size (jaga aspek rasio) untuk visualisasi
    scale_d = display_size / max(Ho, Wo)
    Hd, Wd = int(Ho * scale_d), int(Wo * scale_d)
    img_rgb = cv2.resize(img_rgb_orig, (Wd, Hd), interpolation=cv2.INTER_AREA)
    H, W = Hd, Wd

    def _b(mask):
        m = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        return m > 127

    pb = _b(pred_mask)
    gb = _b(gt_mask)
    tp = pb &  gb
    fp = pb & ~gb
    fn = ~pb & gb

    ov = img_rgb.astype(np.float32).copy()
    ov[tp] = ov[tp] * (1 - alpha) + np.array([255, 230,  0]) * alpha
    ov[fp] = ov[fp] * (1 - alpha) + np.array([230,  30, 30]) * alpha
    ov[fn] = ov[fn] * (1 - alpha) + np.array([ 30, 210, 60]) * alpha
    ov = ov.clip(0, 255).astype(np.uint8)
    cnts, _ = cv2.findContours((gb.astype(np.uint8) * 255),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (255, 255, 255), 2)

    seg = np.zeros((H, W, 3), dtype=np.uint8)
    seg[tp]  = [255, 230,  0]
    seg[fp]  = [230,  30, 30]
    seg[fn]  = [ 30, 210, 60]
    seg[~pb & ~gb] = [30, 30, 30]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor('white')
    title = (scenario + '  mIoU=' + str(metrics['mIoU'])
             + '  IoU_w=' + str(metrics['iou_weed'])
             + '  P=' + str(metrics['precision'])
             + '  R=' + str(metrics['recall']))
    fig.suptitle(title, fontsize=9, fontweight='bold', y=1.01)

    axes[0].imshow(img_rgb)
    axes[0].set_title('Input Image', fontsize=9, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(ov)
    axes[1].set_title('Overlay Gulma (kontur putih = GT)',
                      fontsize=9, fontweight='bold')
    axes[1].axis('off')
    axes[1].legend(handles=[
        mpatches.Patch(color=(1.0, 0.90, 0.0),
                       label='TP=' + str(metrics['TP']) + ' px'),
        mpatches.Patch(color=(0.90, 0.12, 0.12),
                       label='FP=' + str(metrics['FP']) + ' px'),
        mpatches.Patch(color=(0.12, 0.82, 0.24),
                       label='FN=' + str(metrics['FN']) + ' px'),
    ], loc='lower right', fontsize=7, framealpha=0.85, edgecolor='gray')

    axes[2].imshow(seg)
    axes[2].set_title('Peta TP/FP/FN', fontsize=9, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)


# =================================================================
# COLLAGE GRID: SEG1 vs SEG4 vs SEG5 per gambar
# =================================================================

def make_comparison_collage(image_stems: List[str],
                            scenario_dirs: Dict[str, str],
                            collage_dir: str,
                            scenarios_show: Optional[List[str]] = None,
                            max_collages: int = 10) -> List[str]:
    """
    Buat collage grid yang membandingkan panel overlay antar skenario
    untuk gambar yang SAMA -- output visual utama untuk laporan tesis.

    Setiap kolom = panel tengah dari save_weed_overlay (overlay TP/FP/FN).
    Border biru tebal = SEG4 (model yang dievaluasi).

    Args:
        image_stems:    List nama file stem dari test set
        scenario_dirs:  {nama_skenario: folder output skenario}
        collage_dir:    Folder output collage
        scenarios_show: Skenario yang ditampilkan (default: SEG1, SEG4, SEG5)
        max_collages:   Jumlah maksimum collage

    Returns:
        List path file collage yang dihasilkan
    """
    os.makedirs(collage_dir, exist_ok=True)

    if scenarios_show is None:
        scenarios_show = [
            'SEG1_LR_Baseline',
            'SEG4_UAV_IRE_Full',
            'SEG5_HR_UpperBound',
        ]

    avail = [s for s in scenarios_show if s in scenario_dirs]
    if len(avail) < 2:
        print('[WARN] Skenario tidak cukup untuk collage')
        return []

    n_col = len(avail)
    paths = []

    for stem in image_stems[:max_collages]:
        # Kumpulkan file overlay yang tersedia
        overlays = {
            sc: os.path.join(scenario_dirs[sc], 'visualizations',
                             stem + '_overlay.png')
            for sc in avail
        }
        overlays = {sc: p for sc, p in overlays.items()
                    if os.path.isfile(p)}

        if len(overlays) < 2:
            continue

        fig, axes = plt.subplots(1, n_col,
                                  figsize=(6.5 * n_col, 6.5),
                                  facecolor='white')
        if n_col == 1:
            axes = [axes]

        for col_i, sc in enumerate(avail):
            ax = axes[col_i]
            ov_p = overlays.get(sc)

            if ov_p:
                full = cv2.imread(ov_p)
                if full is not None:
                    rgb = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)
                    # [REVISI] Ambil panel INPUT (panel kiri = gambar SR/LR/HR asli),
                    # bukan panel tengah (overlay) yang hanya 1/3 dari gambar composite.
                    # Panel overlay_3panel: [Input | TP/FP/FN overlay | Peta segmen]
                    # Kita tampilkan panel 0 (input gambar penuh) agar detail maksimal.
                    _H, _W = rgb.shape[:2]
                    pw = _W // 3
                    input_panel = rgb[:, :pw, :]          # panel kiri = gambar input
                    overlay_panel = rgb[:, pw:2 * pw, :]  # panel tengah = overlay TP/FP/FN

                    # Buat sub-figure: atas=gambar SR penuh, bawah=overlay TP/FP/FN
                    # dengan pembagian 60/40
                    from matplotlib.gridspec import GridSpecFromSubplotSpec
                    inner_gs = GridSpecFromSubplotSpec(
                        2, 1, subplot_spec=axes[col_i].get_subplotspec(),
                        hspace=0.05, height_ratios=[3, 2])
                    ax.remove()  # hapus axis lama
                    ax_top = fig.add_subplot(inner_gs[0])
                    ax_bot = fig.add_subplot(inner_gs[1])

                    ax_top.imshow(input_panel)
                    ax_top.set_title(_SC_LABELS.get(sc, sc),
                                     fontsize=11, fontweight='bold', pad=6)
                    ax_top.axis('off')
                    ax_bot.imshow(overlay_panel)
                    ax_bot.set_title('Overlay TP/FP/FN', fontsize=8, pad=2)
                    ax_bot.axis('off')

                    if 'SEG4' in sc or 'Full' in sc:
                        for spine in ax_top.spines.values():
                            spine.set_visible(True)
                            spine.set_edgecolor('#1565C0')
                            spine.set_linewidth(3)
                        for spine in ax_bot.spines.values():
                            spine.set_visible(True)
                            spine.set_edgecolor('#1565C0')
                            spine.set_linewidth(3)
                    continue  # lewati set_title di bawah (sudah di-handle)

                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            transform=ax.transAxes, fontsize=12)
            else:
                ax.text(0.5, 0.5, sc + '\n(tidak tersedia)',
                        ha='center', va='center',
                        transform=ax.transAxes, fontsize=9)

            ax.set_title(_SC_LABELS.get(sc, sc),
                         fontsize=11, fontweight='bold', pad=6)
            ax.axis('off')

            # Border biru untuk SEG4
            if 'SEG4' in sc or 'Full' in sc:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor('#1565C0')
                    spine.set_linewidth(3)

        fig.suptitle(
            'Perbandingan Segmentasi Gulma - ' + stem + '\n'
            'Kuning=TP  Merah=FP  Hijau=FN  Kontur_Putih=GT',
            fontsize=10, fontweight='bold', y=1.01)
        plt.tight_layout(pad=0.8)

        out_p = os.path.join(collage_dir, 'collage_' + stem + '.png')
        plt.savefig(out_p, dpi=130, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        paths.append(out_p)

    print('Collage: ' + str(len(paths)) + ' gambar -> ' + collage_dir)
    return paths


# =================================================================
# EVALUASI SATU SKENARIO
# =================================================================

def run_scenario(scenario_name: str, image_dir: str, gt_mask_dir: str,
                 yolo_model_path: str, output_dir: str,
                 file_list: Optional[str] = None,
                 conf: float = 0.25, imgsz: int = 640,
                 device: str = '', save_visuals: bool = True,
                 max_images: Optional[int] = None,
                 ) -> Tuple[List[dict], dict]:
    """
    Evaluasi segmentasi untuk satu skenario.
    GT mask diambil dari Masks/ WeedyRice (ground truth resmi).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError('pip install ultralytics')

    vis_dir    = os.path.join(output_dir, 'visualizations')
    mask_cache = os.path.join(output_dir, '_pred_masks')
    if save_visuals:
        os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(mask_cache, exist_ok=True)

    print('\n' + '=' * 60)
    print('  ' + scenario_name)
    print('  ' + image_dir)
    print('-' * 60)

    model = YOLO(yolo_model_path)

    # Kumpulkan gambar
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    if file_list and os.path.isfile(file_list):
        with open(file_list) as f:
            names = [ln.strip() for ln in f if ln.strip()]
        img_pairs = []
        for name in names:
            stem = Path(name).stem
            for cand in [
                os.path.join(image_dir, name),
                os.path.join(image_dir, 'SR_' + stem + '.png'),
                os.path.join(image_dir, stem + '.png'),
                os.path.join(image_dir, stem + '.JPG'),
            ]:
                if os.path.isfile(cand):
                    img_pairs.append((cand, stem))
                    break
    else:
        img_pairs = [(str(p), p.stem)
                     for p in sorted(Path(image_dir).iterdir())
                     if p.suffix.lower() in exts]

    if max_images:
        img_pairs = img_pairs[:max_images]

    if not img_pairs:
        print('  [WARN] Tidak ada gambar di: ' + image_dir)
        return [], {}

    print('  Gambar: ' + str(len(img_pairs)))
    print('  {:>4}  {:<30}  {:>6}  {:>6}  {:>6}  {:>6}'.format(
          '#', 'File', 'mIoU', 'IoU_w', 'P', 'R'))
    print('  ' + '-' * 58)

    records = []
    for idx, (img_path, stem) in enumerate(img_pairs, 1):
        clean = stem[3:] if stem.startswith('SR_') else stem
        gt_p  = os.path.join(gt_mask_dir, clean + '.png')
        if os.path.isfile(gt_p):
            gt_mask = cv2.imread(gt_p, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                gt_mask = np.zeros((640, 640), dtype=np.uint8)
        else:
            tmp = cv2.imread(img_path)
            h, w = (tmp.shape[:2] if tmp is not None else (640, 640))
            gt_mask = np.zeros((h, w), dtype=np.uint8)

        t0        = time.time()
        pred_mask = predict_weed_mask(model, img_path, conf, 0.45, imgsz, device)
        ms        = (time.time() - t0) * 1000

        m = compute_miou(pred_mask, gt_mask)
        print('  {:>4}  {:<30}  {:>6.4f}  {:>6.4f}  {:>6.4f}  {:>6.4f}'.format(
              idx, Path(img_path).name[:30],
              m['mIoU'], m['iou_weed'], m['precision'], m['recall']))

        if save_visuals:
            vis_p = os.path.join(vis_dir, clean + '_overlay.png')
            try:
                save_weed_overlay(img_path, pred_mask, gt_mask,
                                  vis_p, m, scenario_name)
            except Exception as e:
                print('    [WARN] Visual gagal: ' + str(e))

        # Cache pred_mask untuk collage
        cv2.imwrite(os.path.join(mask_cache, clean + '.png'), pred_mask)

        records.append({
            'index': idx, 'scenario': scenario_name,
            'filename':  Path(img_path).name,
            'mIoU':      m['mIoU'],
            'iou_weed':  m['iou_weed'],
            'iou_bg':    m['iou_bg'],
            'precision': m['precision'],
            'recall':    m['recall'],
            'F1':        m['F1'],
            'TP':        m['TP'],
            'FP':        m['FP'],
            'FN':        m['FN'],
            'time_ms':   round(ms, 1),
        })

    keys = ('mIoU', 'iou_weed', 'precision', 'recall', 'F1')
    avg  = {k: round(float(np.mean([r[k] for r in records])), 4)
            for k in keys}
    print('  ' + '-' * 58)
    print('  {:<32}  {:>6.4f}  {:>6.4f}  {:>6.4f}  {:>6.4f}'.format(
          'AVG', avg['mIoU'], avg['iou_weed'],
          avg['precision'], avg['recall']))
    return records, avg


# =================================================================
# RUNNER SEMUA SKENARIO SEG1-SEG5
# =================================================================

def run_all_scenarios(dataset_root: str, sr_results_base: str,
                      yolo_model_path: str, output_dir: str,
                      test_list: str, device: str = '',
                      save_visuals: bool = True,
                      max_per_scenario: Optional[int] = None,
                      ) -> Dict[str, dict]:
    """
    Jalankan semua skenario Tabel 3.2 secara berurutan.
    GT mask untuk mIoU diambil dari Masks/ WeedyRice (sama dengan
    yang dipakai sebagai weed_mask di VSD loss saat training SR).
    """
    os.makedirs(output_dir, exist_ok=True)
    rgb_dir  = os.path.join(dataset_root, 'RGB')
    mask_dir = os.path.join(dataset_root, 'Masks')

    all_records:   List[dict]      = []
    all_summaries: Dict[str, dict] = {}

    def _infer_dir(exp: str) -> str:
        for c in [
            os.path.join(sr_results_base, exp, exp + '_inference'),
            os.path.join(sr_results_base, exp, 'inference'),
            os.path.join(sr_results_base, exp),
        ]:
            if os.path.isdir(c):
                return c
        return os.path.join(sr_results_base, exp, exp + '_inference')

    def _run(name: str, img_dir: str):
        if not os.path.isdir(img_dir):
            print('\n[SKIP] ' + name)
            return
        recs, avg = run_scenario(
            scenario_name=name, image_dir=img_dir,
            gt_mask_dir=mask_dir, yolo_model_path=yolo_model_path,
            output_dir=os.path.join(output_dir, name),
            file_list=test_list, device=device,
            save_visuals=save_visuals, max_images=max_per_scenario)
        all_records.extend(recs)
        all_summaries[name] = avg

    lr_dir = os.path.join(output_dir, '_temp_LR')
    _make_lr_images(rgb_dir, lr_dir, test_list, scale=4)
    _run('SEG1_LR_Baseline', lr_dir)
    _run('SEG2_IRE_SR',      _infer_dir('SR1_IRE_Baseline'))

    for exp in ['SR2_UAV-IRE_no_NRDB', 'SR3_UAV-IRE_no_MBCM',
                'SR4_UAV-IRE_no_EGA',  'SR5_UAV-IRE_no_VSD']:
        _run('SEG3_' + exp, _infer_dir(exp))

    _run('SEG4_UAV_IRE_Full',   _infer_dir('SR6_UAV-IRE_Full'))
    _run('SEG5_HR_UpperBound',  rgb_dir)

    # Collage grid SEG1 vs SEG4 vs SEG5
    if save_visuals and all_summaries:
        with open(test_list) as f:
            stems = [Path(ln.strip()).stem for ln in f if ln.strip()]
        sc_dirs = {n: os.path.join(output_dir, n) for n in all_summaries}
        make_comparison_collage(
            image_stems=stems,
            scenario_dirs=sc_dirs,
            collage_dir=os.path.join(output_dir, 'collage_comparison'),
            scenarios_show=['SEG1_LR_Baseline',
                            'SEG4_UAV_IRE_Full',
                            'SEG5_HR_UpperBound'],
            max_collages=10,
        )

    _export_excel(all_records, all_summaries, output_dir)
    _plot_bar_chart(all_summaries, output_dir)
    _make_zip(output_dir)
    return all_summaries


# =================================================================
# HELPERS
# =================================================================

def _make_lr_images(rgb_dir: str, lr_dir: str,
                    test_list: str, scale: int = 4):
    """
    [REVISI] Buat LR image untuk SEG1 dengan 2 langkah:
      1. Downscale HR → LR (÷scale) via bicubic
      2. Upsample LR → ukuran HR kembali via bicubic (bicubic ×scale)

    Hasilnya adalah gambar PENUH berukuran sama dengan HR dan SR,
    sehingga evaluasi segmentasi (mIoU) dilakukan secara fair pada
    resolusi yang identik di semua skenario SEG1-SEG5.

    Sebelumnya: output hanya 1/4 ukuran → gambar kecil di collage.
    """
    os.makedirs(lr_dir, exist_ok=True)
    with open(test_list) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    for name in names:
        src = os.path.join(rgb_dir, name)
        dst = os.path.join(lr_dir, name)
        if os.path.isfile(dst) or not os.path.isfile(src):
            continue
        img = cv2.imread(src)
        if img is None:
            continue
        H, W = img.shape[:2]
        # Step 1: Downscale ke LR (simulasi resolusi rendah UAV)
        lr = cv2.resize(img, (W // scale, H // scale),
                        interpolation=cv2.INTER_CUBIC)
        # Step 2: [REVISI] Upsample kembali ke ukuran HR (bicubic upscale)
        # Ini adalah representasi SEG1 yang benar: LR bicubic ×4
        lr_upscaled = cv2.resize(lr, (W, H), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(dst, lr_upscaled)


def _export_excel(records: List[dict], summaries: Dict[str, dict],
                  output_dir: str) -> str:
    try:
        import pandas as pd
    except ImportError:
        return ''
    path   = os.path.join(output_dir, 'segmentation_results.xlsx')
    df_all = pd.DataFrame(records)
    df_sum = pd.DataFrame([{'Skenario': k, **v}
                            for k, v in summaries.items() if v])
    with pd.ExcelWriter(path, engine='openpyxl') as w:
        df_sum.to_excel(w, sheet_name='Summary SEG1-SEG5', index=False)
        df_all.to_excel(w, sheet_name='Per-Image Metrics',  index=False)
        for sc in (df_all['scenario'].unique()
                   if not df_all.empty else []):
            df_all[df_all['scenario'] == sc].to_excel(
                w, sheet_name=sc[:28], index=False)
    print('Excel: ' + path)
    return path


def _plot_bar_chart(summaries: Dict[str, dict], output_dir: str) -> str:
    valid = {k: v for k, v in summaries.items() if v}
    if not valid:
        return ''
    names  = list(valid.keys())
    labels = [_SC_LABELS.get(n, n[:12]) for n in names]
    miou   = [valid[n].get('mIoU',     0) for n in names]
    iou_w  = [valid[n].get('iou_weed', 0) for n in names]
    f1     = [valid[n].get('F1',       0) for n in names]
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.7), 6))
    for bars, vals in [
        (ax.bar(x - w, miou,  w, color='#1565C0', alpha=0.88, label='mIoU'),     miou),
        (ax.bar(x,     iou_w, w, color='#2E7D32', alpha=0.88, label='IoU Weed'), iou_w),
        (ax.bar(x + w, f1,    w, color='#E65100', alpha=0.88, label='F1'),       f1),
    ]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.005,
                    '{:.3f}'.format(val),
                    ha='center', va='bottom', fontsize=7.5)
    if 'SEG5_HR_UpperBound' in valid:
        ub = valid['SEG5_HR_UpperBound'].get('mIoU', 0)
        ax.axhline(ub, color='gray', ls='--', lw=1.2,
                   label='HR upper bound mIoU=' + '{:.3f}'.format(ub))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Skor', fontsize=11)
    ax.set_xlabel('Skenario Evaluasi Segmentasi', fontsize=11)
    ax.set_title('Pengaruh SR UAV-IRE terhadap Akurasi Segmentasi Gulma\n'
                 'YOLOv8n-seg | WeedyRice-RGBMS-DB | Tabel 3.2',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    path = os.path.join(output_dir, 'segmentation_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Bar chart: ' + path)
    return path


def _make_zip(output_dir: str):
    zip_p = os.path.join(output_dir, 'segmentation_results.zip')
    with zipfile.ZipFile(zip_p, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in Path(output_dir).rglob('*'):
            if (not p.is_file() or str(p) == zip_p
                    or p.suffix in {'.pt', '.pth'}
                    or '_temp_LR' in p.parts):
                continue
            zf.write(p, p.relative_to(output_dir))
    print('ZIP: ' + zip_p + ' ({:.1f} MB)'.format(
          os.path.getsize(zip_p) / 1e6))