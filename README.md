# UAV-IRE: Super-Resolution untuk Segmentasi Gulma di Lahan Persawahan

**Tesis Magister | Departemen Matematika | ITS Surabaya 2025**  
Muhamad Syaiful Huda | NRP 6002241012 | Pembimbing: Dr. Budi Setiyono, S.Si, MT

---

## Struktur Folder

```
uav-ire/
│
├── models/                      # Arsitektur model neural network
│   ├── network.py               # IRE baseline (RRDB, ChannelAttention, PatchGAN)
│   ├── nrdb.py                  # NRDB — denoising noise sensor UAV
│   ├── mbcm.py                  # MBCM — kompensasi motion blur
│   ├── ega.py                   # EGA  — perhatian berbasis tepi
│   ├── vsd.py                   # VSD  — discriminator vegetasi
│   └── uav_ire_generator.py     # Generator UAV-IRE (semua modul terintegrasi)
│
├── data/                        # Dataset dan pipeline degradasi
│   ├── weedyrice_dataset.py     # Dataset WeedyRice-RGBMS-DB (RGB + Mask)
│   ├── uav_degradation.py       # Degradasi khas UAV (blur, noise, exposure, rotasi)
│   ├── degradation.py           # Degradasi IRE second-order (baseline)
│   └── dataset.py               # Dataset generik (fallback)
│
├── losses/
│   ├── uav_ire_losses.py        # Total loss UAV-IRE: Lrec+Lperc+Ladv+Ledge+LVSD
│   └── losses.py                # Losses IRE baseline
│
├── training/
│   └── epoch_trainer.py         # Trainer utama: per-epoch, auto-resume, Excel export
│
├── segmentation/                # Pipeline evaluasi segmentasi (Tabel 3.2)
│   ├── yolo_dataset_builder.py  # Konversi mask PNG → YOLO polygon label
│   ├── yolo_finetune.py         # Fine-tune YOLOv8n-seg (3 epoch, backbone frozen)
│   └── seg_eval.py              # Evaluasi SEG1–SEG5 + collage + Excel + ZIP
│
├── utils/
│   ├── metrics.py               # PSNR, SSIM, NIQE
│   ├── visualization.py         # Plot loss, comparison grid
│   └── kaggle_utils.py          # ZIP packaging, merge ablation
│
├── train_uav_ire.py             # Entry point training (lokal/Kaggle CLI)
├── inference.py                 # Inferensi SR + tile support + PSNR/SSIM
├── run_segmentation.py          # Evaluasi segmentasi SEG1–SEG5
├── verify_uav_ire.py            # Verifikasi implementasi vs proposal tesis
├── kaggle_notebook.ipynb        # Notebook Kaggle siap pakai (26 sel)
└── requirements.txt
```

---

## Referensi Cepat

| Tujuan | Perintah |
|---|---|
| Verifikasi implementasi | `python verify_uav_ire.py` |
| Training (lokal) | `python train_uav_ire.py --experiment full` |
| Inferensi SR | `python inference.py --model gen.pth --test_hr_dir RGB/ --output_dir hasil/` |
| Evaluasi segmentasi | `python run_segmentation.py` |
