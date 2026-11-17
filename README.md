# Valencia Flood Mapping with Sentinel-1 SAR

Mapping the flooding caused by the **Valencia DANA storm (29 October 2024)** using Sentinel-1 radar imagery.
The project compares classical, training-free SAR image processing with a small deep-learning baseline,
and checks every result against an independent optical reference from Sentinel-2.

The full write-up is in [remote_sensor_Report.pdf](remote_sensor_Report.pdf).

---

## Overview

Radar sees through clouds, which makes it the main tool for mapping floods while a storm is still over the area.
Calm open water reflects the radar signal away from the sensor, so flooded land appears **dark** in SAR backscatter.
This project uses that effect to build flood maps.

**Study area:** Valencia, Spain (Albufera and surrounding rice fields), UTM zone 31N, 10 m pixels
**SAR data:** Sentinel-1 IW GRD, pre-event 20 Oct 2024 and post-event 1 Nov 2024 (VV + VH)
**Optical reference:** Sentinel-2 L2A, 26 Oct 2024 (pre) and 31 Oct 2024 (post)
**Permanent water mask:** JRC Global Surface Water (occurrence ≥ 50 %)

## Methods

### Pre-processing (SNAP)
The Copernicus GRD products were processed in ESA SNAP: apply orbit file, thermal noise removal, slice assembly,
σ⁰ calibration, coregistration, terrain correction and subset. The result is
`valencia_flood_coregisteration.tif`, with 4 bands (linear σ⁰):

| Band | Content |
|------|---------|
| 1 | VH, post-event (01/11/2024) |
| 2 | VV, post-event (01/11/2024) |
| 3 | VH, pre-event (20/10/2024) |
| 4 | VV, pre-event (20/10/2024) |

### Classical flood mapping (no training)
1. **Speckle reduction:** Lee filter (7×7, ENL ≈ 4.4).
2. **Method A, single date:** Otsu threshold on post-event VV backscatter (dB).
3. **Method B, change detection:** log-ratio `LR = 10·log10(VV_post / VV_pre)`, thresholded with
   **Kittler–Illingworth** (minimum error), combined with a "dark after the event" condition.
4. **Method C, spatial regularisation:** Method B refined with a **Markov Random Field** (Potts prior, ICM optimisation).
5. **Ablation:** Method B without the Lee filter, plus all methods repeated on the VH channel.

Permanent water (JRC GSW) is removed from every map.

### Deep-learning baseline
A small U-Net (VV + VH input, trained from scratch, BCE loss) is trained on the hand-labelled split of
[Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) (Bonafilia et al., 2020) and then applied to the Valencia scene.

### Validation
Flood maps are scored against water masks derived from Sentinel-2 **MNDWI** (Otsu threshold, cloud-free pixels only):

- **R1, post-event water:** water on 31/10 outside permanent water
- **R2, new water:** R1 minus pixels that were already water on 26/10

Metrics: precision, recall, F1, IoU, overall accuracy and Cohen's kappa.

## Results

### Valencia: classical methods, VV polarisation

| Method | Area (km²) | F1 vs R1 | IoU vs R1 | F1 vs R2 | IoU vs R2 |
|--------|-----------:|---------:|----------:|---------:|----------:|
| A: single-date Otsu | 58.00 | **0.915** | **0.843** | 0.623 | 0.452 |
| B: log-ratio + KI | 46.46 | 0.829 | 0.708 | 0.655 | 0.487 |
| C: log-ratio + KI + MRF | 44.64 | 0.810 | 0.680 | **0.661** | **0.493** |
| B without Lee filter | 49.48 | 0.819 | 0.693 | 0.637 | 0.468 |

Thresholds found: Otsu on post VV = −14.43 dB, Kittler–Illingworth on log-ratio = −2.95 dB.

- The single-date method is best at finding **all post-event water**.
- Change detection with the MRF is best at isolating **newly flooded areas**.
- The Lee filter improves precision (0.942 → 0.974 vs R1).
- VV gives slightly better results than VH for every method.

### Sen1Floods11 test split

| Method | IoU | F1 |
|--------|----:|---:|
| Otsu per chip (VV) | 0.210 | 0.346 |
| Global Otsu learnt on train (VV < −12.5 dB) | 0.330 | 0.497 |
| **Vanilla U-Net (VV + VH)** | **0.650** | **0.788** |

### Valencia: U-Net vs Otsu (against R1)

| Method | Precision | Recall | F1 | IoU | Kappa |
|--------|----------:|-------:|---:|----:|------:|
| Method A (Otsu) | 0.965 | 0.870 | 0.915 | 0.843 | 0.884 |
| **U-Net (Sen1Floods11)** | 0.910 | 0.954 | **0.931** | **0.872** | **0.904** |

The U-Net, trained on floods elsewhere in the world, transfers well to Valencia without retraining.

Full numbers are in [results.json](results.json), [results_sen1floods11.json](results_sen1floods11.json) and [results_dl.json](results_dl.json).

## Repository structure

```
.
├── valencia_flood_coregisteration.tif   # SNAP-processed Sentinel-1 stack (input)
├── remote_sensor_Report.pdf             # project report
├── requirements.txt
├── results.json                         # classical methods (VV and VH)
├── results_sen1floods11.json            # Sen1Floods11 benchmark
├── results_dl.json                      # U-Net applied to Valencia
├── data/
│   └── unet_sen1floods11.pt             # trained U-Net weights
└── src/
    ├── download_references.py           # Sentinel-2 + JRC GSW, resampled onto the SNAP grid
    ├── flood_mapping.py                 # classical pipeline, validation and figures
    ├── download_sen1floods11.py         # Sen1Floods11 hand-labelled data (~680 MB)
    ├── train_unet_sen1floods11.py       # U-Net training and benchmark
    └── unet_valencia.py                 # apply the trained U-Net to Valencia
```

Folders `data/` (reference layers, flood maps) and `figures/` are filled in when the scripts run.

## Getting started

### Requirements
Python 3.11+ and the packages in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No account is needed: Sentinel-2 and JRC GSW data come from the
[Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) STAC API.

### Run the pipeline

Run the scripts from the repository root, in this order:

```bash
# 1. Download the validation layers (Sentinel-2, JRC GSW) onto the SNAP grid
python src/download_references.py

# 2. Classical flood mapping: writes results.json, data/flood_maps.npz and figures/
python src/flood_mapping.py

# 3. Apply the provided U-Net to Valencia: writes results_dl.json
python src/unet_valencia.py
```

Optional, to reproduce the deep-learning benchmark:

```bash
# 4. Download Sen1Floods11 (hand-labelled split, ~680 MB)
python src/download_sen1floods11.py

# 5. Train the U-Net (40 epochs, ~8 min on an Apple MPS GPU)
#    Note: this overwrites data/unet_sen1floods11.pt
python src/train_unet_sen1floods11.py
```

The scripts use an Apple MPS or CUDA GPU when available and fall back to the CPU otherwise.

### Output figures

| File | Content |
|------|---------|
| `figures/overview.png` | Pre/post VV, log-ratio, Sentinel-2 view, final flood map and error map |
| `figures/histograms.png` | Histograms with the Otsu, Kittler–Illingworth and MNDWI thresholds |
| `figures/zoom_filtering_mrf.png` | Close-up showing the effect of the Lee filter and the MRF |
| `figures/sen1floods11_examples.png` | Sen1Floods11 test chips: label vs Otsu vs U-Net |
| `figures/valencia_unet_vs_otsu.png` | Valencia: Otsu vs U-Net water maps |

## Data sources and references

- **Sentinel-1 and Sentinel-2:** Copernicus Programme, European Space Agency
- **JRC Global Surface Water:** Pekel, J.-F. et al. (2016). *High-resolution mapping of global surface water and its long-term changes.* Nature, 540, 418–422.
- **Sen1Floods11:** Bonafilia, D. et al. (2020). *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.* CVPR Workshops.
- **Lee filter:** Lee, J.-S. (1980). *Digital image enhancement and noise filtering by use of local statistics.* IEEE TPAMI.
- **Otsu thresholding:** Otsu, N. (1979). *A threshold selection method from gray-level histograms.* IEEE TSMC.
- **Kittler–Illingworth thresholding:** Kittler, J. & Illingworth, J. (1986). *Minimum error thresholding.* Pattern Recognition, 19(1).
- **ICM / MRF:** Besag, J. (1986). *On the statistical analysis of dirty pictures.* JRSS B, 48(3).
- **U-Net:** Ronneberger, O. et al. (2015). *U-Net: Convolutional networks for biomedical image segmentation.* MICCAI.
