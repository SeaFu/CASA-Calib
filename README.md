# CASA-Calib (MATLAB Release)
CASA-Calib: A Context-Aware Semantic Alignment Method for LiDAR-Camera Extrinsic Calibration for Vehicle Perception Systems
Author: Yuan-Ting Fu

This repository provides the official MATLAB implementation of the core components used in the CASA-Calib paper, including:

CASA-Loss (full loss formulation used during calibration)

Cost landscape visualization (Fig. 7)

Tau sensitivity analysis (Fig. 4)

Perturbation robustness experiments (Fig. 5)

All scripts are self-contained and assume you have already prepared the curated Waymo dataset described in the paper.

```text
CASA_Calib/
│
├── CASA_Loss.m                    # Core CASA-Loss (Section III of paper)
│
├── cost_landscape.m               # J(Δty, Δtz) landscape → Fig. 7
│
├── Tau_Sensitivity_Analysis.m     # Tau sweep & stability band → Fig. 4
│
├── perturbation_analysis.m        # Perturbation robustness → Fig. 5
│
├── img_contour_seq_fast.m         # Contour sequencing (used by CASA-Loss)
├── LiDAR_contour_extraction_opt.m # LiDAR contour extraction
├── loss_proj.m                    # Local SDS similarity (1D/2D)
├── loss_shape_optimized.m         # IoU, centroid consistency (global terms)
│
└── README.md                      # This file
```

```text
waymo_segment_data/
    pair_summary_xx_deduplicated.csv
    ├── <seq_id>/
        ├── <tfrecord_name>/
            ├── <frame_id>/
                ├── <pixel_file>.txt   # 2D car mask pixels
                ├── <lidar_file>.txt   # LiDAR points of car instance
                ├── copyy_chao.mat     # (ignored in CASA-Calib)
                ├── calib.txt          # Intrinsics + extrinsics
```


🎯 How to Reproduce Figures in the Paper
1. Figure 4 — Tau Sensitivit
Run:
Tau_Sensitivity_Analysis

Outputs:
Tau sweep (% improvement)
Pareto plot
Distance-to-ideal score + 2% stability band
Matches Fig. 4(a) and Fig. 4(b).

2. Figure 5 — Perturbation Robustness

Run:
perturbation_analysis

Choose:
Rotation-only
Translation-only
Rotation + translation (default)

Outputs:
average rotation error vs perturbation
average translation error vs perturbation
optional: loss / keep-ratio visualization
Reproduces Fig. 5(a)(b).

3. Figure 7 — Cost Landscape (2D + 3D)
Run:
cost_landscape

The script computes the multi-frame CASA cost around ground-truth:
3D surface of J(Δty, Δtz)
2D contour + metrics (d*, FWHM, Aε)
Reproduces Fig. 7.

4. CASA-Loss (Core Loss Function)

CASA_Loss.m implements the exact formulation in Section III:

| Term                          | Description                             |
| ----------------------------- | --------------------------------------- |
| **IoU similarity**            | Global shape alignment                  |
| **Centroid consistency (CC)** | Penalizes shifts between contours       |
| **SDS-1D**                    | Line-like local distribution similarity |
| **SDS-2D**                    | Area-like local distribution similarity |
| **α coupling**                | IoU-guided weighting                    |

This function is used by all optimization scripts.

🔗 Function Dependency Graph
```text
CASA_Loss
 ├── img_contour_seq_fast
 ├── LiDAR_contour_extraction_opt
 ├── loss_proj
 └── loss_shape_optimized

perturbation_analysis
 └── CASA_Loss

cost_landscape
 └── CASA_Loss
```
📩 Questions / Issues

If you encounter missing files, dataset format questions, or need help adapting the code, feel free to open a GitHub issue or contact the author.

