# CASA-Calib (Paper with Code)
CASA-Calib: A Context-Aware Semantic Alignment Method for LiDAR-Camera Extrinsic Calibration for Vehicle Perception Systems

> **Note:** This repository is currently being actively updated.  
> Components related to dataset construction, visualization tooling, and CASA-Calib modules  
> are under continuous refinement. Additional documentation and examples will be released soon.


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

Semantic–Geometric Dataset Builder

Contribution III — Semantic–Geometric Test Set Construction

This repository includes a custom data extraction tool that constructs a curated semantic–geometric test set derived from the Waymo Open Dataset, as described in Contribution 3 of our paper:


“We construct and release a curated semantic–geometric test set based on the Waymo Open Dataset, providing reliable instance-level correspondences for accurate evaluation and benchmarking of semantic-assisted LiDAR–camera calibration methods.”


Unlike standard datasets—where


LiDAR instance IDs and image instance IDs do not correspond,


camera–LiDAR associations must be manually aligned, and


segmentation labels may contain annotation errors,


our tool automatically aligns per-instance LiDAR and camera semantic labels, and exports a cleaned, structured dataset suitable for semantic-assisted calibration research (e.g., CASA-Calib).

🛠 Semantic–Geometric Dataset Builder
File: waymo_semantic_geometric_builder.py

This script processes raw .tfrecord files from the Waymo Open Dataset and generates a pairwise-consistent LiDAR–camera dataset with:

✔ Reliable instance-level correspondences

✔ Pixel-level image segmentation masks

✔ LiDAR point-level semantic & instance labels

✔ Synchronized calibration matrices

✔ A directory structure compatible with CASA-Calib


📦 Output Directory Structure

After running the tool, each valid frame will be exported as:

```text
waymo_segment_data/
 └── <sequence_id>/
      └── <tfrecord_name>/
           └── <frame_id>/
                ├── calib.txt                 # KITTI-style camera–LiDAR extrinsic
                ├── img_raw.png               # RGB image
                ├── panoptic_label_front.png  # image segmentation (uint16)
                ├── instance_label_front.png  # instance map (uint16)
                ├── instance_waymo.png        # original Waymo instance ID map
                ├── points_all.txt            # LiDAR XYZ points (all beams)
                ├── point_labels_all.txt      # corresponding semantic/instance IDs
                ├── lidar.bin                 # binary point cloud file (float32)
                └── ... (additional metadata)
```

This format is fully compatible with CASA-Calib, and can also be used for:

1.Semantic calibration

2.Instance-matching research

3.LiDAR-camera fusion

4.3D instance segmentation training

📦 Description for Semantic–Geometric Dataset Builder
*A core component for constructing the CASA-Calib semantic–geometric evaluation dataset.*

This repository includes a dedicated extraction tool to convert raw Waymo Open Dataset
`.tfrecord` files into a curated semantic–geometric dataset.  
This dataset is required for evaluating instance-level LiDAR–camera calibration methods
and corresponds to **Contribution 3** of the CASA-Calib paper.

### ✨ Purpose  
Waymo provides high-quality but **independently generated** camera and LiDAR segmentation
annotations. CASA-Calib requires a dataset in which:

- semantic labels from *both* modalities are consistently extracted  
- LiDAR segmentation is converted into per-point labels  
- camera instance masks can be paired with LiDAR object clusters  
- each frame contains complete segmentation annotations  
- manual (or automatic) LiDAR–image instance correspondence can be established  

This extractor performs all the above and exports a clean, structured dataset suitable
for downstream calibration evaluation.

---

## Features of the Extractor  
- Extracts RGB images, panoptic labels, semantic labels, and instance masks  
- Converts Waymo’s LiDAR range-image segmentation into **3D point-level labels**  
- Saves LiDAR point clouds (`lidar.bin`), semantic labels, and projection information  
- Exports KITTI-style calibration matrices (`calib.txt`)  
- Filters invalid frames lacking segmentation labels  
- Provides a **Qt-based GUI** for interactive LiDAR–image instance matching  
- GUI buttons: **Save Match**, **Clear Selection**, **Skip Frame**, **Reset View**  
- Produces a CSV summary listing all matched instances per sequence  

## Matching GUI Overview

CASA-Calib includes an interactive Qt-based GUI for establishing instance-level
correspondences between camera segmentation masks and LiDAR point clusters.
This tool is essential for validating semantic–geometric consistency and for
constructing high-quality benchmark data.

The interface consists of four synchronized visualizations:

1. **Image Instance Label (Top-Left)**  
   Displays the instance-level segmentation mask from the camera.  
   Each instance is assigned a unique color and labeled with its instance ID.

2. **LiDAR BEV Projection — Top Instances (Top-Right)**  
   Shows the most prominent LiDAR object clusters in Bird’s-Eye-View (BEV).  
   Each cluster (instance ID) is color-coded consistently across subplots.

3. **All LiDAR Points Projected to the Image (Bottom-Left)**  
   Projects every LiDAR point into the camera frame, colored by depth (m).  
   This allows users to inspect geometric alignment and calibration consistency.

4. **Semantic Point Cloud Projection (Bottom-Right)**  
   Projects only semantically valid LiDAR points and colors them by instance ID.
   This view highlights object-level alignment between modalities.

### GUI Interaction

The interface supports the following actions:

| Button | Function |
|--------|----------|
| **Clear Selection** | Removes all selected pixel and LiDAR selections |
| **Save Match** | Stores the current pixel ↔ LiDAR instance correspondence |
| **Skip Frame** | Discards the current frame and removes temporary output |
| **Reset View** | Resets zoom and panning for all subplots |

### Example Screenshots

Below are example GUI screenshots captured during matching:

#### Image Instance Label + LiDAR BEV Projection
   
<img width="1919" height="1027" alt="GUI_1" src="https://github.com/user-attachments/assets/a5e32c39-c8eb-489e-981e-1c725f575ad2" />


#### Pixel Selection + LiDAR Cluster Highlighting

<img width="1919" height="1029" alt="GUI_2" src="https://github.com/user-attachments/assets/d5f07935-f514-4716-a946-7eb1ccefe39b" />


#### Final Instance Projection and Correspondence Visualization

<img width="1919" height="1029" alt="GUI_3" src="https://github.com/user-attachments/assets/75ec98dc-a7a5-4c56-8a9d-e6a09e17878e" />



   

📩 Questions / Issues

If you encounter missing files, dataset format questions, or need help adapting the code, feel free to open a GitHub issue or contact the author.

