# EC327 KITTI_splat

KITTI Splat: High-Fidelity Street Reconstruction
Software Engineering Final Project • EC327 • 2024
This repository contains a full 3D Gaussian Splatting (3DGS) pipeline optimized for the sparse KITTI autonomous driving dataset. The core of this project is to create a high-fidelity "digital twin" of a street by synchronizing RGB camera data with Velodyne LiDAR geometry and OXTS GPS trajectory. To solve the unique challenges of outdoor driving data, the pipeline integrates custom sensor fusion logic, semantic AI masking (SegFormer) for sky removal, and a safety-constrained densification process.

Visual Guide (Optional Graphics)
The original workflow image is a perfect visual summary of the steps below.

Note to user: When you create this file on GitHub, you can add your provided workflow image here by uploading it and linking to it:
![KITTI Splat Workflow Graphic](path/to/your/uploaded/image_edbe23.jpg)

Key Features
Sensor Fusion: Mathematically aligns 3D Velodyne LiDAR points, OXTS trajectory data, and 2D camera intrinsic/extrinsic matrices into a unified 3D world space.

Semantic AI Masking: Leverages the SegFormer model (pre-trained on Cityscapes) to generate per-frame masks, which are used as a "mathematical blindfold" to force the optimizer to focus 100% of compute on ground-level geometry (the street) and ignore the sky.

Safety-Constrained Densification: Solves the common "reaching artifacts" and sky-bleeding issues in 3DGS by implementing hard physical boundaries (scale clamping at 1.5m and height ceiling at 8.0m) and optimized learning rates.

Work in Progress (Status)
This pipeline is currently in an active, stable state for standard street reconstruction. The recent visual and geometric optimizations effectively suppress "sky bleeding."

Current visual achievements:

A stable, non-splotchy red car.

A clean asphalt road.

No diagonal "sky spears" or fuzzy "hairy trees."

Getting Started
1. Prerequisite Setup
The primary environment for running this code is Google Colab (using the provided main_notebook.ipynb).

2. Repo Cloning
Start by creating a proj_kittisplat directory in your Google Drive and cloning this repository:

Bash
cd /content/drive/MyDrive/proj_kittisplat
git clone https://github.com/deaceda/ec327_kittisplat.git
3. Dependency Installation
The notebook will automatically handle the installation of dependencies. Key libraries include:

Standard Python libs: torch, imageio, matplotlib, numpy, opencv-python.

Gaussian Splatting: gaussian-splatting, diff-gaussian-rasterization (the rasterizer must be compiled from source on Colab).

Masking: transformers (to run the SegFormer model).

Data Organization
You must organize the data precisely within the proj_kittisplat/data folder to match the code's expectations.

Note on Data: For your project, the synced data option is used. Specifically, the drive 2011_09_26_drive_0064_sync/ is used, as it aligns LiDAR and camera data temporally.

Required Dataset Files:
Ensure you have downloaded and placed these specific components from the KITTI Residential category (for drive 0064):

devkit

calibration

tracklets

velodyne

The final data structure must look exactly like this:

Plaintext
proj_kittisplat/data/
├── 2011_09_26_drive_0064_sync/
│   ├── image_02/         # Main RGB camera images
│   ├── image_03/         # Stereo camera images (not required for 3DGS)
│   └── velodyne_points/  # Raw LiDAR scans (.bin)
├── calibration/          # calibration.txt, calib_cam_to_velodyne.txt
├── oxts/                 # trajectory/gps data
├── devkit/               # KITTI development kit
└── masks/                # Place for SegFormer-generated sky masks
Usage
1. Generating Training Files
The first critical step is transforming the raw sensor data into the training files (transforms_train.json and points3d.ply) used by the 3DGS optimizer. This involves stitching the separate LiDAR scans together based on the OXTS trajectory.

Run this script in the notebook:

Python
!python -m src.data.create_dataset \
    --image_path data/2011_09_26_drive_0064_sync/image_02 \
    --mask_path data/masks \
    --oxts_path data/oxts/residential_0064_oxts.txt \
    --output_path data
2. Standard Training
With the dataset generated, you can start the 3DGS training. The pipeline uses the parameters defined in the configuration file (configs/residential_0064.yaml). The default number of iterations is 30,000.

Run this script to begin:

Python
!python train.py --config configs/residential_0064.yaml
Scene Clean-up: The Geometric Guardrails
The "reaching" artifacts (splats stretching to the sky) were fixed by a triple optimization. If you are experiencing geometric noise, check and adjust these two files.

1. src/models/densifier.py
To physically ban the spikes, use these hard mathematical constraints:

Scale Clamp (1.5m): Prevents individual splats from growing into giant 15-foot needles.

Height Ceiling (8.0m): Instantly deletes any point that grows too tall (roughly 26 feet).

2. configs/residential_0064.yaml
Slow down the scaling speed to stop the splats from "exploding" toward the horizon before the optimizer can prune them:

Lower lr_scaling to 0.001.

Visualization
The outputs of the main_notebook.ipynb will be synced to your Google Drive (/MyDrive/proj_kittisplat/output/).

1. Reference Video
To visualize the masked data and sensor synchronization, download the generated reference video:

Location: output/exp_0064/kitti_reference.mp4

2. Navigable 3D Scene
This is the rewarding final step. To walk through your reconstructed street:

Download the PLY file: Grab output/exp_0064/point_cloud_radiancekit.ply from your Drive.

Go to SuperSplat: Open the PlayCanvas SuperSplat viewer in your browser.

Load: Drag and drop your .ply file directly onto the viewer.

Navigation Tip: You might need to use the Rotate tool to lay the scene flat. Switch to Walk mode (usually the W key) to drive or walk down the street.
