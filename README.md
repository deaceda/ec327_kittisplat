# KITTI Splat: High-Fidelity Street Reconstruction

This repository contains a 3D Gaussian Splatting (3DGS) pipeline optimized for the sparse KITTI autonomous driving dataset. It uses sensor fusion and semantic AI masking to reconstruct a stable, high-fidelity 3D street scene. An example output can be seen below!

![KITTI Splat Output Example](splat_example.png)

> **Note:** This pipeline is configured specifically to be run end-to-end within Google Colab.

---

## 1. Setup & Installation (Google Drive)

Because this project relies on heavy GPU computation, everything is orchestrated through Google Drive and Google Colab.

1. **Upload the Repository:** Download or clone this repository to your local machine using your terminal:
   ```bash
   git clone [https://github.com/deaceda/ec327_kittisplat.git](https://github.com/deaceda/ec327_kittisplat.git) KITTI_Project
