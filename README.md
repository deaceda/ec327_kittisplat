# EC327 KITTI_splat

# KITTI Splat: High-Fidelity Street Reconstruction

This repository contains a 3D Gaussian Splatting (3DGS) pipeline optimized for the sparse KITTI autonomous driving dataset. It uses sensor fusion and semantic AI masking to reconstruct a stable, high-fidelity 3D street scene.

## 1. Setup & Data Preparation

First, clone the repository and install the necessary dependencies:

```bash
git clone [https://github.com/deaceda/ec327_kittisplat.git](https://github.com/deaceda/ec327_kittisplat.git)
cd ec327_kittisplat
pip install -r requirements.txt
```

### Fetching the Data
The project requires the **synced** KITTI dataset (specifically Residential Drive 0064) to properly align the LiDAR, GPS, and Camera data. 

We have provided an automated script to handle the downloading, extraction, and formatting of the dataset, as well as generating the SegFormer AI sky masks.

Navigate to `src/data` and run the data preparation script/notebook:
* Open and run all cells in `src/data/get_kitti_data.ipynb`.
* This will automatically build the required `data/` directory structure with the synced images, velodyne points, OXTS trajectories, and AI masks.

## 2. Running the Model

Once your data is prepared, you can start the 3D Gaussian Splatting optimizer. We use a custom Densifier with hard-coded physical limits (1.5m scale clamp, 8.0m height ceiling) to prevent geometric artifacts.

Run the training pipeline using the provided configuration file:

```bash
python train.py --config configs/residential_0064.yaml
```

The model will train for 30,000 iterations. By default, the Densifier shuts off at 25,000 iterations, dedicating the final 5,000 steps entirely to baking the high-resolution textures.

* **Outputs:** The trained 3D geometry will be saved in your `output/exp_0064/` directory. Be sure to run the final export function to generate the `point_cloud_radiancekit.ply` file.

## 3. Viewing in SuperSplat

To interact with your reconstructed street in real-time, we recommend using the PlayCanvas SuperSplat web viewer.

1. Locate the **`point_cloud_radiancekit.ply`** file generated in your output directory.
2. Open your web browser and go to: [PlayCanvas SuperSplat](https://playcanvas.com/supersplat)
3. Drag and drop your `.ply` file directly into the browser window.
4. **Navigation:** Switch to **Walk** mode (usually the `W` key or the shoe icon in the toolbar) to drive or walk down the 3D street using WASD controls.
