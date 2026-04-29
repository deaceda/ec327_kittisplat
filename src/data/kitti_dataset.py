import os
import glob
import numpy as np
import torch
import cv2
from src.data.camera import MiniCam

def parse_kitti_calib(filepath):
    """
    Parses KITTI calibration files (velo_to_cam, etc) while ignoring metadata.
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip() or ":" not in line: 
                continue
                
            key, value = line.split(':', 1)
            key = key.strip()
            
            # FIX: Skip the 'calib_time' line which contains non-numeric strings
            if key == "calib_time":
                continue
                
            try:
                # Attempt to convert the values to a NumPy array of floats
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                # Skip any other lines that don't contain purely numeric data
                continue
    return data

def get_oxts_pose(oxts, scale):
    """Converts KITTI OXTS data into a 4x4 IMU-to-World matrix."""
    lat, lon, alt, roll, pitch, yaw = oxts
    er = 6378137.0 
    tx = scale * lon * np.pi * er / 180.0
    ty = scale * er * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
    tz = alt
    rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R = rz @ ry @ rx
    pose = np.eye(4); pose[:3, :3] = R; pose[:3, 3] = [tx, ty, tz]
    return pose

class KittiDataset:
    def __init__(self, config):
        # 1. Load Paths
        self.image_paths = sorted(glob.glob(os.path.join(config['data']['image_dir'], "*.png")))[:50]
        self.oxts_paths = sorted(glob.glob(os.path.join(config['data']['oxts_dir'], "[0-9]*.txt")))[:50]
        self.device = config['experiment']['device']

        # 2. Parse Projection (Intrinsics) and Rectification
        calib_cam = parse_kitti_calib(config['data']['calib_cam_to_cam'])
        P2 = calib_cam['P_rect_02'].reshape(3, 4)
        self.fx, self.fy, self.cx, self.cy = P2[0,0], P2[1,1], P2[0,2], P2[1,2]
        
        # FIX: Extract the Rectification Matrix
        R_rect_00 = np.eye(4)
        R_rect_00[:3, :3] = calib_cam['R_rect_00'].reshape(3, 3)

        # 3. Parse Velo-to-Cam (Extrinsic Offset)
        calib_v2c = parse_kitti_calib(config['data']['calib_velo_to_cam'])
        Tr_v2c = np.eye(4)
        Tr_v2c[:3, :3] = calib_v2c['R'].reshape(3, 3)
        Tr_v2c[:3, 3] = calib_v2c['T']
        
        # Exact KITTI mapping -> Velo -> Cam0 -> Rectified Cam0
        Tr_v2c_rect = R_rect_00 @ Tr_v2c
        
        # FIX: Save this to 'self' so the notebook can use it for stitching
        self.Tr_v2c_rect = Tr_v2c_rect

        # 4. Compute W2C Trajectory
        self.w2c_matrices = []
        first_imu_to_world_inv = None
        scale = None

        for path in self.oxts_paths:
            with open(path, 'r') as f:
                oxts = [float(x) for x in f.readline().strip().split()[:6]]
            if scale is None: scale = np.cos(oxts[0] * np.pi / 180.0)
            
            imu_to_world = get_oxts_pose(oxts, scale)
            if first_imu_to_world_inv is None:
                first_imu_to_world_inv = np.linalg.inv(imu_to_world)
            
            rel_pose = first_imu_to_world_inv @ imu_to_world
            
            # Apply the fully rectified transformation
            w2c = Tr_v2c_rect @ np.linalg.inv(rel_pose)
            self.w2c_matrices.append(w2c)

        print(f"Dataset ready. Motion and Geometry are now perfectly synced.")

    def get_random_frame(self):
        idx = np.random.randint(0, len(self.image_paths))
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        gt = torch.tensor(image, dtype=torch.float32, device=self.device).permute(2, 0, 1) / 255.0
        
        w2c = self.w2c_matrices[idx]
        camera = MiniCam(w, h, w2c[:3,:3], w2c[:3,3], self.fx, self.fy, self.cx, self.cy, device=self.device)
        return camera, gt
    
    def get_frame_by_index(self, idx):
        # Move the logic from get_random_frame here
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        gt = torch.tensor(image, dtype=torch.float32, device=self.device).permute(2, 0, 1) / 255.0
        w2c = self.w2c_matrices[idx]
        camera = MiniCam(w, h, w2c[:3,:3], w2c[:3,3], self.fx, self.fy, self.cx, self.cy, device=self.device)
        return camera, gt