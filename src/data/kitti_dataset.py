import os
import glob
import numpy as np
import torch
import cv2
from src.data.camera import MiniCam

def parse_calib_cam_to_cam(filepath):
    """
    Parses the KITTI calib_cam_to_cam.txt file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Calibration file not found at: {filepath}")

    calib_data = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('calib_time'):
                continue

            key, value = line.split(':', 1)
            key = key.strip()
            float_values = [float(x) for x in value.split()]

            if key.startswith('K_') or key.startswith('R_') or key.startswith('R_rect_'):
                calib_data[key] = np.array(float_values).reshape(3, 3)
            elif key.startswith('P_rect_'):
                calib_data[key] = np.array(float_values).reshape(3, 4)
            else:
                calib_data[key] = np.array(float_values)

    return calib_data

def get_oxts_pose(oxts, scale):
    """
    Converts KITTI OXTS GPS/IMU data into a 4x4 transformation matrix.
    Applies a local Mercator projection relative to the first frame.
    """
    lat, lon, alt, roll, pitch, yaw = oxts
    er = 6378137.0  # Earth radius in meters
    
    # Translation
    tx = scale * lon * np.pi * er / 180.0
    ty = scale * er * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
    tz = alt
    
    # Rotation (Euler to Matrix)
    rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    
    R = rz @ ry @ rx
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = [tx, ty, tz]
    return pose

class KittiDataset:
    """
    Manages data loading, intrinsic/extrinsic mapping, and frame fetching for training.
    """
    def __init__(self, config):
        self.image_dir = config['data']['image_dir']
        self.oxts_dir = config['data']['oxts_dir']
        self.calib_file = config['data']['calib_cam_to_cam']
        self.device = config['experiment']['device']
        
        # 1. Load calibration data and extract the Left Color Camera projection matrix
        self.calib = parse_calib_cam_to_cam(self.calib_file)
        self.P_rect_02 = self.calib.get('P_rect_02') 
        
        if self.P_rect_02 is None:
            raise ValueError("P_rect_02 not found in calibration file.")

        # 2. Extract intrinsics directly from the Projection matrix
        self.fx = self.P_rect_02[0, 0]
        self.fy = self.P_rect_02[1, 1]
        self.cx = self.P_rect_02[0, 2]
        self.cy = self.P_rect_02[1, 2]
        
        # 3. Locate ground-truth images and OXTS trajectory files
        # FIX: Slice the lists to only train on the first 50 frames
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))[:50]
        self.oxts_paths = sorted(glob.glob(os.path.join(self.oxts_dir, "[0-9]*.txt")))[:50]
        self.num_frames = len(self.image_paths)
        
        if self.num_frames == 0 or len(self.oxts_paths) == 0:
            print(f"WARNING: Missing image or OXTS files.")
            
        # 4. Precompute relative Camera Extrinsics (World-to-Camera matrices)
        self.w2c_matrices = []
        
        # Matrix to swap KITTI IMU axes to standard Camera axes
        T_cam_imu = np.array([
            [ 0, -1,  0, 0],
            [ 0,  0, -1, 0],
            [ 1,  0,  0, 0],
            [ 0,  0,  0, 1]
        ], dtype=np.float32)

        scale = None
        first_pose_inv = None
        
        for path in self.oxts_paths:
            with open(path, 'r') as f:
                # Extract lat, lon, alt, roll, pitch, yaw
                oxts = [float(x) for x in f.readline().strip().split()[:6]]
            
            # Set the map scale based on the first frame's latitude
            if scale is None:
                scale = np.cos(oxts[0] * np.pi / 180.0)
            
            # Get IMU pose in world coordinates
            imu_pose = get_oxts_pose(oxts, scale)
            
            # Make the trajectory relative to the very first frame
            if first_pose_inv is None:
                first_pose_inv = np.linalg.inv(imu_pose)
            rel_imu_pose = first_pose_inv @ imu_pose 
            
            # Convert to World-to-Camera (W2C) matrix for gsplat
            w2c = T_cam_imu @ np.linalg.inv(rel_imu_pose)
            self.w2c_matrices.append(w2c)
            
        print(f"Successfully loaded KITTI dataset indexing {self.num_frames} frames with GPS/IMU trajectories.")

    def get_random_frame(self):
        """
        Pulls a random image from the dataset, converts it to a PyTorch tensor, 
        and generates the corresponding MiniCam object.
        """
        idx = np.random.randint(0, self.num_frames)
        
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        gt_image = torch.tensor(image, dtype=torch.float32, device=self.device) / 255.0
        gt_image = gt_image.permute(2, 0, 1) 
        
        # 5. Fetch dynamically calculated Extrinsics
        w2c = self.w2c_matrices[idx]
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        
        camera = MiniCam(
            width=width, 
            height=height, 
            R=R, 
            T=T, 
            fx=self.fx, 
            fy=self.fy, 
            cx=self.cx, 
            cy=self.cy, 
            device=self.device
        )
        
        return camera, gt_image

# --- Debugging Execution ---
if __name__ == "__main__":
    mock_config = {
        'data': {
            'image_dir': "data/image_02",
            'oxts_dir': "data/oxts",
            'calib_cam_to_cam': "data/calib_cam_to_cam.txt"
        },
        'experiment': {
            'device': "cpu" 
        }
    }
    
    try:
        dataset = KittiDataset(mock_config)
        cam, img = dataset.get_random_frame()
        print(f"Successfully loaded random frame!")
        print(f"Camera Translation (X,Y,Z): {cam.camera_center.numpy()}")
    except Exception as e:
        print(f"Testing failed: {e}")