import os
import glob
import numpy as np
import torch
import cv2
from src.data.camera import MiniCam

def parse_calib_cam_to_cam(filepath):
    """
    Parses the KITTI calib_cam_to_cam.txt file.
    
    Args:
        filepath (str): Path to the calibration text file.
        
    Returns:
        dict: A dictionary mapping string keys to NumPy arrays.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Calibration file not found at: {filepath}")

    calib_data = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or the timestamp
            if not line or line.startswith('calib_time'):
                continue

            # Split the line into the key and the values
            key, value = line.split(':', 1)
            key = key.strip()
            
            # Convert the space-separated string of numbers into a list of floats
            float_values = [float(x) for x in value.split()]

            # Reshape known matrices based on their expected mathematical dimensions
            if key.startswith('K_') or key.startswith('R_') or key.startswith('R_rect_'):
                calib_data[key] = np.array(float_values).reshape(3, 3)
            elif key.startswith('P_rect_'):
                calib_data[key] = np.array(float_values).reshape(3, 4)
            else:
                calib_data[key] = np.array(float_values)

    return calib_data


class KittiDataset:
    """
    Manages data loading, intrinsic/extrinsic mapping, and frame fetching for training.
    """
    def __init__(self, config):
        self.image_dir = config['data']['image_dir']
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
        
        # 3. Locate all ground-truth images
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))
        self.num_frames = len(self.image_paths)
        
        if self.num_frames == 0:
            print(f"WARNING: No .png files found in {self.image_dir}")
        else:
            print(f"Successfully loaded KITTI dataset indexing {self.num_frames} frames.")

    def get_random_frame(self):
        """
        Pulls a random image from the dataset, converts it to a PyTorch tensor, 
        and generates the corresponding MiniCam object.
        """
        # Pick a random frame index for stochastic gradient descent
        idx = np.random.randint(0, self.num_frames)
        
        # Load Image (OpenCV loads as BGR, so we must convert to RGB)
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Convert to PyTorch Tensor [C, H, W] and normalize pixel values to [0, 1]
        gt_image = torch.tensor(image, dtype=torch.float32, device=self.device) / 255.0
        gt_image = gt_image.permute(2, 0, 1) 
        
        # Fetch Extrinsics (Rotation and Translation)
        # Note: A full KITTI implementation uses the 'oxts' GPS/IMU data to calculate 
        # world-space movement per frame. For initial baseline testing and debugging, 
        # we initialize the camera at the origin (Identity matrix).
        R = np.eye(3, dtype=np.float32)
        T = np.zeros(3, dtype=np.float32)
        
        # Instantiate the PyTorch camera for gsplat
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
    # Mock config to test the class independently
    mock_config = {
        'data': {
            'image_dir': "data/image_02",
            'calib_cam_to_cam': "data/calib_cam_to_cam.txt"
        },
        'experiment': {
            'device': "cpu" # Test on CPU locally
        }
    }
    
    try:
        dataset = KittiDataset(mock_config)
        cam, img = dataset.get_random_frame()
        print(f"Successfully loaded random frame!")
        print(f"Image Tensor Shape: {img.shape}")
        print(f"Camera FOV (X/Y): {cam.fovX:.2f} / {cam.fovY:.2f} rad")
    except Exception as e:
        print(f"Testing failed: {e}")