import torch
import numpy as np
from src.utils.graphics_utils import focal2fov, getProjectionMatrix

class MiniCam:
    """
    A lightweight PyTorch camera class designed to feed into gsplat's rasterizer.
    """
    def __init__(self, width, height, R, T, fx, fy, cx, cy, znear=0.01, zfar=100.0, device="cuda"):
        self.image_width = width
        self.image_height = height
        
        # Intrinsics
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        # Field of View calculations using our utility functions
        self.fovY = focal2fov(self.fy, self.image_height)
        self.fovX = focal2fov(self.fx, self.image_width)
        
        # Extrinsics: Construct the 4x4 World-to-Camera (W2C) matrix
        # W2C = [R | T]
        #       [0 | 1]
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T
        
        # Move to PyTorch and target device
        self.world_view_transform = torch.tensor(w2c, dtype=torch.float32, device=device)
        
        # The camera center in world coordinates (inverse of W2C translation)
        c2w = torch.inverse(self.world_view_transform)
        self.camera_center = c2w[:3, 3]
        
        # Projection Matrix (used for depth clipping and bounding in 3D viewers)
        # Note: We transpose it because PyTorch uses column-major multiplication for these transforms
        self.projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=self.fovX, fovY=self.fovY).transpose(0, 1).to(device)
        
        # Full Projection Transform (World to Screen)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

    def get_intrinsics_matrix(self):
        """Returns the 3x3 intrinsics matrix K."""
        K = torch.tensor([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.world_view_transform.device)
        return K