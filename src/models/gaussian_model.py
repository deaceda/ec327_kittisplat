import os
import torch
import torch.nn as nn
import numpy as np
from plyfile import PlyData, PlyElement
from src.utils.sh_utils import RGB2SH

class GaussianModel(nn.Module):
    def __init__(self, sh_degree=3):
        super().__init__()
        self.max_sh_degree = sh_degree
        
        # Tensors to hold our Gaussian parameters
        self._xyz = torch.empty(0)       # Spatial means (mu)
        self._features_dc = torch.empty(0) # Spherical Harmonics (base color)
        self._features_rest = torch.empty(0) # Spherical Harmonics (view-dependent)
        self._scaling = torch.empty(0)   # Scale for covariance
        self._rotation = torch.empty(0)  # Quaternion for covariance
        self._opacity = torch.empty(0)   # Alpha / Opacity

    def create_from_pcd(self, pcd_xyz, pcd_colors=None, spatial_lr_scale=1.0, device="cuda"):
        """
        Initializes the Gaussians from a LiDAR point cloud (pcd_xyz).
        """
        # 1. Spatial Means (mu): Direct mapping from LiDAR xyz
        fused_point_cloud = torch.tensor(np.asarray(pcd_xyz)).float().to(device)
        
        # 2. Spherical Harmonics (Color):
        # If LiDAR has no color, initialize with a neutral gray (0.5)
        num_pts = fused_point_cloud.shape[0]
        if pcd_colors is None:
            # SH DC term for gray (roughly 0.5 RGB converted to SH)
            fused_color = torch.ones((num_pts, 3), device=device) * 0.5 
        else:
            fused_color = torch.tensor(np.asarray(pcd_colors)).float().to(device)
            
        # Convert RGB to SH base (DC term) using the new utility function
        features = torch.zeros((num_pts, 3, (self.max_sh_degree + 1) ** 2), device=device)
        features[:, :3, 0] = RGB2SH(fused_color) 
        
        # 3. Covariance (Scales and Rotations):
        # Initialize scales to be small and uniform based on nearest-neighbor distance
        dist2 = torch.clamp_min(self._get_nearest_neighbor_dist(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        
        # Initialize rotations as identity quaternions [1, 0, 0, 0]
        rots = torch.zeros((num_pts, 4), device=device)
        rots[:, 0] = 1.0 

        # 4. Opacity (alpha):
        # Initialize to a low value (e.g., 0.1) using inverse sigmoid
        opacities = torch.logit(0.1 * torch.ones((num_pts, 1), device=device))

        # Register as optimizable parameters
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        print(f"Initialized {num_pts} Gaussians from point cloud.")

    def _get_nearest_neighbor_dist(self, xyz):
        """Helper to calculate initial scale based on point density."""
        return torch.ones((xyz.shape[0]), device=xyz.device) * 0.01 

    # --- Properties to access activated variables during training ---
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        # Exponentiate to ensure scales are strictly positive
        scales = torch.exp(self._scaling)
        
        # The Hard Clamp tightened to 0.08
        return torch.clamp(scales, max=0.08)

    @property
    def get_rotation(self):
        # Normalize quaternions
        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_opacity(self):
        # Sigmoid bounds opacity between 0 and 1
        return torch.sigmoid(self._opacity)

    # --- PLY Exporter ---

    def construct_list_of_attributes(self):
        """Helper to build the expected PLY attribute headers."""
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels of DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        # All channels of Rest
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        """
        Exports the optimized 3D Gaussians to a standard .ply format.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Detach from Autograd, move to CPU, and convert to numpy
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz) # 3DGS format requires normal columns, usually left as 0
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # Concatenate all properties into one massive array
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))

        # Write to disk
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        print(f"Successfully saved 3D Gaussians to {path}")