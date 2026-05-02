import torch
import math
import numpy as np
from tqdm import tqdm
from src.models.loss_functions import combined_loss
from gsplat import project_gaussians, rasterize_gaussians
from src.utils.sh_utils import SH2RGB

class SplatTrainer:
    def __init__(self, gaussian_model, dataset, densifier=None, iterations=15000, device="cuda"):
        self.model = gaussian_model
        self.dataset = dataset 
        self.config = config
        self.densifier = densifier 
        self.iterations = iterations
        self.device = device
        
        # 1. Define initial and final learning rates for the decay
        self.xyz_lr_init = 0.00016
        self.xyz_lr_final = 0.0000016 # Decay to 1% of initial value
        
        # Define learning rates for each parameter group
        self.optimizer = torch.optim.Adam([
            {'params': [self.model._xyz], 'lr': self.xyz_lr_init, "name": "xyz"},
            {'params': [self.model._features_dc], 'lr': 0.01, "name": "f_dc"},
            {'params': [self.model._features_rest], 'lr': 0.0005, "name": "f_rest"},
            {'params': [self.model._opacity], 'lr': 0.1, "name": "opacity"},
            {'params': [self.model._scaling], 'lr': 0.005, "name": "scaling"},
            {'params': [self.model._rotation], 'lr': 0.001, "name": "rotation"}
        ], lr=0.001, eps=1e-15)

    def get_expon_lr(self, iteration):
        """Calculates the decayed learning rate for the current iteration."""
        t = iteration / self.iterations
        return self.xyz_lr_init * (self.xyz_lr_final / self.xyz_lr_init) ** t

    def update_learning_rate(self, iteration):
        """Updates the learning rate for the 'xyz' parameter group in the optimizer."""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.get_expon_lr(iteration)
                param_group['lr'] = lr

    def train(self):
        self.model.train()
        progress_bar = tqdm(range(1, self.iterations + 1), desc="Training Splats")
        
        for iteration in progress_bar:
            self.update_learning_rate(iteration)
            
            # CURRICULUM: Gradually introduce more frames
            # Start with 5 frames, reach all 50 frames by iteration 10,000
            curriculum_end = int(self.iterations * 0.8)
            num_available = len(self.dataset.image_paths)
            current_max = min(num_available, 5 + int((iteration / curriculum_end) * num_available))
            idx = np.random.randint(0, current_max)
            
            camera, gt_image = self.dataset.get_frame_by_index(idx)
            render_dict = self.render(camera)
            pred_image = render_dict["render"]
            
            # Calculate Loss
            pred_image = pred_image.permute(2, 0, 1)
            loss = combined_loss(pred_image, gt_image)
            
            # Backpropagation
            loss.backward()
            
            # Adaptive Density Control (ends at 25k)
            end_iter = self.config['densification']['end_iteration']
            interval = self.config['densification']['interval']
            
            if self.densifier is not None and iteration < end_iter:
                self.densifier.track_gradients(render_dict["viewspace_points"], render_dict["visibility_filter"])
                if iteration % interval == 0:
                    self.densifier.densify_and_prune(self.optimizer, iteration)
            
            # Optimizer Step
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            # Fetch the current decayed LR to show in the progress bar
            current_xyz_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.{5}f}",
                "LR_xyz": f"{current_xyz_lr:.{8}f}"
            })
            
    def render(self, camera, bg_color=torch.tensor([0.0, 0.0, 0.0], device="cuda")):
        means3D = self.model.get_xyz
        scales = self.model.get_scaling
        rotations = self.model.get_rotation
        opacities = self.model.get_opacity
        sh_dc = self.model._features_dc.squeeze(1)
        
        # FIX: Convert raw SH coefficients to actual RGB colors!
        colors = SH2RGB(sh_dc)
        colors = torch.clamp(colors, 0.0, 1.0)
        
        # 2. Extract camera matrices
        # FIX: Remove .transpose(0, 1)! gsplat expects a standard row-major tensor.
        viewmat = camera.world_view_transform.contiguous() 
        
        K = camera.get_intrinsics_matrix()
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(
            means3D, scales, 1.0, rotations, viewmat, 
            fx, fy, cx, cy, camera.image_height, camera.image_width, 
            block_width=16, clip_thresh=0.01
        )
        xys.retain_grad()
        
        render_colors = rasterize_gaussians(
            xys, depths, radii, conics, num_tiles_hit, 
            colors, opacities, camera.image_height, camera.image_width, 
            block_width=16, background=bg_color
        )
        
        return {
            "render": render_colors,
            "viewspace_points": xys,
            "visibility_filter": radii > 0,
            "radii": radii
        }
