import torch
import math
import numpy as np
from tqdm import tqdm
from src.models.loss_functions import combined_loss
from gsplat import project_gaussians, rasterize_gaussians
from src.utils.sh_utils import SH2RGB

class SplatTrainer:
    def __init__(self, gaussian_model, dataset, config, densifier=None, iterations=15000, device="cuda"):
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
            {'params': [self.model._features_dc], 'lr': self.config['training']['lr_f_dc'], "name": "f_dc"},
            {'params': [self.model._features_rest], 'lr': self.config['training']['lr_f_rest'], "name": "f_rest"},
            {'params': [self.model._opacity], 'lr': self.config['training']['lr_opacity'], "name": "opacity"},
            {'params': [self.model._scaling], 'lr': self.config['training']['lr_scaling'], "name": "scaling"},
            {'params': [self.model._rotation], 'lr': self.config['training']['lr_rotation'], "name": "rotation"}
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
            
            # Unpack the new mask variable from the dataset
            camera, gt_image, mask = self.dataset.get_frame_by_index(idx)
            render_dict = self.render(camera)
            pred_image = render_dict["render"]
            
            # Match dimensions
            pred_image = pred_image.permute(2, 0, 1)
            
            # --- NEW: Apply the Sky Mask! ---
            # Multiplying by the mask instantly turns the sky to 0.0 in both images.
            # The optimizer will see 0 mathematical difference and ignore it completely.
            loss = combined_loss(pred_image * mask, gt_image * mask)
            # --------------------------------
            
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
            
            # --- LIVE PREVIEW SCRIPT ---
            if iteration % 1000 == 0:
                import matplotlib.pyplot as plt
                from IPython.display import clear_output
                
                with torch.no_grad():
                    # Format tensors for matplotlib (H, W, C)
                    render_np = pred_image.permute(1, 2, 0).detach().cpu().numpy()
                    gt_np = gt_image.permute(1, 2, 0).detach().cpu().numpy()
                    
                    # Clear the previous image so it looks like a live feed
                    clear_output(wait=True)
                    
                    # Create a side-by-side comparison
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    
                    axes[0].set_title(f"Ground Truth (Target)")
                    axes[0].imshow(gt_np)
                    axes[0].axis('off')
                    
                    axes[1].set_title(f"Live Splat Render (Iteration: {iteration})")
                    axes[1].imshow(render_np)
                    axes[1].axis('off')
                    
                    plt.show()
            # ---------------------------
            
            # Fetch the current decayed LR to show in the progress bar
            current_xyz_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.{5}f}",
                "LR_xyz": f"{current_xyz_lr:.{8}f}"
            })
            
    def render(self, camera, bg_color=None):
        # --- NEW: Dynamically set background color from config ---
        if bg_color is None:
            is_white = self.config['model'].get('white_background', False)
            bg_val = [1.0, 1.0, 1.0] if is_white else [0.0, 0.0, 0.0]
            bg_color = torch.tensor(bg_val, device=self.device)
        # ---------------------------------------------------------
            
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
            block_width=16, clip_thresh=0.2
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
