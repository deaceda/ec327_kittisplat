import torch
from tqdm import tqdm
from src.models.loss_functions import combined_loss
from gsplat import project_gaussians, rasterize_gaussians

class SplatTrainer:
    def __init__(self, gaussian_model, dataset, densifier=None, iterations=30000, device="cuda"):
        self.model = gaussian_model
        self.dataset = dataset 
        self.densifier = densifier 
        self.iterations = iterations
        self.device = device
        
        # Define learning rates for each parameter group
        self.optimizer = torch.optim.Adam([
            {'params': [self.model._xyz], 'lr': 0.00016, "name": "xyz"},
            {'params': [self.model._features_dc], 'lr': 0.0025, "name": "f_dc"},
            {'params': [self.model._features_rest], 'lr': 0.000125, "name": "f_rest"},
            {'params': [self.model._opacity], 'lr': 0.05, "name": "opacity"},
            {'params': [self.model._scaling], 'lr': 0.005, "name": "scaling"},
            {'params': [self.model._rotation], 'lr': 0.001, "name": "rotation"}
        ], lr=0.0, eps=1e-15)

    def train(self):
        self.model.train()
        
        # Training loop with a progress bar
        progress_bar = tqdm(range(1, self.iterations + 1), desc="Training Splats")
        
        for iteration in progress_bar:
            # 1. Fetch a random camera frame and ground truth image from KITTI
            camera, gt_image = self.dataset.get_random_frame()
            
            # 2. Render the scene from this camera's perspective
            render_dict = self.render(camera)
            pred_image = render_dict["render"]
            
            # 3. Calculate Loss (Permute from HWC to CHW for loss functions)
            pred_image = pred_image.permute(2, 0, 1)
            loss = combined_loss(pred_image, gt_image)
            
            # 4. Backpropagation
            loss.backward()
            
            # 5. Adaptive Density Control (Clone/Split/Prune)
            if self.densifier is not None and iteration < 15000:
                # Track gradients every step
                self.densifier.track_gradients(render_dict["viewspace_points"], render_dict["visibility_filter"])
                
                # Densify and prune every 100 steps
                if iteration % 100 == 0:
                    self.densifier.densify_and_prune(self.optimizer, iteration)
            
            # 6. Optimizer Step
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            # Update progress bar
            progress_bar.set_postfix({"Loss": f"{loss.item():.{5}f}"})
            
    def render(self, camera, bg_color=torch.tensor([0.0, 0.0, 0.0], device="cuda")):
        """
        Projects 3D Gaussians to 2D and rasterizes them into an image.
        """
        # 1. Get current state of the Gaussians from the model
        means3D = self.model.get_xyz
        scales = self.model.get_scaling
        rotations = self.model.get_rotation
        opacities = self.model.get_opacity
        
        # FIX: Squeeze out the extra SH band dimension so shape goes from (N, 1, 3) to (N, 3)
        sh_dc = self.model._features_dc.squeeze(1)
        
        # 2. Extract camera matrices (gsplat expects column-major for viewmat)
        viewmat = camera.world_view_transform.transpose(0, 1) 
        K = camera.get_intrinsics_matrix()
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # 3. Project Gaussians into 2D screen space
        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(
            means3D, scales, 1.0, rotations, viewmat, 
            fx, fy, cx, cy, camera.image_height, camera.image_width, 
            block_width=16, clip_thresh=0.01
        )
        
        # Retain the 2D gradients for the Adaptive Density Control module
        xys.retain_grad()
        
        # 4. Rasterize the 2D splats into an RGB image
        render_colors, render_alphas = rasterize_gaussians(
            xys, depths, radii, conics, num_tiles_hit, 
            sh_dc, opacities, camera.image_height, camera.image_width, 
            block_width=16, background=bg_color
        )
        
        return {
            "render": render_colors,
            "viewspace_points": xys,
            "visibility_filter": radii > 0,
            "radii": radii
        }