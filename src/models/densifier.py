import torch
import torch.nn as nn
import numpy as np

class Densifier:
    def __init__(self, model, grad_threshold=0.0002, opacity_threshold=0.005, extent=10.0):
        self.model = model
        self.grad_threshold = grad_threshold
        self.opacity_threshold = opacity_threshold
        self.percent_dense = 0.01
        self.spatial_extent = extent 
        
        # Keep a running tally of view-space gradients
        self.xyz_gradient_accum = torch.zeros((self.model.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.model.get_xyz.shape[0], 1), device="cuda")

    def track_gradients(self, viewspace_points, visibility_filter):
        """Called every iteration to accumulate the 2D positional gradients."""
        if viewspace_points.grad is not None:
            grads = torch.norm(viewspace_points.grad[visibility_filter, :2], dim=-1, keepdim=True)
            self.xyz_gradient_accum[visibility_filter] += grads
            self.denom[visibility_filter] += 1

    def densify_and_prune(self, optimizer, iteration):
        """Called every N iterations to update the point cloud."""
        grads = self.xyz_gradient_accum / self.denom.clamp(min=1)
        grads[grads.isnan()] = 0.0

        # 1. Identify ALL Gaussians that have high gradients
        high_grad_mask = torch.norm(grads, dim=-1) >= self.grad_threshold

        # 2. Divide into Clone and Split
        # Small points get duplicated. Large points get chopped in half.
        split_threshold = 0.5 
        
        max_scales = torch.max(self.model.get_scaling, dim=1).values
        clone_mask = torch.logical_and(high_grad_mask, max_scales <= split_threshold)
        split_mask = torch.logical_and(high_grad_mask, max_scales > split_threshold)

        # 3. Clone the small Gaussians
        self._clone_gaussians(clone_mask, optimizer)

        # 4. Pad the split mask to account for the newly cloned points
        if clone_mask.any():
            num_new_clones = clone_mask.sum().item()
            pad = torch.zeros(num_new_clones, dtype=torch.bool, device="cuda")
            split_mask = torch.cat([split_mask, pad])

        # 5. Split the large Gaussians
        self._split_gaussians(split_mask, optimizer)

        # 6. Prune nearly transparent AND excessively large Gaussians
        current_max_scales = torch.max(self.model.get_scaling, dim=1).values
        opacity_mask = (self.model.get_opacity < self.opacity_threshold).squeeze()
        
        # Stop mass-murdering the background! Increase limit to 5.0 meters
        max_scale_limit = 1.5
        scale_mask = (current_max_scales > max_scale_limit).squeeze()
        
        prune_mask = torch.logical_or(opacity_mask, scale_mask)
        self._prune_gaussians(prune_mask, optimizer)

        # Reset accumulators
        self.xyz_gradient_accum = torch.zeros((self.model.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.model.get_xyz.shape[0], 1), device="cuda")

    # --- Optimizer & Tensor Utilities ---

    def _replace_tensor(self, tensor, mask, new_tensor):
        """Utility to keep valid points and append new ones."""
        valid_pts = ~mask
        merged = torch.cat([tensor[valid_pts], new_tensor], dim=0)
        return nn.Parameter(merged.requires_grad_(True))

    def _update_optimizer_state(self, optimizer, mask, new_tensors_dict):
        """Updates the Adam optimizer state dictionaries dynamically."""
        valid_pts = ~mask
        for group in optimizer.param_groups:
            name = group["name"]
            if name in new_tensors_dict:
                p = group['params'][0]
                new_p = new_tensors_dict[name]
                state = optimizer.state.get(p, None)
                if state is not None:
                    # FIX: Calculate exact number of new points added
                    kept_pts = valid_pts.sum().item()
                    num_new_pts = new_p.shape[0] - kept_pts
                    
                    # Create zero padding strictly for the newly added points
                    zero_shape = list(new_p.shape)
                    zero_shape[0] = num_new_pts
                    
                    zeros_exp_avg = torch.zeros(zero_shape, dtype=state["exp_avg"].dtype, device=new_p.device)
                    zeros_exp_avg_sq = torch.zeros(zero_shape, dtype=state["exp_avg_sq"].dtype, device=new_p.device)
                    
                    state["exp_avg"] = torch.cat([state["exp_avg"][valid_pts], zeros_exp_avg], dim=0)
                    state["exp_avg_sq"] = torch.cat([state["exp_avg_sq"][valid_pts], zeros_exp_avg_sq], dim=0)
                    del optimizer.state[p]
                
                group['params'][0] = new_p
                if state is not None:
                    optimizer.state[group['params'][0]] = state

    # --- Density Operations ---

    def _clone_gaussians(self, mask, optimizer):
        if not mask.any(): return
        
        # Duplicate parameters exactly
        new_xyz = self.model._xyz[mask]
        new_f_dc = self.model._features_dc[mask]
        new_f_rest = self.model._features_rest[mask]
        new_opacities = self.model._opacity[mask]
        new_scaling = self.model._scaling[mask]
        new_rotation = self.model._rotation[mask]

        empty_mask = torch.zeros_like(mask, dtype=torch.bool)
        new_tensors = {
            "xyz": self._replace_tensor(self.model._xyz, empty_mask, new_xyz),
            "f_dc": self._replace_tensor(self.model._features_dc, empty_mask, new_f_dc),
            "f_rest": self._replace_tensor(self.model._features_rest, empty_mask, new_f_rest),
            "opacity": self._replace_tensor(self.model._opacity, empty_mask, new_opacities),
            "scaling": self._replace_tensor(self.model._scaling, empty_mask, new_scaling),
            "rotation": self._replace_tensor(self.model._rotation, empty_mask, new_rotation)
        }
        
        self._update_optimizer_state(optimizer, empty_mask, new_tensors)
        self._reassign_model_tensors(new_tensors)

    def _split_gaussians(self, mask, optimizer):
        if not mask.any(): return
        
        # Sample new positions based on the current scale
        stds = self.model.get_scaling[mask].repeat(2, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        
        # Offset the split splats
        new_xyz = self.model._xyz[mask].repeat(2, 1) + samples 
        
        # Scale them down by a factor of 1.6 (subtract log(1.6) in log space)
        new_scaling = self.model._scaling[mask].repeat(2, 1) - np.log(1.6) 
        
        # Duplicate remaining attributes
        new_f_dc = self.model._features_dc[mask].repeat(2, 1, 1)
        new_f_rest = self.model._features_rest[mask].repeat(2, 1, 1)
        new_opacities = self.model._opacity[mask].repeat(2, 1)
        new_rotation = self.model._rotation[mask].repeat(2, 1)

        new_tensors = {
            "xyz": self._replace_tensor(self.model._xyz, mask, new_xyz),
            "f_dc": self._replace_tensor(self.model._features_dc, mask, new_f_dc),
            "f_rest": self._replace_tensor(self.model._features_rest, mask, new_f_rest),
            "opacity": self._replace_tensor(self.model._opacity, mask, new_opacities),
            "scaling": self._replace_tensor(self.model._scaling, mask, new_scaling),
            "rotation": self._replace_tensor(self.model._rotation, mask, new_rotation)
        }
        
        self._update_optimizer_state(optimizer, mask, new_tensors)
        self._reassign_model_tensors(new_tensors)

    def _prune_gaussians(self, mask, optimizer):
        if not mask.any(): return
        
        empty_tensor = torch.empty((0,), device="cuda")
        
        new_tensors = {
            "xyz": self._replace_tensor(self.model._xyz, mask, empty_tensor),
            "f_dc": self._replace_tensor(self.model._features_dc, mask, empty_tensor),
            "f_rest": self._replace_tensor(self.model._features_rest, mask, empty_tensor),
            "opacity": self._replace_tensor(self.model._opacity, mask, empty_tensor),
            "scaling": self._replace_tensor(self.model._scaling, mask, empty_tensor),
            "rotation": self._replace_tensor(self.model._rotation, mask, empty_tensor)
        }
        
        self._update_optimizer_state(optimizer, mask, new_tensors)
        self._reassign_model_tensors(new_tensors)

    def _reassign_model_tensors(self, new_tensors):
        """Helper to reassign tensors back to the main model."""
        self.model._xyz = new_tensors["xyz"]
        self.model._features_dc = new_tensors["f_dc"]
        self.model._features_rest = new_tensors["f_rest"]
        self.model._opacity = new_tensors["opacity"]
        self.model._scaling = new_tensors["scaling"]
        self.model._rotation = new_tensors["rotation"]