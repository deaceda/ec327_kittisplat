import torch

# Constant for the 0th degree Spherical Harmonic (DC term)
C0 = 0.28209479177387814

def RGB2SH(rgb):
    """
    Converts standard RGB values (0 to 1) into 0th-degree Spherical Harmonics.
    """
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    """
    Converts 0th-degree Spherical Harmonics back to RGB.
    """
    return sh * C0 + 0.5

def eval_sh(deg, sh, dirs):
    """
    Evaluates spherical harmonics at specific viewing directions.
    (gsplat's CUDA kernels usually handle this during the forward pass, 
    but having this CPU/PyTorch implementation is standard for debugging).
    """
    # Base DC term
    result = C0 * sh[..., 0]
    
    # In a full implementation, you would add C1, C2, C3 constants 
    # and polynomials for degrees 1, 2, and 3 here.
    # For KITTI Splat initialization, we primarily rely on the DC term (degree 0).
    return result