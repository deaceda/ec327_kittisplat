import torch
import torch.nn.functional as F
from math import exp

def l1_loss(network_output, gt):
    """Calculates the mean absolute error between the render and the ground truth."""
    return torch.abs((network_output - gt)).mean()

# --- SSIM Helper Functions ---
def gaussian(window_size, sigma):
    """Generates a 1D Gaussian kernel for SSIM."""
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """Creates a 2D Gaussian window."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    """Calculates the Structural Similarity Index between two images."""
    channel = img1.size(0) # Assuming input is [C, H, W]
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    # Add batch dimension for conv2d
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def combined_loss(pred_image, gt_image, lambda_dssim=0.2):
    """
    Combines L1 and D-SSIM loss.
    lambda_dssim controls the weight of the SSIM penalty. 
    0.2 is the standard weight used in the original 3DGS paper.
    """
    Ll1 = l1_loss(pred_image, gt_image)
    # D-SSIM is defined as 1 - SSIM
    L_ssim = 1.0 - ssim(pred_image, gt_image)
    
    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * L_ssim
    return loss