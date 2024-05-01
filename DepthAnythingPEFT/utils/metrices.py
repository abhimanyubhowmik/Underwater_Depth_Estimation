import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# Scale invarient PSNR & SSIM
def scale_offset_invariance_psnr(depth_map1, depth_map2):

    # Check if depth_map1 is empty
    if depth_map1.size == 0:
        raise ValueError("depth_map1 is empty.")
    
    # Check if depth_map2 is empty
    if depth_map2.size == 0:
        raise ValueError("depth_map2 is empty.")
    
    # Calculate the mean value of both depth maps
    mean1 = np.mean(depth_map1)
    mean2 = np.mean(depth_map2)

    # Calculate the scale factor
    scale_factor = mean1 / mean2

    # Adjust the second depth map by the scale factor
    depth_map2_scaled = depth_map2 * scale_factor

    # Calculate the offset
    offset = mean1 - (scale_factor * mean2)

    # Adjust the second depth map by the offset
    depth_map2_adjusted = depth_map2_scaled + offset

    val_min = depth_map1.min()
    val_range = depth_map1.max() - val_min + 1e-7

    depth_map1_normed = (depth_map1 - val_min) / val_range
    # apply identical normalization to the denoised image (important!)
    depth_map2_adjusted_normed = (depth_map2_adjusted - val_min) / val_range

    #Calculate the PSNR
    psnr_val = psnr(depth_map1_normed, depth_map2_adjusted_normed, data_range=1.0)

    return psnr_val


def scale_offset_invariance_ssim( depth_map1, depth_map2):

    # Check if depth_map1 is empty
    if depth_map1.size == 0:
        raise ValueError("depth_map1 is empty.")
    
    # Check if depth_map2 is empty
    if depth_map2.size == 0:
        raise ValueError("depth_map2 is empty.")

    # Calculate the mean value of both depth maps
    mean1 = np.mean(depth_map1)
    mean2 = np.mean(depth_map2)

    # Calculate the scale factor
    scale_factor = mean1 / mean2

    # Adjust the second depth map by the scale factor
    depth_map2_scaled = depth_map2 * scale_factor

    # Calculate the offset
    offset = mean1 - (scale_factor * mean2)

    # Adjust the second depth map by the offset
    depth_map2_adjusted = depth_map2_scaled + offset

    val_min = depth_map1.min()
    val_range = depth_map1.max() - val_min + 1e-7

    depth_map1_normed = (depth_map1 - val_min) / val_range
    # apply identical normalization to the denoised image (important!)
    depth_map2_adjusted_normed = (depth_map2_adjusted - val_min) / val_range

    # Calculate the SSIM
    ssim_value, _ = ssim(depth_map1_normed, depth_map2_adjusted_normed, full=True, data_range=1.0)

    return ssim_value

# Soft Edge Error
def shift_2d_replace(data, dx, dy, constant=False):
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D array, but received an array with {data.ndim} dimensions.")
    
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data

def soft_edge_error(pred, gt, radius=1):

    # print("Shape of gt:", gt.shape)
    # print("Shape of pred:", pred.shape)
    abs_diff=[]
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            abs_diff.append(np.abs(shift_2d_replace(gt, i, j, 0) - pred))
    return np.minimum.reduce(abs_diff)


def compute_errors(gt, pred):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
            'psnr': Scale invariant psnr
            'ssim' : Scale invariant ssim
            'pearson_corr': Pearson Correlation
    """

    if gt.size != 0 and pred.size != 0:
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt ** 2)

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        err = np.log(pred) - np.log(gt)
        silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2)

        log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
        
        psnr = scale_offset_invariance_psnr(gt,pred)
        ssim = scale_offset_invariance_ssim(gt,pred)
        pearson_corr = scipy.stats.pearsonr(gt.flatten(), pred.flatten())[0]

        matrices = dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel, psnr = psnr,ssim = ssim,pearson_corr = pearson_corr)


    else:
        # If the masked array is empty
        matrices = None
    
    return matrices

def compute_metrics(gt, pred, min_depth_eval=0.1, max_depth_eval=10, disp_gt_edges=None):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. 
    Refer to compute_errors for more details on metrics.
    """
    pred = pred.squeeze()
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval

    gt_depth = gt.squeeze()
    valid_mask = np.logical_and(gt_depth > min_depth_eval, gt_depth < max_depth_eval)
    metrics = compute_errors(gt_depth[valid_mask], pred[valid_mask])

    # For SSE calcualtion of non NaN metrices
    if disp_gt_edges is not None and metrics is not None:
            
            edges = disp_gt_edges.squeeze()
            mask = valid_mask.squeeze() # squeeze
            mask = np.logical_and(mask, edges)
            see_depth = torch.tensor([0])
            if mask.sum() > 0:
                see_depth_map = soft_edge_error(pred, gt_depth)
                see_depth_map_valid = see_depth_map[mask]
                see_depth = see_depth_map_valid.mean()
            metrics['see'] = see_depth
    
    return metrics
