
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.transforms.functional as fn
from torch import nn
import numpy as np
import torch
import scipy
import wandb

class EvaluationMetric:

    def __init__(self, wandb_logging:bool) -> None:
        self.logging = wandb_logging

    def scale_offset_invariance_psnr(self, depth_map1, depth_map2):
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


    def scale_offset_invariance_ssim(self, depth_map1, depth_map2):
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


    def compute_metrics(self, input_image, predicted_depth, gt_depth):

        with torch.no_grad():
            img_size = fn.get_image_size(input_image)
            img_size.reverse()
            prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size = img_size,
            mode = "bicubic",
            align_corners=False)
            
            # Convert depth prediction to numpy array and resize to match ground truth depth map size
            depth_output = prediction.squeeze().cpu().numpy()
            gt_depth = gt_depth.cpu().numpy()

            # Handle invalid or unexpected depth values
            depth_output[depth_output <= 0] = 1e-7  # Replace negative or zero values with a small epsilon
            gt_depth = np.squeeze(gt_depth)
            gt_depth[gt_depth <= 0] = 1e-7 
            # Calculate metrics
            #non_zero_mask = depth_output != 0
            absrel = np.mean(np.abs(depth_output - gt_depth) / gt_depth)

            d = np.log(gt_depth + 1e-7) - np.log(depth_output + 1e-7)
            silog = np.mean(np.square(d)) - np.square(np.sum(d))/ np.square(d.size)
            pearson_corr = scipy.stats.pearsonr(depth_output.flatten(), gt_depth.flatten())[0]
            psnr_val = self.scale_offset_invariance_psnr(gt_depth,depth_output)
            ssim_list = []
            for i in range(gt_depth.shape[0]):
                ssim_list.append(self.scale_offset_invariance_ssim(gt_depth[i],depth_output[i]))
            ssim_val = np.mean(np.array(ssim_list))
            

            if self.logging == True:

                wandb.log({"Absolute Relative error (AbsRel)":absrel})
                wandb.log({"Scale Invarience MSE (Logscale)": silog})
                wandb.log({"Pearson Correlation": pearson_corr})
                wandb.log({"PSNR (Scale and offset Invarience)": psnr_val})
                wandb.log({"SSIM (Scale and offset Invarience)": ssim_val})

            metrics = [absrel,silog,pearson_corr,psnr_val,ssim_val]
            
            return metrics
        