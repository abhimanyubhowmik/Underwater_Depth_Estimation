import torchvision.transforms.functional as fn
import numpy as np
import torch
import wandb
from utils import metrices
import statistics

class EvaluationMetric:

    def __init__(self, wandb_logging:bool) -> None:
        self.logging = wandb_logging


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
            gt_depth = gt_depth.cpu().squeeze().numpy()
            metrices_list = []
            # Calculate metrics
            
            # Iterate over batch size
            for i in range(gt_depth.shape[0]):
                 # Create a binary mask that covers the entire depth map : for SSE Loss only
                full_mask = np.ones(gt_depth[i].shape, dtype=bool)
                matrices = metrices.compute_metrics(gt_depth[i],depth_output[i],max_depth_eval=20,
                                                    disp_gt_edges=full_mask)
                # Remove None before computing mean
                if matrices is not None:
                    metrices_list.append(matrices)


            mean_values = {}

            # Iterate over each dictionary in the list
            for metric in metrices_list:
                # Iterate over each key-value pair in the dictionary
                for key, value in metric.items():
                    # If the key is not in the mean_values dictionary, add it with the current value as the first element of a list
                    if key not in mean_values:
                        mean_values[key] = [value]
                    # If the key is already in the mean_values dictionary, append the current value to the list
                    else:
                        mean_values[key].append(value)

            # Calculate the mean for each key
            for key, values in mean_values.items():
                mean_values[key] = statistics.mean(values)
            

            if self.logging == True:
                wandb.log(mean_values)
            return mean_values
        