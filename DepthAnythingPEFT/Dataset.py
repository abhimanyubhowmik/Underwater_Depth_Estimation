import os
from torch.utils.data import Dataset
from PIL import Image
import torch 

class FlSeaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.tif'):
                    if 'depth' in root:
                        depth_path = os.path.join(root, file)
                        image_path = os.path.join(root.replace('depth', 'imgs'), file.replace('_SeaErra_abs_depth.tif', '.tiff'))
                        data.append((image_path, depth_path))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, depth_path = self.data[idx]
        image = Image.open(image_path)
        depth = Image.open(depth_path)
        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)
        return image, depth

class USODDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.images = []
            self.depths = []

            # Loop through each subfolder
            for subfolder in os.listdir(root_dir):
                subfolder_path = os.path.join(root_dir, subfolder)
                if os.path.isdir(subfolder_path):
                    # Check if the subfolder contains 'depth' and 'RGB' folders
                    if all([os.path.isdir(os.path.join(subfolder_path, folder)) for folder in ['depth', 'RGB']]):
                        # Load images and depths from the subfolders
                        images_folder = os.path.join(subfolder_path, 'RGB')
                        depth_folder = os.path.join(subfolder_path, 'depth')
                        images_list = sorted(os.listdir(images_folder))
                        depth_list = sorted(os.listdir(depth_folder))

                        for img_name, depth_name in zip(images_list, depth_list):
                            img_path = os.path.join(images_folder, img_name)
                            depth_path = os.path.join(depth_folder, depth_name)
                            self.images.append(img_path)
                            self.depths.append(depth_path)

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = self.images[idx]
            depth_path = self.depths[idx]

            image = Image.open(img_path)
            depth = Image.open(depth_path)

            if self.transform:
                image = self.transform(image)
                depth = self.transform(depth)

            return image, depth
        
class VAROSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                    if 'A' in root:
                        image_path = os.path.join(root, file)
                        depth_path = os.path.join(root.replace('A', 'D'), file.replace('A', 'D'))
                        data.append((image_path, depth_path))
        return data

    def __len__(self):
        return len(self.data)
    
    def min_max_normalize(self,image):
    # Get the minimum and maximum pixel values
        min_val = torch.min(image)
        max_val = torch.max(image)

        # Normalize the image
        normalized_image = (image - min_val) / (max_val - min_val)
        return normalized_image

    def __getitem__(self, idx):
        image_path, depth_path = self.data[idx]
        image = Image.open(image_path)
        depth = Image.open(depth_path)
        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)
            depth = self.min_max_normalize(depth)
            depth = 1 - depth   
        return image, depth
