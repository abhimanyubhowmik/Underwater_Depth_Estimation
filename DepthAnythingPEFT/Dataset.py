import os
from torch.utils.data import Dataset
from PIL import Image

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
