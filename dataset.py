import albumentations
import torch
import numpy as np
from PIL import Image


class Classification:
    def __init__(self, image_paths, targets, resize=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.aug = albumentations.Compose([
            albumentations.Normalize(always_apply=True)
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        target = self.targets[idx]

        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)

        image = np.array(image)
        augmentation = self.aug(image=image)
        image = augmentation['image']

        image = np.transpose(image, (2,1,0)).astype(np.float32)

        return {
            'images': torch.tensor(image, dtype=torch.float),
            'targets': torch.tensor(target, dtype=torch.long)
        }