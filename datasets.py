import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size, scale=4, transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_images = sorted(os.listdir(hr_dir))
        self.lr_images = sorted(os.listdir(lr_dir))
        self.transform = transform
        self.patch_size = patch_size # lr size
        self.scale = scale

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])

        # print(hr_image_path)
        
        hr_image = np.array(Image.open(hr_image_path).convert('RGB'))
        lr_image = np.array(Image.open(lr_image_path).convert('RGB'))
        # hr_image = np.load(hr_image_path, encoding='bytes', allow_pickle=True)
        # lr_image = np.load(lr_image_path, encoding='bytes', allow_pickle=True)
        if self.patch_size != 0:
            width = lr_image.shape[1]
            height = lr_image.shape[0]
            crop_x = random.randint(0, height - self.patch_size)
            crop_y = random.randint(0, width - self.patch_size)
            hr_image = hr_image[crop_x * self.scale: (crop_x + self.patch_size) * self.scale, crop_y * self.scale: (crop_y + self.patch_size) * self.scale, :]
            lr_image = lr_image[crop_x: crop_x + self.patch_size, crop_y: crop_y + self.patch_size, :]

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return {'HR': hr_image, 'LR': lr_image, 'HR_path': hr_image_path, 'LR_path': lr_image_path}
    


class ImageDataseLoss(Dataset):
    def __init__(self, hr_dir, lr_dir, b0_dir, patch_size=0, scale=4, transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_images = sorted(os.listdir(hr_dir))
        self.lr_images = sorted(os.listdir(lr_dir))
        self.b0_images = sorted(os.listdir(b0_dir))
        self.transform = transform
        self.patch_size = patch_size # lr size
        self.scale = scale

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])
        b0_image_path = os.path.join(self.b0_dir, self.b0_images[idx])

        # print(hr_image_path)
        
        hr_image = np.array(Image.open(hr_image_path).convert('RGB'))
        lr_image = np.array(Image.open(lr_image_path).convert('RGB'))
        b0_image = np.array(Image.open(b0_image_path).convert('RGB'))
        # hr_image = np.load(hr_image_path, encoding='bytes', allow_pickle=True)
        # lr_image = np.load(lr_image_path, encoding='bytes', allow_pickle=True)
        if self.patch_size != 0:
            width = lr_image.shape[1]
            height = lr_image.shape[0]
            crop_x = random.randint(0, height - self.patch_size)
            crop_y = random.randint(0, width - self.patch_size)
            hr_image = hr_image[crop_x * self.scale: (crop_x + self.patch_size) * self.scale, crop_y * self.scale: (crop_y + self.patch_size) * self.scale, :]
            b0_image = b0_image[crop_x * self.scale: (crop_x + self.patch_size) * self.scale, crop_y * self.scale: (crop_y + self.patch_size) * self.scale, :]
            lr_image = lr_image[crop_x: crop_x + self.patch_size, crop_y: crop_y + self.patch_size, :]

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
            b0_image = self.transform(b0_image)

        return {'HR': hr_image, 'LR': lr_image, 'HR_path': hr_image_path, 'LR_path': lr_image_path, 'B0': b0_image, 'B0_path': b0_image_path}
    

