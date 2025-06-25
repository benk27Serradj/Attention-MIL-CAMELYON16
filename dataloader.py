import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
import cv2

class CamelyonBags(Dataset):
    def __init__(self, image_folder, patch_size, train=True, seed=None):
        """
        Initialize the CamelyonBags dataset.
        
        Args:
            image_folder (str): Path to the folder containing image data.
            patch_size (tuple): Desired patch size (height, width).
            train (bool): Flag to load training data if True, else testing data.
            seed (int, optional): Random seed for reproducibility.
        """
        print("Initializing CamelyonBags...")
        self.image_folder = image_folder
        self.patch_size = patch_size
        self.train = train
        self.seed = seed
        self.image_files = self._load_image_files()
        self.labels = self._generate_labels()
        print(f"Total samples: {len(self)}")

    def _load_image_files(self):
        """
        Load all the image files from the dataset (bags of images).
        
        Returns:
            list: A list of image file paths.
        """
        image_files = []
        dataset_type = 'train' if self.train else 'test'
        for label_folder in ['normal', 'tumor']:
            label_folder_path = os.path.join(self.image_folder, dataset_type, label_folder)
            for filename in os.listdir(label_folder_path):
                if filename.endswith('.tif'):
                    image_files.append(os.path.join(label_folder_path, filename))
        return image_files

    def _generate_labels(self):
        """
        Generate labels for the images based on the folder structure.
        
        Returns:
            list: A list of labels corresponding to each image.
        """
        labels = []
        for image_file in self.image_files:
            if 'normal' in image_file:
                labels.append(0)  # Label 0 for normal
            elif 'tumor' in image_file:
                labels.append(1)  # Label 1 for tumor
        return labels
    
    def __len__(self):
        """
        Return the number of images in the dataset.
        
        Returns:
            int: The total number of images.
        """
        return len(self.image_files)
    
    def _is_tissue(self,patch):

        patch_np = (patch.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        
        r, g, b = patch_np[..., 0], patch_np[..., 1], patch_np[..., 2]

        brightness = (r + g + b) / 3
        if np.mean(brightness) < 40:  # threshold in 0-255 scale
            return False

        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        color_variation = max_rgb - min_rgb
        if np.mean(color_variation) < 2:
            return False

        mask = (r > 120) & (b > 80) & (g < 150)
        return np.sum(mask) > 0


    def is_not_blurry(self, patch, threshold=2000):
        patch_np = (patch.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance >= threshold


    
    def __getitem__(self, idx):
        """
        Fetch a sample (all patches) and corresponding label from the dataset.
        
        Args:
            idx (int): Index of the sample to fetch.
        
        Returns:
            tuple: A tuple (bag, label), where `bag` is a tensor containing all patches,
                and `label` is the corresponding label for the image.
        """
        try:
            print(f"Fetching index: {idx}")
            tiff_path = self.image_files[idx]
            print(f"Opening TIFF file: {tiff_path}")
            with tifffile.TiffFile(tiff_path) as tiff_file:
                print(f"TIFF file opened successfully: {tiff_path}")
                print(f"Number of pages in TIFF: {len(tiff_file.pages)}")
                if len(tiff_file.pages) > 10:
                    page_index = 5
                else:
                    page_index = 0
                
                page = tiff_file.pages[page_index]
                print(f"Selected page {page_index} with shape: {page.shape}")

                height, width = page.shape[:2]
                patch_size = self.patch_size
                if height < patch_size[0] or width < patch_size[1]:
                    raise ValueError(f"Image too small ({height}, {width}) for patch size {patch_size} — file: {tiff_path}")

                image = page.asarray(out='memmap')
                patches = []
                stride_y, stride_x = patch_size  # No overlap

                for y_start in range(0, height - patch_size[0] + 1, stride_y):
                    for x_start in range(0, width - patch_size[1] + 1, stride_x):
                        patch = image[y_start:y_start + patch_size[0], x_start:x_start + patch_size[1]]
                        patch = np.array(patch).astype(np.float32) / 255.0
                        patch = torch.tensor(patch).float()
                        if patch.ndimension() == 3 and patch.shape[2] == 3:
                            patch = patch.permute(2, 0, 1)  # C x H x W
                        else:
                            patch = patch.unsqueeze(0)  # Grayscale: 1 x H x W
                        if self.is_not_blurry(patch) and self._is_tissue(patch):
                            patches.append(patch)

                bag = torch.stack(patches)
                label = self.labels[idx]
                return bag, label

        except Exception as e:
            print(f"⚠️ Error fetching index {idx} ({self.image_files[idx]}): {e}")
            import traceback
            traceback.print_exc()




def mil_collate_fn(batch):
    bags, labels = zip(*batch)  # list of (bag_tensor, label)

    # Find max number of patches in the batch
    max_patches = max(bag.shape[0] for bag in bags)
    c, h, w = bags[0].shape[1:]  # assume all patches same shape

    padded_bags = []
    for bag in bags:
        num_patches = bag.shape[0]
        if num_patches < max_patches:
            # pad with zeros
            pad_size = max_patches - num_patches
            padding = torch.zeros((pad_size, c, h, w), dtype=bag.dtype)
            padded_bag = torch.cat([bag, padding], dim=0)
        else:
            # truncate if too long
            padded_bag = bag[:max_patches]
        padded_bags.append(padded_bag)

    # Stack into a single tensor
    bags_tensor = torch.stack(padded_bags)  # [B, max_patches, C, H, W]
    labels_tensor = torch.tensor(labels)

    return bags_tensor, labels_tensor