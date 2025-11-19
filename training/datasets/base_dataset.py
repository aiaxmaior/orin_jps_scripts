"""
Base dataset class for ADAS/DMS training.
Provides common functionality for all dataset loaders.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import json
import cv2
from PIL import Image


class BaseADASDataset(Dataset):
    """Base class for ADAS datasets with common functionality."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transforms=None,
        cache_images: bool = False,
    ):
        """
        Args:
            root_dir: Root directory of dataset
            split: 'train', 'val', or 'test'
            transforms: Albumentations transforms
            cache_images: Cache images in RAM (faster, but memory-intensive)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms
        self.cache_images = cache_images

        self.samples = []
        self.image_cache = {} if cache_images else None

    def __len__(self) -> int:
        return len(self.samples)

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from path with optional caching."""
        if self.cache_images and image_path in self.image_cache:
            return self.image_cache[image_path].copy()

        # Load image
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.cache_images:
            self.image_cache[image_path] = img.copy()

        return img

    def __getitem__(self, idx: int):
        raise NotImplementedError("Subclass must implement __getitem__")

    def get_statistics(self) -> Dict:
        """Compute dataset statistics for normalization."""
        print(f"Computing statistics for {len(self)} images...")

        mean = np.zeros(3)
        std = np.zeros(3)

        for idx in range(min(1000, len(self))):  # Sample 1000 images
            sample = self[idx]
            img = sample['image']

            if isinstance(img, torch.Tensor):
                img = img.numpy()

            mean += img.mean(axis=(1, 2))
            std += img.std(axis=(1, 2))

        mean /= min(1000, len(self))
        std /= min(1000, len(self))

        return {'mean': mean.tolist(), 'std': std.tolist()}

    def visualize_sample(self, idx: int, save_path: Optional[str] = None):
        """Visualize a sample with annotations."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        sample = self[idx]
        img = sample['image']

        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()

        # Denormalize if needed
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)

        # Draw bounding boxes if available
        if 'boxes' in sample and len(sample['boxes']) > 0:
            boxes = sample['boxes']
            labels = sample.get('labels', [0] * len(boxes))

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f"Class {label}",
                       bbox=dict(facecolor='red', alpha=0.5),
                       fontsize=8, color='white')

        ax.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()

        plt.close()


class DetectionDataset(BaseADASDataset):
    """Dataset for object detection tasks."""

    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        split: str = "train",
        transforms=None,
        cache_images: bool = False,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__(root_dir, split, transforms, cache_images)

        self.annotation_file = Path(annotation_file)
        self.class_names = class_names or []

        self._load_annotations()

    def _load_annotations(self):
        """Load annotations from file (to be implemented by subclasses)."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dict with keys:
                - image: Tensor [C, H, W]
                - boxes: Tensor [N, 4] in xyxy format
                - labels: Tensor [N]
                - image_id: int
        """
        sample = self.samples[idx]

        # Load image
        img = self.load_image(sample['image_path'])

        # Load annotations
        boxes = np.array(sample['boxes'], dtype=np.float32)
        labels = np.array(sample['labels'], dtype=np.int64)

        # Apply transformations
        if self.transforms:
            transformed = self.transforms(
                image=img,
                bboxes=boxes,
                labels=labels
            )
            img = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64)

        # Convert to tensors
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).permute(2, 0, 1).float()

        return {
            'image': img,
            'boxes': torch.from_numpy(boxes) if len(boxes) > 0 else torch.zeros((0, 4)),
            'labels': torch.from_numpy(labels) if len(labels) > 0 else torch.zeros((0,), dtype=torch.long),
            'image_id': idx,
        }


class SegmentationDataset(BaseADASDataset):
    """Dataset for semantic segmentation tasks."""

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dict with keys:
                - image: Tensor [C, H, W]
                - mask: Tensor [H, W] with class indices
                - image_id: int
        """
        sample = self.samples[idx]

        # Load image and mask
        img = self.load_image(sample['image_path'])
        mask = cv2.imread(str(sample['mask_path']), cv2.IMREAD_GRAYSCALE)

        # Apply transformations
        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        # Convert to tensors
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).permute(2, 0, 1).float()

        return {
            'image': img,
            'mask': torch.from_numpy(mask).long(),
            'image_id': idx,
        }


class ClassificationDataset(BaseADASDataset):
    """Dataset for classification tasks (e.g., DMS distraction detection)."""

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dict with keys:
                - image: Tensor [C, H, W]
                - label: int
                - image_id: int
        """
        sample = self.samples[idx]

        # Load image
        img = self.load_image(sample['image_path'])
        label = sample['label']

        # Apply transformations
        if self.transforms:
            transformed = self.transforms(image=img)
            img = transformed['image']

        # Convert to tensor
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).permute(2, 0, 1).float()

        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long),
            'image_id': idx,
        }


def get_class_weights(dataset: DetectionDataset) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.

    Args:
        dataset: Detection dataset

    Returns:
        Tensor of class weights (inverse frequency)
    """
    class_counts = {}

    for idx in range(len(dataset)):
        sample = dataset.samples[idx]
        for label in sample['labels']:
            class_counts[label] = class_counts.get(label, 0) + 1

    num_classes = len(dataset.class_names)
    total_samples = sum(class_counts.values())

    weights = torch.zeros(num_classes)
    for cls_id, count in class_counts.items():
        weights[cls_id] = total_samples / (num_classes * count)

    return weights


if __name__ == "__main__":
    # Example usage
    print("Base dataset classes for ADAS/DMS training")
    print("Supported dataset types:")
    print("  - DetectionDataset: Object detection")
    print("  - SegmentationDataset: Semantic segmentation")
    print("  - ClassificationDataset: Image classification")
