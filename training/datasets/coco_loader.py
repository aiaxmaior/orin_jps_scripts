"""
COCO dataset loader for ADAS transfer learning.
Supports standard COCO format with flexible class filtering.
"""

from .base_dataset import DetectionDataset
from pycocotools.coco import COCO
from pathlib import Path
import numpy as np
from typing import List, Optional


class COCODetection(DetectionDataset):
    """
    COCO dataset loader for object detection.

    COCO classes relevant for ADAS:
    - person (1)
    - bicycle (2)
    - car (3)
    - motorcycle (4)
    - bus (6)
    - truck (8)
    - traffic light (10)
    - stop sign (13)
    """

    # ADAS-relevant COCO class IDs
    ADAS_CLASSES = {
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        6: 'bus',
        8: 'truck',
        10: 'traffic_light',
        13: 'stop_sign',
    }

    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        split: str = "train",
        transforms=None,
        cache_images: bool = False,
        filter_classes: Optional[List[int]] = None,
        min_bbox_area: int = 100,
    ):
        """
        Args:
            root_dir: Path to COCO images directory
            annotation_file: Path to COCO annotations JSON
            split: 'train', 'val', or 'test'
            transforms: Albumentations transforms
            cache_images: Cache images in RAM
            filter_classes: List of COCO class IDs to keep (None = all)
            min_bbox_area: Minimum bbox area (width * height) to keep
        """
        self.filter_classes = filter_classes or list(self.ADAS_CLASSES.keys())
        self.min_bbox_area = min_bbox_area

        # Map COCO class IDs to contiguous indices
        self.coco_id_to_label = {
            coco_id: idx for idx, coco_id in enumerate(self.filter_classes)
        }
        self.label_to_coco_id = {
            idx: coco_id for coco_id, idx in self.coco_id_to_label.items()
        }

        # Get class names
        class_names = [
            self.ADAS_CLASSES.get(coco_id, f"class_{coco_id}")
            for coco_id in self.filter_classes
        ]

        super().__init__(
            root_dir=root_dir,
            annotation_file=annotation_file,
            split=split,
            transforms=transforms,
            cache_images=cache_images,
            class_names=class_names,
        )

    def _load_annotations(self):
        """Load COCO annotations."""
        print(f"Loading COCO annotations from {self.annotation_file}...")

        self.coco = COCO(str(self.annotation_file))

        # Get all image IDs
        img_ids = self.coco.getImgIds()
        print(f"Found {len(img_ids)} images in COCO {self.split} set")

        # Filter images that contain relevant classes
        self.samples = []
        skipped_no_annotations = 0
        skipped_small_boxes = 0

        for img_id in img_ids:
            # Get annotations for this image
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.filter_classes)

            if len(ann_ids) == 0:
                skipped_no_annotations += 1
                continue

            anns = self.coco.loadAnns(ann_ids)

            # Load image info
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = self.root_dir / img_info['file_name']

            if not img_path.exists():
                continue

            # Extract boxes and labels
            boxes = []
            labels = []

            for ann in anns:
                # Get bbox in [x, y, w, h] format
                x, y, w, h = ann['bbox']

                # Filter small boxes
                if w * h < self.min_bbox_area:
                    skipped_small_boxes += 1
                    continue

                # Convert to [x1, y1, x2, y2]
                boxes.append([x, y, x + w, y + h])

                # Map COCO class ID to contiguous label
                coco_class_id = ann['category_id']
                label = self.coco_id_to_label[coco_class_id]
                labels.append(label)

            # Skip images with no valid boxes after filtering
            if len(boxes) == 0:
                continue

            self.samples.append({
                'image_path': str(img_path),
                'boxes': boxes,
                'labels': labels,
                'image_id': img_id,
                'width': img_info['width'],
                'height': img_info['height'],
            })

        print(f"Loaded {len(self.samples)} images with annotations")
        print(f"Skipped {skipped_no_annotations} images (no relevant annotations)")
        print(f"Skipped {skipped_small_boxes} small boxes (< {self.min_bbox_area} px²)")

        # Print class distribution
        self._print_class_distribution()

    def _print_class_distribution(self):
        """Print distribution of classes in dataset."""
        class_counts = {idx: 0 for idx in range(len(self.class_names))}

        for sample in self.samples:
            for label in sample['labels']:
                class_counts[label] += 1

        print("\nClass distribution:")
        for idx, count in sorted(class_counts.items()):
            print(f"  {self.class_names[idx]:20s}: {count:6d} instances")

    def get_coco_metrics(self, predictions, iou_threshold=0.5):
        """
        Evaluate predictions using COCO metrics.

        Args:
            predictions: List of dicts with keys:
                - image_id: int
                - boxes: np.array [N, 4] in xyxy format
                - scores: np.array [N]
                - labels: np.array [N]
            iou_threshold: IoU threshold for matching

        Returns:
            Dict with mAP metrics
        """
        from pycocotools.cocoeval import COCOeval

        # Convert predictions to COCO format
        coco_predictions = []
        for pred in predictions:
            image_id = pred['image_id']
            boxes = pred['boxes']  # [N, 4] xyxy
            scores = pred['scores']  # [N]
            labels = pred['labels']  # [N]

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1

                # Convert label back to COCO class ID
                coco_class_id = self.label_to_coco_id[label]

                coco_predictions.append({
                    'image_id': int(image_id),
                    'category_id': int(coco_class_id),
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(score),
                })

        # Run COCO evaluation
        coco_dt = self.coco.loadRes(coco_predictions)
        coco_eval = COCOeval(self.coco, coco_dt, 'bbox')

        # Filter to only evaluate on relevant classes
        coco_eval.params.catIds = self.filter_classes

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        metrics = {
            'mAP@0.5:0.95': coco_eval.stats[0],
            'mAP@0.5': coco_eval.stats[1],
            'mAP@0.75': coco_eval.stats[2],
            'mAP_small': coco_eval.stats[3],
            'mAP_medium': coco_eval.stats[4],
            'mAP_large': coco_eval.stats[5],
        }

        return metrics


def create_adas_coco_loader(
    root_dir: str,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    transforms=None,
):
    """
    Create COCO dataloader for ADAS training.

    Args:
        root_dir: Path to COCO dataset root
        split: 'train' or 'val'
        batch_size: Batch size
        num_workers: Number of dataloader workers
        transforms: Albumentations transforms

    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    import os

    # Determine annotation file
    if split == "train":
        ann_file = os.path.join(root_dir, "annotations", "instances_train2017.json")
        img_dir = os.path.join(root_dir, "train2017")
    else:
        ann_file = os.path.join(root_dir, "annotations", "instances_val2017.json")
        img_dir = os.path.join(root_dir, "val2017")

    # Create dataset
    dataset = COCODetection(
        root_dir=img_dir,
        annotation_file=ann_file,
        split=split,
        transforms=transforms,
        cache_images=False,  # Don't cache COCO (too large)
    )

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=True,
    )

    return loader


def detection_collate_fn(batch):
    """
    Custom collate function for detection datasets.
    Handles variable number of boxes per image.
    """
    images = []
    boxes = []
    labels = []
    image_ids = []

    for sample in batch:
        images.append(sample['image'])
        boxes.append(sample['boxes'])
        labels.append(sample['labels'])
        image_ids.append(sample['image_id'])

    import torch
    images = torch.stack(images, dim=0)

    return {
        'images': images,
        'boxes': boxes,  # List of tensors (variable length)
        'labels': labels,  # List of tensors (variable length)
        'image_ids': image_ids,
    }


if __name__ == "__main__":
    # Test COCO loader
    import sys

    if len(sys.argv) < 2:
        print("Usage: python coco_loader.py <path_to_coco_dataset>")
        print("Example: python coco_loader.py ~/datasets/coco")
        sys.exit(1)

    coco_root = sys.argv[1]

    # Create dataset
    dataset = COCODetection(
        root_dir=f"{coco_root}/val2017",
        annotation_file=f"{coco_root}/annotations/instances_val2017.json",
        split="val",
    )

    print(f"\nDataset size: {len(dataset)} images")
    print(f"Classes: {dataset.class_names}")

    # Visualize first sample
    dataset.visualize_sample(0, save_path="coco_sample.jpg")
