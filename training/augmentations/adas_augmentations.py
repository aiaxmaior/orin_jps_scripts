"""
Augmentation pipelines for ADAS/DMS training.
Includes weather, lighting, and geometric augmentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


def get_adas_train_transforms(input_size=640, aggressive=False):
    """
    Training augmentations for ADAS (road-facing camera).

    Args:
        input_size: Target input size (square)
        aggressive: Use aggressive augmentations for robustness

    Returns:
        Albumentations Compose transform
    """
    if aggressive:
        # Aggressive augmentations for varying conditions
        transforms = [
            A.Resize(input_size, input_size),

            # Weather augmentations (high probability)
            A.RandomRain(
                slant_lower=-10, slant_upper=10,
                drop_length=20, drop_width=1,
                brightness_coefficient=0.9,
                p=0.4
            ),
            A.RandomFog(
                fog_coef_lower=0.2, fog_coef_upper=0.5,
                alpha_coef=0.1,
                p=0.3
            ),
            A.RandomSnow(
                snow_point_lower=0.1, snow_point_upper=0.3,
                brightness_coeff=2.5,
                p=0.2
            ),

            # Lighting augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.4, contrast_limit=0.4, p=0.6
            ),
            A.RandomGamma(gamma_limit=(50, 200), p=0.4),
            A.RandomToneCurve(scale=0.3, p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.4
            ),

            # Shadows (common in real driving)
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1, num_shadows_upper=2,
                shadow_dimension=5,
                p=0.3
            ),

            # Blur/noise (camera artifacts)
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
            ], p=0.4),

            A.GaussNoise(var_limit=(10, 70), p=0.3),

            # Compression artifacts
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),

            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.3, rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT, p=0.5
            ),

            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    else:
        # Standard augmentations
        transforms = [
            A.Resize(input_size, input_size),

            # Moderate weather
            A.RandomRain(p=0.2),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15),

            # Lighting
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),

            # Blur
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
            ], p=0.2),

            # Geometric
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.3
            ),

            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_area=100,  # Filter boxes smaller than 100 px²
            min_visibility=0.3,  # Keep boxes with >30% visible
        )
    )


def get_adas_val_transforms(input_size=640):
    """Validation transforms (no augmentation, only resize+normalize)."""
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def get_dms_train_transforms(input_size=224):
    """
    Training augmentations for DMS (driver-facing camera).

    DMS operates in controlled environment (cabin), so augmentations
    focus on lighting variation and driver appearance variation.
    """
    return A.Compose([
        A.Resize(input_size, input_size),

        # Lighting (cabin lighting varies)
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.6
        ),
        A.RandomGamma(gamma_limit=(70, 150), p=0.4),

        # IR illumination can cause lens flare
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_lower=0, angle_upper=1,
            num_flare_circles_lower=1, num_flare_circles_upper=3,
            src_radius=50,
            p=0.2
        ),

        # Blur (motion, low light)
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        A.GaussNoise(var_limit=(5, 30), p=0.2),

        # Geometric (driver position varies)
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5
        ),

        # Affine (perspective changes)
        A.Perspective(scale=(0.05, 0.1), p=0.3),

        # Normalize
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_dms_val_transforms(input_size=224):
    """Validation transforms for DMS."""
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_night_specific_transforms(input_size=640):
    """
    Specialized augmentations for night driving.
    Use this for fine-tuning on night-specific datasets.
    """
    return A.Compose([
        A.Resize(input_size, input_size),

        # Low light simulation
        A.RandomBrightnessContrast(
            brightness_limit=(-0.5, -0.2),  # Darken
            contrast_limit=0.2,
            p=0.8
        ),

        # Headlight glare
        A.RandomSunFlare(
            flare_roi=(0, 0.5, 1, 1),  # Lower half (headlights)
            angle_lower=0, angle_upper=1,
            num_flare_circles_lower=2, num_flare_circles_upper=4,
            src_radius=100,
            p=0.5
        ),

        # Motion blur (longer exposure at night)
        A.MotionBlur(blur_limit=7, p=0.4),

        # Noise (high ISO at night)
        A.GaussNoise(var_limit=(20, 80), p=0.5),

        # Standard geometric
        A.HorizontalFlip(p=0.5),

        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.2,  # Lower threshold for night (harder to see)
    ))


def get_fisheye_transforms(input_size=640, distortion_scale=0.2):
    """
    Augmentations for fisheye lens cameras.
    Includes custom distortion simulation.
    """
    return A.Compose([
        A.Resize(input_size, input_size),

        # Simulate varying fisheye distortion
        A.OpticalDistortion(
            distort_limit=distortion_scale, shift_limit=0.05, p=0.5
        ),

        # Standard ADAS augmentations
        A.RandomRain(p=0.2),
        A.RandomFog(p=0.15),
        A.RandomBrightnessContrast(p=0.5),

        A.HorizontalFlip(p=0.5),

        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_area=50,  # Fisheye makes objects smaller at edges
    ))


def visualize_augmentations(
    image_path: str,
    bbox=None,
    output_dir="augmentation_samples",
    num_samples=9
):
    """
    Visualize augmentation pipeline.

    Args:
        image_path: Path to sample image
        bbox: Optional bounding box [x1, y1, x2, y2]
        output_dir: Where to save visualizations
        num_samples: Number of augmented samples to generate
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    import cv2

    Path(output_dir).mkdir(exist_ok=True)

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get transform
    transform = get_adas_train_transforms(aggressive=True)

    # Generate samples
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for idx in range(num_samples):
        if bbox is not None:
            transformed = transform(image=image, bboxes=[bbox], labels=[0])
            aug_image = transformed['image'].permute(1, 2, 0).numpy()

            # Denormalize
            aug_image = aug_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            aug_image = np.clip(aug_image, 0, 1)

            # Draw bbox if present
            if len(transformed['bboxes']) > 0:
                bbox_aug = transformed['bboxes'][0]
                x1, y1, x2, y2 = [int(coord) for coord in bbox_aug]
                import matplotlib.patches as patches
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                axes[idx].add_patch(rect)
        else:
            transformed = transform(image=image)
            aug_image = transformed['image'].permute(1, 2, 0).numpy()
            aug_image = aug_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            aug_image = np.clip(aug_image, 0, 1)

        axes[idx].imshow(aug_image)
        axes[idx].axis('off')
        axes[idx].set_title(f"Sample {idx+1}")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/augmentations_grid.jpg", dpi=150, bbox_inches='tight')
    print(f"Saved augmentation samples to {output_dir}/augmentations_grid.jpg")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python adas_augmentations.py <path_to_image>")
        print("Generates 9 augmented samples for visualization")
        sys.exit(1)

    visualize_augmentations(sys.argv[1])
