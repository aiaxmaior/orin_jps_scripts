# Dataset Guide for ADAS/DMS Transfer Learning

## Overview

This guide covers publicly available datasets for training ADAS and DMS models, along with preparation strategies for Jetson Orin edge deployment.

---

## 1. ADAS Datasets (Road-Facing Camera)

### 1.1 Object Detection Datasets

#### **COCO (Common Objects in Context)**
- **URL:** https://cocodataset.org/
- **Size:** 330K images, 1.5M object instances
- **Classes:** 80 object categories (person, car, truck, bicycle, etc.)
- **Annotations:** Bounding boxes, segmentation masks, keypoints
- **Conditions:** Mostly daytime, indoor/outdoor, varied weather
- **Use Case:** Pre-training, general object detection
- **Download:**
  ```bash
  # Train/Val 2017 (18GB)
  wget http://images.cocodataset.org/zips/train2017.zip
  wget http://images.cocodataset.org/zips/val2017.zip
  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  ```

**Pros:** Massive scale, high quality, standard benchmark
**Cons:** Not automotive-specific, limited weather variation

---

#### **BDD100K (Berkeley DeepDrive)**
- **URL:** https://bdd-data.berkeley.edu/
- **Size:** 100K videos (40 seconds each), 100K images
- **Classes:** 10 object categories (car, pedestrian, truck, bus, motorcycle, bicycle, traffic light, traffic sign, rider, train)
- **Annotations:**
  - Bounding boxes (2D, 3D)
  - Drivable area segmentation
  - Lane markings
  - Instance segmentation
- **Conditions:**
  - Time: day, night, dawn/dusk
  - Weather: clear, rainy, cloudy, foggy, snowy
  - Scene: city, highway, residential, tunnel
- **Use Case:** **PRIMARY DATASET FOR ADAS**
- **Download:** Requires registration at https://bdd-data.berkeley.edu/login.html

**Pros:** Real-world driving, diverse conditions, comprehensive annotations
**Cons:** Large download (1.8TB for all), requires account

**Subsets for Quick Start:**
```bash
# BDD100K Detection (12GB - manageable for Orin training)
bdd100k_images_100k_train.zip       # 70K images
bdd100k_images_100k_val.zip         # 10K images
bdd100k_labels_detection.json       # Bounding boxes
```

---

#### **KITTI (Karlsruhe Institute of Technology)**
- **URL:** http://www.cvlibs.net/datasets/kitti/
- **Size:** 7,481 training images, 7,518 test images
- **Classes:** Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram
- **Annotations:**
  - 2D bounding boxes
  - 3D bounding boxes with orientation
  - Depth maps (LiDAR)
  - Stereo images
- **Conditions:** Urban/highway, daytime only, Germany
- **Use Case:** 3D object detection, depth estimation
- **Download:**
  ```bash
  # Object Detection (12GB)
  wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
  wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
  ```

**Pros:** 3D annotations, depth data, academic standard
**Cons:** Limited diversity (Germany only), daytime only, small size

---

#### **Waymo Open Dataset**
- **URL:** https://waymo.com/open/
- **Size:** 1,150 scenes, 200K frames
- **Classes:** Vehicle, Pedestrian, Cyclist, Sign
- **Annotations:**
  - 2D/3D bounding boxes
  - LiDAR point clouds
  - Camera images (5 cameras)
- **Conditions:** Day/night, varied weather, US roads
- **Use Case:** Multi-sensor fusion, 3D detection
- **Download:** Requires Google Cloud account, TFRecord format

**Pros:** High quality, multi-sensor, industry-grade
**Cons:** Very large (1TB+), requires Google Cloud, complex format

---

#### **nuScenes**
- **URL:** https://www.nuscenes.org/
- **Size:** 1,000 scenes, 1.4M camera images
- **Classes:** 23 object categories
- **Annotations:** 3D bounding boxes, tracking IDs, maps
- **Conditions:** Day/night, rain, Boston/Singapore
- **Use Case:** 3D detection, tracking, prediction
- **Download:** Requires registration, ~400GB

**Pros:** Rich annotations, tracking data, map context
**Cons:** Large download, complex data structure

---

### 1.2 Segmentation Datasets

#### **Cityscapes**
- **URL:** https://www.cityscapes-dataset.com/
- **Size:** 5,000 fine-annotated images, 20,000 coarse
- **Classes:** 30 categories (road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle)
- **Annotations:** Dense pixel-level segmentation
- **Conditions:** Urban, daytime, clear weather, 50 European cities
- **Use Case:** Lane detection, drivable area segmentation
- **Download:** Requires registration

**Pros:** High-quality pixel-level labels, standard benchmark
**Cons:** Limited to urban scenes, daytime only, Europe only

---

#### **Mapillary Vistas**
- **URL:** https://www.mapillary.com/dataset/vistas
- **Size:** 25,000 images
- **Classes:** 66 object categories, 28 stuff categories
- **Annotations:** Dense segmentation
- **Conditions:** Worldwide, varied weather/lighting
- **Use Case:** Universal segmentation
- **Download:** Requires registration, 15GB

**Pros:** Global diversity, high variety
**Cons:** Inconsistent quality (crowdsourced)

---

### 1.3 Lane Detection Datasets

#### **TuSimple Lane Detection**
- **URL:** https://github.com/TuSimple/tusimple-benchmark
- **Size:** 6,408 images
- **Annotations:** Lane line pixel coordinates (JSON)
- **Conditions:** US highways, daytime, good weather
- **Use Case:** Lane detection training/evaluation
- **Download:** Direct download, ~10GB

**Pros:** Focused on lanes, simple format
**Cons:** Limited diversity

---

#### **CULane**
- **URL:** https://xingangpan.github.io/projects/CULane.html
- **Size:** 133,235 images
- **Annotations:** Lane markings with 9 scenarios (normal, crowded, dazzle, shadow, no line, arrow, curve, cross, night)
- **Conditions:** Very diverse (urban/highway, day/night)
- **Use Case:** Robust lane detection
- **Download:** ~20GB

**Pros:** Large scale, diverse conditions, challenging
**Cons:** Annotation quality varies

---

### 1.4 Depth Estimation Datasets

#### **KITTI Depth**
- **URL:** http://www.cvlibs.net/datasets/kitti/eval_depth.php
- **Size:** 93K depth maps
- **Annotations:** LiDAR-projected depth (semi-dense)
- **Use Case:** Monocular depth estimation
- **Download:** ~14GB

---

### 1.5 Adverse Weather Datasets

#### **DAWN (Detection in Adverse Weather Nature)**
- **URL:** https://people.ee.ethz.ch/~csakarid/DAWN/
- **Size:** 1,000 images
- **Annotations:** Bounding boxes
- **Conditions:** Fog, snow, rain, sandstorm
- **Use Case:** Weather robustness testing
- **Download:** ~2GB

---

#### **ACDC (Adverse Conditions Dataset with Correspondences)**
- **URL:** https://acdc.vision.ee.ethz.ch/
- **Size:** 4,006 images
- **Annotations:** Segmentation, depth
- **Conditions:** Fog, rain, snow, night
- **Use Case:** Domain adaptation for weather
- **Download:** Requires registration

---

### 1.6 Night Driving Datasets

#### **ExDark (Exclusively Dark)**
- **URL:** https://github.com/cs-chan/Exclusively-Dark-Image-Dataset
- **Size:** 7,363 images
- **Classes:** 12 object categories
- **Conditions:** Low-light (indoor/outdoor)
- **Use Case:** Night detection
- **Download:** ~1GB

---

## 2. DMS Datasets (Driver-Facing Camera)

### 2.1 Face Detection

#### **WIDER FACE**
- **URL:** http://shuoyang1213.me/WIDERFACE/
- **Size:** 32,203 images, 393,703 faces
- **Annotations:** Bounding boxes
- **Conditions:** Varied scale, pose, occlusion
- **Use Case:** Face detection pre-training
- **Download:** ~3GB

---

### 2.2 Gaze Estimation

#### **MPIIGaze**
- **URL:** https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild
- **Size:** 213,659 images, 15 participants
- **Annotations:** Gaze direction (yaw, pitch)
- **Conditions:** In-the-wild laptop usage
- **Use Case:** Gaze estimation
- **Download:** ~7GB

---

#### **Columbia Gaze**
- **URL:** http://www.cs.columbia.edu/CAVE/databases/columbia_gaze/
- **Size:** 5,880 images, 56 people
- **Annotations:** Head pose, gaze direction
- **Conditions:** Controlled lab
- **Use Case:** Gaze baseline
- **Download:** ~1GB

---

### 2.3 Driver Drowsiness & Distraction

#### **NTHU-DDD (Driver Drowsiness Detection)**
- **URL:** http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/
- **Size:** 2,500 video clips, 28 drivers
- **Annotations:** Drowsy/alert labels, yawning, facial landmarks
- **Conditions:** Simulated driving
- **Use Case:** Drowsiness detection
- **Download:** Requires request

---

#### **State Farm Distracted Driver Detection**
- **URL:** https://www.kaggle.com/c/state-farm-distracted-driver-detection
- **Size:** 22,424 images
- **Classes:** 10 distraction types (safe driving, texting, phone call, adjusting radio, drinking, reaching behind, hair/makeup, talking to passenger)
- **Annotations:** Class labels
- **Use Case:** Distraction classification
- **Download:** ~2GB via Kaggle

---

#### **DMD (Driver Monitoring Dataset)**
- **URL:** https://dmd.vicomtech.org/
- **Size:** 40 hours video, 37 drivers
- **Annotations:** Action labels, gaze, head pose
- **Conditions:** Real driving
- **Use Case:** Comprehensive DMS
- **Download:** Requires request

---

## 3. Synthetic Datasets

### 3.1 CARLA Simulator

- **URL:** https://carla.org/
- **Type:** Driving simulator (Unreal Engine)
- **Capabilities:**
  - Generate unlimited images
  - Perfect ground truth (bboxes, segmentation, depth)
  - Control weather, lighting, traffic
- **Use Case:** Data augmentation, RL training
- **Setup:**
  ```bash
  # Install CARLA
  wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.15.tar.gz
  tar -xzf CARLA_0.9.15.tar.gz
  cd CARLA_0.9.15
  ./CarlaUE4.sh
  ```

**Pros:** Infinite data, perfect labels, controllable
**Cons:** Sim-to-real gap, requires GPU for generation

---

### 3.2 GTA V Dataset

- **URL:** https://download.visinf.tu-darmstadt.de/data/from_games/
- **Size:** 24,966 frames
- **Annotations:** Dense segmentation, depth
- **Use Case:** Cheap pre-training data
- **Download:** ~28GB

**Pros:** Realistic graphics, free annotations
**Cons:** Sim-to-real gap, limited diversity

---

## 4. Recommended Dataset Combinations

### 4.1 Minimal Setup (Quick Start, <50GB)

**ADAS:**
- **BDD100K Detection subset:** 10K images (validation only) - 1.2GB
- **COCO 2017 Val:** 5K images (general pre-training) - 1GB
- **TuSimple Lanes:** 6.4K images - 10GB

**DMS:**
- **State Farm Distracted Driver:** 22K images - 2GB
- **WIDER FACE:** 32K images - 3GB

**Total:** ~17GB

**Training strategy:**
1. Pre-train on COCO (general object detection)
2. Fine-tune on BDD100K (automotive domain)
3. DMS: Train from scratch on State Farm + WIDER

---

### 4.2 Recommended Setup (Production, <200GB)

**ADAS:**
- **COCO 2017 Train+Val:** 330K images - 25GB
- **BDD100K Detection:** 100K images - 12GB
- **Cityscapes:** 5K images - 11GB (for segmentation)
- **CULane:** 133K images - 20GB (for lanes)
- **KITTI:** 14K images - 12GB (for depth)
- **DAWN:** 1K images - 2GB (adverse weather)

**DMS:**
- **State Farm:** 22K images - 2GB
- **WIDER FACE:** 32K images - 3GB
- **MPIIGaze:** 213K images - 7GB
- **DMD (if accessible):** 40h video - ~50GB

**Total:** ~144GB

**Training strategy:**
1. Pre-train backbone on COCO/ImageNet
2. Multi-task learning: BDD100K (detection) + Cityscapes (segmentation) + CULane (lanes)
3. Fine-tune on KITTI for depth awareness
4. Fine-tune on DAWN for weather robustness
5. DMS: Multi-task (face + gaze + distraction)

---

### 4.3 Full Setup (Research, <1TB)

Add to Recommended:
- **Waymo Open Dataset:** 200K frames - 1TB
- **nuScenes:** 1.4M images - 400GB
- **Mapillary Vistas:** 25K images - 15GB
- **CARLA synthetic:** 100K generated - 50GB

**Total:** ~1.5TB

---

## 5. Dataset Preparation Pipeline

### 5.1 Download Script

```bash
#!/bin/bash
# download_datasets.sh

# Create dataset directory
mkdir -p ~/datasets
cd ~/datasets

# COCO (25GB)
echo "Downloading COCO..."
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip && unzip val2017.zip && unzip annotations_trainval2017.zip

# BDD100K (requires manual download with account)
echo "Download BDD100K manually from https://bdd-data.berkeley.edu/"

# TuSimple Lanes (10GB)
echo "Downloading TuSimple..."
wget https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/train_set.zip
wget https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_set.zip
unzip train_set.zip && unzip test_set.zip

# State Farm (2GB, Kaggle)
echo "Download State Farm from Kaggle: kaggle competitions download -c state-farm-distracted-driver-detection"

# WIDER FACE (3GB)
echo "Downloading WIDER FACE..."
wget https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip
wget https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip
unzip WIDER_train.zip && unzip WIDER_val.zip

echo "Dataset download complete!"
```

### 5.2 Data Structure

```
~/datasets/
├── coco/
│   ├── train2017/          # 118K images
│   ├── val2017/            # 5K images
│   └── annotations/
│       ├── instances_train2017.json
│       └── instances_val2017.json
│
├── bdd100k/
│   ├── images/
│   │   ├── 100k/
│   │   │   ├── train/      # 70K images
│   │   │   └── val/        # 10K images
│   └── labels/
│       └── det_20/
│           ├── det_train.json
│           └── det_val.json
│
├── tusimple_lanes/
│   ├── train_set/
│   └── test_set/
│
├── state_farm/
│   ├── train/
│   │   ├── c0/             # Safe driving
│   │   ├── c1/             # Texting right
│   │   └── ...
│   └── test/
│
└── wider_face/
    ├── WIDER_train/
    │   └── images/
    └── WIDER_val/
        └── images/
```

### 5.3 Dataset Loader (PyTorch)

```python
# datasets/coco_loader.py

from pycocotools.coco import COCO
from torch.utils.data import Dataset
from PIL import Image
import os

class COCODetection(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Load image
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        # Extract boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(ann['category_id'])

        target = {'boxes': boxes, 'labels': labels}

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
```

### 5.4 Data Augmentation

```python
# datasets/augmentations.py

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return A.Compose([
        # Resize to target input size
        A.Resize(640, 640),

        # Weather augmentations (critical for ADAS)
        A.RandomRain(p=0.3),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
        A.RandomSnow(p=0.1),

        # Lighting augmentations
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.RandomGamma(gamma_limit=(50, 150), p=0.3),
        A.RandomToneCurve(p=0.2),

        # Blur/noise (camera artifacts)
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GlassBlur(),
        ], p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.2),

        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),

        # Normalize and convert to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_val_transforms():
    return A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
```

---

## 6. Custom Dataset Collection

### 6.1 When to Collect Custom Data

You need custom data when:
- Operating in unique geographic region (different signage, road markings)
- Using fisheye lens (different distortion profile)
- Targeting specific vehicle mounting position
- Addressing unique edge cases (local traffic patterns)

### 6.2 Minimum Custom Dataset Size

**For fine-tuning pre-trained model:**
- **Minimum:** 200-500 images per new condition/class
- **Recommended:** 1,000-2,000 images
- **With augmentation:** 10x multiplier (effectively 10,000-20,000)

**For training from scratch:**
- **Minimum:** 10,000 images (not recommended)
- **Recommended:** 50,000+ images

**Always prefer fine-tuning over training from scratch.**

### 6.3 Collection Strategy

```python
# Example: Collect 1 hour of driving = ~108,000 frames @ 30 FPS
# Subsample every 10th frame = 10,800 images
# After deduplication/filtering = ~2,000 diverse images

import cv2

def collect_driving_data(video_path, output_dir, sample_rate=10):
    """
    Extract frames from driving video for custom dataset.

    Args:
        video_path: Path to dashcam video
        output_dir: Where to save frames
        sample_rate: Save every Nth frame
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_rate == 0:
            # Save frame
            output_path = f"{output_dir}/frame_{saved_count:06d}.jpg"
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames from {frame_count} total")
```

### 6.4 Annotation Tools

**For bounding boxes:**
- **LabelImg** (local, free): https://github.com/heartexlabs/labelImg
- **CVAT** (web-based, free): https://cvat.org/
- **Label Studio** (local/cloud, free): https://labelstud.io/

**For segmentation:**
- **Labelme** (local, free): https://github.com/wkentaro/labelme
- **Supervisely** (cloud, paid): https://supervise.ly/

**Cost:**
- DIY annotation: Free (your time)
- Outsourced (Amazon MTurk, Scale AI): $0.10-0.50 per image for bboxes

---

## 7. Data Management

### 7.1 Version Control with DVC

```bash
# Install DVC
pip install dvc

# Initialize DVC
cd ~/datasets
dvc init

# Track large files
dvc add coco/train2017.zip
dvc add bdd100k/images/

# Commit to git (only tracks metadata, not data)
git add coco/train2017.zip.dvc .gitignore
git commit -m "Add COCO dataset"

# Push data to remote storage (S3, Google Drive, etc.)
dvc remote add -d storage s3://my-bucket/datasets
dvc push
```

### 7.2 Dataset Registry

```yaml
# datasets.yaml - Track what datasets are used in each experiment

experiments:
  yolov8n_baseline:
    train_datasets:
      - coco/train2017
      - bdd100k/train
    val_datasets:
      - bdd100k/val

  yolov8n_night:
    train_datasets:
      - coco/train2017
      - bdd100k/train
      - exdark/train  # Added night data
    val_datasets:
      - bdd100k/val
      - exdark/test
```

---

## 8. Summary

### Quick Start (This Week)

1. Download BDD100K validation (1.2GB) - test existing models
2. Download COCO val2017 (1GB) - baseline evaluation
3. Download State Farm (2GB) - DMS testing

**Total:** ~4GB, can start experimentation immediately

### Short Term (Next Month)

1. Download COCO train2017 (25GB)
2. Download BDD100K full (12GB)
3. Download CULane (20GB) for lanes
4. Start fine-tuning experiments

**Total:** ~60GB

### Long Term (Production)

1. Download Waymo or nuScenes for advanced 3D detection
2. Collect custom data for your specific deployment
3. Generate synthetic data from CARLA

**Total:** ~500GB-1TB

---

**Next Steps:** Download minimal datasets and run baseline inference on Jetson Orin to establish performance benchmarks.
