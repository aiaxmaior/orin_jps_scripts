# Implementation Guide: Transfer Learning & RL for ADAS/DMS

## Quick Start

This guide walks you through implementing the complete transfer learning and RL pipeline for ADAS/DMS on Jetson Orin.

---

## Phase 1: Environment Setup (Day 1)

### 1.1 Install Dependencies

```bash
# On Jetson Orin (JetPack 6.0+)
cd /home/user/orin_jps_scripts

# Install PyTorch for Jetson
wget https://nvidia.box.com/shared/static/...pytorch_for_jetson.whl
pip3 install pytorch_for_jetson.whl

# Install training dependencies
pip3 install -r requirements_training.txt
```

**requirements_training.txt:**
```
torch>=2.0.0
torchvision>=0.15.0
albumentations>=1.3.0
pycocotools>=2.0.6
opencv-python>=4.8.0
matplotlib>=3.7.0
tensorboard>=2.13.0
mlflow>=2.5.0
tqdm>=4.65.0
pyyaml>=6.0
```

### 1.2 Download Datasets

**Quick Start (BDD100K validation only - 1.2GB):**
```bash
cd ~/datasets
mkdir -p bdd100k
cd bdd100k

# Download BDD100K val set (requires account at bdd-data.berkeley.edu)
# After logging in, download:
# - bdd100k_images_100k_val.zip
# - bdd100k_labels_detection_20_val.json

unzip bdd100k_images_100k_val.zip
```

**Full Setup (COCO + BDD100K - ~40GB):**
```bash
# COCO
cd ~/datasets
mkdir -p coco
cd coco

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

# BDD100K (follow manual download steps)
```

### 1.3 Verify Installation

```bash
cd /home/user/orin_jps_scripts

# Test dataset loader
python3 training/datasets/coco_loader.py ~/datasets/coco

# Test augmentation pipeline
python3 training/augmentations/adas_augmentations.py test_image.jpg

# Test PPO agent
python3 rl/agents/ppo_agent.py
```

---

## Phase 2: Transfer Learning - ADAS Object Detection (Week 1-2)

### 2.1 Baseline Evaluation

First, evaluate a pre-trained model to establish baseline:

```bash
# Download pre-trained YOLOv8n
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Run inference on validation set
python3 training/finetune/evaluate_baseline.py \
    --model yolov8n.pt \
    --data ~/datasets/bdd100k/val \
    --output results/baseline_yolov8n.json
```

Expected baseline (YOLOv8n on BDD100K val):
- mAP@0.5: ~0.35
- mAP@0.5:0.95: ~0.20
- Inference time: ~15ms @ 640x640 on Orin NX

### 2.2 Fine-tuning on BDD100K

**Config:** `training/finetune/config/yolov8n_bdd100k.yaml`

```yaml
# Model
model: yolov8n
pretrained: yolov8n.pt

# Data
data_root: ~/datasets/bdd100k
train_split: train
val_split: val
num_classes: 10
class_names: [car, pedestrian, truck, bus, motorcycle, bicycle, traffic_light, traffic_sign, rider, train]

# Training
epochs: 50
batch_size: 16
lr: 0.001
weight_decay: 0.0005
warmup_epochs: 3

# Augmentation
augmentation: aggressive  # or 'standard'
input_size: 640

# Hardware
device: cuda
num_workers: 4
mixed_precision: true  # FP16 training
```

**Run fine-tuning:**

```bash
python3 training/finetune/finetune_adas.py \
    --config training/finetune/config/yolov8n_bdd100k.yaml \
    --output experiments/yolov8n_bdd100k_run1
```

**Expected timeline:**
- Epoch time: ~15 minutes (10K images, batch=16)
- Total training: ~12 hours for 50 epochs
- Early stopping around epoch 30-40

**Expected results (after fine-tuning):**
- mAP@0.5: ~0.55 (+20 points)
- mAP@0.5:0.95: ~0.35 (+15 points)

### 2.3 Model Optimization for Edge

**Quantization (FP32 → INT8):**

```bash
python3 training/optimization/quantization.py \
    --model experiments/yolov8n_bdd100k_run1/best.pt \
    --calibration-data ~/datasets/bdd100k/val \
    --calibration-samples 500 \
    --output models/custom_trained/yolov8n_bdd100k_int8
```

**Export to TensorRT:**

```bash
python3 training/export/to_tensorrt.py \
    --model models/custom_trained/yolov8n_bdd100k_int8/model.onnx \
    --precision int8 \
    --calibration models/custom_trained/yolov8n_bdd100k_int8/calibration.cache \
    --output models/custom_trained/yolov8n_bdd100k_int8/model.engine \
    --dla-core 0  # Use DLA core 0 if available (Orin NX/AGX)
```

**Validate optimized model:**

```bash
python3 training/export/validate_engine.py \
    --engine models/custom_trained/yolov8n_bdd100k_int8/model.engine \
    --data ~/datasets/bdd100k/val \
    --output results/yolov8n_int8_validation.json
```

Expected after INT8:
- mAP@0.5: ~0.54 (-1 point, acceptable)
- Inference time: ~8ms (2x speedup!)
- Memory: 20MB (vs 40MB FP16)

---

## Phase 3: Transfer Learning - Night Conditions (Week 3)

### 3.1 Fine-tune for Night Driving

**Dataset:** BDD100K night subset + ExDark

**Config:** `training/finetune/config/yolov8n_night.yaml`

```yaml
model: yolov8n
pretrained: experiments/yolov8n_bdd100k_run1/best.pt  # Start from BDD100K model

data_root: ~/datasets/bdd100k
train_filter: night  # Only night images
val_filter: night

epochs: 20  # Fewer epochs (fine-tuning fine-tuned model)
batch_size: 16
lr: 0.0001  # Lower LR

augmentation: night_specific  # Custom night augmentations
```

**Run:**

```bash
python3 training/finetune/finetune_adas.py \
    --config training/finetune/config/yolov8n_night.yaml \
    --output experiments/yolov8n_night_run1
```

Expected improvement on night test set:
- Before: mAP@0.5 ~0.40
- After: mAP@0.5 ~0.52 (+12 points)

---

## Phase 4: Transfer Learning - DMS (Week 4)

### 4.1 Driver Distraction Detection

**Dataset:** State Farm Distracted Driver

**Model:** MobileNetV3-Small (lightweight for DMS)

**Config:** `training/finetune/config/mobilenetv3_dms.yaml`

```yaml
model: mobilenetv3_small
pretrained: imagenet

data_root: ~/datasets/state_farm
num_classes: 10
class_names: [safe_driving, texting_right, phone_right, texting_left, phone_left,
              radio, drinking, reaching_behind, hair_makeup, talking_passenger]

epochs: 30
batch_size: 32
lr: 0.001
input_size: 224

augmentation: dms  # DMS-specific augmentations
```

**Run:**

```bash
python3 training/finetune/finetune_dms.py \
    --config training/finetune/config/mobilenetv3_dms.yaml \
    --output experiments/mobilenetv3_dms_run1
```

Expected results:
- Accuracy: ~95% on validation set
- Inference time: ~5ms @ 224x224 on Orin
- Memory: 15MB (INT8)

---

## Phase 5: Reinforcement Learning Setup (Week 5-6)

### 5.1 Install CARLA Simulator

**On Desktop PC (not Jetson - CARLA is GPU-intensive):**

```bash
# Download CARLA 0.9.15
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.15.tar.gz
tar -xzf CARLA_0.9.15.tar.gz
cd CARLA_0.9.15

# Install Python API
pip install carla==0.9.15

# Test CARLA
./CarlaUE4.sh -quality-level=Low -RenderOffScreen &
sleep 10
python3 PythonAPI/examples/spawn_npc.py -n 50
```

### 5.2 Setup RL Environment

**Create CARLA environment wrapper:**

`rl/environments/carla_env.py` should interface with CARLA and provide:

```python
class CARLAEnv:
    def reset(self) -> np.ndarray:
        """Reset environment, return initial state."""
        pass

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take action, return (next_state, reward, done, info).
        """
        pass

    def get_state(self) -> np.ndarray:
        """
        Get state vector from:
        - Camera image features (from ADAS model)
        - Vehicle speed, acceleration
        - Detected objects + distances
        - Lane position
        """
        pass

    def compute_reward(self) -> float:
        """
        Multi-objective reward:
        - Safety: -1000 if collision
        - Lane keeping: -10 * abs(lane_offset)
        - Speed: -0.5 * abs(speed - target_speed)
        - Smoothness: -5 * jerk
        """
        pass
```

### 5.3 Train RL Policy

**Config:** `rl/training/config/ppo_highway.yaml`

```yaml
# Environment
env: carla_highway
num_envs: 4  # Parallel environments
max_episode_steps: 1000

# Agent
state_dim: 1024  # From CNN feature extractor
action_space: discrete
num_actions: 175  # 5 throttle x 5 brake x 7 steering

# PPO hyperparameters
lr: 0.0003
gamma: 0.99
gae_lambda: 0.95
clip_epsilon: 0.2
epochs_per_update: 10
batch_size: 64

# Training
total_timesteps: 1000000  # 1M steps
update_every: 2048  # Update every 2048 steps
save_every: 10000

# Hardware
device: cuda
```

**Run training:**

```bash
# On desktop with CARLA
python3 rl/training/train_ppo.py \
    --config rl/training/config/ppo_highway.yaml \
    --output experiments/ppo_highway_run1
```

**Expected training time:**
- 1M timesteps: ~24-48 hours (depends on GPU)
- Convergence: Check reward plot in TensorBoard

```bash
tensorboard --logdir experiments/ppo_highway_run1/logs
```

### 5.4 Evaluate RL Policy

```bash
python3 rl/training/eval_policy.py \
    --checkpoint experiments/ppo_highway_run1/checkpoint_best.pt \
    --env carla_highway \
    --num_episodes 100 \
    --render  # Visualize
```

**Metrics to track:**
- Collision rate: Target <5% over 100 episodes
- Lane deviation: Target <0.2m average
- Comfort (jerk): Target <2 m/s³ average

---

## Phase 6: Deployment on Jetson Orin (Week 7)

### 6.1 Deploy ADAS Model

**Copy optimized model to Orin:**

```bash
scp models/custom_trained/yolov8n_bdd100k_int8/model.engine \
    orin:/home/user/orin_jps_scripts/models/custom_trained/
```

**Create DeepStream config:**

`deepstream_custom_yolov8n.txt`:

```ini
[application]
enable-perf-measurement=1
perf-measurement-interval-sec=5

[source0]
enable=1
type=3  # CSI camera
camera-id=0
camera-width=1920
camera-height=1080
camera-fps-n=30
camera-fps-d=1

[streammux]
width=640
height=640
batch-size=1
batched-push-timeout=-1
gpu-id=0

[primary-gie]
enable=1
model-engine-file=models/custom_trained/yolov8n_bdd100k_int8/model.engine
batch-size=1
interval=0
gie-unique-id=1
network-mode=2  # INT8
labelfile-path=models/custom_trained/yolov8n_bdd100k_int8/labels.txt
output-blob-names=output0
parse-bbox-func-name=NvDsInferParseYolo

[sink0]
enable=1
type=2  # EGL window
sync=0
```

**Run inference:**

```bash
deepstream-app -c deepstream_custom_yolov8n.txt
```

### 6.2 Deploy RL Policy (Edge Optimized)

**Export RL policy to TensorRT:**

```bash
python3 rl/deployment/export_policy.py \
    --checkpoint experiments/ppo_highway_run1/checkpoint_best.pt \
    --output models/custom_trained/ppo_policy_v1/policy.engine \
    --precision fp16
```

**Run edge policy:**

```bash
python3 rl/deployment/edge_policy.py \
    --policy-engine models/custom_trained/ppo_policy_v1/policy.engine \
    --adas-engine models/custom_trained/yolov8n_bdd100k_int8/model.engine \
    --camera-id 0
```

**Expected performance on Orin NX:**
- ADAS inference: 8ms
- RL policy: 3ms
- Total latency: ~15ms (66 FPS)

---

## Phase 7: Evaluation & Refinement (Week 8-10)

### 7.1 Benchmarking

**Create benchmark suite:**

```bash
python3 experiments/benchmark_suite.py \
    --models yolov8n_int8,yolov8n_fp16,dashcamnet \
    --datasets bdd100k_val,bdd100k_night,coco_val \
    --output results/benchmark_report.pdf
```

**Metrics to report:**
- mAP per model per dataset
- Inference time (mean, std, p99)
- Memory usage (GPU, CPU)
- Power consumption (Jetson tools)

### 7.2 Error Analysis

**Find failure cases:**

```bash
python3 experiments/error_analysis.py \
    --model models/custom_trained/yolov8n_bdd100k_int8/model.engine \
    --data ~/datasets/bdd100k/val \
    --output results/error_analysis/
```

**Common failure modes:**
- Small/distant objects
- Occlusions
- Rare classes (e.g., motorcycles)
- Adverse weather

**Mitigation:**
1. Collect more data for failure cases
2. Adjust augmentations
3. Fine-tune with hard negative mining

### 7.3 Continuous Improvement Loop

```
1. Deploy model → 2. Collect edge cases → 3. Label data → 4. Fine-tune → 5. Deploy
     ↑                                                                      ↓
     └──────────────────────────────────────────────────────────────────────┘
```

**Tools:**
- Edge logging: Save failure cases to SD card
- Active learning: Prioritize labeling of uncertain predictions
- MLflow: Track all experiments

---

## Troubleshooting

### Issue: OOM (Out of Memory) on Jetson

**Solutions:**
1. Reduce batch size
2. Use gradient accumulation
3. Enable mixed precision (FP16)
4. Offload models to DLA
5. Disable SWAP (causes slowdown)

```bash
# Check memory
sudo jetson_release

# Disable SWAP
sudo swapoff -a
```

### Issue: Low mAP after fine-tuning

**Checklist:**
- [ ] Is the pre-trained model from a similar domain?
- [ ] Are augmentations too aggressive?
- [ ] Is learning rate too high/low?
- [ ] Is the dataset balanced?
- [ ] Are bounding boxes accurate?

**Debug steps:**
1. Visualize augmentations: `python3 training/augmentations/adas_augmentations.py <image>`
2. Check class distribution: `dataset.print_class_distribution()`
3. Plot learning curves: TensorBoard
4. Reduce augmentations, increase epochs

### Issue: RL policy not learning

**Checklist:**
- [ ] Is reward function correct? (Positive for good behavior)
- [ ] Is reward too sparse? (Add intermediate rewards)
- [ ] Are actions having an effect? (Check environment)
- [ ] Is value network learning? (Plot value loss)

**Debug steps:**
1. Test random policy: Should get some reward
2. Log episode rewards: Should increase over time
3. Visualize policy: `render=True` in eval
4. Reduce action space complexity

---

## Next Steps

1. **Collect Custom Data:** Use dashcam to record driving in your region
2. **Multi-Task Learning:** Train single model for detection + segmentation + depth
3. **Temporal Fusion:** Use multiple frames (video) instead of single images
4. **Sim-to-Real:** Transfer RL policy from CARLA to real Jetson
5. **Safety Layer:** Add rule-based fallback for RL policy

---

## Resources

- **NVIDIA TAO Toolkit:** Pre-trained models optimized for Jetson
  - https://developer.nvidia.com/tao-toolkit

- **DeepStream SDK:** GPU-accelerated inference
  - https://developer.nvidia.com/deepstream-sdk

- **MLflow:** Experiment tracking
  - https://mlflow.org/

- **CARLA Simulator:** Autonomous driving simulation
  - https://carla.org/

---

**Estimated Total Timeline:** 10 weeks from setup to production deployment

**Budget:**
- Hardware (Orin NX + cameras): $1,144
- Cloud compute (optional, RL training): $100-200
- Datasets: Free (public datasets)
- **Total: ~$1,300**

Good luck with your implementation!
