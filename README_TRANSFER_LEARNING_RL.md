# Transfer Learning & Reinforcement Learning for ADAS/DMS Edge Systems

**Comprehensive framework for training, fine-tuning, and deploying custom ADAS/DMS models on NVIDIA Jetson Orin**

---

## Quick Overview

This repository contains a complete pipeline for developing edge-optimized ADAS (Advanced Driver Assistance Systems) and DMS (Driver Monitoring Systems) using:

- **Transfer Learning:** Fine-tune pre-trained models for custom driving conditions
- **Reinforcement Learning:** Learn driving policies from simulation (CARLA)
- **Edge Optimization:** Deploy on Jetson Orin with TensorRT INT8 quantization
- **Production-Ready:** Safety monitors, OTA updates, real-world robustness

---

## What's Included

### 📚 Documentation

| Document | Description |
|----------|-------------|
| **[TRANSFER_LEARNING_RL_STRATEGY.md](TRANSFER_LEARNING_RL_STRATEGY.md)** | Overall strategy, architecture, and approach |
| **[IR_CUT_CAMERA_REQUIREMENTS.md](IR_CUT_CAMERA_REQUIREMENTS.md)** | Camera selection guide (ADAS vs DMS, IR-CUT analysis) |
| **[DATASET_GUIDE.md](DATASET_GUIDE.md)** | Public datasets, data preparation, loaders |
| **[HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md)** | Complete BOM, hardware selection, costs |
| **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** | Step-by-step implementation (Phase 1-7) |
| **[REAL_WORLD_DEPLOYMENT.md](REAL_WORLD_DEPLOYMENT.md)** | Production deployment, safety, testing |
| **[ADAS_DMS_Full_Pipeline_Strategy.md](ADAS_DMS_Full_Pipeline_Strategy.md)** | Existing inference pipeline (Phase 1-7) |

### 💻 Code Framework

```
orin_jps_scripts/
├── training/                       # Transfer learning infrastructure
│   ├── datasets/
│   │   ├── base_dataset.py        # Base dataset classes
│   │   ├── coco_loader.py         # COCO dataset loader
│   │   ├── bdd100k_loader.py      # (To be implemented)
│   │   └── state_farm_loader.py   # DMS dataset (To be implemented)
│   │
│   ├── augmentations/
│   │   └── adas_augmentations.py  # ADAS/DMS augmentation pipelines
│   │
│   ├── models/                     # Model architectures (To be implemented)
│   ├── finetune/                   # Fine-tuning scripts (To be implemented)
│   ├── optimization/               # Pruning, quantization (To be implemented)
│   └── export/                     # ONNX, TensorRT export (To be implemented)
│
├── rl/                             # Reinforcement learning
│   ├── agents/
│   │   └── ppo_agent.py           # PPO implementation (complete)
│   │
│   ├── environments/               # CARLA env wrapper (To be implemented)
│   ├── perception/                 # State encoding (To be implemented)
│   ├── training/                   # RL training loop (To be implemented)
│   └── deployment/                 # Edge policy deployment (To be implemented)
│
├── inference/                      # Existing DeepStream inference
│   ├── deepstream_*.txt           # DeepStream configs
│   ├── advanced_proximity_sensor.py
│   └── ...
│
├── models/                         # Model storage
│   ├── dashcamnet_vpruned_v1.0.4/ # Existing
│   └── custom_trained/            # Your fine-tuned models
│
└── experiments/                    # MLflow experiment tracking
    ├── mlruns/
    └── configs/
```

---

## System Architecture

### Complete Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    DEVELOPMENT PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. DATA                                                     │
│     [COCO] [BDD100K] [Custom] → [Augmentation]             │
│                                        ↓                     │
│  2. TRAINING                                                │
│     [Pre-trained Model] → [Transfer Learning] → [Fine-tune] │
│                                        ↓                     │
│  3. OPTIMIZATION                                            │
│     [Pruning] → [QAT] → [TensorRT INT8]                   │
│                                        ↓                     │
│  4. RL (Optional)                                           │
│     [CARLA Sim] → [PPO Training] → [Policy Network]        │
│                                        ↓                     │
│  5. DEPLOYMENT                                              │
│     [Jetson Orin] → [DeepStream] → [Real-time Inference]   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Edge Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   JETSON ORIN RUNTIME                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Camera 0: ADAS] ──→ [YOLOv8n INT8] ──→ [Detections]      │
│        ↓                  (8ms)               ↓             │
│   [IMX390 w/                              [Tracking]        │
│    IR-CUT]                                    ↓             │
│                                           [Distance Est]     │
│                                               ↓             │
│  [Camera 1: DMS] ──→ [MobileNetV3] ──→ [Gaze/Distraction]  │
│        ↓                (5ms)               ↓               │
│   [OV7251]                                                  │
│                                                              │
│              ↓           ↓           ↓                       │
│         [State Encoder] (2ms)                               │
│              ↓                                               │
│         [RL Policy Network] (3ms) ← Optional                │
│              ↓                                               │
│         [Safety Monitor] (2ms)                              │
│              ↓                                               │
│         [Decision/Action]                                   │
│                                                              │
│  Total latency: ~20ms (50 FPS)                              │
│  GPU memory: ~150 MB                                        │
│  Power: ~18W                                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Features

### ✅ Completed

- [x] Comprehensive documentation (5 guides, 50+ pages)
- [x] Dataset management framework (COCO loader, base classes)
- [x] Augmentation pipelines (ADAS, DMS, night, fisheye)
- [x] PPO agent implementation (complete, tested)
- [x] Hardware BOM and selection guide
- [x] IR-CUT camera analysis and recommendations
- [x] Real-world deployment best practices
- [x] Safety architecture design

### 🚧 To Be Implemented (Next Steps)

- [ ] BDD100K dataset loader
- [ ] YOLOv8 fine-tuning script
- [ ] Model quantization pipeline
- [ ] CARLA environment wrapper
- [ ] RL training loop
- [ ] Edge policy deployment
- [ ] MLflow integration
- [ ] Practice training runs

---

## Quick Start

### 1. Setup Environment

```bash
# Clone repository (already done)
cd /home/user/orin_jps_scripts

# Install dependencies
pip3 install torch torchvision albumentations pycocotools opencv-python

# Download sample dataset (COCO val, 1GB)
cd ~/datasets
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_val2017.zip
unzip val2017.zip
unzip annotations_val2017.zip
```

### 2. Test Framework

```bash
# Test dataset loader
python3 training/datasets/coco_loader.py ~/datasets/coco

# Test augmentations (visualize)
python3 training/augmentations/adas_augmentations.py sample_image.jpg

# Test PPO agent
python3 rl/agents/ppo_agent.py
```

### 3. Read Documentation

**Start here:**
1. [TRANSFER_LEARNING_RL_STRATEGY.md](TRANSFER_LEARNING_RL_STRATEGY.md) - Understand the approach
2. [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - Follow step-by-step
3. [DATASET_GUIDE.md](DATASET_GUIDE.md) - Download datasets
4. [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md) - Order hardware if needed

---

## Hardware Recommendations

### Minimum (Development): **$1,144**
- Jetson Orin NX 16GB Dev Kit: $799
- IMX390 ADAS camera (IR-CUT): $85
- OV7251 DMS camera: $50
- 256GB NVMe SSD: $35
- IR LEDs, cooling, misc: $175

### Production: **$2,072**
- See [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md) for full BOM

---

## Model Zoo

### Pre-trained (Already Available)

| Model | Task | Size | Latency (Orin) | Notes |
|-------|------|------|----------------|-------|
| DashCamNet | Detection | 7MB | 15ms | ResNet18, 4 classes |
| YOLOv8n | Detection | 6MB | 12ms | 80 COCO classes |
| PeopleNet | Detection | 15MB | 18ms | 3 classes (person/bag/face) |

### Custom (To Be Trained)

| Model | Task | Dataset | Target Performance |
|-------|------|---------|-------------------|
| YOLOv8n-BDD100K-INT8 | ADAS detection | BDD100K | mAP@0.5: 0.55, 8ms |
| YOLOv8n-Night-INT8 | Night detection | BDD100K night | mAP@0.5: 0.52, 8ms |
| MobileNetV3-DMS | Distraction | State Farm | Acc: 95%, 5ms |
| PPO-Highway | Driving policy | CARLA | Collision <5% |

---

## Datasets

### Recommended Minimal Setup (<50GB)

| Dataset | Size | Use Case | Download |
|---------|------|----------|----------|
| COCO val2017 | 1GB | Baseline eval | [Link](http://images.cocodataset.org/zips/val2017.zip) |
| BDD100K val | 1.2GB | ADAS eval | [Link](https://bdd-data.berkeley.edu/) |
| State Farm | 2GB | DMS training | [Kaggle](https://www.kaggle.com/c/state-farm-distracted-driver-detection) |
| TuSimple Lanes | 10GB | Lane detection | [Link](https://github.com/TuSimple/tusimple-benchmark) |

See [DATASET_GUIDE.md](DATASET_GUIDE.md) for comprehensive list.

---

## Timeline Estimate

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1: Setup** | 1 day | Environment, datasets |
| **Phase 2: ADAS Fine-tuning** | 1-2 weeks | YOLOv8n on BDD100K |
| **Phase 3: Night Fine-tuning** | 1 week | Night-specific model |
| **Phase 4: DMS Training** | 1 week | Distraction classifier |
| **Phase 5: RL Setup** | 1-2 weeks | CARLA environment |
| **Phase 6: RL Training** | 1-2 weeks | Driving policy |
| **Phase 7: Deployment** | 1-2 weeks | Integration, testing |
| **Phase 8: Validation** | 2-3 weeks | Real-world testing |
| **TOTAL** | **10-12 weeks** | Production system |

---

## Performance Targets

### Jetson Orin NX 16GB

| Metric | Target | Notes |
|--------|--------|-------|
| **Latency** | <50ms | End-to-end (camera to decision) |
| **FPS** | 30+ | Sustained |
| **GPU Memory** | <512MB | Dual camera system |
| **Power** | <25W | Including cameras, IR LEDs |
| **mAP@0.5** | >0.55 | ADAS detection (day) |
| **mAP@0.5** | >0.50 | Night detection |
| **Accuracy** | >95% | DMS distraction classification |

---

## Safety & Compliance

### Safety Architecture

- **Safety Monitor:** Rule-based validation of ML outputs
- **Fail-Safe:** Conservative fallback on uncertainty
- **Multi-Sensor Fusion:** Cross-validate with radar/ultrasonic
- **Uncertainty Estimation:** Monte Carlo Dropout
- **Graceful Degradation:** Reduce functionality if sensors fail

### Standards

- **ISO 26262:** Functional safety (ASIL B/C target)
- **ISO/SAE 21434:** Cybersecurity
- **UN R157:** Automated Lane Keeping Systems

See [REAL_WORLD_DEPLOYMENT.md](REAL_WORLD_DEPLOYMENT.md) for details.

---

## Troubleshooting

### Common Issues

**Issue:** Out of memory on Jetson
- **Solution:** Reduce batch size, use INT8, enable DLA offloading

**Issue:** Low mAP after fine-tuning
- **Solution:** Check augmentations, increase epochs, validate dataset

**Issue:** High latency
- **Solution:** Use INT8, temporal scheduling, async inference

**Issue:** RL policy not learning
- **Solution:** Check reward function, visualize episodes, reduce action space

See full troubleshooting guide in [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md#troubleshooting).

---

## Contributing

This is a personal research project, but improvements welcome:

1. Implement missing components (BDD100K loader, training scripts)
2. Add new datasets (Waymo, nuScenes)
3. Optimize models (pruning, NAS)
4. Test on other Jetson platforms (Nano, AGX)

---

## Resources

### Official Documentation

- [NVIDIA Jetson Orin](https://developer.nvidia.com/embedded/jetson-orin)
- [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)
- [TAO Toolkit](https://developer.nvidia.com/tao-toolkit)
- [TensorRT](https://developer.nvidia.com/tensorrt)

### External Tools

- [PyTorch](https://pytorch.org/)
- [Albumentations](https://albumentations.ai/)
- [MLflow](https://mlflow.org/)
- [CARLA Simulator](https://carla.org/)

### Datasets

- [COCO](https://cocodataset.org/)
- [BDD100K](https://bdd-data.berkeley.edu/)
- [KITTI](http://www.cvlibs.net/datasets/kitti/)
- [Waymo Open Dataset](https://waymo.com/open/)

---

## License

This project is for research and educational purposes. Public datasets have their own licenses - please review before use.

Automotive deployment requires certification and regulatory compliance. Consult legal and safety experts before real-world deployment.

---

## Status

**Current Phase:** Documentation & Framework Complete ✅

**Next Steps:**
1. Implement BDD100K loader
2. Create fine-tuning training script
3. Run first training experiment
4. Document results in MLflow

**Estimated Time to MVP:** 4-6 weeks

---

## Contact

Questions? Issues? Suggestions?

- Open an issue in this repository
- Review documentation (likely answered there!)
- Check troubleshooting guide

---

**Happy Training! 🚗🤖**

*Last Updated: 2025-11-18*
