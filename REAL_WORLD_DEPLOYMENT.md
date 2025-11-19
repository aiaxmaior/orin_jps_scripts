# Real-World Deployment Guide for ADAS/DMS Systems

## Overview

This document covers the critical considerations, best practices, and gotchas for deploying ADAS/DMS systems from lab to real-world vehicles.

---

## 1. The Lab-to-Road Reality Gap

### What Works in Lab vs. Road

| Aspect | Lab | Real World |
|--------|-----|------------|
| **Lighting** | Controlled, consistent | Extreme variation (sun glare, tunnels, night) |
| **Weather** | Clear | Rain, fog, snow, dust |
| **Camera** | Clean lens, fixed position | Dirty lens, vibration, thermal expansion |
| **Power** | Stable 19V | 12V automotive (9-16V range, spikes) |
| **Thermal** | Room temp (20-25°C) | -40°C to 85°C (interior can hit 70°C in summer) |
| **Latency** | Acceptable | Critical (<50ms for safety) |
| **Reliability** | Can reboot | Must run 24/7, no crashes |

**Key Insight:** Your model that gets 95% accuracy in the lab might drop to 70% on the road. Plan for this.

---

## 2. Safety-Critical Design Principles

### 2.1 Fail-Safe Architecture

**Never rely on ML alone for safety-critical decisions.**

```
┌─────────────────────────────────────────────────┐
│          SAFETY ARCHITECTURE                     │
├─────────────────────────────────────────────────┤
│                                                  │
│  [ML Model] ─→ [Safety Monitor] ─→ [Decision]  │
│                      ↓                           │
│                [Rule-Based                       │
│                 Fallback]                        │
│                                                  │
└─────────────────────────────────────────────────┘
```

**Safety Monitor Example:**

```python
class SafetyMonitor:
    """Validates ML outputs against physics-based constraints."""

    def validate_detection(self, detection, sensor_fusion):
        # Rule 1: Object cannot move faster than physically possible
        if detection.velocity > MAX_VELOCITY:
            return False, "Implausible velocity"

        # Rule 2: Cross-check with other sensors (e.g., radar, lidar)
        if not sensor_fusion.confirms(detection):
            return False, "Not confirmed by other sensors"

        # Rule 3: Temporal consistency (object can't teleport)
        if detection.sudden_appearance and no_radar_confirmation:
            return False, "Sudden appearance, likely false positive"

        # Rule 4: Safety-critical objects must meet high confidence
        if detection.class in ['pedestrian', 'cyclist']:
            if detection.confidence < 0.85:  # Higher threshold
                return False, "Low confidence for safety-critical class"

        return True, "Valid"

    def decide_action(self, ml_action, vehicle_state):
        # Rule-based override for emergencies
        if vehicle_state.collision_imminent:
            return "EMERGENCY_BRAKE"  # Override ML

        # Sanity check ML action
        if ml_action == "ACCELERATE" and vehicle_state.obstacle_ahead:
            return "MAINTAIN_SPEED"  # Override unsafe ML decision

        return ml_action  # Trust ML if passes checks
```

### 2.2 Uncertainty Estimation

**Know when your model is uncertain.**

```python
def get_prediction_with_uncertainty(model, input, num_samples=10):
    """
    Use Monte Carlo Dropout to estimate uncertainty.

    Low uncertainty → Trust prediction
    High uncertainty → Fall back to conservative action
    """
    model.train()  # Enable dropout at inference

    predictions = []
    for _ in range(num_samples):
        pred = model(input)
        predictions.append(pred)

    mean_pred = torch.mean(torch.stack(predictions), dim=0)
    std_pred = torch.std(torch.stack(predictions), dim=0)

    # High std = high uncertainty
    uncertainty = std_pred.max().item()

    return mean_pred, uncertainty
```

**Action based on uncertainty:**
- Uncertainty < 0.1: High confidence → Use ML decision
- Uncertainty 0.1-0.3: Medium → Reduce speed, increase caution
- Uncertainty > 0.3: Low confidence → Fall back to conservative rule-based

---

## 3. Robustness to Real-World Conditions

### 3.1 Camera Degradation

**Problem:** Camera lens gets dirty, water droplets, sun glare, condensation

**Solutions:**

**A. Hardware:**
- Hydrophobic coating on lens
- Active lens heater (prevent condensation)
- Lens washer system (optional, expensive)
- Sunshield visor

**B. Software:**

```python
def detect_degraded_image(image):
    """Detect if camera is compromised."""

    # Check 1: Excessive blur
    blur_score = cv2.Laplacian(image, cv2.CV_64F).var()
    if blur_score < 100:  # Threshold depends on camera
        return "BLUR", 0.8

    # Check 2: Overexposure (sun glare)
    bright_pixels = (image > 250).sum() / image.size
    if bright_pixels > 0.3:  # >30% pixels saturated
        return "OVEREXPOSURE", 0.9

    # Check 3: Underexposure (night, tunnel)
    dark_pixels = (image < 10).sum() / image.size
    if dark_pixels > 0.5:
        return "UNDEREXPOSURE", 0.7

    # Check 4: Water droplets (detect circular artifacts)
    # ... (use Hough circles or learned classifier)

    return "OK", 0.0
```

**Action:**
- If degraded: Alert driver, reduce system confidence, switch to backup sensor

### 3.2 Environmental Adaptation

**Dynamic model switching based on conditions:**

```python
class AdaptiveADAS:
    def __init__(self):
        self.day_model = load_model("yolov8n_day_fp16.engine")
        self.night_model = load_model("yolov8n_night_fp16.engine")
        self.rain_model = load_model("yolov8n_rain_fp16.engine")

    def select_model(self, conditions):
        """Select appropriate model based on conditions."""

        if conditions.time_of_day == "night":
            return self.night_model
        elif conditions.weather == "rain":
            return self.rain_model
        else:
            return self.day_model

    def infer(self, image, vehicle_state, gps, time):
        conditions = self.estimate_conditions(image, vehicle_state, gps, time)
        model = self.select_model(conditions)
        return model(image)
```

**Condition estimation:**
- Time of day: GPS time + sunrise/sunset tables
- Weather: Brightness histogram, windshield wiper status (CAN bus)
- Road type: GPS + map data (highway vs urban)

---

## 4. Latency & Real-Time Performance

### 4.1 Latency Budget

**Critical path:** Camera capture → Inference → Decision → Actuation

| Stage | Target | Notes |
|-------|--------|-------|
| Camera capture | 10ms | 30 FPS = 33ms period, process at 10ms mark |
| Preprocessing | 2ms | Resize, normalize |
| Inference (ADAS) | 8ms | YOLOv8n INT8 on Orin |
| Inference (DMS) | 5ms | MobileNetV3 INT8 |
| RL policy (if used) | 3ms | Small network, FP16 |
| Postprocessing | 2ms | NMS, tracking |
| Decision logic | 2ms | Safety checks, fusion |
| **Total** | **32ms** | Within 33ms budget ✓ |

**Optimization strategies:**

**A. Temporal scheduling:**
```python
# Not all models need to run every frame
frame_schedule = {
    'adas_detection': 1,       # Every frame (30 FPS)
    'lane_segmentation': 2,    # Every 2nd frame (15 FPS)
    'dms_gaze': 3,             # Every 3rd frame (10 FPS)
    'dms_drowsiness': 6,       # Every 6th frame (5 FPS)
}
```

**B. Asynchronous inference:**
```python
import threading
import queue

class AsyncInference:
    def __init__(self, model):
        self.model = model
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)
        self.thread = threading.Thread(target=self._inference_loop)
        self.thread.start()

    def _inference_loop(self):
        while True:
            input_data = self.input_queue.get()
            output = self.model(input_data)
            self.output_queue.put(output)

    def infer_async(self, input_data):
        """Non-blocking inference."""
        self.input_queue.put(input_data)

    def get_result(self, timeout=0.01):
        """Get latest result (or None if not ready)."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
```

**C. Zero-copy memory:**
```python
# Use CUDA streams and pinned memory
import torch

# Allocate pinned memory (faster CPU→GPU)
input_buffer = torch.zeros((1, 3, 640, 640), dtype=torch.float32).pin_memory()

# Create CUDA stream for async copy
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    input_gpu = input_buffer.to('cuda', non_blocking=True)
    output = model(input_gpu)
```

### 4.2 Profiling & Optimization

**Use NVIDIA tools:**

```bash
# Profile inference
nsys profile -o adas_profile python3 run_inference.py

# View in NVIDIA Nsight Systems
# Identify bottlenecks (memory copy, kernel launch overhead, etc.)

# Optimize TensorRT engine
trtexec --onnx=model.onnx \
        --saveEngine=model_optimized.engine \
        --int8 \
        --best \
        --workspace=4096 \
        --verbose
```

---

## 5. Data Management & Privacy

### 5.1 On-Device Data Logging

**What to log:**
- Edge cases (low confidence detections)
- Driver interventions (when driver overrides system)
- Near-misses (collision risk > threshold)
- System failures (errors, exceptions)

**What NOT to log:**
- Full video (privacy, storage)
- Driver face images (privacy)
- GPS traces (privacy)

**Implementation:**

```python
class EdgeCaseLogger:
    def __init__(self, storage_path="/media/sdcard/logs"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

    def should_log(self, detection, confidence):
        # Log only edge cases
        if confidence < 0.6:
            return True, "low_confidence"

        if detection.class == "pedestrian" and detection.distance < 5.0:
            return True, "close_pedestrian"

        return False, None

    def log_sample(self, image, detection, metadata, reason):
        """Log single sample for later review."""

        # Anonymize: Blur faces, license plates
        image = self.anonymize(image)

        # Compress
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 70])

        # Save with metadata
        timestamp = time.time()
        filename = f"{reason}_{timestamp}.jpg"
        metadata_file = filename.replace('.jpg', '.json')

        with open(self.storage_path / filename, 'wb') as f:
            f.write(buffer)

        with open(self.storage_path / metadata_file, 'w') as f:
            json.dump(metadata, f)

    def anonymize(self, image):
        """Blur faces and license plates."""
        # Use DMS face detector
        faces = detect_faces(image)
        for face_bbox in faces:
            image = blur_region(image, face_bbox)

        # Blur license plates (simple approach: blur small rectangular regions)
        # Better: Use license plate detector
        return image
```

### 5.2 Over-the-Air (OTA) Updates

**Model update workflow:**

```
1. Train new model in cloud
   ↓
2. Validate on test set (ensure no regression)
   ↓
3. A/B test on subset of vehicles (10%)
   ↓
4. If metrics improve → Roll out to all
   ↓
5. Monitor post-deployment metrics
```

**Implementation:**

```python
class OTAModelUpdater:
    def __init__(self, model_server="https://models.example.com"):
        self.model_server = model_server
        self.current_version = self.load_version()

    def check_for_update(self):
        """Check if new model is available."""
        response = requests.get(f"{self.model_server}/latest_version")
        latest_version = response.json()['version']

        if latest_version > self.current_version:
            return latest_version
        return None

    def download_and_install(self, version):
        """Download new model and install."""

        # Download
        model_url = f"{self.model_server}/models/adas_v{version}.engine"
        response = requests.get(model_url, stream=True)

        temp_path = f"/tmp/adas_v{version}.engine"
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Verify checksum
        if not self.verify_checksum(temp_path, version):
            raise Exception("Checksum mismatch")

        # Install (atomic swap)
        backup_path = "/home/user/models/adas_current.engine.bak"
        current_path = "/home/user/models/adas_current.engine"

        os.rename(current_path, backup_path)  # Backup old
        os.rename(temp_path, current_path)    # Install new

        # Update version
        self.current_version = version

        print(f"Updated to version {version}")

    def rollback(self):
        """Rollback to previous version if new model fails."""
        backup_path = "/home/user/models/adas_current.engine.bak"
        current_path = "/home/user/models/adas_current.engine"

        if os.path.exists(backup_path):
            os.rename(current_path, "/tmp/failed_model.engine")
            os.rename(backup_path, current_path)
            print("Rolled back to previous version")
```

---

## 6. Testing & Validation

### 6.1 Test Scenarios

**Minimum test coverage:**

| Scenario | Description | Pass Criteria |
|----------|-------------|---------------|
| **Sunny day** | Clear weather, good lighting | mAP > 0.90 |
| **Night** | <1 lux, IR illumination | mAP > 0.75 |
| **Rain** | Wet roads, water droplets | mAP > 0.70 |
| **Tunnel** | Rapid lighting change | No crashes, <2s adaptation |
| **Sun glare** | Direct sunlight in camera | Degrades gracefully, alerts driver |
| **Occlusion** | Partial object visibility | Detects >50% visible objects |
| **High speed** | >100 km/h | Detection at 50m+ distance |
| **Urban** | Crowded, many objects | FP rate < 5% |

### 6.2 Hardware-in-the-Loop (HIL) Testing

**Setup:**
```
[CARLA Simulator] → [Video Injector] → [Jetson Orin] → [Logging]
                     (HDMI or CSI)
```

**Benefits:**
- Test edge cases without real-world risk
- Reproducible scenarios
- Automated regression testing

**Implementation:**

```bash
# Inject CARLA video into Jetson CSI input (requires HDMI→CSI adapter)
python3 carla/scenarios/run_scenario.py \
    --scenario highway_overtake \
    --output hdmi

# On Jetson, run system normally
python3 adas_system.py --camera-id 0

# Compare output to ground truth
python3 test/validate_hil.py \
    --ground-truth carla/scenarios/highway_overtake_gt.json \
    --predictions logs/output.json
```

---

## 7. Regulatory & Certification

### 7.1 Automotive Standards

**Relevant standards:**

- **ISO 26262:** Functional safety for automotive systems
  - ASIL (Automotive Safety Integrity Level): ASIL A (lowest) to ASIL D (highest)
  - ADAS typically requires ASIL B or C

- **ISO/SAE 21434:** Cybersecurity for automotive
  - Protects against hacking, unauthorized access

- **UN Regulation No. 157:** Automated Lane Keeping Systems (ALKS)
  - Defines requirements for Level 3 automation

**Implications for ML:**
- Must document training data, model architecture, validation
- Explainability: Why did the model make this decision?
- Fail-safe: What happens if model fails?

### 7.2 Certification Process

**Steps:**
1. Hazard analysis (HARA): What can go wrong?
2. Safety goals: Define acceptable risk levels
3. Technical safety requirements: How to achieve safety goals?
4. Implementation: Build system
5. Validation: Test against requirements
6. Certification: Third-party audit (TÜV, SGS, etc.)

**Cost:** $50,000-500,000 depending on ASIL level

**Timeline:** 6-18 months

---

## 8. Edge Cases & Known Limitations

### 8.1 Failure Modes to Monitor

**Visual failures:**
- Missed detections (false negatives)
- False positives (ghost objects)
- Misclassification (truck as car)

**Temporal failures:**
- ID switches (tracking failure)
- Latency spikes (thermal throttling)

**Sensor failures:**
- Camera malfunction
- IR LED failure (night blindness)
- GPU failure

**Environmental:**
- Snow covering camera
- Extreme cold (LCD freezes)
- Extreme heat (thermal shutdown)

### 8.2 Graceful Degradation

**Example degradation strategy:**

```
Normal → Reduced confidence → Limited function → Disabled → Emergency stop
  ↓              ↓                   ↓               ↓            ↓
Full ADAS   Conservative    Speed limit   Alert only   Stop car
100%        80% capability  50 km/h max   Driver must  safely
                                          take over
```

**Trigger conditions:**
- Multiple sensors fail → Reduced confidence
- Camera obscured → Limited function
- Critical failure → Disabled, alert driver
- Driver unresponsive + critical failure → Emergency stop

---

## 9. Best Practices Checklist

**Before Deployment:**
- [ ] Validate on >10,000 real-world images from target environment
- [ ] Test in all weather conditions (day, night, rain, fog)
- [ ] Measure latency with real hardware (not simulation)
- [ ] Implement safety monitor and fail-safe
- [ ] Test camera degradation (spray water, dirt on lens)
- [ ] Validate power stability (9V to 16V range)
- [ ] Test thermal performance (-20°C to 70°C)
- [ ] Document all assumptions and limitations
- [ ] Create driver manual with clear instructions
- [ ] Setup logging and telemetry
- [ ] Plan for OTA updates
- [ ] Legal review (liability, privacy)

**Post-Deployment:**
- [ ] Monitor fleet metrics (daily)
- [ ] Collect edge cases for retraining
- [ ] Track failure modes
- [ ] A/B test model updates
- [ ] Respond to incidents within 24 hours
- [ ] Quarterly model retraining

---

## 10. Lessons Learned (Common Pitfalls)

**Pitfall 1: "It works on my laptop"**
- **Problem:** Model works on RTX 4090 but fails on Jetson
- **Solution:** Always test on target hardware early

**Pitfall 2: "The dataset is good enough"**
- **Problem:** Public datasets don't match your deployment
- **Solution:** Collect at least 1,000 images from target environment

**Pitfall 3: "We'll add safety later"**
- **Problem:** Safety is hard to retrofit
- **Solution:** Design for safety from day 1

**Pitfall 4: "False positives don't matter"**
- **Problem:** Frequent false alarms → driver disables system
- **Solution:** Optimize for precision, not just recall

**Pitfall 5: "One model fits all"**
- **Problem:** Single model poor in edge cases
- **Solution:** Use ensemble or condition-specific models

---

## Conclusion

Real-world deployment is **10x harder** than lab development. Budget time and resources accordingly.

**Rule of thumb:** If it takes 1 month to train a model, budget 3 months for deployment (testing, integration, validation).

**Success metrics:**
- System uptime: >99.9%
- False positive rate: <5%
- Latency: <50ms p99
- User trust: Drivers keep system enabled

Good luck shipping to production!
