# Full ADAS/DMS Pipeline Implementation Strategy
## Jetson Orin Nano 8GB - Multi-Model Orchestration

---

## Pipeline Architecture: Tiered Inference

### Core Principle: **Frame Skipping + Priority Scheduling**

Not all models need to run on every frame. We'll implement:
- **Critical models**: Every frame (30 FPS)
- **Important models**: Every 2nd frame (15 FPS)
- **Analytics models**: Every 3-5th frames (6-10 FPS)

---

## Phase 1: Current Working Setup ✓

**Status:** Operational
- DashCamNet on dual cameras
- 640x360 inference resolution
- ~30 FPS stable

---

## Phase 2: Add Object Tracking (Next Step)

### Enable NvDCF Tracker

**Benefit:** Minimal overhead, huge value
- Adds ~5ms latency
- ~50 MB GPU memory
- Provides object IDs, trajectories, velocities

**Implementation:**

```ini
[tracker]
enable=1
tracker-width=640
tracker-height=360
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
ll-config-file=/opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml
gpu-id=0
display-tracking-id=1
```

**Testing:** Verify tracking works before adding more models.

---

## Phase 3: Add DMS Models (FaceNet + GazeNet)

### 3.1 Download Models

```bash
cd ~/jps/models
../ngc-cli/ngc registry model download-version nvidia/tao/facenet:pruned_quantized_v2.0.1
../ngc-cli/ngc registry model download-version nvidia/tao/gazenet:pruned_v1.0
```

### 3.2 Pipeline Configuration

**Stream-specific inference:**
- ADAS stream (camera 1): DashCamNet only
- DMS stream (camera 0): FaceNet → GazeNet (cascade)

```
Camera 0 (DMS) → FaceNet (Primary) → GazeNet (Secondary on faces)
                                   → FacialLandmarks (Secondary)

Camera 1 (ADAS) → DashCamNet (Primary) → Tracker
```

### 3.3 DeepStream Multi-GIE Configuration

```ini
# Primary GIE for ADAS stream
[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
config-file=config_infer_dashcamnet.txt
# Only process source 1 (ADAS camera)
process-mode=1

# Primary GIE for DMS stream
[primary-gie]
enable=1
gpu-id=0
gie-unique-id=2
config-file=config_infer_facenet.txt
# Only process source 0 (DMS camera)
process-mode=1

# Secondary GIE for gaze on DMS
[secondary-gie0]
enable=1
gpu-id=0
gie-unique-id=3
operate-on-gie-id=2
config-file=config_infer_gazenet.txt
```

**Memory Impact:** +130 MB (FaceNet + GazeNet)
**Latency Impact:** +13ms
**Total so far:** ~350 MB, ~35ms → **Still fits!**

---

## Phase 4: Add Lane Detection

### Option A: Classical CV (Recommended First)

**Pros:**
- Minimal GPU usage (~10 MB)
- Fast (~3ms)
- Good for highways/clear lanes

**Implementation:** Custom probe using OpenCV
```python
# In custom DeepStream probe
def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                            minLineLength=100, maxLineGap=50)
    # Fit polynomial to lane lines
    return left_lane, right_lane
```

### Option B: Deep Learning (Later)

Download PeopleSegNet or custom lane segmentation model:
```bash
../ngc-cli/ngc registry model download-version nvidia/tao/peoplesegnet:pruned_v2.0.2
```

Adapt for lane segmentation.

**Memory Impact (Classical):** ~10 MB
**Latency Impact:** ~3ms
**Total so far:** ~360 MB, ~38ms

---

## Phase 5: Add Monocular Depth Estimation

### Approach: Geometric + Learning Hybrid

#### 5.1 Geometric Depth (Immediate - No Model Needed)

Use camera calibration + known object sizes:

```python
def estimate_distance(bbox_height, object_class, focal_length):
    """
    Known heights:
    - Average car: 1.5m
    - Average person: 1.7m
    - Road signs: 0.6-1.0m
    """
    known_heights = {
        'car': 1.5,
        'person': 1.7,
        'bicycle': 1.2,
        'road_sign': 0.8
    }

    real_height = known_heights.get(object_class, 1.5)
    distance = (real_height * focal_length) / bbox_height
    return distance
```

**Implementation:** Custom metadata probe
**Memory Impact:** 0 MB (pure math)
**Latency Impact:** <1ms

#### 5.2 Deep Learning Depth (Optional, Advanced)

Use MonoDepth or similar:
- Only run every 5th frame
- Use for depth map refinement
- ~200 MB, ~20ms per frame

**For now, skip deep learning depth** - use geometric method.

---

## Phase 6: Add Facial Landmarks + Drowsiness Detection

### 6.1 Download Models

```bash
../ngc-cli/ngc registry model download-version nvidia/tao/fpenet:pruned_v2.0.2
../ngc-cli/ngc registry model download-version nvidia/tao/emotionnet:pruned_v1.0
```

### 6.2 Cascade Configuration

```
FaceNet → GazeNet (every frame)
       → FacialLandmarks (every 2nd frame)
       → EmotionNet (every 5th frame)
```

**Benefit of frame skipping:**
- Facial landmarks for eye tracking: 15 FPS is sufficient
- Emotion/drowsiness: 6 FPS is adequate (changes slowly)

**Memory Impact:** +70 MB
**Latency Impact:** +7ms (amortized)
**Total so far:** ~430 MB, ~45ms

---

## Phase 7: Analytics & Distance Calculation

### 7.1 Enable nvdsanalytics

```ini
[nvds-analytics]
enable=1
config-file=config_nvdsanalytics.txt
```

Define ROIs and rules:
- Forward collision zone
- Lane departure zones
- Driver attention zones (looking at road vs away)

### 7.2 Custom Metadata Probe

Create C++/Python probe for:
- Distance calculation (geometric method)
- Time-to-collision (TTC)
- Driver attention score
- PERCLOS (drowsiness metric)

**Implementation:**
```cpp
static GstPadProbeReturn
analytics_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                                gpointer u_data)
{
    // Extract object metadata
    // Calculate distances for each detected object
    // Compute TTC = distance / relative_velocity
    // Calculate driver attention metrics
    // Attach custom metadata

    return GST_PAD_PROBE_OK;
}
```

**Memory Impact:** Minimal (~10 MB)
**Latency Impact:** ~2ms

---

## Final Pipeline Configuration

### Resource Summary

| Component | GPU Memory | Latency | Frame Rate |
|-----------|------------|---------|------------|
| **ADAS Stream** |
| DashCamNet | 150 MB | 15ms | 30 FPS |
| NvDCF Tracker | 50 MB | 5ms | 30 FPS |
| Lane Detection (CV) | 10 MB | 3ms | 30 FPS |
| Depth (Geometric) | 0 MB | <1ms | 30 FPS |
| **DMS Stream** |
| FaceNet | 80 MB | 8ms | 30 FPS |
| GazeNet | 50 MB | 5ms | 30 FPS |
| Facial Landmarks | 30 MB | 3ms | 15 FPS |
| EmotionNet | 40 MB | 4ms | 6 FPS |
| **Infrastructure** |
| nvdsanalytics | 10 MB | 2ms | 30 FPS |
| Custom probes | 10 MB | 2ms | 30 FPS |
| **TOTAL** | **~430 MB** | **~45ms** | **30 FPS** |

### Performance Projections

**GPU Utilization:** ~85%
**Memory Usage:** ~5.5 GB total (430 MB models + 5 GB system/buffers)
**Latency:** 45ms → **22 FPS sustainable**
**Power:** ~22W

---

## Implementation Order (Recommended)

### Week 1: Tracking & Basic DMS
1. ✅ DashCamNet dual camera working
2. ⬜ Enable NvDCF tracker
3. ⬜ Download FaceNet + GazeNet
4. ⬜ Configure dual primary GIE (one per stream)
5. ⬜ Test ADAS + DMS together

### Week 2: Lane Detection & Distance
1. ⬜ Implement classical lane detection (OpenCV probe)
2. ⬜ Camera calibration
3. ⬜ Geometric distance estimation
4. ⬜ TTC calculation
5. ⬜ Test full ADAS pipeline

### Week 3: Advanced DMS
1. ⬜ Download Facial Landmarks + EmotionNet
2. ⬜ Configure cascaded secondary GIEs
3. ⬜ Implement frame skipping logic
4. ⬜ PERCLOS drowsiness detection
5. ⬜ Test full DMS pipeline

### Week 4: Analytics & Optimization
1. ⬜ Enable nvdsanalytics with ROIs
2. ⬜ Implement custom metadata probes
3. ⬜ Alert engine integration
4. ⬜ Performance profiling & optimization
5. ⬜ End-to-end testing

---

## Alternative: Frame Skipping Implementation

If we hit performance limits, implement smart frame skipping:

```python
frame_count = 0

def on_frame(frame):
    frame_count += 1

    # Run every frame (critical)
    run_dashcamnet(frame)
    run_facenet(frame)
    run_gazenet(frame)
    run_tracker(frame)

    # Run every 2nd frame
    if frame_count % 2 == 0:
        run_landmarks(frame)
        run_lane_detection(frame)

    # Run every 5th frame
    if frame_count % 5 == 0:
        run_emotion_drowsiness(frame)

    # Analytics every frame (cheap)
    run_analytics(frame)
```

This reduces average compute to ~30ms/frame → **sustained 30 FPS**.

---

## Fallback Strategy: Model Quantization

If still hitting limits:

1. **Use INT8 for all models** (already planned)
2. **Reduce input resolution** to 480x360 (from 640x360)
3. **Use model pruning** (already using pruned models)
4. **Disable non-critical models** temporarily

---

## Monitoring & Profiling

Use these tools to validate performance:

```bash
# GPU utilization
tegrastats --interval 1000

# Pipeline profiling
export GST_DEBUG=3
export GST_DEBUG_DUMP_DOT_DIR=~/pipeline_graphs

# TensorRT profiling
nsys profile --stats=true deepstream-app -c config.txt
```

---

## Next Immediate Action

**Test Phase 2 first:** Enable tracking and verify it works without degrading performance.

```bash
# Edit deepstream_csi_camera.txt
[tracker]
enable=1  # Change from 0

# Run and check FPS
deepstream-app -c deepstream_csi_camera.txt
```

Should still get ~30 FPS. If yes, proceed to Phase 3 (DMS models).

---

## Questions to Consider

1. **Do you need ALL models running simultaneously?**
   - Maybe drowsiness only matters when vehicle is moving?
   - Maybe depth only needed when objects detected?

2. **What's the minimum viable product?**
   - Start with: DashCamNet + FaceNet + GazeNet + Tracker
   - Add others incrementally

3. **What's your target frame rate?**
   - 30 FPS for demos
   - 15-20 FPS acceptable for production?

Let me know which phase you'd like to tackle first!
