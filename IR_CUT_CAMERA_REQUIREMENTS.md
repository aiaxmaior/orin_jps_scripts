# IR-CUT Camera Requirements for ADAS/DMS Systems

## Executive Summary

**TL;DR:**
- **DMS (Driver Monitoring):** IR-CUT **NOT required** - use dedicated IR illumination instead
- **ADAS (Road-facing):** IR-CUT **REQUIRED** for day/night performance
- **Fisheye ADAS:** IR-CUT **STRONGLY RECOMMENDED**

---

## 1. What is an IR-CUT Filter?

An **IR-CUT filter** is a **switchable mechanical filter** that:
- **DAY mode:** Blocks infrared light (> 700nm), passes visible light (400-700nm)
  - Prevents IR contamination → accurate color reproduction
  - Improves image sharpness (IR causes chromatic aberration)

- **NIGHT mode:** Removes IR filter, allows IR + visible light
  - Maximizes light sensitivity
  - Enables night vision with IR illumination
  - Provides usable images in low light (< 1 lux)

**Mechanism:** Solenoid-driven mechanical filter that physically moves in/out

**Cost:** Adds $5-15 to camera module

---

## 2. Camera Requirements by Position

### 2.1 ADAS Camera (Road-Facing, Forward-Looking)

**Recommendation: IR-CUT FILTER REQUIRED**

#### Why IR-CUT is Critical for ADAS:

**1. Day/Night Performance Differential**

| Condition | Without IR-CUT | With IR-CUT |
|-----------|----------------|-------------|
| **Bright daylight** | Washed out colors, purple tint | Accurate colors |
| **Dusk/Dawn** | Poor contrast | Good contrast |
| **Night (streetlights)** | Very dark, <5m visibility | 20-30m visibility with IR LEDs |
| **Night (no lights)** | Blind | 10-20m visibility with IR |
| **Tunnels** | Slow adaptation | Fast adaptation |

**2. Object Detection Accuracy**

Without IR-CUT (night):
- Pedestrian detection: 60-70% recall @ 20m
- Vehicle detection: 75-85% recall @ 30m
- Sign detection: 40-50% recall @ 15m

With IR-CUT + IR illumination (night):
- Pedestrian detection: 85-90% recall @ 20m
- Vehicle detection: 90-95% recall @ 30m
- Sign detection: 70-80% recall @ 15m

**Improvement:** +20-30% detection recall in night conditions

**3. Color Accuracy (Critical for Sign/Light Recognition)**

Traffic lights and signs rely on color:
- Red vs Green traffic light
- Yellow vs White lane markings
- Red stop signs

Without IR-CUT during day: Colors shift toward purple/magenta
- **Risk:** Misclassifying red light as yellow → SAFETY CRITICAL

**4. Automotive Standard Compliance**

Most automotive-grade cameras (OV2312, IMX390, AR0233) include IR-CUT by default
- It's an industry expectation for ADAS

#### Recommended ADAS Cameras with IR-CUT:

| Camera Model | Resolution | FOV | IR-CUT | Night Mode | Price | Use Case |
|--------------|-----------|-----|--------|------------|-------|----------|
| **IMX490** | 2880x1860 | 120° | ✅ | Excellent (NIR) | $80-120 | Premium ADAS |
| **IMX390** | 1920x1280 | 100° | ✅ | Very good | $50-80 | Mid-range ADAS |
| **AR0233** | 1920x1200 | 60° | ✅ | Good | $40-60 | Budget ADAS |
| **OV2312** | 1920x1080 | 140° | ✅ | Good | $45-65 | Wide FOV ADAS |

**Note:** All support NVIDIA Jetson Orin via MIPI CSI-2

#### Fisheye ADAS (Wide FOV: 170°-190°)

**Fisheye + IR-CUT = STRONGLY RECOMMENDED**

Challenges with fisheye at night:
- **Distortion** amplifies noise at image edges
- **Vignetting** (darker edges) more pronounced in low light
- **Wide FOV** captures more irrelevant dark areas → harder for model

**Solution:** IR-CUT + powerful IR illumination (850nm LEDs, 940nm for stealth)

Recommended fisheye cameras:
- **Leopard Imaging LI-AR0233-FISHEYE** (190° FOV, IR-CUT, $85)
- **e-con Systems See3CAM_CU135** (AR0135, 170° FOV, IR-CUT, $120)

---

### 2.2 DMS Camera (Driver-Facing, Interior)

**Recommendation: IR-CUT NOT REQUIRED (use dedicated IR instead)**

#### Why DMS is Different:

**1. Controlled Lighting Environment**
- Interior cabin lighting is relatively stable
- No extreme day/night transitions like exterior
- Dashboard lights, dome lights provide base illumination

**2. Dedicated IR Illumination is Superior**
- **Active IR LEDs (850nm or 940nm)** provide consistent, controlled lighting
- **No mechanical parts** → more reliable, lower cost
- **Always-on IR** works in all conditions (day/night/tunnels)
- **Privacy-friendly:** 940nm IR is invisible to human eye

**3. DMS Works in IR-Only Mode**

DMS tasks don't require color:
- Face detection: Works perfectly in grayscale/IR
- Gaze estimation: Pupil tracking works better in IR (high contrast)
- Drowsiness: Eyelid closure, head pose → no color needed
- Distraction: Head orientation → no color needed

**Color is NOT needed for DMS → IR-CUT adds no value**

#### Recommended DMS Camera Setup:

**Option 1: IR-Only Camera (Best for DMS)**
```
Camera: IMX219 (no IR-CUT) or OV7251 (global shutter, no IR-CUT)
+ IR LEDs: 4x 850nm LEDs (60° beam angle)
+ Placement: Near camera, 10-15cm from driver face
```

**Advantages:**
- Lower cost ($25 camera vs $50 with IR-CUT)
- No moving parts (higher reliability)
- Consistent illumination 24/7
- Better pupil contrast in IR

**Option 2: Visible Light + Ambient IR**
```
Camera: IMX219 (no IR-CUT, allows some IR)
+ No dedicated IR LEDs
+ Rely on dashboard ambient light
```

**Advantages:**
- Lowest cost
- Simplest setup

**Disadvantages:**
- Poor performance in dark tunnels
- Driver comfort issues (dashboard brightness at night)

#### DMS Camera Placement

**Optimal position:** Dashboard center or A-pillar
- Distance to driver: 50-80 cm
- Angle: 15-25° downward
- FOV required: 60-80° (covers driver face + hands on wheel)

**IR Illumination:**
- **850nm:** Visible as faint red glow (better performance)
- **940nm:** Completely invisible (better driver comfort, slightly lower camera sensitivity)

**Recommended IR LED setup:**
- 4x high-power IR LEDs (1W each)
- Total: 4W IR illumination
- Current: ~350mA per LED
- Placement: Symmetrically around camera lens

---

### 2.3 Multi-Camera System

**Typical ADAS/DMS System:**

```
Vehicle Setup:
├── Front ADAS (Road-facing)
│   ├── Camera: IMX390 with IR-CUT ✅
│   ├── FOV: 100-120°
│   ├── IR LEDs: 6x 850nm (10W total) for night
│   └── Mount: Windshield, behind rearview mirror
│
├── Driver Monitor (DMS)
│   ├── Camera: IMX219 without IR-CUT ❌
│   ├── FOV: 60-80°
│   ├── IR LEDs: 4x 940nm (4W total, always-on)
│   └── Mount: Dashboard center or A-pillar
│
└── (Optional) Rear Camera
    ├── Camera: OV2312 with IR-CUT ✅
    ├── FOV: 140° fisheye
    └── IR LEDs: 4x 850nm (parking assist)
```

---

## 3. IR Illumination Design

### 3.1 IR LED Selection

| Wavelength | Visibility | Camera Sensitivity | Use Case |
|------------|-----------|-------------------|----------|
| **850nm** | Faint red glow | Excellent (90-95%) | ADAS night vision |
| **940nm** | Invisible | Good (60-70%) | DMS (driver comfort) |

**Recommendation:**
- **ADAS:** 850nm (performance > stealth)
- **DMS:** 940nm (comfort > max performance)

### 3.2 IR Power Budget

**ADAS (Forward Camera):**
- Range: 20-30m effective illumination
- Power: 10-15W total
- LEDs: 6-8x high-power IR LEDs (60° beam angle)
- Placement: Near camera, symmetrical

**DMS (Interior):**
- Range: 0.5-1.0m (driver face)
- Power: 2-4W total
- LEDs: 4x IR LEDs (60-80° beam angle)
- Placement: Around camera lens

**Total IR Power:** ~15W (ADAS) + 3W (DMS) = **18W**

### 3.3 IR Synchronization with Exposure

**Advanced:** Pulse IR LEDs in sync with camera exposure
- Reduces power consumption by 50%
- Minimizes heat
- Requires PWM control from Orin GPIO

**Implementation:**
```python
# Example: Sync IR with camera frame rate (30 FPS)
frame_rate = 30  # FPS
exposure_time = 10  # ms

# IR LED duty cycle
duty_cycle = (exposure_time / (1000 / frame_rate)) * 100  # 30%

# PWM control
import Jetson.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(33, GPIO.OUT)
pwm = GPIO.PWM(33, frame_rate)  # 30 Hz
pwm.start(duty_cycle)  # 30% duty cycle
```

**Power savings:** 15W → 4.5W average (for ADAS)

---

## 4. IR-CUT Control from Jetson Orin

### 4.1 Automatic Day/Night Switching

**Method 1: Ambient Light Sensor**
```python
# Use I2C light sensor (e.g., TSL2561)
import smbus

bus = smbus.SMBus(1)
TSL2561_ADDR = 0x39

def read_lux():
    # Read ambient light level
    data = bus.read_i2c_block_data(TSL2561_ADDR, 0x0C, 2)
    lux = (data[1] << 8) | data[0]
    return lux

# Switch IR-CUT based on lux
lux = read_lux()
if lux < 10:  # < 10 lux = night
    enable_ir_cut(False)  # Remove filter, allow IR
    enable_ir_leds(True)   # Turn on IR illumination
else:
    enable_ir_cut(True)    # Insert filter, block IR
    enable_ir_leds(False)  # Turn off IR
```

**Method 2: Image Brightness Analysis**
```python
import cv2
import numpy as np

def estimate_brightness(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate mean brightness
    mean_brightness = np.mean(gray)

    return mean_brightness

# Auto-switch based on image brightness
brightness = estimate_brightness(frame)
if brightness < 30:  # Dark image (0-255 scale)
    switch_to_night_mode()
else:
    switch_to_day_mode()
```

**Hysteresis:** Avoid rapid switching
```python
# Add hysteresis to prevent oscillation
DAY_THRESHOLD = 50
NIGHT_THRESHOLD = 30

if current_mode == 'day' and brightness < NIGHT_THRESHOLD:
    switch_to_night_mode()
elif current_mode == 'night' and brightness > DAY_THRESHOLD:
    switch_to_day_mode()
```

### 4.2 IR-CUT GPIO Control

**Hardware connection:**
```
Jetson Orin GPIO Pin → IR-CUT Solenoid Driver → IR-CUT Filter
```

**Example GPIO control:**
```python
import Jetson.GPIO as GPIO

IR_CUT_PIN = 33  # Pin 33 = GPIO13

GPIO.setmode(GPIO.BOARD)
GPIO.setup(IR_CUT_PIN, GPIO.OUT)

def enable_ir_cut(enable):
    """
    Enable = True  → Insert IR filter (day mode)
    Enable = False → Remove IR filter (night mode)
    """
    if enable:
        GPIO.output(IR_CUT_PIN, GPIO.HIGH)  # 3.3V
    else:
        GPIO.output(IR_CUT_PIN, GPIO.LOW)   # 0V
```

---

## 5. Impact on ML Models

### 5.1 Day vs Night Model Performance

**Without IR-CUT + IR illumination:**
- Night model accuracy: 60-70% of day performance
- Requires separate night-specific models
- Increased model storage and memory

**With IR-CUT + IR illumination:**
- Night model accuracy: 85-95% of day performance
- Can use same model with minor fine-tuning
- Reduced model complexity

### 5.2 Training Data Requirements

**Without IR-CUT:**
- Need 50%+ night-time labeled data
- Night data is expensive to collect
- More augmentation needed

**With IR-CUT:**
- 20-30% night data sufficient
- Better generalization
- Simpler augmentation pipeline

---

## 6. Cost-Benefit Analysis

### Per-Camera Cost

| Component | Without IR-CUT | With IR-CUT | Delta |
|-----------|----------------|-------------|-------|
| **Camera module** | $25 | $50 | +$25 |
| **IR LEDs (powerful)** | $15 | $10 | -$5 |
| **Driver circuit** | $3 | $5 | +$2 |
| **Total** | **$43** | **$65** | **+$22** |

### System Cost (2 cameras)

| Configuration | ADAS | DMS | Total |
|---------------|------|-----|-------|
| **Budget** | $43 (no IR-CUT) | $43 (no IR-CUT) | **$86** |
| **Recommended** | $65 (with IR-CUT) | $43 (no IR-CUT) | **$108** |
| **Premium** | $65 (with IR-CUT) | $65 (with IR-CUT) | **$130** |

**ROI Calculation:**

Additional $22 for ADAS IR-CUT:
- Improves night detection by 20-30%
- Reduces false negatives (missed pedestrians)
- **Safety value:** Priceless
- **Regulatory compliance:** Required for most markets

**Verdict:** **IR-CUT for ADAS is worth the cost**

---

## 7. Recommended Camera Selection

### Final Recommendations

**Option 1: Budget (Total: $110)**
- ADAS: AR0233 with IR-CUT ($50)
- DMS: IMX219 no IR-CUT + 940nm IR LEDs ($30)
- IR LEDs: 10x 850nm/940nm ($15)
- Drivers/cables: ($15)

**Option 2: Recommended (Total: $180)**
- ADAS: IMX390 with IR-CUT ($70)
- DMS: OV7251 global shutter + 940nm IR LEDs ($45)
- IR LEDs: High-power array ($35)
- Drivers/cables/housing: ($30)

**Option 3: Premium (Total: $280)**
- ADAS: IMX490 with IR-CUT ($110)
- DMS: IMX390 with IR-CUT + 940nm IR LEDs ($70)
- IR LEDs: Synchronized pulsed array ($60)
- Drivers/auto-switching/housing: ($40)

### Compatibility Matrix

| Camera | Jetson Orin | CSI Lanes | FPS @ Max Res | IR-CUT Support |
|--------|------------|-----------|---------------|----------------|
| IMX219 | ✅ Native | 2 | 30 | ❌ (aftermarket) |
| IMX477 | ✅ Native | 2 | 30 | ❌ (aftermarket) |
| **IMX390** | ✅ Via GMSL2 | 4 | 30 | ✅ Built-in |
| **IMX490** | ✅ Via GMSL2 | 4 | 30 | ✅ Built-in |
| AR0233 | ✅ Via deserializer | 4 | 60 | ✅ Built-in |
| OV2312 | ✅ Native | 2 | 30 | ⚠️ Optional |

**Note:** GMSL2 cameras require FPD-Link III or GMSL deserializer board (~$100 additional)

---

## 8. Summary & Decision Matrix

### Decision Framework

**Question 1:** Is this camera for ADAS (road-facing)?
- **YES → IR-CUT REQUIRED**
- NO → Continue

**Question 2:** Is this camera for DMS (driver-facing)?
- **YES → IR-CUT NOT NEEDED** (use dedicated IR instead)
- NO → Continue

**Question 3:** Will the camera operate in varying light conditions?
- **YES → IR-CUT RECOMMENDED**
- NO → IR-CUT optional

**Question 4:** Do you need accurate color reproduction?
- **YES → IR-CUT REQUIRED**
- NO → IR-CUT optional

### Final Verdict

| Camera Position | IR-CUT Required? | Alternative Solution |
|----------------|------------------|---------------------|
| **ADAS Front** | ✅ YES | None (IR-CUT is best) |
| **ADAS Fisheye** | ✅ YES | None (IR-CUT is best) |
| **DMS Driver** | ❌ NO | Dedicated IR LEDs (940nm) |
| **Rear Camera** | ⚠️ OPTIONAL | Parking-only = no IR-CUT OK |
| **Side Cameras** | ⚠️ OPTIONAL | Depends on use case |

---

## 9. Implementation Checklist

- [ ] **ADAS Camera:** Select IMX390 or IMX490 with built-in IR-CUT
- [ ] **DMS Camera:** Select IMX219 or OV7251 without IR-CUT
- [ ] **IR LEDs for ADAS:** 6-8x 850nm high-power LEDs (10-15W total)
- [ ] **IR LEDs for DMS:** 4x 940nm LEDs (3-4W total)
- [ ] **GPIO Control:** Connect IR-CUT and IR LED control to Orin GPIO
- [ ] **Light Sensor:** TSL2561 or similar for auto day/night switching
- [ ] **Driver Circuits:** Solenoid driver for IR-CUT, LED driver for IR
- [ ] **Sync Firmware:** Implement PWM sync for pulsed IR (optional optimization)
- [ ] **Validate Night Performance:** Test detection accuracy in <1 lux conditions
- [ ] **Thermal Management:** Ensure IR LEDs have heatsinking (15W total heat)

---

**Conclusion:** IR-CUT is essential for ADAS cameras, unnecessary for DMS. Invest in IR-CUT for the road-facing camera, use dedicated IR illumination for driver monitoring.
