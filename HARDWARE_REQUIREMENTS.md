# Hardware System Requirements for ADAS/DMS Edge Deployment

## System Overview

This document defines the complete hardware bill of materials (BOM) for a production ADAS/DMS system based on NVIDIA Jetson Orin.

---

## 1. Compute Platform Options

### 1.1 Jetson Orin Variants Comparison

| Specification | Orin Nano 8GB | Orin NX 8GB | Orin NX 16GB | Orin AGX 32GB |
|--------------|---------------|-------------|--------------|---------------|
| **GPU** | 1024-core | 1024-core | 1024-core | 2048-core |
| **CPU** | 6-core ARM A78AE | 8-core ARM A78AE | 8-core ARM A78AE | 12-core ARM A78AE |
| **DLA** | 0 | 1 | 1 | 2 |
| **RAM** | 8GB LPDDR5 | 8GB LPDDR5 | 16GB LPDDR5 | 32GB LPDDR5 |
| **Storage** | microSD/NVMe | microSD/NVMe | microSD/NVMe | NVMe SSD |
| **AI Perf** | 40 TOPS | 70 TOPS | 70 TOPS | 200 TOPS |
| **Power** | 7-15W | 10-25W | 10-25W | 15-60W |
| **Price** | $499 | $599 | $799 | $1,999 |
| **ADAS Suitability** | Budget | **Recommended** | Premium | Overkill |

**Recommendation: Orin NX 16GB ($799)**

**Justification:**
- 16GB RAM: Comfortable for dual-camera + model loading + RL policy
- 1 DLA core: Offload one model (e.g., DashCamNet) → free GPU for others
- 70 TOPS: Sufficient for multi-model pipeline
- Power efficiency: 10-25W fits automotive thermal budget
- Cost: ~$800 sweet spot (Nano too constrained, AGX unnecessary)

---

### 1.2 Carrier Board

**Option 1: NVIDIA Jetson Orin NX Developer Kit**
- **Price:** $799 (includes module + carrier board)
- **Features:**
  - 2x MIPI CSI-2 (15-pin, 22-pin)
  - 1x M.2 Key M (NVMe SSD)
  - 4x USB 3.2
  - Gigabit Ethernet
  - 40-pin GPIO header
- **Use Case:** Development, prototyping
- **Pros:** All-in-one, ready to use
- **Cons:** Large form factor (110mm x 100mm), not automotive-grade

**Option 2: Connect Tech Forge Carrier (Automotive)**
- **Price:** $600-900
- **Features:**
  - IP67 rated enclosure
  - Extended temperature (-40°C to 85°C)
  - 4x GMSL2 camera inputs
  - CAN bus interface
  - Automotive power supply (9-36V)
  - Vibration/shock resistant
- **Use Case:** Production vehicle deployment
- **Pros:** Automotive-grade, compact (100mm x 72mm)
- **Cons:** Expensive, requires custom integration

**Recommendation for Phase:**
- **Prototyping (now):** NVIDIA Dev Kit ($799)
- **Production (later):** Connect Tech Forge or Auvidea J20 (~$700)

---

## 2. Camera System

### 2.1 ADAS Camera (Road-Facing)

**Recommended: IMX390 with IR-CUT**

| Specification | Value |
|--------------|-------|
| **Sensor** | Sony IMX390 |
| **Resolution** | 1920 x 1280 (2.3 MP) |
| **Frame Rate** | 30 FPS @ full res |
| **FOV** | 100-120° (depends on lens) |
| **Interface** | GMSL2 or MIPI CSI-2 |
| **IR-CUT** | Yes (mechanical, auto-switching) |
| **HDR** | 120 dB (excellent for varying light) |
| **Price** | $70-100 |

**Vendors:**
- **Leopard Imaging:** LI-IMX390-GMSL2 ($85)
- **e-con Systems:** See3CAM_24CUG (MIPI CSI-2, $95)
- **FRAMOS:** FSM-IMX390 ($110, industrial-grade)

**Alternative (Fisheye):**
- **Leopard Imaging LI-AR0233-FISHEYE** (190° FOV, IR-CUT, $85)

---

### 2.2 DMS Camera (Driver-Facing)

**Recommended: OV7251 Global Shutter (IR-optimized)**

| Specification | Value |
|--------------|-------|
| **Sensor** | OmniVision OV7251 |
| **Resolution** | 640 x 480 (VGA) |
| **Frame Rate** | 100 FPS (30 FPS sufficient for DMS) |
| **Shutter** | Global (no rolling shutter artifacts) |
| **FOV** | 60-80° |
| **Interface** | MIPI CSI-2 |
| **IR Sensitivity** | Excellent (no IR-CUT filter) |
| **Price** | $45-60 |

**Vendors:**
- **ArduCam:** OV7251 120FPS Camera Module ($50)
- **e-con Systems:** e-CAM20_CUTK2 (OV7251, $55)

**Budget Alternative:**
- **Raspberry Pi Camera v2.1 (IMX219):** $25
  - Lower frame rate (30 FPS)
  - Rolling shutter (OK for DMS)
  - Good IR sensitivity
  - Widely available

---

### 2.3 Camera Accessories

**IR Illumination:**
- **ADAS (850nm, 10W):** 6x Osram SFH 4715S ($3 each) = $18
- **DMS (940nm, 4W):** 4x Vishay VSMB2948SL ($2.50 each) = $10

**Lenses (if not included):**
- M12 lens, 100° FOV, IR-corrected: $15-30 each

**Cables:**
- MIPI CSI-2 ribbon cable (15cm): $5 each
- GMSL2 coaxial cable (3m): $20 each

**Mounts:**
- Camera housing (weatherproof): $10-20 each
- Vibration damping mount: $5-10

**Total Camera System Cost:**
- ADAS camera: $85
- DMS camera: $50
- IR LEDs + drivers: $30
- Cables + mounts: $50
- **Total: $215**

---

## 3. Storage

### 3.1 Primary Storage (OS + Models)

**NVMe SSD (M.2 2280, PCIe Gen 3)**

| Capacity | Recommended Model | Price | Use Case |
|----------|------------------|-------|----------|
| **128GB** | Samsung PM991a | $25 | Minimal (OS + 1-2 models) |
| **256GB** | WD Black SN770 | $35 | **Recommended** (OS + 5-10 models) |
| **512GB** | Samsung 980 | $50 | Development (datasets on device) |
| **1TB** | Samsung 980 PRO | $90 | Full local training |

**Recommendation: 256GB NVMe ($35)**

**Breakdown:**
- JetPack OS: 10GB
- DeepStream + dependencies: 5GB
- PyTorch + TensorRT: 8GB
- Models (10x TensorRT engines): 20GB
- RL environment: 5GB
- Datasets (local subset for testing): 50GB
- Logs, experiments: 20GB
- **Total: ~120GB → 256GB gives headroom**

**Installation:** M.2 slot on carrier board, boot from NVMe

---

### 3.2 Secondary Storage (Optional)

**microSD Card (for data logging, model backups)**

- **Capacity:** 128GB-256GB
- **Speed:** UHS-I U3 (95 MB/s write)
- **Model:** SanDisk Extreme ($20-35)
- **Use Case:** Log driving data, backup models

---

## 4. Power System

### 4.1 Power Budget

| Component | Typical | Peak | Notes |
|-----------|---------|------|-------|
| Jetson Orin NX 16GB | 15W | 25W | GPU/CPU load |
| ADAS Camera (IMX390) | 2W | 3W | Active imaging |
| DMS Camera (OV7251) | 1W | 1.5W | Lower resolution |
| IR LEDs (ADAS 10W) | 3W | 10W | Pulsed (30% duty) |
| IR LEDs (DMS 4W) | 1W | 4W | Pulsed (30% duty) |
| NVMe SSD | 2W | 5W | Read/write peaks |
| Cooling fan | 2W | 2W | Always on |
| **Total** | **26W** | **50W** | |

**Peak power: 50W → Add 20% margin → 60W system**

---

### 4.2 Power Supply Options

**For Development (Bench Power):**
- **NVIDIA Jetson Orin NX Power Supply:** 19V 4.74A (90W) barrel jack - $40
- Included with dev kit

**For Automotive Deployment:**

**Option 1: DC-DC Converter (12V automotive → 19V Jetson)**
- **Model:** MORNSUN LM50-23B12 (12V → 19V, 50W)
- **Price:** $45
- **Features:**
  - Input: 9-18V (automotive 12V nominal)
  - Output: 19V 2.6A (50W)
  - Efficiency: 89%
  - Protections: OVP, OCP, SCP
  - Operating temp: -40°C to 85°C

**Option 2: Automotive Power Module (12/24V → multi-voltage)**
- **Model:** Auvidea J106 Power Module
- **Price:** $120
- **Features:**
  - Input: 9-36V (supports 12V and 24V vehicles)
  - Output: 19V + 5V + 3.3V rails
  - Reverse polarity protection
  - Automotive-grade

**Recommendation:**
- **Dev:** Use included 19V PSU
- **Production:** MORNSUN DC-DC ($45) for cost, Auvidea ($120) for robustness

---

### 4.3 Battery Backup (Optional)

For safe shutdown on power loss:
- **LiPo Battery:** 3S 11.1V 2200mAh ($30)
- **UPS Module:** Waveshare UPS HAT ($25)
- **Runtime:** ~10 minutes at 15W (enough for safe shutdown)

---

## 5. Cooling System

### 5.1 Jetson Orin Thermal Management

**Thermal Design Power (TDP):** 25W (NX 16GB max)

**Passive Cooling (7-15W modes):**
- Large heatsink (included with dev kit)
- Sufficient for low-power modes (nvpmodel 2, 3)
- **Not sufficient for full ADAS workload**

**Active Cooling (Required for 25W mode):**
- **Fan:** Noctua NF-A4x10 FLX (40mm, 5V, 4500 RPM) - $15
- **Airflow:** 14.9 CFM
- **Noise:** 17.9 dBA (quiet)
- **Control:** PWM via Jetson GPIO

**Thermal Paste:** Arctic MX-4 ($8)

**Temperature Targets:**
- GPU: <70°C (sustained)
- CPU: <75°C
- Throttling starts: 80°C
- Emergency shutdown: 95°C

**Cooling Cost:** ~$25

---

### 5.2 Camera Thermal Management

ADAS camera (outside, direct sun):
- **Issue:** Lens heating causes image quality degradation
- **Solution:** Sunshield visor ($10) + thermal pad ($5)

DMS camera (interior):
- **Issue:** Minimal, controlled environment
- **Solution:** None needed (ambient cabin cooling)

---

## 6. Enclosure & Mounting

### 6.1 Jetson Enclosure

**For Development:**
- Acrylic case with fan mount: $25
- Jetson Orin NX aluminum case: $40

**For Automotive:**
- IP67 rated aluminum enclosure: $80-150
- Vibration-dampening mounts: $20
- EMI shielding gaskets: $15

---

### 6.2 Camera Mounting

**ADAS Camera (Windshield):**
- 3M VHB adhesive mount: $5
- Adjustable angle bracket: $10

**DMS Camera (Dashboard/A-pillar):**
- Magnetic base (easy repositioning): $8
- 3D-printed custom mount: $5 (material cost)

---

## 7. Networking & Connectivity

### 7.1 Ethernet (For Development)

- Built-in Gigabit Ethernet on dev kit
- CAT6 cable: $5

### 7.2 WiFi/Bluetooth (Optional)

- **Intel AX210 M.2 WiFi 6E Module:** $25
  - WiFi 6E (6 GHz)
  - Bluetooth 5.3
  - M.2 2230 slot

### 7.3 Cellular (Optional, for production)

- **Quectel EC25-A LTE Module:** $40
  - 4G LTE Cat 4
  - USB interface
  - GPS/GNSS included

### 7.4 CAN Bus (Automotive)

- **MCP2515 CAN Controller:** $12
  - SPI interface to Jetson
  - 1 Mbps CAN 2.0B
  - OBD-II connector for vehicle data (speed, RPM, etc.)

---

## 8. Sensors & Peripherals

### 8.1 GPS/GNSS (For mapping context)

- **U-blox NEO-M8N:** $25
  - 10 Hz update rate
  - UART interface
  - Supports GPS, GLONASS, Galileo

### 8.2 IMU (Inertial Measurement Unit)

- **MPU9250 9-DOF IMU:** $10
  - 3-axis gyroscope, accelerometer, magnetometer
  - I2C interface
  - For vehicle dynamics (yaw, pitch, roll)

### 8.3 Ambient Light Sensor

- **TSL2561 Light Sensor:** $5
  - I2C interface
  - Auto day/night switching for IR-CUT
  - 0.1-40,000 lux range

---

## 9. Complete Bill of Materials (BOM)

### 9.1 Development System (Prototyping)

| Component | Model | Qty | Unit Price | Total |
|-----------|-------|-----|-----------|-------|
| **Compute** | Jetson Orin NX 16GB Dev Kit | 1 | $799 | $799 |
| **Storage** | Samsung 980 NVMe 256GB | 1 | $35 | $35 |
| **ADAS Camera** | Leopard LI-IMX390-GMSL2 | 1 | $85 | $85 |
| **DMS Camera** | ArduCam OV7251 | 1 | $50 | $50 |
| **IR LEDs** | ADAS (850nm 10W) + DMS (940nm 4W) | 1 | $30 | $30 |
| **Cables** | CSI/GMSL cables | 1 | $30 | $30 |
| **Cooling** | Noctua fan + thermal paste | 1 | $25 | $25 |
| **Power** | Included with dev kit | 1 | $0 | $0 |
| **Enclosure** | Acrylic case | 1 | $25 | $25 |
| **Sensors** | Light sensor + IMU | 1 | $15 | $15 |
| **Misc** | Cables, mounts, connectors | 1 | $50 | $50 |
| | | | **TOTAL** | **$1,144** |

---

### 9.2 Production System (Automotive)

| Component | Model | Qty | Unit Price | Total |
|-----------|-------|-----|-----------|-------|
| **Compute** | Jetson Orin NX 16GB Module | 1 | $650 | $650 |
| **Carrier Board** | Connect Tech Forge (automotive) | 1 | $700 | $700 |
| **Storage** | Samsung PM991a NVMe 256GB | 1 | $35 | $35 |
| **ADAS Camera** | FRAMOS FSM-IMX390 (industrial) | 1 | $110 | $110 |
| **DMS Camera** | e-con e-CAM20_CUTK2 | 1 | $55 | $55 |
| **IR LEDs** | High-power automotive-grade | 1 | $50 | $50 |
| **Power** | Auvidea J106 Power Module | 1 | $120 | $120 |
| **Cooling** | Industrial fan + heatsink | 1 | $40 | $40 |
| **Enclosure** | IP67 aluminum housing | 1 | $120 | $120 |
| **Sensors** | GPS + IMU + light sensor | 1 | $40 | $40 |
| **CAN Bus** | MCP2515 CAN controller | 1 | $12 | $12 |
| **Cellular** | Quectel EC25-A LTE | 1 | $40 | $40 |
| **Cables/Mounts** | Automotive-grade | 1 | $100 | $100 |
| | | | **TOTAL** | **$2,072** |

---

## 10. System Integration

### 10.1 Physical Layout

```
┌─────────────────────────────────────────────────┐
│          ADAS/DMS System Physical Layout         │
├─────────────────────────────────────────────────┤
│                                                  │
│  [Windshield Mount]                              │
│      ↓                                           │
│  ┌─────────────┐                                │
│  │ ADAS Camera │ ← IMX390 with IR-CUT            │
│  │   + IR LEDs │ ← 6x 850nm LEDs                 │
│  └─────────────┘                                │
│        ↓ GMSL2 cable (3m)                        │
│        ↓                                         │
│  ┌──────────────────────────────┐               │
│  │   Main Enclosure (IP67)      │               │
│  │  ┌────────────────────────┐  │               │
│  │  │ Jetson Orin NX 16GB    │  │               │
│  │  │ + Carrier Board        │  │               │
│  │  ├────────────────────────┤  │               │
│  │  │ NVMe SSD (256GB)       │  │               │
│  │  ├────────────────────────┤  │               │
│  │  │ Cooling Fan            │  │               │
│  │  ├────────────────────────┤  │               │
│  │  │ Power Module (12V→19V) │  │               │
│  │  └────────────────────────┘  │               │
│  │                               │               │
│  │  Connections:                 │               │
│  │  • 12V from vehicle          │               │
│  │  • CAN bus to OBD-II         │               │
│  │  • GPS antenna               │               │
│  │  • 4G LTE antenna            │               │
│  │  • Ethernet (optional)       │               │
│  └──────────────────────────────┘               │
│        ↑ CSI cable (1m)                          │
│        ↑                                         │
│  ┌─────────────┐                                │
│  │ DMS Camera  │ ← OV7251                        │
│  │   + IR LEDs │ ← 4x 940nm LEDs                 │
│  └─────────────┘                                │
│      ↑                                           │
│  [Dashboard/A-pillar Mount]                      │
│                                                  │
└─────────────────────────────────────────────────┘
```

### 10.2 Wiring Diagram

```
Vehicle 12V ─→ [Power Module] ─→ 19V ─→ Jetson Orin NX
                     │
                     ├─→ 5V ─→ IR LED Driver (ADAS)
                     ├─→ 5V ─→ IR LED Driver (DMS)
                     └─→ 5V ─→ Cooling Fan

OBD-II Port ─→ [CAN Controller] ─→ SPI ─→ Jetson GPIO

GPS Antenna ─→ [GPS Module] ─→ UART ─→ Jetson

ADAS Camera ─→ GMSL2 ─→ Jetson CSI-2 Port 0
DMS Camera ─→ MIPI CSI ─→ Jetson CSI-2 Port 1

Light Sensor ─→ I2C ─→ Jetson
IMU Sensor ─→ I2C ─→ Jetson

Jetson GPIO ─→ IR-CUT Control (ADAS camera)
Jetson PWM ─→ IR LED PWM Control
```

---

## 11. Cost Summary

| Configuration | Development | Production |
|--------------|-------------|------------|
| **Core System** | $799 | $1,350 |
| **Cameras** | $165 | $215 |
| **Power** | $0 | $120 |
| **Storage** | $35 | $35 |
| **Cooling** | $25 | $40 |
| **Enclosure** | $25 | $120 |
| **Sensors** | $15 | $92 |
| **Misc** | $80 | $100 |
| **TOTAL** | **$1,144** | **$2,072** |

**Per-Unit Cost at Scale (1000+ units):**
- Bulk pricing: -30% on most components
- **Estimated:** $1,450 per unit

---

## 12. Development Roadmap Hardware Needs

### Phase 1: Current (Basic Inference) ✅
**Hardware:** Jetson Orin NX Dev Kit + IMX219 cameras (already have?)
**Cost:** $0 (assuming you have this)

### Phase 2: Transfer Learning Training (Next)
**Hardware:** Add NVMe SSD (256GB) for datasets + models
**Cost:** $35

### Phase 3: RL Training
**Hardware:** External GPU workstation (for CARLA simulation)
**Recommended:** Desktop with RTX 3080+ ($800-1500)
**Why:** RL training in CARLA is compute-intensive, offload to desktop

### Phase 4: Full ADAS/DMS (Production Prototype)
**Hardware:** Automotive cameras (IMX390 + OV7251) + IR LEDs
**Cost:** $215

### Phase 5: Vehicle Integration
**Hardware:** Full production BOM
**Cost:** $2,072

**Immediate Purchase (This Week):**
- [ ] NVMe SSD 256GB ($35) - for datasets and models
- [ ] Light sensor TSL2561 ($5) - for auto IR-CUT switching
- [ ] Total: $40

**Next Month:**
- [ ] Automotive ADAS camera ($85-110)
- [ ] DMS camera ($50-55)
- [ ] IR LEDs + drivers ($30-50)
- [ ] Total: ~$200

---

## 13. Alternative Configurations

### Budget Configuration ($800 total)

- Jetson Orin Nano 8GB Dev Kit: $499
- 2x Raspberry Pi Camera v2.1 (IMX219): $50
- 128GB NVMe SSD: $25
- IR LEDs (basic): $20
- Acrylic case: $25
- Cables/misc: $30
- **Total: $649**

**Trade-offs:**
- Less RAM (8GB vs 16GB) - limits model size
- No DLA - must run all inference on GPU
- Cheaper cameras - lower image quality
- Still functional for prototyping

### Premium Configuration ($4,000 total)

- Jetson Orin AGX 64GB: $1,999
- Connect Tech Rogue Carrier: $900
- 2x IMX490 cameras (premium): $220
- 1TB NVMe SSD: $90
- Automotive power + backup: $200
- IP67 enclosure with thermal management: $300
- Full sensor suite (GPS, IMU, CAN, cellular): $150
- Professional integration: $141
- **Total: $4,000**

**Benefits:**
- Overkill compute (200 TOPS) - future-proof
- Best cameras (IMX490)
- Large storage for on-device training
- Production-ready automotive integration

---

**Recommendation for Your Phase:**
Start with **Development BOM ($1,144)**, upgrade cameras later when ready for vehicle deployment. The Orin NX 16GB gives you room to grow without overspending on AGX.

Let me know if you'd like help sourcing any specific components or need alternatives!
