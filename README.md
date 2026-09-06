# 🚗 NavDR — AI-Powered Resilient Vehicle Navigation

### AI/ML Enhanced Smartphone Dead Reckoning & GNSS Fusion for Resilient Vehicle Navigation

NavDR is an AI-assisted navigation prototype designed to maintain vehicle positioning when GNSS/GPS signals become temporarily unavailable.

The system uses smartphone accelerometer and gyroscope data, AI-based speed estimation, inertial dead reckoning, Non-Holonomic Constraints (NHC), and GNSS recovery to provide continuous vehicle navigation without requiring OBD-II or dedicated vehicle hardware.

---

## 🎯 Problem

GNSS-based navigation can become unreliable or unavailable in:

* 🚇 Tunnels
* 🛣️ Underpasses
* 🏙️ Urban canyons
* 🌲 Forests
* ⛰️ Valleys
* 📡 GNSS interference or temporary signal loss

During these outages, conventional navigation systems may lose accurate positioning.

Consumer smartphones already contain IMU sensors that provide continuous motion information. However, raw smartphone IMU data contains noise, bias, vibration, and orientation errors.

---

## 💡 Our Solution

NavDR combines **AI + inertial navigation + physical constraints** to estimate vehicle motion during GNSS outages.

### Core Pipeline

```text
                 Smartphone
              Accelerometer + Gyroscope
                        │
                        ▼
               Signal Processing
             Calibration & Filtering
                        │
                        ▼
              AI Speed Estimation
                   GRU Model
                        │
                        ▼
                Heading Estimation
                        │
                        ▼
              Dead Reckoning Engine
                        │
                        ▼
              NHC Constraint Layer
                        │
              ┌─────────┴─────────┐
              │                   │
        GNSS Available        GNSS Lost
              │                   │
              ▼                   ▼
       GNSS + INS Fusion     Inertial Prediction
              │                   │
              └─────────┬─────────┘
                        ▼
               Continuous Position
```

---

## 🚀 Key Features

* 📱 Smartphone-only sensing
* 🤖 GRU-based AI speed estimation
* 🧭 Gyroscope-based heading estimation
* 📍 Inertial dead reckoning
* 🚫 GNSS blackout simulation
* 🛞 Non-Holonomic Constraint (NHC)
* 🔄 GNSS recovery and re-anchoring
* 🧠 AI-assisted IMU motion filtering
* 💻 Edge-deployment oriented architecture
* 📊 Interactive Streamlit dashboard
* 📚 Trained and evaluated using the IO-VNBD dataset

---

## 🧠 AI Component

NavDR uses a **Gated Recurrent Unit (GRU)** neural network to estimate vehicle speed from temporal smartphone IMU measurements.

### Input Sensors

The model uses:

* Accelerometer X
* Accelerometer Y
* Accelerometer Z
* Gyroscope Yaw
* Gyroscope Pitch
* Gyroscope Roll

A temporal window of IMU samples is provided to the GRU.

### Why GRU?

Vehicle motion is temporal. A single IMU measurement does not contain enough information to reliably estimate motion.

GRU can learn patterns across a sequence of sensor measurements while being lighter than many larger recurrent architectures, making it suitable for edge-oriented applications.

---

## 📊 Preliminary AI Results

The preliminary GRU V2 model achieved:

| Metric          |         Result |
| --------------- | -------------: |
| MAE             |  **3.38 km/h** |
| RMSE            |  **4.29 km/h** |
| R²              |     **0.0414** |
| Sampling Rate   |      **10 Hz** |
| Sequence Length | **20 samples** |

These results represent the current preliminary AI speed-estimation prototype.

The navigation pipeline is still under optimization, particularly for heading estimation and long-duration inertial drift.

---

## 🛰️ GNSS-Denied Navigation

NavDR simulates a temporary GNSS outage.

### Normal Operation

```text
GNSS + IMU
   ↓
Fusion
   ↓
Vehicle Position
```

### During GNSS Loss

```text
IMU
 ↓
AI Speed Estimation
 ↓
Heading Estimation
 ↓
Dead Reckoning
 ↓
NHC
 ↓
Estimated Position
```

### After GNSS Recovery

```text
GNSS Returns
     ↓
State Correction
     ↓
Re-anchor Navigation
     ↓
Continue Tracking
```

---

## 🛞 Non-Holonomic Constraint

Ground vehicles generally cannot move freely sideways or vertically like aerial vehicles.

NavDR uses this physical property as a constraint during navigation.

The NHC layer helps prevent physically unrealistic motion estimates caused by noisy smartphone IMU measurements.

---

## 🧪 Dataset

The project uses the **IO-VNBD dataset** for preliminary training and evaluation.

IO-VNBD provides synchronized smartphone and vehicle sensor data that can be used for intelligent vehicle navigation research.

### Dataset

[https://github.com/onyekpeu/IO-VNBD](https://github.com/onyekpeu/IO-VNBD)

### Research Paper

[https://doi.org/10.1016/j.dib.2021.106885](https://doi.org/10.1016/j.dib.2021.106885)

---

## 🏗️ Project Structure

```text
NavDR/
│
├── app/
│   └── dashboard.py
│
├── models/
│   ├── imu_anomaly_model.pkl
│   ├── speed_feature_scaler_v2.pkl
│   ├── speed_gru_v2.pth
│   └── speed_target_scaler_v2.pkl
│
├── notebooks/
│   ├── 01_explore_data.py
│   └── 02_visualize_data.py
│
├── results/
│   ├── navdr_final_results.csv
│   ├── navdr_final_trajectory.png
│   ├── navdr_ai_v2_results.csv
│   └── ...
│
├── src/
│   ├── ai_dead_reckoning.py
│   ├── ai_motion_filter.py
│   ├── calibrated_heading.py
│   ├── dead_reckoning.py
│   ├── dead_reckoning_v2.py
│   ├── ekf_fusion.py
│   ├── fusion_engine.py
│   ├── gnss_fusion_demo.py
│   ├── heading_diagnostic.py
│   ├── map_matching.py
│   ├── navdr_engine.py
│   ├── nhc_fusion.py
│   ├── train_gru.py
│   ├── train_speed_gru.py
│   └── train_speed_model.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Likitha3126/NavDR.git
cd NavDR
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🖥️ Run the Dashboard

The project includes a Streamlit-based demonstration dashboard.

Run:

```bash
streamlit run app/dashboard.py
```

The dashboard displays:

* GNSS status
* AI-estimated speed
* Heading
* Navigation trajectory
* Position error
* AI speed estimation graph
* Navigation architecture
* Prototype configuration

---

# 🧪 Run the AI Pipeline

If the IO-VNBD dataset is available locally, the navigation pipeline can be executed using:

```bash
python src/ai_dead_reckoning.py
```

Other experimental navigation modules are available inside the `src/` directory.

---

# 🧰 Technologies Used

### Programming

* Python

### AI / ML

* PyTorch
* GRU
* Scikit-learn
* Isolation Forest

### Navigation

* GNSS
* IMU
* Dead Reckoning
* Heading Estimation
* Non-Holonomic Constraints
* GNSS/INS Fusion

### Data Processing

* NumPy
* Pandas
* SciPy

### Visualization / Demo

* Matplotlib
* Streamlit

### Dataset

* IO-VNBD

---

# 🎯 Target Applications

NavDR can be extended for:

* 🚚 Commercial trucks
* 🛵 Two-wheelers
* 🚗 Older vehicles without advanced navigation hardware
* 🚑 Emergency and rescue vehicles
* 🗺️ Navigation applications
* 🚇 Tunnel navigation
* 🏙️ Urban environments
* 📡 GNSS-denied environments

---

# 🌟 Innovation

The key idea behind NavDR is combining **AI-based motion estimation with physics-based navigation constraints** using sensors that already exist in consumer smartphones.

### Our approach avoids requiring:

* ❌ OBD-II connection
* ❌ Vehicle wheel-speed sensors
* ❌ Dedicated navigation computer
* ❌ Expensive vehicle INS hardware

Instead:

```text
Consumer Smartphone
        +
      AI/ML
        +
Physics-based Constraints
        +
   GNSS/INS Fusion
        ↓
 Resilient Navigation
```

---

# 🔮 Future Improvements

The current prototype provides a foundation for further development.

Future work includes:

* Real-time smartphone sensor integration
* Improved orientation and heading estimation
* Advanced GNSS/INS sensor fusion
* Extended Kalman Filter optimization
* Robust map matching using OpenStreetMap
* Better handling of phone misalignment
* Vehicle-motion classification
* Improved long-duration drift correction
* External IMU support
* On-device model optimization
* Real-world vehicle testing
* Lane-level positioning

---

# ⚠️ Current Prototype Status

NavDR is currently a **research and demonstration prototype**.

The AI speed-estimation component has been trained and evaluated using IO-VNBD data.

The navigation pipeline demonstrates the intended integration of:

```text
AI
+
IMU
+
Dead Reckoning
+
NHC
+
GNSS Recovery
```

Further optimization is required to achieve production-level long-duration navigation accuracy.

---

# 👥 Team

Developed for:

**SMART INDIA HACKATHON 2026**

### Project

**AI/ML Enhanced Smartphone Dead Reckoning & GNSS Fusion for Resilient Vehicle Navigation**

### Solution

**NavDR — AI-Powered Resilient Navigation**

---

# 📚 References

1. **IO-VNBD Dataset**
   [https://github.com/onyekpeu/IO-VNBD](https://github.com/onyekpeu/IO-VNBD)

2. **IO-VNBD Research Paper**
   [https://doi.org/10.1016/j.dib.2021.106885](https://doi.org/10.1016/j.dib.2021.106885)

3. **OpenStreetMap**
   [https://www.openstreetmap.org/](https://www.openstreetmap.org/)

---

⭐ **NavDR — Making smartphone navigation more resilient when GNSS disappears.**
