import streamlit as st
import pandas as pd
import numpy as np
import os
import glob

# ============================================================
# NavDR — AI-Powered Resilient Navigation Dashboard
# ============================================================

st.set_page_config(
    page_title="NavDR — Resilient Navigation",
    page_icon="🛰️",
    layout="wide"
)

# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

st.title("🛰️ NavDR")
st.subheader("AI-Powered Smartphone Dead Reckoning for GNSS-Denied Navigation")

st.markdown(
    """
    **Real-time concept prototype**

    NavDR combines smartphone IMU sensing, AI-based speed estimation,
    inertial dead reckoning, Non-Holonomic Constraints (NHC), and GNSS
    recovery to maintain navigation during temporary GNSS outages.
    """
)

st.divider()


# ============================================================
# FIND RESULT FILE
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

possible_files = [
    os.path.join(PROJECT_ROOT, "results", "navdr_ai_results.csv"),
    os.path.join(PROJECT_ROOT, "results", "navdr_results.csv"),
    os.path.join(PROJECT_ROOT, "results", "navdr_fusion_results.csv"),
]

result_file = None

for file in possible_files:
    if os.path.exists(file):
        result_file = file
        break

if result_file is None:

    st.error(
        "No NavDR result CSV found. Run the AI dead-reckoning pipeline first."
    )

    st.code(
        "python src\\ai_dead_reckoning.py",
        language="powershell"
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:
    df = pd.read_csv(result_file)
except Exception as e:
    st.error(f"Could not read result file: {e}")
    st.stop()


# ------------------------------------------------------------
# Normalize column names
# ------------------------------------------------------------

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
)

# Convert possible numeric columns

for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except Exception:
        pass


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(possible_names):

    for name in possible_names:

        if name in df.columns:
            return name

    # relaxed matching

    for col in df.columns:

        col_lower = col.lower()

        for name in possible_names:

            if name.lower() in col_lower:
                return col

    return None


time_col = find_column([
    "time",
    "Time",
    "TIME",
    "timestamp",
    "seconds"
])

speed_ref_col = find_column([
    "gnss_speed",
    "GPS_SPEED",
    "gps_speed",
    "speed_reference",
    "reference_speed"
])

speed_ai_col = find_column([
    "ai_speed",
    "predicted_speed",
    "AI_Predicted_Speed",
    "ai_predicted_speed",
    "predicted"
])

error_col = find_column([
    "position_error",
    "NavDR_Position_Error",
    "error",
    "position_error_m"
])

east_col = find_column([
    "dr_east",
    "ai_east",
    "east",
    "navdr_east",
    "x"
])

north_col = find_column([
    "dr_north",
    "ai_north",
    "north",
    "navdr_north",
    "y"
])

heading_col = find_column([
    "heading",
    "estimated_heading",
    "NavDR_Heading"
])


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ NavDR Configuration")

st.sidebar.success("AI Engine: GRU")
st.sidebar.info("IMU Sampling: 10 Hz")
st.sidebar.info("GNSS Mode: Simulated Blackout")

st.sidebar.markdown("---")

st.sidebar.markdown("### Navigation Pipeline")

st.sidebar.markdown(
    """
    🟢 Smartphone IMU

    ↓

    🧠 GRU Speed Estimation

    ↓

    🧭 Gyroscope Heading

    ↓

    🚗 Dead Reckoning

    ↓

    📐 NHC Constraint

    ↓

    🛰️ GNSS Recovery
    """
)


# ============================================================
# METRICS
# ============================================================

st.header("📊 Navigation Status")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "GNSS Status",
        "SIGNAL LOST",
        "60 s blackout"
    )

with col2:

    if speed_ai_col:

        latest_speed = float(df[speed_ai_col].iloc[-1])

        st.metric(
            "AI Speed",
            f"{latest_speed:.1f} km/h"
        )

    else:
        st.metric("AI Speed", "—")


with col3:

    if heading_col:

        latest_heading = float(df[heading_col].iloc[-1])

        st.metric(
            "Heading",
            f"{latest_heading:.1f}°"
        )

    else:
        st.metric(
            "Heading",
            "INS"
        )


with col4:

    if error_col:

        max_error = float(df[error_col].max())

        st.metric(
            "Max Position Error",
            f"{max_error:.1f} m"
        )

    else:
        st.metric(
            "Position Error",
            "—"
        )


with col5:

    st.metric(
        "IMU Rate",
        "10 Hz"
    )


st.divider()


# ============================================================
# GNSS STATUS PANEL
# ============================================================

st.header("🛰️ GNSS Availability")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:

    st.success("GNSS AVAILABLE")

    st.caption(
        "Normal operation: GNSS position is used for absolute correction."
    )


with status_col2:

    st.warning("GNSS LOST")

    st.caption(
        "NavDR automatically switches to AI-assisted dead reckoning."
    )


with status_col3:

    st.info("GNSS RESTORED")

    st.caption(
        "Navigation state can be re-anchored to GNSS."
    )


st.divider()


# ============================================================
# SPEED ESTIMATION
# ============================================================

st.header("🧠 AI Speed Estimation")

if speed_ref_col and speed_ai_col and time_col:

    speed_chart = df[
        [time_col, speed_ref_col, speed_ai_col]
    ].copy()

    speed_chart = speed_chart.rename(
        columns={
            time_col: "Time",
            speed_ref_col: "GNSS Reference",
            speed_ai_col: "AI Predicted"
        }
    )

    speed_chart = speed_chart.set_index("Time")

    st.line_chart(speed_chart)

    st.caption(
        "GRU estimates vehicle speed from a temporal window of smartphone IMU measurements."
    )

else:

    st.warning(
        "Speed columns were not detected in the result CSV."
    )


# ============================================================
# TRAJECTORY
# ============================================================

st.header("📍 Navigation Trajectory")

if east_col and north_col:

    trajectory = df[
        [east_col, north_col]
    ].copy()

    trajectory.columns = [
        "East displacement",
        "North displacement"
    ]

    trajectory.index.name = "Sample"

    st.line_chart(trajectory)

    st.caption(
        "Estimated trajectory generated by the NavDR dead-reckoning engine."
    )

else:

    st.info(
        "Trajectory columns were not detected. "
        "The CSV may contain only navigation metrics."
    )


# ============================================================
# POSITION ERROR
# ============================================================

st.header("📏 Position Error During GNSS Blackout")

if error_col and time_col:

    error_chart = df[
        [time_col, error_col]
    ].copy()

    error_chart = error_chart.rename(
        columns={
            time_col: "Time",
            error_col: "Position Error (m)"
        }
    )

    error_chart = error_chart.set_index("Time")

    st.line_chart(error_chart)

    max_error = float(df[error_col].max())
    mean_error = float(df[error_col].mean())
    final_error = float(df[error_col].iloc[-1])

    e1, e2, e3 = st.columns(3)

    with e1:
        st.metric(
            "Mean Error",
            f"{mean_error:.2f} m"
        )

    with e2:
        st.metric(
            "Maximum Error",
            f"{max_error:.2f} m"
        )

    with e3:
        st.metric(
            "Final Error",
            f"{final_error:.2f} m"
        )

else:

    st.warning(
        "Position-error column was not detected."
    )


# ============================================================
# HEADING
# ============================================================

if heading_col and time_col:

    st.header("🧭 Estimated Heading")

    heading_chart = df[
        [time_col, heading_col]
    ].copy()

    heading_chart = heading_chart.rename(
        columns={
            time_col: "Time",
            heading_col: "Heading"
        }
    )

    heading_chart = heading_chart.set_index("Time")

    st.line_chart(heading_chart)


# ============================================================
# SYSTEM ARCHITECTURE
# ============================================================

st.divider()

st.header("🏗️ NavDR System Architecture")

st.markdown(
    """
    ### Smartphone Sensors
    **Accelerometer + Gyroscope**
    
    ↓
    
    ### Signal Processing
    Calibration → Noise filtering → Motion analysis
    
    ↓
    
    ### AI Layer
    **GRU-based speed estimation**
    
    ↓
    
    ### Navigation Engine
    Speed + Heading → Dead Reckoning
    
    ↓
    
    ### Physical Constraint
    **Non-Holonomic Constraint (NHC)**
    
    ↓
    
    ### GNSS Fusion
    GNSS available → correction  
    GNSS unavailable → inertial prediction
    
    ↓
    
    ### Output
    **Continuous vehicle position**
    """
)


# ============================================================
# RESULTS
# ============================================================

st.divider()

st.header("📋 Prototype Information")

info1, info2 = st.columns(2)

with info1:

    st.markdown(
        f"""
        **Dataset**

        IO-VNBD

        **Sampling Frequency**

        10 Hz

        **GNSS Blackout**

        1200 s → 1260 s

        **AI Model**

        GRU
        """
    )


with info2:

    st.markdown(
        """
        **Input Sensors**

        • Accelerometer  
        • Gyroscope

        **Navigation**

        • Dead Reckoning  
        • Heading estimation  
        • NHC  
        • GNSS recovery

        **Deployment Target**

        Smartphone / Edge Device
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NavDR — AI/ML Enhanced Smartphone Dead Reckoning & GNSS Fusion | SIH 2026 Prototype"
)