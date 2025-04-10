# Get available classes from COCO dataset
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from ultralytics import YOLO
import os
from pathlib import Path

# Page config must be the first Streamlit command
st.set_page_config(
    layout="wide",
    page_title="Live Video Processing",
    page_icon="🦁",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'detection_data' not in st.session_state:
    st.session_state.detection_data = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'csv_save_path' not in st.session_state:
    st.session_state.csv_save_path = None

# Load external CSS
with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Create data directory if it doesn't exist
DATA_DIR = Path("detection_data")
DATA_DIR.mkdir(exist_ok=True)

# Sidebar with company logo
try:
    logo_path = Path("assets/logo.webp")
    if logo_path.exists():
        st.sidebar.image(str(logo_path), width=200)
    else:
        st.sidebar.title("Lionsight")
except Exception as e:
    st.sidebar.title("Lionsight")

# Main content
st.title("Live Video Processing")

# Detection settings in sidebar
st.sidebar.header("Detection Settings")

# Camera selection with modern styling
available_cameras = []
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        available_cameras.append(f"Camera {i}")
        cap.release()

if not available_cameras:
    st.error("No cameras found. Please connect a camera and refresh the page.")
    st.stop()

selected_camera = st.sidebar.selectbox(
    "Select Camera",
    available_cameras,
    index=0
)
camera_index = int(selected_camera.split()[-1])

# Modern controls
enable_tracking = st.sidebar.checkbox("Enable Object Tracking", value=True)
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    format="%0.2f"
)

# Class selection with search
selected_classes = st.sidebar.multiselect(
    "Select Classes to Detect",
    COCO_CLASSES,
    default=["person"]
)

# Start/Stop button at the bottom of the sidebar
st.sidebar.markdown("---")  # Add a separator line
if st.sidebar.button("Start/Stop Detection"):
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()
        # Create new CSV file for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.csv_save_path = DATA_DIR / f"detection_data_{timestamp}.csv"
    else:
        st.session_state.start_time = None
        # Save final data when stopping
        if st.session_state.detection_data:
            df = pd.DataFrame(st.session_state.detection_data)
            df.to_csv(st.session_state.csv_save_path, index=False)
            st.success(f"Data saved to {st.session_state.csv_save_path}")

# Initialize YOLO model
@st.cache_resource
def load_model():
    model_path = 'yolo11n.pt'
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found in the current directory. Please ensure the model file is present.")
        return None
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()
if model is None:
    st.stop()

# Main content layout
col1, col2 = st.columns([7, 3])

with col1:
    st.title("Live Detection")
    video_placeholder = st.empty()
    
    # Detection metrics in a row
    metrics_cols = st.columns(4)
    with metrics_cols[0]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">98%</div>
            <div class="metric-label">Confidence Level</div>
        </div>
        """, unsafe_allow_html=True)
    with metrics_cols[1]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" id="processed-frames">0</div>
            <div class="metric-label">Frames Processed</div>
        </div>
        """, unsafe_allow_html=True)
    with metrics_cols[2]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" id="total-detections">0</div>
            <div class="metric-label">Total Detections</div>
        </div>
        """, unsafe_allow_html=True)
    with metrics_cols[3]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" id="fps">0</div>
            <div class="metric-label">FPS</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.title("Insights")
    tabs = st.tabs(["Real-time", "Analytics"])
    
    with tabs[0]:
        st.subheader("Detection Trends")
        graph1_placeholder = st.empty()
        
        st.subheader("Cumulative Counts")
        graph2_placeholder = st.empty()
        
        # Key metrics
        st.markdown("""
        <div class="metric-card">
            <h4>Key Insights</h4>
            <div class="metric-row">
                <span class="metric-label">Average Dwell Time:</span>
                <span class="metric-value">+191%</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Detection Rate:</span>
                <span class="metric-value">+82%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Video capture and processing
if st.session_state.start_time is not None:
    cap = cv2.VideoCapture(camera_index)
    
    while st.session_state.start_time is not None:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera")
            break
        
        # Run YOLO inference with tracking if enabled
        if enable_tracking:
            results = model.track(
                frame,
                conf=confidence_threshold,
                classes=[COCO_CLASSES.index(cls) for cls in selected_classes],
                persist=True,
                tracker="bytetrack.yaml"  # Using ByteTrack for better performance
            )
        else:
            results = model(
                frame,
                conf=confidence_threshold,
                classes=[COCO_CLASSES.index(cls) for cls in selected_classes]
            )
        
        # Process results
        annotated_frame = results[0].plot()
        
        # Count detections with tracking IDs
        current_time = time.time() - st.session_state.start_time
        detections = {cls: 0 for cls in selected_classes}
        tracked_ids = set()  # Keep track of unique IDs
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                cls_name = COCO_CLASSES[cls]
                if cls_name in selected_classes:
                    # If tracking is enabled, use the tracking ID
                    if enable_tracking and hasattr(box, 'id') and box.id is not None:
                        if box.id not in tracked_ids:
                            tracked_ids.add(box.id)
                            detections[cls_name] += 1
                    else:
                        detections[cls_name] += 1
        
        # Add to session state
        st.session_state.detection_data.append({
            'timestamp': current_time,
            'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **detections
        })
        
        # Update video feed
        video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
        
        # Update graphs
        if len(st.session_state.detection_data) > 0:
            df = pd.DataFrame(st.session_state.detection_data)
            
            # Time series graph
            fig1 = px.line(df, x='datetime', y=selected_classes,
                         title="Objects Detected Over Time")
            fig1.update_xaxes(title_text="Time")
            fig1.update_yaxes(title_text="Count")
            graph1_placeholder.plotly_chart(fig1, use_container_width=True)
            
            # Cumulative graph
            df_cumulative = df.copy()
            for cls in selected_classes:
                df_cumulative[cls] = df_cumulative[cls].cumsum()
            
            fig2 = px.line(df_cumulative, x='datetime', y=selected_classes,
                         title="Cumulative Objects Detected")
            fig2.update_xaxes(title_text="Time")
            fig2.update_yaxes(title_text="Cumulative Count")
            graph2_placeholder.plotly_chart(fig2, use_container_width=True)
            
            # Save data to CSV every minute
            if len(st.session_state.detection_data) % 60 == 0:
                df.to_csv(st.session_state.csv_save_path, index=False)
    
    cap.release()

# Analytics tabs
tabs = st.tabs(["Overview", "Trends", "Reports"])

with tabs[0]:
    # Summary metrics
    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">24,750</div>
            <div class="metric-label">Total Detections</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">1,248</div>
            <div class="metric-label">Unique Objects</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">88.6%</div>
            <div class="metric-label">Avg. Confidence</div>
        </div>
        """, unsafe_allow_html=True)

with tabs[1]:
    st.title("Analytics Dashboard")
    # Add analytics content here

with tabs[2]:
    st.title("Analytics Reports")
    # Add reports content here 