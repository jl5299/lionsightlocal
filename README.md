# Lionsight Live Inference Dashboard

This application provides a real-time object detection dashboard using YOLOv8 and Streamlit. It allows you to:
- Perform live object detection using your webcam
- Select specific classes to detect
- Adjust confidence thresholds
- View real-time detection statistics
- Track object counts over time
- Save detection data to CSV files

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## Features

- **Live Video Feed**: Shows real-time object detection with bounding boxes
- **Class Selection**: Choose which objects to detect from the COCO dataset
- **Confidence Threshold**: Adjust the minimum confidence level for detections
- **Real-time Statistics**: View the number of detected objects over time
- **Cumulative Tracking**: See the total number of objects detected since starting
- **Data Export**: Automatic CSV export of detection data every minute

## Usage

1. Click the "Start/Stop Detection" button to begin/end the detection process
2. Use the sidebar to:
   - Select which classes to detect
   - Adjust the confidence threshold
3. View the live video feed and detection statistics
4. Data is automatically saved to CSV files in the current directory

## Notes

- The application uses the YOLO11n model by default
- Webcam access is required for the live feed
- CSV files are saved with timestamps in the filename 