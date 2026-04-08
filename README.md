# AI Cattle Monitoring

A computer vision prototype for real-time cattle detection using YOLO.  
It processes video frames, runs object detection, and visualizes results to validate a practical monitoring pipeline.

## What This Code Does

- `realtime_cattle_detector.py`  
  Reads video/camera frames, performs YOLO inference, and renders detection boxes in real time.

- `test_cam1_api.py`  
  Tests API-facing data flow for detection outputs before backend integration.

- `requirements.txt`, `api_requirements.txt`  
  Separates runtime dependencies for detection and API testing to keep setup reproducible.

- `yolov8n.pt`  
  Model weights used for inference.

## Tech Stack

Python, YOLO (Ultralytics), OpenCV

## Why It Matters

This repository serves as the ML baseline for the Smart Farm Monitoring system, validating the core pipeline:  
**video input -> object detection -> result delivery**.
