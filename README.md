# End-to-End Video Interaction & Custom YOLOv8 Pipeline

## Overview
This project is a complete Computer Vision system designed to detect specific interactions and identities within video footage. It combines a **custom-trained YOLOv8 model** (detecting "Managers" and "Tables") with a scalable engineering pipeline (Docker & AWS Batch) to analyze video feeds for "touch" events in defined zones.

The repository contains both the **application logic** for processing videos and the **data science environment** used to train and evaluate the object detection model.

## Project Components

### 1. Computer Vision Application
The core application ingests video, runs inference, and generates analytical reports.
* **Video Processing:** `process_video.py` / `process_video_v2.py` run the detection loop.
* **Interaction Logic:** `find_touches.py` calculates overlaps between detected objects and zones defined in `table_zones.json`.
* **Face Recognition:** Specific logic handles "Manager" identification using reference images in `known_faces/`.
* **Infrastructure:**
    * **Docker:** The application is containerized via `Dockerfile` for consistent deployment.
    * **AWS Batch:** `submit_jobs.py` orchestrates parallel processing of video datasets in the cloud.

### 2. Custom Model Training (YOLOv8)
The object detection model was fine-tuned on a custom dataset to specifically identify project-relevant classes.
* **Configuration:** `custom_data.yaml` defines the dataset paths and class mappings.
* **Dataset:** `images/` (Train/Val) and `labels/` contain the annotated data.
* **Classes:**
    * `0`: Table
    * `1`: Manager
* **Performance Tracking:** The `runs/` directory contains MLflow logs (`runs/mlflow/`) and training metrics (confusion matrices, F1 curves).

## Repository Structure

```text
├── custom_data.yaml       # YOLOv8 dataset configuration
├── Dockerfile             # Container definition for AWS Batch
├── process_video.py       # Main inference script
├── find_touches.py        # Interaction detection logic
├── submit_jobs.py         # AWS Batch job submission script
├── table_zones.json       # Coordinates for interaction zones
├── requirements.txt       # Python dependencies
├── yolov8n.pt             # Base model weights
│
├── images/                # Training and Validation images
├── labels/                # YOLO format annotations
├── known_faces/           # Reference images for specific identity matching
├── debug_videos/          # Output folder for visualized inferences
├── final_reports/         # CSV/JSON summaries of detections
│
└── runs/                  # Training artifacts & MLflow logs
    └── detect/train2/     # Best model weights & performance plots
