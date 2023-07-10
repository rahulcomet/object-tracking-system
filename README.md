# Object Tracking System

This repository contains a real-time object tracking system that integrates YOLOv5 for object detection, DeepSORT for tracking, and optionally TensorRT for model optimization. The project is designed to work with publicly available datasets (e.g., MOT Challenge) or custom video data.

## Project Overview

- **Detection:**  
  YOLOv5 (implemented in PyTorch) detects objects in each video frame. The pre-trained model (on the COCO dataset) provides bounding boxes and confidence scores.
  
- **Tracking:**  
  DeepSORT is used to maintain consistent tracking IDs for detected objects across frames. The tracker handles temporal associations and re-identification.
  
- **Optimization (Optional):**  
  TensorRT can be used to optimize the YOLOv5 model for faster inference on supported hardware.
  
- **Demo:**  
  The system processes video input frame-by-frame, outputs annotated frames with bounding boxes and tracking IDs, and saves the final result as a video.

## Usage

To run the full tracking pipeline on a sample video (e.g., a WebM file from MOT Challenge), execute the following command from the project root:

```bash
python -m src.main --video data/sample_video.webm --output demos/tracking_output.avi --conf 0.5
```

### Command-Line Arguments

- `--video`: Path to the input video file.
- `--output`: Path where the output video with annotations will be saved.
- `--conf`: Detection confidence threshold (default is 0.5).
