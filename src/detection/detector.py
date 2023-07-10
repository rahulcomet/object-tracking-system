import torch

class YOLOv5Detector:
    def __init__(self, model_name='yolov5s'):
        # Load the YOLOv5 model from Torch Hub (pre-trained on COCO)
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
    
    def detect(self, frame):
        # Run detection on the input frame and return results
        results = self.model(frame)
        # returns detections as a tensor with shape: [x1, y1, x2, y2, confidence, class]
        return results.xyxy[0]
