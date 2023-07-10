import cv2
from detector import YOLOv5Detector

def main():
    # Initialize the detector
    detector = YOLOv5Detector()
    
    # Load a sample image (ensure it's in your data folder)
    image = cv2.imread("data/sample_image.jpg")
    if image is None:
        print("Error: Sample image not found!")
        return

    # Print the image dimensions (height, width, channels)
    print("Original image dimensions:", image.shape)
    
    # Convert from BGR to RGB for detection
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run detection
    detections = detector.detect(image_rgb)
    
    # Print out detection coordinates and confidence values
    print("Detections:")
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        print(f"Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), Confidence: {conf:.2f}, Class: {cls}")
    
    # Draw detections on the original image
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f"{conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    
    #Save the image with detections
    cv2.imwrite("data/test_detector_output.jpg", image)

if __name__ == "__main__":
    main()
