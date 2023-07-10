import cv2
import argparse
from detection.detector import YOLOv5Detector
from tracking.tracker import ObjectTracker
from helpers.video_utils import init_video_writer

def shrink_box(bbox, shrink_factor=0.5):
    """
    Shrinks the bounding box by a given factor but keeps the top-left corner fixed.
    bbox: list or tuple [x1, y1, x2, y2]
    shrink_factor: factor to scale the width and height (e.g., 0.9 reduces width and height to 90%)
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    new_x2 = x1 + shrink_factor * w
    new_y2 = y1 + shrink_factor * h
    return [x1, y1, new_x2, new_y2]


def main(video_path, output_path, conf_threshold=0.5):
    # Open the input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = init_video_writer(output_path, width, height, fps)

    # Initialize detector and tracker
    detector = YOLOv5Detector()
    tracker = ObjectTracker(max_age=30)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame from BGR to RGB for detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections_tensor = detector.detect(frame_rgb)
        detection_list = []
        for det in detections_tensor:
            if det[4] >= conf_threshold and int(det[5]) == 0:  # Only 'person' detections
                detection_list.append([[float(det[0]), float(det[1]), float(det[2]), float(det[3])],
                                        float(det[4]), int(det[5])])

        
        # Update tracker with current detections (tracker handles the temporal associations)
        tracks = tracker.update(detection_list, frame)
        
        # Draw each track's bounding box and track ID on the frame
        for track in tracks:
            bbox = track.to_ltrb()  # bounding box in [x1, y1, x2, y2]
            bbox_shrunk = shrink_box(bbox, shrink_factor= 0.5)
            xmin, ymin, xmax, ymax = map(int, bbox_shrunk)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track.track_id}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        writer.write(frame)
        cv2.imshow("Object Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("Tracking complete. Output saved to:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the object tracking pipeline using YOLOv5 and DeepSORT.")
    parser.add_argument("--video", type=str, default="data/sample_video.webm", help="Path to the input video")
    parser.add_argument("--output", type=str, default="demos/tracking_output.avi", help="Path to save the output video")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    args = parser.parse_args()
    
    main(args.video, args.output, args.conf)
