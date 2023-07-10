from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self, max_age=30):
        # Initialize DeepSORT with a default max_age (adjust as needed)
        self.tracker = DeepSort(max_age=max_age)
    
    def update(self, detections, frame):
        # Update the tracker with detections for the current frame.
        # Expected detections format: list of [x1, y1, x2, y2, confidence]
        tracks = self.tracker.update_tracks(detections, frame=frame)
        return tracks
