from ultralytics import YOLO


class DroneTracker:

    def __init__(self, model_path):

        print("[INFO] Loading YOLO + ByteTrack...")

        self.model = YOLO(model_path)

        print("[INFO] Tracker Ready.")

    def track(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        return results