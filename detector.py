from ultralytics import YOLO
import cv2


class DroneDetector:

    def __init__(self, model_path):

        print("[INFO] Loading YOLO model...")

        self.model = YOLO(model_path)

        print("[INFO] YOLO Loaded Successfully.")

    def process_video(self, input_video, output_video):

        cap = cv2.VideoCapture(input_video)

        if not cap.isOpened():
            raise Exception(f"Cannot open {input_video}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        writer = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        frame_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = self.model(frame, verbose=False)

            annotated = results[0].plot()

            writer.write(annotated)

            cv2.imshow("Detection", annotated)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        print(f"\nProcessed {frame_count} frames.")
        print(f"Output saved to {output_video}")