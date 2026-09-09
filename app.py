# ============================================================
# DRONE THREAT DETECTION SYSTEM
# ============================================================

from csv import writer
from pyexpat import features
import cv2
import os
import statistics

from tracker import DroneTracker
from feature_extractor import FeatureExtractor
from lstm_predictor import LSTMPredictor
from threat import ThreatAssessment


# ============================================================
# PATHS
# ============================================================

YOLO_MODEL = "best_model.pt"

LSTM_MODEL = "best_lstm_model.pth"

INPUT_SCALER = "input_scaler.pkl"

TARGET_SCALER = "target_scaler.pkl"

VIDEO_PATH = "video/video_04.mp4"

OUTPUT_PATH = "outputs/video23_final.mp4"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DRONE THREAT DETECTION SYSTEM")
    print("=" * 60)

    tracker = DroneTracker(YOLO_MODEL)

    extractor = FeatureExtractor()

    predictor = LSTMPredictor(
        LSTM_MODEL,
        INPUT_SCALER,
        TARGET_SCALER
    )

    assessor = ThreatAssessment()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise Exception("Unable to open video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_number = 0

    # ========================================================
    # REPORT VARIABLES
    # ========================================================

    video_name = os.path.splitext(
        os.path.basename(VIDEO_PATH)
    )[0]

    REPORT_PATH = f"outputs/{video_name}_report.txt"

    track_ids = set()

    # Per-drone statistics
    drone_stats = {}

    first_prediction = None

    middle_prediction = None

    last_prediction = None
    
    current_threat = "NO THREAT"
    current_score = 0
    max_threat = "NO THREAT"
    max_score = 0
    
    # ========================================================
    # VIDEO LOOP
    # ========================================================
    
    while True:
        current_threat = "NO THREAT"
        current_score = 0
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        results = tracker.track(frame)

        annotated = results[0].plot()

        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()

            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):

                features = extractor.extract(track_id, box)

                predictor.update(track_id, features)

                track_ids.add(track_id)

                # ---------------------------------------------------------
                # Per-drone statistics
                # ---------------------------------------------------------

                if track_id not in drone_stats:
                    drone_stats[track_id] = {
                        "speeds": [],
                        "accelerations": [],
                        "frames": 0,
                    }

                drone_stats[track_id]["speeds"].append(
                    features["speed"]
                )

                drone_stats[track_id]["accelerations"].append(
                    features["acceleration"]
                )

                drone_stats[track_id]["frames"] += 1

                if predictor.ready(track_id):

                    future = predictor.predict(track_id)

                    if first_prediction is None:
                        first_prediction = future.copy()

                    last_prediction = future.copy()

                    print("\n" + "=" * 60)
                    print(f"Frame : {frame_number}")
                    print(f"Track : {track_id}")
                    print("=" * 60)

                    for i, point in enumerate(future):

                        print(
                            f"Future Frame +{i+1}: "
                            f"({point[0]:.2f}, {point[1]:.2f})"
                        )

                        cv2.circle(
                            annotated,
                            (int(point[0]), int(point[1])),
                            4,
                            (0, 0, 255),
                            -1
                        )

            # ---------------------------------------------------------
            # Debug: show per-drone statistics
            # ---------------------------------------------------------

            for tid, stats in drone_stats.items():

                avg_speed = statistics.mean(stats["speeds"])
                max_speed = max(stats["speeds"])

                avg_acceleration = statistics.mean(
                    stats["accelerations"]
                )

                max_acceleration = max(
                    stats["accelerations"],
                    key=abs
                )

                result = assessor.assess(
                    avg_speed=avg_speed,
                    max_speed=max_speed,
                    avg_acceleration=avg_acceleration,
                    max_acceleration=max_acceleration,
                    track_duration=stats["frames"],
                )

                print(
                    f"Track {tid} | "
                    f"Threat={result['level']} | "
                    f"Score={result['score']}"
                )

                # Temporary display: show the highest threat
                # among currently tracked drones
                if result["score"] > current_score:
                    current_score = result["score"]
                    current_threat = result["level"]

                # Save the highest threat seen throughout the video
                if result["score"] > max_score:
                    max_score = result["score"]
                    max_threat = result["level"]
    
        writer.write(annotated)

        # ---------------------------------------------------------
        # Draw threat overlay AFTER threat calculation
        # ---------------------------------------------------------

        color = (0, 255, 0)

        if current_threat == "LOW":
            color = (0, 255, 255)

        elif current_threat == "MEDIUM":
            color = (0, 165, 255)

        elif current_threat == "HIGH":
            color = (0, 0, 255)

        cv2.putText(
            annotated,
            f"Threat : {current_threat}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            annotated,
            f"Score : {current_score}/100",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        display_frame = cv2.resize(
            annotated,
            (1280, 720)
        )

        cv2.imshow(
            "Drone Threat Detection",
            display_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    writer.release()

    cv2.destroyAllWindows()

    # ========================================================
    # REPORT
    # ========================================================

    with open(REPORT_PATH, "w") as f:

        f.write("=" * 70 + "\n")
        f.write("DRONE THREAT REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Video Name           : {video_name}\n")
        f.write(f"Total Frames         : {frame_number}\n")
        f.write(f"Track IDs            : {sorted(track_ids)}\n")

        f.write("MOTION STATISTICS\n")
        f.write("-" * 40 + "\n")

        for tid, stats in drone_stats.items():

            f.write(f"Track {tid}\n")
            f.write(
                f"Average Speed        : "
                f"{statistics.mean(stats['speeds']):.2f}\n"
            )
            f.write(
                f"Maximum Speed        : "
                f"{max(stats['speeds']):.2f}\n"
            )
            f.write(
                f"Average Acceleration : "
                f"{statistics.mean(stats['accelerations']):.2f}\n"
            )
            f.write(
                f"Maximum Acceleration : "
                f"{max(stats['accelerations'], key=abs):.2f}\n"
            )
            f.write(
                f"Track Duration       : "
                f"{stats['frames']} frames\n\n"
            )
        def write_prediction(title, prediction):

            if prediction is None:
                return

            f.write(title + "\n")

            for i, p in enumerate(prediction, start=1):

                f.write(
                    f"Frame +{i}: "
                    f"({p[0]:.2f}, {p[1]:.2f})\n"
                )

            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("LSTM PREDICTION SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        write_prediction(
            "FIRST PREDICTION",
            first_prediction
        )

        write_prediction(
            "MIDDLE PREDICTION",
            middle_prediction
        )

        write_prediction(
            "LAST PREDICTION",
            last_prediction
        )

        f.write("="*70 + "\n")
        f.write("THREAT ASSESSMENT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Threat Level : {max_threat}\n")
        f.write(f"Threat Score : {max_score}/100\n\n")

    print("\n")
    print("=" * 60)
    print("PROCESSING COMPLETED")
    print("=" * 60)
    print(f"Output Video : {OUTPUT_PATH}")
    print(f"Report Saved : {REPORT_PATH}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()