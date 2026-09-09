# # # # # from utils.detector import DroneDetector

# # # # # MODEL_PATH = "models/best.pt"

# # # # # VIDEO_PATH = "videos/video01.mp4"

# # # # # OUTPUT_PATH = "outputs/video01_detected.mp4"


# # # # # def main():

# # # # #     detector = DroneDetector(MODEL_PATH)

# # # # #     detector.process_video(
# # # # #         VIDEO_PATH,
# # # # #         OUTPUT_PATH
# # # # #     )


# # # # # if __name__ == "__main__":
# # # # #     main()

# # # # from utils.tracker import DroneTracker
# # # # import cv2

# # # # MODEL_PATH = "models/best.pt"

# # # # VIDEO_PATH = "videos/video10.mp4"

# # # # OUTPUT_PATH = "outputs/video01_tracked.mp4"


# # # # def main():

# # # #     tracker = DroneTracker(MODEL_PATH)

# # # #     cap = cv2.VideoCapture(VIDEO_PATH)

# # # #     if not cap.isOpened():
# # # #         raise Exception("Unable to open video.")

# # # #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# # # #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # # #     fps = cap.get(cv2.CAP_PROP_FPS)

# # # #     writer = cv2.VideoWriter(
# # # #         OUTPUT_PATH,
# # # #         cv2.VideoWriter_fourcc(*"mp4v"),
# # # #         fps,
# # # #         (width, height)
# # # #     )

# # # #     while True:

# # # #         ret, frame = cap.read()

# # # #         if not ret:
# # # #             break

# # # #         results = tracker.track(frame)

# # # #         annotated = results[0].plot()

# # # #         writer.write(annotated)

# # # #         cv2.imshow("ByteTrack", annotated)

# # # #         if cv2.waitKey(1) & 0xFF == ord("q"):
# # # #             break

# # # #     cap.release()
# # # #     writer.release()
# # # #     cv2.destroyAllWindows()

# # # #     print("\n✅ Tracking Completed")
# # # #     print(f"Saved : {OUTPUT_PATH}")


# # # # if __name__ == "__main__":
# # # #     main()

# # # from utils.tracker import DroneTracker
# # # from utils.feature_extractor import FeatureExtractor

# # # import cv2

# # # MODEL_PATH = "models/best.pt"

# # # VIDEO_PATH = "videos/video01.mp4"

# # # OUTPUT_PATH = "outputs/video01_tracked.mp4"


# # # def main():

# # #     tracker = DroneTracker(MODEL_PATH)

# # #     extractor = FeatureExtractor()

# # #     cap = cv2.VideoCapture(VIDEO_PATH)

# # #     if not cap.isOpened():
# # #         raise Exception("Unable to open video.")

# # #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# # #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # #     fps = cap.get(cv2.CAP_PROP_FPS)

# # #     writer = cv2.VideoWriter(
# # #         OUTPUT_PATH,
# # #         cv2.VideoWriter_fourcc(*"mp4v"),
# # #         fps,
# # #         (width, height)
# # #     )

# # #     while True:

# # #         ret, frame = cap.read()

# # #         if not ret:
# # #             break

# # #         results = tracker.track(frame)

# # #         annotated = results[0].plot()

# # #         writer.write(annotated)

# # #         # -----------------------------
# # #         # Extract Features
# # #         # -----------------------------

# # #         if results[0].boxes.id is not None:

# # #             boxes = results[0].boxes.xyxy.cpu().numpy()
# # #             ids = results[0].boxes.id.cpu().numpy().astype(int)

# # #             for box, track_id in zip(boxes, ids):

# # #                 features = extractor.extract(track_id, box)

# # #                 print("=" * 60)
# # #                 print(f"Track ID : {track_id}")

# # #                 for key, value in features.items():
# # #                     print(f"{key:15s}: {value:.2f}")

# # #         cv2.imshow("ByteTrack", annotated)

# # #         if cv2.waitKey(1) & 0xFF == ord("q"):
# # #             break

# # #     cap.release()
# # #     writer.release()
# # #     cv2.destroyAllWindows()

# # #     print("\nTracking Completed")


# # # if __name__ == "__main__":
# # #     main()


# # #  now app.py with lstm prediction above was for tracker above that bytetrack above that only object detector 


# # # ============================================================
# # # DRONE THREAT DETECTION SYSTEM
# # # ============================================================

# # import cv2

# # from utils.tracker import DroneTracker
# # from utils.feature_extractor import FeatureExtractor
# # from utils.lstm_predictor import LSTMPredictor


# # # ============================================================
# # # PATHS
# # # ============================================================

# # YOLO_MODEL = "models/best.pt"

# # LSTM_MODEL = "models/best_lstm_retrained.pth"

# # INPUT_SCALER = "models/input_scaler.pkl"

# # TARGET_SCALER = "models/target_scaler.pkl"

# # VIDEO_PATH = "videos/video01.mp4"

# # OUTPUT_PATH = "outputs/video01_final.mp4"


# # # ============================================================
# # # MAIN
# # # ============================================================

# # def main():

# #     print("=" * 60)
# #     print("DRONE THREAT DETECTION SYSTEM")
# #     print("=" * 60)

# #     # --------------------------------------------------------
# #     # Load Modules
# #     # --------------------------------------------------------

# #     tracker = DroneTracker(YOLO_MODEL)

# #     extractor = FeatureExtractor()

# #     predictor = LSTMPredictor(
# #         LSTM_MODEL,
# #         INPUT_SCALER,
# #         TARGET_SCALER
# #     )

# #     # --------------------------------------------------------
# #     # Open Video
# #     # --------------------------------------------------------

# #     cap = cv2.VideoCapture(VIDEO_PATH)

# #     if not cap.isOpened():
# #         raise Exception("Unable to open video.")

# #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# #     fps = cap.get(cv2.CAP_PROP_FPS)

# #     writer = cv2.VideoWriter(
# #         OUTPUT_PATH,
# #         cv2.VideoWriter_fourcc(*"mp4v"),
# #         fps,
# #         (width, height)
# #     )

# #     frame_number = 0

# #     # ========================================================
# #     # VIDEO LOOP
# #     # ========================================================

# #     while True:

# #         ret, frame = cap.read()

# #         if not ret:
# #             break

# #         frame_number += 1

# #         # ----------------------------------------------------
# #         # Tracking
# #         # ----------------------------------------------------

# #         results = tracker.track(frame)

# #         annotated = results[0].plot()

# #         # ----------------------------------------------------
# #         # Feature Extraction + Prediction
# #         # ----------------------------------------------------

# #         if results[0].boxes.id is not None:

# #             boxes = results[0].boxes.xyxy.cpu().numpy()

# #             ids = results[0].boxes.id.cpu().numpy().astype(int)

# #             for box, track_id in zip(boxes, ids):

# #                 # ---------------------------------------------
# #                 # Feature Extraction
# #                 # ---------------------------------------------

# #                 features = extractor.extract(track_id, box)

# #                 predictor.update(track_id, features)

# #                 # ---------------------------------------------
# #                 # LSTM Prediction
# #                 # ---------------------------------------------

# #                 if predictor.ready(track_id):

# #                     future = predictor.predict(track_id)

# #                     print("\n" + "=" * 60)
# #                     print(f"Frame : {frame_number}")
# #                     print(f"Track : {track_id}")
# #                     print("=" * 60)

# #                     for i, point in enumerate(future):

# #                         print(
# #                             f"Future Frame +{i+1}: "
# #                             f"({point[0]:.2f}, {point[1]:.2f})"
# #                         )

# #                         # Draw prediction point

# #                         cv2.circle(
# #                             annotated,
# #                             (int(point[0]), int(point[1])),
# #                             4,
# #                             (0, 0, 255),
# #                             -1
# #                         )

# #         # ----------------------------------------------------
# #         # Show Video
# #         # ----------------------------------------------------

# #         writer.write(annotated)

# #         cv2.imshow("Drone Threat Detection", annotated)

# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break

# #     # ========================================================
# #     # CLEANUP
# #     # ========================================================

# #     cap.release()

# #     writer.release()

# #     cv2.destroyAllWindows()

# #     print("\n")
# #     print("=" * 60)
# #     print("Processing Completed")
# #     print("=" * 60)
# #     print(f"Saved Output : {OUTPUT_PATH}")


# # # ============================================================
# # # ENTRY POINT
# # # ============================================================

# # if __name__ == "__main__":
# #     main()









# # ============================================================
# # DRONE THREAT DETECTION SYSTEM
# # ============================================================

# import cv2
# import os
# import statistics

# from utils.tracker import DroneTracker
# from utils.feature_extractor import FeatureExtractor
# from utils.lstm_predictor import LSTMPredictor


# # ============================================================
# # PATHS
# # ============================================================

# YOLO_MODEL = "models/best.pt"

# LSTM_MODEL = "models/best_lstm_retrained.pth"

# INPUT_SCALER = "models/input_scaler.pkl"

# TARGET_SCALER = "models/target_scaler.pkl"

# VIDEO_PATH = "videos/video20.mp4"

# OUTPUT_PATH = "outputs/video20_final.mp4"


# # ============================================================
# # MAIN
# # ============================================================

# def main():

#     print("=" * 60)
#     print("DRONE THREAT DETECTION SYSTEM")
#     print("=" * 60)

#     tracker = DroneTracker(YOLO_MODEL)

#     extractor = FeatureExtractor()

#     predictor = LSTMPredictor(
#         LSTM_MODEL,
#         INPUT_SCALER,
#         TARGET_SCALER
#     )

#     cap = cv2.VideoCapture(VIDEO_PATH)

#     if not cap.isOpened():
#         raise Exception("Unable to open video.")

#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     writer = cv2.VideoWriter(
#         OUTPUT_PATH,
#         cv2.VideoWriter_fourcc(*"mp4v"),
#         fps,
#         (width, height)
#     )

#     frame_number = 0

#     # ========================================================
#     # REPORT VARIABLES
#     # ========================================================

#     video_name = os.path.splitext(
#         os.path.basename(VIDEO_PATH)
#     )[0]

#     REPORT_PATH = f"outputs/{video_name}_report.txt"

#     track_ids = set()

#     speed_list = []

#     acceleration_list = []

#     prediction_count = 0

#     first_prediction = None

#     middle_prediction = None

#     last_prediction = None

#     # ========================================================
#     # VIDEO LOOP
#     # ========================================================

#     while True:

#         ret, frame = cap.read()

#         if not ret:
#             break

#         frame_number += 1

#         results = tracker.track(frame)

#         annotated = results[0].plot()

#         if results[0].boxes.id is not None:

#             boxes = results[0].boxes.xyxy.cpu().numpy()

#             ids = results[0].boxes.id.cpu().numpy().astype(int)

#             for box, track_id in zip(boxes, ids):

#                 features = extractor.extract(track_id, box)


#                 predictor.update(track_id, features)

#                 track_ids.add(track_id)

#                 speed_list.append(features["speed"])

#                 acceleration_list.append(
#                     features["acceleration"]
#                 )

#                 if predictor.ready(track_id):

#                     future = predictor.predict(track_id)

#                     prediction_count += 1

#                     if first_prediction is None:
#                         first_prediction = future.copy()

#                     if prediction_count == 500:
#                         middle_prediction = future.copy()

#                     last_prediction = future.copy()

#                     print("\n" + "=" * 60)
#                     print(f"Frame : {frame_number}")
#                     print(f"Track : {track_id}")
#                     print("=" * 60)

#                     for i, point in enumerate(future):

#                         print(
#                             f"Future Frame +{i+1}: "
#                             f"({point[0]:.2f}, {point[1]:.2f})"
#                         )

#                         cv2.circle(
#                             annotated,
#                             (int(point[0]), int(point[1])),
#                             4,
#                             (0, 0, 255),
#                             -1
#                         )

#         writer.write(annotated)

#         cv2.imshow(
#             "Drone Threat Detection",
#             annotated
#         )

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()

#     writer.release()

#     cv2.destroyAllWindows()

#     # ========================================================
#     # REPORT
#     # ========================================================

#     with open(REPORT_PATH, "w") as f:

#         f.write("=" * 70 + "\n")
#         f.write("DRONE THREAT REPORT\n")
#         f.write("=" * 70 + "\n\n")

#         f.write(f"Video Name           : {video_name}\n")
#         f.write(f"Total Frames         : {frame_number}\n")
#         f.write(f"Track IDs            : {sorted(track_ids)}\n")
#         f.write(f"Total Predictions    : {prediction_count}\n\n")

#         if speed_list:

#             f.write("MOTION STATISTICS\n")
#             f.write("-" * 40 + "\n")

#             f.write(
#                 f"Average Speed        : {statistics.mean(speed_list):.2f}\n"
#             )

#             f.write(
#                 f"Maximum Speed        : {max(speed_list):.2f}\n"
#             )

#             f.write(
#                 f"Minimum Speed        : {min(speed_list):.2f}\n\n"
#             )

#         if acceleration_list:

#             f.write(
#                 f"Average Acceleration : {statistics.mean(acceleration_list):.2f}\n"
#             )

#             f.write(
#                 f"Maximum Acceleration : {max(acceleration_list):.2f}\n"
#             )

#             f.write(
#                 f"Minimum Acceleration : {min(acceleration_list):.2f}\n\n"
#             )

#         def write_prediction(title, prediction):

#             if prediction is None:
#                 return

#             f.write(title + "\n")

#             for i, p in enumerate(prediction, start=1):

#                 f.write(
#                     f"Frame +{i}: "
#                     f"({p[0]:.2f}, {p[1]:.2f})\n"
#                 )

#             f.write("\n")

#         f.write("=" * 70 + "\n")
#         f.write("LSTM PREDICTION SUMMARY\n")
#         f.write("=" * 70 + "\n\n")

#         write_prediction(
#             "FIRST PREDICTION",
#             first_prediction
#         )

#         write_prediction(
#             "MIDDLE PREDICTION",
#             middle_prediction
#         )

#         write_prediction(
#             "LAST PREDICTION",
#             last_prediction
#         )

#     print("\n")
#     print("=" * 60)
#     print("PROCESSING COMPLETED")
#     print("=" * 60)
#     print(f"Output Video : {OUTPUT_PATH}")
#     print(f"Report Saved : {REPORT_PATH}")


# # ============================================================
# # ENTRY POINT
# # ============================================================

# if __name__ == "__main__":
#     main()



# app_final.py

# # # # from utils.detector import DroneDetector

# # # # MODEL_PATH = "models/best.pt"

# # # # VIDEO_PATH = "videos/video01.mp4"

# # # # OUTPUT_PATH = "outputs/video01_detected.mp4"


# # # # def main():

# # # #     detector = DroneDetector(MODEL_PATH)

# # # #     detector.process_video(
# # # #         VIDEO_PATH,
# # # #         OUTPUT_PATH
# # # #     )


# # # # if __name__ == "__main__":
# # # #     main()

# # # from utils.tracker import DroneTracker
# # # import cv2

# # # MODEL_PATH = "models/best.pt"

# # # VIDEO_PATH = "videos/video10.mp4"

# # # OUTPUT_PATH = "outputs/video01_tracked.mp4"


# # # def main():

# # #     tracker = DroneTracker(MODEL_PATH)

# # #     cap = cv2.VideoCapture(VIDEO_PATH)

# # #     if not cap.isOpened():
# # #         raise Exception("Unable to open video.")

# # #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# # #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # #     fps = cap.get(cv2.CAP_PROP_FPS)

# # #     writer = cv2.VideoWriter(
# # #         OUTPUT_PATH,
# # #         cv2.VideoWriter_fourcc(*"mp4v"),
# # #         fps,
# # #         (width, height)
# # #     )

# # #     while True:

# # #         ret, frame = cap.read()

# # #         if not ret:
# # #             break

# # #         results = tracker.track(frame)

# # #         annotated = results[0].plot()

# # # 
        # ----------------------------------------------------
        # Threat Assessment
        # ----------------------------------------------------
        if speed_list:

            result = assessor.assess(
                avg_speed=statistics.mean(speed_list),
                max_speed=max(speed_list),
                avg_acceleration=statistics.mean(acceleration_list),
                max_acceleration=max(acceleration_list),
                prediction_count=prediction_count,
                total_frames=frame_number
            )

            current_threat = result["level"]
            current_score = result["score"]
            current_reasons = result["reasons"]

            color=(0,255,0)
            if current_threat=="LOW":
                color=(0,255,255)
            elif current_threat=="MEDIUM":
                color=(0,165,255)
            elif current_threat=="HIGH":
                color=(0,0,255)

            cv2.putText(
                annotated,
                f"Threat: {current_threat}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.putText(
                annotated,
                f"Score: {current_score}/100",
                (20,75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        writer.write(annotated)

# # #         cv2.imshow("ByteTrack", annotated)

# # #         if cv2.waitKey(1) & 0xFF == ord("q"):
# # #             break

# # #     cap.release()
# # #     writer.release()
# # #     cv2.destroyAllWindows()

# # #     print("\n✅ Tracking Completed")
# # #     print(f"Saved : {OUTPUT_PATH}")


# # # if __name__ == "__main__":
# # #     main()

# # from utils.tracker import DroneTracker
# # from utils.feature_extractor import FeatureExtractor

# # import cv2

# # MODEL_PATH = "models/best.pt"

# # VIDEO_PATH = "videos/video01.mp4"

# # OUTPUT_PATH = "outputs/video01_tracked.mp4"


# # def main():

# #     tracker = DroneTracker(MODEL_PATH)

# #     extractor = FeatureExtractor()

# #     cap = cv2.VideoCapture(VIDEO_PATH)

# #     if not cap.isOpened():
# #         raise Exception("Unable to open video.")

# #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# #     fps = cap.get(cv2.CAP_PROP_FPS)

# #     writer = cv2.VideoWriter(
# #         OUTPUT_PATH,
# #         cv2.VideoWriter_fourcc(*"mp4v"),
# #         fps,
# #         (width, height)
# #     )

# #     while True:

# #         ret, frame = cap.read()

# #         if not ret:
# #             break

# #         results = tracker.track(frame)

# #         annotated = results[0].plot()

# #         writer.write(annotated)

# #         # -----------------------------
# #         # Extract Features
# #         # -----------------------------

# #         if results[0].boxes.id is not None:

# #             boxes = results[0].boxes.xyxy.cpu().numpy()
# #             ids = results[0].boxes.id.cpu().numpy().astype(int)

# #             for box, track_id in zip(boxes, ids):

# #                 features = extractor.extract(track_id, box)

# #                 print("=" * 60)
# #                 print(f"Track ID : {track_id}")

# #                 for key, value in features.items():
# #                     print(f"{key:15s}: {value:.2f}")

# #         cv2.imshow("ByteTrack", annotated)

# #         if cv2.waitKey(1) & 0xFF == ord("q"):
# #             break

# #     cap.release()
# #     writer.release()
# #     cv2.destroyAllWindows()

# #     print("\nTracking Completed")


# # if __name__ == "__main__":
# #     main()


# #  now app.py with lstm prediction above was for tracker above that bytetrack above that only object detector 


# # ============================================================
# # DRONE THREAT DETECTION SYSTEM
# # ============================================================

# import cv2

# from utils.tracker import DroneTracker
# from utils.feature_extractor import FeatureExtractor
# from utils.lstm_predictor import LSTMPredictor
from utils.threat import ThreatAssessment


# # ============================================================
# # PATHS
# # ============================================================

# YOLO_MODEL = "models/best.pt"

# LSTM_MODEL = "models/best_lstm_retrained.pth"

# INPUT_SCALER = "models/input_scaler.pkl"

# TARGET_SCALER = "models/target_scaler.pkl"

# VIDEO_PATH = "videos/video01.mp4"

# OUTPUT_PATH = "outputs/video01_final.mp4"


# # ============================================================
# # MAIN
# # ============================================================

# def main():

#     print("=" * 60)
#     print("DRONE THREAT DETECTION SYSTEM")
#     print("=" * 60)

#     # --------------------------------------------------------
#     # Load Modules
#     # --------------------------------------------------------

#     tracker = DroneTracker(YOLO_MODEL)

#     extractor = FeatureExtractor()

#     predictor = LSTMPredictor(
#         LSTM_MODEL,
#         INPUT_SCALER,
#         TARGET_SCALER
#     )

#     # --------------------------------------------------------
#     # Open Video
#     # --------------------------------------------------------

#     cap = cv2.VideoCapture(VIDEO_PATH)

#     if not cap.isOpened():
#         raise Exception("Unable to open video.")

#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     writer = cv2.VideoWriter(
#         OUTPUT_PATH,
#         cv2.VideoWriter_fourcc(*"mp4v"),
#         fps,
#         (width, height)
#     )

#     frame_number = 0

#     # ========================================================
#     # VIDEO LOOP
#     # ========================================================

#     while True:

#         ret, frame = cap.read()

#         if not ret:
#             break

#         frame_number += 1

#         # ----------------------------------------------------
#         # Tracking
#         # ----------------------------------------------------

#         results = tracker.track(frame)

#         annotated = results[0].plot()

#         # ----------------------------------------------------
#         # Feature Extraction + Prediction
#         # ----------------------------------------------------

#         if results[0].boxes.id is not None:

#             boxes = results[0].boxes.xyxy.cpu().numpy()

#             ids = results[0].boxes.id.cpu().numpy().astype(int)

#             for box, track_id in zip(boxes, ids):

#                 # ---------------------------------------------
#                 # Feature Extraction
#                 # ---------------------------------------------

#                 features = extractor.extract(track_id, box)

#                 predictor.update(track_id, features)

#                 # ---------------------------------------------
#                 # LSTM Prediction
#                 # ---------------------------------------------

#                 if predictor.ready(track_id):

#                     future = predictor.predict(track_id)

#                     print("\n" + "=" * 60)
#                     print(f"Frame : {frame_number}")
#                     print(f"Track : {track_id}")
#                     print("=" * 60)

#                     for i, point in enumerate(future):

#                         print(
#                             f"Future Frame +{i+1}: "
#                             f"({point[0]:.2f}, {point[1]:.2f})"
#                         )

#                         # Draw prediction point

#                         cv2.circle(
#                             annotated,
#                             (int(point[0]), int(point[1])),
#                             4,
#                             (0, 0, 255),
#                             -1
#                         )

#         # ----------------------------------------------------
#         # Show Video
#         # ----------------------------------------------------

#         writer.write(annotated)

#         cv2.imshow("Drone Threat Detection", annotated)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # ========================================================
#     # CLEANUP
#     # ========================================================

#     cap.release()

#     writer.release()

#     cv2.destroyAllWindows()

#     print("\n")
#     print("=" * 60)
#     print("Processing Completed")
#     print("=" * 60)
#     print(f"Saved Output : {OUTPUT_PATH}")


# # ============================================================
# # ENTRY POINT
# # ============================================================

# if __name__ == "__main__":
#     main()









# ============================================================
# DRONE THREAT DETECTION SYSTEM
# ============================================================

import cv2
import os
import statistics

from utils.tracker import DroneTracker
from utils.feature_extractor import FeatureExtractor
from utils.lstm_predictor import LSTMPredictor
from utils.threat import ThreatAssessment


# ============================================================
# PATHS
# ============================================================

YOLO_MODEL = "models/best.pt"

LSTM_MODEL = "models/best_lstm_retrained.pth"

INPUT_SCALER = "models/input_scaler.pkl"

TARGET_SCALER = "models/target_scaler.pkl"

VIDEO_PATH = "videos/video20.mp4"

OUTPUT_PATH = "outputs/video20_final.mp4"


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

    speed_list = []

    acceleration_list = []

    prediction_count = 0

    first_prediction = None

    middle_prediction = None

    last_prediction = None

    current_threat = "NO THREAT"
    current_score = 0
    current_reasons = []

    # ========================================================
    # VIDEO LOOP
    # ========================================================

    while True:

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

                speed_list.append(features["speed"])

                acceleration_list.append(
                    features["acceleration"]
                )

                if predictor.ready(track_id):

                    future = predictor.predict(track_id)

                    prediction_count += 1

                    if first_prediction is None:
                        first_prediction = future.copy()

                    if prediction_count == 500:
                        middle_prediction = future.copy()

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

        writer.write(annotated)

        cv2.imshow(
            "Drone Threat Detection",
            annotated
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
        f.write(f"Total Predictions    : {prediction_count}\n\n")

        if speed_list:

            f.write("MOTION STATISTICS\n")
            f.write("-" * 40 + "\n")

            f.write(
                f"Average Speed        : {statistics.mean(speed_list):.2f}\n"
            )

            f.write(
                f"Maximum Speed        : {max(speed_list):.2f}\n"
            )

            f.write(
                f"Minimum Speed        : {min(speed_list):.2f}\n\n"
            )

        if acceleration_list:

            f.write(
                f"Average Acceleration : {statistics.mean(acceleration_list):.2f}\n"
            )

            f.write(
                f"Maximum Acceleration : {max(acceleration_list):.2f}\n"
            )

            f.write(
                f"Minimum Acceleration : {min(acceleration_list):.2f}\n\n"
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
        f.write(f"Threat Level : {current_threat}\n")
        f.write(f"Threat Score : {current_score}/100\n\n")
        f.write("Reasons\n")
        for r in current_reasons:
            f.write(f"- {r}\n")


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
