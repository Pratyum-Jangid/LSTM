# ============================================================
# DRONE THREAT DETECTION SYSTEM
# ============================================================

import cv2
import os
import statistics
import time
import math

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

# ------------------------------------------------------------
# ID-stitching parameters
# ------------------------------------------------------------
# If a new track ID appears within MAX_GAP frames of an old
# track disappearing, and its first position is within a
# velocity-projected radius of where the old track last was,
# treat it as the same physical drone (bridges ByteTrack ID
# switches caused by brief detection dropout / motion blur /
# occlusion / failed IoU re-association).
MAX_GAP = 40

# Extra slack (px) added on top of the velocity-projected position,
# to absorb noise in the speed estimate itself.
DIST_SLACK = 40

# Floor/ceiling on the allowed match radius so a near-zero speed
# doesn't make matching impossibly strict, and a huge speed spike
# doesn't make it match anything on screen.
# NOTE: 900px is ~half a 1920px-wide frame — generous enough for a fast
# drone over a 33-frame gap but still prevents cross-frame false matches.
MIN_DIST = 60
MAX_DIST_CAP = 900


# ============================================================
# ID STITCHING HELPER
# ============================================================

def try_stitch(track_frame_range, extractor, new_id, box, frame_number):
    """
    Returns the ID this detection should be treated as.
    If it looks like a continuation of a recently-lost track,
    returns that old track's ID instead of new_id.

    Strategy:
      1. Collect all tracks whose last-seen frame is within MAX_GAP.
      2. If exactly ONE such candidate exists, stitch unconditionally --
         there is no ambiguity so distance is irrelevant (the drone may
         have moved arbitrarily far during the gap).
      3. If MULTIPLE candidates exist, use velocity-projected distance to
         pick the most plausible one and reject any outside allowed_radius
         (to avoid cross-drone false matches).
    """

    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2

    # -----------------------------------------------------------------
    # Pass 1: collect every track within the time window
    # -----------------------------------------------------------------
    temporal_candidates = []   # (old_id, gap, old_pos)

    for old_id, (first_seen, last_seen) in track_frame_range.items():

        if old_id == new_id:
            continue

        gap = frame_number - last_seen

        if gap <= 0 or gap > MAX_GAP:
            continue

        old_pos = extractor.previous.get(old_id)

        if old_pos is None:
            continue

        temporal_candidates.append((old_id, gap, old_pos))

    # -----------------------------------------------------------------
    # No temporal match at all → new track
    # -----------------------------------------------------------------
    if not temporal_candidates:
        return new_id

    # -----------------------------------------------------------------
    # Exactly one candidate → stitch unconditionally (no ambiguity)
    # A fast drone can move > 900 px in 33 frames; distance would only
    # matter if we needed to resolve which of several old tracks to pick.
    # -----------------------------------------------------------------
    if len(temporal_candidates) == 1:
        matched_id = temporal_candidates[0][0]
        print(
            f"[STITCH-UNAMBIGUOUS] raw id {new_id} -> "
            f"existing track {matched_id} "
            f"(frame {frame_number}, gap {temporal_candidates[0][1]})"
        )
        return matched_id

    # -----------------------------------------------------------------
    # Multiple candidates → use distance to pick the best one
    # -----------------------------------------------------------------
    best_match = None
    best_score = None

    for old_id, gap, old_pos in temporal_candidates:

        old_speed = old_pos.get("speed", 0.0)

        allowed_radius = min(
            max(old_speed * gap + DIST_SLACK, MIN_DIST),
            MAX_DIST_CAP,
        )

        dist = ((cx - old_pos["cx"]) ** 2 + (cy - old_pos["cy"]) ** 2) ** 0.5

        if dist > allowed_radius:
            continue

        score = dist / allowed_radius

        if best_score is None or score < best_score:
            best_score = score
            best_match = old_id

    return best_match if best_match is not None else new_id


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
    # REPORT / STATE VARIABLES
    # ========================================================

    video_name = os.path.splitext(
        os.path.basename(VIDEO_PATH)
    )[0]

    REPORT_PATH = f"outputs/{video_name}_report.txt"

    track_ids = set()

    # Per-drone statistics (keyed by STITCHED id, not raw ByteTrack id)
    drone_stats = {}

    # raw_id -> stitched_id, so every later reference to a raw id
    # resolves to the same logical drone
    id_map = {}

    # stitched_id -> [first_frame_seen, last_frame_seen]
    track_frame_range = {}

    first_prediction = None
    middle_prediction = None
    last_prediction = None

    current_threat = "NO THREAT"
    current_score = 0
    max_threat = "NO THREAT"
    max_score = 0
    detected_frames = 0
    tracked_frames = 0

    # FPS tracking
    fps_start_time = time.time()
    fps_display    = 0.0
    fps_frame_count = 0

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

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            writer.write(frame)
            continue

        detected_frames += 1

        raw_ids = results[0].boxes.id

        annotated = results[0].plot()

        if raw_ids is None and results[0].boxes.conf is not None:
            confs = results[0].boxes.conf.cpu().numpy()
            print(f"[NO-ID] frame {frame_number}: confidences {confs}")

        if raw_ids is not None:

            tracked_frames += 1

            boxes = results[0].boxes.xyxy.cpu().numpy()
            raw_ids = raw_ids.cpu().numpy().astype(int)

            # Per-frame list of (stitched_id, box, features) for label drawing
            drone_draw_info = []

            for box, raw_id in zip(boxes, raw_ids):

                # ---------------------------------------------
                # Resolve raw ByteTrack id -> stitched drone id
                # ---------------------------------------------
                if raw_id in id_map:
                    stitched_id = id_map[raw_id]
                else:
                    stitched_id = try_stitch(
                        track_frame_range, extractor, raw_id, box, frame_number
                    )
                    id_map[raw_id] = stitched_id

                    if stitched_id != raw_id:
                        print(
                            f"[STITCH] raw id {raw_id} -> "
                            f"existing track {stitched_id} "
                            f"(frame {frame_number})"
                        )

                # ---------------------------------------------
                # Feature extraction keyed on stitched id, so
                # velocity/heading continue smoothly across the
                # ID switch instead of resetting to zero
                # ---------------------------------------------
                features = extractor.extract(stitched_id, box)

                predictor.update(stitched_id, features)

                track_ids.add(stitched_id)

                if stitched_id not in track_frame_range:
                    track_frame_range[stitched_id] = [frame_number, frame_number]
                else:
                    track_frame_range[stitched_id][1] = frame_number

                # ---------------------------------------------
                # Per-drone statistics
                # ---------------------------------------------
                if stitched_id not in drone_stats:
                    drone_stats[stitched_id] = {
                        "speeds": [],
                        "accelerations": [],
                        "frames": 0,
                    }

                drone_stats[stitched_id]["speeds"].append(features["speed"])
                drone_stats[stitched_id]["accelerations"].append(features["acceleration"])
                drone_stats[stitched_id]["frames"] += 1

                if predictor.ready(stitched_id):

                    future = predictor.predict(stitched_id)

                    if first_prediction is None:
                        first_prediction = future.copy()

                    if middle_prediction is None and frame_number >= 200:
                        middle_prediction = future.copy()

                    last_prediction = future.copy()

                    print("\n" + "=" * 60)
                    print(f"Frame : {frame_number}")
                    print(f"Track : {stitched_id}")
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

                    # Connect prediction dots with a red polyline
                    if len(future) >= 2:
                        pts = [(int(p[0]), int(p[1])) for p in future]
                        for k in range(len(pts) - 1):
                            cv2.line(
                                annotated,
                                pts[k], pts[k + 1],
                                (0, 0, 200), 1, cv2.LINE_AA
                            )

                drone_draw_info.append((stitched_id, box, features))

            # ---------------------------------------------------------
            # Threat assessment across current drone_stats
            # ---------------------------------------------------------
            for tid, stats in drone_stats.items():

                avg_speed = statistics.mean(stats["speeds"])
                max_speed = max(stats["speeds"])
                avg_acceleration = statistics.mean(stats["accelerations"])
                max_acceleration = max(stats["accelerations"], key=abs)

                result = assessor.assess(
                    avg_speed=avg_speed,
                    max_speed=max_speed,
                    avg_acceleration=avg_acceleration,
                    max_acceleration=max_acceleration,
                    track_duration=stats["frames"],
                )

                if result["score"] > current_score:
                    current_score = result["score"]
                    current_threat = result["level"]

                if result["score"] > max_score:
                    max_score = result["score"]
                    max_threat = result["level"]

        else:
            annotated = frame
            drone_draw_info = []

        # ---------------------------------------------------------
        # Draw per-drone speed + heading labels
        # ---------------------------------------------------------
        for sid, box, feat in drone_draw_info:
            x1, y1, x2, y2 = box
            spd  = feat.get("speed", 0.0)
            hdg  = math.degrees(feat.get("heading", 0.0))
            conf_list = []
            if results[0].boxes.conf is not None:
                conf_list = results[0].boxes.conf.cpu().numpy()
            # label: ID  spd  hdg
            label = f"ID:{sid}  spd:{spd:.1f}  hdg:{hdg:.0f}deg"
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            lx = int(x1)
            ly = max(int(y1) - 6, lh + 4)
            cv2.rectangle(
                annotated,
                (lx - 2, ly - lh - 3),
                (lx + lw + 2, ly + 2),
                (30, 30, 30), -1
            )
            cv2.putText(
                annotated, label, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 230), 1, cv2.LINE_AA
            )

        # ---------------------------------------------------------
        # Draw per-frame HUD (threat status + score + frame counter)
        # MUST be done BEFORE writer.write() so it appears in video
        # ---------------------------------------------------------

        # Choose colour based on threat level
        if current_threat == "HIGH":
            hud_color = (0, 0, 255)      # red
            bg_color  = (0, 0, 180)
        elif current_threat == "MEDIUM":
            hud_color = (0, 165, 255)    # orange
            bg_color  = (0, 100, 180)
        elif current_threat == "LOW":
            hud_color = (0, 255, 255)    # yellow
            bg_color  = (0, 160, 160)
        else:
            hud_color = (0, 220, 0)      # green
            bg_color  = (0, 130, 0)

        # Semi-transparent banner at the top
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (width, 95), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

        # Threat level label
        cv2.putText(
            annotated,
            f"THREAT: {current_threat}",
            (18, 38),
            cv2.FONT_HERSHEY_DUPLEX, 1.0, hud_color, 2, cv2.LINE_AA
        )

        # Score  (with filled pill background for readability)
        score_label = f"SCORE: {current_score}/100"
        cv2.putText(
            annotated,
            score_label,
            (18, 75),
            cv2.FONT_HERSHEY_DUPLEX, 0.85, hud_color, 2, cv2.LINE_AA
        )

        # Frame + FPS counter (top-right corner)
        fps_frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps_display    = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_start_time  = time.time()

        frame_label = f"FRAME: {frame_number}  |  FPS: {fps_display:.1f}"
        (fw, fh), _ = cv2.getTextSize(
            frame_label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1
        )
        cv2.putText(
            annotated,
            frame_label,
            (width - fw - 18, 38),
            cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA
        )

        # Score bar (visual fill under the text)
        bar_x, bar_y, bar_w, bar_h = 18, 82, 260, 8
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        fill_w = int(bar_w * current_score / 100)
        if fill_w > 0:
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), hud_color, -1)

        writer.write(annotated)

    cap.release()
    writer.release()

    # ========================================================
    # DIAGNOSTIC: print track frame ranges and gaps
    # ========================================================

    print("\nTRACK FRAME RANGES (stitched IDs):")
    sorted_tracks = sorted(track_frame_range.items(), key=lambda x: x[1][0])
    for i, (tid, (first, last)) in enumerate(sorted_tracks):
        print(f"  Track {tid}: frames {first} - {last}  ({last - first + 1} frames)")
        if i > 0:
            prev_last = sorted_tracks[i - 1][1][1]
            gap = first - prev_last
            print(f"    ^ gap from previous track: {gap} frames")

    # ========================================================
    # REPORT
    # ========================================================

    # --------------------------------------------------------
    # Merge all track segments into one logical drone for the
    # report. This is correct for a single-drone test video
    # where all detections belong to the same physical object.
    # --------------------------------------------------------
    all_speeds        = []
    all_accelerations = []
    total_track_frames = 0

    for stats in drone_stats.values():
        all_speeds.extend(stats["speeds"])
        all_accelerations.extend(stats["accelerations"])
        total_track_frames += stats["frames"]

    merged_avg_speed   = statistics.mean(all_speeds)        if all_speeds else 0
    merged_max_speed   = max(all_speeds)                    if all_speeds else 0
    merged_avg_accel   = statistics.mean(all_accelerations) if all_accelerations else 0
    merged_max_accel   = max(all_accelerations, key=abs)    if all_accelerations else 0

    # Re-run threat assessment on merged stats for final score
    merged_result = assessor.assess(
        avg_speed       = merged_avg_speed,
        max_speed       = merged_max_speed,
        avg_acceleration= merged_avg_accel,
        max_acceleration= merged_max_accel,
        track_duration  = total_track_frames,
    )

    with open(REPORT_PATH, "w") as f:

        f.write("=" * 70 + "\n")
        f.write("DRONE THREAT REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Video Name           : {video_name}\n")
        f.write(f"Total Frames         : {frame_number}\n")
        f.write(f"Detected Frames      : {detected_frames} ({(detected_frames/frame_number)*100:.1f}%)\n")
        f.write(f"Tracked Frames       : {tracked_frames} ({(tracked_frames/frame_number)*100:.1f}%)\n")
        f.write(f"Track Segments       : {sorted(track_ids)} (same drone, ID resets due to dropout)\n")
        f.write(f"Raw id map           : {id_map}\n\n")

        f.write("MOTION STATISTICS (merged — single drone)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average Speed        : {merged_avg_speed:.2f} px/frame\n")
        # f.write(f"Maximum Speed        : {merged_max_speed:.2f} px/frame\n")
        f.write(f"Average Acceleration : {merged_avg_accel:.2f} px/frame²\n")
        # f.write(f"Maximum Acceleration : {merged_max_accel:.2f} px/frame²\n")
        f.write(f"Track Duration       : {total_track_frames} frames\n\n")

        f.write("SEGMENT BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        for tid, stats in drone_stats.items():
            fr = track_frame_range.get(tid, ["?", "?"])
            f.write(f"  Segment (Track {tid}): frames {fr[0]}–{fr[1]}, duration {stats['frames']} frames\n")  
        f.write("\n")

        def write_prediction(title, prediction):

            if prediction is None:
                return

            f.write(title + "\n")

            for i, p in enumerate(prediction, start=1):
                f.write(f"Frame +{i}: ({p[0]:.2f}, {p[1]:.2f})\n")

            f.write("\n")

        print(f"Detected frames: {detected_frames}/{frame_number}")
        print(f"Tracked frames: {tracked_frames}/{frame_number}")
        print(f"Detection coverage: {(detected_frames / frame_number) * 100:.2f}%")
        print(f"Tracking coverage: {(tracked_frames / frame_number) * 100:.2f}%")
        print(f"Merged threat: {merged_result['level']} ({merged_result['score']}/100)")

        f.write("=" * 70 + "\n")
        f.write("LSTM PREDICTION SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        write_prediction("FIRST PREDICTION", first_prediction)
        write_prediction("MIDDLE PREDICTION", middle_prediction)
        write_prediction("LAST PREDICTION", last_prediction)

        f.write("=" * 70 + "\n")
        f.write("THREAT ASSESSMENT (merged — single drone)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Threat Level : {merged_result['level']}\n")
        f.write(f"Threat Score : {merged_result['score']}/100\n")
        f.write(f"Reasons      :\n")
        for reason in merged_result["reasons"]:
            f.write(f"  - {reason}\n")
        f.write("\n")

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