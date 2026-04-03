import os
import cv2
from collections import deque
from ultralytics import YOLO


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
VIDEO_PATH       = "data/montreal.mp4"
IMAGE_DIR        = "andreasmoegelmose/multiview-traffic-intersection-dataset/versions/1/Drone/frames/"
MODEL_PATH       = "yolo11n.pt"

SPEED_LIMIT_KMH  = 50          # Fixed threshold (urban intersection)
FPS              = 30          # Frames per second

ROAD_WIDTH_PX    = 230         # Measured from image (one direction, 2 lanes)
ROAD_WIDTH_M     = 7.0         # Real-world equivalent (2 x 3.5m lanes)
road_width_pixels = 163.5      # Measured per video
road_width_meters = 28         # Determined through research/assumptions
METERS_PER_PIXEL = ROAD_WIDTH_M / ROAD_WIDTH_PX

WINDOW_SIZE      = 5           # Sliding window: number of frames to average over
MIN_PIXEL_DIST   = 2           # Ignore movements smaller than this (detection jitter)
CONF_THRESHOLD   = 0.2         # YOLO confidence threshold


# ─────────────────────────────────────────────
# Speed Estimation (Sliding Window)
# ─────────────────────────────────────────────
def estimate_speed(position_history: deque, fps: int, meters_per_pixel: float) -> float | None:
    """
    Estimate speed using total displacement over the full sliding window.

    Instead of frame[N-1] → frame[N] (noisy), this computes:
        displacement = distance from oldest to newest position in window
        time         = (number of intervals in window) / fps
        speed        = displacement / time

    This is equivalent to averaging per-frame speeds but is more
    numerically stable because single jitter frames don't dominate.

    Returns speed in km/h, or None if not enough history yet.
    """
    positions = list(position_history)
    if len(positions) < 2:
        return None

    x_start, y_start = positions[0]
    x_end,   y_end   = positions[-1]

    dx = x_end - x_start
    dy = y_end - y_start
    pixel_distance = (dx ** 2 + dy ** 2) ** 0.5

    # Ignore tiny movements — likely detection jitter, not real motion
    if pixel_distance < MIN_PIXEL_DIST:
        return 0.0

    distance_meters = pixel_distance * meters_per_pixel
    time_elapsed    = (len(positions) - 1) / fps   # seconds across window

    speed_mps = distance_meters / time_elapsed
    speed_kmh = speed_mps * 3.6
    return speed_kmh


# ─────────────────────────────────────────────
# Frame Processing
# ─────────────────────────────────────────────
def process_frame(frame, result, track_history: dict, fps: int, speed_limit: float):
    """
    Process a single YOLO result:
      - Update sliding window position history per track ID
      - Estimate speed using the window
      - Annotate the frame with bounding box + speed + violation colour
    """
    if result.boxes is None or result.boxes.id is None:
        return frame

    for box, track_id in zip(result.boxes, result.boxes.id):
        track_id = int(track_id)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Initialise deque for new vehicles — maxlen enforces the window size
        if track_id not in track_history:
            track_history[track_id] = deque(maxlen=WINDOW_SIZE)

        track_history[track_id].append((cx, cy))

        speed_kmh = estimate_speed(track_history[track_id], fps, METERS_PER_PIXEL)
        if speed_kmh is None:
            continue   # Not enough history yet, skip annotation

        # Colour: red if speeding, green if compliant
        color = (0, 0, 255) if speed_kmh > speed_limit else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track_id} | {speed_kmh:.1f} km/h"
        cv2.putText(
            frame, label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, color, 2
        )

    # Overlay speed limit in corner
    cv2.putText(
        frame,
        f"Speed limit: {speed_limit} km/h",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2
    )

    return frame


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print(f"Meters per pixel: {METERS_PER_PIXEL:.6f}")
    print(f"Speed limit:      {SPEED_LIMIT_KMH} km/h")
    print(f"Window size:      {WINDOW_SIZE} frames\n")

    # Ask user for data source
    print("Select the data source:")
    print("  1. Video file")
    print("  2. Image dataset")
    choice = input("Enter 1 or 2: ").strip()

    if choice not in ("1", "2"):
        raise ValueError("Invalid choice. Please enter 1 or 2.")

    use_video = (choice == "1")
    model     = YOLO(MODEL_PATH)
    track_history: dict[int, deque] = {}

    if use_video:
        ROAD_WIDTH_M = 28
        ROAD_WIDTH_PX = 163.5
        # ── Video mode ──────────────────────────────
        print(f"\nProcessing video: {VIDEO_PATH}")
        results = model.track(
            source=VIDEO_PATH,
            tracker="bytetrack.yaml",
            conf=CONF_THRESHOLD,
            persist=True,
            stream=True
        )

        for result in results:
            frame = result.orig_img.copy()
            frame = process_frame(frame, result, track_history, FPS, SPEED_LIMIT_KMH)
            cv2.imshow("Traffic Law Compliance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    else:
        ROAD_WIDTH_M = 7
        ROAD_WIDTH_PX = 230
        # ── Image dataset mode ──────────────────────
        print(f"\nProcessing images from: {IMAGE_DIR}")
        valid_ext   = (".jpg", ".jpeg", ".png", ".bmp")
        frame_files = sorted([
            f for f in os.listdir(IMAGE_DIR)
            if f.lower().endswith(valid_ext)
        ])

        for frame_file in frame_files:
            frame_path = os.path.join(IMAGE_DIR, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Skipping unreadable file: {frame_file}")
                continue

            # model.track with persist=True maintains IDs across calls
            results = model.track(
                source=frame,
                tracker="bytetrack.yaml",
                conf=CONF_THRESHOLD,
                persist=True,
                stream=False
            )

            if not results:
                continue

            for result in results:
                frame = process_frame(frame, result, track_history, FPS, SPEED_LIMIT_KMH)

            cv2.imshow("Traffic Law Compliance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()