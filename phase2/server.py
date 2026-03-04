"""
Smart Helmet Phase 2 — Optical Flow Fall Detection Server

Uses ego-motion compensation + optical flow to detect falling objects
WITHOUT any AI model. Pure OpenCV computer vision math.

The camera is on a moving worker's helmet, so we must separate
the worker's head movement from independently falling objects.

Algorithm:
  1. Receive JPEG frame from Pi client
  2. Find strong features (Shi-Tomasi corners) in prev + curr frame
  3. Track features with Lucas-Kanade sparse optical flow
  4. Estimate affine transform → how the camera moved
  5. Warp prev frame to align with curr frame (stabilize)
  6. Frame difference on stabilized pair → moving blobs
  7. Reject if too much global motion (= camera shake)
  8. Track blob centroids across frames
  9. If a blob moves downward fast and consistently → ALERT

Protocol (same as Phase 1):
  Client sends: [4-byte big-endian length][JPEG bytes]
  Server sends: 0x00 (safe) or 0x01 (alert)

Usage:
  python server.py                    # headless
  python server.py --show             # show detection overlay
  python server.py --port 5555        # custom port
"""

import argparse
import cv2
import numpy as np
import socket
import struct
import time

# ============================================================================
# CONFIGURATION — tuned for realistic construction site helmet camera
# ============================================================================

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5555

# Ego-motion feature tracking
FEATURE_MAX_CORNERS = 300
FEATURE_QUALITY = 0.01
FEATURE_MIN_DISTANCE = 20
MIN_FEATURES_FOR_TRANSFORM = 10

# Fall detection thresholds
MIN_BLOB_AREA = 800               # a brick at 3-5m is ~500-800px² in 640x480
FRAME_DIFF_THRESHOLD = 35         # slightly lower to catch distant objects
FALL_VELOCITY_THRESHOLD = 8.0     # at distance, objects move fewer px/frame
MIN_FALL_FRAMES = 3               # need 3 consistent fast-downward frames
ALERT_COOLDOWN = 3.0              # seconds between alerts
MAX_DISPERSED_MOTION = 25.0       # reject if >25% of frame AND motion is spread out
MIN_CONCENTRATION = 0.40          # if largest blob is >40% of total motion, it's localized (not shake)

# Morphology kernel
MORPH_KERNEL_SIZE = 9

# Lucas-Kanade optical flow params
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# Protocol
RESPONSE_SAFE = b'\x00'
RESPONSE_ALERT = b'\x01'


# ============================================================================
# FALLING OBJECT DETECTOR
# ============================================================================

class FallingObjectDetector:
    """Detects falling objects using ego-motion compensated frame differencing.
    
    Designed for a helmet-mounted camera on a moving construction worker.
    Filters out:
      - Camera ego-motion (worker head movement)
      - Global residual motion (imperfect stabilization / camera shake)
      - Small noise blobs (compression artifacts, shadows)
      - Horizontal/upward motion (not falling)
    Only alerts for large objects with consistent, fast downward motion.
    """

    def __init__(self):
        self.prev_gray = None
        self.tracked_blobs = {}  # id -> (cx, cy, fall_count)
        self.next_blob_id = 0
        self.last_alert_time = 0.0
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
        )

    def process(self, frame):
        """
        Process a new frame. Returns (alert, debug_frame).
        alert: True if a falling object was detected.
        debug_frame: frame with detection overlay drawn on it.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        debug_frame = frame.copy()
        h, w = gray.shape

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, debug_frame

        # --- Step 1: Find features in previous frame ---
        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=FEATURE_MAX_CORNERS,
            qualityLevel=FEATURE_QUALITY,
            minDistance=FEATURE_MIN_DISTANCE
        )

        if prev_pts is None or len(prev_pts) < MIN_FEATURES_FOR_TRANSFORM:
            self.prev_gray = gray
            return False, debug_frame

        # --- Step 2: Track features with Lucas-Kanade ---
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **LK_PARAMS
        )

        mask = status.flatten() == 1
        good_prev = prev_pts[mask]
        good_curr = curr_pts[mask]

        if len(good_prev) < MIN_FEATURES_FOR_TRANSFORM:
            self.prev_gray = gray
            return False, debug_frame

        # --- Step 3: Estimate camera motion (affine transform) ---
        M, inliers = cv2.estimateAffinePartial2D(
            good_prev, good_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0
        )

        if M is None:
            self.prev_gray = gray
            return False, debug_frame

        # --- Step 4: Stabilize — warp prev frame to align with current ---
        stabilized_prev = cv2.warpAffine(self.prev_gray, M, (w, h))

        # --- Step 5: Frame difference on stabilized pair ---
        diff = cv2.absdiff(stabilized_prev, gray)
        _, thresh = cv2.threshold(diff, FRAME_DIFF_THRESHOLD, 255,
                                  cv2.THRESH_BINARY)

        # Clean up noise
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel)
        thresh = cv2.dilate(thresh, self.kernel, iterations=1)

        # --- SMART GLOBAL MOTION REJECTION ---
        # Camera shake = many small blobs spread across the whole frame.
        # Falling object = 1-2 large blobs concentrated in one area.
        # Only reject if motion is BOTH high AND dispersed.
        motion_pixels = np.count_nonzero(thresh)
        motion_percent = (motion_pixels / (h * w)) * 100.0

        # Find all contours to check if motion is localized or spread out
        all_contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Calculate concentration: what fraction of motion is in the largest blob?
        largest_area = 0
        for cnt in all_contours:
            a = cv2.contourArea(cnt)
            if a > largest_area:
                largest_area = a

        concentration = largest_area / max(motion_pixels, 1)
        is_dispersed = concentration < MIN_CONCENTRATION

        if motion_percent > MAX_DISPERSED_MOTION and is_dispersed:
            # High motion AND spread out = camera shake
            cv2.putText(debug_frame,
                        f"Camera shake: {motion_percent:.0f}% dispersed (ignored)",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 165, 255), 1)
            self.prev_gray = gray
            self.tracked_blobs = {}
            return False, debug_frame

        # Show motion info on debug
        cv2.putText(debug_frame,
                    f"Motion: {motion_percent:.1f}%  Conc: {concentration:.0%}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (100, 100, 100), 1)

        # --- Step 6: Find moving blobs ---
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        current_blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_BLOB_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            cx = x + bw // 2
            cy = y + bh // 2
            current_blobs.append((cx, cy, x, y, bw, bh, area))

        # --- Step 7: Match blobs & check for falls ---
        alert = False
        now = time.time()
        matched_ids = set()
        new_tracked = {}

        for (cx, cy, x, y, bw, bh, area) in current_blobs:
            best_id = None
            best_dist = 999999

            for blob_id, (px, py, fall_count) in self.tracked_blobs.items():
                if blob_id in matched_ids:
                    continue
                dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                if dist < 200 and dist < best_dist:
                    best_dist = dist
                    best_id = blob_id

            if best_id is not None:
                matched_ids.add(best_id)
                px, py, fall_count = self.tracked_blobs[best_id]
                dy = cy - py  # positive = downward
                dx = abs(cx - px)

                # A real falling object:
                #  1. Moves FAST downward (> threshold)
                #  2. Moves mostly VERTICALLY (dy > dx * 1.5)
                is_falling = (dy > FALL_VELOCITY_THRESHOLD and
                              dy > dx * 1.5)

                if is_falling:
                    fall_count += 1
                else:
                    fall_count = max(0, fall_count - 1)

                # Draw debug info
                color = (0, 0, 255) if fall_count >= MIN_FALL_FRAMES else (0, 255, 0)
                cv2.rectangle(debug_frame, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(debug_frame,
                            f"dy:{dy:.0f} dx:{dx:.0f} f:{fall_count}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
                cv2.arrowedLine(debug_frame, (px, py), (cx, cy), color, 2)

                # Trigger alert
                if fall_count >= MIN_FALL_FRAMES:
                    if (now - self.last_alert_time) > ALERT_COOLDOWN:
                        alert = True
                        self.last_alert_time = now
                        fall_count = 0
                        print(f"🚨 FALLING OBJECT at ({cx},{cy}), "
                              f"dy={dy:.1f}px, area={area}px²")

                new_tracked[best_id] = (cx, cy, fall_count)
            else:
                # New blob — start tracking, don't alert yet
                new_id = self.next_blob_id
                self.next_blob_id += 1
                new_tracked[new_id] = (cx, cy, 0)
                cv2.rectangle(debug_frame, (x, y), (x + bw, y + bh),
                              (255, 255, 0), 1)

        self.tracked_blobs = new_tracked
        self.prev_gray = gray
        return alert, debug_frame


# ============================================================================
# NETWORK HELPERS
# ============================================================================

def recv_exact(sock, n):
    """Receive exactly n bytes."""
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


# ============================================================================
# SERVER
# ============================================================================

def run_server(host, port, show_window):
    detector = FallingObjectDetector()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(1)

    print("=" * 60)
    print("🔬 SMART HELMET — PHASE 2 SERVER (Optical Flow)")
    print("=" * 60)
    print(f"🌐 Listening on {host}:{port}")
    print(f"📺 Display: {'ON' if show_window else 'OFF'}")
    print(f"⚙️  Thresholds: blob>{MIN_BLOB_AREA}px², "
          f"velocity>{FALL_VELOCITY_THRESHOLD}px/f, "
          f"frames>{MIN_FALL_FRAMES}, "
          f"maxDispersed<{MAX_DISPERSED_MOTION}%")
    print("⏳ Waiting for client connection...")

    fps_time = time.time()
    fps_count = 0
    fps_display = 0

    try:
        while True:
            conn, addr = sock.accept()
            print(f"✅ Client connected from {addr}")
            detector = FallingObjectDetector()

            try:
                while True:
                    raw_len = recv_exact(conn, 4)
                    if raw_len is None:
                        break
                    frame_len = struct.unpack('>I', raw_len)[0]

                    if frame_len == 0 or frame_len > 10_000_000:
                        break

                    raw_frame = recv_exact(conn, frame_len)
                    if raw_frame is None:
                        break

                    np_arr = np.frombuffer(raw_frame, dtype=np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        conn.sendall(RESPONSE_SAFE)
                        continue

                    alert, debug_frame = detector.process(frame)
                    conn.sendall(RESPONSE_ALERT if alert else RESPONSE_SAFE)

                    fps_count += 1
                    if time.time() - fps_time > 1.0:
                        fps_display = fps_count
                        fps_count = 0
                        fps_time = time.time()

                    if show_window:
                        cv2.putText(debug_frame, f"Server FPS: {fps_display}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2)
                        cv2.imshow("Phase 2 - Optical Flow Detection",
                                   debug_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            return

            except (ConnectionResetError, BrokenPipeError):
                pass
            finally:
                conn.close()
                print("📡 Client disconnected, waiting for new connection...")

    except KeyboardInterrupt:
        print("\n👋 Server shutting down...")
    finally:
        sock.close()
        if show_window:
            cv2.destroyAllWindows()
        print("✅ Server stopped")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smart Helmet Phase 2 — Optical Flow Detection Server"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--show", action="store_true",
                        help="Show detection debug window")
    args = parser.parse_args()

    run_server(args.host, args.port, args.show)
