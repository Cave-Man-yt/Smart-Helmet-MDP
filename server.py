"""
Smart Helmet — Detection Server
Runs on a powerful computer. Receives camera frames from the Pi client
over TCP, runs YOLOv5 + fall detection, and sends back alert signals.

Usage:
    python server.py                        # default 0.0.0.0:5555
    python server.py --port 6000            # custom port
    python server.py --show                 # show detections in OpenCV window
"""

import argparse
import cv2
import numpy as np
import socket
import struct
import time
import torch

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5555

# Detection settings — same as main script
DETECTION_CONFIDENCE = 0.25
FALL_VELOCITY_THRESHOLD = 5.0
MIN_CONSECUTIVE_FRAMES = 2
ALERT_COOLDOWN = 1.0

TARGET_CLASSES = [
    'cup', 'bottle', 'cell phone', 'book', 'scissors',
    'laptop', 'mouse', 'keyboard', 'remote', 'clock',
    'vase', 'bowl', 'banana', 'apple', 'orange',
    'spoon', 'fork', 'knife', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'sports ball'
]

# Protocol constants
RESPONSE_NO_ALERT = b'\x00'
RESPONSE_ALERT    = b'\x01'

# ============================================================================
# FALL DETECTOR (reused logic)
# ============================================================================

class FallDetector:
    def __init__(self):
        self.tracked_objects = {}
        self.fall_detected_count = 0
        self.last_alert_time = 0

    def update(self, detections, now):
        """Returns True if a fall is detected this frame."""
        current_objects = {}
        fall = False

        for class_name, conf, x1, y1, x2, y2 in detections:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            matched = False
            for oid, (px, py, pc) in self.tracked_objects.items():
                dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                if dist < 300 and class_name == pc:
                    velocity = cy - py  # downward = positive
                    if velocity > FALL_VELOCITY_THRESHOLD:
                        self.fall_detected_count += 1
                        if self.fall_detected_count >= MIN_CONSECUTIVE_FRAMES:
                            if (now - self.last_alert_time) > ALERT_COOLDOWN:
                                fall = True
                                self.last_alert_time = now
                                self.fall_detected_count = 0
                    else:
                        self.fall_detected_count = max(0, self.fall_detected_count - 1)
                    matched = True
                    current_objects[oid] = (cx, cy, class_name)
                    break

            if not matched:
                new_id = len(current_objects) + len(self.tracked_objects)
                current_objects[new_id] = (cx, cy, class_name)

        self.tracked_objects = current_objects
        return fall

# ============================================================================
# FRAME RECEIVER HELPER
# ============================================================================

def recv_exact(sock, n):
    """Receive exactly n bytes from a socket."""
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data

# ============================================================================
# SERVER
# ============================================================================

def run_server(host, port, show_window):
    # Load model
    print("📦 Loading YOLOv5 model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = DETECTION_CONFIDENCE
    print(f"✅ Model loaded (confidence: {DETECTION_CONFIDENCE})")

    fall_detector = FallDetector()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(1)
    print(f"🌐 Listening on {host}:{port}  (waiting for Pi client...)")

    try:
        while True:
            conn, addr = sock.accept()
            print(f"✅ Client connected: {addr}")

            fps_time = time.time()
            fps_count = 0
            fps_display = 0

            try:
                while True:
                    # --- Receive frame ---
                    # Protocol: 4-byte big-endian length prefix, then JPEG bytes
                    header = recv_exact(conn, 4)
                    if header is None:
                        print("⚠️  Client disconnected (header)")
                        break
                    frame_len = struct.unpack('>I', header)[0]
                    if frame_len == 0 or frame_len > 10_000_000:
                        print(f"⚠️  Bad frame length: {frame_len}")
                        break

                    jpeg_data = recv_exact(conn, frame_len)
                    if jpeg_data is None:
                        print("⚠️  Client disconnected (data)")
                        break

                    # Decode JPEG
                    frame = cv2.imdecode(
                        np.frombuffer(jpeg_data, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    if frame is None:
                        conn.sendall(RESPONSE_NO_ALERT)
                        continue

                    # --- Run detection ---
                    results = model(frame)
                    detections = []
                    for *box, conf, cls in results.xyxy[0]:
                        class_name = model.names[int(cls)]
                        if class_name.lower() in [c.lower() for c in TARGET_CLASSES]:
                            x1, y1, x2, y2 = map(int, box)
                            detections.append((class_name, float(conf), x1, y1, x2, y2))

                    # --- Fall detection ---
                    now = time.time()
                    alert = fall_detector.update(detections, now)

                    if alert:
                        print(f"⚠️  FALL DETECTED → sending alert to {addr}")
                        conn.sendall(RESPONSE_ALERT)
                    else:
                        conn.sendall(RESPONSE_NO_ALERT)

                    # FPS counter
                    fps_count += 1
                    if time.time() - fps_time > 1.0:
                        fps_display = fps_count
                        fps_count = 0
                        fps_time = time.time()

                    # Optional display window
                    if show_window:
                        for cname, cf, x1, y1, x2, y2 in detections:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{cname}: {cf:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        status = f"FPS: {fps_display}  Objects: {len(detections)}"
                        if alert:
                            status += "  ** ALERT **"
                        cv2.putText(frame, status, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Server — Detection Feed", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("👋 Quit requested from server window")
                            return

            except (ConnectionResetError, BrokenPipeError):
                print("⚠️  Client connection lost")
            finally:
                conn.close()
                print("🔄 Waiting for next client connection...")

    except KeyboardInterrupt:
        print("\n👋 Server shutting down")
    finally:
        sock.close()
        if show_window:
            cv2.destroyAllWindows()
        print("✅ Server stopped")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Helmet Detection Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind address (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")
    parser.add_argument("--show", action="store_true", help="Show detection feed in an OpenCV window")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🌐 SMART HELMET — DETECTION SERVER")
    print("=" * 60)
    run_server(args.host, args.port, args.show)
