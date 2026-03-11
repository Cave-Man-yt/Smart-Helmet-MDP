"""
AI-Augmented Smart Safety Helmet - ESP32-CAM VERSION
Motion-Based Fall Detection (Class-Agnostic)

Uses OpenCV background subtraction (MOG2) to detect ANY moving object,
then tracks vertical velocity to detect falls. No AI model needed.
Works with any falling object: bricks, tools, debris, phones, etc.

CONTROLS:
  Q - Quit
  D - Toggle debug mode
  S - Toggle sensitivity (Low / Medium / High)
"""

import cv2
import numpy as np
import pygame
import time
import threading
import urllib.request

# ============================================================================
# CONFIGURATION
# ============================================================================

# Fall Detection Thresholds
FALL_VELOCITY_THRESHOLD = 15.0  # px/frame downward speed to flag as falling
MIN_CONSECUTIVE_FRAMES = 2      # Frames above threshold before alerting

# Motion Detection
MIN_CONTOUR_AREA = 1500    # Ignore blobs smaller than this (noise filter)
MAX_CONTOUR_AREA = 200000  # Ignore blobs larger than this (whole-frame noise)
LEARNING_RATE = 0.005      # How fast the background model adapts (lower = more stable)

# Tracking
MAX_TRACKING_DISTANCE = 200  # Max px distance to match a blob between frames
MAX_MISSED_FRAMES = 5        # Remove tracked object after this many missed frames

# Ego-Motion Compensation
CAMERA_MOTION_RATIO_THRESHOLD = 0.25  # If >25% of frame is foreground, it's camera motion
CAMERA_MOTION_FAST_LEARN = 0.05       # Faster learning rate when camera is moving

# Alert Configuration
ALERT_DURATION = 2.0
ALERT_COOLDOWN = 3.0

# Camera Configuration
STREAM_URL = "http://192.168.29.27:81/stream"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Debug mode
DEBUG_MODE = True

# Sensitivity presets
SENSITIVITY_PRESETS = {
    'LOW':    {'min_area': 3000, 'velocity': 25.0, 'consecutive': 3},
    'MEDIUM': {'min_area': 1500, 'velocity': 15.0, 'consecutive': 2},
    'HIGH':   {'min_area': 800,  'velocity': 8.0,  'consecutive': 2},
}
CURRENT_SENSITIVITY = 'MEDIUM'

# ============================================================================
# ROBUST MJPEG STREAM READER (bypasses ffmpeg crash bugs)
# ============================================================================

class VideoStreamWidget:
    def __init__(self, url):
        self.url = url
        self.status = False
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            try:
                stream = urllib.request.urlopen(self.url, timeout=5)
                bytes_data = b''
                while self.running:
                    chunk = stream.read(4096)
                    if not chunk:
                        break
                    bytes_data += chunk
                    a = bytes_data.find(b'\xff\xd8')  # JPEG start
                    b = bytes_data.find(b'\xff\xd9')  # JPEG end
                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        if len(jpg) > 0:
                            frame = cv2.imdecode(
                                np.frombuffer(jpg, dtype=np.uint8),
                                cv2.IMREAD_COLOR
                            )
                            if frame is not None:
                                self.status = True
                                self.frame = frame
            except Exception:
                self.status = False
                if self.running:
                    time.sleep(1.0)

    def read(self):
        return self.status, self.frame

    def release(self):
        self.running = False

# ============================================================================
# AUDIO SYSTEM
# ============================================================================

class AudioAlertSystem:
    def __init__(self):
        self.alert_active = False
        self.audio_available = False
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self.generate_beep_sound()
            self.audio_available = True
        except Exception as e:
            print(f"⚠️  Audio not available ({e}). Visual alerts only.")

    def generate_beep_sound(self):
        sample_rate = 22050
        duration = 0.3
        frequency = 1000

        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        buf = np.sin(2 * np.pi * frequency * t)
        buf = (buf * 32767).astype(np.int16)
        stereo_buf = np.column_stack((buf, buf))

        self.beep_sound = pygame.sndarray.make_sound(stereo_buf)

    def play_alert(self):
        if not self.alert_active:
            if self.audio_available:
                self.beep_sound.play()
            else:
                print('\a', end='', flush=True)
            self.alert_active = True

    def stop_alert(self):
        self.alert_active = False

# ============================================================================
# CENTROID TRACKER (Class-Agnostic)
# ============================================================================

class CentroidTracker:
    """
    Tracks moving blobs by their centroid position across frames.
    No class labels needed — purely position-based matching.
    """
    def __init__(self):
        self.next_id = 0
        self.objects = {}  # id -> {'cx', 'cy', 'fall_count', 'missed', 'trail'}
        self.last_alert_time = 0

    def update(self, contours, current_time, ego_motion_y=0.0):
        """
        Match new contour centroids to existing tracked objects.
        ego_motion_y: estimated vertical camera movement (subtracted from velocity)
        Returns: (fall_detected, tracked_objects_dict)
        """
        fall_detected = False

        # Extract centroids and bounding rects from contours
        input_centroids = []
        input_rects = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2
            input_centroids.append((cx, cy))
            input_rects.append((x, y, w, h))

        # If no contours detected, increment missed count for all
        if len(input_centroids) == 0:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id]['missed'] += 1
                if self.objects[obj_id]['missed'] > MAX_MISSED_FRAMES:
                    del self.objects[obj_id]
            return fall_detected, self.objects

        # If no existing objects, register all new centroids
        if len(self.objects) == 0:
            for i, (cx, cy) in enumerate(input_centroids):
                self._register(cx, cy, input_rects[i])
            return fall_detected, self.objects

        # Match existing objects to new centroids by nearest distance
        obj_ids = list(self.objects.keys())
        obj_centroids = [(self.objects[oid]['cx'], self.objects[oid]['cy']) for oid in obj_ids]

        used_inputs = set()
        used_objects = set()

        # Build distance matrix and greedily match
        matches = []
        for i, (ocx, ocy) in enumerate(obj_centroids):
            best_dist = float('inf')
            best_j = -1
            for j, (icx, icy) in enumerate(input_centroids):
                dist = np.sqrt((ocx - icx)**2 + (ocy - icy)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_dist < MAX_TRACKING_DISTANCE:
                matches.append((i, best_j, best_dist))

        # Sort by distance and assign greedily
        matches.sort(key=lambda x: x[2])
        for obj_idx, inp_idx, dist in matches:
            if obj_idx in used_objects or inp_idx in used_inputs:
                continue

            obj_id = obj_ids[obj_idx]
            new_cx, new_cy = input_centroids[inp_idx]
            old_cy = self.objects[obj_id]['cy']

            # Calculate vertical velocity (positive = downward)
            # Subtract camera ego-motion so camera panning doesn't trigger falls
            raw_velocity = new_cy - old_cy
            velocity = raw_velocity - ego_motion_y

            if DEBUG_MODE:
                print(f"  Object #{obj_id}: raw_vel={raw_velocity:.1f}, "
                      f"ego={ego_motion_y:.1f}, adj_vel={velocity:.1f}, "
                      f"pos=({new_cx},{new_cy})")

            # Check for fall
            if velocity > FALL_VELOCITY_THRESHOLD:
                self.objects[obj_id]['fall_count'] += 1
                if self.objects[obj_id]['fall_count'] >= MIN_CONSECUTIVE_FRAMES:
                    if (current_time - self.last_alert_time) > ALERT_COOLDOWN:
                        fall_detected = True
                        self.last_alert_time = current_time
                    self.objects[obj_id]['fall_count'] = 0
            elif velocity < -2:  # Moving upward significantly
                self.objects[obj_id]['fall_count'] = max(
                    0, self.objects[obj_id]['fall_count'] - 1
                )

            # Update position
            self.objects[obj_id]['cx'] = new_cx
            self.objects[obj_id]['cy'] = new_cy
            self.objects[obj_id]['missed'] = 0
            self.objects[obj_id]['rect'] = input_rects[inp_idx]

            # Keep a short trail for visualization
            trail = self.objects[obj_id]['trail']
            trail.append((new_cx, new_cy))
            if len(trail) > 15:
                trail.pop(0)

            used_objects.add(obj_idx)
            used_inputs.add(inp_idx)

        # Register unmatched inputs as new objects
        for j in range(len(input_centroids)):
            if j not in used_inputs:
                cx, cy = input_centroids[j]
                self._register(cx, cy, input_rects[j])

        # Increment missed for unmatched existing objects
        for i in range(len(obj_ids)):
            if i not in used_objects:
                obj_id = obj_ids[i]
                self.objects[obj_id]['missed'] += 1
                if self.objects[obj_id]['missed'] > MAX_MISSED_FRAMES:
                    del self.objects[obj_id]

        return fall_detected, self.objects

    def _register(self, cx, cy, rect):
        self.objects[self.next_id] = {
            'cx': cx, 'cy': cy,
            'fall_count': 0, 'missed': 0,
            'rect': rect,
            'trail': [(cx, cy)]
        }
        if DEBUG_MODE:
            print(f"  New blob tracked (ID: {self.next_id}) at ({cx}, {cy})")
        self.next_id += 1

# ============================================================================
# VISUAL ALERT RENDERING
# ============================================================================

class VisualAlertRenderer:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.alert_start_time = None

    def render_alert(self, frame):
        # Red border
        cv2.rectangle(frame, (0, 0),
                      (self.frame_width, self.frame_height),
                      (0, 0, 255), 20)

        # Red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0),
                      (self.frame_width, self.frame_height),
                      (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # DANGER text
        text = "DANGER!"
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 3
        thickness = 8

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        tx = (self.frame_width - tw) // 2
        ty = (self.frame_height + th) // 2

        # Shadow + text
        cv2.putText(frame, text, (tx, ty), font, font_scale,
                    (0, 0, 0), thickness + 4, cv2.LINE_AA)
        cv2.putText(frame, text, (tx, ty), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

        return frame

    def start_alert(self):
        self.alert_start_time = time.time()

    def is_alert_active(self):
        if self.alert_start_time is None:
            return False
        if time.time() - self.alert_start_time > ALERT_DURATION:
            self.alert_start_time = None
            return False
        return True

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class SmartHelmetMotionDetector:
    def __init__(self):
        print("🎯 Smart Safety Helmet - Motion-Based Fall Detection")
        print("=" * 70)

        # Initialize background subtractor
        print("📦 Initializing motion detector (MOG2)...")
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True
        )
        # Morphological kernels for cleaning the mask
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        print("✅ Motion detector ready")

        # Initialize components
        self.audio_system = AudioAlertSystem()
        self.tracker = CentroidTracker()
        self.visual_alert = VisualAlertRenderer(FRAME_WIDTH, FRAME_HEIGHT)

        # Optical flow for ego-motion estimation
        self.prev_gray = None
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        self.feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=7
        )

        # Connect to ESP32-CAM
        print(f"📷 Connecting to ESP32-CAM at {STREAM_URL} ...")
        self.cap = VideoStreamWidget(STREAM_URL)
        print("✅ Connected. Waiting for first frame...")
        time.sleep(2)

        print("=" * 70)
        print("🚀 System Ready!")
        print(f"⚡ Fall Velocity Threshold: {FALL_VELOCITY_THRESHOLD} px/frame")
        print(f"📐 Min Contour Area: {MIN_CONTOUR_AREA} px²")
        print(f"🎚️  Sensitivity: {CURRENT_SENSITIVITY}")
        print(f"🐛 Debug: {'ON' if DEBUG_MODE else 'OFF'}")
        print("=" * 70)
        print("💡 Controls:")
        print("   Q - Quit")
        print("   D - Toggle debug mode")
        print("   S - Cycle sensitivity (LOW / MEDIUM / HIGH)")
        print()

    def process_frame(self, frame):
        global MIN_CONTOUR_AREA, FALL_VELOCITY_THRESHOLD, MIN_CONSECUTIVE_FRAMES

        # Resize if needed
        h, w = frame.shape[:2]
        if w != FRAME_WIDTH or h != FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Convert to grayscale for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- EGO-MOTION ESTIMATION ---
        ego_motion_y = 0.0
        camera_is_moving = False

        if self.prev_gray is not None:
            # Find good features to track in previous frame
            prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, **self.feature_params)

            if prev_pts is not None and len(prev_pts) > 10:
                # Calculate optical flow
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, prev_pts, None, **self.lk_params
                )

                if next_pts is not None:
                    # Filter good points
                    good_mask = status.flatten() == 1
                    good_prev = prev_pts[good_mask]
                    good_next = next_pts[good_mask]

                    if len(good_prev) > 5:
                        # Calculate flow vectors
                        flow = good_next - good_prev
                        flow_y = flow[:, 0, 1]  # Vertical component

                        # Median flow = global camera motion estimate
                        ego_motion_y = float(np.median(flow_y))

                        # Check if most features are moving similarly (= camera motion)
                        flow_std = float(np.std(flow_y))
                        if flow_std < 8.0 and abs(ego_motion_y) > 3.0:
                            camera_is_moving = True

        self.prev_gray = gray.copy()

        # --- BACKGROUND SUBTRACTION ---
        # Use faster learning rate when camera is moving so MOG2 adapts quickly
        lr = CAMERA_MOTION_FAST_LEARN if camera_is_moving else LEARNING_RATE
        fg_mask = self.bg_subtractor.apply(frame, learningRate=lr)

        # Remove shadows (MOG2 marks them as 127, foreground as 255)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close)

        # --- GLOBAL MOTION RATIO CHECK ---
        # If too much of the frame is foreground, it's camera motion, not objects
        fg_ratio = np.count_nonzero(fg_mask) / (FRAME_WIDTH * FRAME_HEIGHT)
        if fg_ratio > CAMERA_MOTION_RATIO_THRESHOLD:
            camera_is_moving = True

        # Find contours of moving objects
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area
        valid_contours = []
        if not camera_is_moving:  # Skip contour tracking during heavy camera motion
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA:
                    valid_contours.append(cnt)

        # Update tracker with ego-motion compensation
        current_time = time.time()
        fall_detected, tracked = self.tracker.update(
            valid_contours, current_time, ego_motion_y
        )

        if DEBUG_MODE and (len(valid_contours) > 0 or camera_is_moving):
            status_str = "📷 CAMERA MOVING" if camera_is_moving else ""
            print(f"\n[Frame] Blobs: {len(valid_contours)}, "
                  f"Tracked: {len(tracked)}, "
                  f"ego_y={ego_motion_y:.1f}, fg={fg_ratio:.1%} {status_str}")

        # Draw bounding boxes and trails for tracked objects
        for obj_id, obj in tracked.items():
            if obj['missed'] > 0:
                continue

            x, y, bw, bh = obj['rect']
            fall_count = obj['fall_count']

            if fall_count >= MIN_CONSECUTIVE_FRAMES - 1:
                color = (0, 0, 255)    # Red - imminent
            elif fall_count > 0:
                color = (0, 165, 255)  # Orange - building
            else:
                color = (0, 255, 0)    # Green - normal

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

            label = f"ID:{obj_id}"
            if fall_count > 0:
                label += f" FALLING({fall_count})"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            trail = obj['trail']
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                pt1 = trail[i - 1]
                pt2 = trail[i]
                thickness = max(1, int(alpha * 3))
                cv2.line(frame, pt1, pt2, color, thickness)

        # Camera motion indicator on frame
        if camera_is_moving:
            cv2.putText(frame, "CAMERA MOVING - Fall detection paused",
                        (10, FRAME_HEIGHT - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        return frame, fg_mask, fall_detected, len(valid_contours)

    def run(self):
        global DEBUG_MODE, CURRENT_SENSITIVITY
        global FALL_VELOCITY_THRESHOLD, MIN_CONSECUTIVE_FRAMES, MIN_CONTOUR_AREA

        fps_time = time.time()
        fps_counter = 0
        fps_display = 0
        sensitivity_keys = list(SENSITIVITY_PRESETS.keys())
        current_sens_idx = sensitivity_keys.index(CURRENT_SENSITIVITY)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # Process frame
                processed, fg_mask, fall_detected, blob_count = self.process_frame(frame)

                # Handle fall
                if fall_detected:
                    print("⚠️  FALL DETECTED! Activating alerts...")
                    self.visual_alert.start_alert()
                    self.audio_system.play_alert()

                # Visual alert overlay
                if self.visual_alert.is_alert_active():
                    processed = self.visual_alert.render_alert(processed)
                else:
                    self.audio_system.stop_alert()

                # FPS counter
                fps_counter += 1
                if time.time() - fps_time > 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    fps_time = time.time()

                # Status overlay
                y_offset = 30
                status_items = [
                    f"FPS: {fps_display}",
                    f"Moving Blobs: {blob_count}",
                    f"Tracked: {len(self.tracker.objects)}",
                    f"Sensitivity: {CURRENT_SENSITIVITY}",
                    f"Debug: {'ON' if DEBUG_MODE else 'OFF'}"
                ]
                for item in status_items:
                    cv2.putText(processed, item, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25

                # Instructions
                instructions = [
                    "Q: Quit | D: Debug | S: Sensitivity",
                    "GREEN = Tracked | ORANGE = Falling | RED = Alert imminent"
                ]
                y_bottom = FRAME_HEIGHT - 15
                for inst in reversed(instructions):
                    cv2.putText(processed, inst, (10, y_bottom),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_bottom -= 20

                # Show main view
                cv2.imshow('Smart Safety Helmet - Motion Detection', processed)

                # Show debug mask if debug mode
                if DEBUG_MODE:
                    fg_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
                    cv2.imshow('Motion Mask (Debug)', fg_color)

                # Keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\n👋 Shutting down...")
                    break
                elif key == ord('d'):
                    DEBUG_MODE = not DEBUG_MODE
                    print(f"\n🐛 Debug mode: {'ON' if DEBUG_MODE else 'OFF'}")
                    if not DEBUG_MODE:
                        cv2.destroyWindow('Motion Mask (Debug)')
                elif key == ord('s'):
                    current_sens_idx = (current_sens_idx + 1) % len(sensitivity_keys)
                    CURRENT_SENSITIVITY = sensitivity_keys[current_sens_idx]
                    preset = SENSITIVITY_PRESETS[CURRENT_SENSITIVITY]
                    MIN_CONTOUR_AREA = preset['min_area']
                    FALL_VELOCITY_THRESHOLD = preset['velocity']
                    MIN_CONSECUTIVE_FRAMES = preset['consecutive']
                    print(f"\n🎚️  Sensitivity: {CURRENT_SENSITIVITY} "
                          f"(area>{MIN_CONTOUR_AREA}, vel>{FALL_VELOCITY_THRESHOLD}, "
                          f"frames>={MIN_CONSECUTIVE_FRAMES})")

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            try:
                pygame.mixer.quit()
            except Exception:
                pass
            print("✅ Cleanup complete")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("💡 MOTION-BASED FALL DETECTION")
    print("=" * 70)
    print("Detects ANY falling object — no class labels needed!")
    print("Works with: bricks, tools, phones, debris, anything")
    print()
    print("1. Ensure ESP32-CAM is on and streaming")
    print("2. Keep the camera steady (mounted on helmet)")
    print("3. Press 'S' to change sensitivity if needed")
    print("=" * 70)
    print()

    try:
        app = SmartHelmetMotionDetector()
        app.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()