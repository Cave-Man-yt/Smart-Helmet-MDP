"""
AI-Augmented Smart Safety Helmet - Phase 1: IMPROVED VERSION
Better object detection with lower confidence threshold

IMPROVEMENTS:
- Lower confidence threshold (0.25 instead of 0.5)
- More target object classes
- Better visual feedback
- Debug information display
"""

import cv2
import torch
import numpy as np
import pygame
import time
from collections import deque

# ============================================================================
# CONFIGURATION PARAMETERS - OPTIMIZED FOR BETTER DETECTION
# ============================================================================

# Fall Detection Thresholds
FALL_VELOCITY_THRESHOLD = 5.0
MIN_CONSECUTIVE_FRAMES = 2

# IMPROVED: Lower confidence for better detection
DETECTION_CONFIDENCE = 0.25  # Changed from 0.5 to 0.25

# IMPROVED: More target objects (EXCLUDING 'person' to avoid false positives)
TARGET_CLASSES = [
    'cup', 'bottle', 'cell phone', 'book', 'scissors',
    'laptop', 'mouse', 'keyboard', 'remote', 'clock',
    'vase', 'bowl', 'banana', 'apple', 'orange',
    'spoon', 'fork', 'knife', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'sports ball'
    # NOTE: 'person' is intentionally excluded to prevent 
    # false alerts when workers move or lean down
]

# Alert Configuration
ALERT_DURATION = 2.0
ALERT_COOLDOWN = 1.0

# Camera Configuration
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# Debug mode - set True for verbose logging (floods terminal)
DEBUG_MODE = False

# ============================================================================
# AUDIO SYSTEM
# ============================================================================

class AudioAlertSystem:
    def __init__(self):
        self.audio_available = False
        self.alert_active = False
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self.audio_available = True
            self.generate_beep_sound()
        except (NotImplementedError, Exception) as e:
            print(f"⚠️  Audio mixer not available: {e}")
            print("🔊 System will continue without audio alerts")
    
    def generate_beep_sound(self):
        if not self.audio_available:
            return
        sample_rate = 22050
        duration = 0.2
        frequency = 800
        
        n_samples = int(duration * sample_rate)
        buf = np.sin(2 * np.pi * frequency * np.linspace(0, duration, n_samples))
        buf = (buf * 32767).astype(np.int16)
        stereo_buf = np.column_stack((buf, buf))
        
        self.beep_sound = pygame.sndarray.make_sound(stereo_buf)
    
    def play_alert(self):
        if self.audio_available and not self.alert_active:
            self.beep_sound.play()
            self.alert_active = True
    
    def stop_alert(self):
        self.alert_active = False

# ============================================================================
# FALL DETECTION LOGIC
# ============================================================================

class FallDetector:
    def __init__(self, velocity_threshold=FALL_VELOCITY_THRESHOLD, 
                 min_frames=MIN_CONSECUTIVE_FRAMES):
        self.velocity_threshold = velocity_threshold
        self.min_frames = min_frames
        self.tracked_objects = {}
        self.fall_detected_count = 0
        self.last_alert_time = 0
    
    def calculate_velocity(self, current_y, previous_y):
        return current_y - previous_y
    
    def update_tracking(self, detections, current_frame_time):
        current_objects = {}
        fall_detected = False
        
        for detection in detections:
            class_name, conf, x1, y1, x2, y2 = detection
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            matched = False
            for obj_id, obj_data in self.tracked_objects.items():
                prev_center_x, prev_center_y, prev_class = obj_data
                
                distance = np.sqrt((center_x - prev_center_x)**2 + 
                                 (center_y - prev_center_y)**2)
                
                if distance < 300 and class_name == prev_class:
                    velocity = self.calculate_velocity(center_y, prev_center_y)
                    
                    if DEBUG_MODE:
                        print(f"  Object '{class_name}': velocity = {velocity:.2f} px/frame")
                    
                    if velocity > self.velocity_threshold:
                        self.fall_detected_count += 1
                        
                        if self.fall_detected_count >= self.min_frames:
                            if (current_frame_time - self.last_alert_time) > ALERT_COOLDOWN:
                                fall_detected = True
                                self.last_alert_time = current_frame_time
                                self.fall_detected_count = 0
                    else:
                        self.fall_detected_count = max(0, self.fall_detected_count - 1)
                    
                    matched = True
                    current_objects[obj_id] = (center_x, center_y, class_name)
                    break
            
            if not matched:
                new_id = len(current_objects) + len(self.tracked_objects)
                current_objects[new_id] = (center_x, center_y, class_name)
                if DEBUG_MODE:
                    print(f"  New object tracked: '{class_name}' (ID: {new_id})")
        
        self.tracked_objects = current_objects
        return fall_detected

# ============================================================================
# VISUAL ALERT RENDERING
# ============================================================================

class VisualAlertRenderer:
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.alert_start_time = None
    
    def render_alert(self, frame):
        border_thickness = 20
        cv2.rectangle(frame, 
                     (0, 0), 
                     (self.frame_width, self.frame_height),
                     (0, 0, 255), 
                     border_thickness)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height),
                     (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        text = "DANGER!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        thickness = 8
        
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (self.frame_width - text_width) // 2
        text_y = (self.frame_height + text_height) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale,
                   (0, 0, 0), thickness + 4, cv2.LINE_AA)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale,
                   (255, 255, 255), thickness, cv2.LINE_AA)
        
        return frame
    
    def start_alert(self):
        self.alert_start_time = time.time()
    
    def is_alert_active(self):
        if self.alert_start_time is None:
            return False
        
        elapsed = time.time() - self.alert_start_time
        if elapsed > ALERT_DURATION:
            self.alert_start_time = None
            return False
        
        return True

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class SmartHelmetPhase1:
    def __init__(self):
        print("🎯 Initializing AI-Augmented Smart Safety Helmet - Phase 1 (IMPROVED)")
        print("=" * 70)
        
        # Load YOLOv5 model
        print("📦 Loading YOLOv5 model...")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.conf = DETECTION_CONFIDENCE
        print(f"✅ Model loaded (confidence threshold: {DETECTION_CONFIDENCE})")
        
        # Initialize components
        self.audio_system = AudioAlertSystem()
        self.fall_detector = FallDetector()
        self.visual_alert = VisualAlertRenderer(FRAME_WIDTH, FRAME_HEIGHT)
        
        # Initialize camera - try multiple indices
        print(f"📷 Searching for camera...")
        self.cap = None
        for idx in range(5):
            cap_test = cv2.VideoCapture(idx)
            if cap_test.isOpened():
                ret, _ = cap_test.read()
                if ret:
                    print(f"✅ Camera found at index {idx}")
                    self.cap = cap_test
                    break
                cap_test.release()
        
        if self.cap is None:
            raise RuntimeError("❌ No working camera found")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        print("=" * 70)
        print("🚀 System Ready!")
        print(f"📊 Detecting {len(TARGET_CLASSES)} object types")
        print(f"⚡ Fall Threshold: {FALL_VELOCITY_THRESHOLD} pixels/frame")
        print(f"🎯 Confidence: {DETECTION_CONFIDENCE} (lower = more detections)")
        print(f"🐛 Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}")
        print("=" * 70)
        print("💡 Controls:")
        print("   Q - Quit")
        print("   D - Toggle debug mode")
        print("   C - Change confidence (0.1 / 0.25 / 0.5)")
        print()
    
    def process_frame(self, frame):
        # Run YOLOv5 detection
        results = self.model(frame)
        
        # Parse detections
        detections = []
        all_detections = []  # For display
        
        for *box, conf, cls in results.xyxy[0]:
            class_name = self.model.names[int(cls)]
            x1, y1, x2, y2 = map(int, box)
            all_detections.append((class_name, float(conf), x1, y1, x2, y2))
            
            # Filter for target classes
            if class_name.lower() in [c.lower() for c in TARGET_CLASSES]:
                detections.append((class_name, float(conf), x1, y1, x2, y2))
        
        if DEBUG_MODE and len(all_detections) > 0:
            print(f"\n[Frame] All detections: {len(all_detections)}, Target detections: {len(detections)}")
            for class_name, conf, x1, y1, x2, y2 in all_detections[:5]:  # Show first 5
                print(f"  - {class_name}: {conf:.3f}")
        
        # Update fall detection
        current_time = time.time()
        fall_detected = self.fall_detector.update_tracking(detections, current_time)
        
        # Draw bounding boxes - GREEN for target objects, YELLOW for others
        for class_name, conf, x1, y1, x2, y2 in all_detections:
            is_target = class_name.lower() in [c.lower() for c in TARGET_CLASSES]
            color = (0, 255, 0) if is_target else (0, 255, 255)  # Green or Yellow
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {conf:.2f}"
            if is_target:
                label += " [TARGET]"
            
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame, fall_detected, len(all_detections)
    
    def run(self):
        global DEBUG_MODE  # Declare at function start
        
        fps_time = time.time()
        fps_counter = 0
        fps_display = 0
        confidence_levels = [0.1, 0.25, 0.5]
        current_conf_idx = 1  # Start with 0.25
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                
                # Process frame
                processed_frame, fall_detected, total_detections = self.process_frame(frame)
                
                # Handle fall detection
                if fall_detected:
                    print("⚠️  FALL DETECTED! Activating alerts...")
                    self.visual_alert.start_alert()
                    self.audio_system.play_alert()
                
                # Render visual alert if active
                if self.visual_alert.is_alert_active():
                    processed_frame = self.visual_alert.render_alert(processed_frame)
                else:
                    self.audio_system.stop_alert()
                
                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_time > 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    fps_time = time.time()
                
                # Add comprehensive status overlay
                y_offset = 30
                status_items = [
                    f"FPS: {fps_display}",
                    f"All Objects: {total_detections}",
                    f"Tracked: {len(self.fall_detector.tracked_objects)}",
                    f"Confidence: {self.model.conf:.2f}",
                    f"Debug: {'ON' if DEBUG_MODE else 'OFF'}"
                ]
                
                for item in status_items:
                    cv2.putText(processed_frame, item, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25
                
                # Add instructions
                instructions = [
                    "Q: Quit | D: Debug | C: Change Confidence",
                    "GREEN boxes = Target objects",
                    "YELLOW boxes = Other detected objects"
                ]
                
                y_bottom = FRAME_HEIGHT - 15
                for inst in reversed(instructions):
                    cv2.putText(processed_frame, inst, (10, y_bottom),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_bottom -= 20
                
                # Display frame
                cv2.imshow('Smart Safety Helmet - Phase 1 (IMPROVED)', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Shutting down...")
                    break
                elif key == ord('d'):
                    DEBUG_MODE = not DEBUG_MODE
                    print(f"\n🐛 Debug mode: {'ON' if DEBUG_MODE else 'OFF'}")
                elif key == ord('c'):
                    current_conf_idx = (current_conf_idx + 1) % len(confidence_levels)
                    new_conf = confidence_levels[current_conf_idx]
                    self.model.conf = new_conf
                    print(f"\n🎯 Confidence changed to: {new_conf}")
                
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            try:
                if pygame.mixer.get_init():
                    pygame.mixer.quit()
            except (NotImplementedError, Exception):
                pass
            print("✅ Cleanup complete")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("💡 TIPS FOR BETTER DETECTION:")
    print("=" * 70)
    print("1. Use good lighting - face a window or lamp")
    print("2. Hold objects 50cm - 2m from camera")
    print("3. Try these objects first: phone, cup, laptop, mouse")
    print("4. Object should fill 10-30% of the frame")
    print("5. Press 'C' to cycle confidence levels if needed")
    print("=" * 70)
    print()
    
    try:
        app = SmartHelmetPhase1()
        app.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()