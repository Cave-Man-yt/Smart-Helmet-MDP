"""
Smart Helmet Phase 2 — Pi Client
Captures camera frames, streams to the Phase 2 optical flow server,
and triggers alerts on falling object detection.

Usage:
  python client.py --server localhost           # laptop testing
  python client.py --server 192.168.1.100       # remote server
  python client.py --server 192.168.1.100 --gpio 17   # Pi with buzzer
  python client.py --server localhost --no-show        # headless mode
"""

import cv2
import numpy as np
import socket
import struct
import sys
import threading
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_PORT = 5555
DEFAULT_QUALITY = 70
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
ALERT_DURATION = 2.0

RESPONSE_ALERT = b'\x01'

WINDOW_NAME = "Smart Helmet Phase 2 - Client"

# ============================================================================
# ALERT BACKENDS
# ============================================================================

class BuzzerGPIO:
    """Hardware buzzer via Raspberry Pi GPIO."""
    def __init__(self, pin):
        import RPi.GPIO as GPIO
        self.GPIO = GPIO
        self.pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
        print(f"🔊 GPIO buzzer ready on pin {pin}")

    def trigger(self):
        self.GPIO.output(self.pin, self.GPIO.HIGH)
        print("🚨 BUZZER ON")

    def stop(self):
        self.GPIO.output(self.pin, self.GPIO.LOW)

    def cleanup(self):
        self.GPIO.output(self.pin, self.GPIO.LOW)
        self.GPIO.cleanup()


class BuzzerSoftware:
    """Software fallback — prints + optional pygame beep."""
    def __init__(self):
        self._audio = False
        try:
            import pygame
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            sr = 22050
            dur = 0.3
            freq = 800
            n = int(dur * sr)
            buf = np.sin(2 * np.pi * freq * np.linspace(0, dur, n))
            buf = (buf * 32767).astype(np.int16)
            stereo = np.column_stack((buf, buf))
            self._beep = pygame.sndarray.make_sound(stereo)
            self._audio = True
            print("🔊 Software buzzer ready (pygame audio)")
        except Exception:
            print("🔊 Software buzzer ready (terminal only)")

    def trigger(self):
        print("🚨 *** ALERT — FALLING OBJECT DETECTED! ***")
        if self._audio:
            self._beep.play()

    def stop(self):
        pass

    def cleanup(self):
        if self._audio:
            try:
                import pygame
                if pygame.mixer.get_init():
                    pygame.mixer.quit()
            except Exception:
                pass


# ============================================================================
# NETWORK THREAD — only does socket I/O, NO OpenCV calls
# ============================================================================

class NetworkWorker:
    """Sends pre-encoded JPEG bytes to server, receives alerts.
    
    All OpenCV calls (imencode etc.) happen in the MAIN thread.
    This thread only touches sockets.
    """

    def __init__(self, server_host, server_port, buzzer):
        self.server_host = server_host
        self.server_port = server_port
        self.buzzer = buzzer

        self.alert_active = False
        self.alert_time = 0.0
        self.connected = False
        self.server_fps = 0
        self.running = True

        # Shared: main thread writes JPEG bytes, this thread reads them
        self._jpeg_data = None
        self._data_lock = threading.Lock()
        self._data_ready = threading.Event()

    def set_jpeg(self, jpeg_bytes):
        """Called by main thread with pre-encoded JPEG data."""
        with self._data_lock:
            self._jpeg_data = jpeg_bytes
        self._data_ready.set()

    def stop(self):
        self.running = False
        self._data_ready.set()

    def _connect(self):
        """Try connecting to server."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        try:
            sock.connect((self.server_host, self.server_port))
            sock.settimeout(2.0)
            return sock
        except (ConnectionRefusedError, socket.timeout, OSError):
            sock.close()
            return None

    def run(self):
        """Network loop — runs in background thread."""
        sock = None
        fps_time = time.time()
        fps_count = 0

        while self.running:
            # Connect
            if sock is None:
                self.connected = False
                print(f"🌐 Connecting to {self.server_host}:{self.server_port}...")
                sock = self._connect()
                if sock is None:
                    print("⚠️  Server unavailable, retrying in 2s...")
                    time.sleep(2)
                    continue
                self.connected = True
                print("✅ Connected!")

            # Wait for JPEG data from main thread
            self._data_ready.wait(timeout=1.0)
            if not self.running:
                break
            self._data_ready.clear()

            with self._data_lock:
                data = self._jpeg_data
            if data is None:
                continue

            # Send: [4-byte length][JPEG bytes]
            try:
                sock.sendall(struct.pack('>I', len(data)))
                sock.sendall(data)
            except (BrokenPipeError, ConnectionResetError, OSError):
                print("⚠️  Lost connection, reconnecting...")
                sock.close()
                sock = None
                continue

            # Receive 1-byte response
            try:
                resp = sock.recv(1)
                if not resp:
                    print("⚠️  Server disconnected, reconnecting...")
                    sock.close()
                    sock = None
                    continue
                if resp == RESPONSE_ALERT:
                    self.alert_active = True
                    self.alert_time = time.time()
                    self.buzzer.trigger()
            except socket.timeout:
                pass  # timeout = assume safe, keep going
            except (ConnectionResetError, OSError):
                print("⚠️  Server disconnected, reconnecting...")
                sock.close()
                sock = None
                continue

            # FPS
            fps_count += 1
            if time.time() - fps_time > 1.0:
                self.server_fps = fps_count
                fps_count = 0
                fps_time = time.time()

        if sock:
            sock.close()


# ============================================================================
# MAIN — camera + display + encoding, all on main thread
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Smart Helmet Phase 2 - Pi Client"
    )
    parser.add_argument("--server", required=True, help="Server address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--gpio", type=int, default=None,
                        help="GPIO pin for hardware buzzer (Pi only)")
    parser.add_argument("--cam", type=int, default=None,
                        help="Camera index override (default: auto-detect)")
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY)
    parser.add_argument("--no-show", action="store_true",
                        help="Disable camera preview (headless)")
    args = parser.parse_args()

    show_window = not args.no_show

    print("\n" + "=" * 60)
    print("📡 SMART HELMET PHASE 2 — CLIENT")
    print("=" * 60)

    # --- Buzzer ---
    if args.gpio is not None:
        try:
            buzzer = BuzzerGPIO(args.gpio)
        except (ImportError, Exception) as e:
            print(f"⚠️  GPIO unavailable ({e}), software fallback")
            buzzer = BuzzerSoftware()
    else:
        buzzer = BuzzerSoftware()

    # --- Camera (same search as standalone script) ---
    print("📷 Searching for camera...")
    cap = None
    if args.cam is not None:
        cap_test = cv2.VideoCapture(args.cam)
        if cap_test.isOpened():
            ret, _ = cap_test.read()
            if ret:
                print(f"✅ Camera found at index {args.cam}")
                cap = cap_test
            else:
                cap_test.release()
    else:
        for idx in range(5):
            cap_test = cv2.VideoCapture(idx)
            if cap_test.isOpened():
                ret, _ = cap_test.read()
                if ret:
                    print(f"✅ Camera found at index {idx}")
                    cap = cap_test
                    break
                cap_test.release()

    if cap is None:
        print("❌ No working camera found!")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    # --- Create window BEFORE main loop ---
    if show_window:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    # --- Start network thread ---
    net = NetworkWorker(args.server, args.port, buzzer)
    net_thread = threading.Thread(target=net.run, daemon=True)
    net_thread.start()

    # --- Main loop: camera + encode + display (ALL on main thread) ---
    fps_time = time.time()
    fps_counter = 0
    fps_display = 0

    print("=" * 60)
    print("🚀 Camera running! Streaming in background.")
    if show_window:
        print("💡 Press Q in the window to quit.")
    print()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera read failed")
                break

            # Encode JPEG on main thread (not in network thread!)
            ok, jpeg = cv2.imencode(
                '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality]
            )
            if ok:
                net.set_jpeg(jpeg.tobytes())

            # Check alert state
            is_alert = net.alert_active and \
                       (time.time() - net.alert_time) < ALERT_DURATION
            if not is_alert and net.alert_active:
                net.alert_active = False
                buzzer.stop()

            # FPS
            fps_counter += 1
            if time.time() - fps_time > 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_time = time.time()

            # --- Draw on frame and display ---
            if show_window:
                # Alert overlay
                if is_alert:
                    cv2.rectangle(frame, (0, 0),
                                  (frame.shape[1], frame.shape[0]),
                                  (0, 0, 255), 20)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0),
                                  (frame.shape[1], frame.shape[0]),
                                  (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                    text = "DANGER!"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale, thick = 3, 8
                    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
                    tx = (frame.shape[1] - tw) // 2
                    ty = (frame.shape[0] + th) // 2
                    cv2.putText(frame, text, (tx, ty), font, scale,
                                (0, 0, 0), thick + 4, cv2.LINE_AA)
                    cv2.putText(frame, text, (tx, ty), font, scale,
                                (255, 255, 255), thick, cv2.LINE_AA)

                # Status text
                conn_str = "CONNECTED" if net.connected else "CONNECTING..."
                y = 30
                for item in [f"Cam: {fps_display} FPS",
                             f"Server: {net.server_fps} FPS",
                             conn_str]:
                    cv2.putText(frame, item, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y += 25

                cv2.putText(frame, "Q: Quit", (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                time.sleep(0.03)  # don't spin

    except KeyboardInterrupt:
        print("\n👋 Stopping client...")
    finally:
        net.stop()
        net_thread.join(timeout=3)
        cap.release()
        if show_window:
            cv2.destroyAllWindows()
        buzzer.cleanup()
        print("✅ Client stopped")


if __name__ == "__main__":
    main()
