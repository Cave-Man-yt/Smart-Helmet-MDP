"""
Smart Helmet — Pi Client (also works on laptop for testing)
Captures camera frames, sends them to the detection server,
and triggers the buzzer/speaker when an alert is received.

Usage (laptop testing — no hardware needed):
    python client_pi.py --server localhost
    python client_pi.py --server 192.168.1.100 --port 5555

Usage (on Raspberry Pi):
    python client_pi.py --server 192.168.1.100 --gpio 17

Options:
    --server HOST    Detection server address (required)
    --port PORT      Detection server port (default: 5555)
    --gpio PIN       GPIO pin for buzzer (enables hardware buzzer)
    --cam INDEX      Camera index override (default: auto-detect)
    --quality Q      JPEG quality 1-100 (default: 60, lower = faster)
    --no-show        Disable camera preview (headless mode for Pi)
    --test           Quick connectivity test then exit
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
DEFAULT_QUALITY = 60
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
ALERT_DURATION = 2.0

# Protocol constants (must match server.py)
RESPONSE_ALERT = b'\x01'

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
    """Software fallback — uses pygame beep or just prints."""
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
# NETWORK HELPERS
# ============================================================================

def connect_to_server(host, port, timeout=5):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        sock.settimeout(2.0)
        return sock
    except (ConnectionRefusedError, socket.timeout, OSError):
        sock.close()
        return None


def send_frame(sock, frame, quality):
    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ret:
        return False
    data = jpeg.tobytes()
    try:
        sock.sendall(struct.pack('>I', len(data)))
        sock.sendall(data)
        return True
    except (BrokenPipeError, ConnectionResetError, OSError):
        return False


def recv_response(sock):
    try:
        data = sock.recv(1)
        if not data:
            return None
        return data == RESPONSE_ALERT
    except socket.timeout:
        return False
    except (ConnectionResetError, OSError):
        return None

# ============================================================================
# MAIN CLIENT CLASS — follows same pattern as SmartHelmetPhase1
# ============================================================================

class SmartHelmetClient:
    def __init__(self, server_host, server_port, gpio_pin, cam_override,
                 quality, show_window):
        print("📡 Initializing Smart Helmet Client")
        print("=" * 70)

        self.server_host = server_host
        self.server_port = server_port
        self.quality = quality
        self.show_window = show_window

        # --- Set up buzzer ---
        if gpio_pin is not None:
            try:
                self.buzzer = BuzzerGPIO(gpio_pin)
            except (ImportError, Exception) as e:
                print(f"⚠️  GPIO not available ({e}), using software fallback")
                self.buzzer = BuzzerSoftware()
        else:
            self.buzzer = BuzzerSoftware()

        # --- Initialize camera — SAME method as fall_detection_phase1 ---
        print("📷 Searching for camera...")
        self.cap = None

        if cam_override is not None:
            # User specified a camera index
            cap_test = cv2.VideoCapture(cam_override)
            if cap_test.isOpened():
                ret, _ = cap_test.read()
                if ret:
                    print(f"✅ Camera found at index {cam_override}")
                    self.cap = cap_test
                else:
                    cap_test.release()
        else:
            # Auto-detect: try indices 0-4 (same as standalone script)
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

        # --- Network state (updated by background thread) ---
        self._alert = False
        self._alert_time = 0.0
        self._connected = False
        self._server_fps = 0
        self._net_lock = threading.Lock()
        self._latest_jpeg = None  # pre-encoded JPEG bytes (encoded on main thread)
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()
        self._running = True

        print("=" * 70)
        print("🚀 Client Ready!")
        print(f"🌐 Server: {server_host}:{server_port}")
        print(f"📷 Preview: {'ON' if show_window else 'OFF'}")
        print("=" * 70)
        print("💡 Controls:")
        print("   Q - Quit")
        print()

    def _network_loop(self):
        """Background thread: send pre-encoded JPEG bytes, receive alerts.
        NO OpenCV calls here — all encoding happens on main thread."""
        sock = None
        fps_time = time.time()
        fps_count = 0

        while self._running:
            # Connect
            if sock is None:
                with self._net_lock:
                    self._connected = False
                print(f"🌐 Connecting to {self.server_host}:{self.server_port}...")
                sock = connect_to_server(self.server_host, self.server_port)
                if sock is None:
                    print("⚠️  Server unavailable, retrying in 2s...")
                    time.sleep(2)
                    continue
                with self._net_lock:
                    self._connected = True
                print("✅ Connected to detection server!")

            # Wait for pre-encoded JPEG from main thread
            self._frame_event.wait(timeout=1.0)
            if not self._running:
                break
            self._frame_event.clear()

            with self._frame_lock:
                jpeg_data = self._latest_jpeg
            if jpeg_data is None:
                continue

            # Send: [4-byte length][JPEG bytes]
            try:
                sock.sendall(struct.pack('>I', len(jpeg_data)))
                sock.sendall(jpeg_data)
            except (BrokenPipeError, ConnectionResetError, OSError):
                print("⚠️  Lost connection, reconnecting...")
                sock.close()
                sock = None
                continue

            # Receive
            alert = recv_response(sock)
            if alert is None:
                print("⚠️  Server disconnected, reconnecting...")
                sock.close()
                sock = None
                continue

            if alert:
                with self._net_lock:
                    self._alert = True
                    self._alert_time = time.time()
                self.buzzer.trigger()

            fps_count += 1
            if time.time() - fps_time > 1.0:
                with self._net_lock:
                    self._server_fps = fps_count
                fps_count = 0
                fps_time = time.time()

        if sock:
            sock.close()

    def run(self):
        """Main loop — camera + encode + display. ALL OpenCV on main thread."""

        # Create window before main loop
        if self.show_window:
            cv2.namedWindow('Smart Helmet - Client', cv2.WINDOW_AUTOSIZE)

        # Start network in background
        net_thread = threading.Thread(target=self._network_loop, daemon=True)
        net_thread.start()

        fps_time = time.time()
        fps_counter = 0
        fps_display = 0

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break

                # Encode JPEG on main thread, give bytes to network thread
                ok, jpeg = cv2.imencode(
                    '.jpg', frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                )
                if ok:
                    with self._frame_lock:
                        self._latest_jpeg = jpeg.tobytes()
                    self._frame_event.set()

                # Read alert state
                with self._net_lock:
                    alert_active = self._alert and (time.time() - self._alert_time) < ALERT_DURATION
                    connected = self._connected
                    server_fps = self._server_fps
                    if not alert_active:
                        self._alert = False
                        self.buzzer.stop()

                # --- Render alert overlay (same style as standalone) ---
                if alert_active:
                    # Red border
                    cv2.rectangle(frame, (0, 0),
                                 (FRAME_WIDTH, FRAME_HEIGHT),
                                 (0, 0, 255), 20)
                    # Red tint
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0),
                                 (FRAME_WIDTH, FRAME_HEIGHT),
                                 (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                    # DANGER text
                    text = "DANGER!"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 3
                    thickness = 8
                    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    tx = (FRAME_WIDTH - tw) // 2
                    ty = (FRAME_HEIGHT + th) // 2
                    cv2.putText(frame, text, (tx, ty), font, font_scale,
                                (0, 0, 0), thickness + 4, cv2.LINE_AA)
                    cv2.putText(frame, text, (tx, ty), font, font_scale,
                                (255, 255, 255), thickness, cv2.LINE_AA)

                # --- Calculate FPS ---
                fps_counter += 1
                if time.time() - fps_time > 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    fps_time = time.time()

                # --- Status overlay ---
                y_offset = 30
                conn_str = "CONNECTED" if connected else "CONNECTING..."
                status_items = [
                    f"Cam FPS: {fps_display}",
                    f"Server FPS: {server_fps}",
                    f"Server: {conn_str}",
                ]
                for item in status_items:
                    cv2.putText(frame, item, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25

                # Bottom instructions
                cv2.putText(frame, "Q: Quit", (10, FRAME_HEIGHT - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # --- Display frame ---
                cv2.imshow('Smart Helmet - Client', frame)

                # --- Handle keyboard input ---
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\n👋 Shutting down...")
                    break

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")

        finally:
            self._running = False
            self._frame_event.set()
            net_thread.join(timeout=3)
            self.cap.release()
            cv2.destroyAllWindows()
            self.buzzer.cleanup()
            print("✅ Client stopped")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smart Helmet Pi Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  Laptop testing (no hardware):
    python client_pi.py --server localhost

  On Raspberry Pi (headless):
    python client_pi.py --server 192.168.1.100 --gpio 17 --no-show

  Quick connectivity test:
    python client_pi.py --server localhost --test
"""
    )
    parser.add_argument("--server", required=True, help="Detection server address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--gpio", type=int, default=None,
                        help="GPIO pin for hardware buzzer (Pi only)")
    parser.add_argument("--cam", type=int, default=None,
                        help="Camera index override (default: auto-detect)")
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY)
    parser.add_argument("--no-show", action="store_true",
                        help="Disable camera preview (headless)")
    parser.add_argument("--test", action="store_true",
                        help="Quick connectivity test then exit")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("📡 SMART HELMET — PI CLIENT")
    print("=" * 70)

    if args.test:
        # Quick connectivity test
        print("🔍 Running connectivity test...")
        sock = connect_to_server(args.server, args.port)
        if sock is None:
            print("❌ Could not connect. Is server.py running?")
            sys.exit(1)
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            ok = send_frame(sock, frame, args.quality)
            if ok:
                alert = recv_response(sock)
                print(f"✅ Test passed! Server responded (alert={alert})")
            else:
                print("❌ Failed to send frame")
        else:
            print("❌ Camera not available")
        cap.release()
        sock.close()
        sys.exit(0)

    try:
        client = SmartHelmetClient(
            server_host=args.server,
            server_port=args.port,
            gpio_pin=args.gpio,
            cam_override=args.cam,
            quality=args.quality,
            show_window=not args.no_show,
        )
        client.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
