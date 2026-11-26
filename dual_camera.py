"""
Dual-camera capture for a UR arm VLA pipeline.

  - Global RGB camera  : any UVC/USB camera via OpenCV
  - Wrist RGB-D camera : Intel RealSense (D405/D435/D455) via pyrealsense2
  - Robot pose         : ur_rtde, sampled continuously so each wrist frame
                         gets the TCP pose from the moment it was captured

Each device runs in its own thread and pushes (host_timestamp, data) into a
small ring buffer. The main loop picks the closest-in-time set of samples,
so the slow model (e.g. Grounding DINO) never blocks capture.

pip install opencv-python pyrealsense2 numpy scipy ur_rtde
"""

import threading
import time
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R


# ----------------------------------------------------------------------------
# Generic threaded buffer
# ----------------------------------------------------------------------------
class ThreadedSource:
    def __init__(self, maxlen=30):
        self.buf = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self._close()

    def _push(self, t, data):
        with self.lock:
            self.buf.append((t, data))

    def latest(self):
        with self.lock:
            return self.buf[-1] if self.buf else None

    def nearest(self, t):
        with self.lock:
            if not self.buf:
                return None
            return min(self.buf, key=lambda item: abs(item[0] - t))

    def _loop(self):
        raise NotImplementedError

    def _close(self):
        pass


# ----------------------------------------------------------------------------
# Global RGB camera (OpenCV)
# ----------------------------------------------------------------------------
class GlobalRGBCamera(ThreadedSource):
    def __init__(self, index=0, width=1280, height=720, fps=30):
        super().__init__()
        backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # avoid stale frames
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open global camera {index}")

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                self._push(time.monotonic(), frame)

    def _close(self):
        self.cap.release()


# ----------------------------------------------------------------------------
# Wrist RGB-D camera (RealSense), depth aligned to color
# ----------------------------------------------------------------------------
class WristRGBDCamera(ThreadedSource):
    """
    Intel RealSense D405 (also works for other D4xx).

    D405 notes:
      - Color comes from the same global-shutter sensor as depth, so color and
        depth are already pixel-matched; align is kept for safety (cheap).
      - Best range ~7-50 cm. No IR projector: needs light and surface texture.
      - Depth unit is 0.1 mm (read from the device, not hard-coded).
    """

    def __init__(self, serial=None, width=848, height=480, fps=30,
                 use_filters=True, min_depth=0.07, max_depth=0.5):
        super().__init__()
        self.min_depth, self.max_depth = min_depth, max_depth
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(cfg)

        dev = profile.get_device()
        name = dev.get_info(rs.camera_info.name)
        usb = dev.get_info(rs.camera_info.usb_type_descriptor)
        print(f"[wrist] {name} on USB {usb}")
        if not usb.startswith("3"):
            print("[wrist] WARNING: running at USB 2 -> use 640x480 @ 15 fps or fix cable")

        self.align = rs.align(rs.stream.color)
        self.depth_scale = dev.first_depth_sensor().get_depth_scale()
        self.intrinsics = (profile.get_stream(rs.stream.color)
                           .as_video_stream_profile().get_intrinsics())

        # Spatial + hole filling clean up stereo noise without lag.
        # (No temporal filter: it smears depth while the arm is moving.)
        self.filters = []
        if use_filters:
            self.filters = [
                rs.disparity_transform(True),
                rs.spatial_filter(),
                rs.disparity_transform(False),
                rs.hole_filling_filter(1),
            ]

        # let auto-exposure settle
        for _ in range(30):
            self.pipeline.wait_for_frames()

    def _loop(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue
            t = time.monotonic()
            frames = self.align.process(frames)
            c, d = frames.get_color_frame(), frames.get_depth_frame()
            if not c or not d:
                continue
            for f in self.filters:
                d = f.process(d)
            color = np.asanyarray(c.get_data()).copy()
            depth_m = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale
            # drop readings outside the D405's useful range
            depth_m[(depth_m < self.min_depth) | (depth_m > self.max_depth)] = 0
            self._push(t, (color, depth_m))

    def pixel_to_cam_xyz(self, u, v, depth_m, patch=3):
        """3D point (m) in the camera frame; median over a small patch to skip holes."""
        h, w = depth_m.shape
        u0, u1 = max(0, u - patch), min(w, u + patch + 1)
        v0, v1 = max(0, v - patch), min(h, v + patch + 1)
        vals = depth_m[v0:v1, u0:u1]
        vals = vals[vals > 0]
        if vals.size == 0:
            return None
        z = float(np.median(vals))
        return np.array(rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], z))

    def _close(self):
        self.pipeline.stop()


# ----------------------------------------------------------------------------
# UR TCP pose stream (ur_rtde), so pose matches the frame timestamp
# ----------------------------------------------------------------------------
class URPoseStream(ThreadedSource):
    def __init__(self, robot_ip, hz=125):
        super().__init__(maxlen=500)
        from rtde_receive import RTDEReceiveInterface
        self.rtde_r = RTDEReceiveInterface(robot_ip)
        self.period = 1.0 / hz

    def _loop(self):
        while self.running:
            self._push(time.monotonic(), np.array(self.rtde_r.getActualTCPPose()))
            time.sleep(self.period)

    def _close(self):
        self.rtde_r.disconnect()


def ur_pose_to_matrix(pose):
    """UR [x, y, z, rx, ry, rz] (rotation vector) -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


# ----------------------------------------------------------------------------
# Manager: one call returns a time-matched snapshot of everything
# ----------------------------------------------------------------------------
class RigManager:
    def __init__(self, global_cam, wrist_cam, robot=None, max_dt=0.040):
        self.g, self.w, self.r = global_cam, wrist_cam, robot
        self.max_dt = max_dt  # 40 ms tolerance between cameras

    def start(self):
        for s in (self.g, self.w, self.r):
            if s:
                s.start()
        time.sleep(1.0)
        return self

    def stop(self):
        for s in (self.g, self.w, self.r):
            if s:
                s.stop()

    def snapshot(self, timeout=1.0):
        """Anchor on the newest wrist frame, match the nearest global frame and pose."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            wl = self.w.latest()
            if wl is None:
                time.sleep(0.005)
                continue
            t_w, (color, depth) = wl
            gn = self.g.nearest(t_w)
            if gn is None or abs(gn[0] - t_w) > self.max_dt:
                time.sleep(0.005)
                continue
            snap = {
                "t": t_w,
                "dt_cams": gn[0] - t_w,
                "global_rgb": gn[1],
                "wrist_rgb": color,
                "wrist_depth": depth,
            }
            if self.r:
                pn = self.r.nearest(t_w)
                snap["tcp_pose"] = pn[1] if pn else None
            return snap
        raise TimeoutError("No synchronized frames within timeout")


# ----------------------------------------------------------------------------
# Plug your Grounding DINO / one-shot detector here
# ----------------------------------------------------------------------------
def detect(image_bgr, prompt):
    """
    Return a list of (x1, y1, x2, y2, score, label) in pixel coords.
    Replace with your model call, e.g. Grounding DINO via HuggingFace
    (AutoModelForZeroShotObjectDetection, "IDEA-Research/grounding-dino-tiny").
    """
    return []


# ----------------------------------------------------------------------------
# Example loop
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ROBOT_IP = "192.168.1.10"
    PROMPT = "red cube ."

    # Hand-eye calibration result (camera pose in TCP frame), from
    # cv2.calibrateHandEye. Identity is only a placeholder.
    T_tcp_cam = np.eye(4)

    rig = RigManager(
        GlobalRGBCamera(index=0),
        WristRGBDCamera(serial=None),
        URPoseStream(ROBOT_IP),
    ).start()

    try:
        while True:
            s = rig.snapshot()

            # 1) Coarse: find the target in the global view (scene-level reasoning)
            g_dets = detect(s["global_rgb"], PROMPT)

            # 2) Fine: localize in the wrist view and lift to 3D
            w_dets = detect(s["wrist_rgb"], PROMPT)
            for x1, y1, x2, y2, score, label in w_dets:
                u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
                p_cam = rig.w.pixel_to_cam_xyz(u, v, s["wrist_depth"])
                if p_cam is None or s.get("tcp_pose") is None:
                    continue
                T_base_tcp = ur_pose_to_matrix(s["tcp_pose"])
                p_base = (T_base_tcp @ T_tcp_cam @ np.append(p_cam, 1.0))[:3]
                print(f"{label} {score:.2f} -> base xyz {np.round(p_base, 4)}")

            # Visualize
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(s["wrist_depth"], alpha=255 / 1.0), cv2.COLORMAP_JET)
            cv2.imshow("global", cv2.resize(s["global_rgb"], (640, 360)))
            cv2.imshow("wrist", np.hstack([s["wrist_rgb"], depth_vis]))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        rig.stop()
        cv2.destroyAllWindows()
