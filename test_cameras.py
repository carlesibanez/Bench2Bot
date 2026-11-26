"""
Test the global RGB camera and the wrist RealSense together (no robot needed).

Put this file next to dual_camera.py and run:
    python test_cameras.py --list                 # step 1: find the cameras
    python test_cameras.py --global-index 0       # step 2: live test
    python test_cameras.py --global-index 0 --duration 300   # 5-min soak test

Keys in the preview window:  q = quit,  s = save a snapshot pair
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from dual_camera import GlobalRGBCamera, WristRGBDCamera, RigManager


def list_cameras():
    print("=== RealSense devices ===")
    devs = rs.context().query_devices()
    if len(devs) == 0:
        print("  none found (check USB 3 port / cable)")
    for d in devs:
        print(f"  {d.get_info(rs.camera_info.name)}  "
              f"serial={d.get_info(rs.camera_info.serial_number)}  "
              f"usb={d.get_info(rs.camera_info.usb_type_descriptor)}")
        if not d.get_info(rs.camera_info.usb_type_descriptor).startswith("3"):
            print("  WARNING: RealSense is not on USB 3 -> depth will be limited")

    print("\n=== Stable Linux paths (/dev/v4l/by-id) ===")
    for p in sorted(glob.glob("/dev/v4l/by-id/*")):
        print("  ", p)

    print("\n=== OpenCV indices that open ===")
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, f = cap.read()
            shape = f.shape if ok else "no frame"
            print(f"  index {i}: {shape}")
        cap.release()
    print("\nNote: the RealSense also shows up as some OpenCV indices; "
          "pick the index whose resolution matches your global camera.")


class Stats:
    """Tracks FPS, camera time offset and stalls."""
    def __init__(self):
        self.t0 = time.monotonic()
        self.last_g = self.last_w = None
        self.n_g = self.n_w = 0
        self.dts = []
        self.max_gap_g = self.max_gap_w = 0.0
        self.sync_fail = 0

    def update(self, rig):
        g, w = rig.g.latest(), rig.w.latest()
        if g and g[0] != self.last_g:
            if self.last_g:
                self.max_gap_g = max(self.max_gap_g, g[0] - self.last_g)
            self.last_g = g[0]
            self.n_g += 1
        if w and w[0] != self.last_w:
            if self.last_w:
                self.max_gap_w = max(self.max_gap_w, w[0] - self.last_w)
            self.last_w = w[0]
            self.n_w += 1

    def report(self):
        el = time.monotonic() - self.t0
        dts = np.abs(self.dts) * 1000 if self.dts else np.array([0])
        return (f"[{el:6.1f}s] global {self.n_g/el:5.1f} fps | wrist {self.n_w/el:5.1f} fps | "
                f"cam offset mean {dts.mean():4.1f} ms, max {dts.max():5.1f} ms | "
                f"longest gap G {self.max_gap_g*1000:4.0f} ms, W {self.max_gap_w*1000:4.0f} ms | "
                f"sync fails {self.sync_fail}")


def depth_check(rig, depth):
    """Depth at image centre: 0 or NaN means no depth there."""
    h, w = depth.shape
    p = rig.w.pixel_to_cam_xyz(w // 2, h // 2, depth)
    valid = (depth > 0).mean() * 100
    return p, valid


def run_test(args):
    src = args.global_path if args.global_path else args.global_index
    rig = RigManager(
        GlobalRGBCamera(index=src, width=args.width, height=args.height),
        WristRGBDCamera(serial=args.serial),
        robot=None,
        max_dt=args.max_dt / 1000,
    ).start()

    os.makedirs("snapshots", exist_ok=True)
    stats = Stats()
    last_print = time.monotonic()

    try:
        while True:
            stats.update(rig)
            try:
                s = rig.snapshot(timeout=0.5)
            except TimeoutError:
                stats.sync_fail += 1
                print("  sync timeout: one camera stalled or offset > max_dt")
                continue
            stats.dts.append(s["dt_cams"])
            stats.dts = stats.dts[-1000:]

            p, valid = depth_check(rig, s["wrist_depth"])

            if time.monotonic() - last_print > 2.0:
                print(stats.report())
                centre = f"{p[2]:.3f} m" if p is not None else "no depth"
                print(f"          centre depth {centre}, valid depth pixels {valid:.0f}%")
                last_print = time.monotonic()

            if not args.no_view:
                g = cv2.resize(s["global_rgb"], (640, 360))
                wr = s["wrist_rgb"]
                dv = cv2.applyColorMap(
                    cv2.convertScaleAbs(s["wrist_depth"], alpha=255 / args.max_depth),
                    cv2.COLORMAP_JET)
                h, w = wr.shape[:2]
                cv2.drawMarker(wr, (w // 2, h // 2), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(g, f"dt {s['dt_cams']*1000:+.1f} ms", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("global", g)
                cv2.imshow("wrist RGB | depth", np.hstack([wr, dv]))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"snapshots/{ts}_global.png", s["global_rgb"])
                    cv2.imwrite(f"snapshots/{ts}_wrist.png", s["wrist_rgb"])
                    np.save(f"snapshots/{ts}_depth.npy", s["wrist_depth"])
                    print(f"  saved snapshots/{ts}_*")

            if args.duration and time.monotonic() - stats.t0 > args.duration:
                break
    finally:
        rig.stop()
        cv2.destroyAllWindows()
        print("\n=== FINAL ===")
        print(stats.report())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list cameras and exit")
    ap.add_argument("--global-index", type=int, default=0)
    ap.add_argument("--global-path", type=str, default=None,
                    help="e.g. /dev/v4l/by-id/usb-...-video-index0 (overrides index)")
    ap.add_argument("--serial", type=str, default=None, help="RealSense serial")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--max-dt", type=float, default=40, help="sync tolerance, ms")
    ap.add_argument("--max-depth", type=float, default=1.0, help="depth colormap range, m")
    ap.add_argument("--duration", type=float, default=0, help="seconds, 0 = until q")
    ap.add_argument("--no-view", action="store_true", help="headless (no windows)")
    args = ap.parse_args()

    if args.list:
        list_cameras()
    else:
        run_test(args)
