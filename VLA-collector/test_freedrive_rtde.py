"""Verify programmatic UR5e freedrive / teach mode via ur_rtde in Remote Control mode.

Prerequisites:
- UR5e teach pendant must be switched to **Remote Control** mode (and no other
  PolyScope program or external RTDE control client blocking port 30004).
- Install ``ur-rtde`` (Python 3.10 or 3.11 is recommended for prebuilt Windows wheels):
      pip install ur-rtde

Usage:
    python test_freedrive_rtde.py --ip 192.168.0.2
    python test_freedrive_rtde.py --ip 192.168.0.2 --mode freedrive
    python test_freedrive_rtde.py --ip 192.168.0.2 --mode teach
"""

import argparse
import sys
import threading
import time


def _format_vec(values, precision=4):
    return "[" + ", ".join(f"{v:+.{precision}f}" for v in values) + "]"


def _stream_state_until_enter(rtde_r, prompt_message):
    """Print live TCP pose and joint positions at ~5 Hz until the user presses Enter."""
    stop_event = threading.Event()

    def _printer():
        while not stop_event.is_set():
            try:
                tcp = rtde_r.getActualTCPPose()
                joints = rtde_r.getActualQ()
                line = (
                    f"\rTCP: {_format_vec(tcp, 4)} | "
                    f"Joints: {_format_vec(joints, 3)}"
                )
                sys.stdout.write(line)
                sys.stdout.flush()
            except Exception as exc:
                sys.stdout.write(f"\n[Stream warning] {exc}\n")
                sys.stdout.flush()
                break
            stop_event.wait(0.2)

    print(prompt_message)
    thread = threading.Thread(target=_printer, daemon=True)
    thread.start()
    try:
        input()
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
        print()


def _test_freedrive_mode(rtde_c, rtde_r):
    """Test orientation-locked XYZ translation freedriveMode()."""
    print("\n--- Test 1: Orientation-locked translation freedrive (freedriveMode) ---")
    input(
        "Keep one hand near the arm / E-stop. "
        "Press Enter to ENABLE orientation-locked freedrive [1,1,1,0,0,0]..."
    )

    start_tcp = rtde_r.getActualTCPPose()
    ok = rtde_c.freedriveMode([1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0])
    if not ok:
        print("freedriveMode() returned False.")
        return False

    try:
        _stream_state_until_enter(
            rtde_r,
            "Freedrive ACTIVE (translation only). Try moving the arm, then press Enter to STOP:",
        )
    finally:
        ended = rtde_c.endFreedriveMode()
        print(f"endFreedriveMode() -> {ended}")

    end_tcp = rtde_r.getActualTCPPose()
    rot_deltas = [abs(e - s) for e, s in zip(end_tcp[3:], start_tcp[3:])]
    max_rot_delta = max(rot_deltas) if rot_deltas else 0.0
    print(f"rtde_c.isConnected(): {rtde_c.isConnected()}")
    print(
        f"Max orientation delta [rx, ry, rz] = {max_rot_delta:.4f} rad "
        f"({'orientation stayed locked' if max_rot_delta < 0.05 else 'orientation changed — verify axis lock'})"
    )
    return True


def _test_teach_mode(rtde_c, rtde_r):
    """Test full 6-DoF teachMode()."""
    print("\n--- Test 2: Full 6-DoF freedrive (teachMode) ---")
    input(
        "Keep one hand near the arm / E-stop. "
        "Press Enter to ENABLE full 6-DoF teachMode()..."
    )

    ok = rtde_c.teachMode()
    if not ok:
        print("teachMode() returned False.")
        return False

    try:
        _stream_state_until_enter(
            rtde_r,
            "teachMode ACTIVE (6-DoF). Try moving the arm, then press Enter to STOP:",
        )
    finally:
        ended = rtde_c.endTeachMode()
        print(f"endTeachMode() -> {ended}")

    print(f"rtde_c.isConnected(): {rtde_c.isConnected()}")
    return True


def _safe_cleanup(rtde_c, rtde_r):
    """Ensure freedrive/teach modes are stopped and RTDE connections are closed."""
    if rtde_c is not None:
        for method_name in ("endFreedriveMode", "endTeachMode"):
            method = getattr(rtde_c, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
        try:
            rtde_c.disconnect()
        except Exception:
            pass

    if rtde_r is not None:
        try:
            rtde_r.disconnect()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test UR5e freedrive / teach mode via ur_rtde in Remote Control mode."
    )
    parser.add_argument(
        "--ip",
        default="192.168.0.2",
        help="IP address of the UR5e controller (default: 192.168.0.2).",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "freedrive", "teach"],
        default="auto",
        help=(
            "Which freedrive API to test: 'freedrive' (orientation-locked), "
            "'teach' (6-DoF teachMode), or 'auto' (try freedriveMode, fallback/offer teachMode)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import rtde_control
        import rtde_receive
    except ImportError as exc:
        print(
            f"Error: ur_rtde is not installed ({exc}).\n"
            "Install it in your robot-control environment (Python 3.10/3.11 recommended on Windows):\n"
            "    pip install ur-rtde",
            file=sys.stderr,
        )
        return 1

    rtde_r = None
    rtde_c = None

    try:
        print(f"Connecting RTDEReceiveInterface to {args.ip}...")
        rtde_r = rtde_receive.RTDEReceiveInterface(args.ip)
        print(f"Robot mode: {rtde_r.getRobotMode()}")
        print(f"Current joints (rad): {_format_vec(rtde_r.getActualQ(), 4)}")
        print(f"Current TCP (m, rad): {_format_vec(rtde_r.getActualTCPPose(), 4)}")

        print(f"\nConnecting RTDEControlInterface to {args.ip}...")
        try:
            rtde_c = rtde_control.RTDEControlInterface(args.ip)
        except Exception as exc:
            print(
                f"\nFailed to connect RTDEControlInterface: {exc}\n"
                "Troubleshooting:\n"
                "  1. Verify the UR5e teach pendant is in **Remote Control** mode.\n"
                "  2. Make sure no PolyScope program or other RTDE control client is running.\n"
                "  3. Confirm network reachability to the robot IP.",
                file=sys.stderr,
            )
            return 1

        if args.mode in ("auto", "freedrive"):
            freedrive_ok = False
            try:
                freedrive_ok = _test_freedrive_mode(rtde_c, rtde_r)
            except (AttributeError, Exception) as exc:
                print(f"freedriveMode() unavailable or failed: {exc}")

            if args.mode == "auto":
                if not freedrive_ok:
                    print("Falling back to teachMode()...")
                    _test_teach_mode(rtde_c, rtde_r)
                else:
                    choice = input("\nAlso test full 6-DoF teachMode()? [y/N]: ").strip().lower()
                    if choice in ("y", "yes"):
                        _test_teach_mode(rtde_c, rtde_r)
        else:
            _test_teach_mode(rtde_c, rtde_r)

        print("\nDone. Freedrive exited cleanly.")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user — exiting freedrive and disconnecting safely.")
        return 130
    finally:
        _safe_cleanup(rtde_c, rtde_r)


if __name__ == "__main__":
    raise SystemExit(main())
