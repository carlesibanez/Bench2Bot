import serial
import time
import cv2
from pathlib import Path
from datetime import datetime


# ============================================================
# SETTINGS
# ============================================================

IKA_PORT = "/dev/ttyACM0"
IKA_BAUDRATE = 9600

TARGET_TEMPERATURE = 120.0

# Camera
CAMERA_DEVICE = "/dev/video0"
CAMERA_FPS = 10
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Output
RUN_NAME = datetime.now().strftime("wax_run_%Y%m%d_%H%M%S")

OUTPUT_DIR = (
    Path.home()
    / "Outreach"
    / "tpu_robot_training"
    / "runs"
    / RUN_NAME
)

VIDEO_PATH = OUTPUT_DIR / "camera_recording.mp4"
TEMP_LOG_PATH = OUTPUT_DIR / "temperature_log.csv"


# ============================================================
# IKA CONTROLLER
# ============================================================

class IKARCT:

    def __init__(self, port, baudrate=9600):

        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.SEVENBITS,
            parity=serial.PARITY_EVEN,
            stopbits=serial.STOPBITS_ONE,
            timeout=2
        )

        time.sleep(0.5)


    def send_command(self, command):

        message = command + " \r\n"

        self.ser.write(
            message.encode("ascii")
        )

        time.sleep(0.1)

        response = self.ser.readline().decode(
            "ascii",
            errors="ignore"
        ).strip()

        return response


    # ========================================================
    # TEMPERATURE
    # ========================================================

    def set_temperature(self, temperature):

        return self.send_command(
            f"OUT_SP_1 {temperature}"
        )


    def get_temperature(self):

        response = self.send_command(
            "IN_PV_2"
        )

        try:
            return float(response.split()[0])

        except (ValueError, IndexError):
            return None


    # ========================================================
    # HEATING
    # ========================================================

    def start_heating(self):

        return self.send_command(
            "START_1"
        )


    def stop_heating(self):

        return self.send_command(
            "STOP_1"
        )


    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        if self.ser.is_open:
            self.ser.close()


# ============================================================
# CAMERA
# ============================================================

def start_camera():

    camera = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS
    )

    if not camera.isOpened():

        raise RuntimeError(
            f"Could not open camera: {CAMERA_DEVICE}"
        )

    return camera


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ika = None
    camera = None
    video_writer = None

    experiment_start = time.time()

    try:

        # ====================================================
        # CONNECT TO IKA
        # ====================================================

        print("Connecting to IKA...")

        ika = IKARCT(
            IKA_PORT,
            IKA_BAUDRATE
        )

        print("IKA connected.")


        # ====================================================
        # CAMERA
        # ====================================================

        print("Starting camera...")

        camera = start_camera()

        actual_width = int(
            camera.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        actual_height = int(
            camera.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        actual_fps = camera.get(
            cv2.CAP_PROP_FPS
        )

        if actual_fps <= 0:
            actual_fps = CAMERA_FPS


        # ====================================================
        # VIDEO WRITER
        # ====================================================

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        video_writer = cv2.VideoWriter(
            str(VIDEO_PATH),
            fourcc,
            actual_fps,
            (
                actual_width,
                actual_height
            )
        )


        # ====================================================
        # TEMPERATURE
        # ====================================================

        print(
            f"Setting target temperature: "
            f"{TARGET_TEMPERATURE} °C"
        )

        ika.set_temperature(
            TARGET_TEMPERATURE
        )


        # ====================================================
        # START HEATING ONLY
        # ====================================================

        print("Starting heating...")

        response = ika.start_heating()

        print(
            f"Heating response: {response}"
        )


        # ====================================================
        # START MESSAGE
        # ====================================================

        print()
        print("===================================")
        print("PARAFFIN WAX EXPERIMENT")
        print("===================================")
        print(
            f"Target temperature: "
            f"{TARGET_TEMPERATURE} °C"
        )
        print("Stirring: OFF")
        print("Camera recording: ON")
        print()
        print("Press Q when you want to stop.")
        print("===================================")


        # ====================================================
        # TEMPERATURE LOG
        # ====================================================

        with open(
            TEMP_LOG_PATH,
            "w"
        ) as log_file:

            log_file.write(
                "timestamp,"
                "elapsed_seconds,"
                "temperature_C\n"
            )


            # =================================================
            # EXPERIMENT LOOP
            # =================================================

            while True:

                # --------------------------------------------
                # CAMERA
                # --------------------------------------------

                ret, frame = camera.read()

                if not ret:

                    print(
                        "\nWARNING: Camera frame failed."
                    )

                    continue


                video_writer.write(frame)


                # --------------------------------------------
                # TEMPERATURE
                # --------------------------------------------

                temperature = (
                    ika.get_temperature()
                )


                # --------------------------------------------
                # TIME
                # --------------------------------------------

                elapsed = (
                    time.time()
                    - experiment_start
                )

                timestamp = (
                    datetime.now().isoformat()
                )


                # --------------------------------------------
                # LOG
                # --------------------------------------------

                log_file.write(
                    f"{timestamp},"
                    f"{elapsed:.2f},"
                    f"{temperature}\n"
                )

                log_file.flush()


                # --------------------------------------------
                # TERMINAL STATUS
                # --------------------------------------------

                print(
                    f"\r"
                    f"Time: {elapsed:7.1f}s | "
                    f"Temp: {temperature} °C | "
                    f"Stirring: OFF",
                    end=""
                )


                # --------------------------------------------
                # CAMERA WINDOW
                # --------------------------------------------

                cv2.imshow(
                    "Paraffin Wax Experiment",
                    frame
                )


                # --------------------------------------------
                # STOP WITH Q
                # --------------------------------------------

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):

                    print()
                    print(
                        "Stop requested."
                    )

                    break


    # ========================================================
    # CTRL+C
    # ========================================================

    except KeyboardInterrupt:

        print()
        print(
            "Keyboard interrupt received."
        )


    # ========================================================
    # ERROR
    # ========================================================

    except Exception as error:

        print()
        print(
            f"ERROR: {error}"
        )


    # ========================================================
    # SAFE SHUTDOWN
    # ========================================================

    finally:

        print()
        print(
            "Stopping experiment safely..."
        )


        if ika is not None:

            try:

                ika.stop_heating()

                print(
                    "Heating OFF."
                )

            except Exception:
                pass


            try:

                ika.close()

            except Exception:
                pass


        if video_writer is not None:

            video_writer.release()


        if camera is not None:

            camera.release()


        cv2.destroyAllWindows()


        print()
        print("===================================")
        print("WAX RUN FINISHED")
        print("===================================")
        print(
            f"Video: {VIDEO_PATH}"
        )
        print(
            f"Temperature log: {TEMP_LOG_PATH}"
        )
        print("===================================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()