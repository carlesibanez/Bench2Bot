import os
import sys
import time
import math
import threading
from time import sleep

from utils.util import degree_to_radians
try:
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

try:
    from utils.UR_Functions import URfunctions as URControl
    HAS_UR = True
except ImportError:
    HAS_UR = False
    URControl = None

try:
    from robotiq.robotiq_gripper import RobotiqGripper
    HAS_GRIPPER = True
except ImportError:
    HAS_GRIPPER = False
    RobotiqGripper = None

try:
    from PyLabware.devices.ika_rct_digital import RCTDigitalHotplate
    HAS_HOTPLATE = True
except ImportError:
    HAS_HOTPLATE = False
    RCTDigitalHotplate = None

 

PIPETTE_POSITIONS = {
    "Open": 0,
    "Pick": 190,
    "Squeeze": 235
}
ROBOT_POSITIONS_= {
    "Front_Home_Position": degree_to_radians([72.5, -108, 122, -105, -87, 5]), 
    "Pipette_1_without": degree_to_radians([75.3, -65.3, 74.8, -98.9, -90, -2.2]),
    "Pipette_1": degree_to_radians([75.3, -63.23, 81.42, -107.65, -90.58, -2.2]),
    "Pipette_1_pick": degree_to_radians([75.3, -66.17, 67.03, -90.32, -90.54, -2.28]),
    "Vial_1_above": degree_to_radians([80.12, -121.68, 120.34, -88.05, -90.49, 2.42]),
    "Vial_1_inside": degree_to_radians([80.11, -119.01, 135.96, -106.34, -90.53, 2.49]),
    "Vial_2_above": degree_to_radians([95.65, -86.9, 82.64, -84.91, -90.0, 18.0]),
    "Vial_2_inside": degree_to_radians([95.6, -87.0, 88.7, -90.9, -90.3, 18.03]),
    
    
    # "Front_Home_Position": [0.9858925938606262, -1.7097555599608363, 2.087477986012594, -0.3821602624705811, 0.917646586894989, 0.006281518843024969],
    "Left_Home_Position": [0.9858925938606262, -1.7097555599608363, 2.087477986012594, -0.3821602624705811, 0.917646586894989, 0.006281518843024969],

    "Pipette_Holder_P3": [1.0045684576034546, -1.1894803804210206, 2.015272919331686, -0.8287990850261231, 0.9036718606948853, 0.003900587558746338],
    "Above_Pipette_Holder_P3": [1.0050357580184937, -1.4436511558345337, 1.7828071753131312, -0.3437257570079346, 0.9020593166351318, 0.006543457508087158],

    "Pipette_Holder_P2": [0.9176647663116455, -1.1644064050963898, 1.9716804663287562, -0.8108180326274415, 0.816744863986969, 0.004359245300292969],
    "Above_Pipette_Holder_P2": [0.9181051254272461, -1.4039693635753174, 1.7547457853900355, -0.356087402706482, 0.8152208924293518, 0.0070192813873291016],

    "Pipette_Holder_P1": [0.8514202237129211, -1.1253174108317872, 1.9113529364215296, -0.7901094716838379, 0.7504305243492126, 0.004778623580932617],
    "Above_Pipette_Holder_P1": [0.8518240451812744, -1.3797095578959961, 1.6381872336017054, -0.26484520853076177, 0.7485695481300354, 0.007914340123534203],
    
    "Pipette_Above_SVH2_P1": [1.1697930097579956, -1.6097046337523402, 2.1604750792132776, -0.5488625329783936, 2.167518138885498, 0.004422497935593128],
    "Pipette_Just_In_SVH2_P1": [1.1696596145629883, -1.5474370897612353, 2.2160642782794397, -0.6669634145549317, 2.167961359024048, 0.003825228661298752],
    "Pipette_In_SVH2_P1": [1.169288158416748, -1.3945873540690918, 2.296798292790548, -0.9013450902751465, 2.1683976650238037, 0.0024986625649034977],
    
    "Pipette_Above_SVH2_P2": [1.1187692880630493, -1.7053538761534632, 2.237337891255514, -0.5302449029735108, 2.116180896759033, 0.004398844670504332],
    "Pipette_Just_In_SVH2_P2": [1.1260229349136353, -1.6343280277647914, 2.3094587961779993, -0.6737151902965088, 2.124135971069336, 0.0037398100830614567],
    "Pipette_In_SVH2_P2": [1.1255658864974976, -1.4603781265071412, 2.397318426762716, -0.9363041681102295, 2.1246302127838135, 0.0023139920085668564],
    
    "Pipette_Above_SVH2_P3": [1.0569829940795898, -1.7725392780699671, 2.323096815739767, -0.5491878551295777, 2.0544216632843018, 0.004179716110229492],
    "Pipette_Just_In_SVH2_P3": [1.0568301677703857, -1.7128902874388636, 2.375284973775045, -0.6612522763064881, 2.0547215938568115, 0.003672876628115773],
    "Pipette_In_SVH2_P3": [1.0505099296569824, -1.5446308844587584, 2.474193874989645, -0.9290871185115357, 2.0493617057800293, 0.002337896730750799],

    "Pipette_Near_Hotplate_Vial": [1.6094765663146973, -1.7065593204894007, 1.9414098898517054, -0.2339969438365479, 1.8890506029129028, 0.00498750526458025],
    "Pipette_Above_Hotplate_Vial": [1.7327353954315186, -1.515752987270691, 1.883343521748678, -0.36736448228869634, 1.7660012245178223, 0.004570244811475277],
    "Pipette_In_Hotplate_Vial": [1.7327182292938232, -1.4854273361018677, 1.9352906386004847, -0.44974978387866216, 1.7663404941558838, 0.0042514801025390625],

    "Vial_5": [1.6295762062072754, -0.7691212457469483, 0.8135131041156214, 1.464686556453369, 1.5105781555175781, 1.6134917736053467],
    "Vial_4": [1.6293808221817017, -0.7574216288379212, 0.6149275938617151, 1.6516057687946777, 1.51029372215271, 1.6123936176300049],
    "Vial_3": [1.5694230794906616, -1.6029640636839808, 1.988000218068258, 1.072641058559082, 1.5237963199615479, 1.544532299041748],
    "Vial_2": [1.557105302810669, -1.4412485522082825, 2.0756834189044397, 0.8969403940388183, 1.506769061088562, 1.5452169179916382],
    "Vial_1": [1.5603711605072021, -1.5427539025298138, 2.003195110951559, 1.063420220012329, 1.5177810192108154, 1.587868571281433]
}

MOVEMENT_PARAMS = {
    "speed": 0.8,
    "acceleration": 0.9,
    "blending": 0.01,
}

# Choices
# 1. Full synthesis using Mettler Toledo Quantos
# 2. Copper chloride and piroctone olamine solutions already prepared

# Choices once syngthesis is ready
# 1. Synthesise and measure one sample of piroctone olamine
# 2. Measure particle size distribution over time
# 3. Measure impact of copper chloride concentration on particle size distribution


# # === IKA RCT Digial Hotplate Configuration

def find_hotplate_port():
    """
    Detects and returns the serial port of an IKA RCT hotplate.
    Tries all ports until one responds successfully.
    """
    if not HAS_SERIAL or not HAS_HOTPLATE:
        print("Warning: serial or hotplate driver not available. Skipping lookup.")
        return None, None
        
    ports = serial.tools.list_ports.comports()
    plate = None

    for port in ports:
        # Windows ports: COMx
        if sys.platform.startswith('win'):
            if port.device.startswith('COM'):
                if 'IKA' in port.description or 'USB' in port.description:
                    print(f"Found IKA device on {port.device} ({port.description})")
                    return port.device, None
                print(f"Found possible device on {port.device} ({port.description})")
                return port.device, None
        
        # Linux ports: /dev/ttyUSBx or /dev/ttyACMx
        elif sys.platform.startswith('linux'):
            print(f"Trying port: {port.device} ({port.description})")
            try:
                serial_port = port.device
                # Create the hotplate instance
                plate = RCTDigitalHotplate(
                    device_name="IKA RCT Digital",
                    connection_mode="serial",
                    address=None,
                    port=serial_port
                )

                plate.connect()
                plate.initialize_device()
            except:
                print("No IKA hotplate on this serial port found", "\n")
            
        # macOS ports: /dev/tty.* or /dev/cu.*
        elif sys.platform.startswith('darwin'):
            if port.device.startswith('/dev/tty.') or port.device.startswith('/dev/cu.'):
                print(f"Trying port: {port.device} ({port.description})")
                try:
                    serial_port = port.device
                    plate = RCTDigitalHotplate(
                        device_name="IKA RCT Digital",
                        connection_mode="serial",
                        address=None,
                        port=serial_port
                    )
                    plate.connect()
                    plate.initialize_device()
                except:
                    print("No IKA hotplate on this serial port found", "\n")
                
    return port.device, plate   # THIS PART CAUSES THE ERROR, KEEP IT AS IS

# === Helper Functions ===
def degreestorad(angles_deg):
    return [angle * math.pi / 180 for angle in angles_deg]


class OutreachDemo:
    """Educational wrapper exposing only add_indicator/add_acid/add_base.

    Usage (for students):
        demo = OutreachDemo()
        demo.initialize()
        demo.add_indicator()
        demo.add_acid(0.25)
        demo.add_base(4.5)
        demo.dispose()  # optional; also called automatically
    """

    def __init__(
        self,
        robot_ip="192.168.0.2",
        robot_port=30003,
        gripper_port=63352,
        simulation=False,
    ):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.gripper_port = gripper_port
        self.simulation_mode = bool(simulation)
        self.robot = None
        self.gripper = None
        self.plate = None
        self._initialized = False
        self._busy_lock = threading.Lock()
        self._is_busy = False

    @property
    def is_busy(self):
        return self._is_busy

    # --- Lifecycle ---
    def initialize(self, front_home_position=ROBOT_POSITIONS_["Front_Home_Position"]):
        """Connect robot, gripper, hotplate; move home and start stirring."""
        if self.simulation_mode:
            self._initialized = True
            print("[SIMULATION] Initialization complete (no hardware connections attempted).")
            return

        # Check for missing hardware dependencies if not in simulation mode
        if not HAS_UR:
            raise ImportError("URControl dependency (ur-rtde) is missing. Switch to simulation mode or install dependencies.")
        if not HAS_GRIPPER:
            print("Warning: RobotiqGripper dependency missing. Gripper movements will fail.")
        
        if self.robot is None:
            self.robot = URControl(ip=self.robot_ip, port=self.robot_port)

        if self.gripper is None and HAS_GRIPPER:
            self.gripper = RobotiqGripper()
            self.gripper.connect(self.robot_ip, self.gripper_port)

        if self.plate is None:
            self.plate = find_hotplate_port()[1]
            print("Retrieved plate:", self.plate)

        # Initial position
        self._operate_gripper(0)
        self._move_robot_tcp(front_home_position)
        print("Starting in home position", "\n")

        # Turn on Hotplate stirring
        if self.plate:
            self.plate.set_speed(500)
            self.plate.start_stirring()
            time.sleep(0.1)
            print(f"Speed: {self.plate.get_speed()} rpm")

        self._initialized = True

    def dispose(self):
        """Safely stop stirring/cleanup. Called automatically on destruction."""
        if self.simulation_mode:
            self._initialized = False
            print("[SIMULATION] Dispose complete (no hardware cleanup required).")
            return

        try:
            if self.plate:
                self.plate.stop_stirring()
                print("Stirring stopped.")
            self._initialized = False
        except Exception as exc:
            print(f"Warning: failed to stop plate cleanly: {exc}")

    def __del__(self):
        # Automatic cleanup if object is garbage-collected
        try:
            self.dispose()
        except Exception:
            pass

    # --- Public student-facing methods ---
    def add_indicator(self):
        self._run_exclusive(
            self._add_indicator_impl,
            "Robot is currently busy. Wait for the current action to finish.",
        )

    def add_acid(self, amount_ml):
        self._run_exclusive(
            lambda: self._add_acid_impl(amount_ml),
            "Robot is currently busy. Wait for the current action to finish.",
        )

    def add_base(self, amount_ml):
        self._run_exclusive(
            lambda: self._add_base_impl(amount_ml),
            "Robot is currently busy. Wait for the current action to finish.",
        )

    def _add_indicator_impl(self):
        self._ensure_ready()
        print("Adding a dew drops of indicator to the hotplate sample vial.", "\n")
        self._pipetting_routine(input_vial=3, which_pipette=3, amount_ml=0.1)

    def _add_acid_impl(self, amount_ml):
        amount = self._validate_volume_ml(amount_ml, reagent_name="acid")
        self._ensure_ready()
        print(f"Adding {amount} ml of acid to the hotplate sample vial.", "\n")
        self._pipetting_routine(input_vial=1, which_pipette=1, amount_ml=amount)

    def _add_base_impl(self, amount_ml):
        amount = self._validate_volume_ml(amount_ml, reagent_name="base")
        self._ensure_ready()
        print(f"Adding {amount} ml of base to the hotplate sample vial.", "\n")
        self._pipetting_routine(input_vial=2, which_pipette=2, amount_ml=amount)

    # --- Internal helpers (kept private-ish) ---
    def _ensure_ready(self):
        if not self._initialized:
            raise RuntimeError("Call initialize() before performing operations.")
        if self.simulation_mode:
            return
        if not (self.robot and self.gripper):
            raise RuntimeError("Call initialize() before performing operations.")

    def _run_exclusive(self, action, busy_message):
        with self._busy_lock:
            if self._is_busy:
                raise RuntimeError(busy_message)
            self._is_busy = True

        try:
            action()
        finally:
            with self._busy_lock:
                self._is_busy = False

    def _validate_volume_ml(self, amount_ml, reagent_name):
        try:
            amount = float(amount_ml)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid {reagent_name} volume: enter a number in mL.")

        if amount <= 0:
            raise ValueError(f"Invalid {reagent_name} volume: value must be > 0 mL.")

        # Classroom-safe guardrail for accidental large inputs.
        if amount > 10:
            raise ValueError(
                f"Invalid {reagent_name} volume: value must be <= 10 mL for this demo."
            )

        return amount

    def _move_robot(self, position, speed=None, acceleration=None, blending=None):
        if self.simulation_mode:
            print(f"[SIMULATION] Move robot to position: {position}")
            return

        speed = MOVEMENT_PARAMS["speed"] if speed is None else speed
        acceleration = MOVEMENT_PARAMS["acceleration"] if acceleration is None else acceleration
        blending = MOVEMENT_PARAMS["blending"] if blending is None else blending

        self.robot.move_joint_list(
            position,
            speed,
            acceleration,
            blending,
        )
    
    def _move_robot_tcp(self, position):
        self.robot.movel_tcp(position, 0.2, 0.2)
        
        
    def _operate_gripper(self, position, speed=250):
        if self.simulation_mode:
            print(f"[SIMULATION] Move gripper to position: {position}")
            return
        self.gripper.move(position, speed, 125)
        
    def _pipetting_squeeze(self, starting_pos, amount_ml=2):
        if amount_ml <= 2:
            steps = int((amount_ml / 2) * 40)  # 40 steps from 201 to 241
            
        for step in range(steps):
            self._operate_gripper(starting_pos + step, speed=5)
        
            

    def _pipetting_routine(self, input_vial, which_pipette, amount_ml):
        """Automated pipetting routine to transfer liquid from input vial to hotplate sample vial."""
        # Decode pipette choice into positions (1-4)
        if which_pipette not in [1, 2, 3, 4]:
            raise ValueError("which_pipette must be an integer between 1 and 4.")
        which_pipette = str(which_pipette)
        Above_Pipette_Holder = f"Above_Pipette_Holder_P{which_pipette}"
        Pipette_Holder = f"Pipette_Holder_P{which_pipette}"

        # Decode input vial choice into positions (1-3)
        if input_vial not in [1, 2, 3]:
            raise ValueError("input_vial must be an integer between 1 and 3.")
        input_vial = str(input_vial)

        # Higher gripper values mean more squeezing, 201 is just starting to squeeze, 241 is fully squeezed with 2ml dispensed
        if amount_ml <= 2:
            steps = int((amount_ml / 2) * 40)  # 40 steps from 201 to 241
            repeats = 1
        else:
            repeats = int(amount_ml // 2)
            remainder = amount_ml % 2
            steps = int((remainder / 2) * 40) if remainder > 0 else 0
            if remainder > 0:
                repeats += 1

        print(
            f"Pipetting {amount_ml} ml using pipette {which_pipette} from input vial {input_vial} in {repeats} full repeats and {steps} steps for remainder.",
            "\n",
        )

        # Pick up pipette
        self._move_robot(ROBOT_POSITIONS[Above_Pipette_Holder])
        print(f"Above Pipette Holder Position {which_pipette}", "\n")
        time.sleep(0.1)
        self._operate_gripper(180)

        self._move_robot(ROBOT_POSITIONS[Pipette_Holder])
        print(f"Picked up pipette from Pipette Holder Position {which_pipette}", "\n")
        time.sleep(0.1)
        self._operate_gripper(200)
        time.sleep(0.1)

        self._move_robot(ROBOT_POSITIONS[Above_Pipette_Holder])
        print(f"Above Pipette Holder Position {which_pipette}", "\n")
        time.sleep(0.1)

        for r in range(repeats):
            Above_SVH = f"Pipette_Above_SVH2_P{input_vial}"
            Just_In_SVH = f"Pipette_Just_In_SVH2_P{input_vial}"
            In_SVH = f"Pipette_In_SVH2_P{input_vial}"

            self._move_robot(ROBOT_POSITIONS[Above_SVH])
            print(f"Above Sample Vial Holder Position {input_vial}", "\n")
            time.sleep(0.1)

            self._move_robot(ROBOT_POSITIONS[Just_In_SVH])
            print(f"Just inside Sample Vial Holder Position {input_vial}", "\n")
            time.sleep(0.1)
            self._operate_gripper(240)
            time.sleep(0.1)
            self._move_robot(ROBOT_POSITIONS[In_SVH])
            for position in range(235, 199, -5):
                
                print("Sucking up liquid", "\n")
                self._operate_gripper(position)
                time.sleep(0.1)

            self._move_robot(ROBOT_POSITIONS[Above_SVH])
            print(f"Above Sample Vial Holder Position {input_vial}", "\n")
            time.sleep(0.1)

            self._move_robot(ROBOT_POSITIONS["Pipette_Near_Hotplate_Vial"])
            print("Pipette on its way to the hotplate sample vial")
            time.sleep(0.1)

            self._move_robot(ROBOT_POSITIONS["Pipette_Above_Hotplate_Vial"])
            print("Above sample vial on the hotplate", "\n")
            time.sleep(0.1)

            self._move_robot(ROBOT_POSITIONS["Pipette_In_Hotplate_Vial"])
            print("In sample vial on the hotplate", "\n")
            time.sleep(0.1)

            if r < repeats - 1:
                dispense_steps = 40
            else:
                dispense_steps = steps
            for position in range(201, 201 + dispense_steps, 1):
                print("Dispensing liquid", "\n")
                self._operate_gripper(position)
                time.sleep(0.1)
            
            for position in range(201 + dispense_steps - 1, 199, -5):
                print("Unsqueezing pipette", "\n")
                self._operate_gripper(position)
                time.sleep(0.1)


            self._move_robot(ROBOT_POSITIONS["Pipette_Above_Hotplate_Vial"])
            self._move_robot(ROBOT_POSITIONS["Pipette_Near_Hotplate_Vial"])
            print("Pipette on its way to the sample vial")
            time.sleep(0.1)

        #getting rid of excess
        Above_SVH = f"Pipette_Above_SVH2_P{input_vial}"
        Just_In_SVH = f"Pipette_Just_In_SVH2_P{input_vial}"
        In_SVH = f"Pipette_In_SVH2_P{input_vial}"

        self._move_robot(ROBOT_POSITIONS[Above_SVH])
        print(f"Above Sample Vial Holder Position {input_vial}", "\n")
        time.sleep(0.1)

        self._move_robot(ROBOT_POSITIONS[Just_In_SVH])
        print(f"Just inside Sample Vial Holder Position {input_vial}", "\n")

        for position in range(201, 241, 1):
            print("Dispensing liquid", "\n")
            self._operate_gripper(position)
            time.sleep(0.1)
            
        for position in range(240, 199, -5):
            print("Unsqueezing pipette", "\n")
            self._operate_gripper(position)
            time.sleep(0.1)

        self._move_robot(ROBOT_POSITIONS[Above_SVH])
        print(f"Above Sample Vial Holder Position {input_vial}", "\n")
        time.sleep(0.1)

        self._move_robot(ROBOT_POSITIONS[Above_Pipette_Holder])
        print(f"Above Pipette Holder Position {which_pipette}", "\n")
        time.sleep(0.1)

        self._move_robot(
            ROBOT_POSITIONS[Pipette_Holder],
            speed=0.05,
            acceleration=MOVEMENT_PARAMS["acceleration"],
            blending=MOVEMENT_PARAMS["blending"],
        )
        print(f"Replaced Pipette {which_pipette}", "\n")
        time.sleep(0.1)
        self._operate_gripper(180)
        time.sleep(0.1)

        self._move_robot(ROBOT_POSITIONS[Above_Pipette_Holder])
        print(f"Above Pipette Holder Position {which_pipette}", "\n")
        time.sleep(0.1)

        self._move_robot(ROBOT_POSITIONS["Front_Home_Position"])
        print("Starting in home position", "\n")
        time.sleep(0.1)

    def pick_up_empty_pipette(self, positions: dict, speed: int = 1, n_sleep: int = 2):
        print("Opening gripper to pick up pipette", "\n")
        print(f'{PIPETTE_POSITIONS["Open"]}')
        
        self._operate_gripper(PIPETTE_POSITIONS["Open"], speed=speed)
        sleep(n_sleep)
        
        self._move_robot_tcp(positions[f"Pipette_1_without"])
        
        self._move_robot_tcp(positions["Pipette_1"])
        self._operate_gripper(PIPETTE_POSITIONS["Pick"], speed=speed)
        sleep(n_sleep)
        
        self._move_robot_tcp(positions["Pipette_1_pick"])
        
    def fill_pipette_from_vial(self, positions: dict, speed: int = 1, n_sleep: int = 2):
        self._move_robot_tcp(positions["Vial_1_above"])
        self._move_robot_tcp(positions["Vial_1_inside"])
        self._operate_gripper(PIPETTE_POSITIONS["Squeeze"], speed=speed)
        sleep(n_sleep)
        self._operate_gripper(PIPETTE_POSITIONS["Pick"], speed=speed)
        sleep(n_sleep)
        self._move_robot_tcp(positions["Vial_1_above"])
        
    def dispense_pipette_to_vial(self, positions: dict, speed: int = 1, n_sleep: int = 2):    
        self._move_robot_tcp(positions["Vial_2_above"])
        self._move_robot_tcp(positions["Vial_2_inside"])
        self._operate_gripper(PIPETTE_POSITIONS["Squeeze"], speed=speed)
        sleep(n_sleep)
        # demo._pipetting_squeeze(PIPETTE_POSITIONS["Squeeze"])
        self._move_robot_tcp(positions["Vial_2_above"])
        self._operate_gripper(PIPETTE_POSITIONS["Pick"], speed=speed)
        sleep(n_sleep)

__all__ = ["OutreachDemo"]
