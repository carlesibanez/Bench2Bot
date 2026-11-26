import sys
import os
import time
import argparse
import math
import json 
# Add the directory containing robotiq_preamble.py to the Python search path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'robotiq'))

from utils.UR_Functions import URfunctions as URControl

HOST = "192.168.0.2"
PORT = 30003

def main():
    robot = URControl(ip="192.168.0.2", port=30003)
    print(robot.get_current_joint_positions().tolist())
    print(robot.get_current_tcp())
    
    positions = {}
    for run in range(1):
        
        input("Get robot into starting position...")
        pos_start = robot.get_current_joint_positions().tolist()
        pos_start_tcp = robot.get_current_tcp().tolist()
        
        input("Get robot to pipette...")
        pos_pip = robot.get_current_joint_positions().tolist()
        pos_pip_tcp = robot.get_current_tcp().tolist()
        
        input("Get robot with pipette to vial 1 (at vial entrance)...")
        pos_vial_1 = robot.get_current_joint_positions().tolist()
        pos_vial_1_tcp = robot.get_current_tcp().tolist()
    
        input("Get robot with pipette to vial 2 (at vial entrance...")
        pos_vial_2 = robot.get_current_joint_positions().tolist()
        pos_vial_2_tcp = robot.get_current_tcp().tolist()
        
        positions[f"run_{run}"] = {
            "pos_start": pos_start,
            "pos_start_tcp": pos_start_tcp,
            "pos_pip": pos_pip,
            "pos_pip_tcp": pos_pip_tcp,
            "pos_vial_1": pos_vial_1,
            "pos_vial_1_tcp": pos_vial_1_tcp,
            "pos_vial_2": pos_vial_2,
            "pos_vial_2_tcp": pos_vial_2_tcp
        }
    
    filename = "positions.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(positions, f, indent=4)
        print(f"Positions saved to {filename}")

    
if __name__ == '__main__':
     main()