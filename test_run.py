import json
from subprocess import PIPE
from time import sleep

from Outreach_Core import OutreachDemo
from Outreach_Core import PIPETTE_POSITIONS, ROBOT_POSITIONS_

import numpy as np

def load_positions(filename):
    with open(filename, "r", encoding="utf-8") as file:
        raw_pos = json.load(file)["run_0"]
    
    pipette_hover = np.array(raw_pos["pos_pip_tcp"]) + np.array([0, 0, 0.05, 0, 0, 0])
    pipette_pick = np.array(raw_pos["pos_pip_tcp"]) + np.array([0, 0, 0.15, 0, 0, 0])
    vial_1_hover = np.array(raw_pos["pos_vial_1_tcp"])
    vial_1_above = np.array(vial_1_hover) + np.array([0, 0, 0.05, 0, 0, 0])
    vial_1_inside = np.array(vial_1_hover) + np.array([0, 0, -0.07, 0, 0, 0])
    
    vial_2_hover = np.array(raw_pos["pos_vial_2_tcp"])
    vial_2_above = np.array(vial_2_hover) + np.array([0, 0, 0.05, 0, 0, 0])
    vial_2_inside = np.array(vial_2_hover) + np.array([0, 0, -0.05, 0, 0, 0])

    positions = {
        "Front_Home_Position": np.array(raw_pos["pos_start_tcp"]),
        "Pipette_1_without": pipette_hover,
        "Pipette_1": np.array(raw_pos["pos_pip_tcp"]),
        "Pipette_1_pick": pipette_pick,
        "Vial_1_above": vial_1_above, 
        "Vial_1_inside": vial_1_inside, 
        "Vial_2_above": vial_2_above, 
        "Vial_2_inside": vial_2_inside, 
    }
    return positions
    
    
    
def main():
    ROBOT_POSITIONS = load_positions("positions.json")
    # ROBOT_POSITIONS = ROBOT_POSITIONS_
    demo = OutreachDemo(simulation=False)
    demo.initialize(ROBOT_POSITIONS["Front_Home_Position"])
    
    try:        
        
        speed = 1
        sleep_gripper = 2
        # pick up the empty pipette
        demo.pick_up_empty_pipette(ROBOT_POSITIONS)
        # demo._move_robot(ROBOT_POSITIONS["Pipette_1_without"])
        # demo._operate_gripper(PIPETTE_POSITIONS["Open"], speed=speed)
        # sleep(sleep_gripper)
        # demo._move_robot(ROBOT_POSITIONS["Pipette_1"])
        # demo._operate_gripper(PIPETTE_POSITIONS["Pick"], speed=speed)
        # sleep(sleep_gripper)
        # demo._move_robot(ROBOT_POSITIONS["Pipette_1_pick"])
        
        # fill up the pipette
        demo.fill_pipette_from_vial(ROBOT_POSITIONS, speed=speed, n_sleep=sleep_gripper)
        # demo._move_robot(ROBOT_POSITIONS["Vial_1_above"])
        # demo._operate_gripper(PIPETTE_POSITIONS["Squeeze"], speed=speed)
        # sleep(sleep_gripper)
        # demo._move_robot(ROBOT_POSITIONS["Vial_1_inside"])
        # demo._operate_gripper(PIPETTE_POSITIONS["Pick"], speed=speed)
        # sleep(sleep_gripper)
        # demo._move_robot(ROBOT_POSITIONS["Vial_1_above"])
    
        # tranfer liquid to vial 2
        demo.dispense_pipette_to_vial(ROBOT_POSITIONS, speed=speed, n_sleep=sleep_gripper)
        # demo._move_robot(ROBOT_POSITIONS["Vial_2_above"])
        # demo._move_robot(ROBOT_POSITIONS["Vial_2_inside"])
        # demo._operate_gripper(PIPETTE_POSITIONS["Squeeze"], speed=speed)
        # sleep(sleep_gripper)
        # # demo._pipetting_squeeze(PIPETTE_POSITIONS["Squeeze"])
        # demo._move_robot(ROBOT_POSITIONS["Vial_2_above"])
        # demo._operate_gripper(PIPETTE_POSITIONS["Pick"], speed=speed)
        # sleep(sleep_gripper)
        
        
                

        
        # print(f"Current joint positions: {demo.get_current_joint_positions()}")
        # sleep(1)
        # demo._operate_gripper(0)
        # # demo._move_robot(ROBOT_POSITIONS["Vial_2"])
        
        # demo._operate_gripper(119)
        # sleep(2)
        # demo._move_robot(ROBOT_POSITIONS["Vial_3"])
        # sleep(2)
        # demo._move_robot(ROBOT_POSITIONS["Vial_4"])
        # sleep(2)
        # demo._move_robot(ROBOT_POSITIONS["Vial_5"])
        # demo._operate_gripper(0)
        # sleep(2)
        # demo._move_robot(ROBOT_POSITIONS["Front_Home_Position"])
        
        # New stuff
        
        # Add indicator
        # demo.add_acid(5)
        # demo.add_indicator()
        # demo.add_base(4)

    # try:
    #     demo.add_indicator()
    #     demo.add_acid(5)
    # finally:
    
    finally:
        demo.dispose()


if __name__ == "__main__":
    main()