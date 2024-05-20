import torch
import mujoco
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot


robot_cfg = {
    "mesh": True,
    "rel_joint_lm": False,
    "upright_start": False,
    "remove_toe": False,
    "real_weight": True,
    "real_weight_porpotion_capsules": True,
    "real_weight_porpotion_boxes": True,
    "replace_feet": True,
    "big_ankle": True,
    "freeze_hand": False,
    "box_body": True,
    "model": "smpl",
    "body_params": {},
    "joint_params": {},
    "geom_params": {},
    "actuator_params": {},
    "ball_joint": False,
    "create_vel_sensors": False,  # Create global and local velocities sensors.
    "sim": "isaacgym"
}

robot = SMPL_Robot(robot_cfg, data_dir='smpl')
betas = torch.zeros(1, 10)
betas[0] = 1
gender = [0]
robot.load_from_skeleton(betas=betas)
robot.write_xml('smpl.xml')
mj_mode = mujoco.MjModel.from_xml_path('smpl.xml')
mj_data = mujoco.MjData(mj_mode)