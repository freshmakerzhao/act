from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
import numpy as np

xml_path = "/home/zhaoshuai/workspace_act/act/assets/vx300s_single/bimanual_viperx_ee_transfer_cube.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
print("qpos_len:", model.nq)
joint_name_list = [model.joint(i).name for i in range(model.njnt)]
for i, name in enumerate(joint_name_list):
    print(f"joint {i}: {name}")

physics = mujoco.Physics.from_xml_path(xml_path)
# 输出所有joint
for i in range(physics.model.njnt):
    joint_name = physics.model.joint(i).name
    joint_pos = physics.data.qpos[i]
    print(f"Joint {i} ({joint_name}): Position = {joint_pos}")