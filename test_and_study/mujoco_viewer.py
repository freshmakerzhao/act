import mujoco
from mujoco import viewer
# model = mujoco.MjModel.from_xml_path("/home/zhaoshuai/workspace_act/act/assets/vx300s_single/single_viperx_ee_transfer_cube.xml")
# model = mujoco.MjModel.from_xml_path("/home/zhaoshuai/workspace_act/act/assets/fairino5_single/single_viperx_ee_transfer_cube.xml")
model = mujoco.MjModel.from_xml_path("/home/zhaoshuai/workspace_act/act/assets/excavator_simple/single_viperx_ee_transfer_cube.xml")

data = mujoco.MjData(model)

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        mujoco.mj_step(model, data)