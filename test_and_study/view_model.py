import mujoco
from mujoco import viewer
absolute_import_path = "/home/zhaoshuai/workspace_act/act/assets/vx300s_single"
model = mujoco.MjModel.from_xml_path(f"{absolute_import_path}/single_viperx_ee_transfer_cube.xml")
data = mujoco.MjData(model)

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        mujoco.mj_step(model, data)