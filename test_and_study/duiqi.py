import mujoco
import mujoco.viewer
import numpy as np

# 加载你的单臂场景 XML
XML_PATH = r'E:\Workspace_robot\act\assets\fairino5_single\single_fairino_fr5_ee_transfer_cube.xml'
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# 获取相关 ID
mocap_id = model.body('mocap_right').mocapid[0]
gripper_id = model.body('vx300s_right/gripper_link').id

with mujoco.viewer.launch_passive(model, data) as viewer:
    # 【核心步骤】启动瞬间，将 Mocap 的位姿强制设为夹爪目前的位姿
    data.mocap_pos[mocap_id] = data.xpos[gripper_id]
    data.mocap_quat[mocap_id] = data.xquat[gripper_id]
    
    # 同步物理状态
    mujoco.mj_forward(model, data)
    
    print("Mocap 已强制吸附至夹爪中心。")
    print("现在拖动红球，机械臂将‘零距离’跟随。")

    while viewer.is_running():
        mujoco.mj_step(model, data)
        
        # 实时打印用于替换 XML 的 15 位 qpos
        qpos_list = [f"{x:.4f}" for x in data.qpos]
        print(f"\r当前 qpos: {' '.join(qpos_list)}", end="")
        
        viewer.sync()