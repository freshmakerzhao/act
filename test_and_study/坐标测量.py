import mujoco
import mujoco.viewer
import numpy as np
import time

# 加载你的单臂场景 XML
XML_PATH = r'E:\Workspace_robot\act\assets\fairino5_single\single_fairino_fr5_ee_transfer_cube.xml'

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# 临时关闭 weld 约束，以便测量自由姿态
model.eq_active[0] = 0 

with mujoco.viewer.launch_passive(model, data) as viewer:
    # 1. 设定你认为合理的初始关节角度 (j1-j6)
    # 示例：让法奥稍微弯曲伸手
    grasp_ready_qpos = [0.0305, -1.6, -1.55, -1.81, 1.53, -1.53] 
    data.qpos[:6] = grasp_ready_qpos
    
    # 2. 执行前向动力学计算坐标
    mujoco.mj_forward(model, data)
    
    # 3. 获取夹爪目前的全局坐标 (xpos)
    gripper_id = model.body('vx300s_right/gripper_link').id
    actual_pos = data.xpos[gripper_id]
    
    print("\n" + "="*50)
    print("--- 测量结果 (请根据此数值更新 XML) ---")
    print(f"1. Mocap 应设定的 pos: {actual_pos[0]:.4f} {actual_pos[1]:.4f} {actual_pos[2]:.4f}")
    print(f"2. Keyframe 应设定的 qpos: {' '.join([f'{x:.4f}' for x in data.qpos])}")
    print("="*50 + "\n")

    while viewer.is_running():
        step_start = time.time()
        viewer.sync()
        # 维持实时仿真速度
        time_to_sleep = model.opt.timestep - (time.time() - step_start)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)