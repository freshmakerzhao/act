import mujoco
import mujoco.viewer
import numpy as np
import time

# ================= 设置区域 =================
# 1. 确认 XML 路径
XML_PATH = '/home/zhaoshuai/workspace_act/act/assets/fairino5_single/single_viperx_ee_transfer_cube.xml'

# 2. L型 8位关节角度 (j1-j6 + 2个手指)
START_POSE = [0, -1.54, 1.45, -1.4, -1.56, -1.56, 0.057,   -0.057]

EE_NAME = 'fairino_fr5/wrist3_link' 
# ===========================================

def debug_initialize(model, data):
    
    data.qpos[:8] = START_POSE
    
    mujoco.mj_forward(model, data)

    site_id = model.site('gripper_tip').id
    ee_id = model.body(EE_NAME).id
    
    data.mocap_pos[0] = data.site_xpos[site_id]
    data.mocap_pos[0] = [0.00394128,0.39627973,0.4555668 ]
    data.mocap_quat[0] = [0.7071, 0.7071, 0, 0]

    quat_res = np.zeros(4)
    mujoco.mju_mat2Quat(quat_res, data.site_xmat[site_id])
    data.mocap_quat[0] = quat_res

            
def main():
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 启动交互式查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 执行初始化
        debug_initialize(model, data)
        
        while viewer.is_running():
            step_start = time.time()
            
            # 如果你在查看器中按了 Backspace，手动触发重置
            # 注意：launch_passive 下通常需要手动逻辑处理快捷键，此处简化为持续步进
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # 维持频率
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()