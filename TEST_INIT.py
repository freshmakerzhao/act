import mujoco
import mujoco.viewer
import numpy as np
import time

# ================= 设置区域 =================
# 1. 选择设备
EQUIPMENT_MODEL = 'excavator_simple'
# EQUIPMENT_MODEL = 'fairino5_single'

if EQUIPMENT_MODEL == 'excavator_simple':
    XML_PATH = '/home/zhaoshuai/workspace_act/act/assets/excavator_simple/single_viperx_ee_transfer_cube.xml'
    EXCAVATOR_JOINTS = ['j1_swing', 'j2_boom', 'j3_stick', 'j4_bucket']
    START_POSE = [0.0, -0.25, -0.5, -0.5]
    MOCAP_ALIGN_BODY = 'excavator_simple/bucket'
    BASE_MOCAP_POS = np.array([5.3, 0.0, 0.4])
    DYNAMIC_MOCAP = False
else:
    XML_PATH = '/home/zhaoshuai/workspace_act/act/assets/fairino5_single/single_viperx_ee_transfer_cube.xml'
    # L型 8位关节角度 (j1-j6 + 2个手指)
    START_POSE = [0, -1.54, 1.45, -1.4, -1.56, -1.56, 0.057, -0.057]
    START_POSE2 = [0, 0, 0, 0, 0, 0, 0.057, -0.057]
    EE_NAME = 'fairino_fr5/wrist3_link'
# ===========================================

def debug_initialize(model, data):
    if EQUIPMENT_MODEL == 'excavator_simple':
        for joint_name, joint_value in zip(EXCAVATOR_JOINTS, START_POSE):
            joint_id = model.joint(joint_name).id
            qpos_adr = model.jnt_qposadr[joint_id]
            data.qpos[qpos_adr] = joint_value
        mujoco.mj_forward(model, data)
        data.mocap_pos[0] = BASE_MOCAP_POS
        data.mocap_quat[0] = [1, 0, 0, 0]
    else:
        data.qpos[:8] = START_POSE2
        mujoco.mj_forward(model, data)

        # site_id = model.site('gripper_tip').id
        # ee_id = model.body(EE_NAME).id
        # data.mocap_pos[0] = data.site_xpos[site_id]
        data.mocap_pos[0] = [0, 0.396, 0.38]
        # data.mocap_quat[0] = [0.7071, 0.7071, 0, 0]
        data.mocap_quat[0] = [1, 0, 0, 0]

        # quat_res = np.zeros(4)
        # mujoco.mju_mat2Quat(quat_res, data.site_xmat[site_id])
        # data.mocap_quat[0] = quat_res

            
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

        t = 0.0
        keyframes = None
        keyframe_idx = 0
        if EQUIPMENT_MODEL == 'excavator_simple' and model.nmocap > 0:
            keyframes = [
                {"t": 0.0, "pos": BASE_MOCAP_POS + np.array([0.0, 0.0, 0.0]), "quat": [1, 0, 0, 0]},
                {"t": 20.0, "pos": BASE_MOCAP_POS + np.array([-1.5, 0.0, 0.5]), "quat": [1, 0, 0, 0]},
            ]
        
        while viewer.is_running():
            step_start = time.time()
            
            # 如果你在查看器中按了 Backspace，手动触发重置
            # 注意：launch_passive 下通常需要手动逻辑处理快捷键，此处简化为持续步进
            
            if (EQUIPMENT_MODEL == 'excavator_simple' and model.nmocap > 0 and
                    keyframes is not None and DYNAMIC_MOCAP):
                if keyframe_idx < len(keyframes) - 1 and t >= keyframes[keyframe_idx + 1]["t"]:
                    keyframe_idx += 1

                curr_kf = keyframes[keyframe_idx]
                next_kf = keyframes[min(keyframe_idx + 1, len(keyframes) - 1)]
                dt = next_kf["t"] - curr_kf["t"]
                if dt <= 0:
                    frac = 0.0
                else:
                    frac = (t - curr_kf["t"]) / dt

                data.mocap_pos[0] = curr_kf["pos"] + (next_kf["pos"] - curr_kf["pos"]) * frac
                data.mocap_quat[0] = curr_kf["quat"]

            mujoco.mj_step(model, data)
            viewer.sync()
            
            # 维持频率
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            t += model.opt.timestep

if __name__ == "__main__":
    main()