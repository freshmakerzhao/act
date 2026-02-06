import mujoco
import mujoco.viewer
import time
import numpy as np

# 配置文件路径
XML_PATH = r'E:\Workspace_robot\act\assets\fairino5_single\single_fairino_fr5_ee_transfer_cube.xml'

def main():
    # 1. 加载模型与数据
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # 2. 获取关键 ID
    # 对应你 XML 中的 body name
    gripper_id = model.body('vx300s_right/gripper_link').id
    mocap_id = model.body('mocap_right').mocapid[0]

    # 3. 启动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # --- 核心吸附逻辑 ---
        # 强制将 Mocap 移动到夹爪当前的全局位置
        data.mocap_pos[mocap_id] = data.xpos[gripper_id]
        data.mocap_quat[mocap_id] = data.xquat[gripper_id]
        
        # 必须调用前向计算刷新约束矩阵，防止启动瞬间产生巨大排斥力
        mujoco.mj_forward(model, data) 
        
        print("\n" + "="*50)
        print("法奥机械臂吸附测试已启动")
        print("1. 此时 Mocap 红球应刚好包裹在夹爪中心。")
        print("2. 按住 Ctrl + 鼠标左键 拖动红球观察跟随效果。")
        print("3. 按住 Ctrl + 鼠标右键 旋转红球观察末端自转。")
        print("="*50 + "\n")

        while viewer.is_running():
            step_start = time.time()

            # 物理步进
            mujoco.mj_step(model, data)

            # 获取当前 qpos 用于回填 XML
            qpos_str = " ".join([f"{x:.4f}" for x in data.qpos])
            print(f"\r实时 qpos (可复制): {qpos_str}", end="")

            viewer.sync()

            # 维持实时仿真速度
            time_to_sleep = model.opt.timestep - (time.time() - step_start)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

if __name__ == "__main__":
    main()