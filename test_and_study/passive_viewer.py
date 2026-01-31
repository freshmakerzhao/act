import mujoco
import mujoco.viewer
import time
import numpy as np

# 1. 填入你的主 XML 路径
# XML_PATH = r'E:\Workspace_robot\act\assets\fairino5_single\single_fairino_fr5_ee_transfer_cube.xml'
XML_PATH = r'E:\Workspace_robot\act\assets\vx300s_single\single_viperx_ee_transfer_cube.xml'

def main():
    # 加载模型与数据
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # 启动交互式查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n" + "="*50)
        print("MuJoCo 交互式示教助手已启动")
        print("操作指南：")
        print("1. 按住 【Ctrl + 鼠标左键】 拖拽机械臂末端或 Mocap 红点。")
        print("2. 调整好姿态后，观察下方控制台输出。")
        print("3. 直接复制打印出的那行 qpos 字符串。")
        print("="*50 + "\n")

        while viewer.is_running():
            step_start = time.time()

            # 物理仿真步进
            mujoco.mj_step(model, data)

            # 获取当前 qpos 状态
            # 顺序：j1-j6 (6位) + 夹爪 (2位) + 方块 (7位) = 15位
            current_qpos = data.qpos.copy()
            
            # 将数组转为 XML 格式的字符串
            qpos_str = " ".join([f"{x:.4f}" for x in current_qpos])
            
            # 实时打印到控制台（带清除行，防止刷屏）
            print(f"\r当前 qpos: {qpos_str}", end="")

            # 同步渲染
            viewer.sync()

            # 控制仿真速度
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()