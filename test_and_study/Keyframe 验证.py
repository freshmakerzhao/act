import mujoco
import mujoco.viewer
import time

# 指定你的主场景文件
XML_PATH = r'E:\Workspace_robot\act\assets\fairino5_single\single_fairino_fr5_ee_transfer_cube.xml'

def main():
    # 1. 加载模型与数据
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # 2. 【核心步骤】将数据重置为索引为 0 的关键帧
    # 这会强制覆盖所有的关节角度 (qpos) 和物体位姿
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        # 执行前向计算以更新所有零部件的全局坐标 (xpos)
        mujoco.mj_forward(model, data)
        print(f"成功加载 Keyframe (共 {model.nkey} 个)，已重置至初始姿态。")
    else:
        print("错误：XML 中未找到任何 keyframe 定义！")
        return

    # 3. 启动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n" + "="*50)
        print("验证指南：")
        print("1. 机械臂现在应处于你调节好的‘准备抓取’姿态。")
        print("2. 检查 Mocap 红球是否在夹爪中心。")
        print("3. 如果红球偏离太远，请将控制台打印的 actual_pos 回填到 XML。")
        print("="*50 + "\n")

        # 获取夹爪 ID 用于实时位置监控
        gripper_id = model.body('vx300s_right/gripper_link').id

        while viewer.is_running():
            step_start = time.time()

            # 物理步进
            mujoco.mj_step(model, data)

            # 实时输出夹爪坐标，方便你核对 Mocap 的 pos 属性
            current_ee_pos = data.xpos[gripper_id]
            print(f"\r夹爪当前位置 (用于对齐 Mocap): {current_ee_pos[0]:.4f} {current_ee_pos[1]:.4f} {current_ee_pos[2]:.4f}", end="")

            viewer.sync()

            # 维持仿真速度
            time_to_sleep = model.opt.timestep - (time.time() - step_start)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

if __name__ == "__main__":
    main()