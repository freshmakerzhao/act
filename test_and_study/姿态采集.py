import mujoco
import mujoco.viewer

# 加载模型
XML_PATH = r'E:\Workspace_robot\act\assets\fairino5_single\single_fairino_fr5_ee_transfer_cube.xml'

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# 【架构补丁】暂时关闭所有可能导致“锁定”的因素
model.eq_active[0] = 0  # 禁用第一个 equality 约束 (weld)

print("\n" + "="*50)
print("--- 法奥 FR5 姿态采集助手 ---")
print("1. 如果右侧滑块仍不可用，请点击左上角的 'Reload' 按钮。")
print("2. 调整好姿态后，在控制台观察实时打印。")
print("="*50 + "\n")

# 使用 launch (非 passive)，这通常会启用更多 GUI 权限
mujoco.viewer.launch(model, data)

# 窗口关闭后，你可以在终端看到最后停留的 qpos
print(f"\n最终 qpos: {' '.join([f'{x:.4f}' for x in data.qpos])}")