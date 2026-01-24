# interactive_box.py，通过键盘交互控制箱子位置，方便理解mujoco坐标系

from ee_sim_env import make_ee_sim_env
import matplotlib.pyplot as plt
import numpy as np

env = make_ee_sim_env('sim_transfer_cube')
ts = env.reset()

# 获取初始箱子位置
physics = env._physics
box_start_idx = 16  # 箱子在 qpos 中的索引

# 初始位置
box_pos = physics.data.qpos[box_start_idx:box_start_idx+3].copy()

print("=" * 70)
print("交互式箱子控制")
print("=" * 70)
print("\n键盘控制：")
print("  W/S - 前后移动 (Y轴)")
print("  A/D - 左右移动 (X轴)")
print("  Q/E - 上下移动 (Z轴)")
print("  R   - 重置位置")
print("  ESC - 退出")
print("\n点击图像窗口，然后按键控制")
print("=" * 70 + "\n")

# 显示窗口
fig, (ax_img, ax_info) = plt.subplots(1, 2, figsize=(14, 6))
ax_img.axis('off')
ax_info.axis('off')
plt.ion()

step_size = 0.01  # 每次移动 1cm

def on_key(event):
    global box_pos
    
    if event.key == 'w':
        box_pos[1] += step_size  # Y+ (前)
    elif event.key == 's':
        box_pos[1] -= step_size  # Y- (后)
    elif event.key == 'a': 
        box_pos[0] -= step_size  # X- (左)
    elif event.key == 'd':
        box_pos[0] += step_size  # X+ (右)
    elif event.key == 'q':
        box_pos[2] += step_size  # Z+ (上)
    elif event.key == 'e': 
        box_pos[2] -= step_size  # Z- (下)
    elif event.key == 'r':
        # 重置到初始位置
        box_pos = physics.data.qpos[box_start_idx:box_start_idx+3].copy()
        print("已重置箱子位置")
    elif event.key == 'escape':
        plt.close()
        return
    
    # 更新箱子位置
    physics.data.qpos[box_start_idx:box_start_idx+3] = box_pos
    
    # 推进一步物理仿真
    left_pose = ts.observation['mocap_pose_left']
    right_pose = ts.observation['mocap_pose_right']
    action = np.concatenate([left_pose, [1], right_pose, [1]])
    new_ts = env.step(action)
    
    # 更新显示
    update_display(new_ts)

def update_display(ts):
    # 更新图像
    img = ts.observation['images']['angle']
    ax_img.clear()
    ax_img.imshow(img)
    ax_img.set_title("MuJoCo View", fontsize=14)
    ax_img.axis('off')
    
    # 更新信息
    ax_info.clear()
    ax_info.axis('off')
    
    right_pos = ts.observation['mocap_pose_right'][: 3]
    box = ts.observation['env_state'][:3]
    
    info_text = f"""
Current Positions: 

Box: 
  X = {box[0]:.3f} m  {'(Right)' if box[0] > 0 else '(Left)'}
  Y = {box[1]:.3f} m  (Front/Back)
  Z = {box[2]:.3f} m  (Height)

Right End-Effector:
  X = {right_pos[0]:.3f} m
  Y = {right_pos[1]:.3f} m
  Z = {right_pos[2]:.3f} m

Distance:
  Horizontal: {np.linalg.norm(right_pos[: 2] - box[:2]):.3f} m
  Height Difference: {right_pos[2] - box[2]:.3f} m
Control:
  W/S:  Front/Back
  A/D: Left/Right
  Q/E: Up/Down
  R: Reset
    """
    
    ax_info.text(0.1, 0.5, info_text, fontsize=11, 
                verticalalignment='center', family='monospace')
    
    plt.draw()

# 绑定键盘事件
fig.canvas.mpl_connect('key_press_event', on_key)

# 初始显示
update_display(ts)

plt.show(block=True)