# view_mujoco.py

from ee_sim_env import make_ee_sim_env
import matplotlib.pyplot as plt
import numpy as np

# 创建环境
env = make_ee_sim_env('sim_transfer_cube')
ts = env.reset()

# 显示窗口
plt.ion()
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

print("=" * 70)
print("MuJoCo 实时查看器")
print("=" * 70)
print("\n按 Ctrl+C 停止\n")

step = 0
try:
    while True: 
        # 获取当前状态
        right_pose = ts.observation['mocap_pose_right']  # 7维：xyz + quat
        left_pose = ts.observation['mocap_pose_left']    # 7维
        box_pose = ts.observation['env_state']           # 7维
        
        # 打印信息
        print(f"\n[第 {step} 步]")
        print(f"右臂末端位置: [{right_pose[0]:.3f}, {right_pose[1]:.3f}, {right_pose[2]:.3f}]")
        print(f"右臂末端姿态:  [{right_pose[3]:.3f}, {right_pose[4]:.3f}, {right_pose[5]:.3f}, {right_pose[6]:.3f}]")
        print(f"箱子位置: [{box_pose[0]:.3f}, {box_pose[1]:.3f}, {box_pose[2]:.3f}]")
        
        # 显示图像
        img = ts.observation['images']['angle']
        ax.clear()
        ax.imshow(img)
        ax.set_title(f"Step {step}", fontsize=14)
        ax.axis('off')
        plt.pause(0.1)
        
        # 构造动作：左臂 8维 + 右臂 8维 = 16维
        # mocap_pose 是 7维，需要加上夹爪（1维）
        action = np.concatenate([
            left_pose,   # 7维:  xyz + quat
            [1],         # 1维: 左臂夹爪（1=打开）
            right_pose,  # 7维
            [1]          # 1维: 右臂夹爪
        ])
        
        ts = env.step(action)
        step += 1
        
except KeyboardInterrupt: 
    print("\n\n停止")
    plt.close()