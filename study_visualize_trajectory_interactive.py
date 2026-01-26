# 文件名：visualize_trajectory_interactive.py，可视化关键路点轨迹，按钮控制切换路点

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pyquaternion import Quaternion
from ee_sim_env import make_ee_sim_env
from matplotlib.font_manager import FontProperties

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False

class ButtonControlledVisualizer:
    
    def __init__(self, task_name='sim_transfer_cube'):
        self.env = make_ee_sim_env(task_name)
        self.ts = self.env.reset()
        self.generate_trajectory()
        
        # 当前路点索引（左右独立）
        self.current_left_idx = 0
        self.current_right_idx = 0
        
        # 设置图形界面
        self.fig = plt.figure(figsize=(16, 9))
        
        # MuJoCo 视图（左侧）
        self.ax_img = plt.subplot(2, 1, 1)
        img = self.ts.observation['images']['angle']
        self.plt_img = self.ax_img.imshow(img)
        self.ax_img.axis('off')
        self.ax_img.set_title("MuJoCo 视图", fontsize=14)

        # 信息文本（左下）
        self.ax_info = plt.subplot(2, 1, 2)
        self.ax_info.axis('off')

        # 调整布局，为右侧按钮留空间
        self.ax_img.set_position([0.05, 0.55, 0.45, 0.4])
        self.ax_info.set_position([0.05, 0.10, 0.45, 0.35])
        
        # 控制按钮（右下，左右独立）
        ax_btn_left_prev = plt.axes([0.55, 0.25, 0.18, 0.075])
        ax_btn_left_next = plt.axes([0.75, 0.25, 0.18, 0.075])
        ax_btn_right_prev = plt.axes([0.55, 0.15, 0.18, 0.075])
        ax_btn_right_next = plt.axes([0.75, 0.15, 0.18, 0.075])
        ax_btn_left_play = plt.axes([0.55, 0.35, 0.18, 0.075])
        ax_btn_right_play = plt.axes([0.75, 0.35, 0.18, 0.075])
        
        self.btn_left_prev = Button(ax_btn_left_prev, '左臂 ← 上一个')
        self.btn_left_next = Button(ax_btn_left_next, '左臂 下一个 →')
        self.btn_right_prev = Button(ax_btn_right_prev, '右臂 ← 上一个')
        self.btn_right_next = Button(ax_btn_right_next, '右臂 下一个 →')
        self.btn_left_play = Button(ax_btn_left_play, '左臂 ▶ 播放')
        self.btn_right_play = Button(ax_btn_right_play, '右臂 ▶ 播放')
        
        self.btn_left_prev.on_clicked(self.prev_left_waypoint)
        self.btn_left_next.on_clicked(self.next_left_waypoint)
        self.btn_right_prev.on_clicked(self.prev_right_waypoint)
        self.btn_right_next.on_clicked(self.next_right_waypoint)
        self.btn_left_play.on_clicked(self.play_left_trajectory)
        self.btn_right_play.on_clicked(self.play_right_trajectory)
        
        plt.ion()
        plt.show()
        
        # 显示第一个路点
        self.update_display()
    
    def generate_trajectory(self):
        """生成轨迹"""
        init_mocap_pose_right = self.ts.observation['mocap_pose_right']
        init_mocap_pose_left = self.ts.observation['mocap_pose_left']
        
        box_info = np.array(self.ts.observation['env_state'])
        box_xyz = box_info[:3]
        
        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)
        
        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)
        meet_xyz = np.array([0, 0.5, 0.25])
        
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[: 3], "quat": init_mocap_pose_left[3:], "gripper": 0, "desc": "初始位置"},
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1, "desc": "接近交接点"},
            {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1, "desc": "到达交接点"},
            {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0, "desc": "闭合夹爪"},
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0, "desc": "后退"},
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0, "desc": "保持"},
        ]
        
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[: 3], "quat": init_mocap_pose_right[3:], "gripper": 0, "desc": "初始位置"},
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1, "desc": "箱子上方"},
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1, "desc": "下降"},
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper":  0, "desc": "闭合夹爪"},
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0, "desc": "接近交接点"},
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0, "desc":  "到达交接点"},
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1, "desc":  "打开夹爪"},
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1, "desc": "后退"},
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1, "desc": "保持"},
        ]
        
        self.box_xyz = box_xyz
        self.meet_xyz = meet_xyz
    
    def update_display(self):
        """更新显示"""
        left_wp = self.left_trajectory[self.current_left_idx]
        right_wp = self.right_trajectory[self.current_right_idx]
        
        # 更新信息文本
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        info_text = f"""
【左臂 路点 {self.current_left_idx + 1}/{len(self.left_trajectory)}】

时间步: t = {left_wp['t']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
左臂状态: 
  动作: {left_wp['desc']}
  位置: [{left_wp['xyz'][0]:.3f}, {left_wp['xyz'][1]:.3f}, {left_wp['xyz'][2]:.3f}]
  夹爪: {'打开' if left_wp['gripper'] > 0.5 else '闭合'}

【右臂 路点 {self.current_right_idx + 1}/{len(self.right_trajectory)}】

右臂状态:
  动作: {right_wp['desc']}
  位置: [{right_wp['xyz'][0]:.3f}, {right_wp['xyz'][1]:.3f}, {right_wp['xyz'][2]:.3f}]
  夹爪: {'打开' if right_wp['gripper'] > 0.5 else '闭合'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        
        font = FontProperties(family='WenQuanYi Micro Hei', size=11)
        self.ax_info.text(0.1, 0.5, info_text, fontproperties=font,
                        verticalalignment='center')

        # self.ax_info.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
        #                  verticalalignment='center')
        
        # 移动到路点
        self.move_to_waypoint(left_wp, right_wp)
        
        plt.draw()
    
    def move_to_waypoint(self, left_wp, right_wp, steps=30):
        """移动到路点"""
        current_left = self.env.physics.named.data.mocap_pos['mocap_left'].copy()
        current_right = self.env.physics.named.data.mocap_pos['mocap_right'].copy()
        
        for step in range(steps):
            frac = step / steps
            left_xyz = current_left + (left_wp['xyz'] - current_left) * frac
            right_xyz = current_right + (right_wp['xyz'] - current_right) * frac
            
            action = np.concatenate([
                left_xyz, left_wp['quat'], [left_wp['gripper']],
                right_xyz, right_wp['quat'], [right_wp['gripper']]
            ])
            
            self.ts = self.env.step(action)
            img = self.ts.observation['images']['angle']
            self.plt_img.set_data(img)
            plt.pause(0.01)
    
    def next_left_waypoint(self, event):
        """左臂下一个路点"""
        if self.current_left_idx < len(self.left_trajectory) - 1:
            self.current_left_idx += 1
            self.update_display()
        else:
            print("\n左臂已经是最后一个路点了！")
    
    def prev_left_waypoint(self, event):
        """左臂上一个路点"""
        if self.current_left_idx > 0:
            self.current_left_idx -= 1
            self.update_display()
        else:
            print("\n左臂已经是第一个路点了！")

    def next_right_waypoint(self, event):
        """右臂下一个路点"""
        if self.current_right_idx < len(self.right_trajectory) - 1:
            self.current_right_idx += 1
            self.update_display()
        else:
            print("\n右臂已经是最后一个路点了！")
    
    def prev_right_waypoint(self, event):
        """右臂上一个路点"""
        if self.current_right_idx > 0:
            self.current_right_idx -= 1
            self.update_display()
        else:
            print("\n右臂已经是第一个路点了！")
    
    def play_left_trajectory(self, event):
        """播放左臂轨迹"""
        print("\n播放左臂轨迹...")
        for idx in range(len(self.left_trajectory)):
            self.current_left_idx = idx
            self.update_display()
            plt.pause(0.5)
        print("左臂播放完成！")
    
    def play_right_trajectory(self, event):
        """播放右臂轨迹"""
        print("\n播放右臂轨迹...")
        for idx in range(len(self.right_trajectory)):
            self.current_right_idx = idx
            self.update_display()
            plt.pause(0.5)
        print("右臂播放完成！")

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("交互式轨迹可视化工具")
    print("=" * 80)
    print("\n使用说明：")
    print("  - 左臂/右臂按钮分别控制各自路点")
    print("  - 左臂/右臂播放按钮分别播放各自轨迹")
    print("  - 关闭窗口：退出程序")
    print("\n" + "=" * 80 + "\n")
    
    visualizer = ButtonControlledVisualizer('sim_transfer_cube')
    plt.show(block=True)