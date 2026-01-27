# interactive_waypoint_designer_improved.py

from ee_sim_env import make_ee_sim_env
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
from pyquaternion import Quaternion

env = make_ee_sim_env('sim_transfer_cube')
ts = env.reset()

# Get initial state
box_xyz = ts.observation['env_state'][:3]
init_right = ts.observation['mocap_pose_right']
init_left = ts.observation['mocap_pose_left']
print(f"Box position: {box_xyz}")
# Create adjustable target point
target_offset = np.array([0, 0, 0.08])

# Create interface with better layout
fig = plt.figure(figsize=(16, 8))

# Left: MuJoCo view (larger)
ax_img = plt.subplot(1, 2, 1)
ax_img.set_position([0.05, 0.1, 0.5, 0.85])  # [left, bottom, width, height]

# Right: Control panel
ax_info = plt.subplot(1, 2, 2)
ax_info.set_position([0.60, 0.55, 0.35, 0.40])
ax_info.axis('off')

# Sliders with better spacing
slider_height = 0.03
slider_gap = 0.055
slider_left = 0.62
slider_width = 0.32
slider_bottom_start = 0.22

ax_slider_x = plt.axes([slider_left, slider_bottom_start + 6*slider_gap, slider_width, slider_height])
ax_slider_y = plt.axes([slider_left, slider_bottom_start + 5*slider_gap, slider_width, slider_height])
ax_slider_z = plt.axes([slider_left, slider_bottom_start + 4*slider_gap, slider_width, slider_height])
ax_slider_qw = plt.axes([slider_left, slider_bottom_start + 3*slider_gap, slider_width, slider_height])
ax_slider_qx = plt.axes([slider_left, slider_bottom_start + 2*slider_gap, slider_width, slider_height])
ax_slider_qy = plt.axes([slider_left, slider_bottom_start + slider_gap, slider_width, slider_height])
ax_slider_qz = plt.axes([slider_left, slider_bottom_start, slider_width, slider_height])

slider_x = Slider(ax_slider_x, 'Offset X (m)', -0.2, 0.2, valinit=0, valstep=0.001)
slider_y = Slider(ax_slider_y, 'Offset Y (m)', -0.2, 0.2, valinit=0, valstep=0.001)
slider_z = Slider(ax_slider_z, 'Offset Z (m)', -0.1, 0.2, valinit=0.08, valstep=0.001)
slider_qw = Slider(ax_slider_qw, 'Quat w', -1.0, 1.0, valinit=init_right[3], valstep=0.001)
slider_qx = Slider(ax_slider_qx, 'Quat x', -1.0, 1.0, valinit=init_right[4], valstep=0.001)
slider_qy = Slider(ax_slider_qy, 'Quat y', -1.0, 1.0, valinit=init_right[5], valstep=0.001)
slider_qz = Slider(ax_slider_qz, 'Quat z', -1.0, 1.0, valinit=init_right[6], valstep=0.001)

# Buttons with better spacing
button_width = 0.15
button_height = 0.05
button_left = 0.62
button_gap = 0.02

ax_btn_save = plt.axes([button_left, 0.15, button_width, button_height])
ax_btn_reset = plt.axes([button_left + button_width + button_gap, 0.15, button_width, button_height])
ax_btn_export = plt.axes([button_left, 0.08, button_width + button_width + button_gap, button_height])

btn_save = Button(ax_btn_save, 'Save Waypoint', color='lightblue', hovercolor='skyblue')
btn_reset = Button(ax_btn_reset, 'Reset', color='lightcoral', hovercolor='salmon')
btn_export = Button(ax_btn_export, 'Export Python Code', color='lightgreen', hovercolor='lime')

waypoints = []

def update(val):
    offset = np.array([slider_x.val, slider_y.val, slider_z.val])
    target = box_xyz + offset

    quat_raw = np.array([slider_qw.val, slider_qx.val, slider_qy.val, slider_qz.val])
    quat_norm = np.linalg.norm(quat_raw)
    target_quat = (quat_raw / quat_norm) if quat_norm > 1e-8 else init_right[3:].copy()
    
    # Update robot position
    action = np.concatenate([
        init_left, [1],
        target, target_quat, [1]
    ])
    new_ts = env.step(action)
    
    # Display image
    img = new_ts.observation['images']['angle']
    ax_img.clear()
    ax_img.imshow(img)
    ax_img.set_title(f"Target: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]", 
                     fontsize=14, pad=10)
    ax_img.axis('off')
    
    # Display info with larger font
    ax_info.clear()
    ax_info.axis('off')
    
    distance_to_box = np.linalg.norm(target - box_xyz)
    
    # 机械臂当前目标位置和基于box的偏移信息
    info_text1 = f"""
CURRENT TARGET
  X: {target[0]:7.3f} m
  Y: {target[1]:7.3f} m
  Z: {target[2]:7.3f} m

OFFSET FROM BOX
  dX: {offset[0]:+7.3f} m
  dY: {offset[1]:+7.3f} m
  dZ: {offset[2]:+7.3f} m

DISTANCE: {distance_to_box:.3f} m

SAVED WAYPOINTS: {len(waypoints)}
"""
    
    info_text2 = f"""
QUAT (w, x, y, z)
    {target_quat[0]:+7.4f}  {target_quat[1]:+7.4f}
    {target_quat[2]:+7.4f}  {target_quat[3]:+7.4f}
"""

    info_text3 = f"""
BOX POSITION
  X: {box_xyz[0]:7.3f} m
  Y: {box_xyz[1]:7.3f} m
  Z: {box_xyz[2]:7.3f} m
"""

    ax_info.text(-0.05, 0.95, info_text1, 
                fontsize=11,
                family='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax_info.text(0.30, 0.95, info_text2, 
                fontsize=11,
                family='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax_info.text(0.70, 0.95, info_text3, 
                fontsize=11,
                family='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    fig.canvas.draw_idle()

def save_waypoint(event):
    offset = np.array([slider_x.val, slider_y.val, slider_z.val])
    target = box_xyz + offset
    quat_raw = np.array([slider_qw.val, slider_qx.val, slider_qy.val, slider_qz.val])
    quat_norm = np.linalg.norm(quat_raw)
    target_quat = (quat_raw / quat_norm) if quat_norm > 1e-8 else init_right[3:].copy()
    
    wp = {
        'xyz': target.copy(),
        'offset': offset.copy(),
        'quat': target_quat.copy(),
        'gripper': 1,
        't': len(waypoints) * 50  # Auto-assign time
    }
    waypoints.append(wp)
    
    print(f"✓ Waypoint {len(waypoints)} saved:")
    print(f"    t={wp['t']}, xyz={wp['xyz']}, offset={wp['offset']}, quat={wp['quat']}")
    
    update(None)

def reset(event):
    slider_x.reset()
    slider_y.reset()
    slider_z.reset()
    slider_qw.reset()
    slider_qx.reset()
    slider_qy.reset()
    slider_qz.reset()
    update(None)

def export_code(event):
    if len(waypoints) == 0:
        print("No waypoints to export!")
        return
    
    print("\n" + "="*70)
    print("GENERATED PYTHON CODE")
    print("="*70)
    print("\nself.right_trajectory = [")
    
    for i, wp in enumerate(waypoints):
        gripper_state = wp['gripper']
        comment = f"waypoint {i+1}"
        print(f'    {{"t": {wp["t"]}, '
            f'"xyz": box_xyz + np.array([{wp["offset"][0]:.3f}, {wp["offset"][1]:.3f}, {wp["offset"][2]:.3f}]), '
            f'"quat": np.array([{wp["quat"][0]:.6f}, {wp["quat"][1]:.6f}, {wp["quat"][2]:.6f}, {wp["quat"][3]:.6f}]), '
            f'"gripper": {gripper_state}, "desc": "{comment}"}},')
    
    print("]\n")
    print("="*70)
    print(f"Total waypoints: {len(waypoints)}")
    print("="*70 + "\n")

# Bind events
slider_x.on_changed(update)
slider_y.on_changed(update)
slider_z.on_changed(update)
slider_qw.on_changed(update)
slider_qx.on_changed(update)
slider_qy.on_changed(update)
slider_qz.on_changed(update)
btn_save.on_clicked(save_waypoint)
btn_reset.on_clicked(reset)
btn_export.on_clicked(export_code)

# Initial display
update(None)

print("="*70)
print("INTERACTIVE WAYPOINT DESIGNER")
print("="*70)
print("\nInstructions:")
print("  1. Adjust sliders to move the robot arm")
print("  2. Click 'Save Waypoint' when position is good")
print("  3. Repeat for all waypoints")
print("  4. Click 'Export Python Code' to generate trajectory")
print("\n" + "="*70 + "\n")

plt.show()