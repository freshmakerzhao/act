import time
import numpy as np
import mujoco
import mujoco.viewer

from ee_sim_env import make_ee_sim_env

# Create environment and reset
env = make_ee_sim_env("sim_lifting_cube", "vx300s_single")
ts = env.reset()

# Access MuJoCo model/data from dm_control physics
model = env.physics.model.ptr
data = env.physics.data.ptr

# Names of mocap bodies in this environment
LEFT_MOCAP = "mocap_left"
RIGHT_MOCAP = "mocap_right"

# Fixed gripper state (open)
GRIPPER_OPEN = 1.0

# Launch passive MuJoCo viewer (dragging works on mocap bodies)
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Read current mocap pose (may be updated by mouse dragging)
        left_xyz = env.physics.named.data.mocap_pos[LEFT_MOCAP].copy()
        right_xyz = env.physics.named.data.mocap_pos[RIGHT_MOCAP].copy()
        left_quat = env.physics.named.data.mocap_quat[LEFT_MOCAP].copy()
        right_quat = env.physics.named.data.mocap_quat[RIGHT_MOCAP].copy()

        # Build action: left pose + gripper, right pose + gripper
        action = np.concatenate([
            left_xyz, left_quat, [GRIPPER_OPEN],
            right_xyz, right_quat, [GRIPPER_OPEN],
        ])

        # Step environment to apply mocap targets
        ts = env.step(action)

        # Sync viewer and pace the loop
        viewer.sync()
        time.sleep(0.01)