import os
import time
import json
import numpy as np
import mujoco
import mujoco.viewer

# ================== 你需要改的部分 ==================
XML_PATH = r"D:\code\mujoco_xml\xml\scene_3.xml"
SAVE_DIR = r"D:\code\mujoco_xml\dataset\scripted"
EPISODES = 1
MAX_STEPS = 1200
RENDER = True
# ====================================================

# ---- 关节/控制器/对齐点 ----
ACT_NAMES   = ["act_base", "act_boom", "act_stick", "act_bucket"]
JOINT_NAMES = ["base_to_cheshen", "cheshen_to_03", "03_to_04", "04_to_chandou"]
ALIGN_SITE  = "stick_ref"

# ---- 土堆目标点：写死 ----
PILE_TARGET = np.array([6.916, 0.0, 1.5], dtype=float)

# ---- IK 对齐参数 ----
ALIGN_TOL_XY = 0.01     # m
IK_GAIN = 1.2
DAMPING = 0.10
MAX_DQ = 0.12           # rad per step
HOLD_STEPS = 20
ALIGN_MAX_STEPS = 250

# ---- “纯关节脚本”参数：用相对轨迹 Δq(t) ----
# 这些是总幅值（不是每步增量），会自动线性插值成轨迹
INSERT_STEPS = 80
CURL_STEPS   = 80
LIFT_STEPS   = 60
RESET_STEPS  = 90

# 插入：boom 下压、stick 伸出（相对 q_align）
BOOM_DOWN_TOTAL = -0.35     # rad（太大可能卡住，建议 -0.2 ~ -0.5 之间试）
STICK_OUT_TOTAL = +0.35     # rad（建议 0.2 ~ 0.6）
# 卷斗：bucket 卷、stick 回收一点、boom 微抬一点（相对插入末端）
BUCKET_CURL_TOTAL = +0.55   # rad（注意别超过你的 bucket range 上限 0.6）
STICK_IN_BACK = -0.30       # rad
BOOM_LIFT_DURING_CURL = +0.08  # rad
# 抬起：boom 抬、stick 回到对齐姿态附近，bucket 保持卷起
BOOM_LIFT_TOTAL = +0.25     # rad


def clamp_norm(v, max_norm):
    n = float(np.linalg.norm(v))
    if n > max_norm:
        return v * (max_norm / (n + 1e-9))
    return v


def linspace_delta(n, start, end):
    if n <= 0:
        return np.zeros((0,), dtype=float)
    return np.linspace(start, end, n, dtype=float)


def build_dig_delta_traj():
    """
    生成一条相对轨迹 delta_traj: (T,4) 对应 [base, boom, stick, bucket]
    后续用：q_target = q_align + delta_traj[t]
    """
    T = INSERT_STEPS + CURL_STEPS + LIFT_STEPS
    delta = np.zeros((T, 4), dtype=float)

    # 1) INSERT：boom down + stick out，bucket保持
    boom_ins = linspace_delta(INSERT_STEPS, 0.0, BOOM_DOWN_TOTAL)
    stick_ins = linspace_delta(INSERT_STEPS, 0.0, STICK_OUT_TOTAL)
    delta[:INSERT_STEPS, 1] = boom_ins
    delta[:INSERT_STEPS, 2] = stick_ins

    # 2) CURL：从插入末端开始，stick 回收一些 + bucket 卷斗 + boom 微抬
    s0 = INSERT_STEPS
    s1 = INSERT_STEPS + CURL_STEPS
    boom_curl = linspace_delta(CURL_STEPS, BOOM_DOWN_TOTAL, BOOM_DOWN_TOTAL + BOOM_LIFT_DURING_CURL)
    stick_curl = linspace_delta(CURL_STEPS, STICK_OUT_TOTAL, STICK_OUT_TOTAL + STICK_IN_BACK)
    bucket_c = linspace_delta(CURL_STEPS, 0.0, BUCKET_CURL_TOTAL)
    delta[s0:s1, 1] = boom_curl
    delta[s0:s1, 2] = stick_curl
    delta[s0:s1, 3] = bucket_c

    # 3) LIFT：boom 抬起更多 + stick 回到 0（对齐姿态）+ bucket 保持卷起
    s2 = s1
    s3 = s1 + LIFT_STEPS
    boom_l = linspace_delta(LIFT_STEPS, delta[s1-1, 1], delta[s1-1, 1] + BOOM_LIFT_TOTAL)
    stick_l = linspace_delta(LIFT_STEPS, delta[s1-1, 2], 0.0)
    bucket_l = np.full((LIFT_STEPS,), delta[s1-1, 3], dtype=float)
    delta[s2:s3, 1] = boom_l
    delta[s2:s3, 2] = stick_l
    delta[s2:s3, 3] = bucket_l

    return delta


class ExcavatorEnv:
    """
    action: q_target(4,) -> position actuator ctrl
    obs: qpos4, stick_ref world pos, pile_target, error_align
    """
    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)

        self.act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACT_NAMES]

        self.qpos_addrs = []
        self.dof_addrs = []
        for jn in JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            self.qpos_addrs.append(int(self.model.jnt_qposadr[jid]))
            self.dof_addrs.append(int(self.model.jnt_dofadr[jid]))

        self.align_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ALIGN_SITE)

        self.jacp = np.zeros((3, self.model.nv), dtype=float)
        self.jacr = np.zeros((3, self.model.nv), dtype=float)

        mujoco.mj_forward(self.model, self.data)
        sid = self.align_site_id
        bid = int(self.model.site_bodyid[sid])
        print("[CHECK] stick_ref site body =", mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid))
        print("[CHECK] stick_ref site_xpos =", self.data.site_xpos[sid].copy())
        J4 = self.jacobian_site_pos_wrt_4joints()
        print("[CHECK] ||J4||_F =", float(np.linalg.norm(J4)))
        print("[CHECK] J4 =\n", J4)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def get_qpos4(self):
        return np.array([self.data.qpos[a] for a in self.qpos_addrs], dtype=float)

    def set_ctrl_qtarget(self, q_target4):
        for i, aid in enumerate(self.act_ids):
            self.data.ctrl[aid] = float(q_target4[i])

    def get_obs(self):
        tip = self.data.site_xpos[self.align_site_id].copy()
        tgt = PILE_TARGET.copy()
        q4  = self.get_qpos4()
        return {
            "qpos4": q4,
            "stick_ref": tip,
            "pile_target": tgt,
            "error_align": (tgt - tip),
        }

    def step(self, q_target4, n_substeps=10):
        self.set_ctrl_qtarget(q_target4)
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)
        return self.get_obs()

    def jacobian_site_pos_wrt_4joints(self):
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.align_site_id)
        cols = self.dof_addrs
        return self.jacp[:, cols].copy()


class ScriptedPolicy:
    """
    HOLD -> ALIGN_XY(用IK) -> DIG_TRAJ(纯关节相对轨迹) -> RESET -> HOLD
    """
    def __init__(self, env: ExcavatorEnv):
        self.env = env
        self.phase = "HOLD"
        self.t = 0

        self.bucket_hold = None
        self.q_home = None

        self.q_align = None
        self.delta_traj = None  # (T,4)

    def _ik_align_xy_step(self, obs):
        q = obs["qpos4"].copy()

        if self.bucket_hold is None:
            self.bucket_hold = float(q[3])
        q[3] = self.bucket_hold  # bucket锁住

        e = obs["error_align"].copy()
        e[2] = 0.0  # 只对齐xy
        err_xy = float(np.linalg.norm(e[:2]))

        J4 = self.env.jacobian_site_pos_wrt_4joints()  # (3,4)
        J = J4[:, :3]  # 只用 base/boom/stick

        JJt = J @ J.T
        A = JJt + (DAMPING ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(A, e)

        dq = IK_GAIN * dq
        dq = clamp_norm(dq, MAX_DQ)

        q[:3] += dq
        return q, err_xy

    def __call__(self, obs):
        q = obs["qpos4"].copy()

        if self.q_home is None:
            self.q_home = q.copy()

        if self.phase == "HOLD":
            if self.bucket_hold is None:
                self.bucket_hold = float(q[3])
            q[3] = self.bucket_hold
            self.t += 1
            if self.t >= HOLD_STEPS:
                self.phase = "ALIGN_XY"
                self.t = 0
            return q, {"phase": self.phase}

        if self.phase == "ALIGN_XY":
            q_next, err_xy = self._ik_align_xy_step(obs)
            self.t += 1

            if (err_xy < ALIGN_TOL_XY) or (self.t >= ALIGN_MAX_STEPS):
                # ✅ 记录对齐姿态：后续纯关节相对轨迹都从这里开始
                self.q_align = q_next.copy()
                self.bucket_hold = float(self.q_align[3])

                # ✅ 生成相对轨迹 Δq(t)
                self.delta_traj = build_dig_delta_traj()

                self.phase = "DIG_TRAJ"
                self.t = 0

            return q_next, {"phase": self.phase, "err_xy": err_xy}

        if self.phase == "DIG_TRAJ":
            i = self.t
            if i >= len(self.delta_traj):
                self.phase = "RESET"
                self.t = 0
                return q, {"phase": self.phase}

            q_target = self.q_align + self.delta_traj[i]
            # base 默认不动（delta_traj[:,0]=0），bucket 相对卷斗
            q_target[3] = self.bucket_hold + self.delta_traj[i, 3]

            self.t += 1
            return q_target, {"phase": self.phase, "traj_i": i}

        if self.phase == "RESET":
            alpha = min(1.0, (self.t + 1) / float(RESET_STEPS))
            q_reset = (1 - alpha) * obs["qpos4"] + alpha * self.q_home
            self.t += 1
            if alpha >= 1.0 - 1e-6:
                self.phase = "HOLD"
                self.t = 0
                self.bucket_hold = None
                self.q_home = None
                self.q_align = None
                self.delta_traj = None
            return q_reset, {"phase": self.phase}

        return q, {"phase": self.phase}


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def run_collect():
    ensure_dir(SAVE_DIR)
    env = ExcavatorEnv(XML_PATH)

    for ep in range(EPISODES):
        obs = env.reset()
        policy = ScriptedPolicy(env)

        episode = {
            "meta": {
                "xml_path": XML_PATH,
                "pile_target": PILE_TARGET.tolist(),
                "dt": env.dt,
                "episode_id": ep,
                "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "steps": []
        }

        viewer = mujoco.viewer.launch_passive(env.model, env.data) if RENDER else None

        for step in range(MAX_STEPS):
            q_target, info = policy(obs)
            obs_next = env.step(q_target, n_substeps=10)

            if step % 20 == 0:
                e = obs["error_align"]
                print(f"[STEP {step:04d}] phase={info.get('phase')} "
                      f"q={np.round(obs['qpos4'],3)} "
                      f"err_xy={np.linalg.norm(e[:2]):.3f} err_z={e[2]:.3f}")

            episode["steps"].append({
                "qpos4": obs["qpos4"].tolist(),
                "q_target4": np.array(q_target, dtype=float).tolist(),
                "stick_ref": obs["stick_ref"].tolist(),
                "pile_target": obs["pile_target"].tolist(),
                "error_align": obs["error_align"].tolist(),
                "info": info,
            })

            obs = obs_next

            if viewer is not None:
                viewer.sync()
                time.sleep(0.05)
                if not viewer.is_running():
                    break

        if viewer is not None:
            print("[INFO] Episode finished. Close the viewer window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.01)
            viewer.close()

        out_path = os.path.join(SAVE_DIR, f"episode_{ep:04d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(episode, f, ensure_ascii=False, indent=2)

        print(f"[OK] saved: {out_path}  steps={len(episode['steps'])}")


if __name__ == "__main__":
    run_collect()
