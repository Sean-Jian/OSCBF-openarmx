"""
VR 遥操 + OSCBF 安全控制 - OpenArmX 右臂
基于 oscbf_openarmx_ee_cbf.py 的 CBF 逻辑 + piper VR 遥操的控制框架

控制逻辑：
- 右手 grip 按住：激活控制，手柄相对移动映射到右臂末端位姿
- 右手 A 键：右臂回到 L 构型
- 末端位姿控制：操作空间 PD → J^T → OSCBF-QP → 关节力矩
- 安全约束：小球避碰 + 本体避碰

坐标系说明：
- VR 坐标系 → 右臂 baselink 坐标系（不是世界坐标系）
- 相对移动：手柄从 T0 到 T1，右臂末端从 A0 到 A1
- A0/A1 均在右臂 baselink 坐标系下表示
"""
import os
import sys
import asyncio
import threading
import time
import warnings
warnings.filterwarnings("ignore", message=".*fixed link.*")

import numpy as np
from scipy.linalg import inv
from scipy.spatial.transform import Rotation as R

import mujoco
import mujoco_viewer
import fcl
import transforms3d as tf

# VR 服务器（复用 telegrip 的 WebSocket 服务）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, os.path.join(project_root, 'telegrip'))
from telegrip.config import TelegripConfig
from telegrip.inputs.vr_ws_server import VRWebSocketServer

MODEL_PATH = os.path.join(os.path.dirname(__file__), "openarmx_mujoco.xml")


# ============================================================
# CBF 控制器（直接复用 ee_cbf 的核心逻辑，不修改原文件）
# ============================================================
class VROSCBFController(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 1.8, azimuth=180, elevation=-30)
        self.path = path

        # ---- L 构型 ----
        self.L_pose = np.array([0.0, 0.0, 0.0, np.pi/2, 0.0, 0.0, 0.0])

        # ---- 右臂目标末端位姿（世界坐标系）----
        self.right_target_ee_pos = np.zeros(3)
        self.right_target_ee_orientation = np.eye(3)
        self.right_ik_joint_angles = self.L_pose.copy()  # 零空间目标

        # ---- 小球碰撞参数 ----
        self.ball_radius     = 0.04
        self.safety_distance = 0.05
        self.right_robot_collision_bodies = [
            "openarmx_right_link1", "openarmx_right_link2", "openarmx_right_link3",
            "openarmx_right_link4", "openarmx_right_link5", "openarmx_right_link6",
            "openarmx_right_link7",
        ]
        self.right_robot_collision_radii = [0.07, 0.06, 0.07, 0.06, 0.05, 0.05, 0.05]
        self.right_robot_collision_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in self.right_robot_collision_bodies
        ]
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_ball")

        # ---- 本体碰撞 FCL ----
        self.right_robot_collision_geoms = [
            "openarmx_right_link3_collision",
            "openarmx_right_link4_collision",
            "openarmx_right_link7_collision",
        ]
        self.body_collision_geoms = ["robot_body_collision"]
        self.right_fcl_robot_objects = self._init_fcl_objects(self.right_robot_collision_geoms)
        self.fcl_body_objects        = self._init_fcl_objects(self.body_collision_geoms)

        # ---- 本体 CBF 参数 ----
        self.body_box_center      = np.array([0.0, 0.0, 0.34])
        self.body_box_half        = np.array([0.10, 0.03, 0.32])
        self.safety_distance_body = 0.05
        self.alpha_body           = 300.0
        self.alpha2_body          = 80.0
        self.w_body               = 10.0

        # ---- 任务空间限制 ----
        self.task_space_limits = {
            'x': {'min': -0.5, 'max': 1.5},
            'y': {'min': -1.5, 'max': 1.5},
            'z': {'min':  0.3, 'max': 1.5},
        }
        self.alpha_task  = 1.0
        self.alpha2_task = 1.0

        # ---- 关节/执行器索引 ----
        self.right_controlled_joints = [f"openarmx_right_joint{i}" for i in range(1, 8)]
        self.right_joint_qpos_indices = []
        self.right_dof_indices        = []
        self.right_actuator_indices   = []
        self._init_joint_indices()

        # ---- 控制参数（VR 遥操需要快速响应，Kp 调大）----
        self.Kp_pos = np.diag([1200]*3)  # 位置刚度（大 → 跟随快）
        self.Kd_pos = np.diag([80]*3)    # 位置阻尼
        self.Kp_ori = np.diag([80]*3)    # 姿态刚度
        self.Kd_ori = np.diag([18]*3)    # 姿态阻尼
        self.Kpj    = 0    # 零空间关节刚度（0=纯笛卡尔控制）
        self.Kdj    = 5    # 零空间阻尼

        self.index = 0
        self._reset_to_L = False  # A 键重置标志位（由 VR 线程设置，主线程处理）
        print(f"FCL 右臂胶囊体: {len(self.right_fcl_robot_objects)}/3  本体: {len(self.fcl_body_objects)}/1")

    # ------------------------------------------------------------------ helpers
    def _init_joint_indices(self):
        for jname in self.right_controlled_joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                self.right_joint_qpos_indices.append(self.model.jnt_qposadr[jid])
                self.right_dof_indices.append(self.model.jnt_dofadr[jid])
        for i in range(1, 8):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"right_motor_{i}")
            if aid != -1:
                self.right_actuator_indices.append(aid)

    def _init_fcl_objects(self, geom_names):
        fcl_objects = []
        for name in geom_names:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id == -1:
                print(f"警告：未找到 geom {name}")
                continue
            geom_type = self.model.geom_type[geom_id]
            size = self.model.geom_size[geom_id].copy()
            pos  = self.data.geom_xpos[geom_id].copy()
            mat  = self.data.geom_xmat[geom_id].copy()
            rot_mat = mat.reshape(3, 3)
            quat_xyzw = tf.quaternions.mat2quat(rot_mat)
            fcl_quat = np.array([quat_xyzw[1], quat_xyzw[2], quat_xyzw[3], quat_xyzw[0]])
            if   geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:   fcl_geom = fcl.Sphere(size[0])
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:      fcl_geom = fcl.Box(size[0]*2, size[1]*2, size[2]*2)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:  fcl_geom = fcl.Capsule(size[0], size[1]*2)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER: fcl_geom = fcl.Cylinder(size[0], size[1]*2)
            else: continue
            fcl_objects.append(fcl.CollisionObject(fcl_geom, fcl.Transform(fcl_quat, pos)))
        return fcl_objects

    def _update_fcl_poses(self, geom_names, fcl_objects):
        for i, name in enumerate(geom_names):
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id == -1 or i >= len(fcl_objects): continue
            pos = self.data.geom_xpos[geom_id].copy()
            mat = self.data.geom_xmat[geom_id].copy()
            rot_mat = mat.reshape(3, 3)
            quat_xyzw = tf.quaternions.mat2quat(rot_mat)
            fcl_quat = np.array([quat_xyzw[1], quat_xyzw[2], quat_xyzw[3], quat_xyzw[0]])
            fcl_objects[i].setTransform(fcl.Transform(fcl_quat, pos))

    def get_current_qpos(self):
        return np.array([self.data.qpos[i] for i in self.right_joint_qpos_indices])

    def get_current_qvel(self):
        return np.array([self.data.qvel[i] for i in self.right_dof_indices])

    def get_ee_pos(self):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarmx_right_link7")
        return self.data.body(bid).xpos.copy() if bid != -1 else None

    def get_ee_ori(self):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarmx_right_link7")
        return self.data.body(bid).xmat.reshape(3,3).copy() if bid != -1 else None

    def get_ball_pos(self):
        return self.data.body(self.ball_body_id).xpos.copy()

    def compute_orientation_error(self, R_cur, R_tgt):
        R_err = R_tgt @ R_cur.T
        e = 0.5 * (R_err - R_err.T)
        return np.array([e[2,1], e[0,2], e[1,0]])

    def compute_damped_pseudoinverse(self, J, lam=0.2):
        JT = J.T
        return JT @ np.linalg.inv(J @ JT + lam**2 * np.eye(J.shape[0]))

    # ------------------------------------------------------------------ body CBF
    def compute_body_distance(self):
        self._update_fcl_poses(self.right_robot_collision_geoms, self.right_fcl_robot_objects)
        self._update_fcl_poses(self.body_collision_geoms, self.fcl_body_objects)
        if not self.right_fcl_robot_objects or not self.fcl_body_objects:
            return 1.0, self.right_robot_collision_ids[2]
        min_distances = []
        for ro in self.right_fcl_robot_objects:
            for bo in self.fcl_body_objects:
                req = fcl.DistanceRequest(); res = fcl.DistanceResult()
                fcl.distance(ro, bo, req, res)
                min_distances.append(res.min_distance)
        min_idx = int(np.argmin(min_distances))
        n_body  = len(self.fcl_body_objects)
        robot_geom_name = self.right_robot_collision_geoms[np.clip(min_idx // n_body, 0, len(self.right_fcl_robot_objects)-1)]
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom_name)
        min_bid = self.model.geom_bodyid[geom_id] if geom_id != -1 else 0
        return min_distances[min_idx], min_bid

    # ------------------------------------------------------------------ QP
    def solve_qp(self, P, q_vec, Lg_obs, rhs_obs, Lg_task, rhs_task, Lg_body, rhs_body, tau_min, tau_max):
        try:
            gamma = -np.linalg.solve(P + 1e-6*np.eye(7), q_vec)
        except Exception:
            gamma = -q_vec / (np.diag(P) + 1e-6)
        # 小球 + 任务空间约束
        for _ in range(50):
            obs_viol  = rhs_obs  - Lg_obs  @ gamma
            step = np.zeros(7)
            if np.any(obs_viol > 1e-4):
                g = Lg_obs[np.argmax(obs_viol)]
                step += (obs_viol.max() / (g @ g + 1e-6)) * g
            gamma += step
            task_viol = rhs_task - Lg_task @ gamma
            if np.any(task_viol > 0):
                g = Lg_task[np.argmax(task_viol)]
                gamma += (task_viol[np.argmax(task_viol)] / (g @ g + 1e-6)) * g
            if np.all(obs_viol <= 0) and np.all(task_viol <= 0):
                break
        gamma = np.clip(gamma, tau_min, tau_max)
        # 本体约束（clip 后强制执行）
        for _ in range(100):
            body_viol = rhs_body - Lg_body @ gamma
            if np.all(body_viol <= 1e-6): break
            worst = np.argmax(body_viol)
            g = Lg_body[worst]
            g_norm2 = g @ g + 1e-8
            if g_norm2 < 1e-10: break
            gamma += (body_viol[worst] / g_norm2) * self.w_body * g
        return gamma

    # ------------------------------------------------------------------ 主控制
    def compute_torque(self):
        q    = self.get_current_qpos()
        qdot = self.get_current_qvel()
        ee_pos = self.get_ee_pos()
        ee_ori = self.get_ee_ori()
        if ee_pos is None: return np.zeros(7)

        jidx = self.right_dof_indices
        ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarmx_right_link7")
        jac_pos_full = np.zeros((3, self.model.nv))
        jac_rot_full = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jac_pos_full, jac_rot_full, ee_body_id)
        J_pos = jac_pos_full[:, jidx]
        J_ori = jac_rot_full[:, jidx]
        J_full = np.vstack([J_pos, J_ori])
        J_pinv = self.compute_damped_pseudoinverse(J_full)
        N_q    = np.eye(7) - J_pinv @ J_full

        # 名义力矩（操作空间 PD）
        pos_err = self.right_target_ee_pos - ee_pos
        ori_err = self.compute_orientation_error(ee_ori, self.right_target_ee_orientation)
        F_cart  = np.concatenate([
            self.Kp_pos @ pos_err - self.Kd_pos @ (J_pos @ qdot),
            self.Kp_ori @ ori_err - self.Kd_ori @ (J_ori @ qdot),
        ])
        Gamma_0   = -self.Kpj*(q - self.right_ik_joint_angles) - self.Kdj*qdot
        Gamma_nom = J_full.T @ F_cart + N_q @ Gamma_0

        # ---- 小球 CBF ----
        ball_pos = self.get_ball_pos()
        ball_vel = np.zeros(3)  # 小球固定
        h_obs = np.array([
            np.linalg.norm(self.data.body(bid).xpos - ball_pos) - r - self.ball_radius - self.safety_distance
            for bid, r in zip(self.right_robot_collision_ids, self.right_robot_collision_radii)
        ])
        Lf_h, Lg_obs_list = [], []
        for bid in self.right_robot_collision_ids:
            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jac, None, bid)
            bvel = jac @ self.data.qvel
            bpos = self.data.body(bid).xpos
            d = bpos - ball_pos; dn = d / (np.linalg.norm(d) + 1e-8)
            Lf_h.append(np.dot(bvel - ball_vel, dn))
            Jp = jac[:, jidx]
            Lg_obs_list.append(dn @ Jp)
        Lf_h   = np.array(Lf_h)
        Lg_obs = np.array(Lg_obs_list)
        h2_obs = Lf_h + 10.0 * h_obs
        rhs_obs = -10.0 * h2_obs

        # ---- 本体 CBF ----
        body_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot_body_collision_body")
        if body_body_id != -1:
            self.body_box_center = self.data.body(body_body_id).xpos.copy() + np.array([0,0,0.34])
        min_dist, min_bid = self.compute_body_distance()
        h_body = min_dist - self.safety_distance_body
        if h_body < -0.01:
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, min_bid)
            print(f"[Body CBF] ⚠️穿透 dist={min_dist:.4f} h={h_body:.4f} link={bname}")

        body_pos_vec = self.data.body(min_bid).xpos.copy()
        direction_norm = (body_pos_vec - self.body_box_center)
        direction_norm /= (np.linalg.norm(direction_norm) + 1e-10)
        # 用 FCL 最近点方向（更精确）
        if self.right_fcl_robot_objects and self.fcl_body_objects:
            best_d, best_res = np.inf, None
            for ro in self.right_fcl_robot_objects:
                for bo in self.fcl_body_objects:
                    req = fcl.DistanceRequest(); res = fcl.DistanceResult()
                    fcl.distance(ro, bo, req, res)
                    if res.min_distance < best_d:
                        best_d = res.min_distance; best_res = res
            if best_res and best_res.nearest_points[0] is not None and best_res.nearest_points[1] is not None:
                d_vec = best_res.nearest_points[0] - best_res.nearest_points[1]
                if np.linalg.norm(d_vec) > 1e-6:
                    direction_norm = d_vec / np.linalg.norm(d_vec)
        jac_b = np.zeros((3, self.model.nv)); jac_b_rot = np.zeros((3, self.model.nv))
        if self.right_fcl_robot_objects and self.fcl_body_objects:
            best_d2, best_res2 = np.inf, None
            for ro in self.right_fcl_robot_objects:
                for bo in self.fcl_body_objects:
                    req2 = fcl.DistanceRequest(); res2 = fcl.DistanceResult()
                    fcl.distance(ro, bo, req2, res2)
                    if res2.min_distance < best_d2:
                        best_d2 = res2.min_distance; best_res2 = res2
            if best_res2 and best_res2.nearest_points[0] is not None:
                mujoco.mj_jac(self.model, self.data, jac_b, jac_b_rot,
                               np.array(best_res2.nearest_points[0]), min_bid)
            else:
                mujoco.mj_jacBody(self.model, self.data, jac_b, None, min_bid)
        else:
            mujoco.mj_jacBody(self.model, self.data, jac_b, None, min_bid)
        bvel_b = jac_b @ self.data.qvel
        Lf_hb  = np.dot(bvel_b, direction_norm)
        Jp_b   = jac_b[:, jidx]
        Lg_body_vec = direction_norm @ Jp_b
        alpha_b  = self.alpha_body  * (3.0 if h_body < 0 else 1.0)
        alpha2_b = self.alpha2_body * (3.0 if h_body < 0 else 1.0)
        h2_body  = Lf_hb + alpha_b * h_body
        Lg_body  = np.array([Lg_body_vec])
        rhs_body = np.array([-alpha2_b * h2_body])

        # ---- 任务空间 CBF ----
        M_inv = np.eye(7); C = np.ones(7)
        h_task = np.array([
            ee_pos[0] - self.task_space_limits['x']['min'],
            self.task_space_limits['x']['max'] - ee_pos[0],
            ee_pos[1] - self.task_space_limits['y']['min'],
            self.task_space_limits['y']['max'] - ee_pos[1],
            ee_pos[2] - self.task_space_limits['z']['min'],
            self.task_space_limits['z']['max'] - ee_pos[2],
        ])
        grad_h = np.zeros((6, 14))
        grad_h[:, :7] = np.vstack([J_pos[0:1,:], -J_pos[0:1,:], J_pos[1:2,:], -J_pos[1:2,:], J_pos[2:3,:], -J_pos[2:3,:]])
        fz = np.concatenate([qdot, -M_inv @ C])
        Lf_ht   = grad_h @ fz
        h2_task = Lf_ht + self.alpha_task * h_task
        gLf = np.zeros((6, 14)); gLf[:, 7:] = grad_h[:, :7]
        gh2 = gLf + self.alpha_task * grad_h
        Lf_h2t = gh2 @ fz
        gz = np.zeros((14, 7)); gz[7:, :] = M_inv
        Lg_task  = gh2 @ gz
        rhs_task = -self.alpha2_task * h2_task - Lf_h2t

        # ---- QP ----
        Wj = np.diag([1]*7); Wo = 3*np.diag([1]*6); Ws = 0.8*np.eye(7)
        P_qp = N_q @ Wj @ N_q.T + J_full.T @ Wo @ J_full + Ws
        q_qp = -Gamma_nom @ P_qp
        tau_min = np.array([-80,-80,-60,-60,-20,-20,-20], dtype=float)
        tau_max = -tau_min
        return self.solve_qp(P_qp, q_qp, Lg_obs, rhs_obs, Lg_task, rhs_task, Lg_body, rhs_body, tau_min, tau_max)

    # ------------------------------------------------------------------ runBefore / runFunc
    def runBefore(self):
        # 设置 L 构型初始关节角
        for i, jname in enumerate(self.right_controlled_joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                self.data.qpos[self.model.jnt_qposadr[jid]] = self.L_pose[i]
        mujoco.mj_forward(self.model, self.data)

        # 初始目标 = 当前末端位姿（L 构型）
        ee_pos = self.get_ee_pos()
        ee_ori = self.get_ee_ori()
        self.right_target_ee_pos         = ee_pos.copy()
        self.right_target_ee_orientation = ee_ori.copy()
        self.right_ik_joint_angles       = self.L_pose.copy()

        self.total_time = 3600.0
        self.dt         = self.model.opt.timestep
        self.num_steps  = int(self.total_time / self.dt)
        self.index      = 0
        print(f"右臂 L 构型初始末端: {np.round(ee_pos, 3)}")
        print("等待 VR 连接... 按住 grip 激活控制，A 键回到 L 构型")

    def runFunc(self):
        if self.index >= self.num_steps:
            return

        # A 键重置（在主线程处理，线程安全）
        if self._reset_to_L:
            self._reset_to_L = False
            for i, jname in enumerate(self.right_controlled_joints):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid != -1:
                    self.data.qpos[self.model.jnt_qposadr[jid]] = self.L_pose[i]
            mujoco.mj_forward(self.model, self.data)
            ee = self.get_ee_pos(); ee_ori = self.get_ee_ori()
            if ee is not None:
                self.right_target_ee_pos         = ee.copy()
                self.right_target_ee_orientation = ee_ori.copy()
                self.right_ik_joint_angles       = self.L_pose.copy()
            print(f"已回到 L 构型，末端: {np.round(ee, 3)}")
        tau = self.compute_torque()
        for i, aid in enumerate(self.right_actuator_indices):
            self.data.ctrl[aid] = tau[i]
        if self.index % 500 == 0:
            ee = self.get_ee_pos()
            ball = self.get_ball_pos()
            dist = np.linalg.norm(ee - ball) if ee is not None else -1
            print(f"[{self.index}] 末端={np.round(ee,3)} 目标={np.round(self.right_target_ee_pos,3)} 距小球={dist:.3f}m")
        self.index += 1
        time.sleep(0.001)

    def runAfter(self):
        print("仿真结束")


# ============================================================
# VR 控制系统
# ============================================================
class VRControlSystem:
    """
    VR 手柄 → 右臂末端位姿目标 → OSCBF 控制器

    坐标系：
      VR 世界坐标系 --[T]--> 右臂 baselink 坐标系
      相对移动：A1 = A0 @ inv(T @ T0 @ m_offset) @ (T @ T1 @ m_offset)
      其中 A0/A1 在右臂 baselink 坐标系下
    """
    def __init__(self, controller: VROSCBFController, config: TelegripConfig):
        self.ctrl   = controller
        self.config = config

        # VR 服务器
        self._command_queue = asyncio.Queue()
        self.vr_server = VRWebSocketServer(self._command_queue, config)

        # HTTPS 服务器（提供网页界面给头显）
        self.https_server = None
        self.https_thread  = None

        # VR → 右臂 baselink 坐标变换矩阵（与 piper 代码一致，按需调整）
        self.T = np.array([[0, 0,-1, 0],
                           [0, 1, 0, 0],
                           [1, 0, 0, 0],
                           [0, 0, 0, 1]], dtype=float)

        # 右臂 baselink 在世界坐标系中的位姿（从 XML 读取）
        # openarmx_right_link1: pos=[0, -0.089, 0.698], quat=[0.707105, 0.707108, 0, 0]
        self._right_base_pos = np.array([0.0, -0.089, 0.698])
        def quat_wxyz_to_R(w, x, y, z):
            return np.array([
                [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
                [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
                [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
            ])
        self._R_base = quat_wxyz_to_R(0.707105, 0.707108, 0, 0)
        # 世界坐标系 → 右臂 baselink 坐标系
        self._T_world_to_base = np.eye(4)
        self._T_world_to_base[:3, :3] = self._R_base.T
        self._T_world_to_base[:3,  3] = -self._R_base.T @ self._right_base_pos

        # grip 状态
        self.grip_active      = False
        self.grip_origin_data = None   # {"T0": 4x4, "A0_base": 4x4}
        self.m_offset         = None   # 4x4
        self._cached_transform = None  # 预计算缓存

        # 低通滤波
        self.filter_alpha  = 0.7   # 滤波系数（越大越快但越抖）
        self._filtered_pos = None
        self._filtered_rot = None  # 四元数

        self.is_running = False

    def _quat_to_R(self, q):
        """q = [x,y,z,w]"""
        from scipy.spatial.transform import Rotation
        return Rotation.from_quat(q).as_matrix()

    def _get_ee_in_base(self):
        """获取右臂末端在 baselink 坐标系下的 4x4 位姿"""
        ee_pos = self.ctrl.get_ee_pos()
        ee_ori = self.ctrl.get_ee_ori()
        if ee_pos is None: return None
        T_world = np.eye(4)
        T_world[:3, :3] = ee_ori
        T_world[:3,  3] = ee_pos
        return self._T_world_to_base @ T_world

    def _base_to_world_pos(self, pos_base):
        """baselink 坐标系位置 → 世界坐标系位置"""
        return self._R_base @ pos_base + self._right_base_pos

    def _base_to_world_ori(self, R_base):
        """baselink 坐标系旋转 → 世界坐标系旋转"""
        return self._R_base @ R_base

    def on_grip_press(self, right_controller):
        """grip 按下：记录初始状态"""
        if right_controller.origin_position is None or right_controller.origin_quaternion is None:
            return
        # 当前末端在 baselink 坐标系下
        A0_base = self._get_ee_in_base()
        if A0_base is None: return

        # 构造 T0（VR 坐标系，单位米）
        T0_rot = self._quat_to_R(right_controller.origin_quaternion)
        T0_pos = np.array([
            right_controller.origin_position['x'],
            right_controller.origin_position['y'],
            right_controller.origin_position['z'],
        ])
        T0 = np.eye(4); T0[:3,:3] = T0_rot; T0[:3,3] = T0_pos

        # 偏移矩阵（旋转部分）
        m_offset = inv(self.T @ T0) @ A0_base
        m_offset[:3, 3] = 0  # 平移部分清零

        self.grip_origin_data  = {"T0": T0, "A0_base": A0_base}
        self.m_offset          = m_offset
        self._cached_transform = A0_base @ inv(self.T @ T0 @ m_offset)
        self.grip_active       = True
        self._filtered_pos     = None
        self._filtered_rot     = None
        print("grip 激活，开始控制右臂")

    def on_grip_release(self):
        self.grip_active = False
        print("grip 释放，停止控制")

    def on_a_button(self):
        """A 键：回到 L 构型"""
        print("A 键：右臂回到 L 构型")
        ee_pos = self.ctrl.get_ee_pos()
        ee_ori = self.ctrl.get_ee_ori()
        if ee_pos is not None:
            self.ctrl.right_target_ee_pos         = ee_pos.copy()
            self.ctrl.right_target_ee_orientation = ee_ori.copy()
        # 重置关节到 L 构型
        for i, jname in enumerate(self.ctrl.right_controlled_joints):
            jid = mujoco.mj_name2id(self.ctrl.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                self.ctrl.data.qpos[self.ctrl.model.jnt_qposadr[jid]] = self.ctrl.L_pose[i]
        mujoco.mj_forward(self.ctrl.model, self.ctrl.data)
        ee_pos2 = self.ctrl.get_ee_pos()
        ee_ori2 = self.ctrl.get_ee_ori()
        if ee_pos2 is not None:
            self.ctrl.right_target_ee_pos         = ee_pos2.copy()
            self.ctrl.right_target_ee_orientation = ee_ori2.copy()
        self.grip_active = False
        self.grip_origin_data = None
        print(f"已回到 L 构型，末端: {np.round(ee_pos2, 3)}")

    def update_target(self, right_controller):
        """根据手柄当前位姿更新右臂末端目标（在 baselink 坐标系下计算，再转世界坐标）"""
        if not self.grip_active or self.grip_origin_data is None or self.m_offset is None:
            return
        if right_controller.accumulated_rotation_quat is None:
            return

        # 构造当前 T1（VR 坐标系，单位米）
        T1_rot = self._quat_to_R(right_controller.accumulated_rotation_quat)
        rel = right_controller  # 相对位移已累积在 origin_position + relative_pos
        # 从 vr_server 获取相对位移
        rel_pos = getattr(self.vr_server, 'relative_pos_right', [0,0,0])
        T1_pos = np.array([
            right_controller.origin_position['x'] + rel_pos[0],
            right_controller.origin_position['y'] + rel_pos[1],
            right_controller.origin_position['z'] + rel_pos[2],
        ])
        T1 = np.eye(4); T1[:3,:3] = T1_rot; T1[:3,3] = T1_pos

        # 计算目标位姿（baselink 坐标系）
        A1_base = self._cached_transform @ (self.T @ T1 @ self.m_offset)

        # 低通滤波
        pos_new = A1_base[:3, 3]
        rot_new = A1_base[:3, :3]
        if self._filtered_pos is None:
            self._filtered_pos = pos_new.copy()
            self._filtered_rot = rot_new.copy()
        else:
            self._filtered_pos = self.filter_alpha * pos_new + (1 - self.filter_alpha) * self._filtered_pos
            self._filtered_rot = rot_new  # 旋转直接用（可改为 slerp）

        # 转换到世界坐标系，更新控制器目标
        self.ctrl.right_target_ee_pos         = self._base_to_world_pos(self._filtered_pos)
        self.ctrl.right_target_ee_orientation = self._base_to_world_ori(self._filtered_rot)

    async def _vr_loop(self):
        """VR 数据处理主循环"""
        prev_grip = False
        while self.is_running:
            rc = self.vr_server.right_controller

            # 监听 command_queue（A 键通过 handle_a_button_press 放入队列）
            try:
                while True:
                    cmd = self._command_queue.get_nowait()
                    if cmd.metadata and cmd.metadata.get("action") == "return_to_home":
                        # 设置标志位，让主线程 runFunc 处理（线程安全）
                        self.ctrl._reset_to_L = True
                        self.grip_active = False
                        print("A 键触发，等待主线程回到 L 构型")
                        break
            except Exception:
                pass

            # grip 状态变化
            curr_grip = rc.grip_active
            if curr_grip and not prev_grip:
                self.on_grip_press(rc)
            elif not curr_grip and prev_grip:
                self.on_grip_release()
            prev_grip = curr_grip

            # 更新目标
            if self.grip_active:
                self.update_target(rc)

            await asyncio.sleep(0.002)  # 500 Hz

    async def run(self):
        import http.server, ssl, json
        self.is_running = True

        # 启动 HTTPS 服务器（复用 piper 代码的完整 APIHandler）
        try:
            import http.server, ssl as _ssl
            from telegrip.utils import get_absolute_path

            vr_sys_ref = self  # 闭包引用

            class _APIHandler(http.server.BaseHTTPRequestHandler):
                def log_message(self, *a): pass
                def end_headers(self):
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                    self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                    try: super().end_headers()
                    except: pass
                def do_OPTIONS(self):
                    self.send_response(200); self.end_headers()
                def _serve(self, path, ctype):
                    try:
                        abs_path = get_absolute_path(path)
                        with open(abs_path, 'rb') as f: data = f.read()
                        self.send_response(200)
                        self.send_header('Content-Type', ctype)
                        self.end_headers(); self.wfile.write(data)
                    except FileNotFoundError: self.send_error(404)
                    except Exception: pass
                def do_GET(self):
                    p = self.path
                    if p in ('/', '/index.html'):    self._serve('web-ui/index.html', 'text/html')
                    elif p.endswith('.css'):          self._serve(f'web-ui{p}', 'text/css')
                    elif p.endswith('.js'):           self._serve(f'web-ui{p}', 'application/javascript')
                    elif p.endswith('.ico'):          self._serve(p[1:], 'image/x-icon')
                    elif p.endswith(('.jpg','.jpeg')): self._serve(f'web-ui{p}', 'image/jpeg')
                    elif p.endswith('.png'):          self._serve(f'web-ui{p}', 'image/png')
                    elif p.endswith('.gif'):          self._serve(f'web-ui{p}', 'image/gif')
                    elif p == '/api/status':
                        import json
                        status = {"vrConnected": len(vr_sys_ref.vr_server.clients) > 0,
                                  "clients": len(vr_sys_ref.vr_server.clients)}
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(status).encode())
                    else: self.send_error(404)

            self.https_server = http.server.HTTPServer(
                (str(self.config.host_ip), int(self.config.https_port)), _APIHandler)
            ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
            cert_path, key_path = self.config.get_absolute_ssl_paths()
            ctx.load_cert_chain(cert_path, key_path)
            self.https_server.socket = ctx.wrap_socket(self.https_server.socket, server_side=True)
            self.https_thread = threading.Thread(target=self.https_server.serve_forever, daemon=True)
            self.https_thread.start()
            print(f"HTTPS 服务器已启动: https://{self.config.host_ip}:{int(self.config.https_port)}/")
            import socket as _sock
            try:
                with _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM) as s:
                    s.connect(("8.8.8.8", 80))
                    local_ip = s.getsockname()[0]
                print(f"头显请访问: https://{local_ip}:{int(self.config.https_port)}/")
            except: pass
        except Exception as e:
            print(f"HTTPS 服务器启动失败（不影响 WebSocket）: {e}")

        await self.vr_server.start()
        print(f"WebSocket 服务器已启动，等待 VR 连接...")
        await self._vr_loop()

    def stop(self):
        self.is_running = False
        if self.https_server:
            self.https_server.shutdown()


# ============================================================
# 入口
# ============================================================
def main():
    # TelegripConfig 是 dataclass，直接实例化（自动读取 config.yaml）
    config = TelegripConfig()

    # 创建 MuJoCo 控制器
    mj_ctrl = VROSCBFController(MODEL_PATH)

    # 创建 VR 控制系统
    try:
        vr_system = VRControlSystem(mj_ctrl, config)
    except Exception as e:
        print(f"VR 系统初始化失败: {e}")
        print("仅运行 MuJoCo 仿真（无 VR 控制）")
        mj_ctrl.run_loop()
        return

    # 在后台线程运行 VR 异步循环
    def run_vr():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(vr_system.run())
        except Exception as e:
            print(f"VR 循环异常: {e}")
        finally:
            loop.close()

    vr_thread = threading.Thread(target=run_vr, daemon=True)
    vr_thread.start()
    print("VR 线程已启动")

    # MuJoCo 主循环（在主线程运行）
    mj_ctrl.run_loop()
    vr_system.stop()


if __name__ == "__main__":
    main()
