"""
OpenArmX 双臂 OSCBF 控制器
基于 oscbf_wholebody_double.py 适配 openarmx_mujoco.xml
- 关节名前缀：openarmx_left_joint1~7 / openarmx_right_joint1~7
- 末端 link: openarmx_left_link7 / openarmx_right_link7
- 本体碰撞：robot_body_collision (box geom)
- 小球：moving_ball (可拖动)
- 胶囊体：link3/4 使用胶囊体进行本体碰撞检测（与 qijia 一致）
"""
import os
import warnings
warnings.filterwarnings("ignore", message=".*fixed link.*")
import mujoco
import mujoco_viewer
import time
import transforms3d as tf
import roboticstoolbox as rtb
from spatialmath import SE3
import osqp
import fcl
from scipy.optimize import minimize
import numpy as np
import scipy.sparse as sparse

MODEL_PATH = os.path.join(os.path.dirname(__file__), "openarmx_mujoco.xml")


class OpenArmXDualController(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 1.8, azimuth=180, elevation=-30)
        self.path = path

        # ---- DH 参数 ----
        dh_params = [
            rtb.RevoluteMDH(alpha=0,        a=0,       d=0.058,   offset=0),
            rtb.RevoluteMDH(alpha=np.pi/2,  a=-0.0205, d=0.081,   offset=0),
            rtb.RevoluteMDH(alpha=-np.pi/2, a=0.02,    d=0,       offset=0),
            rtb.RevoluteMDH(alpha=np.pi/2,  a=0,       d=0.14181, offset=np.pi/2),
            rtb.RevoluteMDH(alpha=np.pi/2,  a=0,       d=0.126,   offset=0),
            rtb.RevoluteMDH(alpha=-np.pi/2, a=0.037426,d=0,       offset=-np.pi/2),
            rtb.RevoluteMDH(alpha=-np.pi/2, a=-0.0375, d=0,       offset=-np.pi/2),
        ]
        self.left_robot_mdh  = rtb.DHRobot(dh_params, name="OpenArmX_Left")
        self.right_robot_mdh = rtb.DHRobot(dh_params, name="OpenArmX_Right")

        # ---- 基座变换（MuJoCo世界坐标 → DH基座坐标）----
        # 左臂 link1: pos=[0, 0.089, 0.698], quat=[0.707105, -0.707108, 0, 0] (w,x,y,z)
        # 右臂 link1: pos=[0, -0.089, 0.698], quat=[0.707105, 0.707108, 0, 0]
        # quat wxyz → 旋转矩阵
        def quat_wxyz_to_R(w, x, y, z):
            return np.array([
                [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
                [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
                [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
            ])
        # 左臂基座旋转（quat: w=0.707105, x=-0.707108, y=0, z=0）
        self.R_base_left  = quat_wxyz_to_R(0.707105, -0.707108, 0, 0)
        self.p_base_left  = np.array([0.0,  0.089, 0.698])
        # 右臂基座旋转（quat: w=0.707105, x=0.707108, y=0, z=0）
        self.R_base_right = quat_wxyz_to_R(0.707105,  0.707108, 0, 0)
        self.p_base_right = np.array([0.0, -0.089, 0.698])

        # ---- L构型目标关节角度（joint4=90°）----
        self.L_pose = np.array([0.0, 0.0, 0.0, np.pi/2, 0.0, 0.0, 0.0])

        # ---- 目标末端位姿（由 L构型正运动学计算，runBefore 里更新）----
        self.left_target_ee_pos        = np.array([0.3,  0.25, 0.9])
        self.left_target_ee_euler      = np.array([-np.pi/2, 0, -np.pi/2])
        self.left_target_ee_orientation = tf.euler.euler2mat(*self.left_target_ee_euler)

        self.right_target_ee_pos        = np.array([0.3, -0.25, 0.9])
        self.right_target_ee_euler      = np.array([-np.pi/2, 0, -np.pi/2])
        self.right_target_ee_orientation = tf.euler.euler2mat(*self.right_target_ee_euler)

        # ---- 小球碰撞参数 ----
        self.ball_radius      = 0.04
        self.safety_distance  = 0.05
        self.ball_actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ball_force_x"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ball_force_y"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ball_force_z"),
        ]
        self.left_robot_collision_bodies  = [
            "openarmx_left_link1", "openarmx_left_link2", "openarmx_left_link3",
            "openarmx_left_link4", "openarmx_left_link5", "openarmx_left_link6",
            "openarmx_left_link7",
        ]
        self.right_robot_collision_bodies = [
            "openarmx_right_link1", "openarmx_right_link2", "openarmx_right_link3",
            "openarmx_right_link4", "openarmx_right_link5", "openarmx_right_link6",
            "openarmx_right_link7",
        ]
        self.left_robot_collision_radii  = [0.07, 0.06, 0.07, 0.06, 0.05, 0.05, 0.05]
        self.right_robot_collision_radii = [0.07, 0.06, 0.07, 0.06, 0.05, 0.05, 0.05]
        self.left_robot_collision_ids  = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in self.left_robot_collision_bodies
        ]
        self.right_robot_collision_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in self.right_robot_collision_bodies
        ]
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_ball")

        # ---- 本体碰撞检测【与 qijia 一致：2 个胶囊体】----
        self.left_robot_collision_geoms  = [
            "openarmx_left_link3_collision",
            "openarmx_left_link4_collision",
            "openarmx_left_link7_collision",   # 末端也加入 body 碰撞检测
        ]
        self.right_robot_collision_geoms = [
            "openarmx_right_link3_collision",
            "openarmx_right_link4_collision",
            "openarmx_right_link7_collision",
        ]
        self.body_collision_geoms = ["robot_body_collision"]

        self.left_fcl_robot_objects  = self._init_fcl_objects(self.left_robot_collision_geoms)
        self.right_fcl_robot_objects = self._init_fcl_objects(self.right_robot_collision_geoms)
        self.fcl_body_objects        = self._init_fcl_objects(self.body_collision_geoms)

        # 【调试】打印 FCL 初始化状态
        print(f"\n{'='*50}")
        print(f"=== FCL 胶囊体初始化状态 ===")
        print(f"左臂胶囊体数：{len(self.left_fcl_robot_objects)}/{len(self.left_robot_collision_geoms)}")
        print(f"右臂胶囊体数：{len(self.right_fcl_robot_objects)}/{len(self.right_robot_collision_geoms)}")
        print(f"本体 FCL 对象数：{len(self.fcl_body_objects)}")
        if len(self.fcl_body_objects) == 0:
            print(f"⚠️  警告：本体 FCL 对象为空！请检查 XML 中是否有 robot_body_collision")
        if len(self.left_fcl_robot_objects) == 0:
            print(f"⚠️  警告：左臂胶囊体未加载！请检查 XML 中的 geom 名称")
        if len(self.right_fcl_robot_objects) == 0:
            print(f"⚠️  警告：右臂胶囊体未加载！请检查 XML 中的 geom 名称")
        print(f"{'='*50}\n")

        self.left_body_check_ids  = self.left_robot_collision_ids[:2]
        self.right_body_check_ids = self.right_robot_collision_ids[:2]

        # ---- 本体碰撞 CBF 参数【增强版】----
        self.body_box_center = np.array([0.0, 0.0, 0.34])
        self.body_box_half   = np.array([0.10, 0.03, 0.32])
        self.safety_distance_body = 0.05       # 安全距离 5cm
        self.alpha_body  = 500.0               # CBF 增益
        self.alpha2_body = 150.0               # 二阶 CBF 增益
        self.w_body = 10.0                     # body 约束权重（clip 后强制执行）

        # ---- 任务空间 CBF 参数 ----
        self.task_space_limits = {
            'x': {'min': -0.5, 'max': 1.5},
            'y': {'min': -1.5, 'max': 1.5},
            'z': {'min':  0.3, 'max': 1.5},
        }
        self.alpha_task  = 1.0
        self.alpha2_task = 1.0

        # ---- IK 状态 ----
        self.left_ik_joint_angles  = None
        self.right_ik_joint_angles = None
        self.left_q_prev  = None
        self.right_q_prev = None

        # ---- 关节/执行器索引 ----
        self.left_controlled_joints  = [f"openarmx_left_joint{i}"  for i in range(1, 8)]
        self.right_controlled_joints = [f"openarmx_right_joint{i}" for i in range(1, 8)]
        self.left_joint_qpos_indices  = []
        self.right_joint_qpos_indices  = []
        self.left_dof_indices         = []
        self.right_dof_indices        = []
        self.left_actuator_indices    = []
        self.right_actuator_indices   = []
        self._init_joint_indices()

        self.index = 0
        self.osqp_solver = osqp.OSQP()

    def _init_joint_indices(self):
        for jname in self.left_controlled_joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                self.left_joint_qpos_indices.append(self.model.jnt_qposadr[jid])
                self.left_dof_indices.append(self.model.jnt_dofadr[jid])
        for jname in self.right_controlled_joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                self.right_joint_qpos_indices.append(self.model.jnt_qposadr[jid])
                self.right_dof_indices.append(self.model.jnt_dofadr[jid])
        for i in range(1, 8):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"left_motor_{i}")
            if aid != -1:
                self.left_actuator_indices.append(aid)
        for i in range(1, 8):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"right_motor_{i}")
            if aid != -1:
                self.right_actuator_indices.append(aid)

    def _init_fcl_objects(self, geom_names):
        fcl_objects = []
        for name in geom_names:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id == -1:
                print(f"警告：未找到名为{name}的 geom")
                continue
            geom_type = self.model.geom_type[geom_id]
            size = self.model.geom_size[geom_id].copy()
            pos  = self.data.geom_xpos[geom_id].copy()
            mat  = self.data.geom_xmat[geom_id].copy()
            rot_mat = mat.reshape(3, 3)
            quat_xyzw = tf.quaternions.mat2quat(rot_mat)
            fcl_quat = np.array([quat_xyzw[1], quat_xyzw[2], quat_xyzw[3], quat_xyzw[0]])
            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                fcl_geom = fcl.Sphere(size[0])
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                fcl_geom = fcl.Box(size[0]*2, size[1]*2, size[2]*2)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                fcl_geom = fcl.Capsule(size[0], size[1]*2)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                fcl_geom = fcl.Cylinder(size[0], size[1]*2)
            else:
                print(f"不支持的几何类型：{geom_type}")
                continue
            fcl_obj = fcl.CollisionObject(fcl_geom, fcl.Transform(fcl_quat, pos))
            fcl_objects.append(fcl_obj)
        return fcl_objects

    def _update_fcl_poses(self, geom_names, fcl_objects):
        for i, name in enumerate(geom_names):
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id == -1 or i >= len(fcl_objects):
                continue
            pos = self.data.geom_xpos[geom_id].copy()
            mat = self.data.geom_xmat[geom_id].copy()
            rot_mat = mat.reshape(3, 3)
            quat_xyzw = tf.quaternions.mat2quat(rot_mat)
            fcl_quat = np.array([quat_xyzw[1], quat_xyzw[2], quat_xyzw[3], quat_xyzw[0]])
            fcl_objects[i].setTransform(fcl.Transform(fcl_quat, pos))

    def solve_inverse_kinematics(self, arm_side="left"):
        robot  = self.left_robot_mdh  if arm_side == "left" else self.right_robot_mdh
        target_pos    = self.left_target_ee_pos    if arm_side == "left" else self.right_target_ee_pos
        target_euler  = self.left_target_ee_euler  if arm_side == "left" else self.right_target_ee_euler
        q_prev = self.left_q_prev if arm_side == "left" else self.right_q_prev

        T_target = SE3(target_pos) * SE3.RPY(*target_euler)

        def cost(q):
            T = robot.fkine(q)
            return (np.linalg.norm(T_target.t - T.t) +
                    np.linalg.norm(T_target.R - T.R) +
                    0.1 * np.linalg.norm(q))

        x0 = np.zeros(7) if q_prev is None else q_prev
        res = minimize(cost, x0, method='L-BFGS-B', tol=1e-5)
        if res.success:
            if arm_side == "left":
                self.left_ik_joint_angles = res.x
                self.left_q_prev = res.x
            else:
                self.right_ik_joint_angles = res.x
                self.right_q_prev = res.x
            return True
        return False

    def get_current_qpos(self, arm_side="left"):
        idx = self.left_joint_qpos_indices if arm_side == "left" else self.right_joint_qpos_indices
        return np.array([self.data.qpos[i] for i in idx])

    def get_current_qvel(self, arm_side="left"):
        idx = self.left_dof_indices if arm_side == "left" else self.right_dof_indices
        return np.array([self.data.qvel[i] for i in idx])

    def get_end_effector_position(self, arm_side="left"):
        link = "openarmx_left_link7" if arm_side == "left" else "openarmx_right_link7"
        bid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, link)
        return self.data.body(bid).xpos.copy() if bid != -1 else None

    def get_end_orientation(self, arm_side="left"):
        link = "openarmx_left_link7" if arm_side == "left" else "openarmx_right_link7"
        bid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, link)
        return self.data.body(bid).xmat.reshape(3, 3).copy() if bid != -1 else None

    def get_ball_position(self):
        return self.data.body(self.ball_body_id).xpos.copy()

    def compute_orientation_error(self, R_cur, R_tgt):
        R_err = R_tgt @ R_cur.T
        e = 0.5 * (R_err - R_err.T)
        return np.array([e[2,1], e[0,2], e[1,0]])

    def compute_damped_pseudoinverse(self, J, lam=0.2):
        JT = J.T
        return JT @ np.linalg.inv(J @ JT + lam**2 * np.eye(J.shape[0]))

    def compute_collision_distances(self, arm_side="left"):
        ball_pos = self.get_ball_position()
        ids  = self.left_robot_collision_ids  if arm_side == "left" else self.right_robot_collision_ids
        rads = self.left_robot_collision_radii if arm_side == "left" else self.right_robot_collision_radii
        return np.array([
            np.linalg.norm(self.data.body(bid).xpos - ball_pos) - r - self.ball_radius
            for bid, r in zip(ids, rads)
        ])

    def compute_body_distance(self, arm_side="left"):
        """用 FCL 计算胶囊体到本体 box 的精确距离"""
        if arm_side == "left":
            fcl_robot_objects = self.left_fcl_robot_objects
            collision_geoms   = self.left_robot_collision_geoms
        else:
            fcl_robot_objects = self.right_fcl_robot_objects
            collision_geoms   = self.right_robot_collision_geoms

        # 更新 FCL 位姿
        self._update_fcl_poses(collision_geoms, fcl_robot_objects)
        self._update_fcl_poses(self.body_collision_geoms, self.fcl_body_objects)

        # 如果 FCL 对象为空，使用简化距离计算
        if not fcl_robot_objects or not self.fcl_body_objects:
            bid = (self.left_robot_collision_ids[2] if arm_side == "left"
                   else self.right_robot_collision_ids[2])  # link3
            arm_pos = self.data.body(bid).xpos
            body_dist = np.linalg.norm(arm_pos - self.body_box_center) - 0.15
            return max(body_dist, 0.0), bid, 0.05

        min_distances = []
        for robot_obj in fcl_robot_objects:
            for body_obj in self.fcl_body_objects:
                request = fcl.DistanceRequest()
                result  = fcl.DistanceResult()
                fcl.distance(robot_obj, body_obj, request, result)
                min_distances.append(result.min_distance)

        if not min_distances:
            fallback_bid = self.left_body_check_ids[0] if arm_side == "left" else self.right_body_check_ids[0]
            return 1.0, fallback_bid, 0.05

        min_idx  = int(np.argmin(min_distances))
        min_dist = min_distances[min_idx]

        # min_idx 是 robot_obj_i * n_body + body_obj_j 的展开索引
        n_body = len(self.fcl_body_objects)
        robot_obj_idx = min_idx // n_body
        robot_obj_idx = np.clip(robot_obj_idx, 0, len(fcl_robot_objects) - 1)
        robot_geom_name = collision_geoms[robot_obj_idx]
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom_name)
        min_body_id = self.model.geom_bodyid[geom_id] if geom_id != -1 else 0

        return min_dist, min_body_id, 0.05

    def solve_qp_analytical(self, P, q_vec, Lg_obs, rhs_obs,
                             Lg_task, rhs_task, Lg_body, rhs_body,
                             tau_min, tau_max):
        try:
            gamma = -np.linalg.solve(P + 1e-6*np.eye(7), q_vec)
        except Exception:
            gamma = -q_vec / (np.diag(P) + 1e-6)

        # 同时处理所有约束（body 和 obs 合并，与 wholebody 一致）
        for _ in range(100):
            body_viol = rhs_body - Lg_body @ gamma
            obs_viol  = rhs_obs  - Lg_obs  @ gamma
            total_step = np.zeros(7)

            if np.any(body_viol > 1e-4):
                g = Lg_body[np.argmax(body_viol)]
                total_step += (body_viol.max() / (g @ g + 1e-6)) * self.w_body * g

            if np.any(obs_viol > 1e-4):
                g = Lg_obs[np.argmax(obs_viol)]
                total_step += (obs_viol.max() / (g @ g + 1e-6)) * 1.0 * g

            gamma += total_step

            task_viol = rhs_task - Lg_task @ gamma
            if np.any(task_viol > 0):
                idx = np.argmax(task_viol)
                g   = Lg_task[idx]
                gamma += (task_viol[idx] / (g @ g + 1e-6)) * g

            if np.all(body_viol <= 1e-4) and np.all(obs_viol <= 1e-4) and np.all(task_viol <= 0):
                break

        return np.clip(gamma, tau_min, tau_max)

    def compute_arm_torque(self, arm_side="left"):
        q    = self.get_current_qpos(arm_side)
        qdot = self.get_current_qvel(arm_side)
        ee_pos    = self.get_end_effector_position(arm_side)
        ee_orient = self.get_end_orientation(arm_side)
        if ee_pos is None:
            return np.zeros(7)

        robot   = self.left_robot_mdh  if arm_side == "left" else self.right_robot_mdh
        tgt_pos = self.left_target_ee_pos    if arm_side == "left" else self.right_target_ee_pos
        tgt_ori = self.left_target_ee_orientation if arm_side == "left" else self.right_target_ee_orientation
        q_des   = self.left_ik_joint_angles  if arm_side == "left" else self.right_ik_joint_angles
        jidx    = self.left_dof_indices if arm_side == "left" else self.right_dof_indices
        col_ids = self.left_robot_collision_ids if arm_side == "left" else self.right_robot_collision_ids

        # 用 MuJoCo 自带雅可比（世界坐标系，不依赖 DH 参数）
        ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                        "openarmx_left_link7" if arm_side == "left" else "openarmx_right_link7")
        jac_pos_full = np.zeros((3, self.model.nv))
        jac_rot_full = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jac_pos_full, jac_rot_full, ee_body_id)
        J_pos_mj = jac_pos_full[:, jidx]   # (3, 7) 位置雅可比
        J_ori_mj = jac_rot_full[:, jidx]   # (3, 7) 旋转雅可比
        J_full_mj = np.vstack([J_pos_mj, J_ori_mj])  # (6, 7)
        J_pinv = self.compute_damped_pseudoinverse(J_full_mj)
        N_q    = np.eye(7) - J_pinv @ J_full_mj

        # 笛卡尔误差直接在世界坐标系计算（ee_pos 和 tgt_pos 都是世界坐标）
        pos_err = tgt_pos - ee_pos

        # Impedance nominal torque
        Kp_pos = np.diag([300]*3); Kd_pos = np.diag([40]*3)
        Kp_ori = np.diag([20]*3);  Kd_ori = np.diag([7]*3)
        if arm_side == "right" and hasattr(self, 'right_Kpj'):
            Kpj = self.right_Kpj; Kdj = self.right_Kdj
        elif arm_side == "left" and hasattr(self, 'left_Kpj'):
            Kpj = self.left_Kpj;  Kdj = self.left_Kdj
        else:
            Kpj = 50; Kdj = 10
        ori_err = self.compute_orientation_error(ee_orient, tgt_ori)
        # 笛卡尔空间控制（末端位姿误差驱动）
        F_cart = np.concatenate([
            Kp_pos @ pos_err - Kd_pos @ (J_pos_mj @ qdot),
            Kp_ori @ ori_err - Kd_ori @ (J_ori_mj @ qdot),
        ])
        Gamma_0   = -Kpj*(q - q_des) - Kdj*qdot
        Gamma_nom = J_full_mj.T @ F_cart + N_q @ Gamma_0

        # ---- Ball avoidance CBF ----
        ball_pos  = self.get_ball_position()
        # 小球固定不动，速度为零
        ball_vel = np.zeros(3)
        h_obs = self.compute_collision_distances(arm_side) - self.safety_distance
        Lf_h  = []
        for bid in col_ids:
            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jac, None, bid)
            bvel = jac @ self.data.qvel
            bpos = self.data.body(bid).xpos
            d    = bpos - ball_pos
            dn   = d / (np.linalg.norm(d) + 1e-8)
            Lf_h.append(np.dot(bvel - ball_vel, dn))
        Lf_h   = np.array(Lf_h)
        h2_obs = Lf_h + 10.0 * h_obs
        Lg_obs = []
        for bid in col_ids:
            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jac, None, bid)
            Jp  = jac[:, jidx]
            d   = self.data.body(bid).xpos - ball_pos
            dn  = d / (np.linalg.norm(d) + 1e-8)
            Lg_obs.append(dn @ Jp)
        Lg_obs   = np.array(Lg_obs)
        rhs_obs  = -10.0 * h2_obs

        # 诊断：每500步打印小球 CBF 状态
        if self.index % 500 == 0 and arm_side == "right":
            min_h = h_obs.min()
            min_idx = h_obs.argmin()
            print(f"[Ball CBF {arm_side}] ball={np.round(ball_pos,3)} "
                  f"min_h={min_h:.4f} (link={self.right_robot_collision_bodies[min_idx]}) "
                  f"|Lg|={np.linalg.norm(Lg_obs[min_idx]):.4f} "
                  f"rhs={rhs_obs[min_idx]:.3f}")

        # ---- Body avoidance CBF ----
        # 实时更新 body box 中心位置（box 可拖动）
        body_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot_body_collision_body")
        if body_body_id != -1:
            body_origin = self.data.body(body_body_id).xpos.copy()
            # box geom 在 body 内的偏移是 pos="0 0 0.34"
            self.body_box_center = body_origin + np.array([0.0, 0.0, 0.34])

        min_dist, min_bid, min_rad = self.compute_body_distance(arm_side)
        h_body = min_dist - self.safety_distance_body

        # 仅在真正穿透时打印
        if h_body < -0.01:
            min_bid_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, min_bid)
            print(f"[Body CBF {arm_side}] ⚠️穿透 dist={min_dist:.4f} h={h_body:.4f} link={min_bid_name}")

        # 计算方向向量（无论是否穿透都需要）
        if arm_side == "left":
            fcl_robot_objects = self.left_fcl_robot_objects
        else:
            fcl_robot_objects = self.right_fcl_robot_objects

        # 用 body 位置到 box 中心的方向作为排斥方向（稳定可靠）
        body_pos = self.data.body(min_bid).xpos.copy()
        direction = body_pos - self.body_box_center
        direction_norm = direction / (np.linalg.norm(direction) + 1e-10)

        # 尝试用 FCL 最近点方向（更精确）
        if fcl_robot_objects and self.fcl_body_objects:
            best_result = None
            best_d = np.inf
            for robot_obj in fcl_robot_objects:
                for body_obj in self.fcl_body_objects:
                    req = fcl.DistanceRequest()
                    res = fcl.DistanceResult()
                    fcl.distance(robot_obj, body_obj, req, res)
                    if res.min_distance < best_d:
                        best_d = res.min_distance
                        best_result = res
            if (best_result is not None and
                    best_result.nearest_points[0] is not None and
                    best_result.nearest_points[1] is not None):
                d_vec = best_result.nearest_points[0] - best_result.nearest_points[1]
                if np.linalg.norm(d_vec) > 1e-6:
                    direction_norm = d_vec / np.linalg.norm(d_vec)

        # 计算 Jacobian：用 FCL 最近点位置计算点雅可比（而非 body 原点）
        # 这样能正确反映所有上游关节对该点位置的影响
        jac_b = np.zeros((3, self.model.nv))
        jac_b_rot = np.zeros((3, self.model.nv))

        # 用最近点位置计算雅可比（mj_jac 需要世界坐标系中的点）
        if fcl_robot_objects and self.fcl_body_objects:
            best_result2 = None
            best_d2 = np.inf
            for robot_obj in fcl_robot_objects:
                for body_obj in self.fcl_body_objects:
                    req2 = fcl.DistanceRequest()
                    res2 = fcl.DistanceResult()
                    fcl.distance(robot_obj, body_obj, req2, res2)
                    if res2.min_distance < best_d2:
                        best_d2 = res2.min_distance
                        best_result2 = res2
            if (best_result2 is not None and
                    best_result2.nearest_points[0] is not None):
                closest_pt = np.array(best_result2.nearest_points[0])
                mujoco.mj_jac(self.model, self.data, jac_b, jac_b_rot, closest_pt, min_bid)
            else:
                mujoco.mj_jacBody(self.model, self.data, jac_b, None, min_bid)
        else:
            mujoco.mj_jacBody(self.model, self.data, jac_b, None, min_bid)

        bvel_b = jac_b @ self.data.qvel
        Lf_hb  = np.dot(bvel_b, direction_norm)
        Jp_b   = jac_b[:, jidx]
        Lg_body_vec = direction_norm @ Jp_b  # shape (7,)

        # 穿透时用更强的增益
        if h_body < 0:
            alpha_b  = self.alpha_body  * 3.0
            alpha2_b = self.alpha2_body * 3.0
        else:
            alpha_b  = self.alpha_body
            alpha2_b = self.alpha2_body

        h2_body  = Lf_hb + alpha_b * h_body
        Lg_body  = np.array([Lg_body_vec])
        rhs_body = np.array([-alpha2_b * h2_body])

        # ---- Task space CBF ----
        M_inv = np.eye(7)
        C     = np.ones(7)
        h_task = np.array([
            ee_pos[0] - self.task_space_limits['x']['min'],
            self.task_space_limits['x']['max'] - ee_pos[0],
            ee_pos[1] - self.task_space_limits['y']['min'],
            self.task_space_limits['y']['max'] - ee_pos[1],
            ee_pos[2] - self.task_space_limits['z']['min'],
            self.task_space_limits['z']['max'] - ee_pos[2],
        ])
        grad_h = np.zeros((6, 14))
        grad_h[:, :7] = np.vstack([
            J_pos_mj[0:1,:], -J_pos_mj[0:1,:],
            J_pos_mj[1:2,:], -J_pos_mj[1:2,:],
            J_pos_mj[2:3,:], -J_pos_mj[2:3,:],
        ])
        fz = np.concatenate([qdot, -M_inv @ C])
        Lf_ht  = grad_h @ fz
        h2_task = Lf_ht + self.alpha_task * h_task
        gLf    = np.zeros((6, 14)); gLf[:, 7:] = grad_h[:, :7]
        gh2    = gLf + self.alpha_task * grad_h
        Lf_h2t = gh2 @ fz
        gz     = np.zeros((14, 7)); gz[7:, :] = M_inv
        Lg_task = gh2 @ gz
        rhs_task = -self.alpha2_task * h2_task - Lf_h2t

        # ---- QP ----
        Wj = np.diag([1]*7)
        Wo = 3 * np.diag([1]*6)
        Ws = 0.8 * np.eye(7)
        P_qp  = N_q @ Wj @ N_q.T + J_full_mj.T @ Wo @ J_full_mj + Ws
        q_qp  = -Gamma_nom @ P_qp
        tau_min = np.array([-80, -80, -60, -60, -20, -20, -20], dtype=float)
        tau_max = -tau_min

        return self.solve_qp_analytical(
            P_qp, q_qp, Lg_obs, rhs_obs,
            Lg_task, rhs_task, Lg_body, rhs_body,
            tau_min, tau_max
        )

    def runBefore(self):
        # 用 L构型作为目标关节角度
        self.left_ik_joint_angles  = self.L_pose.copy()
        self.right_ik_joint_angles = self.L_pose.copy()
        self.left_q_prev  = self.L_pose.copy()
        self.right_q_prev = self.L_pose.copy()

        # 由 MuJoCo 实际末端位置作为目标（不用 DH，避免坐标系不匹配）
        left_ee  = self.get_end_effector_position("left")
        right_ee = self.get_end_effector_position("right")
        left_ori  = self.get_end_orientation("left")
        right_ori = self.get_end_orientation("right")
        self.left_target_ee_pos         = left_ee.copy()
        self.left_target_ee_orientation = left_ori.copy()
        self.right_target_ee_pos        = right_ee.copy()
        self.right_target_ee_orientation = right_ori.copy()

        # 把仿真初始关节角也设为 L构型
        for i, jname in enumerate(self.left_controlled_joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                self.data.qpos[self.model.jnt_qposadr[jid]] = self.L_pose[i]
        for i, jname in enumerate(self.right_controlled_joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                self.data.qpos[self.model.jnt_qposadr[jid]] = self.L_pose[i]

        mujoco.mj_forward(self.model, self.data)  # 刷新正运动学

        # ---- DH 参数验证：对比正运动学和 MuJoCo 实际末端 ----
        q_now = self.get_current_qpos("right")
        T_fk  = self.right_robot_mdh.fkine(q_now)
        ee_mj = self.get_end_effector_position("right")

        # 用 MuJoCo 实际的基座旋转矩阵（更准确）
        link1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarmx_right_link1")
        R_base_mj = self.data.body(link1_id).xmat.reshape(3,3).copy()
        p_base_mj = self.data.body(link1_id).xpos.copy()
        ee_dh_world = R_base_mj @ np.array(T_fk.t) + p_base_mj

        print(f"\n=== DH 验证 ===")
        print(f"当前关节角: {np.round(np.degrees(q_now),1)} deg")
        print(f"基座位置: {np.round(p_base_mj,4)}")
        print(f"DH→世界坐标末端: {np.round(ee_dh_world, 4)}")
        print(f"MuJoCo 实际末端: {np.round(ee_mj, 4)}")
        print(f"误差: {np.round(ee_dh_world - ee_mj, 4)}  |误差|={np.linalg.norm(ee_dh_world - ee_mj):.4f}m")
        print(f"================\n")

        # 保存基座变换供控制使用
        self.R_base_right_mj = R_base_mj
        self.p_base_right_mj = p_base_mj
        link1_id_l = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarmx_left_link1")
        self.R_base_left_mj = self.data.body(link1_id_l).xmat.reshape(3,3).copy()
        self.p_base_left_mj = self.data.body(link1_id_l).xpos.copy()

        self.total_time = 100.0
        self.dt         = self.model.opt.timestep
        self.num_steps  = int(self.total_time / self.dt)
        self.index      = 0

        # ---- 末端位姿目标模式 ----
        # 右臂目标末端 = 小球中心（阻抗控制驱动往小球走，CBF 阻止靠近）
        # 左臂目标末端 = 本体 box 中心（阻抗控制驱动往本体走，CBF 阻止靠近）
        ball_pos = self.get_ball_position()
        body_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot_body_collision_body")
        body_center = self.data.body(body_bid).xpos.copy() + np.array([0, 0, 0.34])

        # 右臂：目标末端 = 小球位置，关节目标 = 当前关节角（不产生关节空间干扰力矩）
        self.right_target_ee_pos = ball_pos.copy()
        self.right_target_ee_orientation = self.get_end_orientation("right").copy()
        self.right_ik_joint_angles = self.get_current_qpos("right").copy()

        # 左臂：目标末端 = 本体内部（直接穿进去的位置）
        self.left_target_ee_pos = np.array([0.0, 0.0, 0.34])  # 本体 box 中心内部
        self.left_target_ee_orientation = self.get_end_orientation("left").copy()
        self.left_ik_joint_angles = self.get_current_qpos("left").copy()

        # 笛卡尔刚度（关节空间刚度设为0，完全靠笛卡尔项驱动）
        self.right_Kpj = 0;  self.right_Kdj = 5
        self.left_Kpj  = 0;  self.left_Kdj  = 5

        print(f"右臂目标末端（小球）: {np.round(ball_pos, 3)}")
        print(f"左臂目标末端（本体内部）: {np.round(np.array([0.0, 0.0, 0.34]), 3)}")
        print("CBF 有效版本：手臂应在安全距离外停下")

    def runFunc(self):
        if self.index >= self.num_steps:
            return

        left_tau  = self.compute_arm_torque("left")
        right_tau = self.compute_arm_torque("right")
        for i, aid in enumerate(self.left_actuator_indices):
            self.data.ctrl[aid] = left_tau[i]
        for i, aid in enumerate(self.right_actuator_indices):
            self.data.ctrl[aid] = right_tau[i]

        if self.index % 200 == 0:
            ee_r = self.get_end_effector_position("right")
            ee_l = self.get_end_effector_position("left")
            dist_ball  = np.linalg.norm(ee_r - self.right_target_ee_pos) if ee_r is not None else -1
            dist_body  = np.linalg.norm(ee_l - self.left_target_ee_pos)  if ee_l is not None else -1
            print(f"[{self.index}] 右臂末端={np.round(ee_r,3)} 距小球={dist_ball:.3f}m | "
                  f"左臂末端={np.round(ee_l,3)} 距本体={dist_body:.3f}m")

        self.index += 1
        time.sleep(0.001)
        for i, aid in enumerate(self.right_actuator_indices):
            self.data.ctrl[aid] = right_tau[i]

        # 只在力矩明显（说明 CBF 在工作）时打印
        left_tau_norm  = np.linalg.norm(left_tau)
        right_tau_norm = np.linalg.norm(right_tau)
        if self.index % 200 == 0 and (left_tau_norm > 5.0 or right_tau_norm > 5.0):
            print(f"[{self.index}] CBF激活 左臂|τ|={left_tau_norm:.1f} 右臂|τ|={right_tau_norm:.1f}")
            print(f"       左臂 tau={np.round(left_tau,2)}  右臂 tau={np.round(right_tau,2)}")

        self.index += 1
        time.sleep(0.001)

    def runAfter(self):
        print("仿真结束")


if __name__ == "__main__":
    ctrl = OpenArmXDualController(MODEL_PATH)
    ctrl.run_loop()