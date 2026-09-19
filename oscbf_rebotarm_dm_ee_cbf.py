import argparse
import os
import sys
import time
import warnings
import xml.etree.ElementTree as ET

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ModuleNotFoundError as exc:
    mujoco = None
    MUJOCO_IMPORT_ERROR = exc
else:
    MUJOCO_IMPORT_ERROR = None

warnings.filterwarnings("ignore", message=".*fixed link.*")


def _script_directory():
    """Recover the Unicode script path on Windows builds using a legacy code page."""
    if os.name == "nt":
        import ctypes

        get_command_line = ctypes.windll.kernel32.GetCommandLineW
        get_command_line.restype = ctypes.c_wchar_p
        command_line_to_argv = ctypes.windll.shell32.CommandLineToArgvW
        command_line_to_argv.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
        command_line_to_argv.restype = ctypes.POINTER(ctypes.c_wchar_p)
        command_line = get_command_line()
        argc = ctypes.c_int()
        argv = command_line_to_argv(command_line, ctypes.byref(argc))
        try:
            for index in range(1, argc.value):
                argument = argv[index]
                if argument.lower().endswith("oscbf_rebotarm_dm_ee_cbf.py"):
                    if not os.path.isabs(argument):
                        buffer = ctypes.create_unicode_buffer(32768)
                        ctypes.windll.kernel32.GetCurrentDirectoryW(len(buffer), buffer)
                        argument = os.path.join(buffer.value, argument)
                    return os.path.dirname(os.path.abspath(argument))
        finally:
            ctypes.windll.kernel32.LocalFree(argv)
    return os.path.dirname(os.path.abspath(__file__))


ROOT_DIR = _script_directory()
URDF_PATH = os.path.join(
    ROOT_DIR, "reBotArm_control_py", "urdf", "DM", "urdf", "ReBot_Arm_DM.urdf"
)
MODEL_PATH = os.path.join(ROOT_DIR, "rebotarm_dm_oscbf_mujoco.xml")
TARGET_BALL_RADIUS = 0.025
OBSTACLE_BALL_RADIUS = 0.055
FLOOR_Z = 0.0
ROBOT_BASE_Z = 0.0

# Capsule endpoints are expressed in each link body's local coordinate frame.
# radius controls thickness; start/end control position, direction, and length.
CBF_CAPSULE_CONFIG = {
    "link2": {
        "radius": 0.035,
        "start": np.array([0.0, -0.0031, -0.0308]),
        "end": np.array([-0.24, -0.0031, -0.0308]),
    },
    "link3": {
        "radius": 0.035,
        "start": np.array([0.0, -0.0536, -0.0310]),
        "end": np.array([0.2426, -0.0536, -0.0310]),
    },
    "link6": {
        "radius": 0.035,
        "start": np.array([0.0, 0.0, 0.0]),
        "end": np.array([0.0, 0.0, 0.1]),
    },
}


def _urdf_visual_mesh_records(urdf_path=URDF_PATH):
    robot = ET.parse(urdf_path).getroot()
    records = []
    urdf_dir = os.path.dirname(urdf_path)
    for link in robot.findall("link"):
        link_name = link.attrib["name"]
        for visual_index, visual in enumerate(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            origin = visual.find("origin")
            xyz = np.zeros(3)
            rpy = np.zeros(3)
            if origin is not None:
                xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
                rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
            filename = mesh.attrib["filename"]
            records.append({
                "link": link_name,
                "name": f"dm_visual_{link_name}_{visual_index}",
                "file": f"dm_visual_{len(records):03d}.stl",
                "source": os.path.normpath(os.path.join(urdf_dir, filename)),
                "xyz": xyz,
                "rpy": rpy,
                "scale": mesh.attrib.get("scale"),
            })
    return records


def load_mujoco_model(path):
    # MuJoCo's Windows file loader can reject non-ASCII paths. Python's Unicode
    # file API is reliable, so compile the already-resolved XML text instead.
    with open(path, "r", encoding="utf-8") as xml_file:
        xml_text = xml_file.read()
    assets = {}
    for record in _urdf_visual_mesh_records():
        with open(record["source"], "rb") as mesh_file:
            assets[record["file"]] = mesh_file.read()
    return mujoco.MjModel.from_xml_string(xml_text, assets=assets)


def _rpy_to_quat_wxyz(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _fmt(values):
    return " ".join(f"{float(v):.8g}" for v in values)


def generate_rebotarm_mjcf(urdf_path=URDF_PATH, xml_path=MODEL_PATH):
    """Generate MJCF using the URDF kinematics and its original visual meshes."""
    tree = ET.parse(urdf_path)
    robot = tree.getroot()
    joints = {}
    children_by_parent = {}
    for joint in robot.findall("joint"):
        name = joint.attrib["name"]
        jtype = joint.attrib.get("type", "fixed")
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        origin = joint.find("origin")
        xyz = np.zeros(3)
        rpy = np.zeros(3)
        if origin is not None:
            xyz = np.array([float(x) for x in origin.attrib.get("xyz", "0 0 0").split()])
            rpy = np.array([float(x) for x in origin.attrib.get("rpy", "0 0 0").split()])
        axis_node = joint.find("axis")
        axis = np.array([0.0, 0.0, 1.0])
        if axis_node is not None:
            axis = np.array([float(x) for x in axis_node.attrib.get("xyz", "0 0 1").split()])
        limit = joint.find("limit")
        lower, upper, effort = -3.14, 3.14, 20.0
        if limit is not None:
            lower = float(limit.attrib.get("lower", lower))
            upper = float(limit.attrib.get("upper", upper))
            effort = float(limit.attrib.get("effort", effort))
        joints[name] = {
            "name": name,
            "type": jtype,
            "parent": parent,
            "child": child,
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis,
            "range": (lower, upper),
            "effort": effort,
        }
        children_by_parent.setdefault(parent, []).append(joints[name])

    chain = []
    link = "base_link"
    while link != "end_link":
        candidates = [
            j for j in children_by_parent.get(link, [])
            if j["type"] in ("revolute", "continuous", "fixed")
            and not j["name"].startswith("finger")
        ]
        if not candidates:
            break
        joint = candidates[0]
        chain.append(joint)
        link = joint["child"]

    visual_meshes = _urdf_visual_mesh_records(urdf_path)
    visuals_by_link = {}
    for record in visual_meshes:
        visuals_by_link.setdefault(record["link"], []).append(record)

    mujoco_root = ET.Element("mujoco", {"model": "rebotarm_dm_oscbf"})
    ET.SubElement(
        mujoco_root,
        "compiler",
        {
            "angle": "radian",
            "autolimits": "true",
        },
    )
    ET.SubElement(mujoco_root, "option", {"timestep": "0.002", "gravity": "0 0 -9.81"})
    default = ET.SubElement(mujoco_root, "default")
    ET.SubElement(
        default,
        "joint",
        {"damping": "1.2", "armature": "0.02", "limited": "true"},
    )
    ET.SubElement(
        default,
        "geom",
        {"friction": "0.8 0.1 0.1", "density": "450", "contype": "0", "conaffinity": "0"},
    )
    asset = ET.SubElement(mujoco_root, "asset")
    ET.SubElement(asset, "material", {"name": "arm_gray", "rgba": "0.48 0.48 0.48 1"})
    ET.SubElement(asset, "material", {"name": "dark_gray", "rgba": "0.25 0.25 0.25 1"})
    ET.SubElement(asset, "material", {"name": "collision_capsule", "rgba": "0.95 0.78 0.08 0.38"})
    ET.SubElement(asset, "material", {"name": "red", "rgba": "1 0.05 0.03 0.82"})
    ET.SubElement(asset, "material", {"name": "orange", "rgba": "1 0.48 0.02 0.72"})
    ET.SubElement(asset, "material", {"name": "cyan", "rgba": "0.05 0.55 1 0.72"})
    ET.SubElement(asset, "material", {"name": "floor_mat", "rgba": "0.18 0.20 0.22 1"})
    for record in visual_meshes:
        attrs = {"name": record["name"], "file": record["file"]}
        if record["scale"]:
            attrs["scale"] = record["scale"]
        ET.SubElement(asset, "mesh", attrs)

    world = ET.SubElement(mujoco_root, "worldbody")
    ET.SubElement(world, "light", {"pos": "0 -1.5 2.5", "dir": "0.2 0.5 -1", "diffuse": "0.8 0.8 0.8"})
    ET.SubElement(world, "camera", {"name": "main", "pos": "0.75 -1.25 0.75", "xyaxes": "0.86 0.51 0 -0.22 0.37 0.9"})
    ET.SubElement(world, "geom", {"name": "floor", "type": "plane", "pos": f"0 0 {FLOOR_Z}", "size": "2 2 0.01", "material": "floor_mat"})

    def add_slide_ball(name, pos, radius, material):
        body = ET.SubElement(
            world,
            "body",
            {"name": name, "pos": _fmt(pos), "gravcomp": "1"},
        )
        ET.SubElement(body, "joint", {"name": f"{name}_x", "type": "slide", "axis": "1 0 0", "damping": "1.5", "range": "-1.0 1.0"})
        ET.SubElement(body, "joint", {"name": f"{name}_y", "type": "slide", "axis": "0 1 0", "damping": "1.5", "range": "-1.0 1.0"})
        ET.SubElement(body, "joint", {"name": f"{name}_z", "type": "slide", "axis": "0 0 1", "damping": "1.5", "range": "-0.4 1.0"})
        ET.SubElement(body, "geom", {"name": f"{name}_geom", "type": "sphere", "size": f"{radius}", "material": material, "contype": "1", "conaffinity": "1"})
        ET.SubElement(body, "inertial", {"pos": "0 0 0", "mass": "0.1", "diaginertia": "0.001 0.001 0.001"})

    add_slide_ball("target_ball", [0.28, -0.18, 0.32], TARGET_BALL_RADIUS, "red")
    add_slide_ball("obstacle_ball_1", [0.20, 0.02, 0.28], OBSTACLE_BALL_RADIUS, "orange")
    add_slide_ball("obstacle_ball_2", [0.40, -0.08, 0.42], OBSTACLE_BALL_RADIUS, "cyan")

    def add_link_visuals(body, link_name):
        for record in visuals_by_link.get(link_name, []):
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{record['name']}_geom",
                    "type": "mesh",
                    "mesh": record["name"],
                    "pos": _fmt(record["xyz"]),
                    "quat": _fmt(_rpy_to_quat_wxyz(*record["rpy"])),
                    "material": "arm_gray",
                    "mass": "0",
                    "contype": "0",
                    "conaffinity": "0",
                },
            )

    base = ET.SubElement(world, "body", {"name": "base_link", "pos": f"0 0 {ROBOT_BASE_Z}"})
    add_link_visuals(base, "base_link")
    ET.SubElement(base, "geom", {"name": "base_collision", "type": "cylinder", "size": "0.085 0.035", "rgba": "0 0 0 0"})
    current = base
    for joint in chain:
        body = ET.SubElement(
            current,
            "body",
            {
                "name": joint["child"],
                "pos": _fmt(joint["xyz"]),
                "quat": _fmt(_rpy_to_quat_wxyz(*joint["rpy"])),
            },
        )
        if joint["type"] != "fixed":
            attrs = {
                "name": joint["name"],
                "type": "hinge",
                "axis": _fmt(joint["axis"]),
                "range": _fmt(joint["range"]),
                "actuatorfrcrange": f"{-joint['effort']} {joint['effort']}",
            }
            ET.SubElement(body, "joint", attrs)
        add_link_visuals(body, joint["child"])

        next_joints = [
            j for j in children_by_parent.get(joint["child"], [])
            if not j["name"].startswith("finger")
        ]
        capsule = CBF_CAPSULE_CONFIG.get(joint["child"])
        if capsule is not None:
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{joint['child']}_collision",
                    "type": "capsule",
                    "fromto": _fmt([*capsule["start"], *capsule["end"]]),
                    "size": f"{capsule['radius']}",
                    "material": "collision_capsule",
                    "contype": "0",
                    "conaffinity": "0",
                },
            )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{joint['child']}_joint_ball",
                "type": "sphere",
                "size": "0.042",
                "rgba": "0 0 0 0",
            },
        )
        current = body

    for finger_joint in children_by_parent.get("end_link", []):
        if not finger_joint["name"].startswith("finger"):
            continue
        finger_body = ET.SubElement(
            current,
            "body",
            {
                "name": finger_joint["child"],
                "pos": _fmt(finger_joint["xyz"]),
                "quat": _fmt(_rpy_to_quat_wxyz(*finger_joint["rpy"])),
            },
        )
        add_link_visuals(finger_body, finger_joint["child"])

    ET.SubElement(current, "site", {"name": "ee_site", "pos": "0 0 0", "size": "0.012", "rgba": "0.1 0.1 0.1 1"})

    actuator = ET.SubElement(mujoco_root, "actuator")
    for joint in chain:
        if joint["type"] != "fixed":
            effort = joint["effort"]
            ET.SubElement(
                actuator,
                "motor",
                {
                    "name": f"{joint['name']}_motor",
                    "joint": joint["name"],
                    "gear": "1",
                    "ctrllimited": "true",
                    "ctrlrange": f"{-effort} {effort}",
                },
            )

    xml = ET.tostring(mujoco_root, encoding="unicode")
    xml = "<?xml version='1.0' encoding='utf-8'?>\n" + xml
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml)


class ReBotArmDMCBFController:
    def __init__(self, path):
        if mujoco is None:
            raise RuntimeError(
                "MuJoCo is not installed in this Python environment.\n"
                "Run with D:\\Anaconda_envs\\envs\\mujoco_env\\python.exe, "
                "or install it with: python -m pip install mujoco"
            ) from MUJOCO_IMPORT_ERROR
        self.path = path
        self.model = load_mujoco_model(path)
        self.data = mujoco.MjData(self.model)
        self.n = 6
        self.controlled_joints = [f"joint{i}" for i in range(1, 7)]
        self.collision_geom_names = [
            f"{link_name}_collision" for link_name in CBF_CAPSULE_CONFIG
        ]
        self.obstacle_balls = ["obstacle_ball_1", "obstacle_ball_2"]
        self.ball_radius = OBSTACLE_BALL_RADIUS
        self.target_radius = TARGET_BALL_RADIUS
        self.safety_distance = 0.055
        self.alpha_obs = 18.0
        self.alpha2_obs = 12.0
        self.task_limits = {
            "x": (-0.10, 0.72),
            "y": (-0.55, 0.55),
            "z": (0.06, 0.80),
        }
        self.alpha_task = 8.0
        self.alpha2_task = 5.0
        self.Kp_pos = np.diag([800, 800.0, 800.0])
        self.Kd_pos = np.diag([34.0, 34.0, 34.0])
        self.Kp_ori = np.diag([16.0, 16.0, 16.0])
        self.Kd_ori = np.diag([4.0, 4.0, 4.0])
        self.q_home = np.array([0.0, -1.2, -1.15, 0.35, 0.0, 0.0])

        self.joint_qpos_indices = []
        self.dof_indices = []
        self.actuator_indices = []
        self.collision_geom_ids = []
        self.obstacle_body_ids = []
        self._init_indices()
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_ball")
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "end_link")
        self.index = 0
        self.last_tau_nom = np.zeros(self.n)
        self.last_tau_clipped_nom = np.zeros(self.n)
        self.last_tau_safe = np.zeros(self.n)
        self.last_min_safety_margin = np.inf

    def _init_indices(self):
        for name in self.controlled_joints:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise RuntimeError(f"Joint not found in MuJoCo model: {name}")
            self.joint_qpos_indices.append(self.model.jnt_qposadr[jid])
            self.dof_indices.append(self.model.jnt_dofadr[jid])
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor")
            if aid == -1:
                raise RuntimeError(f"Actuator not found in MuJoCo model: {name}_motor")
            self.actuator_indices.append(aid)
        for name in self.collision_geom_names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid == -1:
                raise RuntimeError(f"CBF collision capsule not found: {name}")
            self.collision_geom_ids.append(gid)
        for name in self.obstacle_balls:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid != -1:
                self.obstacle_body_ids.append(bid)

    def get_q(self):
        return np.array([self.data.qpos[i] for i in self.joint_qpos_indices])

    def get_qdot(self):
        return np.array([self.data.qvel[i] for i in self.dof_indices])

    def get_target_pos(self):
        return self.data.body(self.target_body_id).xpos.copy()

    def get_ee_pos(self):
        return self.data.site(self.ee_site_id).xpos.copy()

    def get_ee_rot(self):
        return self.data.body(self.ee_body_id).xmat.reshape(3, 3).copy()

    def body_velocity(self, bid):
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jac, None, bid)
        return jac @ self.data.qvel

    def capsule_distance_data(self, geom_id, obstacle_pos):
        center = self.data.geom_xpos[geom_id]
        rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
        half_length = self.model.geom_size[geom_id, 1]
        axis = rotation[:, 2]
        segment_start = center - half_length * axis
        segment_end = center + half_length * axis
        segment = segment_end - segment_start
        segment_norm_sq = float(segment @ segment)
        if segment_norm_sq > 1e-12:
            fraction = np.clip(
                float((obstacle_pos - segment_start) @ segment) / segment_norm_sq,
                0.0,
                1.0,
            )
        else:
            fraction = 0.0
        closest_point = segment_start + fraction * segment
        delta = closest_point - obstacle_pos
        distance = np.linalg.norm(delta)
        direction = delta / (distance + 1e-8)
        body_id = self.model.geom_bodyid[geom_id]
        jacobian = np.zeros((3, self.model.nv))
        mujoco.mj_jac(
            self.model,
            self.data,
            jacobian,
            None,
            closest_point,
            body_id,
        )
        radius = self.model.geom_size[geom_id, 0]
        return distance, direction, jacobian, radius

    @staticmethod
    def orientation_error(R_cur, R_tgt):
        R_err = R_tgt @ R_cur.T
        e = 0.5 * (R_err - R_err.T)
        return np.array([e[2, 1], e[0, 2], e[1, 0]])

    def damped_pinv(self, J, lam=0.08):
        return J.T @ np.linalg.inv(J @ J.T + lam * lam * np.eye(J.shape[0]))

    def solve_projected_qp(self, tau_nom, constraints, tau_min, tau_max):
        tau = np.clip(tau_nom.copy(), tau_min, tau_max)
        for _ in range(80):
            max_violation = 0.0
            for a, rhs, weight in constraints:
                denom = float(a @ a) + 1e-8
                if denom < 1e-7:
                    continue
                violation = float(rhs - a @ tau)
                max_violation = max(max_violation, violation)
                if violation > 1e-5:
                    tau += weight * (violation / denom) * a
                    tau = np.clip(tau, tau_min, tau_max)
            if max_violation <= 1e-5:
                break
        return tau

    def compute_torque(self):
        q = self.get_q()
        qdot = self.get_qdot()
        target_pos = self.get_target_pos()
        ee_pos = self.get_ee_pos()
        ee_rot = self.get_ee_rot()

        jac_pos_full = np.zeros((3, self.model.nv))
        jac_rot_full = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_pos_full, jac_rot_full, self.ee_site_id)
        J_pos = jac_pos_full[:, self.dof_indices]
        J_rot = jac_rot_full[:, self.dof_indices]
        J_full = np.vstack([J_pos, J_rot])

        pos_err = target_pos - ee_pos
        ori_err = self.orientation_error(ee_rot, self.target_orientation)
        cart_vel = J_pos @ qdot
        ang_vel = J_rot @ qdot
        wrench = np.concatenate([
            self.Kp_pos @ pos_err - self.Kd_pos @ cart_vel,
            self.Kp_ori @ ori_err - self.Kd_ori @ ang_vel,
        ])
        tau_nom = J_full.T @ wrench - np.array([4.0, 4.0, 3.0, 1.3, 1.0, 0.8]) * qdot

        # Compensate gravity and velocity-dependent generalized forces.
        tau_nom += self.data.qfrc_bias[self.dof_indices]

        # Small posture term. This is not a null-space controller; it is just damping toward
        # a comfortable shape so the 6-DoF arm stays away from singular folded poses.
        tau_nom += -np.array([3.0, 2.0, 2.0, 0.8, 0.6, 0.4]) * (q - self.q_home)

        constraints = []
        min_safety_margin = np.inf
        for obs_id in self.obstacle_body_ids:
            obs_pos = self.data.body(obs_id).xpos.copy()
            obs_vel = self.body_velocity(obs_id)
            for geom_id in self.collision_geom_ids:
                dist, direction, jac, radius = self.capsule_distance_data(
                    geom_id,
                    obs_pos,
                )
                h = dist - radius - self.ball_radius - self.safety_distance
                min_safety_margin = min(min_safety_margin, h)
                J_link = jac[:, self.dof_indices]
                rel_vel = jac @ self.data.qvel - obs_vel
                h_dot = float(direction @ rel_vel)
                a = direction @ J_link
                rhs = -self.alpha2_obs * (h_dot + self.alpha_obs * h)
                weight = 2.5 if h < 0.04 else 1.0
                constraints.append((a, rhs, weight))

        limits = [
            (ee_pos[0] - self.task_limits["x"][0], J_pos[0, :]),
            (self.task_limits["x"][1] - ee_pos[0], -J_pos[0, :]),
            (ee_pos[1] - self.task_limits["y"][0], J_pos[1, :]),
            (self.task_limits["y"][1] - ee_pos[1], -J_pos[1, :]),
            (ee_pos[2] - self.task_limits["z"][0], J_pos[2, :]),
            (self.task_limits["z"][1] - ee_pos[2], -J_pos[2, :]),
        ]
        for h, grad in limits:
            h_dot = float(grad @ qdot)
            rhs = -self.alpha2_task * (h_dot + self.alpha_task * h)
            constraints.append((grad, rhs, 0.8))

        tau_min = np.array([-27.0, -27.0, -27.0, -7.0, -7.0, -7.0])
        tau_max = -tau_min
        tau_safe = self.solve_projected_qp(tau_nom, constraints, tau_min, tau_max)
        self.last_tau_nom = tau_nom.copy()
        self.last_tau_clipped_nom = np.clip(tau_nom, tau_min, tau_max)
        self.last_tau_safe = tau_safe.copy()
        self.last_min_safety_margin = min_safety_margin
        return tau_safe

    def runBefore(self):
        for i, value in enumerate(self.q_home):
            self.data.qpos[self.joint_qpos_indices[i]] = value
        mujoco.mj_forward(self.model, self.data)
        self.target_orientation = self.get_ee_rot().copy()
        self.total_time = 300.0
        self.dt = self.model.opt.timestep
        self.num_steps = int(self.total_time / self.dt)
        self.index = 0
        print("ReBotArm DM OSCBF demo")
        print("Drag the red target ball: the end effector follows it.")
        print("Drag either obstacle ball toward the arm: the arm should avoid it.")

    def runFunc(self):
        if self.index >= self.num_steps:
            return
        tau = self.compute_torque()
        for i, aid in enumerate(self.actuator_indices):
            self.data.ctrl[aid] = tau[i]
        if self.index % 250 == 0:
            ee = self.get_ee_pos()
            tgt = self.get_target_pos()
            min_h = np.inf
            for obs_id in self.obstacle_body_ids:
                obs_pos = self.data.body(obs_id).xpos
                for geom_id in self.collision_geom_ids:
                    dist, _, _, radius = self.capsule_distance_data(geom_id, obs_pos)
                    h = dist - radius - self.ball_radius
                    min_h = min(min_h, h)
            print(
                f"[{self.index}] ee={np.round(ee, 3)} target={np.round(tgt, 3)} "
                f"err={np.linalg.norm(tgt - ee):.3f} min_obs_gap={min_h:.3f}"
            )
        self.index += 1
        time.sleep(0.001)

    def runAfter(self):
        print("Simulation finished")

    def run_loop(self):
        self.runBefore()
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=True,
            show_right_ui=True,
        ) as viewer:
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            viewer.cam.distance = 1.4
            viewer.cam.lookat[:] = [0.20, 0.0, 0.28]
            while viewer.is_running() and self.index < self.num_steps:
                step_start = time.perf_counter()
                with viewer.lock():
                    self.runFunc()
                    mujoco.mj_step(self.model, self.data)
                viewer.sync()
                remaining = self.dt - (time.perf_counter() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
        self.runAfter()


def check_environment():
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    if mujoco is None:
        print(f"MuJoCo: MISSING ({MUJOCO_IMPORT_ERROR})")
        print("Recommended interpreter:")
        print(r"  D:\Anaconda_envs\envs\mujoco_env\python.exe")
        return False

    print(f"MuJoCo: {mujoco.__version__}")
    if not os.path.isfile(URDF_PATH):
        print(f"URDF: MISSING ({URDF_PATH})")
        return False
    print(f"URDF: OK ({URDF_PATH})")

    try:
        generate_rebotarm_mjcf()
        model = load_mujoco_model(MODEL_PATH)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        ReBotArmDMCBFController(MODEL_PATH)
    except Exception as exc:
        print(f"Model/controller: FAILED ({type(exc).__name__}: {exc})")
        return False

    print(
        f"Model/controller: OK (nq={model.nq}, nv={model.nv}, "
        f"actuators={model.nu})"
    )
    print("All required dependencies and model checks passed.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReBotArm DM end-effector CBF demo")
    parser.add_argument(
        "--check",
        action="store_true",
        help="check dependencies and compile the model without opening a window",
    )
    args = parser.parse_args()
    if args.check:
        raise SystemExit(0 if check_environment() else 1)
    generate_rebotarm_mjcf()
    ctrl = ReBotArmDMCBFController(MODEL_PATH)
    ctrl.run_loop()
