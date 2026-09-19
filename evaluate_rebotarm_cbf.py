import csv
import os

import numpy as np

import oscbf_rebotarm_dm_ee_cbf as demo


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT_DIR, "rebotarm_cbf_evaluation.csv")


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def move_between(time_value, start_time, end_time, start, end):
    phase = smoothstep((time_value - start_time) / (end_time - start_time))
    return start + phase * (end - start)


def sphere_trajectory(time_value, scene):
    center = scene["target_center"]
    target = center + np.array([
        0.040 * np.sin(0.62 * time_value),
        0.035 * np.sin(0.48 * time_value),
        0.025 * np.sin(0.76 * time_value),
    ])

    obstacle_1_far = scene["obstacle_1_far"]
    obstacle_1_near = scene["obstacle_1_near"]
    if time_value < 1.5:
        obstacle_1 = obstacle_1_far
    elif time_value < 3.0:
        obstacle_1 = move_between(time_value, 1.5, 3.0, obstacle_1_far, obstacle_1_near)
    elif time_value < 4.5:
        obstacle_1 = obstacle_1_near
    else:
        obstacle_1 = move_between(time_value, 4.5, 6.0, obstacle_1_near, obstacle_1_far)

    obstacle_2_far = scene["obstacle_2_far"]
    obstacle_2_near = scene["obstacle_2_near"]
    if time_value < 3.0:
        obstacle_2 = obstacle_2_far
    elif time_value < 4.4:
        obstacle_2 = move_between(time_value, 3.0, 4.4, obstacle_2_far, obstacle_2_near)
    elif time_value < 6.3:
        obstacle_2 = obstacle_2_near
    else:
        obstacle_2 = move_between(time_value, 6.3, 7.5, obstacle_2_near, obstacle_2_far)
    return target, obstacle_1, obstacle_2


def set_sphere_state(controller, body_name, position, previous_position, dt):
    body_id = demo.mujoco.mj_name2id(
        controller.model,
        demo.mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )
    body_base = controller.model.body_pos[body_id]
    velocity = (position - previous_position) / dt
    for axis_index, axis_name in enumerate("xyz"):
        joint_id = demo.mujoco.mj_name2id(
            controller.model,
            demo.mujoco.mjtObj.mjOBJ_JOINT,
            f"{body_name}_{axis_name}",
        )
        controller.data.qpos[controller.model.jnt_qposadr[joint_id]] = (
            position[axis_index] - body_base[axis_index]
        )
        controller.data.qvel[controller.model.jnt_dofadr[joint_id]] = velocity[axis_index]


def run_evaluation(duration=8.0, sample_period=0.01):
    demo.generate_rebotarm_mjcf()
    controller = demo.ReBotArmDMCBFController(demo.MODEL_PATH)
    controller.runBefore()
    dt = controller.model.opt.timestep
    steps = int(duration / dt)
    sample_every = max(1, int(round(sample_period / dt)))
    capsule_centers = {
        demo.mujoco.mj_id2name(
            controller.model,
            demo.mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        ): controller.data.geom_xpos[geom_id].copy()
        for geom_id in controller.collision_geom_ids
    }
    first_center = capsule_centers["link2_collision"]
    last_name = controller.collision_geom_names[-1]
    last_center = capsule_centers[last_name]
    scene = {
        "target_center": controller.get_ee_pos().copy(),
        "obstacle_1_far": first_center + np.array([0.0, 0.29, 0.0]),
        "obstacle_1_near": first_center + np.array([0.0, 0.138, 0.0]),
        "obstacle_2_far": last_center + np.array([0.0, -0.29, 0.0]),
        "obstacle_2_near": last_center + np.array([0.0, -0.138, 0.0]),
    }
    previous = sphere_trajectory(0.0, scene)
    rows = []

    for step in range(steps):
        time_value = step * dt
        positions = sphere_trajectory(time_value, scene)
        for name, position, previous_position in zip(
            ("target_ball", "obstacle_ball_1", "obstacle_ball_2"),
            positions,
            previous,
        ):
            set_sphere_state(controller, name, position, previous_position, dt)
        demo.mujoco.mj_forward(controller.model, controller.data)

        torque = controller.compute_torque()
        controller.data.ctrl[controller.actuator_indices] = torque

        if step % sample_every == 0:
            ee = controller.get_ee_pos()
            target = positions[0]
            correction = np.linalg.norm(
                controller.last_tau_safe - controller.last_tau_clipped_nom
            )
            rows.append([
                time_value,
                *target,
                *ee,
                *positions[1],
                *positions[2],
                np.linalg.norm(target - ee),
                controller.last_min_safety_margin,
                correction,
            ])

        demo.mujoco.mj_step(controller.model, controller.data)
        previous = positions

    header = [
        "time_s",
        "target_x", "target_y", "target_z",
        "ee_x", "ee_y", "ee_z",
        "obstacle1_x", "obstacle1_y", "obstacle1_z",
        "obstacle2_x", "obstacle2_y", "obstacle2_z",
        "tracking_error_m",
        "min_safety_margin_m",
        "cbf_torque_correction_nm",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved {len(rows)} samples to {CSV_PATH}")


if __name__ == "__main__":
    run_evaluation()
