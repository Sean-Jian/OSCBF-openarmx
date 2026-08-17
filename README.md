# OSCBF-openarmx

<p align="center">
  <a href="./README.md"><img alt="English" src="https://img.shields.io/badge/English-README-0969da"></a>
  <a href="./README.zh-CN.md"><img alt="简体中文" src="https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-README-d73a49"></a>
</p>

An open-source reproduction of Operational-Space Control Barrier Functions (OSCBF) on the OpenArmX dual-arm platform. The project covers the simulation-to-real workflow and demonstrates robot-body collision avoidance, inter-arm collision avoidance, and extensions toward singularity and general obstacle avoidance.

## Highlights

- Torque-level operational-space control with CBF safety constraints
- Dual-arm MuJoCo simulation and OpenArmX robot deployment
- Robot-body, arm-arm, obstacle, and workspace-boundary constraints
- FCL-based collision-distance queries and Jacobian-based constraint construction
- VR end-effector teleoperation with OSCBF safety filtering
- Paired CBF / no-CBF examples for comparison

## System architecture

```mermaid
flowchart LR
    A[Task command<br/>target pose / IK / VR] --> B[Nominal controller<br/>operational-space impedance or joint PD]
    S[Robot state<br/>q and q_dot] --> B
    S --> D[Kinematics and dynamics<br/>FK / Jacobian / gravity]
    G[Collision geometry<br/>robot / body / obstacle] --> E[Distance queries<br/>FCL or bounding primitives]
    D --> E
    E --> F[Safety functions<br/>h and h_dot]
    D --> F
    F --> H[CBF constraints<br/>obstacle / body / arm-arm / workspace]
    B --> Q[OSCBF-QP<br/>minimize deviation from nominal torque]
    H --> Q
    D --> Q
    Q --> T[Safe joint torque tau*]
    T --> P{Execution backend}
    P -->|Simulation| M[MuJoCo dual-arm model]
    P -->|Real robot| R[ROS 2 MIT torque controller]
    M --> S
    R --> S
```

The nominal controller generates the torque required to track the requested motion. In parallel, robot state and collision geometry are converted into second-order CBF inequalities. The QP then finds the closest admissible torque, subject to collision, workspace, and actuator-limit constraints, before sending it to either MuJoCo or the real robot.

## Repository contents

| File | Purpose |
| --- | --- |
| `oscbf_openarmx_ee_cbf.py` | Dual-arm end-effector control with OSCBF constraints |
| `oscbf_openarmx_ee_no_cbf.py` | End-effector-control baseline without active CBF filtering |
| `oscbf_openarmx_double.py` | Dual-arm OSCBF simulation scenario |
| `oscbf_openarmx_no_cbf.py` | Dual-arm baseline used for comparison |
| `vr_oscbf_openarmx.py` | VR teleoperation with right-arm OSCBF protection |
| `oscbf_mit_controller.cpp` | Real-robot ROS 2 torque controller |
| `openarmx_mujoco.xml` | OpenArmX dual-arm MuJoCo model |
| `oscbf.pdf` | Reference OSCBF paper included in this repository |

## Quick start

The simulation code depends on MuJoCo, NumPy, SciPy, OSQP, Python-FCL, Robotics Toolbox for Python, SpatialMath, Transforms3D, and a MuJoCo viewer. Install versions compatible with your Python and system FCL installation, then run a scenario from the repository root:

```bash
python oscbf_openarmx_ee_cbf.py
```

Compare it with the corresponding unfiltered baseline:

```bash
python oscbf_openarmx_ee_no_cbf.py
```

The VR example additionally expects the `telegrip` configuration and WebSocket server referenced by `vr_oscbf_openarmx.py`. The real-robot controller must be integrated into the corresponding OpenArmX ROS 2 workspace before it can be built and launched.

## Results

### OSCBF paper

![OSCBF paper](./oscbf.png)

### OpenArmX simulation

![OpenArmX simulation](./openarmx_mujoco_oscbf.png)

### Qijia simulation

![Qijia simulation](./qijia_mujoco_oscbf.png)

### OpenArmX real-robot validation

![OpenArmX real-robot validation](./openarmx_oscbf_real.png)

### Qijia RViz view

![Qijia RViz view](./qijia_rviz.jpg)

## License

See [LICENSE](./LICENSE).
