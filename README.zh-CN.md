# OSCBF-openarmx

<p align="center">
  <a href="./README.md"><img alt="English" src="https://img.shields.io/badge/English-README-0969da"></a>
  <a href="./README.zh-CN.md"><img alt="简体中文" src="https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-README-d73a49"></a>
</p>

本项目基于 OpenArmX 双臂平台，使用开源硬件复现操作空间控制屏障函数（OSCBF），打通从仿真到实机的完整流程。项目实现了机械臂与本体避碰、双臂互碰避免，并可进一步扩展到奇异点避免和通用障碍物避碰场景。

## 项目亮点

- 基于力矩控制的操作空间控制与 CBF 安全约束
- OpenArmX 双臂 MuJoCo 仿真及实机部署
- 支持机械臂与本体、双臂之间、外部障碍物及工作空间边界约束
- 使用 FCL 计算碰撞距离，并结合雅可比构造安全约束
- 支持带 OSCBF 安全过滤的 VR 末端遥操作
- 提供 CBF 与无 CBF 的对照示例

## 系统架构

```mermaid
flowchart LR
    A[任务指令<br/>目标位姿 / IK / VR] --> B[标称控制器<br/>操作空间阻抗或关节 PD]
    S[机器人状态<br/>q 与 q_dot] --> B
    S --> D[运动学与动力学<br/>正运动学 / 雅可比 / 重力]
    G[碰撞几何<br/>机械臂 / 本体 / 障碍物] --> E[距离查询<br/>FCL 或包络几何体]
    D --> E
    E --> F[安全函数<br/>h 与 h_dot]
    D --> F
    F --> H[CBF 约束<br/>障碍物 / 本体 / 双臂 / 工作空间]
    B --> Q[OSCBF-QP<br/>最小化与标称力矩的偏差]
    H --> Q
    D --> Q
    Q --> T[安全关节力矩 tau*]
    T --> P{执行后端}
    P -->|仿真| M[MuJoCo 双臂模型]
    P -->|实机| R[ROS 2 MIT 力矩控制器]
    M --> S
    R --> S
```

标称控制器首先根据目标运动生成跟踪力矩；与此同时，系统根据机器人状态和碰撞几何构造二阶 CBF 不等式。QP 在碰撞、工作空间和执行器力矩限制下，求出与标称力矩最接近的安全力矩，并将其发送到 MuJoCo 或真实机器人。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `oscbf_openarmx_ee_cbf.py` | 带 OSCBF 约束的双臂末端控制 |
| `oscbf_openarmx_ee_no_cbf.py` | 未启用 CBF 过滤的末端控制基线 |
| `oscbf_openarmx_double.py` | 双臂 OSCBF 仿真场景 |
| `oscbf_openarmx_no_cbf.py` | 用于效果对照的双臂基线 |
| `vr_oscbf_openarmx.py` | 带右臂 OSCBF 保护的 VR 遥操作 |
| `oscbf_mit_controller.cpp` | 实机 ROS 2 力矩控制器 |
| `openarmx_mujoco.xml` | OpenArmX 双臂 MuJoCo 模型 |
| `oscbf.pdf` | 仓库内附的 OSCBF 参考论文 |

## 快速开始

仿真代码依赖 MuJoCo、NumPy、SciPy、OSQP、Python-FCL、Robotics Toolbox for Python、SpatialMath、Transforms3D 和 MuJoCo viewer。请根据当前 Python 环境及系统 FCL 版本安装兼容的依赖，然后在仓库根目录运行：

```bash
python oscbf_openarmx_ee_cbf.py
```

可运行对应的无安全过滤版本进行对比：

```bash
python oscbf_openarmx_ee_no_cbf.py
```

VR 示例还需要 `vr_oscbf_openarmx.py` 中引用的 `telegrip` 配置和 WebSocket 服务。实机控制器需要先集成到对应的 OpenArmX ROS 2 工作空间中，再进行编译和启动。

## 效果展示

### OSCBF 论文

![OSCBF 论文](./oscbf.png)

### OpenArmX 仿真验证

![OpenArmX 仿真验证](./openarmx_mujoco_oscbf.png)

### Qijia 仿真验证

![Qijia 仿真验证](./qijia_mujoco_oscbf.png)

### OpenArmX 实机验证

![OpenArmX 实机验证](./openarmx_oscbf_real.png)

### Qijia RViz 视图

![Qijia RViz 视图](./qijia_rviz.jpg)

## 开源协议

详见 [LICENSE](./LICENSE)。
