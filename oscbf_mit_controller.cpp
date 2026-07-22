// OSCBF MIT Controller - Real Robot Torque-Level Safety Control
//
// 数据流：
//   CAN读取 q_cur, qdot_cur
//     → 发布 /joint_states（供IK节点使用）
//   订阅 /left(right)_forward_position_controller/commands → q_des
//     → 关节PD: tau_nom = Kp*(q_des - q_cur) + Kd*(0 - qdot_cur)
//     → OSCBF-QP: tau* = argmin ||tau - tau_nom||² s.t. CBF约束
//     → send_motion_control_commands(tau*)

#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include <csignal>
#include <thread>
#include <mutex>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

#include <openarmx/can/socket/openarmx.hpp>
#include <openarmx/robstride_motor/rs_motor_constants.hpp>

#include "dynamics.hpp"

using namespace std::chrono_literals;

// ============================================================
// 参数配置
// ============================================================

static constexpr int    N_JOINTS     = 7;
static constexpr double CTRL_HZ      = 200.0;
static constexpr double DT           = 1.0 / CTRL_HZ;

// 关节PD增益（力矩层）- 来自官方参数速查表「常规运动(默认)」
// joint 1-4 (RS04/RS03): KP=50, KD=2.5
// joint 5-7 (RS00):      KP=10, KD=0.5
static const double KP[N_JOINTS] = {50.0, 50.0, 50.0, 50.0, 10.0, 10.0, 10.0};
static const double KD[N_JOINTS] = { 2.5,  2.5,  2.5,  2.5,  0.5,  0.5,  0.5};

// 力矩限制 (Nm) - 参考官方额定扭矩：RS04=40Nm, RS03=20Nm, RS00=5Nm
static const double TAU_MAX[N_JOINTS] = {40.0, 40.0, 20.0, 20.0, 5.0, 5.0, 5.0};

// 方向符号（与 teleop_bimanual_with_gravitycomp_single.cpp 一致）
static const double DIR_SIGNS[N_JOINTS] = {-1,-1,-1,-1,-1,-1,-1};

// 重力补偿缩放（与 gravity_comp_test 一致，稍微欠补偿更安全）
static constexpr double G_SCALE = 0.96;

// L构型目标关节角（rad）- 与 v10_simple_hardware.cpp 的初始化逻辑完全对齐
// 官方初始化：joint 1-3,5-7 归零，joint4 移到 80° = 1.3963 rad（关节坐标系）
// 注意：发给电机时会乘以 direction_multipliers（各关节符号），
//       但 joint_states 发布的是关节坐标系值，所以这里直接用关节坐标系值
static const double L_POSE[N_JOINTS] = {0.0, 0.0, 0.0, 1.3963, 0.0, 0.0, 0.0};

// 初始化阶段：位置模式 kp/kd（用于运动到L构型）
// 与 v10_simple_hardware.cpp 保持一致，使用「常规运动」档
// j4 (RS03) 降低 KP 避免大角度运动时电流冲击触发保护
static const double INIT_KP[N_JOINTS] = {50.0, 50.0, 50.0, 50.0, 10.0, 10.0, 10.0};
static const double INIT_KD[N_JOINTS] = { 2.5,  2.5,  2.5,  2.5,  0.5,  0.5,  0.5};

// 到位判断阈值不再需要（阻塞式初始化）
// static constexpr double INIT_DONE_THRESH = 0.15;

// ============================================================
// 本体立柱 CBF 参数
// ============================================================

// 本体立柱碰撞盒：6cm × 6cm × 70cm
// 半尺寸：x=0.03, y=0.03, z=0.35
// 中心高度：柱子从底部到顶部约 0~0.70m，中心在 z=0.35m（世界坐标系）
static const Eigen::Vector3d BODY_BOX_CENTER_WORLD(0.0, 0.0, 0.35);
static const Eigen::Vector3d BODY_BOX_HALF(0.03, 0.03, 0.35);

// KDL FK 结果在各自 baselink（link0）坐标系下，需要把 box 中心转换过去
// 从 openarmx_robot.urdf 读取的精确值：
//   右臂 link0 joint: xyz=(0, -0.031, 0.698), rpy=(1.5708, 0, 0) → 绕X轴+90°
//   左臂 link0 joint: xyz=(0,  0.031, 0.698), rpy=(-1.5708, 0, 0) → 绕X轴-90°
//
// 绕X轴+90°: R = [[1,0,0],[0,0,-1],[0,1,0]]
// 绕X轴-90°: R = [[1,0,0],[0,0, 1],[0,-1,0]]
//
// p_in_base = R^T * (p_world - t_base)

static const Eigen::Vector3d RIGHT_BASE_POS(0.0, -0.031, 0.698);
static const Eigen::Matrix3d RIGHT_BASE_R = (Eigen::Matrix3d() <<
    1, 0,  0,
    0, 0, -1,
    0, 1,  0).finished();

static const Eigen::Vector3d LEFT_BASE_POS(0.0, 0.031, 0.698);
static const Eigen::Matrix3d LEFT_BASE_R = (Eigen::Matrix3d() <<
    1,  0, 0,
    0,  0, 1,
    0, -1, 0).finished();

// 预计算：box中心在各自baselink坐标系下
// p_in_base = R^T * (p_world - t_base)
static const Eigen::Vector3d RIGHT_BOX_CENTER =
    RIGHT_BASE_R.transpose() * (BODY_BOX_CENTER_WORLD - RIGHT_BASE_POS);
static const Eigen::Vector3d LEFT_BOX_CENTER =
    LEFT_BASE_R.transpose() * (BODY_BOX_CENTER_WORLD - LEFT_BASE_POS);
// 注意：boxDistance 假设 box 在当前坐标系下轴对齐。
// 世界系中的立柱长轴在 z 方向；转换到左右臂 base（绕 X 轴 ±90°）后，
// 长轴会落到 base 的 y 方向，因此半尺寸也要同步变成 [x, z, y]。
static const Eigen::Vector3d BOX_HALF_IN_BASE(BODY_BOX_HALF.x(), BODY_BOX_HALF.z(), BODY_BOX_HALF.y());
static const Eigen::Vector3d LINK4_TO_LINK5_OFFSET(0.0, -0.0309, 0.126);

static constexpr double SAFETY_DIST_BODY = 0.02;  // m，检测球表面到立柱表面的安全距离
static constexpr double SAFETY_DIST_ARM_ARM = 0.02;  // m，双臂检测球表面之间的安全距离
// 边界附近名义PD增益调度：
//   球表面距本体 >= 5cm：保持原始刚度
//   球表面距本体 <= 2cm：降到下限
//   2cm ~ 5cm：连续非线性过渡；前段慢、中段正常、末段快
static constexpr double BODY_GAIN_SCHED_START = 0.05;  // m
static constexpr double BODY_GAIN_SCHED_END   = 0.02;  // m
static const double KP_MIN_BODY[N_JOINTS] = {10.0, 10.0, 10.0, 10.0, 2.0, 2.0, 2.0};
static constexpr double ARM_GAIN_SCHED_START = 0.05;  // m，双臂球表面距离 >= 5cm 保持原始刚度
static constexpr double ARM_GAIN_SCHED_END   = 0.02;  // m，双臂球表面距离 <= 2cm 降到下限
static const double KP_MIN_ARM[N_JOINTS] = {10.0, 10.0, 10.0, 10.0, 2.0, 2.0, 2.0};
static constexpr double ALPHA_CBF        = 60.0;  // 一阶增益
static constexpr double ALPHA2_CBF       = 15.0;  // 二阶增益
static constexpr double W_BODY           = 1.0;   // 修正权重
// ============================================================
// 工具函数
// ============================================================

// box有符号距离（正=外，负=穿透）
static double boxDistance(const Eigen::Vector3d& p,
                          const Eigen::Vector3d& center,
                          const Eigen::Vector3d& half)
{
    Eigen::Vector3d d = (p - center).cwiseAbs() - half;
    double outside = d.cwiseMax(0.0).norm();
    double inside  = std::min(d.maxCoeff(), 0.0);
    return outside + inside;
}

// box距离对点p的梯度
static Eigen::Vector3d boxDistanceGrad(const Eigen::Vector3d& p,
                                       const Eigen::Vector3d& center,
                                       const Eigen::Vector3d& half)
{
    Eigen::Vector3d d  = p - center;
    Eigen::Vector3d ad = d.cwiseAbs() - half;
    if (ad.maxCoeff() > 0.0) {
        // 点在box外
        Eigen::Vector3d clamped = ad.cwiseMax(0.0).cwiseProduct(d.cwiseSign());
        double norm = clamped.norm();
        if (norm > 1e-8) return Eigen::Vector3d(clamped / norm);
        return Eigen::Vector3d::Zero();
    }
    // 点在box内，梯度指向最近面
    Eigen::Vector3d grad = Eigen::Vector3d::Zero();
    int idx; ad.maxCoeff(&idx);
    grad(idx) = (d(idx) >= 0.0) ? 1.0 : -1.0;
    return grad;
}

// 对tau做clip
static void clipTorque(Eigen::VectorXd& tau)
{
    for (int i = 0; i < N_JOINTS; ++i) {
        tau(i) = std::max(-TAU_MAX[i], std::min(TAU_MAX[i], tau(i)));
    }
}

// ============================================================
// OSCBF MIT Controller Node
// ============================================================

class OSCBFMITController : public rclcpp::Node
{
public:
    OSCBFMITController()
    : Node("oscbf_mit_controller")
    {
        // ---- 参数声明 ----
        this->declare_parameter<std::string>("left_can",   "can3");
        this->declare_parameter<std::string>("right_can",  "can2");
        this->declare_parameter<std::string>("urdf_path",  "/home/robotgym/jxj/qijia-teleopvr/telegrip/telegrip/openarmx_ws/src/openarmx_description/urdf/robot/openarmx_robot.urdf");
        this->declare_parameter<bool>("enable_left",  true);
        this->declare_parameter<bool>("enable_right", true);
        this->declare_parameter<bool>("verbose",      false);
        this->declare_parameter<bool>("use_oscbf",        false); // false=正常遥操模式(position+kp+kd)
        this->declare_parameter<bool>("use_gravity_comp", false); // 仅控制 tau_des/tau_nom 是否加重力补偿
        this->declare_parameter<bool>("enable_body_collision_cbf", true);
        this->declare_parameter<bool>("enable_arm_arm_collision_cbf", false);
        // 左右臂 baselink 朝向不同，重力向量分开配置（与 gravity_comp_test 一致）
        // 左臂 rpy=(-1.5708,0,0)：g_left  = R_left^T  * {0,0,-9.81} = {0, +9.81, 0}
        // 右臂 rpy=(+1.5708,0,0)：g_right = R_right^T * {0,0,-9.81} = {0, -9.81, 0}
        this->declare_parameter<std::vector<double>>("left_gdir",  {0.0,  9.81, 0.0});
        this->declare_parameter<std::vector<double>>("right_gdir", {0.0, -9.81, 0.0});

        std::string left_can  = this->get_parameter("left_can").as_string();
        std::string right_can = this->get_parameter("right_can").as_string();
        std::string urdf_path = this->get_parameter("urdf_path").as_string();
        enable_left_  = this->get_parameter("enable_left").as_bool();
        enable_right_ = this->get_parameter("enable_right").as_bool();
        verbose_          = this->get_parameter("verbose").as_bool();
        use_oscbf_        = this->get_parameter("use_oscbf").as_bool();
        use_gravity_comp_ = this->get_parameter("use_gravity_comp").as_bool();
        enable_body_collision_cbf_ = this->get_parameter("enable_body_collision_cbf").as_bool();
        enable_arm_arm_collision_cbf_ = this->get_parameter("enable_arm_arm_collision_cbf").as_bool();
        auto left_gdir  = this->get_parameter("left_gdir").as_double_array();
        auto right_gdir = this->get_parameter("right_gdir").as_double_array();

        RCLCPP_INFO(get_logger(), "=== OSCBF MIT Controller 初始化 ===");
        RCLCPP_INFO(get_logger(), "urdf=%s  left_can=%s  right_can=%s",
                    urdf_path.c_str(), left_can.c_str(), right_can.c_str());
        RCLCPP_INFO(get_logger(), "left_gdir=[%.2f,%.2f,%.2f]  right_gdir=[%.2f,%.2f,%.2f]",
                    left_gdir[0], left_gdir[1], left_gdir[2],
                    right_gdir[0], right_gdir[1], right_gdir[2]);
        RCLCPP_INFO(get_logger(), "use_oscbf=%s  use_gravity_comp=%s",
                    use_oscbf_ ? "true" : "false",
                    use_gravity_comp_ ? "true" : "false");
        RCLCPP_INFO(get_logger(), "enable_body_collision_cbf=%s  enable_arm_arm_collision_cbf=%s",
                    enable_body_collision_cbf_ ? "true" : "false",
                    enable_arm_arm_collision_cbf_ ? "true" : "false");

        // ---- 初始化动力学（左右臂各一个）----
        if (enable_left_) {
            auto dyn = std::make_unique<Dynamics>(urdf_path,
                "openarmx_left_link0", "openarmx_left_link7");
            if (!dyn->Init()) throw std::runtime_error("左臂动力学初始化失败");
            dyn->SetGravityVector(left_gdir[0], left_gdir[1], left_gdir[2]);
            left_dyn_ = std::move(dyn);
        }
        if (enable_right_) {
            auto dyn = std::make_unique<Dynamics>(urdf_path,
                "openarmx_right_link0", "openarmx_right_link7");
            if (!dyn->Init()) throw std::runtime_error("右臂动力学初始化失败");
            dyn->SetGravityVector(right_gdir[0], right_gdir[1], right_gdir[2]);
            RCLCPP_INFO(get_logger(), "右臂 KDL chain: %zu joints, %zu segments",
                dyn->NumJoints(), dyn->NumSegments());
            right_dyn_ = std::move(dyn);
        }

        // ---- 初始化CAN总线 ----
        std::vector<openarmx::robstride_motor::MotorType> motor_types = {
            openarmx::robstride_motor::MotorType::RS04,
            openarmx::robstride_motor::MotorType::RS04,
            openarmx::robstride_motor::MotorType::RS03,
            openarmx::robstride_motor::MotorType::RS03,
            openarmx::robstride_motor::MotorType::RS00,
            openarmx::robstride_motor::MotorType::RS00,
            openarmx::robstride_motor::MotorType::RS00};
        std::vector<uint32_t> ids = {0x01,0x02,0x03,0x04,0x05,0x06,0x07};

        if (enable_left_) {
            left_bus_ = std::make_unique<openarmx::can::socket::OpenArmX>(left_can, false);
            left_bus_->init_arm_motors(motor_types, ids, ids);
            left_bus_->init_gripper_motor(openarmx::robstride_motor::MotorType::RS00, 0x08, 0x08);
            left_bus_->set_callback_mode_all(openarmx::robstride_motor::CallbackMode::STATE);
            // 与 v10 一致：先 disable，再 enable
            left_bus_->disable_all();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (!left_bus_->enable_all()) throw std::runtime_error("左臂电机使能失败");
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            left_bus_->refresh_all();
            left_bus_->recv_all();
            RCLCPP_INFO(get_logger(), "左臂CAN初始化完成: %s", left_can.c_str());
        }
        if (enable_right_) {
            right_bus_ = std::make_unique<openarmx::can::socket::OpenArmX>(right_can, false);
            right_bus_->init_arm_motors(motor_types, ids, ids);
            right_bus_->init_gripper_motor(openarmx::robstride_motor::MotorType::RS00, 0x08, 0x08);
            right_bus_->set_callback_mode_all(openarmx::robstride_motor::CallbackMode::STATE);
            // 与 v10 一致：先 disable，再 enable
            right_bus_->disable_all();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (!right_bus_->enable_all()) throw std::runtime_error("右臂电机使能失败");
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            right_bus_->refresh_all();
            right_bus_->recv_all();
            RCLCPP_INFO(get_logger(), "右臂CAN初始化完成: %s", right_can.c_str());
        }

        // ---- 状态向量初始化 ----
        left_q_.assign(N_JOINTS, 0.0);   left_qdot_.assign(N_JOINTS, 0.0);   left_tau_.assign(N_JOINTS, 0.0);
        right_q_.assign(N_JOINTS, 0.0);  right_qdot_.assign(N_JOINTS, 0.0);  right_tau_.assign(N_JOINTS, 0.0);
        left_q_des_.assign(N_JOINTS + 1, 0.0);  // 7关节 + 1夹爪
        right_q_des_.assign(N_JOINTS + 1, 0.0);
        left_q_des_valid_  = false;
        right_q_des_valid_ = false;

        // ---- ROS接口 ----
        // 发布 joint_states（供IK节点订阅）
        js_pub_ = create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
        // 发布 tau_des 和 tau_star 供数据记录
        left_tau_des_pub_  = create_publisher<std_msgs::msg::Float64MultiArray>("/left_tau_des",  10);
        right_tau_des_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/right_tau_des", 10);
        left_tau_star_pub_  = create_publisher<std_msgs::msg::Float64MultiArray>("/left_tau_star",  10);
        right_tau_star_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/right_tau_star", 10);

        // 订阅IK结果（来自官方遥操节点）
        left_cmd_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
            "/left_forward_position_controller/commands", 10,
            [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
                std::lock_guard<std::mutex> lk(cmd_mutex_);
                if ((int)msg->data.size() >= N_JOINTS) {
                    for (int i = 0; i < N_JOINTS; ++i) left_q_des_[i] = msg->data[i];
                    // 第8个元素是夹爪（单位 m）
                    if ((int)msg->data.size() > N_JOINTS)
                        left_q_des_[N_JOINTS] = msg->data[N_JOINTS];
                    left_q_des_valid_ = true;
                }
            });

        right_cmd_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
            "/right_forward_position_controller/commands", 10,
            [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
                std::lock_guard<std::mutex> lk(cmd_mutex_);
                if ((int)msg->data.size() >= N_JOINTS) {
                    for (int i = 0; i < N_JOINTS; ++i) right_q_des_[i] = msg->data[i];
                    if ((int)msg->data.size() > N_JOINTS)
                        right_q_des_[N_JOINTS] = msg->data[N_JOINTS];
                    right_q_des_valid_ = true;
                }
            });

        // 控制定时器
        auto period = std::chrono::microseconds(static_cast<int>(1e6 / CTRL_HZ));
        timer_ = create_wall_timer(period,
            std::bind(&OSCBFMITController::controlLoop, this));

        start_time_      = std::chrono::high_resolution_clock::now();
        last_hz_display_ = start_time_;

        // 初始化阶段：记录起始关节角（上电后第一次读取）
        // 先读一次当前状态作为插值起点（此时电机已稳定）
        readArmState(left_bus_.get(),  left_q_,  left_qdot_,  left_tau_,  enable_left_);
        readArmState(right_bus_.get(), right_q_, right_qdot_, right_tau_, enable_right_);
        left_q_init_  = left_q_;
        right_q_init_ = right_q_;

        // 打印初始关节角，便于确认读取正确
        if (enable_left_) {
            RCLCPP_INFO(get_logger(), "左臂初始关节角: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                left_q_[0], left_q_[1], left_q_[2], left_q_[3], left_q_[4], left_q_[5], left_q_[6]);
        }
        if (enable_right_) {
            RCLCPP_INFO(get_logger(), "右臂初始关节角: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                right_q_[0], right_q_[1], right_q_[2], right_q_[3], right_q_[4], right_q_[5], right_q_[6]);
        }

        // ================================================================
        // 初始化运动：阻塞式，与 v10_simple_hardware 完全对齐
        // 阶段1（3s）：joint 1-3,5-7 正弦缓动归零，joint4 保持不动
        // 阶段2（5s）：joint4 正弦缓动移到 80°，其余保持 0
        // ================================================================
        runBlockingInit();

        // 初始化完成后读取最新关节角更新 q，q_des 设为目标 L 构型（不用实际值，避免稳态误差累积）
        readArmState(left_bus_.get(),  left_q_,  left_qdot_,  left_tau_,  enable_left_);
        readArmState(right_bus_.get(), right_q_, right_qdot_, right_tau_, enable_right_);
        {
            for (int i = 0; i < N_JOINTS; ++i) {
                left_q_des_[i]  = (i == 3) ? L_POSE[3] : 0.0;
                right_q_des_[i] = (i == 3) ? L_POSE[3] : 0.0;
            }
            left_q_des_valid_  = true;
            right_q_des_valid_ = true;
        }
        RCLCPP_INFO(get_logger(), "✓ 已到达L构型，OSCBF MIT Controller 就绪，等待 IK commands...");
        RCLCPP_INFO(get_logger(), "左臂实际关节角: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
            left_q_[0], left_q_[1], left_q_[2], left_q_[3], left_q_[4], left_q_[5], left_q_[6]);
        RCLCPP_INFO(get_logger(), "右臂实际关节角: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
            right_q_[0], right_q_[1], right_q_[2], right_q_[3], right_q_[4], right_q_[5], right_q_[6]);
    }

    ~OSCBFMITController()
    {
        RCLCPP_INFO(get_logger(), "关闭 OSCBF MIT Controller...");
        auto relax = [](openarmx::can::socket::OpenArmX* bus) {
            if (!bus) return;
            try {
                std::vector<openarmx::robstride_motor::MotionControlParam> cmds;
                for (auto* m : bus->get_arm().get_motors()) {
                    openarmx::robstride_motor::MotionControlParam p{};
                    p.position = m->get_position();
                    p.velocity = 0.0; p.torque = 0.0; p.kp = 0.0; p.kd = 0.0;
                    cmds.push_back(p);
                }
                bus->get_arm().send_motion_control_commands(cmds);
                bus->disable_all();
            } catch (...) {}
        };
        relax(left_bus_.get());
        relax(right_bus_.get());
    }

private:
    // ----------------------------------------------------------------
    // 主控制循环
    // ----------------------------------------------------------------
    void controlLoop()
    {
        frame_count_++;
        auto now = std::chrono::high_resolution_clock::now();

        // 1. 读取CAN状态
        readArmState(left_bus_.get(),  left_q_,  left_qdot_,  left_tau_,  enable_left_);
        readArmState(right_bus_.get(), right_q_, right_qdot_, right_tau_, enable_right_);

        // 2. 发布 joint_states
        publishJointStates(now);

        // ================================================================
        // 初始化运动已在构造函数阻塞完成，直接跑 OSCBF
        runOSCBFPhase(now);

        // 频率显示
        if (verbose_) {
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_hz_display_).count();
            if (ms >= 1000) {
                auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - start_time_).count();
                double hz = frame_count_ * 1000.0 / std::max(1.0, (double)total_ms);
                RCLCPP_INFO(get_logger(), "Loop: %.1f Hz", hz);
                last_hz_display_ = now;
            }
        }
    }

    // ----------------------------------------------------------------
    // 初始化运动（阻塞式，完全照抄 v10_simple_hardware on_activate）
    // 左右臂各自独立执行，每臂的循环结构与 v10 单臂完全一致：
    //   send_motion_control_commands → recv_all(1000) → sleep(5ms)
    // ----------------------------------------------------------------
    void runBlockingInit()
    {
        auto ease_inout = [](double t) -> double {
            return 0.5 - 0.5 * std::cos(M_PI * t);
        };

        constexpr int    STEP_MS   = 5;
        constexpr int    STEPS_P1  = 600;    // 600 * 5ms = 3s
        constexpr int    STEPS_P2  = 1000;   // 1000 * 5ms = 5s
        constexpr double J4_TARGET = 1.3963; // 80°

        // pos_commands 对应电机坐标系原始位置（未乘方向符号），与 v10 的 pos_commands_ 完全一致
        // v10: pos_commands_[i] = arm_motors[i]->get_position() * direction_multipliers[i]
        // 即 pos_commands_ 已经是关节坐标系值，sendOne 里乘 DIR_SIGNS 发给电机时再转回电机坐标系
        // left_q_init_ / right_q_init_ 是 readArmState 读出的关节坐标系值（已乘 DIR_SIGNS），直接用
        std::vector<double> pos_l(N_JOINTS), pos_r(N_JOINTS);
        for (int i = 0; i < N_JOINTS; ++i) {
            pos_l[i] = left_q_init_[i];
            pos_r[i] = right_q_init_[i];
        }

        // 单臂发送函数，与 v10 的 send_motion_control_commands + recv_all 完全一致
        auto sendOne = [&](openarmx::can::socket::OpenArmX* bus,
                           const std::vector<double>& pos, bool enabled) {
            if (!bus || !enabled) return;
            std::vector<openarmx::robstride_motor::MotionControlParam> cmds;
            auto motors = bus->get_arm().get_motors();
            for (int i = 0; i < N_JOINTS && i < (int)motors.size(); ++i) {
                openarmx::robstride_motor::MotionControlParam p{};
                p.kp       = INIT_KP[i];
                p.kd       = INIT_KD[i];
                p.position = DIR_SIGNS[i] * pos[i];
                p.velocity = 0.0;
                p.torque   = 0.0;
                cmds.push_back(p);
            }
            bus->get_arm().send_motion_control_commands(cmds);
            bus->recv_all(1000);
        };

        // ================================================================
        // 阶段1：joint 1-3,5-7 归零，joint4 保持不动（3s）
        // 左右臂各自独立循环，完全照抄 v10 Phase1
        // ================================================================
        RCLCPP_INFO(get_logger(), "[Init Phase1] 关节1-3,5-7归零 (3s)...");

        // 右臂 Phase1（与 v10 一致：先右臂）
        {
            std::vector<double> starts(pos_r);
            for (int step = 1; step <= STEPS_P1; ++step) {
                double alpha = ease_inout(static_cast<double>(step) / STEPS_P1);
                for (int i = 0; i < N_JOINTS; ++i) {
                    if (i == 3) continue;
                    pos_r[i] = starts[i] * (1.0 - alpha);
                }
                sendOne(right_bus_.get(), pos_r, enable_right_);
                std::this_thread::sleep_for(std::chrono::milliseconds(STEP_MS));
            }
        }
        RCLCPP_INFO(get_logger(), "[Init Phase1] 右臂完成");

        // 左臂 Phase1
        {
            std::vector<double> starts(pos_l);
            for (int step = 1; step <= STEPS_P1; ++step) {
                double alpha = ease_inout(static_cast<double>(step) / STEPS_P1);
                for (int i = 0; i < N_JOINTS; ++i) {
                    if (i == 3) continue;
                    pos_l[i] = starts[i] * (1.0 - alpha);
                }
                sendOne(left_bus_.get(), pos_l, enable_left_);
                std::this_thread::sleep_for(std::chrono::milliseconds(STEP_MS));
            }
        }
        RCLCPP_INFO(get_logger(), "[Init Phase1] 左臂完成");

        // 阶段1结束后保持 1s 等收敛
        RCLCPP_INFO(get_logger(), "[Init Phase1] 保持零位 1s...");
        for (int step = 0; step < 200; ++step) {
            sendOne(left_bus_.get(),  pos_l, enable_left_);
            sendOne(right_bus_.get(), pos_r, enable_right_);
            std::this_thread::sleep_for(std::chrono::milliseconds(STEP_MS));
        }

        // ================================================================
        // 阶段2：joint4 移到 80°，其余保持 0（5s）
        // 左右臂各自独立循环，完全照抄 v10 Phase2
        // ================================================================
        RCLCPP_INFO(get_logger(), "[Init Phase2] joint4移到80° (5s)，起点 L=%.3f R=%.3f",
                    pos_l[3], pos_r[3]);

        // 右臂 Phase2（先右臂）
        {
            double j4_start = pos_r[3];
            for (int step = 1; step <= STEPS_P2; ++step) {
                double alpha = ease_inout(static_cast<double>(step) / STEPS_P2);
                pos_r[3] = j4_start + alpha * (J4_TARGET - j4_start);
                sendOne(right_bus_.get(), pos_r, enable_right_);
                std::this_thread::sleep_for(std::chrono::milliseconds(STEP_MS));
            }
        }
        RCLCPP_INFO(get_logger(), "[Init Phase2] 右臂完成");

        // 左臂 Phase2
        {
            double j4_start = pos_l[3];
            for (int step = 1; step <= STEPS_P2; ++step) {
                double alpha = ease_inout(static_cast<double>(step) / STEPS_P2);
                pos_l[3] = j4_start + alpha * (J4_TARGET - j4_start);
                sendOne(left_bus_.get(), pos_l, enable_left_);
                std::this_thread::sleep_for(std::chrono::milliseconds(STEP_MS));
            }
        }
        RCLCPP_INFO(get_logger(), "[Init Phase2] 左臂完成，joint4=%.3f rad", J4_TARGET);
    }

    // ----------------------------------------------------------------
    // OSCBF 阶段：订阅IK结果 → 关节PD + 重力补偿 → OSCBF-QP → 力矩
    // use_oscbf_=false 时：直接发 position+kp+kd（电机内部PD，与原遥操一致）
    // use_gravity_comp_ 仅控制 tau_des/tau_nom 是否加重力补偿，与 use_oscbf_ 无关
    // 两种模式下均打印 tau_des 和 tau_star 供对比
    // ----------------------------------------------------------------
    void runOSCBFPhase(const std::chrono::high_resolution_clock::time_point& /*now*/)
    {
        std::vector<double> left_q_des(N_JOINTS), right_q_des(N_JOINTS);
        bool left_valid, right_valid;
        {
            std::lock_guard<std::mutex> lk(cmd_mutex_);
            left_q_des  = left_q_des_;
            right_q_des = right_q_des_;
            left_valid  = left_q_des_valid_;
            right_valid = right_q_des_valid_;
        }

        if (!left_valid || !right_valid) return;

        if (!use_oscbf_) {
            // ---- 正常遥操模式：position+kp+kd，电机内部PD ----
            auto sendPosKpKd = [&](openarmx::can::socket::OpenArmX* bus,
                                   const std::vector<double>& q_des,
                                   bool enabled) {
                if (!bus || !enabled) return;
                // 7 个关节
                std::vector<openarmx::robstride_motor::MotionControlParam> cmds;
                auto motors = bus->get_arm().get_motors();
                for (int i = 0; i < N_JOINTS && i < (int)motors.size(); ++i) {
                    openarmx::robstride_motor::MotionControlParam p{};
                    p.position = DIR_SIGNS[i] * q_des[i];
                    p.velocity = 0.0;
                    p.torque   = 0.0;
                    p.kp       = KP[i];
                    p.kd       = KD[i];
                    cmds.push_back(p);
                }
                bus->get_arm().send_motion_control_commands(cmds);
                // 夹爪：commands 第8个元素（index 7），单位 m，转换为电机 rad
                // motor_rad = joint_m / 0.044 * 1.0472（与 v10_simple_hardware 一致）
                if ((int)q_des.size() > N_JOINTS) {
                    double gripper_joint_m = q_des[N_JOINTS];
                    double motor_rad = gripper_joint_m / 0.044 * 1.0472;
                    openarmx::robstride_motor::MotionControlParam gp{};
                    gp.position = motor_rad;
                    gp.velocity = 0.0;
                    gp.torque   = 0.0;
                    gp.kp       = 5.0;
                    gp.kd       = 0.5;
                    bus->get_gripper().send_motion_control_commands({gp});
                }
            };
            if (enable_left_)
                sendPosKpKd(left_bus_.get(),  left_q_des,  enable_left_);
            if (enable_right_)
                sendPosKpKd(right_bus_.get(), right_q_des, enable_right_);
            // 计算 tau_des / tau_star 仅用于打印观察（不发送）
            if (enable_left_ && left_dyn_)
                computeOSCBFTorque(left_q_, left_qdot_, left_q_des, left_dyn_.get(), true);
            if (enable_right_ && right_dyn_)
                computeOSCBFTorque(right_q_, right_qdot_, right_q_des, right_dyn_.get(), false);
        } else {
            // ---- OSCBF 力矩模式 ----
            if (enable_left_ && left_dyn_) {
                auto tau = computeOSCBFTorque(left_q_, left_qdot_, left_q_des,
                                              left_dyn_.get(), true);
                sendTorque(left_bus_.get(), tau);
            }
            if (enable_right_ && right_dyn_) {
                auto tau = computeOSCBFTorque(right_q_, right_qdot_, right_q_des,
                                              right_dyn_.get(), false);
                sendTorque(right_bus_.get(), tau);
            }
        }
    }

    // ----------------------------------------------------------------
    // 读取单臂CAN状态 → q, qdot（应用方向符号）
    // ----------------------------------------------------------------
    void readArmState(openarmx::can::socket::OpenArmX* bus,
                      std::vector<double>& q,
                      std::vector<double>& qdot,
                      std::vector<double>& tau,
                      bool enabled)
    {
        if (!bus || !enabled) return;
        try {
            bus->recv_all(100);  // 超时缩短，减少阻塞
            auto motors = bus->get_arm().get_motors();
            for (int i = 0; i < N_JOINTS && i < (int)motors.size(); ++i) {
                double s = DIR_SIGNS[i];
                q[i]    = s * motors[i]->get_position();
                qdot[i] = s * motors[i]->get_velocity();
                tau[i]  = s * motors[i]->get_torque();
            }
        } catch (const std::exception& e) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "读取CAN状态失败: %s", e.what());
        }
    }

    // ----------------------------------------------------------------
    // 发布 /joint_states（左右臂7+7关节）
    // ----------------------------------------------------------------
    void publishJointStates(const std::chrono::high_resolution_clock::time_point& now)
    {
        sensor_msgs::msg::JointState msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = "base_link";

        // 读夹爪位置（motor rad → joint m，与 v10_simple_hardware 一致）
        // motor 0 rad=closed→joint 0m, motor -1.0472 rad=open→joint 0.044m
        auto gripper_pos = [](openarmx::can::socket::OpenArmX* bus) -> double {
            if (!bus) return 0.0;
            try {
                auto motors = bus->get_gripper().get_motors();
                if (!motors.empty()) {
                    double motor_rad = motors[0]->get_position();
                    return 0.044 * (motor_rad / 1.0472);
                }
            } catch (...) {}
            return 0.0;
        };
        double left_gripper  = gripper_pos(left_bus_.get());
        double right_gripper = gripper_pos(right_bus_.get());

        // 左臂 7 关节 + 夹爪
        for (int i = 0; i < N_JOINTS; ++i) {
            msg.name.push_back("openarmx_left_joint" + std::to_string(i + 1));
            msg.position.push_back(left_q_[i]);
            msg.velocity.push_back(left_qdot_[i]);
            msg.effort.push_back(left_tau_[i]);
        }
        msg.name.push_back("openarmx_left_finger_joint1");
        msg.position.push_back(left_gripper);
        msg.velocity.push_back(0.0);
        msg.effort.push_back(0.0);

        // 右臂 7 关节 + 夹爪
        for (int i = 0; i < N_JOINTS; ++i) {
            msg.name.push_back("openarmx_right_joint" + std::to_string(i + 1));
            msg.position.push_back(right_q_[i]);
            msg.velocity.push_back(right_qdot_[i]);
            msg.effort.push_back(right_tau_[i]);
        }
        msg.name.push_back("openarmx_right_finger_joint1");
        msg.position.push_back(right_gripper);
        msg.velocity.push_back(0.0);
        msg.effort.push_back(0.0);

        js_pub_->publish(msg);
        (void)now;
    }

    // ----------------------------------------------------------------
    // 核心：计算 OSCBF 力矩
    //
    // 步骤：
    //   1. 关节PD → tau_nom
    //   2. 重力补偿 → tau_nom += tau_g
    //   3. 对每个检测link构造二阶CBF约束
    //   4. 投影QP（解析解）→ tau*
    // ----------------------------------------------------------------
    Eigen::VectorXd computeOSCBFTorque(
        const std::vector<double>& q_vec,
        const std::vector<double>& qdot_vec,
        const std::vector<double>& q_des_vec,
        Dynamics* dyn,
        bool is_left_arm)
    {
        Eigen::VectorXd q(N_JOINTS), qdot(N_JOINTS), q_des(N_JOINTS);
        for (int i = 0; i < N_JOINTS; ++i) {
            q(i)     = q_vec[i];
            qdot(i)  = qdot_vec[i];
            q_des(i) = q_des_vec[i];
        }

        // 根据左右臂选择对应 baselink 坐标系下的 box 中心
        const Eigen::Vector3d& box_center = is_left_arm ? LEFT_BOX_CENTER : RIGHT_BOX_CENTER;
        const Eigen::Matrix3d& self_base_R = is_left_arm ? LEFT_BASE_R : RIGHT_BASE_R;
        const Eigen::Vector3d& self_base_pos = is_left_arm ? LEFT_BASE_POS : RIGHT_BASE_POS;
        const Eigen::Matrix3d& other_base_R = is_left_arm ? RIGHT_BASE_R : LEFT_BASE_R;
        const Eigen::Vector3d& other_base_pos = is_left_arm ? RIGHT_BASE_POS : LEFT_BASE_POS;

        // ---- 3. 构造 CBF 约束（检测 link4~link7，每个用精确雅可比）----
        // KDL segment 索引（从0开始）：
        //   该链路中包含一个固定段 openarmx_*_link4_ext，因此从 link5 开始索引会整体 +1。
        //   chain: link1, link2, link3, link4, link4_ext, link5, link6, link7
        //   link4 = seg3, link5 = seg5, link6 = seg6, link7 = seg7（末端）
        // GetJacobianAtSegment(q, seg_idx) 返回截止到该segment的雅可比

        // ----------------------------------------------------------------
        // 检测点定义：关节原点 + 杆件中段偏移点
        // 偏移量来自URDF各关节xyz的一半（在父link坐标系下）
        //
        // KDL chain seg_idx: 0=link1, 1=link2, 2=link3, 3=link4, 4=link5, 5=link6, 6=link7(EE)
        // link1~link3 有硬物理限制不会碰立柱，只检测 link4~link7
        //
        // 杆件中段偏移（link自身坐标系下，即下一关节xyz的一半）：
        //   link4→link5: joint5 xyz=(0, -0.0309, 0.126) → 中点=(0, -0.01545, 0.063)
        //   link5→link6: joint6 xyz=(0.037426, 0, 0.131) → 中点=(0.018713, 0, 0.0655)
        //   link6→link7: joint7 xyz=(-0.0375, 0, 0) → 杆件太短(3.75cm)，不加中点
        //   link7末端:   末端执行器约在link7坐标系z=0.10处
        // ----------------------------------------------------------------
        struct CheckPoint {
            int seg_idx;
            Eigen::Vector3d offset;  // 在该link坐标系下的偏移（零向量=关节原点）
            double sphere_radius;    // 检测球半径（m）
            const char* name;
        };
        static const CheckPoint CHECK_POINTS[] = {
            // 最终采用 5 个粗包络球，位置已在 Mujoco 中按实物外形人工校准。
            // 注意：以下 offset 都保持与 Mujoco 可视化中确认过的位置一致。
            // link4 两个球是按 link4 根部坐标系定的，后面会在运行时换到 KDL seg3 末端 frame。
            {3, Eigen::Vector3d(-1.060656e-07, -3.049987e-02, 2.260000e-02), 0.050, "link4_a"},
            {3, Eigen::Vector3d(-1.000000e-02, -3.070000e-02, 8.400000e-02), 0.050, "link4_b"},
            // link5：一个直径 10cm 球，包住 link5 / J6 电机区域。
            {5, Eigen::Vector3d(-4.999853e-03, -6.364003e-08, 3.999991e-02), 0.050, "link5_j6"},
            // link6：一个直径 9cm 球，球心在 J7 电机中心。
            {6, Eigen::Vector3d(-0.0375, 0.0, 0.0),                                0.045, "link6_j7"},
            // link7：一个直径 8cm 球，球心在 J8 电机中心附近。
            {7, Eigen::Vector3d(0.0, 0.0, 0.08),                                   0.040, "link7_j8"},
        };
        static constexpr int N_CHECK_POINTS = sizeof(CHECK_POINTS) / sizeof(CHECK_POINTS[0]);

        std::vector<Eigen::VectorXd> A_rows;
        std::vector<double>          b_vals;

        // 收集每个检测点的信息，用于后面统一打印
        struct PointInfo {
            const char* name;
            Eigen::Vector3d p;
            double raw_dist;
            double margin;
            double h;
            double Lf_h;
            double h2;
            bool active;  // h < 0.1
        };
        std::vector<PointInfo> point_infos;
        point_infos.reserve(N_CHECK_POINTS);
        double min_body_surface_dist = std::numeric_limits<double>::infinity();
        double min_arm_surface_dist = std::numeric_limits<double>::infinity();

        struct PairInfo {
            std::string name;
            double center_dist;
            double margin;
            double h;
            double Lf_h;
            double h2;
            bool active;
        };
        std::vector<PairInfo> pair_infos;

        struct PointKinematics {
            const CheckPoint* cp;
            Eigen::Vector3d p;
            Eigen::MatrixXd J_cp;
        };
        std::vector<PointKinematics> self_points;
        self_points.reserve(N_CHECK_POINTS);

        // 打印总计数器（左右臂各自）
        static int dbg_cnt_l = 0, dbg_cnt_r = 0;        int& dbg_cnt = is_left_arm ? dbg_cnt_l : dbg_cnt_r;
        bool do_print = (++dbg_cnt >= 500);
        if (do_print) dbg_cnt = 0;

        for (int ci = 0; ci < N_CHECK_POINTS; ++ci) {
            const auto& cp = CHECK_POINTS[ci];

            // FK：获取该segment的旋转矩阵R和原点位置p
            Eigen::Matrix3d R_cp; Eigen::Vector3d p_origin;
            dyn->GetCoordinateAtSegment(q_vec.data(), cp.seg_idx, R_cp, p_origin);

            // 实际检测点 = segment原点 + R * offset。
            // 其中 KDL 的 seg3（link4）返回在 segment 末端（接近 joint5）；
            // 但 link4 两个检测球的 offset 是按 link4 根部坐标系定的，因此这里补一次固定换元。
            Eigen::Vector3d offset_seg = cp.offset;
            if (cp.seg_idx == 3) {
                offset_seg -= LINK4_TO_LINK5_OFFSET;
            }
            Eigen::Vector3d p_cp = p_origin + R_cp * offset_seg;

            double raw_dist = boxDistance(p_cp, box_center, BOX_HALF_IN_BASE);
            double surf_dist = raw_dist - cp.sphere_radius;
            min_body_surface_dist = std::min(min_body_surface_dist, surf_dist);
            double margin = SAFETY_DIST_BODY + cp.sphere_radius;
            double h = raw_dist - margin;

            // 先记录基本信息，Lf_h/h2 在激活时才有
            PointInfo info;
            info.name   = cp.name;
            info.p      = p_cp;
            info.raw_dist = raw_dist;
            info.margin = margin;
            info.h      = h;
            info.Lf_h   = 0.0;
            info.h2     = 0.0;
            info.active = (h < 0.1);

            // 雅可比：对偏移后的点 p_cp 求 dp/dq
            // J_cp = J_origin_linear + [z_i × (R*offset)]_i 的修正项
            // 但更简单：直接用 GetJacobianAtSegment 得到原点雅可比，
            // 然后加上偏移引起的修正：J_p = J_v + J_w × offset_world
            // 其中 J_w 是角速度雅可比（后3行），offset_world = R * offset
            Eigen::MatrixXd J_full(6, N_JOINTS);
            dyn->GetJacobianAtSegment(q_vec.data(), cp.seg_idx, J_full);
            Eigen::MatrixXd J_v = J_full.topRows(3);     // 线速度雅可比 (3×7)
            Eigen::MatrixXd J_w = J_full.bottomRows(3);  // 角速度雅可比 (3×7)

            // 偏移修正：p_cp的速度 = v_origin + omega × offset_world
            // dp/dq = J_v + skew(offset_world) * J_w  （注意叉积顺序）
            Eigen::Vector3d offset_world = R_cp * offset_seg;
            // skew(a) * b = a × b，所以 omega × offset = J_w^T 列 × offset
            // 逐列：(J_w[:,i] × offset_world) 加到 J_v[:,i]
            Eigen::MatrixXd J_cp(3, N_JOINTS);
            for (int i = 0; i < N_JOINTS; ++i) {
                Eigen::Vector3d jw_col = J_w.col(i);
                J_cp.col(i) = J_v.col(i) + jw_col.cross(offset_world);
            }

            // 双臂互碰约束只看最后 3 个球，且应独立于本体约束的激活阈值收集。
            if (cp.seg_idx >= 5) {
                Eigen::Vector3d p_cp_world = self_base_R * p_cp + self_base_pos;
                Eigen::MatrixXd J_cp_world = self_base_R * J_cp;
                self_points.push_back(PointKinematics{&cp, p_cp_world, J_cp_world});
            }

            if (!enable_body_collision_cbf_) {
                continue;
            }
            if (h >= 0.1) {
                point_infos.push_back(info);
                continue;  // 距安全边界 >10cm 时跳过本体约束计算
            }
            Eigen::Vector3d grad_h = boxDistanceGrad(p_cp, box_center, BOX_HALF_IN_BASE);
            Eigen::VectorXd Lg_h = J_cp.transpose() * grad_h;
            double Lf_h = Lg_h.dot(qdot);

            double h2  = Lf_h + ALPHA_CBF * h;
            double rhs = -ALPHA2_CBF * h2;

            info.Lf_h = Lf_h;
            info.h2   = h2;
            point_infos.push_back(info);

            A_rows.push_back(Lg_h);
            b_vals.push_back(rhs);

            if (verbose_) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 200,
                    "[CBF] %s h=%.4f Lf_h=%.3f h2=%.3f",
                    cp.name, h, Lf_h, h2);
            }
        }

        if (enable_arm_arm_collision_cbf_) {
            const std::vector<double>& q_other_vec = is_left_arm ? right_q_ : left_q_;
            const std::vector<double>& qdot_other_vec = is_left_arm ? right_qdot_ : left_qdot_;
            Dynamics* dyn_other = is_left_arm ? right_dyn_.get() : left_dyn_.get();

            if (dyn_other && q_other_vec.size() >= N_JOINTS && qdot_other_vec.size() >= N_JOINTS) {
                Eigen::VectorXd qdot_other(N_JOINTS);
                for (int i = 0; i < N_JOINTS; ++i) qdot_other(i) = qdot_other_vec[i];

                std::vector<PointKinematics> other_points;
                other_points.reserve(3);
                for (int ci = 2; ci < N_CHECK_POINTS; ++ci) {
                    const auto& cp = CHECK_POINTS[ci];
                    Eigen::Matrix3d R_cp; Eigen::Vector3d p_origin;
                    dyn_other->GetCoordinateAtSegment(q_other_vec.data(), cp.seg_idx, R_cp, p_origin);

                    Eigen::Vector3d p_cp = p_origin + R_cp * cp.offset;
                    Eigen::MatrixXd J_full(6, N_JOINTS);
                    dyn_other->GetJacobianAtSegment(q_other_vec.data(), cp.seg_idx, J_full);
                    Eigen::MatrixXd J_v = J_full.topRows(3);
                    Eigen::MatrixXd J_w = J_full.bottomRows(3);
                    Eigen::Vector3d offset_world = R_cp * cp.offset;
                    Eigen::MatrixXd J_cp(3, N_JOINTS);
                    for (int i = 0; i < N_JOINTS; ++i) {
                        Eigen::Vector3d jw_col = J_w.col(i);
                        J_cp.col(i) = J_v.col(i) + jw_col.cross(offset_world);
                    }
                    Eigen::Vector3d p_cp_world = other_base_R * p_cp + other_base_pos;
                    Eigen::MatrixXd J_cp_world = other_base_R * J_cp;
                    other_points.push_back(PointKinematics{&cp, p_cp_world, J_cp_world});
                }

                for (const auto& self_pt : self_points) {
                    if (self_pt.cp->seg_idx < 5) continue;
                    for (const auto& other_pt : other_points) {
                        Eigen::Vector3d diff = self_pt.p - other_pt.p;
                        double center_dist = diff.norm();
                        double surf_dist = center_dist - self_pt.cp->sphere_radius - other_pt.cp->sphere_radius;
                        min_arm_surface_dist = std::min(min_arm_surface_dist, surf_dist);
                        double margin = self_pt.cp->sphere_radius + other_pt.cp->sphere_radius + SAFETY_DIST_ARM_ARM;

                        PairInfo pinfo;
                        pinfo.name = std::string(self_pt.cp->name) + " <-> " + other_pt.cp->name;
                        pinfo.center_dist = center_dist;
                        pinfo.margin = margin;
                        pinfo.h = center_dist - margin;
                        pinfo.Lf_h = 0.0;
                        pinfo.h2 = 0.0;
                        pinfo.active = (pinfo.h < 0.1);

                        if (center_dist < 1e-8 || pinfo.h >= 0.1) {
                            pair_infos.push_back(pinfo);
                            continue;
                        }

                        Eigen::Vector3d grad = diff / center_dist;
                        Eigen::VectorXd Lg_h = self_pt.J_cp.transpose() * grad;
                        double Lf_h = grad.dot(self_pt.J_cp * qdot - other_pt.J_cp * qdot_other);
                        double h2 = Lf_h + ALPHA_CBF * pinfo.h;
                        double rhs = -ALPHA2_CBF * h2;

                        pinfo.Lf_h = Lf_h;
                        pinfo.h2 = h2;
                        pair_infos.push_back(pinfo);

                        A_rows.push_back(Lg_h);
                        b_vals.push_back(rhs);
                    }
                }
            }
        }

        // ---- 1. 关节PD名义力矩（靠近本体/另一只手臂时降低刚度）----
        // 这里只调“内部计算 tau_des/tau_nom 用的 PD”，不影响 use_oscbf=false 时
        // 正常遥操那条 position+kp+kd 发送链路。
        auto cubic_ease_out = [](double x) {
            return 1.0 - std::pow(1.0 - x, 3.0);
        };
        auto cubic_fall = [](double x) {
            return x * x * x;
        };

        double body_gain_scale = 1.0;
        if (enable_body_collision_cbf_ && std::isfinite(min_body_surface_dist)) {
            if (min_body_surface_dist <= BODY_GAIN_SCHED_END) {
                body_gain_scale = 0.0;
            } else if (min_body_surface_dist < BODY_GAIN_SCHED_START) {
                double x =
                    (min_body_surface_dist - BODY_GAIN_SCHED_END) /
                    (BODY_GAIN_SCHED_START - BODY_GAIN_SCHED_END);
                body_gain_scale = cubic_ease_out(x);
            }
        }

        double arm_gain_scale = 1.0;
        if (enable_arm_arm_collision_cbf_ && std::isfinite(min_arm_surface_dist)) {
            if (min_arm_surface_dist <= ARM_GAIN_SCHED_END) {
                arm_gain_scale = 0.0;
            } else if (min_arm_surface_dist < ARM_GAIN_SCHED_START) {
                double x =
                    (min_arm_surface_dist - ARM_GAIN_SCHED_END) /
                    (ARM_GAIN_SCHED_START - ARM_GAIN_SCHED_END);
                // 双臂场景更容易在临界区互相顶住，因此这里用更“狠”的连续下降曲线：
                // x=1(5cm) -> 1，x=0(2cm) -> 0，且在中后段会更快压低到接近 0。
                arm_gain_scale = cubic_fall(x);
            }
        }

        double combined_gain_scale = 1.0;
        if (enable_body_collision_cbf_) combined_gain_scale = std::min(combined_gain_scale, body_gain_scale);
        if (enable_arm_arm_collision_cbf_) combined_gain_scale = std::min(combined_gain_scale, arm_gain_scale);

        Eigen::VectorXd tau_nom(N_JOINTS);
        Eigen::VectorXd kp_eff_dbg(N_JOINTS);
        for (int i = 0; i < N_JOINTS; ++i) {
            double kp_eff = KP[i];
            double kd_eff = KD[i];
            if (enable_body_collision_cbf_ || enable_arm_arm_collision_cbf_) {
                double kp_min = KP[i];
                if (enable_body_collision_cbf_ && enable_arm_arm_collision_cbf_) {
                    kp_min = std::min(KP_MIN_BODY[i], KP_MIN_ARM[i]);
                } else if (enable_body_collision_cbf_) {
                    kp_min = KP_MIN_BODY[i];
                } else if (enable_arm_arm_collision_cbf_) {
                    kp_min = KP_MIN_ARM[i];
                }
                kp_eff = kp_min + combined_gain_scale * (KP[i] - kp_min);
            }
            kp_eff_dbg(i) = kp_eff;
            tau_nom(i) = kp_eff * (q_des(i) - q(i)) + kd_eff * (0.0 - qdot(i));
        }

        // ---- 2. 重力补偿（仅由 use_gravity_comp_ 控制；与 use_oscbf_ 无关）----
        if (use_gravity_comp_) {
            std::vector<double> tau_g_vec(N_JOINTS, 0.0);
            dyn->GetGravity(q_vec.data(), tau_g_vec.data());
            for (int i = 0; i < N_JOINTS; ++i)
                tau_nom(i) += G_SCALE * tau_g_vec[i];
        }

        clipTorque(tau_nom);

        // ---- 4. QP求解（解析投影法）----
        // 无约束最优解 = tau_nom
        // 对每个违反的约束做最小修正投影
        Eigen::VectorXd tau_star = tau_nom;

        for (size_t k = 0; k < A_rows.size(); ++k) {
            const Eigen::VectorXd& a = A_rows[k];
            double b = b_vals[k];

            double slack = a.dot(tau_star) - b;
            if (slack >= 0.0) continue;  // 约束已满足

            // 最小修正：tau* += (b - a·tau) / (a·a) * a
            double a_norm2 = a.dot(a);
            if (a_norm2 < 1e-10) continue;
            tau_star += W_BODY * (-slack / a_norm2) * a;
        }

        clipTorque(tau_star);
        bool any_active = !A_rows.empty();

        // ---- 综合打印（每500帧约2.5s，左右臂独立计数）----
        if (do_print) {
            const char* arm = is_left_arm ? "LEFT" : "RIGHT";
            bool cbf_modified = any_active && (tau_star - tau_nom).norm() > 1e-4;
            bool cbf_applied = use_oscbf_;

            printf("──── [%s CBF状态] ────────────────────────────────\n", arm);
            if (enable_body_collision_cbf_) {
                printf("  box_center  =[%6.3f,%6.3f,%6.3f]  half=[%5.3f,%5.3f,%5.3f]\n",
                    box_center(0), box_center(1), box_center(2),
                    BOX_HALF_IN_BASE(0), BOX_HALF_IN_BASE(1), BOX_HALF_IN_BASE(2));
                if (std::isfinite(min_body_surface_dist)) {
                    printf("  gain_sched(body)  surf_min=%6.3f  scale=%5.3f  window=[%4.2f -> %4.2f]m\n",
                        min_body_surface_dist, body_gain_scale,
                        BODY_GAIN_SCHED_START, BODY_GAIN_SCHED_END);
                }
                for (const auto& info : point_infos) {
                    if (info.active) {
                        printf("  %-12s p=[%6.3f,%6.3f,%6.3f]  d_box=%6.3f  margin=%6.3f  h=%7.4f  Lf_h=%6.3f  h2=%6.3f  <<激活>>\n",
                            info.name,
                            info.p(0), info.p(1), info.p(2),
                            info.raw_dist, info.margin,
                            info.h, info.Lf_h, info.h2);
                    } else {
                        printf("  %-12s p=[%6.3f,%6.3f,%6.3f]  d_box=%6.3f  margin=%6.3f  h=%7.4f\n",
                            info.name,
                            info.p(0), info.p(1), info.p(2),
                            info.raw_dist, info.margin,
                            info.h);
                    }
                }
            }
            if (enable_arm_arm_collision_cbf_) {
                if (std::isfinite(min_arm_surface_dist)) {
                    printf("  gain_sched(arm)   surf_min=%6.3f  scale=%5.3f  window=[%4.2f -> %4.2f]m\n",
                        min_arm_surface_dist, arm_gain_scale,
                        ARM_GAIN_SCHED_START, ARM_GAIN_SCHED_END);
                }
                for (const auto& info : pair_infos) {
                    printf("  %-22s d_cc=%6.3f  margin=%6.3f  h=%7.4f  Lf_h=%6.3f  h2=%6.3f  %s\n",
                        info.name.c_str(), info.center_dist, info.margin, info.h, info.Lf_h, info.h2,
                        info.active ? "<<双臂>>" : "");
                }
            }
            if (enable_body_collision_cbf_ || enable_arm_arm_collision_cbf_) {
                printf("  kp_eff:           j1=%5.1f j2=%5.1f j3=%5.1f j4=%5.1f j5=%4.1f j6=%4.1f j7=%4.1f\n",
                    kp_eff_dbg(0), kp_eff_dbg(1), kp_eff_dbg(2), kp_eff_dbg(3),
                    kp_eff_dbg(4), kp_eff_dbg(5), kp_eff_dbg(6));
            }
            // tau_des
            printf("  tau_des:  j1=%5.2f j2=%5.2f j3=%5.2f j4=%5.2f j5=%5.2f j6=%5.2f j7=%5.2f\n",
                tau_nom(0),tau_nom(1),tau_nom(2),tau_nom(3),tau_nom(4),tau_nom(5),tau_nom(6));
            // tau_star 始终打印；当 use_oscbf=false 时，仅作为监视结果，不会发送到电机。
            printf("  tau_star: j1=%5.2f j2=%5.2f j3=%5.2f j4=%5.2f j5=%5.2f j6=%5.2f j7=%5.2f  %s%s\n",
                tau_star(0),tau_star(1),tau_star(2),tau_star(3),tau_star(4),tau_star(5),tau_star(6),
                cbf_modified && cbf_applied ? "<<CBF修正介入>>" : "",
                !cbf_applied ? "<<仅监视，不介入>>" : "");
            if (cbf_modified && cbf_applied) {
                Eigen::VectorXd delta = tau_star - tau_nom;
                printf("  修正量:   j1=%5.2f j2=%5.2f j3=%5.2f j4=%5.2f j5=%5.2f j6=%5.2f j7=%5.2f\n",
                    delta(0),delta(1),delta(2),delta(3),delta(4),delta(5),delta(6));
            }
            printf("─────────────────────────────────────────────────\n");
        }

        // 发布 tau_des 和 tau_star 到 ROS 话题（供数据记录）
        auto pub_vec = [](rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr& pub,
                          const Eigen::VectorXd& v) {
            std_msgs::msg::Float64MultiArray msg;
            msg.data.assign(v.data(), v.data() + v.size());
            pub->publish(msg);
        };
        if (is_left_arm) {
            pub_vec(left_tau_des_pub_,  tau_nom);
            pub_vec(left_tau_star_pub_, tau_star);
        } else {
            pub_vec(right_tau_des_pub_,  tau_nom);
            pub_vec(right_tau_star_pub_, tau_star);
        }

        return tau_star;
    }

    // ----------------------------------------------------------------
    // 发送力矩到电机（MIT模式：kp=kd=0，纯力矩）
    // position 传当前电机位置，防止触发位置保护
    // ----------------------------------------------------------------
    void sendTorque(openarmx::can::socket::OpenArmX* bus,
                    const Eigen::VectorXd& tau)
    {
        if (!bus) return;
        try {
            std::vector<openarmx::robstride_motor::MotionControlParam> cmds;
            auto motors = bus->get_arm().get_motors();
            for (int i = 0; i < N_JOINTS && i < (int)motors.size(); ++i) {
                openarmx::robstride_motor::MotionControlParam p{};
                p.position = motors[i]->get_position();  // 当前电机位置
                p.velocity = 0.0;
                p.kp       = 0.0;
                p.kd       = 0.0;
                // 应用方向符号（电机坐标系）
                p.torque   = 1.0 * DIR_SIGNS[i] * tau(i);
                cmds.push_back(p);
            }
            bus->get_arm().send_motion_control_commands(cmds);
        } catch (const std::exception& e) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "发送力矩失败: %s", e.what());
        }
    }

    // ---- 成员变量 ----
    bool enable_left_  = true;
    bool enable_right_ = true;
    bool verbose_      = false;
    bool use_oscbf_        = false;
    bool use_gravity_comp_ = false;
    bool enable_body_collision_cbf_ = true;
    bool enable_arm_arm_collision_cbf_ = false;

    std::unique_ptr<openarmx::can::socket::OpenArmX> left_bus_;
    std::unique_ptr<openarmx::can::socket::OpenArmX> right_bus_;
    std::unique_ptr<Dynamics> left_dyn_;
    std::unique_ptr<Dynamics> right_dyn_;

    std::vector<double> left_q_,  left_qdot_,  left_tau_,  left_q_des_;
    std::vector<double> right_q_, right_qdot_, right_tau_, right_q_des_;
    bool left_q_des_valid_  = false;
    bool right_q_des_valid_ = false;
    std::mutex cmd_mutex_;

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr js_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr left_tau_des_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr right_tau_des_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr left_tau_star_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr right_tau_star_pub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr left_cmd_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr right_cmd_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point last_hz_display_;
    std::vector<double> left_q_init_;
    std::vector<double> right_q_init_;
    int frame_count_ = 0;
};

// ============================================================
// main
// ============================================================

static std::shared_ptr<OSCBFMITController> g_node;

void signal_handler(int) { rclcpp::shutdown(); }

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    std::signal(SIGINT, signal_handler);

    try {
        g_node = std::make_shared<OSCBFMITController>();
        rclcpp::spin(g_node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("oscbf_mit_controller"),
                     "致命错误: %s", e.what());
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}
