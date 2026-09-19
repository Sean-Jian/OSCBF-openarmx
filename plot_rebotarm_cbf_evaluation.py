import os

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT_DIR, "rebotarm_cbf_evaluation.csv")
PNG_PATH = os.path.join(ROOT_DIR, "rebotarm_cbf_evaluation.png")
PDF_PATH = os.path.join(ROOT_DIR, "rebotarm_cbf_evaluation.pdf")


def main():
    data = np.genfromtxt(CSV_PATH, delimiter=",", names=True)
    time = data["time_s"]
    error_cm = 100.0 * data["tracking_error_m"]
    margin_mm = 1000.0 * data["min_safety_margin_m"]
    correction = data["cbf_torque_correction_nm"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.bbox": "tight",
    })

    colors = {
        "ee": "#1f5a94",
        "target": "#b43b35",
        "obs1": "#d18b22",
        "obs2": "#178f96",
        "safe": "#23805a",
        "cbf": "#6d4c8e",
        "grid": "#d7d7d7",
    }
    fig = plt.figure(figsize=(10.2, 6.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(grid[0, 0], projection="3d")
    ax0.plot(
        data["target_x"], data["target_y"], data["target_z"],
        color=colors["target"], linestyle="--", linewidth=1.7,
        label="Target",
    )
    ax0.plot(
        data["ee_x"], data["ee_y"], data["ee_z"],
        color=colors["ee"], linewidth=1.8, label="End effector",
    )
    ax0.plot(
        data["obstacle1_x"], data["obstacle1_y"], data["obstacle1_z"],
        color=colors["obs1"], linewidth=1.5, label="Obstacle 1",
    )
    ax0.plot(
        data["obstacle2_x"], data["obstacle2_y"], data["obstacle2_z"],
        color=colors["obs2"], linewidth=1.5, label="Obstacle 2",
    )
    ax0.set_xlabel(r"$x$ (m)", labelpad=4)
    ax0.set_ylabel(r"$y$ (m)", labelpad=4)
    ax0.set_zlabel(r"$z$ (m)", labelpad=4)
    ax0.set_title("(a) Cartesian trajectories", loc="left")
    ax0.view_init(elev=24, azim=-56)
    ax0.legend(frameon=False, ncol=2, loc="upper center")
    ax0.grid(True, linewidth=0.45, alpha=0.55)

    ax1 = fig.add_subplot(grid[0, 1])
    ax1.plot(time, error_cm, color=colors["ee"], linewidth=1.5)
    rms_cm = np.sqrt(np.mean(error_cm ** 2))
    ax1.axhline(rms_cm, color=colors["ee"], linewidth=0.9, linestyle=":")
    ax1.text(
        0.98, 0.92, rf"RMSE = {rms_cm:.2f} cm",
        transform=ax1.transAxes, ha="right", va="top",
    )
    ax1.set_xlabel(r"Time $t$ (s)")
    ax1.set_ylabel(r"Tracking error $\|e_p\|_2$ (cm)")
    ax1.set_title("(b) End-effector tracking error", loc="left")

    ax2 = fig.add_subplot(grid[1, 0])
    ax2.plot(time, margin_mm, color=colors["safe"], linewidth=1.5)
    ax2.axhline(0.0, color=colors["target"], linewidth=1.0, linestyle="--")
    min_index = int(np.argmin(margin_mm))
    ax2.plot(time[min_index], margin_mm[min_index], "o", color=colors["safe"], markersize=4)
    ax2.annotate(
        rf"$h_{{\min}}={margin_mm[min_index]:.2f}$ mm",
        xy=(time[min_index], margin_mm[min_index]),
        xytext=(12, 18), textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": colors["safe"], "linewidth": 0.8},
    )
    ax2.set_xlabel(r"Time $t$ (s)")
    ax2.set_ylabel(r"Safety margin $h_{\min}$ (mm)")
    ax2.set_title("(c) Minimum CBF safety margin", loc="left")

    ax3 = fig.add_subplot(grid[1, 1])
    ax3.fill_between(time, correction, color=colors["cbf"], alpha=0.18, linewidth=0)
    ax3.plot(time, correction, color=colors["cbf"], linewidth=1.1)
    ax3.set_xlabel(r"Time $t$ (s)")
    ax3.set_ylabel(r"CBF correction $\|\Delta\tau\|_2$ (N m)")
    ax3.set_title("(d) CBF control intervention", loc="left")

    for axis in (ax1, ax2, ax3):
        axis.set_xlim(time[0], time[-1])
        axis.grid(True, color=colors["grid"], linewidth=0.55, alpha=0.7)
        axis.spines["top"].set_alpha(0.45)
        axis.spines["right"].set_alpha(0.45)

    fig.suptitle(
        "ReBotArm end-effector tracking with CBF collision avoidance",
        fontsize=13,
    )
    fig.savefig(PNG_PATH, dpi=300)
    fig.savefig(PDF_PATH)
    plt.close(fig)
    print(f"Saved {PNG_PATH}")
    print(f"Saved {PDF_PATH}")


if __name__ == "__main__":
    main()
