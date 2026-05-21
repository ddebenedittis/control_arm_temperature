# CLAUDE.md

## Project Overview

ROS 2 (Humble) workspace implementing a temperature-aware whole-body controller for a quadruped leg/arm setup, built on top of Hierarchical Quadratic Programming (HQP) with optional Control Barrier Functions and epigraph reformulation. The MPC layer supports both single-shooting and multiple-shooting; the temperature-aware layer smoothly fades joint position references as actuator temperature crosses a threshold.

Real-hardware target uses moteus actuators via a Raspberry Pi `pi3hat` (see `pi3hat_moteus_int_msgs`); simulation uses Gazebo.

## Build & Run

```bash
# Build (ROS 2 colcon workspace)
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
source install/setup.bash

# Arm experiment launch (the only launch file actually present in robot_gazebo)
ros2 launch robot_gazebo exp_arm.launch.py [log:=False] [use_rviz:=False] [use_yaml:=False] \
    [task:={point,circle}] [single_shooting:=False] [epigraph:=False] [cbf:=False] [hqp:=True]

# Display-only (RViz + joint_state_publisher_gui, no controller)
ros2 launch arm_description display_arm.launch.py
ros2 launch arm_description display_leg.launch.py
```

Console scripts (from `whole_body_controller/setup.py`): `wbc_node`, `exp_wbc_node`, `wbc_leg_node`, `temperature_node`, `bag_recorder`, `republish_command`, `positins2torque` (sic).

Note: the README references `arm.launch.py` / `leg.launch.py` which do **not** exist in `src/robot/robot_gazebo/launch/`. The current entry point is `exp_arm.launch.py`.

## Docker

```bash
./docker/build.bash     # builds image control_arm_temperature on osrf/ros:humble-desktop
./docker/run.bash       # runs with --gpus all, X11 forwarding, host network, repo bind-mounted
```

`run.bash` already handles xauth (`/tmp/.docker.xauth`) and runs `xhost +`. Container needs NVIDIA Container Toolkit.

## Architecture

### Package Dependency Graph

```
hierarchical_qp                  ← Core HQP solver (git submodule)
    ↑
robot_model                      ← Pinocchio wrapper (RobotWrapper)
    ↑
whole_body_controller            ← WBC + control tasks + ROS nodes
    ↑
robot_gazebo / robot_control     ← Sim bring-up, controller configs
arm_description                  ← URDF/xacro (arm + leg variants)
pi3hat_moteus_int_msgs           ← Hardware msg interface (JointsCommand/JointsStates)
logger / batch_sim               ← CSV logging, plotting, sweep runner
rviz_legged_{msgs,plugins}       ← RViz visualization (git submodule)
```

### Key Packages

- **`hierarchical_qp`** (submodule, branch `ros2`): `HierarchicalQP` in `hierarchical_qp/hierarchical_qp.py` wraps `qpsolvers`. First task must have an equality constraint; pass `None` for missing equality/inequality parts.
- **`whole_body_controller`**: Main control package.
  - `arm/whole_body_controller.py` — `WholeBodyController` (façade), `SolutionSS` / `SolutionMS` (result containers).
  - `arm/control_tasks.py` — `ControlTasks` (multiple-shooting base).
  - `arm/control_tasks_ss.py` — `ControlTasksSS` (single-shooting).
  - `arm/control_tasks_ss_epi.py` — `ControlTasksSSEpi(ControlTasksSS)` (single-shooting + epigraph reformulation for torques).
  - `arm/dynamic_matrices_ss{,_epi}.py` — `AState` / `bState` propagation matrices, block accessors.
  - `leg/` — analogous `ControlTasksLeg` + `WholeBodyControllerLeg`.
  - `wbc_node.py`, `exp_wbc_node.py`, `wbc_leg_node.py` — ROS 2 nodes; `exp_wbc_node` is the hardware experiment entry point and contains the `smooth_activation(temp, edge0, edge1)` temperature gate.
  - `utils/temperature_node.py` — simulates per-joint actuator temperatures on `Float64MultiArray`.
  - `utils/fading_filter.py` — `FadingFilter` used for reference smoothing on activation/deactivation.
- **`robot_model`**: `robot_wrapper.py:RobotWrapper` — Pinocchio model loader, exposes `nq`, `joint_names`, kinematics/dynamics.
- **`arm_description`**: URDF/xacro for both `arm.xacro` and `leg.xacro` (robot name internally `mulinex`). Built with `ament_cmake`, installs `urdf/`, `meshes/`, `rviz/`, `launch/`.
- **`logger` / `batch_sim`**: Python-only utilities. `logger/plotter.py` supports time-window selection and NaN-safe plots. `batch_sim/batch_sim.py` drives parameter sweeps.

### Controller mode matrix

The arm WBC has four orthogonal mode flags (set via launch args or `config/arm_wbc.yaml`):

| Flag | Effect | Constraint |
|---|---|---|
| `single_shooting` | Use SS (`ControlTasksSS`) instead of MS (`ControlTasks`) | — |
| `epigraph` | Use epigraph reformulation for torque minimization | Only implemented for SS |
| `cbf` | Add Control Barrier Function constraints | Only implemented for SS / SS+epi |
| `hqp` | Hierarchical QP (else weighted QP) | — |

`task ∈ {point, circle}` selects the end-effector reference trajectory.

### Hardware Configuration

- Xacro entry point: `src/robot/arm_description/urdf/arm.xacro` (uses Gazebo when `use_gazebo:=true`).
- Robot model name: `mulinex`. Shared `links.xacro` + `material.xacro` between arm and leg.
- Real-hardware messages (`pi3hat_moteus_int_msgs`): `JointsCommand`, `JointsStates` — bridge to moteus + pi3hat actuators.

## Notes / Key Concepts

- **Submodules**: `src/control/hierarchical_qp` and `src/rviz_legged` are git submodules; run `git submodule update --init --recursive` after cloning.
- **Temperature gate**: `smooth_activation` in `exp_wbc_node.py` smoothly cross-fades to a safe position reference between `edge0=34.80°C` and `edge1=34.95°C`. Above `edge1` the controller is fully deactivated; the `FadingFilter` then smooths the position reference to avoid jumps.
- **Solver result containers**: `SolutionSS` vs `SolutionMS` are not interchangeable — pick the matching one for the chosen shooting mode.
- **README is partly stale**: launch file names (`arm.launch.py`, `leg.launch.py`) referenced in README don't exist in tree; use `exp_arm.launch.py` and the `display_*.launch.py` files in `arm_description`.
- **YAML vs CLI parameters**: when `use_yaml:=True`, launch args are ignored and `whole_body_controller/config/arm_wbc.yaml` is used instead.
