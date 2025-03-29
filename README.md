# Control Arm Temperature

Implementation of a temperature-aware controller for robotic systems.
It builds upon Hierarchical Quadratic Programming, a method that allows to solve multiple prioritized tasks while respecting the tasks' priorities.
For references on Hierarchical Quadratic Programming, take a look at [Soft Bilinear Inverted Pendulum: A Model to Enable Locomotion With Soft Contacts](https://ieeexplore.ieee.org/document/10777856) or [Prioritized optimization for task-space control](https://ieeexplore.ieee.org/abstract/document/5354341).

## Installation with Docker

Install [Docker Community Edition](https://docs.docker.com/engine/install/ubuntu/) (ex Docker Engine).
You can follow the installation method through `apt`.
Note that it makes you verify the installation by running `sudo docker run hello-world`.
It is better to avoid running this command with `sudo` and instead follow the post installation steps first and then run the command without `sudo`.

Follow with the [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) for Linux.
This will allow you to run Docker without `sudo`.

Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit) (nvidia-docker2).

Build the Docker image with
```bash
./docker/build.bash
```

Run the container with
```bash
./docker/run.bash
```

If the container is run before building the workspace, a warning will be shown.
Do not worry about it and build the workspace with
```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && source install/setup.bash
```

## Usage

Optional arguments are enclosed within square brackets, i.e. `[]`; for multiple option arguments, the available options are enumerated within curly brackets, i.e. `{}`.

### Arm

Run the arm simulation with
```bash
ros2 launch robot_gazebo arm.launch.py
```

<img src="https://raw.githubusercontent.com/ddebenedittis/media/main/control_arm_temperature/arm.webp" width="500">

### Leg

Run the leg simulation with
```bash
ros2 launch robot_gazebo leg.launch.py [controller:={wbc,slider}] [rviz:={False,True}]
```
Where:
- `controller`: choose between `wbc` (default) and `slider` (pd controller with a slider command reference) controller.
- `rviz`: run RViz2 to visualize the robot and the contact forces.

## To Do


## Author

[Davide De Benedittis](https://github.com/ddebenedittis)

## References

- (Docker ROS NVIDIA)[https://github.com/ddebenedittis/docker_ros_nvidia]: Docker image for ROS with NVIDIA support.