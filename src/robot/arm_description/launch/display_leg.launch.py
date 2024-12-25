import os

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    arm_package_path = get_package_share_path("arm_description")

    urdf_path = os.path.join(arm_package_path, "urdf", "leg.xacro") 
    config_path = os.path.join(arm_package_path, "rviz", "config.rviz")
    
    arm_description = ParameterValue(
        Command(["xacro ", urdf_path]),
        value_type=str,
    )

    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': arm_description}]
    )

    robot_state_pub_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', config_path],
    )


    return LaunchDescription([
        robot_state_pub,
        robot_state_pub_gui,
        rviz_node,
    ])
