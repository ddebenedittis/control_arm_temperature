import os

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    urdf_path = os.path.join(
        get_package_share_path('arm_description'),
        'urdf',
        'arm.xacro',
    )

    ros_gz_sim_path = os.path.join(
        get_package_share_path('ros_gz_sim'), 'launch', 'gz_sim.launch.py'
    )

    gz_bridge_params = os.path.join(
        get_package_share_path('robot_gazebo'), 'params', 'gz_bridge.yaml'
    )

    world_path = os.path.join(
        get_package_share_path('robot_gazebo'), 'worlds', 'default.world',
    )

    arm_description = ParameterValue(
        Command(['xacro ', urdf_path]),
        value_type=str,
    )

    # ======================================================================= #

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ros_gz_sim_path),
        launch_arguments={'gz_args': world_path}.items(),
    )

    gz_ros_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '--ros-args',
            '-p',
            f'config_file:={gz_bridge_params}',
        ],
        output='screen',
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {'robot_description': arm_description},
            {'use_sim_time': True}
        ],
    )

    spawn_arm = Node(
        package="ros_gz_sim",
        executable='create',
        arguments=[
            "-topic", '/robot_description',
            '-name', 'arm',
            '-x', '0',
            '-y', '0',
            '-z', '0',
        ],
        output='screen',
    )

    spawn_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager',
        ],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    spawn_effort_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'effort_controller',
            '--controller-manager', '/controller_manager',
        ],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )
    
    spawn_controller = Node(
        package="whole_body_controller",
        executable="wbc_node",
        parameters=[{'use_sim_time': True}],
        output='screen',
        emulate_tty=True,
    )
    
    spawn_temperature_node = Node(
        package="whole_body_controller",
        executable="temperature_node",
        parameters=[{'use_sim_time': True}],
        output='screen',
        emulate_tty=True,
    )
    
    logger = Node(
        package="logger",
        executable="logger",
        parameters=[{'use_sim_time': True}],
        output='screen',
        emulate_tty=True,
    )
    
    change_camera = ExecuteProcess(
        cmd=[
            'gz', 'service', '-s', '/gui/move_to/pose',
            '--reqtype', 'gz.msgs.GUICamera',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '2000',
            '--req',
            'pose: {position: {x: 0.0, y: -2.0, z: 0.5}, orientation: {x: -0.0, y: 0.0, z: 0.707, w: 0.707}}'
        ],
        output='screen'
    )

    return LaunchDescription([
        gz,
        gz_ros_bridge,
        change_camera,
        robot_state_publisher_node,
        spawn_arm,
        spawn_joint_state_broadcaster,
        spawn_effort_controller,
        spawn_controller,
        spawn_temperature_node,
        logger,
    ])
