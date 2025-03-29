import os

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    urdf_path = os.path.join(
        get_package_share_path('arm_description'),
        'urdf',
        'leg.xacro',
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

    leg_description = ParameterValue(
        Command(['xacro ', urdf_path]),
        value_type=str,
    )
    
    # ======================================================================= #
    
    controller = LaunchConfiguration('controller', default='slider')
    
    use_rviz = LaunchConfiguration('rviz', default='False')

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
            {'robot_description': leg_description},
            {'use_sim_time': True}
        ],
    )

    spawn_leg = Node(
        package="ros_gz_sim",
        executable='create',
        arguments=[
            "-topic", '/robot_description',
            '-name', 'leg',
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
        condition=IfCondition(PythonExpression([
            '"', controller, '"', ' == "wbc"'
        ])),
        package="whole_body_controller",
        executable="wbc_leg_node",
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
    
    change_camera = ExecuteProcess(
        cmd=[
            'gz', 'service', '-s', '/gui/move_to/pose',
            '--reqtype', 'gz.msgs.GUICamera',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '2000',
            '--req',
            'pose: {position: {x: 0.0, y: -1.0, z: 0.5}, orientation: {x: -0.0, y: 0.0, z: 0.707, w: 0.707}}'
        ],
        output='screen'
    )
    
    # ======================================================================= #
    
    slider_pub_config = os.path.join(
        get_package_share_path('robot_gazebo'),
        'config',
        'qj_slider_pub.yaml',
    )
    slider_publisher = Node(
        condition=IfCondition(PythonExpression([
            '"', controller, '"', ' == "slider"'
        ])),
        package="slider_publisher",
        executable="slider_publisher",
        parameters=[
            {'use_sim_time': True},
            {'config': slider_pub_config},
        ],
    )
    
    pos2torque = Node(
        condition=IfCondition(PythonExpression([
            '"', controller, '"', ' == "slider"'
        ])),
        package="whole_body_controller",
        executable="positins2torque",
        parameters=[{'use_sim_time': True}],
        output='screen',
        emulate_tty=True,
    )
    
    spawn_debug_controller = Node(
        condition=IfCondition(PythonExpression([
            '"', controller, '"', ' == "slider"'
        ])),
        package="whole_body_controller",
        executable="wbc_leg_node",
        parameters=[
            {'debug': True},
            {'use_sim_time': True},
        ],
        output='screen',
        emulate_tty=True,
    )
    
    # ======================================================================= #
    
    rviz_config_file = LaunchConfiguration('rviz_config_file', default='rviz_leg.rviz')
    rviz_config_file_path = PathJoinSubstitution([
        FindPackageShare('robot_gazebo'),
        'config',
        rviz_config_file
    ])
    
    rviz2 = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        parameters=[{'use_sim_time': True}],
        arguments=['-d', rviz_config_file_path],
    )
    
    # ======================================================================= #

    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='False'),
        gz,
        gz_ros_bridge,
        change_camera,
        robot_state_publisher_node,
        spawn_leg,
        spawn_joint_state_broadcaster,
        spawn_effort_controller,
        spawn_controller,
        spawn_temperature_node,
        rviz2,
        slider_publisher,
        pos2torque,
        spawn_debug_controller,
    ])
