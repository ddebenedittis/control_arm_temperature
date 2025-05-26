import datetime
import os

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
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
    
    rviz_config_file_path = os.path.join(
        get_package_share_path('robot_gazebo'),
        'config',
        'rviz_arm.rviz',
    )
    
    wbc_config_file_path = os.path.join(
        get_package_share_path('whole_body_controller'),
        'config',
        'arm_wbc.yaml',
    )
    
    # ======================================================================= #
    
    time = LaunchConfiguration('time', default=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    
    log = LaunchConfiguration('log', default='False')
    ss = LaunchConfiguration('single_shooting', default='False')
    epi = LaunchConfiguration('epigraph', default='False')
    cbf = LaunchConfiguration('cbf', default='False')
    hqp = LaunchConfiguration('hqp', default='True')
    task = LaunchConfiguration('task', default='point')
    use_rviz = LaunchConfiguration('use_rviz', default='False')
    
    use_yaml = LaunchConfiguration('use_yaml', default='False')
    nc = LaunchConfiguration('nc', default='1')
    dt = LaunchConfiguration('dt', default='0.25')
    kp = LaunchConfiguration('kp', default='1000.0')
    kd = LaunchConfiguration('kd', default='500.0')
    ki = LaunchConfiguration('ki', default='0.0')

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
        condition=UnlessCondition(use_yaml),
        package="whole_body_controller",
        executable="wbc_node",
        parameters=[
            {'use_sim_time': True},
            {'task': task},
            {'single_shooting': ss},
            {'epigraph': epi},
            {'cbf': cbf},
            {'hqp': hqp},
            {'nc': nc},
            {'dt': dt},
            {'kp': kp},
            {'kd': kd},
            {'ki': ki},
        ],
        output='screen',
        emulate_tty=True,
    )
    
    spawn_controller_yaml = Node(
        condition=IfCondition(use_yaml),
        package="whole_body_controller",
        executable="wbc_node",
        # pass params by config file,
        parameters=[
            {'use_sim_time': True},
            wbc_config_file_path,
        ],
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
        condition=IfCondition(log),
        package="logger",
        executable="logger",
        parameters=[
            {'time': LaunchConfiguration('time')},
            {'use_sim_time': True}
        ],
        output='screen',
        emulate_tty=True,
    )
    
    rviz2 = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        parameters=[{'use_sim_time': True}],
        arguments=['-d', rviz_config_file_path],
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
        DeclareLaunchArgument('time', default_value=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
        DeclareLaunchArgument('log', default_value='False'),
        DeclareLaunchArgument('single_shooting', default_value='False'),
        DeclareLaunchArgument('epigraph', default_value='False'),
        DeclareLaunchArgument('cbf', default_value='False'),
        DeclareLaunchArgument('hqp', default_value='True'),
        DeclareLaunchArgument('task', default_value='point'),
        DeclareLaunchArgument('use_rviz', default_value='False'),
        DeclareLaunchArgument('use_yaml', default_value='False'),
        DeclareLaunchArgument('nc', default_value='1'),
        DeclareLaunchArgument('dt', default_value='0.25'),
        DeclareLaunchArgument('kp', default_value='1000.0'),
        DeclareLaunchArgument('kd', default_value='500.0'),
        DeclareLaunchArgument('ki', default_value='0.0'),
        gz,
        gz_ros_bridge,
        change_camera,
        robot_state_publisher_node,
        spawn_arm,
        spawn_joint_state_broadcaster,
        spawn_effort_controller,
        spawn_controller_yaml,
        spawn_controller,
        spawn_temperature_node,
        logger,
        rviz2,
    ])
