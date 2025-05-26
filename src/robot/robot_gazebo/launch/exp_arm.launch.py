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

    arm_description = ParameterValue(
        Command(['xacro ', urdf_path]),
        value_type=str,
    )
    
    rviz_config_file_path = os.path.join(
        get_package_share_path('robot_gazebo'),
        'config',
        'rviz_arm_exp.rviz',
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
    dt = LaunchConfiguration('dt', default='0.01')
    kp = LaunchConfiguration('kp', default='1000.0')
    kd = LaunchConfiguration('kd', default='500.0')
    ki = LaunchConfiguration('ki', default='1.0')

    # ======================================================================= #

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {'robot_description': arm_description},
        ],
    )
    
    spawn_controller = Node(
        condition=UnlessCondition(use_yaml),
        package="whole_body_controller",
        executable="exp_wbc_node",
        parameters=[
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
        executable="exp_wbc_node",
        parameters=[
            wbc_config_file_path,
        ],
        output='screen',
        emulate_tty=True,
    )
    
    logger = Node(
        condition=IfCondition(log),
        package="logger",
        executable="logger",
        parameters=[
            {'time': time},
        ],
        output='screen',
        emulate_tty=True,
    )
    
    rviz2 = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file_path],
    )
    
    bag_recorder = Node(
        package='whole_body_controller',
        executable='bag_recorder',
        name='bag_recorder',
        parameters=[
            {'task': task},
            {'time': time},
        ],
        shell=True,
        emulate_tty=True,
        output = 'screen',
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
        DeclareLaunchArgument('dt', default_value='0.01'),
        DeclareLaunchArgument('kp', default_value='1000.0'),
        DeclareLaunchArgument('kd', default_value='500.0'),
        DeclareLaunchArgument('ki', default_value='1.0'),
        robot_state_publisher_node,
        spawn_controller_yaml,
        spawn_controller,
        logger,
        rviz2,
        bag_recorder,
    ])
