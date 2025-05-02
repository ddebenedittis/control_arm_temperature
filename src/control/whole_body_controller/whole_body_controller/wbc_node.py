import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.exceptions import InvalidParameterTypeException

from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from whole_body_controller.arm.whole_body_controller import WholeBodyController


class WBCController(Node):
    def __init__(self):
        super().__init__('wbc_node')
        
        self.declare_parameter('single_shooting', False)
        self.ss = self.get_parameter('single_shooting').get_parameter_value().bool_value
        
        self.wbc = WholeBodyController('arm', ss=self.ss)
        
        self.k_p = 0.001
        self.k_d = 0.0001
        
        # ============================ Parameters =========================== #
        
        self.declare_parameter('task', 'point')
        self.task = self.get_parameter('task').get_parameter_value().string_value
        
        self.declare_parameter('nc', 1)
        self.wbc.n_c = self.get_parameter('nc').get_parameter_value().integer_value
        
        try:
            self.declare_parameter('dt', 0.25)
            self.wbc._control_tasks.dt = self.get_parameter('dt').get_parameter_value().double_value
        except InvalidParameterTypeException as _:
            try:
                self.declare_parameter('dt', 1)
                self.wbc._control_tasks.dt = float(
                    self.get_parameter('dt').get_parameter_value().integer_value)
            except Exception as e:
                self.get_logger().error(f"Invalid parameter type for 'dt': {e}.")
                raise e

        
        try:
            self.declare_parameter('kp', 10.0)
            self.wbc._control_tasks.k_p = self.get_parameter('kp').get_parameter_value().double_value
        except InvalidParameterTypeException as _:
            try:
                self.declare_parameter('kp', 10)
                self.wbc._control_tasks.k_p = float(
                    self.get_parameter('kp').get_parameter_value().integer_value)
            except Exception as e:
                self.get_logger().error(f"Invalid parameter type for 'kp': {e}.")
                raise e

        
        try:
            self.declare_parameter('kd', 10.0)
            self.wbc._control_tasks.k_d = self.get_parameter('kd').get_parameter_value().double_value
        except InvalidParameterTypeException as _:
            try:
                self.declare_parameter('kd', 10)
                self.wbc._control_tasks.k_d = float(
                    self.get_parameter('kd').get_parameter_value().integer_value)
            except Exception as e:
                self.get_logger().error(f"Invalid parameter type for 'kd': {e}.")
                raise e

        
        try:
            self.declare_parameter('ki', 1.0)
            self.wbc._control_tasks.k_i = self.get_parameter('ki').get_parameter_value().double_value
        except InvalidParameterTypeException as _:
            try:
                self.declare_parameter('ki', 1)
                self.wbc._control_tasks.k_i = float(
                    self.get_parameter('ki').get_parameter_value().integer_value)
            except Exception as e:
                self.get_logger().error(f"Invalid parameter type for 'ki': {e}.")
                raise e

        
        # =========================== Subscribers =========================== #
        
        self.joint_states_subscription = self.create_subscription(
            JointState,
            "joint_states",
            self.joint_states_callback,
            1,
        )
        
        self.temperature_subscription = self.create_subscription(
            Float64MultiArray,
            '/joint_states/temperature',
            self.temperature_callback,
            1,
        )
        
        # ============================ Publishers =========================== #
        
        self.joint_command_pub = self.create_publisher(
            Float64MultiArray, '/effort_controller/commands', 1)
        
        self.ee_pos_pub = self.create_publisher(
            PointStamped, '/ee_position', 1)
        
        self.ee_path_pub = self.create_publisher(
            Path, '/ee_path', 1)
        
        self.ref_pub = self.create_publisher(
            PointStamped, '/ee_reference', 1)
        
        self.ee_reference_path_pub = self.create_publisher(
            Path, '/ee_reference_path', 1)
        
        # ============================== Timer ============================== #
        
        self.timer_period = 0.0025
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        self.time = 0.0
        
        # ======================== Internal Variables ======================= #
        
        self.counter = 0
        
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'base_link'
        self.path_msg.poses = []
        
        self.reference_path_msg = Path()
        self.reference_path_msg.header.frame_id = 'base_link'
        self.reference_path_msg.poses = []
        
        self.joint_positions = np.zeros(self.wbc._control_tasks.robot_wrapper.nq)
        self.joint_velocities = np.zeros(self.wbc._control_tasks.robot_wrapper.nv)
        self.temp = np.ones(self.wbc._control_tasks.robot_wrapper.nq) * 25
        
    def joint_states_callback(self, msg: JointState):
        self.joint_positions = np.array(msg.position)
        self.joint_velocities = np.array(msg.velocity)

        # The joint positions and velocities need to be reordered as specified
        # in robot_model/robots/all_robots.yaml.
        for i, joint_name in enumerate(msg.name):
            idx = self.wbc._control_tasks.robot_wrapper.joint_names.index(joint_name)
            self.joint_positions[idx] = msg.position[i]
            self.joint_velocities[idx] = msg.velocity[i]
            
    def temperature_callback(self, msg: Float64MultiArray):
        self.temp = np.array(msg.data)
        
    def get_ref(self):
        if self.task == 'point':
            p_ref = np.array([0.30, -0.16, -0.4])
            v_ref = np.zeros(3)
            a_ref = np.zeros(3)
        elif self.task == 'circle':
            radius = 0.1
            omega = 1
            
            p_ref = np.array([
                0.30 + radius * np.cos(omega * self.time),
                - 0.16,
                -0.4 + radius * np.sin(omega * self.time)
            ])
            v_ref = np.array([
                - radius * omega * np.sin(omega * self.time),
                0,
                radius * omega * np.cos(omega * self.time)
            ])
            a_ref = np.array([
                - radius * omega**2 * np.cos(omega * self.time),
                0,
                - radius * omega**2 * np.sin(omega * self.time)
            ])
        else:
            raise ValueError(f"Unknown task: {self.task}")
            
        return p_ref, v_ref, a_ref
            
    def timer_callback(self):
        p_ref, v_ref, a_ref = self.get_ref()
        
        sol = self.wbc(
            self.joint_positions, self.joint_velocities, self.temp,
            p_ref, v_ref, a_ref
        )
        
        tau_ff = sol.tau
        if not self.ss:
            tau = tau_ff \
                + self.k_p * (sol.q - self.joint_positions) \
                + self.k_d * (sol.v - self.joint_velocities)
        else:
            tau = tau_ff
        
        msg = Float64MultiArray()
        msg.data = tau.tolist()
        self.joint_command_pub.publish(msg)
        
        decimation = 50
        if self.counter % decimation == 0:
            # Publish the end-effector path
            self.path_msg.header.stamp = self.get_clock().now().to_msg()
            
            reference_pose_stamped = PoseStamped()
            position_ee = self.wbc.get_ee_position()
            
            reference_pose_stamped.pose.position.x = position_ee[0]
            reference_pose_stamped.pose.position.y = position_ee[1]
            reference_pose_stamped.pose.position.z = position_ee[2]
            self.path_msg.poses.append(reference_pose_stamped)
            
            self.path_msg.poses = self.path_msg.poses[-500:]
            self.ee_path_pub.publish(self.path_msg)
            
            # Publish the end-effector reference path
            self.reference_path_msg.header.stamp = self.get_clock().now().to_msg()
            
            reference_pose_stamped = PoseStamped()
            
            reference_pose_stamped.pose.position.x = p_ref[0]
            reference_pose_stamped.pose.position.y = p_ref[1]
            reference_pose_stamped.pose.position.z = p_ref[2]
            self.reference_path_msg.poses.append(reference_pose_stamped)
            
            self.reference_path_msg.poses = self.reference_path_msg.poses[-500:]
            self.ee_reference_path_pub.publish(self.reference_path_msg)
            
            self.counter = self.counter % decimation
            
        decimation = 10
        if self.counter % decimation == 0:
            # Publish the end-effector position
            ee_position = PointStamped()
            ee_position.header.frame_id = 'base_link'
            ee_position.header.stamp = self.get_clock().now().to_msg()
            ee_position.point.x = self.wbc.get_ee_position()[0]
            ee_position.point.y = self.wbc.get_ee_position()[1]
            ee_position.point.z = self.wbc.get_ee_position()[2]
            
            self.ee_pos_pub.publish(ee_position)
            
            # Create a PointStamped message for the reference end-effector position
            point_msg = PointStamped()
            point_msg.header.frame_id = 'base_link'
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.point.x = p_ref[0]
            point_msg.point.y = p_ref[1]
            point_msg.point.z = p_ref[2]
            
            self.ref_pub.publish(point_msg)
            
        self.counter += 1
        self.time += self.timer_period


def main(args=None):
    np.set_printoptions(precision=3, linewidth=300)
    rclpy.init(args=args)

    node = WBCController()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
