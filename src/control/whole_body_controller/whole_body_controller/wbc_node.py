import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from whole_body_controller.whole_body_controller import WholeBodyController


class WBCController(Node):
    def __init__(self):
        super().__init__('wbc_node')
        
        self.declare_parameter('task', 'point')
        self.task = self.get_parameter('task').get_parameter_value().string_value
        
        self.wbc = WholeBodyController('arm')
        
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
        
        self.ee_path_pub = self.create_publisher(
            Path, '/ee_path', 1)
        
        self.ref_pub = self.create_publisher(
            PointStamped, '/ee_reference', 1)
        
        # ============================== Timer ============================== #
        
        self.timer_period = 0.0025
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        self.time = 0.0
        
        # ======================== Internal Variables ======================= #
        
        self.counter = 0
        
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'base_link'
        self.path_msg.poses = []
        
        self.joint_positions = np.zeros(self.wbc.control_tasks.robot_wrapper.nq)
        self.joint_velocities = np.zeros(self.wbc.control_tasks.robot_wrapper.nv)
        self.temp = np.ones(self.wbc.control_tasks.robot_wrapper.nq) * 25
        
    def joint_states_callback(self, msg: JointState):
        self.joint_positions = np.array(msg.position)
        self.joint_velocities = np.array(msg.velocity)

        # The joint positions and velocities need to be reordered as specified
        # in robot_model/robots/all_robots.yaml.
        for i, joint_name in enumerate(msg.name):
            idx = self.wbc.control_tasks.robot_wrapper.joint_names.index(joint_name)
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
        
        tau_opt = self.wbc(
            self.joint_positions, self.joint_velocities, self.temp,
            p_ref, v_ref, a_ref
        )
        
        msg = Float64MultiArray()
        msg.data = tau_opt.tolist()
        self.joint_command_pub.publish(msg)
        
        # Publish the end-effector path
        decimation = 50
        if self.counter % decimation == 0:
            self.path_msg.header.stamp = self.get_clock().now().to_msg()
            
            pose_stamped = PoseStamped()
            position_ee = self.wbc.get_ee_position()
            
            pose_stamped.pose.position.x = position_ee[0]
            pose_stamped.pose.position.y = position_ee[1]
            pose_stamped.pose.position.z = position_ee[2]
            self.path_msg.poses.append(pose_stamped)
            
            self.path_msg.poses = self.path_msg.poses[-100:]
            self.ee_path_pub.publish(self.path_msg)
            
            self.counter = self.counter % decimation
            
        decimation = 10
        if self.counter % decimation == 0:
            # Create a PointStamped message for the end-effector position
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
