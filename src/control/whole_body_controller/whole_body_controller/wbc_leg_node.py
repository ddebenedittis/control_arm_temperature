import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from whole_body_controller.whole_body_controller_leg import WholeBodyController


class WBCController(Node):
    def __init__(self):
        super().__init__('wbc_node')
        
        self.wbc = WholeBodyController('leg')
        
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
        
        # ============================== Timer ============================== #
        
        self.timer_period = 0.0025
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        # ======================== Internal Variables ======================= #
        
        self.generalized_coordinates = np.zeros(self.wbc.control_tasks.n_q)
        self.generalized_velocities = np.zeros(self.wbc.control_tasks.n_q)
        self.temp = np.ones(self.wbc.control_tasks.n_qj) * 25
        
    def joint_states_callback(self, msg: JointState):
        joint_positions = np.array(msg.position)
        joint_velocities = np.array(msg.velocity)

        # The joint positions and velocities need to be reordered as specified
        # in robot_model/robots/all_robots.yaml.
        for i, joint_name in enumerate(msg.name):
            idx = self.wbc.control_tasks.robot_wrapper.joint_names.index(joint_name)
            joint_positions[idx] = msg.position[i]
            joint_velocities[idx] = msg.velocity[i]
            
        self.generalized_coordinates[1:] = joint_positions
        self.generalized_velocities[1:] = joint_velocities
            
    def temperature_callback(self, msg: Float64MultiArray):
        self.temp = np.array(msg.data)
            
    def timer_callback(self):
        h_ref = np.array([0.4])
        h_d_ref = np.zeros(1)
        h_dd_ref = np.zeros(1)
        
        tau_opt = self.wbc(
            self.generalized_coordinates, self.generalized_velocities, self.temp,
            h_ref, h_d_ref, h_dd_ref
        )
        
        msg = Float64MultiArray()
        msg.data = tau_opt.tolist()
        self.joint_command_pub.publish(msg)


def main(args=None):
    np.set_printoptions(precision=3, linewidth=300)
    rclpy.init(args=args)

    node = WBCController()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
