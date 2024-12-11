import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from whole_body_controller.whole_body_controller import WholeBodyController


class WBCController(Node):
    def __init__(self):
        super().__init__('wbc_node')
        
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
        
        # ============================== Timer ============================== #
        
        self.timer_period = 0.0025
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        # ======================== Internal Variables ======================= #
        
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
            
    def timer_callback(self):
        p_ref = np.array([0.30, -0.16, -0.4])
        v_ref = np.zeros(3)
        a_ref = np.zeros(3)
        
        tau_opt = self.wbc(
            self.joint_positions, self.joint_velocities, self.temp,
            p_ref, v_ref, a_ref
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
