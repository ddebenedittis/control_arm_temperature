import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from whole_body_controller.arm.whole_body_controller import WholeBodyController


class Position2Torque(Node):
    def __init__(self):
        super().__init__('position2torque')
        
        self.wbc = WholeBodyController('leg')
        
        self.k_p = 5.0
        self.k_d = 0.5
        
        # =========================== Subscribers =========================== #
        
        self.joint_states_subscription = self.create_subscription(
            JointState,
            "joint_states",
            self.joint_states_callback,
            1,
        )
        
        self.position_subscription = self.create_subscription(
            Float64MultiArray,
            '/position_controller/commands',
            self.position_callback,
            1,
        )
        
        # ============================ Publishers =========================== #
        
        self.joint_command_pub = self.create_publisher(
            Float64MultiArray, '/effort_controller/commands', 1)
        
        # ============================== Timer ============================== #
        
        self.timer_period = 0.0025
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        # ======================== Internal Variables ======================= #
        
        self.joint_positions = np.zeros(3)
        self.joint_velocities = np.zeros(3)
        self.commands_positions = np.zeros(3)
        
    def joint_states_callback(self, msg: JointState):
        self.joint_positions = np.array(msg.position)
        self.joint_velocities = np.array(msg.velocity)

        # The joint positions and velocities need to be reordered as specified
        # in robot_model/robots/all_robots.yaml.
        for i, joint_name in enumerate(msg.name):
            idx = self.wbc._control_tasks.robot_wrapper.joint_names.index(joint_name)
            self.joint_positions[idx] = msg.position[i]
            self.joint_velocities[idx] = msg.velocity[i]
            
    def position_callback(self, msg: Float64MultiArray):
        self.commands_positions = np.array(msg.data)
            
    def timer_callback(self):
        tau_opt = self.k_p * (self.commands_positions - self.joint_positions) \
            + self.k_d * (- self.joint_velocities)
                
        msg = Float64MultiArray()
        msg.data = tau_opt.tolist()
        self.joint_command_pub.publish(msg)


def main(args=None):
    np.set_printoptions(precision=3, linewidth=300)
    rclpy.init(args=args)

    node = Position2Torque()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
