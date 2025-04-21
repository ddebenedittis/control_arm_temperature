import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray

from whole_body_controller.arm.whole_body_controller import WholeBodyController


class ActuatorsTemperature(Node):
    """Node to simulate the temperatures of the actuators."""
    
    def __init__(self):
        super().__init__('temperature_node')
        
        wbc = WholeBodyController('arm')
        self.nq = wbc._control_tasks.robot_wrapper.nq
        self.joint_names = wbc._control_tasks.robot_wrapper.joint_names
        
        # =========================== Subscribers =========================== #
        
        self.joint_command_subscription = self.create_subscription(
            Float64MultiArray,
            '/effort_controller/commands',
            self.joint_command_callback,
            1
        )
        
        # ============================ Publishers =========================== #
        
        self.temperature_pub = self.create_publisher(
            Float64MultiArray, '/joint_states/temperature', 1)
        
        # ============================== Timer ============================== #
        
        self.timer_period = 0.001
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        # ======================== Internal Variables ======================= #
        
        self.alpha = 0.1
        self.beta = 10
        self.temp_0 = np.ones(self.nq) * 25
        
        self.tau = np.zeros(self.nq)
        self.temp = self.temp_0
        
        
    def joint_command_callback(self, msg: Float64MultiArray):
        self.tau = np.array(msg.data)
        
    def timer_callback(self):
        self.temp = self.temp + (- self.alpha * self.timer_period * (self.temp - self.temp_0)) + self.beta * self.timer_period * np.abs(self.tau)
        
        msg = Float64MultiArray()
        msg.data = self.temp
        self.temperature_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    temperature_node = ActuatorsTemperature()

    rclpy.spin(temperature_node)

    temperature_node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
