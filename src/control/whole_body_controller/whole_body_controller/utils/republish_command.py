import numpy as np

import rclpy
from rclpy.node import Node

from pi3hat_moteus_int_msgs.msg import JointsCommand, JointsStates
from sensor_msgs.msg import JointState


class RepublishCommand(Node):
    def __init__(self):
        super().__init__('republish_command')
        
        self.get_logger().info('Republish command node started')
        
        self.joint_state_cmd = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)
        
        self.cmd_publisher = self.create_publisher(
            JointsCommand,
            '/joint_controller/command',
            1)
        
    def joint_state_callback(self, msg):
        # Create a new JointsCommand message
        cmd_msg = JointsCommand()
        
        # Fill the command message with the data from the JointState message
        cmd_msg.name = msg.name
        cmd_msg.position = msg.position
        cmd_msg.velocity = msg.velocity
        cmd_msg.effort = msg.effort
        cmd_msg.kp_scale = [1., 1., 1.]
        cmd_msg.kd_scale = [1., 1., 1.]
        
        # Publish the command message
        self.cmd_publisher.publish(cmd_msg)


def main(args=None):
    np.set_printoptions(precision=3, linewidth=300)
    rclpy.init(args=args)

    node = RepublishCommand()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
