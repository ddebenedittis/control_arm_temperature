import datetime
import numpy as np
import os

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class Logger(Node):
    def __init__(self):
        super().__init__('logger')
        
        # ============================ Parameters =========================== #
        
        self.declare_parameter('time', 'yyyy-mm-dd-hh-mm-ss')
        
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
        
        self.ee_position_subscription = self.create_subscription(
            PointStamped,
            '/ee_position',
            self.ee_position_callback,
            1,
        )
        
        self.reference_position_subscription = self.create_subscription(
            PointStamped,
            '/ee_reference',
            self.reference_position_callback,
            1,
        )
        
        # ============================== Timer ============================== #
        
        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        # ======================== Internal Variables ======================= #
        
        self.joint_names = ['JOINT_1', 'JOINT_2', 'JOINT_3']
        
        self.joint_positions = None
        self.joint_velocities = None
        self.joint_torques = None
        self.temperatures = None
        
        self.ee_position = None
        self.reference_position = None
        
        self.k = 0
        timesteps = 600
        
        self.times_vec = np.zeros(timesteps)
        self.joint_positions_vec = np.zeros((timesteps, 3))
        self.joint_velocities_vec = np.zeros((timesteps, 3))
        self.joint_torques_vec = np.zeros((timesteps, 3))
        self.temperatures_vec = np.zeros((timesteps, 3))
        
        self.ee_position_vec = np.zeros((timesteps, 3))
        self.reference_position_vec = np.zeros((timesteps, 3))
        
    def joint_states_callback(self, msg: JointState):
        self.joint_positions = np.array(msg.position)
        self.joint_velocities = np.array(msg.velocity)
        self.joint_torques = np.array(msg.effort)
        
        # Reorder the joint positions and velocities
        for i, joint_name in enumerate(msg.name):
            idx = self.joint_names.index(joint_name)
            self.joint_positions[idx] = msg.position[i]
            self.joint_velocities[idx] = msg.velocity[i]
            self.joint_torques[idx] = msg.effort[i]
    
    def temperature_callback(self, msg: Float64MultiArray):
        self.temperatures = np.array(msg.data)
        
    def ee_position_callback(self, msg: PointStamped):
        self.ee_position = np.array([msg.point.x, msg.point.y, msg.point.z])
        
    def reference_position_callback(self, msg: PointStamped):
        self.reference_position = np.array([msg.point.x, msg.point.y, msg.point.z])
        
    def timer_callback(self):
        if self.joint_positions is None:
            return
            
        self.times_vec[self.k] = self.get_clock().now().nanoseconds / 1e9
        self.joint_positions_vec[self.k, :] = self.joint_positions
        self.joint_velocities_vec[self.k, :] = self.joint_velocities
        self.joint_torques_vec[self.k, :] = self.joint_torques
        self.temperatures_vec[self.k, :] = self.temperatures
        
        self.ee_position_vec[self.k, :] = self.ee_position
        self.reference_position_vec[self.k, :] = self.reference_position
        
        self.k += 1
            
        if self.k < self.joint_positions_vec.shape[0]:
            return
        
        time = str(self.get_parameter('time').get_parameter_value().string_value)
        workspace_directory = f"{get_package_share_directory('logger')}/../../../../"
        
        path = f"{workspace_directory}/log/csv/{time}"
            
        try:
            os.makedirs(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s" % path)
            
        np.savez(
            path + '/log.npz',
            times=self.times_vec,
            joint_positions=self.joint_positions_vec,
            joint_velocities=self.joint_velocities_vec,
            joint_torques=self.joint_torques_vec,
            temperatures=self.temperatures_vec,
            ee_position=self.ee_position_vec,
            reference_position=self.reference_position_vec,
        )
        
        print("DONE")
        print("Saved the CSVs in %s" % os.getcwd() + "/" + path)
        
        self.destroy_node()
    

def main(args=None):
    rclpy.init(args=args)

    node = Logger()

    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
