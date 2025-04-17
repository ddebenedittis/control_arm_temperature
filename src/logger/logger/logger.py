import datetime
import numpy as np
import os

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class Logger(Node):
    def __init__(self):
        super().__init__('logger')
        
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
        
        # ============================== Timer ============================== #
        
        self.timer_period = 0.01
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        # ======================== Internal Variables ======================= #
        
        self.joint_names = ['JOINT_1', 'JOINT_2', 'JOINT_3']
        
        self.joint_positions = None
        self.joint_velocities = None
        self.joint_torques = None
        self.temperatures = None
        
        self.k = 0
        timesteps = 1000
        
        self.times_vec = np.zeros(timesteps)
        self.joint_positions_vec = np.zeros((timesteps, 3))
        self.joint_velocities_vec = np.zeros((timesteps, 3))
        self.joint_torques_vec = np.zeros((timesteps, 3))
        self.temperatures_vec = np.zeros((timesteps, 3))
        
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
        
    def timer_callback(self):
        if self.joint_positions is None:
            return
            
        self.times_vec[self.k] = self.get_clock().now().nanoseconds / 1e9
        self.joint_positions_vec[self.k, :] = self.joint_positions
        self.joint_velocities_vec[self.k, :] = self.joint_velocities
        self.joint_torques_vec[self.k, :] = self.joint_torques
        self.temperatures_vec[self.k, :] = self.temperatures
        self.k += 1
            
        if self.k < self.joint_positions_vec.shape[0]:
            return
        
        path = "log/csv/" + f"{datetime.datetime.now():%Y-%m-%d-%H:%M:%S}"
            
        try:
            os.makedirs(path)
        except OSError:
            print("Creation of the directories %s failed" % path)
        else:
            print("Successfully created the directories %s" % path)
            
        np.savez(
            path + '/log.npz',
            times=self.times_vec,
            joint_positions=self.joint_positions_vec,
            joint_velocities=self.joint_velocities_vec,
            joint_torques=self.joint_torques_vec,
            temperatures=self.temperatures_vec,
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
