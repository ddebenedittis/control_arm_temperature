import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.exceptions import InvalidParameterTypeException

from nav_msgs.msg import Path
from pi3hat_moteus_int_msgs.msg import JointsCommand, JointsStates

from whole_body_controller.arm.whole_body_controller import WholeBodyController
from whole_body_controller.utils.fading_filter import FadingFilter


class WBCController(Node):
    def __init__(self):
        super().__init__('wbc_node')
        
        self.declare_parameter('single_shooting', False)
        self.ss = self.get_parameter('single_shooting').get_parameter_value().bool_value
        
        self.declare_parameter('epigraph', False)
        self.epi = self.get_parameter('epigraph').get_parameter_value().bool_value
        
        self.declare_parameter('cbf', False)
        self.cbf = self.get_parameter('cbf').get_parameter_value().bool_value
        
        self.declare_parameter('hqp', True)
        self.hqp = self.get_parameter('hqp').get_parameter_value().bool_value
        
        self.wbc = WholeBodyController(
            'arm', ss=self.ss, epi=self.epi, cbf=self.cbf, hqp=self.hqp,
        )
        
        self.k_p = np.array([1.0, 0.5, 0.1]) * 0.0
        self.k_d = np.array([1.0, 0.5, 0.1]) * 0.2
        self.k_tau = np.array([1.0, 1.0, 1.0]) * 0.1
        
        # ============================ Parameters =========================== #
        
        self.declare_parameter('task', 'point')
        self.task = self.get_parameter('task').get_parameter_value().string_value
        self.wbc.task = self.task
        if self.task == 'obs8':
            self.x_min = -100.0
            self.y_min = -0.425
            self.x_max =  0.375
            self.y_max =  100.0
            
            self.wbc.x_min = self.x_min
            self.wbc.y_min = self.y_min
            self.wbc.x_max = self.x_max
            self.wbc.y_max = self.y_max
        
        self.declare_parameter('nc', 1)
        self.wbc.n_c = self.get_parameter('nc').get_parameter_value().integer_value
        
        try:
            self.declare_parameter('dt', 0.01)
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
            self.declare_parameter('kp', 1000.0)
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
            self.declare_parameter('kd', 500.0)
            self.wbc._control_tasks.k_d = self.get_parameter('kd').get_parameter_value().double_value
        except InvalidParameterTypeException as _:
            try:
                self.declare_parameter('kd', 500)
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
            JointsStates,
            "/state_broadcaster/joints_state",
            self.joint_states_callback,
            1,
        )
        
        # ============================ Publishers =========================== #
        
        self.joint_command_pub = self.create_publisher(
            JointsCommand, '/joint_controller/command', 1)
        
        # ============================== Timer ============================== #
        
        self.timer_period = 0.0025
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        self.time = 0.0
        
        # ======================== Internal Variables ======================= #
        
        self.counter = 0
        
        self.joint_positions = np.zeros(self.wbc._control_tasks.robot_wrapper.nq)
        self.joint_velocities = np.zeros(self.wbc._control_tasks.robot_wrapper.nv)
        self.temp = np.ones(self.wbc._control_tasks.robot_wrapper.nq) * 25
        
        self.joint_positions_filter = FadingFilter()
        self.joint_velocities_filter = FadingFilter()
        self.joint_velocities_filter = FadingFilter()
        self.joint_velocities_filter.beta = 0.9
        
    def joint_states_callback(self, msg: JointsStates):
        joint_positions = np.zeros(self.joint_positions.shape)
        joint_velocities = np.zeros(self.joint_velocities.shape)
        
        # The joint positions and velocities need to be reordered as specified
        # in robot_model/robots/all_robots.yaml.
        for i, joint_name in enumerate(msg.name):
            idx = self.wbc._control_tasks.robot_wrapper.joint_names.index(joint_name)
            if not np.isnan(msg.position[i]):
                joint_positions[idx] = msg.position[i]
            if not np.isnan(msg.velocity[i]):
                joint_velocities[idx] = msg.velocity[i]
            if not np.isnan(msg.temperature[i]):
                self.temp[idx] = msg.temperature[i]
                
        self.joint_positions = self.joint_positions_filter.filter(joint_positions)
        self.joint_velocities = self.joint_velocities_filter.filter(joint_velocities)
        
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
        elif self.task == 'obs8':   # Lemniscate trajectory
            radius = 0.1
            omega = 1

            x = radius * np.sin(omega * self.time)
            z = radius * np.sin(2 * omega * self.time)

            dx = radius * omega * np.cos(omega * self.time)
            dz = 2 * radius * omega * np.cos(2 * omega * self.time)

            ddx = -radius * omega**2 * np.sin(omega * self.time)
            ddz = -4 * radius * omega**2 * np.sin(2 * omega * self.time)

            p_ref = np.array([0.30 + x, -0.16, -0.4 + z])
            v_ref = np.array([dx, 0, dz])
            a_ref = np.array([ddx, 0, ddz])
        else:
            raise ValueError(f"Unknown task: {self.task}")
            
        return p_ref, v_ref, a_ref
            
    def timer_callback(self):
        p_ref, v_ref, a_ref = self.get_ref()
        
        sol = self.wbc(
            self.joint_positions, self.joint_velocities, self.temp,
            p_ref, v_ref, a_ref
        )
        
        q = sol.q
        v = sol.v
        tau = sol.tau * self.k_tau
        
        msg = JointsCommand()
        msg.name = self.wbc._control_tasks.robot_wrapper.joint_names
        msg.position = q
        msg.velocity = v
        msg.effort = tau
        msg.kp_scale = [0.1, 0.1, 0.1]
        msg.kd_scale = [1.0, 1.0, 1.0]
        self.joint_command_pub.publish(msg)
            
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
