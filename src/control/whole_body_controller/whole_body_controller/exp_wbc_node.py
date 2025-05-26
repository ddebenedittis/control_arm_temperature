import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.exceptions import InvalidParameterTypeException

from geometry_msgs.msg import Point32, PointStamped, PolygonStamped, PoseStamped
from nav_msgs.msg import Path
from pi3hat_moteus_int_msgs.msg import JointsCommand, JointsStates
from sensor_msgs.msg import JointState

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
        
        self.joint_state_pub = self.create_publisher(
            JointState, '/joint_states', 1)
        
        self.ee_pos_pub = self.create_publisher(
            PointStamped, '/ee_position', 1)
        
        self.ee_path_pub = self.create_publisher(
            Path, '/ee_path', 1)
        
        self.ref_pub = self.create_publisher(
            PointStamped, '/ee_reference', 1)
        
        self.ee_reference_path_pub = self.create_publisher(
            Path, '/ee_reference_path', 1)
        
        self.feasible_region_1_pub = self.create_publisher(
            PolygonStamped, '/feasible_region_1', 1)
        self.feasible_region_2_pub = self.create_publisher(
            PolygonStamped, '/feasible_region_2', 1)
        self.feasible_region_3_pub = self.create_publisher(
            PolygonStamped, '/feasible_region_3', 1)
        self.feasible_region_4_pub = self.create_publisher(
            PolygonStamped, '/feasible_region_4', 1)
        
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
        
        self.joint_positions_filter = FadingFilter()
        self.joint_positions_filter.beta = 0.75
        self.joint_velocities_filter = FadingFilter()
        self.joint_velocities_filter.beta = 0.75
        
    def joint_states_callback(self, msg: JointsStates):
        joint_positions = np.zeros(self.joint_positions.shape)
        joint_velocities = np.zeros(self.joint_velocities.shape)
        
        # The joint positions and velocities need to be reordered as specified
        # in robot_model/robots/all_robots.yaml.
        for i, joint_name in enumerate(msg.name):
            idx = self.wbc._control_tasks.robot_wrapper.joint_names.index(joint_name)
            if not np.isnan(msg.position[i]):
                joint_positions[idx] = msg.position[i]
            else:
                joint_positions[idx] = self.joint_positions[idx]
            if not np.isnan(msg.velocity[i]):
                joint_velocities[idx] = msg.velocity[i]
            else:
                joint_velocities[idx] = self.joint_velocities[idx]
            #! The interface is bugged. The curent is actually the temperature
            #! and vice versa.
            if not np.isnan(msg.current[i]):
                self.temp[idx] = msg.current[i]
                
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
    
    def publish_viz_data(self, p_ref):
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
            
            # Publish the feasible region
            if self.task == 'obs8':
                if self.x_min is not None:
                    polygon_msg_1 = PolygonStamped()
                    polygon_msg_1.header.frame_id = 'base_link'
                    polygon_msg_1.header.stamp = self.get_clock().now().to_msg()
                    polygon_msg_1.polygon.points = [
                        Point32(x=self.x_min,           y=-0.16, z= 100.0,),
                        Point32(x=self.x_min-0.0001,    y=-0.16, z= 100.0,),
                        Point32(x=self.x_min-0.0001,    y=-0.16, z=-100.0,),
                        Point32(x=self.x_min,           y=-0.16, z=-100.0,)
                    ]
                    self.feasible_region_1_pub.publish(polygon_msg_1)
                
                if self.y_min is not None:
                    polygon_msg_2 = PolygonStamped()
                    polygon_msg_2.header.frame_id = 'base_link'
                    polygon_msg_2.header.stamp = self.get_clock().now().to_msg()
                    polygon_msg_2.polygon.points = [
                        Point32(x=-100.0, y=-0.16, z=self.y_min,),
                        Point32(x= 100.0, y=-0.16, z=self.y_min,),
                        Point32(x= 100.0, y=-0.16, z=self.y_min-0.0001,),
                        Point32(x=-100.0, y=-0.16, z=self.y_min-0.0001,)
                    ]
                    self.feasible_region_2_pub.publish(polygon_msg_2)
                
                if self.x_max is not None:
                    polygon_msg_3 = PolygonStamped()
                    polygon_msg_3.header.frame_id = 'base_link'
                    polygon_msg_3.header.stamp = self.get_clock().now().to_msg()
                    polygon_msg_3.polygon.points = [
                        Point32(x=self.x_max,     y=-0.16, z= 100.0,),
                        Point32(x=self.x_max+0.0001, y=-0.16, z= 100.0,),
                        Point32(x=self.x_max+0.0001, y=-0.16, z=-100.0,),
                        Point32(x=self.x_max,     y=-0.16, z=-100.0,)
                    ]
                    self.feasible_region_3_pub.publish(polygon_msg_3)
                
                if self.y_max is not None:
                    polygon_msg_4 = PolygonStamped()
                    polygon_msg_4.header.frame_id = 'base_link'
                    polygon_msg_4.header.stamp = self.get_clock().now().to_msg()
                    polygon_msg_4.polygon.points = [
                        Point32(x=-100.0, y=-0.16, z=self.y_max,),
                        Point32(x= 100.0, y=-0.16, z=self.y_max,),
                        Point32(x= 100.0, y=-0.16, z=self.y_max+0.0001,),
                        Point32(x=-100.0, y=-0.16, z=self.y_max+0.0001,)
                    ]
                    self.feasible_region_4_pub.publish(polygon_msg_4)
            
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
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.wbc._control_tasks.robot_wrapper.joint_names
        msg.position = q
        msg.velocity = v
        msg.effort = tau
        msg.kp_scale = [0.1, 0.1, 0.1]
        msg.kd_scale = [1.0, 1.0, 1.0]
        self.joint_command_pub.publish(msg)
        
        # Publish joint states for visualization
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = self.wbc._control_tasks.robot_wrapper.joint_names
        joint_state_msg.position = self.joint_positions
        joint_state_msg.velocity = self.joint_velocities
        
        self.joint_state_pub.publish(joint_state_msg)
        
        self.publish_viz_data(p_ref)
        
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
