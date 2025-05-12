import numpy as np
import pinocchio as pin
from scipy.linalg import pinv

from robot_model.robot_wrapper import RobotWrapper

from whole_body_controller.arm.dynamic_matrices_ss import AState, bState


"""
All linear tasks are writen in the form:
    A x = b
    C x <= d
"""


class ControlTasksSS:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.robot_wrapper = RobotWrapper(robot_name)
        
        # ============================== Sizes ============================== #
        
        self._n_c = 2                        # n control steps
        
        self.n_q = self.robot_wrapper.nq    # n joint positions
        self.n_s = 3*self.n_q               # n_states = q, qdot, T
        self.n_i = self.n_q                 # n inputs = tau
        self.n_x = self.n_i
        self.n_x_opt = self.n_x * self.n_c
        
        # ========================= Internal States ========================= #
        
        # Joint positions and velocities
        self.q = np.zeros(self.n_q)
        self.v = np.zeros(self.n_q)
        self.temp = np.zeros(self.n_q)
        
        # Temperature coefficients: T_k+1 = (1 - alpha dt) T_k + beta |tau|
        self.alpha = 0.1
        self.beta = 0.01
        self.T_0 = np.ones(self.n_q) * 25
        
        # MPC timestep size
        self.dt = 0.01
        self.Ts = 0.01
        
        self.A_state = None
        self.b_state = None
        self.A_state_dot = None
        self.b_state_dot = None
        
        # =================================================================== #
        
        # Previous optimal torques
        # TODO: update at the end
        self.tau = np.zeros(self.n_q)
        
        # ============================== Limits ============================= #
        
        self.tau_max = 10
        self.tau_min = -10
        self.delta_tau_max =  2.5 * self.Ts
        self.delta_tau_min = -2.5 * self.Ts
        
        self.T_max = 40
        
        # ============================== Gains ============================== #
        
        self.k_p = 10.0
        self.k_d = 10.0
        self.k_i = 0.0
        
    @property
    def n_c(self):
        return self._n_c
    
    @n_c.setter
    def n_c(self, n_c):
        if n_c < 1:
            raise ValueError("n_c must be greater than 0")
        self._n_c = n_c
        self.n_x_opt = self.n_i * self._n_c
             
    def update(self, q: np.ndarray, v: np.ndarray, temp: np.ndarray):
        """Update the dynamic and kinematic quantities in Pinocchio."""
        
        self.q = q
        self.v = v
        self.temp = temp
        self.robot_wrapper.forwardKinematics(self.q, self.v, 0 * self.v)
        pin.computeJointJacobiansTimeVariation(
            self.robot_wrapper.model,
            self.robot_wrapper.data,
            self.q,
            self.v,
        )
        
        self.compute_matrices_state()
        self.compute_matrices_state_dot()
        
    # ======================================================================= #
    
    def _compute_dyn_matrices(self):
        """
        Computes and returns the dynamic matrices A, B and f.
        The system is described by the following equation:
            x_{k+1} = A x_k + B u_k + f

        Returns:
            A: The system evolution matrix
            B: The input matrix
            f: The constant term
        """
        
        M = self.robot_wrapper.mass(self.q)
        h = self.robot_wrapper.nle(self.q, self.v)
        
        A = np.eye(self.n_s)
        A[0:self.n_q, self.n_q:2*self.n_q] = np.eye(self.n_q) * self.dt
        A[2*self.n_q:3*self.n_q, 2*self.n_q:3*self.n_q] = (1 - self.alpha * self.dt) * np.eye(self.n_q)
        
        B = np.zeros((self.n_s, self.n_i))
        B[self.n_q:2*self.n_q, 0:self.n_q] = pinv(M) * self.dt
        B[2*self.n_q:3*self.n_q, 0:self.n_q] = self.beta * self.dt * np.sign(self.tau) * np.eye(self.n_q)
        
        f = np.zeros(self.n_s)
        f[self.n_q:2*self.n_q] = - pinv(M) @ h * self.dt
        f[2*self.n_q:3*self.n_q] = self.alpha * self.dt * self.T_0
        
        return A, B, f
        
    def compute_matrices_state(self):
        """
        Computes the state matrices A_state and b_state w.r.t. the system
        inputs.
        Given the inputs
            u_bar = [u_0; u_1; ...; u_n_c],
        the state is
            x_k = [q_1, qdot_1, T_1, ..., q_{n_c+1}, qdot_{n_c+1}, T_{n_c+1}]^T
        and the system is described by the following equation:
            x_{k+1} = A_state u_bar + b_state
        """
        
        self.A_state = AState(np.zeros((self.n_s * self.n_c, self.n_x_opt)), self)
        self.b_state = bState(np.zeros(self.A_state._A.shape[0]), self)
        
        A, B, f = self._compute_dyn_matrices()
        
        for i in range(1, self.n_c+1):
            self.b_state.si[i] += np.linalg.matrix_power(A, i) @ np.concatenate((self.q, self.v, self.temp))
            
            for j in range(i):
                self.A_state.si[i, j] = np.linalg.matrix_power(A, i-j-1) @ B
                self.b_state.si[i]   += np.linalg.matrix_power(A, i-j-1) @ f
                
    def compute_matrices_state_dot(self):
        self.A_state_dot = AState(np.zeros((self.n_s * self.n_c, self.n_x_opt)), self)
        self.b_state_dot = bState(np.zeros(self.A_state._A.shape[0]), self)
        
        M = self.robot_wrapper.mass(self.q)
        h = self.robot_wrapper.nle(self.q, self.v)
        
        for i in range(self.n_c):
            self.A_state_dot.vi[i+1, i] = pinv(M)
            self.b_state_dot.vi[i+1] = pinv(M) @ h
            
            self.A_state_dot.Ti[i+1, i] = self.beta * np.sign(self.tau) * np.eye(self.n_q)
            
            self.A_state_dot.Ti[i+1] += - self.alpha * self.A_state.Ti[i+1]
            self.b_state_dot.Ti[i+1] += - self.alpha * self.b_state.Ti[i+1]
    
    # ======================================================================= #
    
    def task_torque_limits(self):
        """Implement the torque limits task."""
        
        C = np.zeros((4 * self.n_c * self.n_i, self.n_x_opt))
        d = np.zeros(C.shape[0])
        
        # Limit the maximum torques
        for i in range(self.n_c):
            C[2*i*self.n_i:(2*i+1)*self.n_i, self._id_ui(i)] = np.eye(self.n_i)
            C[(2*i+1)*self.n_i:(2*i+2)*self.n_i, self._id_ui(i)] = - np.eye(self.n_i)
            d[2*i*self.n_i:(2*i+1)*self.n_i] = self.tau_max
            d[(2*i+1)*self.n_i:(2*i+2)*self.n_i] = - self.tau_min
            
        # Limit the maximum torques variation w.r.t. the previous timestep
        off = 2*self.n_c*self.n_i
        for i in range(self.n_c):
            C[off+2*i*self.n_i:off+(2*i+1)*self.n_i, self._id_ui(i)] = np.eye(self.n_i)
            C[off+(2*i+1)*self.n_i:off+(2*i+2)*self.n_i, self._id_ui(i)] = - np.eye(self.n_i)
            d[off+2*i*self.n_i:off+(2*i+1)*self.n_i] = self.delta_tau_max + self.tau
            d[off+(2*i+1)*self.n_i:off+(2*i+2)*self.n_i] = - self.delta_tau_min - self.tau
            
        return C, d
    
    def task_velocity_limits(self):
        C = np.zeros((2 * self.n_c * self.n_q, self.n_x_opt))
        d = np.zeros(C.shape[0])
        
        for i in range(self.n_c):
            C[2*i*self.n_q:(2*i+1)*self.n_q, :] = self.A_state.vi[i+1]
            d[2*i*self.n_q:(2*i+1)*self.n_q] = - self.b_state.vi[i+1] + 2.0
            C[(2*i+1)*self.n_q:(2*i+2)*self.n_q, :] = - self.A_state.vi[i+1]
            d[(2*i+1)*self.n_q:(2*i+2)*self.n_q] = self.b_state.vi[i+1] + 2.0
            
        return C, d
            
    def task_temperature_limits(self):
        C = np.zeros((self.n_c * self.n_q, self.n_x_opt))
        d = np.zeros(C.shape[0])
        
        for i in range(self.n_c):
            C[i*self.n_q:(i+1)*self.n_q, :] = self.A_state.Ti[i+1]
            d[i*self.n_q:(i+1)*self.n_q] = - self.b_state.Ti[i+1] + self.T_max
            
        return C, d
            
    def task_motion_ref(self, p_ref, v_ref, a_ref):
        # Only the first two components are controllable in a planar manipulator.
        
        id_ee = self.robot_wrapper.model.getFrameId(self.robot_wrapper.ee_name)
        
        M = self.robot_wrapper.mass(self.q)
        h = self.robot_wrapper.nle(self.q, self.v)
        
        J_ee = self.robot_wrapper.getFrameJacobian(id_ee, rf_frame=pin.LOCAL_WORLD_ALIGNED)
        J_ee = J_ee[0:3, :]
        
        pos_ee = self.robot_wrapper.framePlacement(self.q, id_ee).translation
        
        J_ee_dot_times_v = pin.getFrameClassicalAcceleration(
            self.robot_wrapper.model,
            self.robot_wrapper.data,
            id_ee,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ).linear
        
        a_des = a_ref + self.k_d * (v_ref - J_ee @ self.v) + self.k_p * (p_ref - pos_ee)
        
        A = np.zeros((3*self.n_c, self.n_x_opt))
        b = np.zeros(A.shape[0])
        
        for i in range(self.n_c):
            A[3*i:3*(i+1), :] = J_ee @ self.A_state_dot.vi[i+1]
            b[3*i:3*(i+1)] = - J_ee @ self.b_state_dot.vi[i+1] \
                - J_ee_dot_times_v \
                + a_des
        
        return A, b
    
    def task_min_torques_qdot(self):
        A = np.zeros((self.n_c * self.n_i + self.n_q * self.n_c, self.n_x_opt * self.n_c))
        b = np.zeros(self.n_c * self.n_i + self.n_q * self.n_c)
        
        off = self.n_i * self.n_c
        for i in range(self.n_c):
            A[i*self.n_i:(i+1)*self.n_i, self._id_ui(i)] = np.eye(self.n_i)
            b[i*self.n_i:(i+1)*self.n_i] = np.zeros(self.n_i)
            A[off+i*self.n_q:off+(i+1)*self.n_q, :] = self.A_state.vi[i+1]
            b[off+i*self.n_q:off+(i+1)*self.n_q] = - self.b_state.vi[i+1]
        
        return A, b
        
    # ======================================================================= #
    
    def _id_ui(self, i):
        if i < 0 or i > self.n_c-1:
            raise ValueError("i must be in [0, n_c-1]")
        
        return np.arange(
            i*self.n_i,
            (i+1)*self.n_i,
        )

    # ======================================================================= #
    
    def get_ee_position(self):
        """Get the end-effector position."""
        
        id_ee = self.robot_wrapper.model.getFrameId(self.robot_wrapper.ee_name)
        self.robot_wrapper.forwardKinematics(self.q, self.v, 0 * self.v)
        return self.robot_wrapper.framePlacement(self.q, id_ee).translation
