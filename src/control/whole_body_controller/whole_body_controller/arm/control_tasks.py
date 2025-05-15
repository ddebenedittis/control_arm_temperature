import numpy as np
import pinocchio as pin
from scipy.linalg import pinv

from robot_model.robot_wrapper import RobotWrapper


class ControlTasks:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.robot_wrapper = RobotWrapper(robot_name)
        
        # ============================== Sizes ============================== #
        
        self._n_c = 2                        # n control steps
        
        self.n_q = self.robot_wrapper.nq    # n joint positions
        self.n_s = 3*self.n_q               # n_states = q, qdot, T
        self.n_i = self.n_q                 # n inputs = tau
        self.n_x = self.n_s + self.n_i      # n states and inputs
        
        # ========================= Internal States ========================= #
        
        # Joint positions and velocities
        self.q = np.zeros(self.n_q)
        self.v = np.zeros(self.n_q)
        self.temp = np.zeros(self.n_q)
        
        # Temperature coefficients: T_k+1 = (1 - alpha dt) T_k + beta |tau|
        self.alpha = 0.1
        self.beta = 10.0
        self.T_0 = np.ones(self.n_q) * 25
        
        # MPC timestep size
        self.dt = 0.01
        
        # =================================================================== #
        
        # Previous optimal torques
        # TODO: update at the end
        self.tau = np.zeros(self.n_q)
        
        # ============================== Limits ============================= #
        
        self.tau_max = 10
        self.tau_min = -10
        self.delta_tau_max =  0.5 * self.dt
        self.delta_tau_min = -0.5 * self.dt
        
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
        
    def task_eom(self):
        """Generate the A and b matrices of the equations of motion task."""

        A_dyn = np.zeros((self.n_c * self.n_s, self.n_x * self.n_c))
        b_dyn = np.zeros(self.n_c * self.n_s)
        
        A, B, f = self._compute_dyn_matrices()
        
        A_dyn[0:self.n_s, self._id_ui(0)] = B
        A_dyn[0:self.n_s, self._id_si(1)] = - np.eye(self.n_s)
        b_dyn[0:self.n_s] = - A @ np.concatenate([self.q, self.v, self.temp]) - f
        
        for i in range(1, self.n_c):
            A_dyn[i*self.n_s:(i+1)*self.n_s, self._id_si(i)] = A
            A_dyn[i*self.n_s:(i+1)*self.n_s, self._id_ui(i)] = B
            A_dyn[i*self.n_s:(i+1)*self.n_s, self._id_si(i+1)] = - np.eye(self.n_s)
            b_dyn[i*self.n_s:(i+1)*self.n_s] = - f
            
        return A_dyn, b_dyn
        
    def _compute_dyn_matrices(self):
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
    
    def task_torque_limits(self):
        """Implement the torque limits task."""
        
        C = np.zeros((4 * self.n_c * self.n_i, self.n_x * self.n_c))
        d = np.zeros(4 * self.n_c * self.n_i)
        
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
            
        # C2 = np.zeros((2 * (self.n_c - 1) * self.n_i, self.n_x * self.n_c))
        # d2 = np.zeros(2 * (self.n_c - 1) * self.n_i)
        
        # for i in range(self.n_c - 1):
        #     C2[i*self.n_i:(i+1)*self.n_i, self._id_ui(i+1)] = np.eye(self.n_i)
        #     C2[i*self.n_i:(i+1)*self.n_i, self._id_ui(i)] = - np.eye(self.n_i)
        #     d2[i*self.n_i:(i+1)*self.n_i] = self.delta_tau_max * 0.01
        #     C2[(i+1)*self.n_i:(i+2)*self.n_i, self._id_ui(i+1)] = - np.eye(self.n_i)
        #     C2[(i+1)*self.n_i:(i+2)*self.n_i, self._id_ui(i)] = np.eye(self.n_i)
        #     d2[(i+1)*self.n_i:(i+2)*self.n_i] = - self.delta_tau_min * 0.01
            
        # C = np.vstack((C, C2))
        # d = np.hstack((d, d2))
            
        return C, d
    
    def task_velocity_limits(self):
        C = np.zeros((2 * self.n_c * self.n_q, self.n_x * self.n_c))
        d = np.zeros(2 * self.n_c * self.n_q)
        
        for i in range(self.n_c):
            C[i*self.n_q:(i+1)*self.n_q, self._id_vi(i+1)] = np.eye(self.n_q)
            d[i*self.n_q:(i+1)*self.n_q] = 2.0
            C[(i+1)*self.n_q:(i+2)*self.n_q, self._id_vi(i+1)] = - np.eye(self.n_q)
            d[(i+1)*self.n_q:(i+2)*self.n_q] = 2.0
            
        return C, d
            
    def task_temperature_limits(self):
        C = np.zeros((self.n_c * self.n_q, self.n_x * self.n_c))
        d = np.zeros(self.n_c * self.n_q)
        
        for i in range(self.n_c):
            C[i*self.n_q:(i+1)*self.n_q, self._id_Ti(i+1)] = np.eye(self.n_q)
            d[i*self.n_q:(i+1)*self.n_q] = self.T_max
            
        return C, d
    
    def task_obs(
        self,
        x_min: float | None = None, y_min: float | None = None,
        x_max: float | None = None, y_max: float | None = None,
    ):
        id_ee = self.robot_wrapper.model.getFrameId(self.robot_wrapper.ee_name)
        
        M = self.robot_wrapper.mass(self.q)
        h = self.robot_wrapper.nle(self.q, self.v)
        
        J_ee = self.robot_wrapper.getFrameJacobian(id_ee, rf_frame=pin.LOCAL_WORLD_ALIGNED)
        J_ee = J_ee[[0, 2], :]
        
        pos_ee = self.robot_wrapper.framePlacement(self.q, id_ee).translation
        pos_ee = pos_ee[[0, 2]]
        vel_ee = J_ee @ self.v
        
        J_ee_dot_times_v = pin.getFrameClassicalAcceleration(
            self.robot_wrapper.model,
            self.robot_wrapper.data,
            id_ee,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ).linear
        J_ee_dot_times_v = J_ee_dot_times_v[[0, 2]]
        
        C = np.zeros((4*self.n_c, self.n_x * self.n_c))
        d = np.zeros(C.shape[0])
        
        C[0:2, self._id_ui(0)] = - 1/2 * self.dt**2 * J_ee @ pinv(M)
        
        d[0:2] = + 1/2 * self.dt**2 * J_ee_dot_times_v \
            - 1/2 * self.dt**2 * J_ee @ pinv(M) @ h \
            + pos_ee \
            + vel_ee * self.dt \
            - np.array([x_min, y_min])
            
        C[2:4, self._id_ui(0)] = 1/2 * self.dt**2 * J_ee @ pinv(M)
        
        d[2:4] = - 1/2 * self.dt**2 * J_ee_dot_times_v \
            + 1/2 * self.dt**2 * J_ee @ pinv(M) @ h \
            - pos_ee \
            - vel_ee * self.dt \
            + np.array([x_max, y_max])
            
        for i in range(2, self.n_c):
            C[4*i:4*i+2, self._id_ui(i)] = - 1/2 * self.dt**2 * J_ee @ pinv(M)
            # C[4*i:4*i+2, self._id_qi(i)] = self.k_p * J_ee
            # C[4*i:4*i+2, self._id_vi(i)] = self.k_d * J_ee
            
            C[4*i+2:4*(i+1), self._id_ui(i)] = + 1/2 * self.dt**2 * J_ee @ pinv(M)
            # C[4*i+2:4*(i+1), self._id_qi(i)] = self.k_p * J_ee
            # C[4*i+2:4*(i+1), self._id_vi(i)] = self.k_d * J_ee
            
            d[4*i:4*i+2] = + 1/2 * self.dt**2 * J_ee_dot_times_v \
                - 1/2 * self.dt**2 * J_ee @ pinv(M) @ h \
                + pos_ee \
                + vel_ee * self.dt \
                - np.array([x_min, y_min])
            
            d[4*i+2:4*(i+1)] = - 1/2 * self.dt**2 * J_ee_dot_times_v \
                + 1/2 * self.dt**2 * J_ee @ pinv(M) @ h \
                - pos_ee \
                - vel_ee * self.dt \
                + np.array([x_max, y_max])
            
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
        
        A = np.zeros((3*self.n_c, self.n_x * self.n_c))
        b = np.zeros(3*self.n_c)
        
        A[0:3, self._id_ui(0)] = J_ee @ pinv(M)
        
        a_des = a_ref + self.k_d * (v_ref - J_ee @ self.v) + self.k_p * (p_ref - pos_ee)
        v_des = self.q + a_des * self.dt
        
        b[0:3] = - J_ee_dot_times_v \
            + J_ee @ pinv(M) @ h \
            + a_ref \
            + self.k_d * (v_ref - J_ee @ self.v) \
            + self.k_p * (p_ref - pos_ee) \
            + self.k_i * (v_ref - v_des)
            
        for i in range(2, self.n_c):
            A[3*i:3*(i+1), self._id_ui(i)] = J_ee @ pinv(M)
            A[3*i:3*(i+1), self._id_qi(i)] = self.k_p * J_ee
            A[3*i:3*(i+1), self._id_vi(i)] = self.k_d * J_ee
            
            b[3*i:3*(i+1)] = - J_ee_dot_times_v \
                + J_ee @ pinv(M) @ h \
                + a_ref \
                + self.k_d * v_ref \
                + self.k_p * (p_ref - pos_ee + J_ee @ self.q) \
                + self.k_i * (v_ref - v_des)
            
        return A, b
    
    def task_min_torques_qdot(self):
        A = np.zeros((self.n_c * self.n_i + self.n_q * self.n_c, self.n_x * self.n_c))
        b = np.zeros(self.n_c * self.n_i + self.n_q * self.n_c)
        
        off = self.n_i * self.n_c
        for i in range(self.n_c):
            A[i*self.n_i:(i+1)*self.n_i, self._id_ui(i)] = np.eye(self.n_i)
            b[i*self.n_i:(i+1)*self.n_i] = np.zeros(self.n_i)
            A[off+i*self.n_q:off+(i+1)*self.n_q, self._id_vi(i+1)] = np.eye(self.n_q) * 0.01
            b[off+i*self.n_q:off+(i+1)*self.n_q] = np.zeros(self.n_q)
            
        return A, b
        
        
    # ======================================================================= #
    
    # The optimization vector is
    # [u_0, q_1, v_1, T_1, u_1, q_2, v_2, T_2, ..., u_{n_c-1}, q_n_c, v_n_c, T_n_c]
    # Or, more compactly:
    # [u_0, s_1, u_1, s_2, ..., u_{n_c-1}, s_n_c]
    
    def _id_ui(self, i):
        if i < 0 or i > self.n_c-1:
            raise ValueError("i must be in [0, n_c-1]")
        
        return np.arange(
            i*self.n_x,
            i*self.n_x + self.n_i,
        )
    
    def _id_si(self, i):
        if i < 1 or i > self.n_c:
            raise ValueError("i must be in [1, n_c]")
        
        im1 = i - 1
        return np.arange(
            im1*self.n_x + self.n_i,
            im1*self.n_x + self.n_i + 3*self.n_q,
        )
        
    def _id_qi(self, i):
        if i < 1 or i > self.n_c:
            raise ValueError("i must be in [1, n_c]")
        
        im1 = i - 1
        return np.arange(
            im1*self.n_x + self.n_i,
            im1*self.n_x + self.n_i + self.n_q,
        )
    
    def _id_vi(self, i):
        if i < 1 or i > self.n_c:
            raise ValueError("i must be in [1, n_c]")
        
        im1 = i - 1
        return np.arange(
            im1*self.n_x + self.n_i + self.n_q,
            im1*self.n_x + self.n_i + 2*self.n_q,
        )
        
    def _id_Ti(self, i):
        if i < 1 or i > self.n_c:
            raise ValueError("i must be in [1, n_c]")
        
        im1 = i - 1
        return np.arange(
            im1*self.n_x + self.n_i + 2*self.n_q,
            im1*self.n_x + self.n_i + 3*self.n_q,
        )

    # ======================================================================= #
    
    def get_ee_position(self):
        """Get the end-effector position."""
        
        id_ee = self.robot_wrapper.model.getFrameId(self.robot_wrapper.ee_name)
        self.robot_wrapper.forwardKinematics(self.q, self.v, 0 * self.v)
        return self.robot_wrapper.framePlacement(self.q, id_ee).translation
