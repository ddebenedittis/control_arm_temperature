import numpy as np
import pinocchio as pin
from scipy.linalg import pinv

from whole_body_controller.arm.control_tasks_ss import ControlTasksSS
from whole_body_controller.arm.dynamic_matrices_ss_epi import AState, bState


class ControlTasksSSEpi(ControlTasksSS):
    def __init__(self, robot_name):
        super().__init__(robot_name)
        
        # ============================== Sizes ============================== #
        
        self._n_c = 2                        # n control steps
        
        self.n_q = self.robot_wrapper.nq    # n joint positions
        self.n_s = 3*self.n_q               # n_states = q, qdot, T
        self.n_i = self.n_q                 # n inputs = tau
        self.n_a = self.n_q                 # n auxiliary variables
        self.n_x = self.n_i + self.n_a
        self.n_x_opt = self.n_x * self.n_c
    
    @property
    def n_c(self):
        return self._n_c
    
    @n_c.setter
    def n_c(self, n_c):
        if n_c < 1:
            raise ValueError("n_c must be greater than 0")
        self._n_c = n_c
        self.n_x_opt = self.n_x * self._n_c
    
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
        
        B = np.zeros((self.n_s, self.n_x))
        B[self.n_q:2*self.n_q, 0:self.n_q] = pinv(M) * self.dt
        B[2*self.n_q:3*self.n_q, self.n_q:self.n_q+self.n_a] = self.beta * self.dt * np.eye(self.n_q)
        
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

    # TODO: make this more general: there is a double indexing in some matrices.
    def compute_matrices_state_dot(self):
        self.A_state_dot = AState(np.zeros((self.n_s * self.n_c, self.n_x_opt)), self)
        self.b_state_dot = bState(np.zeros(self.A_state._A.shape[0]), self)
        
        M = self.robot_wrapper.mass(self.q)
        h = self.robot_wrapper.nle(self.q, self.v)
        
        for i in range(self.n_c):
            self.A_state_dot.vi[i+1, i][:, 0:self.n_q] = pinv(M)
            self.b_state_dot.vi[i+1] = pinv(M) @ h
            
            self.A_state_dot.Ti[i+1, i][:, self.n_q:2*self.n_q] = self.beta * np.eye(self.n_q)
            
            self.A_state_dot.Ti[i+1] += - self.alpha * self.A_state.Ti[i+1]
            self.b_state_dot.Ti[i+1] += - self.alpha * self.b_state.Ti[i+1]
    
    # ======================================================================= #
    
    def task_torque_slack(self):
        """Enforce constraints on the epigraph torque."""
        
        C = np.zeros((2* self.n_c * self.n_i, self.n_x_opt))
        d = np.zeros(C.shape[0])
        
        for i in range(self.n_c):
            C[2*i*self.n_i:(2*i+1)*self.n_i, self._id_taui(i)] = np.eye(self.n_i)
            C[2*i*self.n_i:(2*i+1)*self.n_i, self._id_tau_sl(i)] = - np.eye(self.n_i)
            C[(2*i+1)*self.n_i:(2*i+2)*self.n_i, self._id_taui(i)] = - np.eye(self.n_i)
            C[(2*i+1)*self.n_i:(2*i+2)*self.n_i, self._id_tau_sl(i)] = - np.eye(self.n_i)
            
        return C, d

    def task_torque_limits(self):
        """Implement the torque limits task."""
        
        C = np.zeros((2 * self.n_c * self.n_i, self.n_x_opt))
        d = np.zeros(C.shape[0])
        
        # Limit the maximum torques
        for i in range(self.n_c):
            C[2*i*self.n_i:(2*i+1)*self.n_i, self._id_taui(i)] = np.eye(self.n_i)
            C[(2*i+1)*self.n_i:(2*i+2)*self.n_i, self._id_taui(i)] = - np.eye(self.n_i)
            d[2*i*self.n_i:(2*i+1)*self.n_i] = self.tau_max
            d[(2*i+1)*self.n_i:(2*i+2)*self.n_i] = - self.tau_min
            
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
    
    def task_temperature_limits_cbf(self):
        # if self.cbf_gamma <= self.alpha:
        #     raise ValueError("Choose cbf_gamma > alpha")
        
        C = np.zeros((self.n_c * self.n_q, self.n_x_opt))
        d = np.zeros(C.shape[0])
        
        for i in range(self.n_c):
            C[i*self.n_q:(i+1)*self.n_q, :] = self.A_state.Ti[i+1] * (self.cbf_gamma + self.alpha)
            C[i*self.n_q:(i+1)*self.n_q, self._id_tau_sl(i)] += - self.beta * np.eye(self.n_q)
            d[i*self.n_q:(i+1)*self.n_q] = self.cbf_gamma * self.T_max \
                - (self.cbf_gamma + self.alpha) * self.b_state.Ti[i+1]
            
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
        
        C = np.zeros((4*self.n_c, self.n_x_opt))
        d = np.zeros(C.shape[0])
        
        C[0:2, self._id_taui(0)] = - 1/2 * self.dt**2 * J_ee @ pinv(M)
        
        d[0:2] = + 1/2 * self.dt**2 * J_ee_dot_times_v \
            - 1/2 * self.dt**2 * J_ee @ pinv(M) @ h \
            + pos_ee \
            + vel_ee * self.dt \
            - np.array([x_min, y_min])
            
        C[2:4, self._id_taui(0)] = 1/2 * self.dt**2 * J_ee @ pinv(M)
        
        d[2:4] = - 1/2 * self.dt**2 * J_ee_dot_times_v \
            + 1/2 * self.dt**2 * J_ee @ pinv(M) @ h \
            - pos_ee \
            - vel_ee * self.dt \
            + np.array([x_max, y_max])
            
        for i in range(2, self.n_c):
            C[4*i:4*i+2, self._id_taui(i)] = - 1/2 * self.dt**2 * J_ee @ pinv(M)
            # C[4*i:4*i+2, self._id_qi(i)] = self.k_p * J_ee
            # C[4*i:4*i+2, self._id_vi(i)] = self.k_d * J_ee
            
            C[4*i+2:4*(i+1), self._id_taui(i)] = + 1/2 * self.dt**2 * J_ee @ pinv(M)
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
            A[i*self.n_i:(i+1)*self.n_i, self._id_taui(i)] = np.eye(self.n_i)
            b[i*self.n_i:(i+1)*self.n_i] = np.zeros(self.n_i)
            A[off+i*self.n_q:off+(i+1)*self.n_q, :] = self.A_state.vi[i+1]
            b[off+i*self.n_q:off+(i+1)*self.n_q] = - self.b_state.vi[i+1]
        
        return A, b
        
    # ======================================================================= #
    
    def _id_taui(self, i):
        if i < 0 or i > self.n_c-1:
            raise ValueError("i must be in [0, n_c-1]")
        
        return np.arange(
            i*self.n_x,
            i*self.n_x + self.n_i,
        )
        
    def _id_tau_sl(self, i):
        if i < 0 or i > self.n_c-1:
            raise ValueError("i must be in [0, n_c-1]")
        
        return np.arange(
            i*self.n_x + self.n_i,
            i*self.n_x + self.n_i + self.n_a,
        )

    # ======================================================================= #
    
    def get_ee_position(self):
        """Get the end-effector position."""
        
        id_ee = self.robot_wrapper.model.getFrameId(self.robot_wrapper.ee_name)
        self.robot_wrapper.forwardKinematics(self.q, self.v, 0 * self.v)
        return self.robot_wrapper.framePlacement(self.q, id_ee).translation
