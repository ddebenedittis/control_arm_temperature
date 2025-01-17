import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag, pinv

from hierarchical_qp.hierarchical_qp import HierarchicalQP
from robot_model.robot_wrapper import RobotWrapper


class ControlTasksLeg:
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.robot_wrapper = RobotWrapper(robot_name)
        
        # ============================== Sizes ============================== #
        
        self.n_c = 2                        # n control steps
        self.n_qb = 1                       # n base coordinates
        self.n_qj = self.robot_wrapper.nq-1 # n joint positions
        self.n_q = self.n_qb + self.n_qj    # n joint positions + height
        self.n_s = 2*self.n_q+self.n_qj     # n states = q, qdot, T
        self.n_tau = self.n_qj              # n torques
        self.n_f = 3                        # n forces components
        self.n_i = self.n_tau + self.n_f    # n inputs = tau + f (planar)
        self.n_x = self.n_s + self.n_i      # n states and inputs
        
        # ========================= Internal States ========================= #
        
        # Joint positions and velocities
        self.q = np.zeros(self.n_q)
        self.v = np.zeros(self.n_q)
        self.temp = np.zeros(self.n_qj)
        
        # Temperature coefficients: T_k+1 = (1 - alpha dt) T_k + beta |tau|
        self.alpha = 0.1
        self.beta = 0.01
        self.T_0 = np.ones(self.n_qj) * 25
        
        # MPC timestep size
        self.dt = 0.01
        self.Ts = 0.001
        
        self.mu = 0.8
        
        # =================================================================== #
        
        # Previous optimal torques
        # TODO: update at the end
        self.tau = np.zeros(self.n_qj)
        
        # ============================== Limits ============================= #
        
        self.tau_max = 10
        self.tau_min = -10
        self.delta_tau_max =  5.0 * self.Ts
        self.delta_tau_min = -5.0 * self.Ts
        
        self.T_max = 40
        
        # ============================== Gains ============================== #
        
        self.k_p = 10.0
        self.k_d = 10.0
        
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
        
        S = np.hstack((
            np.zeros((self.n_qj, self.n_qb)),
            np.eye(self.n_qj),
        ))
        
        id_ee = self.robot_wrapper.model.getFrameId(self.robot_wrapper.ee_name)
        J_c = self.robot_wrapper.getFrameJacobian(id_ee, rf_frame=pin.LOCAL_WORLD_ALIGNED)
        J_c = J_c[0:3, :]
        
        A = np.eye(self.n_s)
        A[0:self.n_q, self.n_q:2*self.n_q] = np.eye(self.n_q) * self.dt
        A[2*self.n_q:3*self.n_q, 2*self.n_q:3*self.n_q] = (1 - self.alpha * self.dt) * np.eye(self.n_q)
        
        B = np.zeros((self.n_s, self.n_i))
        B[self.n_q:2*self.n_q, 0:self.n_tau] = pinv(M) @ S.T * self.dt
        B[self.n_q:2*self.n_q, self.n_tau:self.n_tau+self.n_f] = pinv(M) @ J_c.T * self.dt
        B[2*self.n_q:3*self.n_q, 0:self.n_tau] = self.beta * self.dt * np.sign(self.tau) * np.eye(self.n_q)
        
        f = np.zeros(self.n_s)
        f[self.n_q:2*self.n_q] = - pinv(M) @ h * self.dt
        f[2*self.n_q:3*self.n_q] = self.alpha * self.dt * self.T_0
        
        return A, B, f
    
    def task_torque_limits(self):
        """Implement the torque limits task."""
        
        C = np.zeros((4 * self.n_c * self.n_tau, self.n_x * self.n_c))
        d = np.zeros(4 * self.n_c * self.n_tau)
        
        # Limit the maximum torques
        for i in range(self.n_c):
            C[2*i*self.n_tau:(2*i+1)*self.n_tau, self._id_ui(i)] = np.eye(self.n_tau)
            C[(2*i+1)*self.n_tau:(2*i+2)*self.n_tau, self._id_ui(i)] = - np.eye(self.n_tau)
            d[2*i*self.n_tau:(2*i+1)*self.n_tau] = self.tau_max
            d[(2*i+1)*self.n_tau:(2*i+2)*self.n_tau] = - self.tau_min
            
        # Limit the maximum torques variation w.r.t. the previous timestep
        off = 2*self.n_c*self.n_tau
        for i in range(self.n_c):
            C[off+2*i*self.n_tau:off+(2*i+1)*self.n_tau, self._id_ui(i)] = np.eye(self.n_tau)
            C[off+(2*i+1)*self.n_tau:off+(2*i+2)*self.n_tau, self._id_ui(i)] = - np.eye(self.n_tau)
            d[off+2*i*self.n_tau:off+(2*i+1)*self.n_tau] = self.delta_tau_max + self.tau
            d[off+(2*i+1)*self.n_tau:off+(2*i+2)*self.n_tau] = - self.delta_tau_min - self.tau
            
        return C, d
    
    def task_force_limits(self):
        """Implement the force limits task. I.e. the Coulomb friction
        constraint and the positive normal force constraint."""
        
        C = np.zeros((self.n_c * 5, self.n_x * self.n_c))
        d = np.zeros(self.n_c * 5)
        
        for i in range(self.n_c):
            C[5*i:5*(i+1), self._id_fi(i)] =  np.array([
                [ 0,  0,       -1],
                [ 1,  0, -self.mu],
                [-1,  0, -self.mu],
                [ 0,  1, -self.mu],
                [ 0, -1, -self.mu],
            ])
            
        return C, d
        
    def task_temperature_limits(self):
        C = np.zeros((self.n_c * self.n_qj, self.n_x * self.n_c))
        d = np.zeros(self.n_c * self.n_qj)
        
        for i in range(self.n_c):
            C[i*self.n_qj:(i+1)*self.n_qj, self._id_Ti(i+1)] = np.eye(self.n_qj)
            d[i*self.n_qj:(i+1)*self.n_qj] = self.T_max
            
        return C, d
    
    def task_rigid_contact(self):
        A = np.zeros((self.n_f * self.n_c, self.n_x * self.n_c))
        b = np.zeros(self.n_f * self.n_c)
        
        id_ee = self.robot_wrapper.model.getFrameId(self.robot_wrapper.ee_name)
        J_c = self.robot_wrapper.getFrameJacobian(id_ee, rf_frame=pin.LOCAL_WORLD_ALIGNED)
        
        A = np.zeros((self.n_f * self.n_c, self.n_x * self.n_c))
        b = np.zeros(self.n_f * self.n_c)
        
        for i in range(self.n_c):
            A[i*self.n_f:(i+1)*self.n_f, self._id_vi(i+1)] = J_c
            
        return A, b
        
    def task_motion_ref(self, h_ref, h_d_ref, h_dd_ref):
        # Only the first two components are controllable in a planar manipulator.
        
        id_base = self.robot_wrapper.model.getFrameId(self.robot_wrapper.base_name)
        
        M = self.robot_wrapper.mass(self.q)
        h = self.robot_wrapper.nle(self.q, self.v)
        
        J_base = self.robot_wrapper.getFrameJacobian(id_base, rf_frame=pin.LOCAL_WORLD_ALIGNED)
        J_base = J_base[2:3, :]     # Only the linear z component.
        
        pos_base = self.robot_wrapper.framePlacement(self.q, id_base).translation
        h_base = pos_base[2]
        
        J_base_dot_times_v = pin.getFrameClassicalAcceleration(
            self.robot_wrapper.model,
            self.robot_wrapper.data,
            id_base,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ).linear
        J_base_dot_times_v = J_base_dot_times_v[2]
        
        A = np.zeros((1*self.n_c, self.n_x * self.n_c))
        b = np.zeros(1*self.n_c)
        
        n = 1
        A[0:n, self._id_ui(0)] = J_base @ pinv(M)
        
        b[0:n] = - J_base_dot_times_v \
            + J_base @ pinv(M) @ h \
            + h_dd_ref \
            + self.k_d * (h_d_ref - J_base @ self.v) \
            + self.k_p * (h_ref - pos_base)
        
        for i in range(1, self.n_c):
            A[n*i:n*(i+1), self._id_ui(i)] = J_base @ pinv(M)
            A[n*i:n*(i+1), self._id_qi(i)] = self.k_p * J_base
            A[n*i:n*(i+1), self._id_vi(i)] = self.k_d * J_base
            
            b[n*i:n*(i+1)] = - J_base_dot_times_v \
                + J_base @ pinv(M) @ h \
                + h_dd_ref \
                + self.k_d * h_d_ref \
                + self.k_p * (h_ref - pos_base - J_base @ self.q)
            
        return A, b
    
    def task_min_torques_qdot(self):
        A = np.zeros((self.n_c * self.n_tau + self.n_c * self.n_q, self.n_x * self.n_c))
        b = np.zeros(self.n_c * self.n_tau + self.n_q * self.n_c)
        
        off = self.n_tau * self.n_c
        for i in range(self.n_c):
            A[i*self.n_tau:(i+1)*self.n_tau, self._id_ui(i)] = np.eye(self.n_tau)
            b[i*self.n_tau:(i+1)*self.n_tau] = np.zeros(self.n_tau)
            A[off+i*self.n_q:off+(i+1)*self.n_q, self._id_vi(i+1)] = np.eye(self.n_i)
            b[off+i*self.n_q:off+(i+1)*self.n_q] = np.zeros(self.n_i)
            
        return A, b
        
        
    # ======================================================================= #
    
    def _id_ui(self, i):
        if i < 0 or i > self.n_c-1:
            raise ValueError("i must be in [0, n_c-1]")
        
        return np.arange(
            i*self.n_x,
            i*self.n_x + self.n_i,
        )
        
    def _id_taui(self, i):
        if i < 0 or i > self.n_c-1:
            raise ValueError("i must be in [0, n_c-1]")
        
        return np.arange(
            i*self.n_x,
            i*self.n_x + self.n_qj,
        )
        
    def _id_fi(self, i):
        if i < 0 or i > self.n_c-1:
            raise ValueError("i must be in [0, n_c-1]")
        
        return np.arange(
            i*self.n_x + self.n_qj,
            i*self.n_x + self.n_qj+ self.n_f,
        )
    
    def _id_si(self, i):
        if i < 1 or i > self.n_c:
            raise ValueError("i must be in [1, n_c]")
        
        im1 = i - 1
        return np.arange(
            im1*self.n_x + self.n_i,
            im1*self.n_x + self.n_i + self.n_s,
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
            im1*self.n_x + self.n_i + 2*self.n_q + self.n_qj,
        )
