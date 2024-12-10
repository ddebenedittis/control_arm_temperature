import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag, pinv

from hierarchical_qp.hierarchical_qp import HierarchicalQP
from robot_model.robot_wrapper import RobotWrapper


class ControlTasks:
    def __init__(self, robot_name):
        self.robot_wrapper = RobotWrapper(robot_name)
        
        self.q = None
        self.v = None
        
        self.n_c = 2
        self.n_q = self.robot_wrapper.nq
        self.n_state = self.n_q + self.n_q + self.n_q
        self.n_input = self.n_q
        self.n_x = self.n_state + self.n_input
        
        self.alpha = 1.0
        self.beta = 0.1
        
        self.dt = 0.01
        
        # TODO
        self.tau = np.zeros(self.n_q)
        
        self.tau_max = 10
        self.tau_min = -10
        
        self.T_max = 10
        
        self.k_p = 10
        self.k_d = 1
        
    def update(self, q, v):
        self.q = q
        self.v = v
        self.robot_wrapper.forwardKinematics(self.q, self.v)
        pin.computeJointJacobiansTimeVariation(
            self.robot_wrapper.model,
            self.robot_wrapper.data,
            self.q,
            self.v,
        )
        
    def task_eom(self):
        M = self.robot_wrapper.mass(self.q)
        h = self.robot_wrapper.nle(self.q, self.v)
        
        A_dyn = np.zeros((self.n_c * self.n_q, self.n_x * self.n_c))
        b_dyn = np.zeros(self.n_c * self.n_q)
        
        A, B, f = self._compute_dyn_matrices()
        
        A_dyn[0:self.n_x, self._id_ui(0)] = B
        A_dyn[0:self.n_x, self._id_si(1)] = - np.eye(self.n_x)
        b_dyn[0:self.n_x] = A @ np.concatenate([self.q, self.v, self.tau]) - f
        
        for i in range(1, self.n_c):
            A_dyn[i*self.n_x:(i+1)*self.n_x, self._id_si(i)] = A
            A_dyn[i*self.n_x:(i+1)*self.n_x, self._id_ui(i)] = B
            A_dyn[i*self.n_x:(i+1)*self.n_x, self._id_si(i+1)] = - np.eye(self.n_x)
            b_dyn[i*self.n_x:(i+1)*self.n_x] = - f
            
        return A_dyn, b_dyn
        
    def _compute_dyn_matrices(self):
        M = self.robot_wrapper.mass(self.q)
        h = self.robot_wrapper.nle(self.q, self.v)
        
        A = np.eye(3*self.n_q)
        A[0:self.n_q, self.n_q:2*self.n_q] = np.eye(self.n_q) * self.dt
        A[2*self.n_q:3*self.n_q, 2*self.n_q:3*self.n_q] = (1 - self.alpha * self.dt) * np.eye(self.n_q)
        
        B = np.zeros((3*self.n_q, self.n_q))
        B[self.n_q:2*self.n_q, 0:self.n_q] = pinv(M) * self.dt
        B[2*self.n_q:3*self.n_q] = self.beta * self.dt * np.sign(self.tau)
        
        f = np.zeros(self.n_q)
        f[self.n_q:2*self.n_q] = - pinv(M) @ h * self.dt
        
        return A, B, f
    
    def task_torque_limits(self):
        C = np.zeros((2 * self.n_c * self.n_input, self.n_x * self.n_c))
        d = np.zeros(2 * self.n_c * self.n_input)
        
        for i in range(self.n_c):
            C[2*i*self.n_input:(2*i+1)*self.n_input, self._id_ui(i)] = np.eye(self.n_input)
            C[(2*i+1)*self.n_input:(2*i+2)*self.n_input, self._id_ui(i)] = - np.eye(self.n_input)
            d[2*i*self.n_input:(2*i+1)*self.n_input] = self.tau_max
            d[(2*i+1)*self.n_input:(2*i+2)*self.n_input] = - self.tau_min
            
        return C, d
            
    def task_temperature_limits(self):
        C = np.zeros((self.n_c * self.n_q, self.n_x * self.n_c))
        d = np.zeros(self.n_c * self.n_q)
        
        for i in range(self.n_c):
            C[i*self.n_q:(i+1)*self.n_q, self._id_Ti(i)] = np.eye(self.n_q)
            d[i*self.n_q:(i+1)*self.n_q] = self.T_max
            
        return C, d
            
    def task_motion_ref(self, p_ref, v_ref, a_ref):
        id_ee = self.robot_wrapper.model.getFrameId(self.robot_wrapper.ee_name)
        
        M = self.robot_wrapper.mass(self.q)
        h = self.robot_wrapper.nle(self.q, self.v)
        
        J_ee = self.robot_wrapper.getFrameJacobian(id_ee, rf_frame=pin.LOCAL_WORLD_ALIGNED)
        J_ee = J_ee[0:3, :]
        
        self.pos_ee = self.robot_wrapper.framePlacement(id_ee).translation
        
        J_ee_dot_times_v = pin.getFrameClassicalAcceleration(
            self.robot_wrapper.model,
            self.robot_wrapper.data,
            id_ee,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ).linear()
        
        A = np.zeros((3, self.n_x))
        A[0:3, self._id_ui(0)] = J_ee @ pinv(M)
        A[0:3, self._id_vi(1)] = self.k_p * J_ee
        
        b = - J_ee_dot_times_v \
            + J_ee @ pinv(M) @ h \
            + a_ref \
            + self.k_d * (v_ref - J_ee @ self.q) \
            + self.k_p * (p_ref - self.pos_ee)
            
        return A, b
        
        
    # ======================================================================= #
    
    def _id_ui(self, i):
        if i < 0 or i > self.n_c-1:
            raise ValueError("i must be in [0, n_c-1]")
        
        return np.arange(
            i*self.n_x,
            i*self.n_x + self.n_input,
        )
    
    def _id_si(self, i):
        if i < 1 or i > self.n_c:
            raise ValueError("i must be in [1, n_c]")
        
        im1 = i - 1
        return np.arange(
            im1*self.n_x + self.n_input,
            im1*self.n_x + self.n_input + 3*self.n_q,
        )
        
    def _id_qi(self, i):
        if i < 1 or i > self.n_c:
            raise ValueError("i must be in [1, n_c]")
        
        im1 = i - 1
        return np.arange(
            im1*self.n_x + self.n_input,
            im1*self.n_x + self.n_input + self.n_q,
        )
    
    def _id_vi(self, i):
        if i < 1 or i > self.n_c:
            raise ValueError("i must be in [1, n_c]")
        
        im1 = i - 1
        return np.arange(
            im1*self.n_x + self.n_input + self.n_q,
            im1*self.n_x + self.n_input + 2*self.n_q,
        )
        
    def _id_Ti(self, i):
        if i < 1 or i > self.n_c:
            raise ValueError("i must be in [1, n_c]")
        
        im1 = i - 1
        return np.arange(
            im1*self.n_x + self.n_input + 2*self.n_q,
            im1*self.n_x + self.n_input + 3*self.n_q,
        )
