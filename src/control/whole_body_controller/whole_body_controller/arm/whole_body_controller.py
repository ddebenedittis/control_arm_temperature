from hierarchical_qp.hierarchical_qp import HierarchicalQP
from whole_body_controller.arm.control_tasks import ControlTasks


class Solution:
    def __init__(self, robot_name: str, sol = None):
        self.control_tasks = ControlTasks(robot_name)
        self.sol = sol
        
    @property
    def tau(self):
        return self.sol[self.control_tasks._id_ui(0)]

    @property
    def q(self):
        return self.sol[self.control_tasks._id_qi(1)]
    
    @property
    def v(self):
        return self.sol[self.control_tasks._id_vi(1)]
    
    @property
    def T(self):
        return self.sol[self.control_tasks._id_Ti(1)]
    

class WholeBodyController:
    def __init__(self, robot_name):
        self._control_tasks = ControlTasks(robot_name)
        self._control_tasks.n_c = 1
        self._control_tasks.dt = 0.25
        
        self._hqp = HierarchicalQP()
        self._hqp.regularization = 1e-6
        
        self._solution = Solution(robot_name)
        self._solution.control_tasks.n_c = self._control_tasks.n_c
        
    @property
    def n_c(self):
        return self._control_tasks.n_c
    
    @n_c.setter
    def n_c(self, n_c):
        self._control_tasks.n_c = n_c
        self._solution.control_tasks.n_c = n_c
        
    # ======================================================================= #
    
    def update(self, q, v, temp):
        self._control_tasks.update(q, v, temp)
        
    def __call__(
        self,
        q, v, temp,
        pos_ref, vel_ref, acc_ref):
        self.update(q, v, temp)
        
        A = []
        b = []
        C = []
        d = []
        
        A_dyn, b_dyn = self._control_tasks.task_eom()
        A.append(A_dyn)
        b.append(b_dyn)
        C.append(None)
        d.append(None)
        
        C_torque, d_torque = self._control_tasks.task_torque_limits()
        A.append(None)
        b.append(None)
        C.append(C_torque)
        d.append(d_torque)
        
        C_temp, d_temp = self._control_tasks.task_temperature_limits()
        A.append(None)
        b.append(None)
        C.append(C_temp)
        d.append(d_temp)
        
        C_vel_lim, d_vel_lim = self._control_tasks.task_velocity_limits()
        A.append(None)
        b.append(None)
        C.append(C_vel_lim)
        d.append(d_vel_lim)
        
        A_ref, b_ref = self._control_tasks.task_motion_ref(
            pos_ref, vel_ref, acc_ref
        )
        A.append(A_ref)
        b.append(b_ref)
        C.append(None)
        d.append(None)
        
        A_min, b_min = self._control_tasks.task_min_torques_qdot()
        A.append(A_min)
        b.append(b_min)
        C.append(None)
        d.append(None)
        
        sol = self._hqp(A, b, C, d)
        
        self._solution.sol = sol
        
        self._control_tasks.tau = self._solution.tau
        
        return self._solution

    def get_ee_position(self):
        return self._control_tasks.get_ee_position()
