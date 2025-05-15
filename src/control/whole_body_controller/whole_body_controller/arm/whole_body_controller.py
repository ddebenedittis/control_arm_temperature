from hierarchical_qp.hierarchical_qp import HierarchicalQP

from whole_body_controller.arm.control_tasks import ControlTasks
from whole_body_controller.arm.control_tasks_ss import ControlTasksSS
from whole_body_controller.arm.control_tasks_ss_epi import ControlTasksSSEpi


class SolutionMS:
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
    
class SolutionSS:
    def __init__(self, robot_name: str, sol = None):
        self.control_tasks = ControlTasksSS(robot_name)
        self.sol = sol
        
    @property
    def tau(self):
        return self.sol[self.control_tasks._id_ui(0)]
    

class WholeBodyController:
    def __init__(
        self, robot_name,
        ss: bool = False,
        epi: bool = False,
        cbf: bool = False,
        hqp: bool = True,
    ):
        self.ss = ss
        self.epi = epi
        self.cbf = cbf
        self.hqp = hqp
        
        # If hqp is False, the tasks are weighted with 1 / (decay_factor**p)
        # where p is the task "priority".
        self.decay_factor = 5
        
        self.task = 'point'
        self.x_min = -100
        self.y_min = -100
        self.x_max =  100
        self.y_max =  100
        
        if ss:
            if epi:
                self._control_tasks = ControlTasksSSEpi(robot_name)
            else:
                self._control_tasks = ControlTasksSS(robot_name)
        else:
            self._control_tasks = ControlTasks(robot_name)
        self._control_tasks.n_c = 1
        self._control_tasks.dt = 0.25
        
        self._hqp = HierarchicalQP(hierarchical=self.hqp)
        self._hqp.regularization = 1e-9
        
        self._solution = SolutionMS(robot_name)
        self._solution.control_tasks.n_c = self._control_tasks.n_c
        
    @property
    def n_c(self):
        return self._control_tasks.n_c
    
    @n_c.setter
    def n_c(self, n_c):
        self._control_tasks.n_c = n_c
        self._solution.control_tasks.n_c = n_c
        
    # ======================================================================= #
    
    def _solve_qp(self, A, b, C, d):
        if self.hqp:
            return self._hqp(A, b, C, d)
        
        we = [1/self.decay_factor**i for i in range(len(A))]
        wi = we
        return self._hqp(A, b, C, d, we, wi)
    
    def update(self, q, v, temp):
        self._control_tasks.update(q, v, temp)
        
    def wbc_ms(
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
        
        # C_temp, d_temp = self._control_tasks.task_temperature_limits()
        # A.append(None)
        # b.append(None)
        # C.append(C_temp)
        # d.append(d_temp)
        
        C_vel_lim, d_vel_lim = self._control_tasks.task_velocity_limits()
        A.append(None)
        b.append(None)
        C.append(C_vel_lim)
        d.append(d_vel_lim)
        
        if self.task == 'obs8':
            C_obs, d_obs = self._control_tasks.task_obs(
                x_min=self.x_min, y_min=self.y_min,
                x_max=self.x_max, y_max=self.y_max
            )
            A.append(None)
            b.append(None)
            C.append(C_obs)
            d.append(d_obs)
        
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
        
        sol = self._solve_qp(A, b, C, d)
        
        self._solution.sol = sol
        
        self._control_tasks.tau = self._solution.tau
        
        return self._solution
    
    def wbc_ss(
        self,
        q, v, temp,
        pos_ref, vel_ref, acc_ref
    ):
        self.update(q, v, temp)
        
        A = []
        b = []
        C = []
        d = []
        
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
        
        # A_min, b_min = self._control_tasks.task_min_torques_qdot()
        # A.append(A_min)
        # b.append(b_min)
        # C.append(None)
        # d.append(None)
        
        sol = self._solve_qp(A, b, C, d)
        
        self._solution.sol = sol
        
        self._control_tasks.tau = self._solution.tau
        
        return self._solution
    
    def wbc_ss_epi(
        self,
        q, v, temp,
        pos_ref, vel_ref, acc_ref
    ):
        self.update(q, v, temp)
        
        A = []
        b = []
        C = []
        d = []
        
        C_sl, d_sl = self._control_tasks.task_torque_slack()
        A.append(None)
        b.append(None)
        C.append(C_sl)
        d.append(d_sl)
        
        C_torque, d_torque = self._control_tasks.task_torque_limits()
        A.append(None)
        b.append(None)
        C.append(C_torque)
        d.append(d_torque)
        
        if self.cbf:
            C_temp, d_temp = self._control_tasks.task_temperature_limits_cbf()
        else:
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
        
        sol = self._solve_qp(A, b, C, d)
        
        self._solution.sol = sol
        
        self._control_tasks.tau = self._solution.tau
        
        return self._solution
    
    def __call__(
        self,
        q, v, temp,
        pos_ref, vel_ref, acc_ref
    ):
        if self.ss:
            if self.epi:
                return self.wbc_ss_epi(q, v, temp, pos_ref, vel_ref, acc_ref)
            return self.wbc_ss(q, v, temp, pos_ref, vel_ref, acc_ref)
        else:
            return self.wbc_ms(q, v, temp, pos_ref, vel_ref, acc_ref)

    def get_ee_position(self):
        return self._control_tasks.get_ee_position()
