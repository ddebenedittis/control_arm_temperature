from hierarchical_qp.hierarchical_qp import HierarchicalQP
from whole_body_controller.control_tasks import ControlTasks


class WholeBodyController:
    def __init__(self, robot_name):
        self.control_tasks = ControlTasks(robot_name)
        self.control_tasks.n_c = 5
        self.control_tasks.dt = 0.25
        
        self.hqp = HierarchicalQP()
        self.hqp.regularization = 1e-4
        
    def update(self, q, v, temp):
        self.control_tasks.update(q, v, temp)
        
    def __call__(
        self,
        q, v, temp,
        pos_ref, vel_ref, acc_ref):
        self.update(q, v, temp)
        
        A = []
        b = []
        C = []
        d = []
        
        A_dyn, b_dyn = self.control_tasks.task_eom()
        A.append(A_dyn)
        b.append(b_dyn)
        C.append(None)
        d.append(None)
        
        C_torque, d_torque = self.control_tasks.task_torque_limits()
        A.append(None)
        b.append(None)
        C.append(C_torque)
        d.append(d_torque)
        
        C_temp, d_temp = self.control_tasks.task_temperature_limits()
        A.append(None)
        b.append(None)
        C.append(C_temp)
        d.append(d_temp)
        
        A_ref, b_ref = self.control_tasks.task_motion_ref(
            pos_ref, vel_ref, acc_ref
        )
        A.append(A_ref)
        b.append(b_ref)
        C.append(None)
        d.append(None)
        
        A_min, b_min = self.control_tasks.task_min_torques_qdot()
        A.append(A_min)
        b.append(b_min)
        C.append(None)
        d.append(None)
        
        sol = self.hqp(A, b, C, d)
        
        tau_opt = sol[0:self.control_tasks.n_q]
        
        self.control_tasks.tau = tau_opt
        
        return tau_opt

    def get_ee_position(self):
        return self.control_tasks.get_ee_position()
