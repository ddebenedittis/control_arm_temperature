from hierarchical_qp.hierarchical_qp import HierarchicalQP
from whole_body_controller.control_tasks import ControlTasks


class WholeBodyController:
    def __init__(self, robot_name):
        self.control_tasks = ControlTasks(robot_name)
        
        self.hqp = HierarchicalQP()
        
        
    def update(self, q, v):
        self.control_tasks.update(q, v)
        
    def __call__(self, q, v, pos_ref, vel_ref, acc_ref):
        self.update(q, v)
        
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
        
        C_temp, d_temp = self.control_tasks.task_torque_limits()
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
        
        sol = self.hqp(A, b, C, d)
        
        tau_opt = sol[0:self.control_tasks.n_q]
        
        return tau_opt
