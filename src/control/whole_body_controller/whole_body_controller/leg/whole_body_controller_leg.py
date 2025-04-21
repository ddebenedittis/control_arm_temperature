from hierarchical_qp.hierarchical_qp import HierarchicalQP
from whole_body_controller.leg.control_tasks_leg import ControlTasksLeg


class WholeBodyController:
    def __init__(self, robot_name):
        self._control_tasks = ControlTasksLeg(robot_name)
        self._control_tasks.n_c = 1
        self._control_tasks.dt = 0.25
        
        self.hqp = HierarchicalQP()
        self.hqp.regularization = 1e-6
        
    def update(self, q, v, temp):
        self._control_tasks.update(q, v, temp)
        
    def __call__(
        self,
        q, v, temp,
        h_ref, h_d_ref, h_dd_ref
    ):
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
        
        C_force, d_force = self._control_tasks.task_force_limits()
        A.append(None)
        b.append(None)
        C.append(C_force)
        d.append(d_force)
        
        A_contact, b_contact = self._control_tasks.task_rigid_contact()
        A.append(A_contact)
        b.append(b_contact)
        C.append(None)
        d.append(None)
        
        # C_temp, d_temp = self._control_tasks.task_temperature_limits()
        # A.append(None)
        # b.append(None)
        # C.append(C_temp)
        # d.append(d_temp)
        
        A_ref, b_ref = self._control_tasks.task_motion_ref(
            h_ref, h_d_ref, h_dd_ref
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
        
        sol = self.hqp(A, b, C, d)
        
        n_qj = self._control_tasks.n_qj
        n_f = self._control_tasks.n_f
        
        tau_opt = sol[0:n_qj]
        f_opt = sol[n_qj:n_qj+n_f]
        
        self._control_tasks.tau = tau_opt
        
        return tau_opt, f_opt
