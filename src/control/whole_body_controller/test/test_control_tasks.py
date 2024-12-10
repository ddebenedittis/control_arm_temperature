import pytest
import traceback

import numpy as np
from whole_body_controller.control_tasks import ControlTasks


def test_control_tasks():
    try:
        control_tasks = ControlTasks('arm')
    except Exception as e:
        e = traceback.format_exc()
        assert False, f"ControlTasks object creation raised the following exception:\n{e}"

    try:
        q = np.zeros(control_tasks.robot_wrapper.nq)
        v = np.zeros(control_tasks.robot_wrapper.nv)
        control_tasks.update(q, v)
    except Exception as e:
        e = traceback.format_exc()
        assert False, f"ControlTasks.update() raised the following exception:\n{e}"
        
    try:
        A_dyn, b_dyn = control_tasks.task_eom()
    except Exception as e:
        e = traceback.format_exc()
        assert False, f"ControlTasks.task_eom() raised the following exception:\n{e}"
    
    try:
        C, d = control_tasks.task_torque_limits()
    except Exception as e:
        e = traceback.format_exc()
        assert False, f"ControlTasks.task_torque_limits() raised the following exception:\n{e}"
    
    try:
        C, d = control_tasks.task_temperature_limits()
    except Exception as e:
        e = traceback.format_exc()
        assert False, f"ControlTasks.task_temperature_limits() raised the following exception:\n{e}"
    
    try:
        id_ee = control_tasks.robot_wrapper.model.getFrameId(control_tasks.robot_wrapper.ee_name)
        
        p_ref = np.array([0.19, -0.16, -0.4])
        v_ref = np.zeros(3)
        a_ref = np.zeros(3)
        A, b = control_tasks.task_motion_ref(p_ref, v_ref, a_ref)
    except Exception as e:
        e = traceback.format_exc()
        assert False, f"ControlTasks.task_motion_ref() raised the following exception:\n{e}"
