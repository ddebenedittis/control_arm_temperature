import pytest
import traceback

import numpy as np
from whole_body_controller.whole_body_controller import WholeBodyController


def test_whole_body_controller():
    try:
        whole_body_controller = WholeBodyController('arm')
    except Exception as e:
        e = traceback.format_exc()
        assert False, f"WholeBodyController object creation raised the following exception:\n{e}"
    
    try:
        q = np.zeros(whole_body_controller.control_tasks.robot_wrapper.nq)
        v = np.zeros(whole_body_controller.control_tasks.robot_wrapper.nv)
        
        p_ref = np.array([0.19, -0.16, -0.4])
        v_ref = np.zeros(3)
        a_ref = np.zeros(3)
        
        tau_opt = whole_body_controller(q, v, p_ref, v_ref, a_ref)
        
    except Exception as e:
        e = traceback.format_exc()
        assert False, f"WholeBodyController.__call__() raised the following exception:\n{e}"
