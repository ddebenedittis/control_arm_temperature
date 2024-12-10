import pytest

import numpy as np
from whole_body_controller.control_tasks import ControlTasks

def test_init():
    try:
        control_tasks = ControlTasks('arm')
    except:
        assert False, "Failed to initialize ControlTasks"
        
    try:
        q = np.zeros(control_tasks.robot_wrapper.nq)
        v = np.zeros(control_tasks.robot_wrapper.nv)
        control_tasks.update(q, v)
    except:
        assert False, "Failed to update ControlTasks"
