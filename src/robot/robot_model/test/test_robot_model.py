import pytest

from robot_model.robot_wrapper import RobotWrapper


def test_robot_wrapper():
    try:
        robot_wrapper = RobotWrapper('arm')
    except:
        assert False, "Failed to initialize RobotWrapper"
