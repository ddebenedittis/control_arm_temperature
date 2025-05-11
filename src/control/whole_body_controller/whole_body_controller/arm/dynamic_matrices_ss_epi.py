import numpy as np

from whole_body_controller.arm.dynamic_matrices_ss import _BlockAccessor


class AState:
    def __init__(self, A: np.ndarray, control_tasks: "ControlTasks"):
        self._A = A
        self.n_c = control_tasks.n_c
        self.n_q = control_tasks.n_q
        self.n_s = control_tasks.n_s
        self.n_i = control_tasks.n_i
        self.n_a = control_tasks.n_a
        self.n_x = control_tasks.n_x

        self.qi = _BlockAccessor(self._A, self._id_qi, self._id_uj, axis=1)
        self.vi = _BlockAccessor(self._A, self._id_vi, self._id_uj, axis=1)
        self.Ti = _BlockAccessor(self._A, self._id_Ti, self._id_uj, axis=1)
        self.si = _BlockAccessor(self._A, self._id_si, self._id_uj, axis=1)
        
    def _id_uj(self, j):
        if j < 0 or j > self.n_c - 1:
            raise ValueError("i must be in [0, n_c-1]")
        
        return np.arange(
            j * self.n_x,
            (j+1) * self.n_x,
        )

    def _id_qi(self, i):
        self._check_index(i)
        im1 = i - 1
        return np.arange(
            im1 * self.n_s,
            im1 * self.n_s + self.n_q,
        )

    def _id_vi(self, i):
        self._check_index(i)
        im1 = i - 1
        return np.arange(
            im1 * self.n_s + self.n_q,
            im1 * self.n_s + 2 * self.n_q,
        )

    def _id_Ti(self, i):
        self._check_index(i)
        im1 = i - 1
        return np.arange(
            im1 * self.n_s + 2 * self.n_q,
            im1 * self.n_s + 3 * self.n_q,
        )

    def _id_si(self, i):
        self._check_index(i)
        im1 = i - 1
        return np.arange(
            im1 * self.n_s,
            i * self.n_s,
        )

    def _check_index(self, i):
        if i < 1 or i > self.n_c:
            raise ValueError("i must be in [1, n_c]")


class bState:
    def __init__(self, b: np.ndarray, control_tasks: "ControlTasks"):
        self._b = b
        self.n_c = control_tasks.n_c
        self.n_q = control_tasks.n_q
        self.n_s = control_tasks.n_s

        self.qi = _BlockAccessor(self._b, self._id_qi)
        self.vi = _BlockAccessor(self._b, self._id_vi)
        self.Ti = _BlockAccessor(self._b, self._id_Ti)
        self.si = _BlockAccessor(self._b, self._id_si)

    def _id_qi(self, i):
        self._check_index(i)
        im1 = i - 1
        return np.arange(
            im1 * self.n_s,
            im1 * self.n_s + self.n_q,
        )

    def _id_vi(self, i):
        self._check_index(i)
        im1 = i - 1
        return np.arange(
            im1 * self.n_s + self.n_q,
            im1 * self.n_s + 2 * self.n_q,
        )

    def _id_Ti(self, i):
        self._check_index(i)
        im1 = i - 1
        return np.arange(
            im1 * self.n_s + 2 * self.n_q,
            im1 * self.n_s + 3 * self.n_q,
        )

    def _id_si(self, i):
        self._check_index(i)
        im1 = i - 1
        return np.arange(
            im1 * self.n_s,
            i * self.n_s,
        )

    def _check_index(self, i):
        if i < 1 or i > self.n_c:
            raise ValueError("i must be in [1, n_c]")
