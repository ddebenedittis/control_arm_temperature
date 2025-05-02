import numpy as np


class _BlockAccessor:
    def __init__(self, array: np.ndarray, id_fn, id_fn2 = None, axis: int = None):
        self._array = array
        self._id_fn = id_fn
        self._id_fn2 = id_fn2
        self._axis = axis  # Use axis=1 for 2D access (AState), None for 1D (bState)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            i, j = key
            ids = self._id_fn(i)
            ids2 = self._id_fn2(j)
            if self._axis == 1:
                return self._array[np.ix_(ids, ids2)]
            else:
                raise ValueError("2D indexing not supported on 1D array")
        else:
            ids = self._id_fn(key)
            return self._array[ids] if self._axis is None else self._array[ids, :]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            i, j = key
            ids = self._id_fn(i)
            ids2 = self._id_fn2(j)
            if self._axis == 1:
                self._array[np.ix_(ids, ids2)] = value
            else:
                raise ValueError("2D indexing not supported on 1D array")
        else:
            ids = self._id_fn(key)
            if self._axis is None:
                self._array[ids] = value
            else:
                self._array[ids, :] = value


class AState:
    def __init__(self, A: np.ndarray, control_tasks: "ControlTasks"):
        self._A = A
        self.n_c = control_tasks.n_c
        self.n_q = control_tasks.n_q
        self.n_s = control_tasks.n_s
        self.n_i = control_tasks.n_i

        self.qi = _BlockAccessor(self._A, self._id_qi, self._id_uj, axis=1)
        self.vi = _BlockAccessor(self._A, self._id_vi, self._id_uj, axis=1)
        self.Ti = _BlockAccessor(self._A, self._id_Ti, self._id_uj, axis=1)
        self.si = _BlockAccessor(self._A, self._id_si, self._id_uj, axis=1)
        
    def _id_uj(self, j):
        if j < 0 or j > self.n_c - 1:
            raise ValueError("i must be in [0, n_c-1]")
        
        return np.arange(
            j * self.n_i,
            (j+1) * self.n_i,
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
