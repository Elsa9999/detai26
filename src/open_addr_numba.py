
import mmh3
import numpy as np
from numba import njit

@njit
def _insert_batch(keys, data_arr, states, size, mask, h1_arr, h2_arr, n_ptr):
    for i in range(len(h1_arr)):
        base = h1_arr[i]
        step = h2_arr[i]
        for j in range(size):
            idx = (base + j * step) & mask
            if states[idx] != 1 or keys[idx] == h1_arr[i]:
                if states[idx] != 1:
                    n_ptr[0] += 1
                keys[idx] = h1_arr[i]
                data_arr[idx] = data_arr[i]
                states[idx] = 1
                break

class OpenAddrNumba:
    def __init__(self, exp=20):
        self.size = 1 << exp
        self.mask = self.size - 1
        self.keys = np.full(self.size, -1, dtype=np.int64)
        self.data = np.full(self.size, -1, dtype=np.int64)
        self.states = np.zeros(self.size, dtype=np.int8)  # 0 empty,1 used,2 deleted
        self.n = 0

    def batch_insert(self, py_keys, py_vals):
        n = len(py_keys)
        h1 = np.array([mmh3.hash(k) & self.mask for k in py_keys], dtype=np.int64)
        h2 = np.array([(mmh3.hash(k[::-1]) & self.mask) | 1 for k in py_keys], dtype=np.int64)
        keys_arr = np.array(h1, dtype=np.int64)
        data_arr = np.array(py_vals, dtype=np.int64)
        n_ptr = np.zeros(1, dtype=np.int64)
        _insert_batch(keys_arr, data_arr, self.states, self.size, self.mask, h1, h2, n_ptr)
        self.n = int(n_ptr[0])