from ray import ray
import numpy as np

class material:
    def __init__(self):
        pass

class hit_record:
    def __init__(self):
        self.t = float()
        self.p = np.array([0.0,0.0,0.0])
        self.normal = np.array([0.0,0.0,0.0])
        self.mat_ptr = material()

class hitable:
    def __init__(self):
        pass