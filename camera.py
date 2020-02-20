from ray import ray
from utils import unit_vector
from random import random
import numpy as np
import math

def random_in_unit_disk():
    p = 2.0 * np.array([random(),random(),0],dtype=np.float32) - np.array([1,1,0],dtype=np.float32)
    while np.dot(p, p) >= 1.0:
        p = 2.0 * np.array([random(),random(),0],dtype=np.float32) - np.array([1,1,0],dtype=np.float32)
    return p

class camera:
    def __init__(self, lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist):
        self.lens_radius = aperture / 2
        theta = vfov * math.pi / 180
        half_height = math.tan(theta / 2)
        half_width = aspect * half_height
        self.origin = lookfrom
        self.w = unit_vector(lookfrom - lookat)
        self.u = unit_vector(np.cross(vup, self.w))
        self.v = np.cross(self.w, self.u)
        self.lower_left_corner = self.origin \
                                - half_width * self.u * focus_dist \
                                - half_height * self.v * focus_dist \
                                - self.w * focus_dist
        self.horizontal = 2 * half_width * self.u * focus_dist
        self.vertical = 2 * half_height * self.v * focus_dist

    def get_ray(self, s, t):
        rd = self.lens_radius * random_in_unit_disk()
        offset = self.u * rd[0] + self.v * rd[1]
        return ray(self.origin + offset, 
                    self.lower_left_corner + s * self.horizontal + t * self.vertical - self.origin - offset)
