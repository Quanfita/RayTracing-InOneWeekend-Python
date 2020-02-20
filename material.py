from ray import ray
from hitable import hit_record, hitable
from random import random
from utils import unit_vector, squared_length, length

import numpy as np
import math
import numba

@numba.jit
def schlick(cosine, ref_idx):
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 ** 2
    return r0 + (1 + r0) * pow((1 - cosine), 5)

def refract(v, n, ni_over_nt, refracted):
    uv = unit_vector(v)
    dt = np.dot(uv, n)
    discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt ** 2)
    if discriminant > 0:
        refracted = ni_over_nt * (uv - n * dt) - n * math.sqrt(discriminant)
        return True, refracted
    else:
        return False, refracted

@numba.jit
def reflect(v, n):
    return v - 2 * np.dot(v, n) * n

def random_in_unit_sphere():
    p = 2.0 * np.array([random(),random(),random()],dtype=np.float32) - np.array([1,1,1],dtype=np.float32)
    while squared_length(p) >= 0.5:
        p = 2.0 * np.array([random(),random(),random()],dtype=np.float32) - np.array([1,1,1],dtype=np.float32)
    return p

class material:
    def __init__(self):
        pass

class lambertian(material):
    def __init__(self, a):
        self.albedo = a
    
    def scatter(self, r_in, rec, attenuation):
        target = rec.p + rec.normal + random_in_unit_sphere()
        scattered = ray(rec.p, target - rec.p)
        attenuation = self.albedo
        return True, attenuation, scattered

class metal(material):
    def __init__(self, a, f):
        self.albedo = a
        self.fuzz = f if f < 1 else 1
    
    def scatter(self, r_in, rec, attenuation):
        reflected = reflect(unit_vector(r_in.direction()), rec.normal)
        scattered = ray(rec.p, reflected + self.fuzz * random_in_unit_sphere())
        attenuation = self.albedo
        return np.any(np.dot(scattered.direction(), rec.normal)) > 0, attenuation, scattered

class dielectric(material):
    def __init__(self, ri):
        self.ref_idx = ri
    
    def scatter(self, r_in, rec, attenuation):
        refracted = np.zeros([3],dtype=np.float32)
        reflected = reflect(r_in.direction(), rec.normal)
        attenuation = np.array([1,1,1],dtype=np.float32)
        if np.dot(r_in.direction(), rec.normal) > 0:
            outward_normal = -rec.normal
            ni_over_nt = self.ref_idx
            cosine = self.ref_idx * np.dot(r_in.direction(), rec.normal) / length(r_in.direction())
            # cosine = np.dot(r_in.direction(), rec.normal) / length(r_in.direction())
            # print(1 - self.ref_idx * self.ref_idx * (1 - cosine ** 2), self.ref_idx, cosine)
            # cosine = math.sqrt(1 - self.ref_idx * self.ref_idx * (1 - cosine ** 2))
        else:
            outward_normal = rec.normal
            ni_over_nt = 1.0 / self.ref_idx
            cosine = (0 - np.dot(r_in.direction(), rec.normal)) / length(r_in.direction())
        boolean, refracted = refract(r_in.direction(), outward_normal, ni_over_nt, refracted)
        if boolean:
            reflect_prob = schlick(cosine, self.ref_idx)
        else:
            reflect_prob = 1.0
        if (random() < reflect_prob):
            scattered = ray(rec.p, reflected)
        else:
            scattered = ray(rec.p, refracted)
        return True, attenuation, scattered