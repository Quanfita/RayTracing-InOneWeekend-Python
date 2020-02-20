from hitable import hitable, material
import numpy as np
import math

class sphere(hitable):
    def __init__(self, cen, r, m):
        self.center = cen
        self.radius = r
        self.mat_ptr = m
    
    def hit(self, r, t_min, t_max, rec):
        oc = r.origin() - self.center
        a = np.dot(r.direction(), r.direction())
        b = np.dot(oc, r.direction())
        c = np.dot(oc, oc) - self.radius ** 2
        discriminant = b ** 2 - a * c
        if discriminant > 0:
            temp = (0 - b - math.sqrt(discriminant)) / a
            if temp < t_max and temp > t_min:
                rec.t = temp
                rec.p = r.point_at_parameter(rec.t)
                rec.normal = (rec.p - self.center) / self.radius
                # print((rec.normal + 1)*0.5*255, (rec.p+1)*0.5*255)
                rec.mat_ptr = self.mat_ptr
                return True
            temp = (0 - b + math.sqrt(discriminant)) / a
            if temp < t_max and temp > t_min:
                rec.t = temp
                rec.p = r.point_at_parameter(rec.t)
                rec.normal = (rec.p - self.center) / self.radius
                rec.mat_ptr = self.mat_ptr
                return True
        return False