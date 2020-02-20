import numpy as np
class vec3:
    def __init__(self, e0, e1, e2):
        self.e = [e0, e1, e2]
    
    def x(self):
        return self.e[0]
    
    def y(self):
        return self.e[1]

    def z(self):
        return self.e[2]
    
    def r(self):
        return self.e[0]
    
    def g(self):
        return self.e[1]

    def b(self):
        return self.e[2]