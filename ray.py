class ray:
    def __init__(self,a,b,ti=0.0):
        self.A = a
        self.B = b
        self._time = ti
    
    def origin(self):
        return self.A
    
    def direction(self):
        return self.B
    
    def point_at_parameter(self, t):
        return self.A + t * self.B
    
    def time(self):
        return self._time