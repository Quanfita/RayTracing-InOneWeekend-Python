from hitable import hitable, hit_record

class hitable_list(hitable):
    def __init__(self, l, n):
        self.list = l
        self.list_size = n
    
    def hit(self, r, t_min, t_max, rec):
        temp_rec = hit_record()
        hit_anything = False
        closest_so_far = t_max
        for i in range(self.list_size):
            if self.list[i].hit(r, t_min, closest_so_far, temp_rec):
                hit_anything = True
                closest_so_far = temp_rec.t
                rec = temp_rec
                # print((rec.normal+1)*0.5*255)
        return hit_anything, rec