from ray import ray
from hitable_list import hitable_list, hit_record, hitable
from sphere import sphere
from camera import camera
from random import random
from utils import unit_vector, squared_length, length
from material import metal, lambertian,dielectric

import cv2
import numpy as np
import math
import numba
# def hitSphere(center, radius, r):
#     oc = r.origin() - center
#     a = np.dot(r.direction(), r.direction())
#     b = 2 * np.dot(r.direction(), oc)
#     c = np.dot(oc, oc) - radius ** 2
#     discriminant = b ** 2 - 4 * a * c
#     if discriminant < 0:
#         return -1.0
#     else:
#         return (-b - math.sqrt(discriminant)) / (2.0 * a)

# def color(r):
#     t = hitSphere(np.array([.0, .0, -1.0]), 0.5, r)
#     if t > 0.0:
#         N = unit_vector(r.point_at_parameter(t) - np.array([.0,.0,-1.0]))
#         return 0.5 * np.array([N[0] + 1, N[1] + 1, N[2] + 1])
#     unit_direction = unit_vector(r.direction())
#     t = 0.5 * (unit_direction[1] + 1.0)
#     return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])

def vec3(a, b, c):
    return np.array([a, b, c], dtype=np.float32)

def color(r, world, depth):
    boolean, rec = world.hit(r, 0.001, float('inf'), hit_record())
    if boolean:
        flag, attenuation, scattered = rec.mat_ptr.scatter(r, rec, np.array([0.0,0.0,0.0],dtype=float))
        if depth < 50 and flag:
            return attenuation * color(scattered, world, depth + 1)
        else:
            return np.zeros([3],dtype=float)
    else:
        unit_direction = unit_vector(r.direction())
        t = 0.5 * (unit_direction[1] + 1.0)
        return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])

# def render(data,cam,world,ny,nx,ns,anti):
#     colors = np.zeros([data.shape[0],data.shape[1],3],dtype=float)
#     for j in range(ny-1,-1,-1):
#         for i in range(nx):
#             # col = np.zeros([3],dtype=np.float32)
#             # if anti:
#             #     for _ in range(ns):
#             #         u, v = (i + random()) / nx, (ny - 1 - j + random()) / ny
#             #         r = cam.get_ray(u, v)
#             #         col += color(r, world, 0)
#             #     col /= ns
#             # else:
#             #     u, v = i / nx, (ny - 1 - j) / ny
#             #     r = cam.get_ray(u, v)
#             #     col = color(r, world, 0)
#             u, v = i / nx, (ny - 1 - j) / ny
#             r = cam.get_ray(u, v)
#             col = color(r, world, 0)
#             colors[j, i,:] = col
#     for j in range(ny-1,-1,-1):
#         for i in range(nx):
#             col = np.zeros([3],dtype=np.float32)
#             for _ in range(ns):
#                 u, v = (i + random()) / nx, (ny - 1 - j + random()) / ny
#                 col += colors[]
#             col /= ns
#             col = np.sqrt(col)
#             data[j, i, :-1] = 255.99*col
#             data[j, i, -1] = 255.0
#     print('\rprocess: '+ str(int(round((ny-j) / ny, 2)*100))+r'%',end='',flush=True)
#     print()
#     return data

def render(data,cam,world,ny,nx,ns,anti):
    # colors = np.zeros([data.shape[0],data.shape[1],3],dtype=float)
    for j in range(ny-1,-1,-1):
        for i in range(nx):
            col = np.zeros([3],dtype=np.float32)
            if anti:
                for _ in range(ns):
                    u, v = (i + random()) / nx, (ny - 1 - j + random()) / ny
                    r = cam.get_ray(u, v)
                    col += color(r, world, 0)
                col /= ns
            else:
                u, v = i / nx, (ny - 1 - j) / ny
                r = cam.get_ray(u, v)
                col = color(r, world, 0)
            col = np.sqrt(col)
            data[j, i, :-1] = 255.99*col
            data[j, i, -1] = 255.0
        print('\rprocess: '+ str(int(round((ny-j) / ny, 2)*100))+r'%',end='',flush=True)
    print()
    return data

def random_scene(n=500):
    list = []
    list.append(sphere(vec3(0, -1000, 0), 1000, lambertian(vec3(0.5, 0.5, 0.5))))
    for a in range(-11,11):
        for b in range(-11,11):
            choose_mat = random()
            center = vec3(a + 0.9 * random(), 0.2, b + 0.9 * random())
            if length(center - vec3(4, 0.2, 0)) > 0.9:
                if choose_mat < 0.8:
                    list.append(sphere(center, 0.2, lambertian(vec3(random()*random(), random()*random(), random()*random()))))
                elif choose_mat < 0.95:
                    list.append(sphere(center, 0.2, metal(vec3(0.5*(1+random()), 0.5*(1+random()), 0.5*(1+random())), 0.5*random())))
                else:
                    list.append(sphere(center, 0.2, dielectric(1.5)))
    list.append(sphere(vec3(0, 1, 0), 1.0, dielectric(1.5)))
    list.append(sphere(vec3(-4, 1, 0), 1.0, lambertian(vec3(0.4, 0.2, 0.1))))
    list.append(sphere(vec3(4, 1, 0), 1.0, metal(vec3(0.7, 0.6, 0.5), 0.0)))
    return list

def main():
    nx = 600
    ny = 400
    n = 4
    ns = 10

    anti_aliasing = False

    # a = sphere(np.array([.0,.0,-1.0]), 0.5, lambertian(np.array([0.1,0.2,0.5],dtype=float)))
    # b = sphere(np.array([.0,-100.5,-1.0]), 100, lambertian(np.array([0.8,0.8,0.0],dtype=float)))
    # c = sphere(np.array([1.0,.0,-1.0]), 0.5, metal(np.array([0.8,0.6,0.2],dtype=float), 0.0))
    # d = sphere(np.array([-1.0,0.0,-1.0]), 0.5, dielectric(1.5))
    # e = sphere(np.array([-1.0,0.0,-1.0]), -0.45, dielectric(1.5))
    scene = random_scene(n)
    world = hitable_list(scene, len(scene))

    lookfrom = vec3(13, 2, 3)
    lookat = vec3(0, 0, 0)
    dist_to_focus = 10.0
    aperture = 0.1
    cam = camera(lookfrom, lookat, vec3(0, 1, 0), 20, nx / ny, aperture, dist_to_focus)

    data = render(np.zeros([ny,nx,n],dtype=np.float32),cam,world,ny,nx,ns,anti_aliasing)
    data = data.astype(np.uint8)
    img = cv2.cvtColor(data,cv2.COLOR_RGBA2BGRA)
    cv2.imwrite('res.png',img)

if __name__ == '__main__':
    main()
    