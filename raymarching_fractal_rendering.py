import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import lil_math as m
import math


def dot(a, b):
    return a.dot(b)


def pixel_to_dir(pixel):
    return m.Vector3((pixel[0] / img_size[0]) * 2 - 1.0, (pixel[1] / img_size[1]) * 2 - 1.0, 1).normalized()


def rot_x(p, s, c):
    return m.Vector3(p.x, c*p.y + s*p.z, c*p.z - s*p.y)


def rot_z(p, s, c):
    return m.Vector3(c * p.x + s * p.y, c * p.y - s * p.x, p.z)


def menger_fold(p):
    a = min(p.x - p.y, 0.0)
    p.x -= a
    p.y += a
    a = min(p.x - p.z, 0.0)
    p.x -= a
    p.z += a
    a = min(p.y - p.z, 0.0)
    p.y -= a
    p.z += a
    return p


def de_box(p, s, w):
    a = m.Vector3(abs(p.x), abs(p.y), abs(p.z)) - s
    return (min(max(max(a.x, a.y), a.z), 0.0) + m.Vector3(max(a.x, 0.0), max(a.y, 0.0), max(a.z, 0.0)).magnitude()) / w


FRACTAL_ITER = 16
FRACTAL_SCALE = 1.8
FRACTAL_SHIFT = m.Vector3(-2.12, -2.75, 0.49)
FRACTAL_ROTATIONS = m.Vector3(-0.12, 0.5, 3.141592/4.0)

sines = m.Vector3(math.sin(FRACTAL_ROTATIONS.x), math.sin(FRACTAL_ROTATIONS.y), 0)
cosines = m.Vector3(math.cos(FRACTAL_ROTATIONS.x), math.cos(FRACTAL_ROTATIONS.y), 0)


def de_fractal(p, min_dist):
    w = 1
    i = stop = 0
    while i < FRACTAL_ITER and stop == 0:
        i += 1
        p = m.Vector3(abs(p.x), abs(p.y), abs(p.z))
        p = rot_z(p, sines.x, cosines.x)
        p = menger_fold(p)
        p = rot_x(p, sines.y, cosines.y)
        p *= FRACTAL_SCALE
        w *= FRACTAL_SCALE
        p += FRACTAL_SHIFT
        if 6.0 / w < min_dist:
            stop += 1

    return de_box(p, m.Vector3(6.0, 6.0, 6.0), w)


def de_scene(p):
    # d_sphere = p.magnitude() - 0.5
    # d_sphere = de_box(p, m.Vector3(0.5,0.5,0.5), 1)
    d_sphere = de_fractal(p, 0)
    d_ground = 10000  # p.y + 0.5
    return min(d_sphere, d_ground)


def get_normal(p):
    x = m.Vector3(0.0001, 0.0, 0.0)
    y = m.Vector3(0.0, 0.0001, 0.0)
    z = m.Vector3(0.0, 0.0, 0.0001)
    return m.Vector3(de_scene(p+x), de_scene(p+y), de_scene(p+z)).normalized()


img_size = [128, 128]
imgArr = np.array(Image.new('RGB', (img_size[0], img_size[1]), color='black'))

cam_pos = m.Vector3(0.0, 3.5, -5)
cam_dir = m.Vector3(0.0, -0.3, 1.0).normalized()
light_dir = m.Vector3(0.3, -1.0, 0.3).normalized()

start_time = time.time()
for x in range(img_size[0]):
    for y in range(img_size[1]):
        pos = m.Vector3(cam_pos.x, cam_pos.y, cam_pos.z)
        dir = pixel_to_dir([x, y])
        hit = False
        total_dist = 0
        march = 0
        while march < 200:
            march += 1
            dist = de_scene(pos)
            pos += dir * dist
            total_dist += dist

            if total_dist > 30:
                total_dist = 30
                break
            if dist < 0.000001:
                hit = True
                break

        if hit:
            norm = get_normal(pos)
            ao = 1.0/(2.5*0.008*march + 1)
            foo = max(norm.dot(-light_dir) * 255, 0) * ao
            imgArr[img_size[1]-y-1, x] = [foo, foo, foo]
        else:
            imgArr[img_size[1]-y-1, x] = [235, 235, 240]

    print('Rendering: ', x/img_size[0] * 100, '%')
end_time = time.time()
print("Finished in: " + str(end_time - start_time) + "s")

img = Image.fromarray(imgArr)
plt.imshow(img)
plt.show()
