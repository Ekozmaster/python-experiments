import numpy as np
from PIL import Image
import time
from color_correction import hsv_to_rgb


def lerp_color(a, b, t):
    return [(b[0]-a[0]) * t + a[0], (b[1]-a[1]) * t + a[1], (b[2]-a[2]) * t + a[2]]


def generate_mandelbrot_set(image_size=(128, 128), fractal_itterations=20, cam_scale=3.0, cam_pos=[0, 0]):
    img_arr = np.array(Image.new('RGB', image_size, color='black'))

    aspect_ratio = float(image_size[0]) / image_size[1]
    light_dir = (0.7071, -0.7071)
    t = time.time()
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            z = np.complex(0, 0)
            z_sqr_mag = 0
            dz = np.complex(0, 0)
            dz2 = np.complex(0, 0)
            c = np.complex((float(i) / image_size[0] * cam_scale - cam_scale/2 + cam_pos[0]) * aspect_ratio,
                           float(j) / image_size[1] * cam_scale - cam_scale/2 - cam_pos[1])

            for itter in range(fractal_itterations):
                dz2 = 2 * (dz2 * z + dz ** 2)
                dz = 2 * z * dz + 1.0
                z = z**2 + c

                z_sqr_mag = z.real**2 + float(z.imag)**2
                if z_sqr_mag > 100.0:
                    break
            if z_sqr_mag > 4.0:
                lo = 0.5 * np.log(z_sqr_mag)
                u = z * dz * ((1 + lo) * np.conjugate(dz ** 2) - lo * np.conjugate(z * dz2))
                #u = z / dz
                u /= np.absolute(u)

                lightColor = np.dot(np.array([u.real, u.imag]), np.array([light_dir[0], light_dir[1]]))
                if lightColor < 0.0:
                    lightColor = lightColor * 0.3 + 0.3
                else:
                    lightColor = lightColor * 0.7 + 0.3
                lightColor = lightColor * 2 + 0.1
                lightColor = lightColor / (1 + lightColor)
                lightColor = lightColor**0.45

                lightColor = max(0, min(1.0, lightColor))
                img_arr[j, i] = [lightColor * 255, lightColor * 255, lightColor * 255]

                # Antialiasing
                z_mag = z_sqr_mag ** 0.5
                distance = z_mag * 2.0 * np.log(z_mag) / np.absolute(dz)
                antialiasFactor = distance / (1 * cam_scale / image_size[0])
                antialiasFactor = min(1, max(0, antialiasFactor))
                img_arr[j, i] = lerp_color([10, 15, 10], img_arr[j, i], antialiasFactor)
                #img_arr[j, i] = orbit_trap(z)
            else:
                img_arr[j, i] = [10 * z_sqr_mag * 4, 15 * z_sqr_mag * 1.5, 10]

        print('Rendering: {0:.3f} %'.format((float(i) / image_size[0]) * 100))
    print('Render time: ' + str(time.time() - t))

    return Image.fromarray(img_arr)