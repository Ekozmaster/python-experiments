"""
Artistic renders of the mandelbrot set.

For a cool FullHD render, use:
img = generate_mandelbrot_set((1920, 1080), fractal_itterations=42, cam_pos=[-0.85, 0.2], cam_scale=2.5)

There is a utils module with standard display resolutions to use.

The math used is really nicely covered in this link, take a look:
https://www.math.univ-toulouse.fr/~cheritat/wiki-draw/index.php/Mandelbrot_set
"""
import numpy as np
from PIL import Image
import time


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
                # Calculating the set's normal vector (u) used for lighting calculations.
                lo = 0.5 * np.log(z_sqr_mag)
                u = z * dz * ((1 + lo) * np.conjugate(dz ** 2) - lo * np.conjugate(z * dz2))
                u /= np.absolute(u)

                # Dot product between normal and light's direction vector.
                luminance = np.dot(np.array([u.real, u.imag]), np.array(light_dir))
                # Dot returns values between -1 and 1, so i like to set negative values between 0 and '0.something',
                # and positive values between '0.something' and 1, which prevents normals facing the opposite direction
                # from getting full dark, plus, in 3D, can even provide a simple subsurface scattering effect or some
                # pleasant ambient light effects.
                if luminance < 0.0:
                    luminance = luminance * 0.3 + 0.3
                else:
                    luminance = luminance * 0.7 + 0.3
                # Down here i'm using Reinhard Operator to tone-map the color range (luminance / (1 + luminance))
                # between -1 and 1 (HDR back to LDR), which makes it easier to add a fake ambient color and extra
                # exposure. Not very accurate, but i'll implement PBR in the future, which performs calculations on HDR.
                luminance = luminance * 2 + 0.1  # twice the exposure + 0.1 of ambient luminance.
                luminance = luminance / (1 + luminance)
                luminance = luminance**0.45  # gamma correction (pow(x, 1.0 / 2.2)

                luminance = max(0, min(1.0, luminance))
                img_arr[j, i] = [luminance * 255, luminance * 255, luminance * 255]

                # Antialiasing
                # Covered in the link at the top, antialiasing uses the distance estimator (DE) for Mandelbrot sets
                # as a way to interpolate colors at the boundary using the distance as interpolation factor.
                z_mag = z_sqr_mag ** 0.5
                distance = z_mag * 2.0 * np.log(z_mag) / np.absolute(dz)
                antialias_factor = distance / (1 * cam_scale / image_size[0])  # Thickness factor of 1 for 1080p images.
                antialias_factor = min(1, max(0, antialias_factor))
                img_arr[j, i] = lerp_color([10, 15, 10], img_arr[j, i], antialias_factor)
            else:
                img_arr[j, i] = [10 * z_sqr_mag * 4, 15 * z_sqr_mag * 1.5, 10]

        print('Rendering: {0:.3f} %'.format((float(i) / image_size[0]) * 100))
    print('Render time: ' + str(time.time() - t))

    return Image.fromarray(img_arr)
