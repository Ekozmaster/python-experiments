import numpy as np
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
import noise  # Using simplex noise 2D which goes from -1 to 1

matplotlib.use('TkAgg')
img_size = (256, 256)

imag = Image.new('RGB', img_size, color='black')
img_arr = np.array(imag)

cell_size = 30


def random_color(pos_x, pos_y):
    return [int(128 + noise.snoise2(pos_x * 124.324 - 519.6534, pos_y * 312.1253 + 812.1241) * 128),
            int(128 + noise.snoise2(pos_y * 224.324 - 219.6534, pos_x * 414.1253 + 912.1241) * 128),
            int(128 + noise.snoise2(pos_x * 274.324 - 119.6534, pos_y * 114.1253 + 512.1241) * 128)]


def random_pos(pos_x, pos_y):
    return [pos_x * cell_size + noise.snoise2(pos_x * 124.324 - 519.6534, pos_y * 312.1253 + 812.1241) * cell_size * 0.5,
            pos_y * cell_size + noise.snoise2(pos_y * 224.324 - 219.6534, pos_x * 414.1253 + 912.1241) * cell_size * 0.5]


def distance(pos_x, pos_y, cell_pos):
    return ((pos_x - cell_pos[0])**2 + (pos_y - cell_pos[1])**2)**0.5


def generate_voronoi_diagram():
    for x in range(img_arr.shape[0]):
        for y in range(img_arr.shape[1]):
            cur_cell = [int(x / cell_size), int(y / cell_size)]

            closer_cell = [cur_cell[0] - 1, cur_cell[1] - 1]
            closer_cell_pos = random_pos(closer_cell[0], closer_cell[1])
            closer_dist = distance(x, y, closer_cell_pos)

            for i in range(cur_cell[0] - 1, cur_cell[0] + 2):
                for j in range(cur_cell[1] - 1, cur_cell[1] + 2):
                    neigh_cell = [i, j]
                    neigh_pos = random_pos(neigh_cell[0], neigh_cell[1])
                    dist = distance(x, y, neigh_pos)

                    if dist < closer_dist:
                        closer_cell = neigh_cell
                        closer_dist = dist

            img_arr[x, y] = random_color(closer_cell[0], closer_cell[1])

        print("Rendering: {0:.3f} %".format(float(x)/img_arr.shape[0] * 100))

generate_voronoi_diagram()
imag = Image.fromarray(img_arr)

plt.imshow(imag)
plt.show()