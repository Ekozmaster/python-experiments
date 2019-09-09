"""
Voronoi Diagram is a partitioning of a plane in regions determined by a set of points/centers AKA seeds. It is kind of
if you had a set of hospitals spread in your city, and you want to partition the city in regions/cells to map what is
the closest hospital for a given position.

This code implements a fast version of it which is extremely useful for procedural generation of anything that must be
evenly-randomly distributed. They can be used in many applications from trees placement in a game to whole biomes
generation in minecraft-like games.
Take a look in the Cuberite Minecraft Server's write on the topic for a good application:
http://cuberite.xoft.cz/docs/Generator.html

It consists in, for a given point (say, a pixel), find which cell it belongs to, iterating on all 3 by 3 neighbour
cells in a discrete grid/lattice, randomizing their centers using their "grid_id" as seed to the randomizer and
calculating the distances from them. The closest grid cell is the cell it belongs to.
"""
import numpy as np
from PIL import Image
import noise  # Using simplex noise 2D which goes from -1 to 1 to randomize cells positions and colors.

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
    img_size = (256, 256)
    imag = Image.new('RGB', img_size, color='black')
    img_arr = np.array(imag)

    for x in range(img_arr.shape[0]):
        for y in range(img_arr.shape[1]):
            # Current cell is the center cell in the 3 by 3 neighbourhood.
            cur_cell = [int(x / cell_size), int(y / cell_size)]
            closer_cell = [0, 0]
            closer_dist = 9999999

            # Iterating in the 3 by 3 cell' neighbourhood.
            for i in range(cur_cell[0] - 1, cur_cell[0] + 2):
                for j in range(cur_cell[1] - 1, cur_cell[1] + 2):
                    neigh_cell = [i, j]
                    # Using the grid's (x,y) id to randomly offset it's center.
                    neigh_pos = random_pos(neigh_cell[0], neigh_cell[1])
                    dist = distance(x, y, neigh_pos)

                    if dist < closer_dist:
                        closer_cell = neigh_cell
                        closer_dist = dist

            # Using the grid's (x,y) id to pick a random color to identify it.
            img_arr[x, y] = random_color(closer_cell[0], closer_cell[1])

        print("Rendering: {0:.3f} %".format(float(x)/img_arr.shape[0] * 100))
    return Image.fromarray(img_arr)
