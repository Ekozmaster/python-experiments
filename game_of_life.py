# Conway's Game of Life
# Consists in a cellular automata that simulates pretty cool emergent patterns of cells (pixels if you want)
# moving around and morphing based on the environment and surroundings.
# Each cell must follow very simple rules, based on how many neighbours it has:
# 1 - If there is less than two neighbours, it dies.
# 2 - If there is exactly two neighbours, it survives to the next iteration.
# 3 - If there is exactly three neighbours, it borns (comes into existence).
# 4 - If there are more than 3, it dies, kind of from overpopulation/suffocation.
# This implementation uses two boards, one to save the state of the previous frame and the other to be updated,
# then they are swapped, and the process repeat.

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

resolution = (100, 100)


# Here is where cool initialization methods might go in. Right now it is just random.
def init_boards():
    board_a = np.random.randint(0, 2, resolution) * 255
    board_b = np.random.randint(0, 2, resolution) * 255
    return board_a, board_b


def update_boards(board_a, board_b):
    # Not updating edges, so that it can iterate safely on all 3x3 neighbourhood of each cell.
    for i in range(1, resolution[0] - 1):
        for j in range(1, resolution[1] - 1):
            neighbour_count = 0
            for x in range(i - 1, i + 2):
                for y in range(j - 1, j + 2):
                    if board_a[x, y] and (x != i or y != j):
                        neighbour_count += 1

            if neighbour_count < 2:  # Less than two neighbours, die.
                board_b[i, j] = 0
            elif neighbour_count == 2:  # Nothing happens
                board_b[i, j] = board_a[i, j]
            elif neighbour_count == 3:  # A new cell borns
                board_b[i, j] = 255
            else:  # Dies from overpopulation
                board_b[i, j] = 0


def run():
    running = True
    board_a, board_b = init_boards()
    x = int(resolution[0]/2)
    y = int(resolution[1]/2)
    while running:
        plt.cla()
        update_boards(board_a, board_b)

        # swap boards
        temp = board_a
        board_a = board_b
        board_b = temp

        plt.imshow(Image.fromarray(board_b.astype(np.int8)))
        plt.pause(1/60)


run()
