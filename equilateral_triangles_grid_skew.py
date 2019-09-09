import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')


# If you rotate the i and j canonical base's axes by 15 degrees CW and CCW respectively so that they get
# 120 degrees from each other, you get a skewed grid with equilateral triangles. That way instead of making bilinear
# interpolations, you can easily interpolate inside the current triangle.
def skewed_grid():
    system_basis = np.mat([[0.96592582628, -0.2588190451], [-0.2588190451, 0.96592582628]])
    for x in range(-3, 3):
        x1 = np.array(system_basis * np.mat([x, -3]).transpose())
        x2 = np.array(system_basis * np.mat([x, 3]).transpose())
        ax.plot([x1[0], x2[0]], [x1[1], x2[1]], color='gray')

    for y in range(-3, 3):
        y1 = np.array(system_basis * np.mat([-3, y]).transpose())
        y2 = np.array(system_basis * np.mat([3, y]).transpose())
        ax.plot([y1[0], y2[0]], [y1[1], y2[1]], color='gray')


skewed_grid()
bases = np.array([[0.96592582628, -0.2588190451], [-0.2588190451, 0.96592582628]])

ax.quiver(0, 0, bases[:, 0], bases[:, 1], scale=1, angles='xy', scale_units='xy', color=['r', 'g'])

plt.xlim(-2, 2)
plt.ylim(-2, 2)

ax.grid(True)
plt.show()
