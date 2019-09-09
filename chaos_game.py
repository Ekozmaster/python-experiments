import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img_size = [1024, 1024]
imgArr = np.array(Image.new('RGB', (img_size[0], img_size[1]), color='black'))
# [[y, x], [y, x], [y, x], ...]
vertices = np.array([[0.1, 0.5], [0.9, 0.1], [0.9, 0.9]])
lerp_factor = 0.5
cur_point = np.array([0.3, 0.5])
pixel_color = np.array([255, 255, 255])


def lerp_vectors(a, b, t):
    return (b-a) * t + a


for i in range(300000):
    imgArr[int(cur_point[0] * img_size[0]), int(cur_point[1] * img_size[1])] = pixel_color
    cur_point = lerp_vectors(cur_point, vertices[np.random.randint(0, len(vertices))], lerp_factor)

img = Image.fromarray(imgArr)
plt.imshow(img)
plt.show()
