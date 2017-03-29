from matplotlib import pyplot as plt
import numpy as np

lines = 10
colums = 10

imgs = np.load("imgs.npz").items()[-1][1][-2]
index = 1
imgs_num = len(imgs)

for i in range(lines):
    for j in range(colums):
        plt.subplot(lines, colums, index)
        img_index = np.random.randint(imgs_num)
        plt.imshow(imgs[img_index].reshape((28, 28)), cmap="gray")
        plt.axis("off")
        index += 1
plt.show()