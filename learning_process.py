import numpy as np
import torch
import os

import train_data_process

img_data = [[], []]

for root, dirs, files in os.walk("resource\img_data", topdown=False):
    for name in files:
        if os.path.join(root, name).endswith('.chunk'):
            img_data[0].append(os.path.join(root, name))
        elif os.path.join(root, name).endswith('.pixel'):
            img_data[1].append(os.path.join(root, name))

for i in range(len(img_data[0])):
    chunk = np.fromfile(img_data[0][i], dtype=np.uint8)
    chunk.resize(train_data_process.CHUNK_SIZE, train_data_process.CHUNK_SIZE, 25)
    pixel = np.fromfile(img_data[1][i], dtype=np.uint8)
    print(pixel)
    exit()