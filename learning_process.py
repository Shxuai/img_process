import numpy as np
import torch
import os
import time

from tqdm import trange

import train_data_process

CHUNK_SIZE = train_data_process.CHUNK_SIZE

img_data = [[], []]

for root, dirs, files in os.walk('resource\img_data', topdown=False):
    for name in files:
        if os.path.join(root, name).endswith('.chunk'):
            img_data[0].append(os.path.join(root, name))
        elif os.path.join(root, name).endswith('.pixel'):
            img_data[1].append(os.path.join(root, name))

chunk = []
pixel = []

for i in range(len(img_data[0])):
    chunk.append(np.fromfile(img_data[0][i], dtype=np.uint8))
    # chunk[i].resize(CHUNK_SIZE, CHUNK_SIZE, 25)
    local_pixel = np.fromfile(img_data[1][i], dtype=np.uint8).tolist()
    # print(type(local_pixel))
    # exit()
    for j in range(len(local_pixel)):
        local_pixel[j] = train_data_process.binary_encode(local_pixel[j], train_data_process.NUM_DIGITS)
    local_pixel = np.array(local_pixel).flatten()
    pixel.append(local_pixel)

trX = torch.Tensor(chunk)
trY = torch.Tensor(pixel)
del chunk, pixel

# Define the model
NUM_HIDDEN = 100
model = torch.nn.Sequential(
    torch.nn.Linear(CHUNK_SIZE * CHUNK_SIZE * 25, CHUNK_SIZE * 25),
    torch.nn.ReLU(),
    torch.nn.Linear(CHUNK_SIZE * 25, 25),
    torch.nn.Linear(25, 24),
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# Start training it
print('Start training!')
BATCH_SIZE = 128
for epoch in range(10000):
    for start in range(0, len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]

        y_pred = model(batchX)

        loss = loss_fn(y_pred.flatten(), batchY.flatten())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Find loss on training data
    loss = loss_fn(model(trX), trY).item()
    print('Epoch:', epoch, 'Loss:', loss)
