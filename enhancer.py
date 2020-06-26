# Based on examply.py from msd_pytorch github
import msd_pytorch as mp
from torch.utils.data import DataLoader
import numpy as np
import torch

# Network params
c_in = 1
depth = 50
width = 1
dilations = [i for i in range(10)]

loss = "L2"
c_out = 1

# Trainings params
epochs = 10
batch_size = 10

# Data params
train_input_glob = "/export/scratch3/jjow/1000fbp_sub2/*.tiff"
train_target_glob = "/export/scratch3/jjow/1000fbp_full/*.tiff"
train_ds = mp.ImageDataset(train_input_glob, train_target_glob) 
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Create model
model = mp.MSDRegressionModel(c_in, c_out, depth, width, dilations=dilations, loss=loss)
model.set_normalization(train_dl)

best_val_err = np.inf
val_err = 0

for epoch in range(epochs):
    model.train(train_dl,1)
    train_error = model.validate(train_dl)
    print(f"epoch: {epoch} -- training error: {train_error:0.6f}")
model.save(f'msdnet_epoch_{epoch}.torch',epoch)
