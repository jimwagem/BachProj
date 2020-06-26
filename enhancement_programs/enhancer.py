import msd_pytorch as mp
from torch.utils.data import DataLoader
import numpy as np
import torch

# Network params
c_in = 1
depth = 40
width = 1
dilations = [1 + i for i in range(10)]

loss = "L2"
c_out = 1

# Trainings params
epochs = 100
batch_size = 5

# Data params
print('loading data...')
base_dir = "/export/scratch3/jjow/sirt_data/"
input_str = "dec12/"
output_str = "full/"
train_input_glob = base_dir + input_str + "*.tiff"
train_target_glob = base_dir + output_str + "*.tiff"

val_input_glob = base_dir + input_str + "val/*.tiff"
val_target_glob = base_dir + output_str + "val/*.tiff"
train_ds = mp.ImageDataset(train_input_glob, train_target_glob) 
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

val_ds = mp.ImageDataset(val_input_glob, val_target_glob)
val_dl = DataLoader(val_ds, batch_size, shuffle=False)

# Create model
model = mp.MSDRegressionModel(c_in, c_out, depth, width, dilations=dilations, loss=loss)
model.set_normalization(train_dl)

best_val_err = np.inf
val_err = 0

train_errs = []
val_errs = []

save_dir = './sirtmodels/' + input_str

for epoch in range(epochs):
    model.train(train_dl,1)
    train_error = model.validate(train_dl)
    train_errs.append(train_error)
    validation_error = model.validate(val_dl)
    val_errs.append(validation_error)
    print(f"epoch: {epoch} -- val err: {validation_error}, train err: {train_error}")
    if validation_error < best_val_err:
        best_val_err = validation_error
        model.save(save_dir + f'modelparam_e{epoch}.torch', epoch)
np.save(save_dir + f'train_err_e{epoch}.npy', train_errs)
np.save(save_dir + f'val_err_e{epoch}.npy', val_errs)
model.save(save_dir + f'modelparam_e{epoch}.torch',epoch)
