# Based on examply.py from msd_pytorch github
import msd_pytorch as mp
from torch.utils.data import DataLoader
import numpy as np
import torch
import tifffile as tf

# Network params
c_in = 1
depth = 50
width = 1
dilations = [i for i in range(10)]

loss = "L2"
c_out = 1

# Recreate model
print('loading model')
model = mp.MSDRegressionModel(c_in, c_out, depth, width, dilations=dilations, loss=loss)
model_path = '' 
model.load(model_path)

# Get image
new_data_path = './export/scratch3/jjow/fbp_data/fbp_sub8/rec0001.tiff'
print('loading dataset')
ds = mp.ImageDataset(new_data_path, new_data_path)
dl = Dataloader(ds, 1, shuffle=False)

save_path = './network_rec.tiff'
for i, data in enumerate(dl):
    # Unpack data
    inp, _ = data
    output = model.net(inp.cuda())
    output_np = output.detatch().cpu().numpy()
    output_np = output_np.squeeze()
    tf.imsave(save_path, output_np)
