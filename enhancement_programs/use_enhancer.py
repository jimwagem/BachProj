# Based on examply.py from msd_pytorch github
import msd_pytorch as mp
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import tifffile as tf
import imageio as imi

# Network params
c_in = 1
depth = 40
width = 1
dilations = [i + 1 for i in range(10)]

loss = "L2"
c_out = 1

# Recreate model
print('loading model')
model = mp.MSDRegressionModel(c_in, c_out, depth, width, dilations=dilations, loss=loss)
constraint = 'sub12'
alg = 'sirt'
model_path = './programs/' + alg + 'models/' + constraint + '/modelparam_e99.torch'
model.load(model_path)

# Get image
data_path = '/export/scratch3/jjow/' + alg + '_data/' + constraint + '/val/0rec0199.tiff'
print('loading dataset')
ds = mp.ImageDataset(data_path, data_path)
dl = DataLoader(ds, 1, shuffle=False)

for i, data in enumerate(dl):
    # Unpack data
    inp, _ = data
    output = model.net(inp.cuda())
    output_np = output.detach().cpu().numpy()
    output_np = output_np.squeeze()
    
    # Saving image
    # All values above max get truncated
    save_dir = f'./transfer/enh_{alg}_{constraint}.png'
    im_max = 0.01
    im_float = output_np/im_max
    im_float = np.clip(im_float, 0, 1)
    im_uint16 = ((2**16 - 1)*im_float).astype(np.uint16)
    imi.imwrite(save_dir, im_uint16, prefer_uint8=False)
    #plt.imshow(im_uint16, cmap='gray')
    #plt.show()

