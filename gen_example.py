import astra
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import tifffile as tf
import imageio as imi

# Setup geometries
vol_size = 512
detector_size = 1944
angles = np.linspace(0,np.pi,1000)
vol_geom = astra.create_vol_geom(vol_size, vol_size)
proj_geom = astra.create_proj_geom('parallel', 1.0, detector_size, angles)

# Create origonal images
im = Image.new('F', (vol_size, vol_size))
draw = ImageDraw.Draw(im)
draw.ellipse((200, 200, 300, 300), width=2)


# Scale image to max
I = np.array(im)

# Projector
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
sino_id, sino = astra.create_sino(I, proj_id)

# Change these to generate artifacts
# Angular limit goes up to 1000
# detector_subsample must divide detector_size
detector_subsample = 1
angular_subsample = 1
angular_limit = 500

detector_size2 = int(detector_size/detector_subsample)
angles2 = angles

angles2 = angles2[0:angular_limit]
angles2 = angles2[0::angular_subsample]
sino = sino[0:angular_limit]
sino = sino[0::angular_subsample,0::detector_subsample]


proj_geom2 = astra.create_proj_geom('parallel', detector_subsample, detector_size2, angles2)
sino_id2 = astra.data2d.create('-sino', proj_geom2, sino)


# Reconstruction
rec_id = astra.data2d.create('-vol', vol_geom)
cfg = astra.astra_dict('FBP_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sino_id2
cfg['option'] = {'FilterType':'Ram-Lak'}
alg_id = astra.algorithm.create(cfg)

astra.algorithm.run(alg_id)
rec = astra.data2d.get(rec_id)

# # Decomment this to plot image
# cutoff = 0.3
# plt.imshow(rec, cmap='gray', vmax=cutoff)
# plt.show()

# # Decomment this to save as tif
# tif_limit = 0.01
# scale_fac = tif_limit/im.max()
# tf.imsave('./', scale_fac*im)

# Save as png
cutoff = 0.3
rec = np.clip(rec, 0, cutoff)
# Scale max to 1
rec = rec/cutoff
rec_uint16 = ((2**16 - 1)* rec).astype(np.uint16)
imi.imwrite('/ufs/jjow/final_ims/example_lim90.png', rec_uint16, prefer_uint8=False)
