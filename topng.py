# This file is used to convert the tiff results to png
import numpy as np 
import imageio as imi
import tifffile as tf

alg = 'fbp'
constraint = 'lim90'


tiff_path = f'/export/scratch3/jjow/{alg}_data/{constraint}/0rec0199.tiff'
save_path = f'/ufs/jjow/final_ims/rec_{alg}_{constraint}.png'
im = tf.imread(tiff_path)

# Image scaling and conversion to png
# All values above im_max are truncated
im_max = 0.01
im = im/im_max
im = np.clip(im, 0, 1)
im_uint16 = ((2**16 - 1)*im).astype(np.uint16)
imi.imwrite(save_path, im_uint16, prefer_uint8=False)