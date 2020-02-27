import numpy as np
import scipy as scp
import skimage.io
import astra
import matplotlib.pyplot as plt
import glob

def read1dtiff(filename):
    """Returns a 1d array"""
    im = skimage.io.imread(filename)
    return im[0]

def averageField(filenames,size):
    result_field = np.zeros(size)
    for d_filename in filenames:
        temp_field = np.array(read1dtiff(d_filename))
        temp_field.astype(float)
        result_field += temp_field
    result_field = result_field / len(filenames)
    return result_field

# Data and Distance from source to object/detector respectively.
data_dir = "/JimsBreakfast/TestScan_06-02-20_fakecoffee_highres/"
SOD = 423.560547
SDD = 528.014648

# Preprocessing
binning = 1
exclude_last_pro = True

# Reconstruction paramters
nXYBase = 64
scale_fac = 8
ang_sub_samp = 1
lim_ang = 360


# Get light and dark files from directory
dark_field_files = glob.glob(data_dir + "di*.tif")
light_field_files = glob.glob(data_dir + "io*.tif")

detector_size = len(read1dtiff(dark_field_files))

projection_files = glob.glob(data_dir + "scan_*.tif")
angles = np.linspace(0, 2*np.pi, len(projection_files))
if (exclude_last_pro):
    projection_files = projection_files[:-1]
    angles = angles[:-1]
projection_files = projection_files[0::ang_sub_samp]
angles = angles[0::ang_sub_samp]
ang_ind = [index for index, angle in enumerate(angles) if angle <= lim_ang/180*np.pi]
angles = [angles[index] for index in ang_ind]
projection_files = [projection_files[index] for index in ang_ind]
n_pro = len(angles)

voxel_size = 1 / scale_fac

# Reconstruction geometry
nX = nXYBase * scale_fac
nY = nX
n = nX*nY
nXY = [nX, nY]
vol_geom = astra.create_vol_geom(nX, nX)

# Set up fan beam projection geometry
SDD *= scale_fac
SOD *= scale_fac
dec_pixel_size = binning * 0.074800 * scale_fac

proj_geo = astra.create_proj_geom('fanflat', dec_pixel_size, detector_size[1], angles, SOD, SDD - SOD)

# Reading and preprocessing data
dark_field = averageField(dark_field_files)
flat_field = averageField(light_field_files)

# Read data
data = np.zeros(projection_files)
dark_light_diff = (flat_field - dark_field)
if 0 in dark_light_diff:
    print("Division by 0")
    exit

for data_file in projection_files:
    data_temp = np.array(read1dtiff(data_file))
    data_temp.astype(float)
    data_temp = np.divide((data_temp - dark_field),dark_light_diff)
    data += data_temp
print(data)
# # Reading files
# im = skimage.io.imread('test.tif')
# plt.plot(im[0])
# plt.show()

