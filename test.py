import numpy as np
import scipy as scp
import skimage.io
import astra
import matplotlib.pyplot as plt
import glob

def read1dtiff(filename):
    """Returns a 1d array"""
    if isinstance(filename,list):
        filename = filename[0]
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
data_dir = "./JimsBreakfast/TestScan_06-02-20_creamer/"
SOD = 203.549805
SDD = 528.014648

# CPU or GPU
useCPU = True

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

proj_geom = astra.create_proj_geom('fanflat', dec_pixel_size, detector_size, angles, SOD, SDD - SOD)

# Reading and preprocessing data
dark_field = averageField(dark_field_files, detector_size)
flat_field = averageField(light_field_files, detector_size)

# Read data
data = np.zeros([n_pro, detector_size])
dark_light_diff = (flat_field - dark_field)
if 0 in dark_light_diff:
    print("Division by 0")
    exit

for index, data_file in enumerate(projection_files):
    data_partial = np.array(read1dtiff(data_file))
    data_partial.astype(float)
    data_partial = np.divide((data_partial - dark_field),dark_light_diff)
    data[index] = data_partial

# Set all values smaller than 0 to the minimum (of values greater than 0)
# Also clip the values to 1.
min_data = min([elem for elem in data.flatten() if elem > 0])
data = [np.clip(sublist, min_data, 1) for sublist in data]

# Apply logarithm
data = np.array([-np.log(sublist) for sublist in data])

# Display sinogram
# plt.imshow(data)
# plt.show()
# Create astra object for reconstruction
recID = astra.data2d.create('-vol', vol_geom)
sinoID = astra.data2d.create('-sino', proj_geom, data)

if useCPU:
    proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)

    # Create configuration
    cfg = astra.astra_dict('SIRT')
    cfg['ReconstructionDataId'] = recID
    cfg['ProjectionDataId'] = sinoID
    cfg['ProjectorId'] = proj_id
else:
    #TODO implement GPU
    pass

# Create and run algorithm
alg_id = astra.algorithm.create(cfg)
iterations = 5
astra.algorithm.run(alg_id, iterations)

# Recieve reconstruction
rec = astra.data2d.get(recID)
plt.imshow(10*rec)
plt.show()
plt.imshow(100*rec)
plt.show()

# Clean up.
astra.algorithm.delete(alg_id)
astra.data2d.delete(recID)
astra.data2d.delete(sinoID)
if useCPU:
    astra.projector.delete(proj_id)
