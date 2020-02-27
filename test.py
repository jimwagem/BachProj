import numpy as np
import scipy as scp
import skimage.io
import astra
import matplotlib.pyplot as plt
import glob

def read1dtiff(filename):
    """Returns a 1d array"""
    im = skimage.io.imread(filename)
    return im

# Data and Distance from source to object/detector respectively.
dataDir = "./JimsBreakfast/TestScan_06-02-20_fakecoffee_highres/"
SOD = 423.560547
SDD = 528.014648

# Preprocessing
binning = 1
excludeLastPro = True

# Reconstruction paramters
nXYBase = 64
scaleFac = 8
angSubSamp = 1
limAng = 360


# Get light and dark files from directory
darkFieldFiles = glob.glob(dataDir + "di*.tif")
lightFieldFiles = glob.glob(dataDir + "io*.tif")

detectorSize = np.array(read1dtiff(darkFieldFiles[0])).shape

projectionFiles = glob.glob(dataDir + "scan_*.tif")
angles = np.linspace(0, 2*np.pi, len(projectionFiles))
if (excludeLastPro):
    projectionFiles = projectionFiles[:-1]
    angles = angles[:-1]
projectionFiles = projectionFiles[0::angSubSamp]
angles = angles[0::angSubSamp]
angInd = [index for index, angle in enumerate(angles) if angle <= limAng/180*np.pi]
angles = [angles[index] for index in angInd]
projectionFiles = [projectionFiles[index] for index in angInd]
nPro = len(angles)

voxelSize = 1 / scaleFac

# Reconstruction geometry
nX = nXYBase * scaleFac
nY = nX
n = nX*nY
nXY = [nX, nY]
vol_geom = astra.create_vol_geom(nX, nX)

# Set up fan beam projection geometry
SDD *= scaleFac
SOD *= scaleFac
decPixelSize = binning * 0.074800 * scaleFac

projGeo = astra.create_proj_geom('fanflat', decPixelSize, detectorSize[1], angles, SOD, SDD - SOD)


# # Reading files
# im = skimage.io.imread('test.tif')
# plt.plot(im[0])
# plt.show()

