# This file contains a function for image reconstruction using the ASTRA toolbox.
# This is a python adaptation of a Matlab script.
# Running this script will plot a reconstruction.
# To generate multiple reconstruction ans save them, use gen_data.py 
import numpy as np
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

def reconstruct(settings):
    data_dir = settings['data_dir']
    SOD = settings['SOD']
    SDD = settings['SDD']
    useCPU = settings['useCPU']
    binning = settings['binning']
    exclude_last_pro = settings['exclude_last_pro']
    nXYBase = settings['nXYBase']
    scale_fac = settings['scale_fac']
    ang_sub_samp = settings['ang_sub_samp']
    lim_ang = settings['lim_ang']
    alg_name = settings['alg_name']
    offset = settings['offset']
    det_sub_samp = settings['det_sub_samp']

    # Get light and dark files from directory
    dark_field_files = sorted(glob.glob(data_dir + "di*.tif"))
    light_field_files = sorted(glob.glob(data_dir + "io*.tif"))

    detector_size = len(read1dtiff(dark_field_files[0]))
    detector_size = int(detector_size/det_sub_samp)
    print("detector_size: " + str(detector_size))

    projection_files = sorted(glob.glob(data_dir + "scan_*.tif"))
    angles = np.linspace(0, 2*np.pi, len(projection_files))
    if (exclude_last_pro):
        projection_files = projection_files[:-1]
        angles = angles[:-1]
    projection_files = projection_files[0::ang_sub_samp]
    angles = np.roll(angles, offset)
    angles = angles[::ang_sub_samp]
    ang_ind = [index for index, angle in enumerate(angles) if angle <= (lim_ang/180)*np.pi]
    angles = [angles[index] for index in ang_ind]
    projection_files = [projection_files[index] for index in ang_ind]
    n_pro = len(angles) 

    voxel_size = 1 / scale_fac

    # Reconstruction geometry
    nX = nXYBase * scale_fac
    nY = nX
    n = nX*nY
    nXY = [nX, nY]
    vol_geom = astra.create_vol_geom(nX, nY)

    # Set up fan beam projection geometry
    SDD *= scale_fac
    SOD *= scale_fac
    dec_pixel_size = binning * 0.074800 * scale_fac * det_sub_samp

    proj_geom = astra.create_proj_geom('fanflat', dec_pixel_size, detector_size, angles, SOD, SDD - SOD)

    # Reading and preprocessing data
    dark_field = averageField(dark_field_files, detector_size*det_sub_samp)
    flat_field = averageField(light_field_files, detector_size*det_sub_samp)
    dark_field = dark_field[0::det_sub_samp]
    flat_field = flat_field[0::det_sub_samp]

    # Read data
    data = np.zeros([n_pro, detector_size])
    dark_light_diff = (flat_field - dark_field)
    if 0 in dark_light_diff:
        print("Division by 0")
        exit()

    for index, data_file in enumerate(projection_files):
        data_partial = np.array(read1dtiff(data_file))
        data_partial.astype(float)
        data_partial = data_partial[0::det_sub_samp]
        data_partial = np.divide((data_partial - dark_field),dark_light_diff)
        data[index] = data_partial

    # Set all values smaller than 0 to the minimum (of values greater than 0)
    # Also clip the values to 1.
    min_data = min([elem for elem in data.flatten() if elem > 0])
    data = [np.clip(sublist, min_data, 1) for sublist in data]

    # Apply logarithm
    data = np.array([-np.log(sublist) for sublist in data])

    # Display sinogram
    plt.imshow(data, cmap='gray')
    plt.xlabel('t')
    plt.ylabel('$\\theta$')
    plt.show()
    # Create astra object for reconstruction
    recID = astra.data2d.create('-vol', vol_geom)
    sinoID = astra.data2d.create('-sino', proj_geom, data)

    
    if useCPU:
        cfg = astra.astra_dict(alg_name)
        proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)
        cfg['ProjectorId'] = proj_id
    else:
        cfg = astra.astra_dict(alg_name + '_CUDA')

    # Create configuration
    cfg['ReconstructionDataId'] = recID
    cfg['ProjectionDataId'] = sinoID
    cfg['option'] = {'FilterType': 'Ram-Lak',
                    'MinConstraint': 0.}

    # Create and run algorithm
    alg_id = astra.algorithm.create(cfg)
    iterations = settings['iterations']
    astra.algorithm.run(alg_id, iterations)

    # Recieve reconstruction
    rec = astra.data2d.get(recID)
    rec = np.maximum(rec,0)
    # plt.imshow(rec, cmap='gray')
    # plt.show()
    # disp(rec)


    # Clean up.
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(recID)
    astra.data2d.delete(sinoID)
    if useCPU:
        astra.projector.delete(proj_id)
    return rec

if __name__ == "__main__":
    # Data and Distance from source to object/detector respectively.
    data_dir_str = "/export/scratch3/jjow/Scan-12-03-2020/slice0001/"
    SOD = 423.562012
    SDD = 527.997070

    # CPU or GPU
    useCPU = False

    # Preprocessing
    binning = 1
    exclude_last_pro = True

    # Reconstruction paramters
    nXYBase = 128
    scale_fac = 4
    ang_sub_samp = 1
    det_sub_samp = 1
    lim_ang = 360

    settings = {}

    settings['SOD'] = SOD 
    settings['SDD'] = SDD
    settings['data_dir'] = data_dir_str
    settings['useCPU'] = useCPU
    settings['binning'] = binning
    settings['exclude_last_pro'] = exclude_last_pro 
    settings['nXYBase'] = nXYBase 
    settings['scale_fac'] = scale_fac 
    settings['ang_sub_samp'] = ang_sub_samp 
    settings['lim_ang'] = lim_ang
    settings['alg_name'] = 'FBP'
    settings['iterations'] = 500
    settings['offset'] = 0
    settings['det_sub_samp'] = det_sub_samp
    rec = reconstruct(settings)
    plt.imshow(rec, cmap='gray')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
