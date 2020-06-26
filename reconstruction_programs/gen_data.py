import reconstruct_func
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import tifffile as tf

def number_to_text(inp):
    """convert integer to 4 digit string with leading zero's"""
    lead_zeros = "0"*4 + str(inp)
    return lead_zeros[-4:]


if __name__ == "__main__":
    # Data and Distance from source to object/detector respectively.
    data_dir_str = "/export/scratch3/jjow/Scan-12-03-2020/slice"
    save_dir_str = "/export/scratch3/jjow/fbp_data/dec12/"
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
    lim_ang = 360
    det_sub_samp = 12

    settings = {}

    settings['SOD'] = SOD 
    settings['SDD'] = SDD
    settings['useCPU'] = useCPU
    settings['binning'] = binning
    settings['exclude_last_pro'] = exclude_last_pro 
    settings['nXYBase'] = nXYBase 
    settings['scale_fac'] = scale_fac 
    settings['ang_sub_samp'] = ang_sub_samp 
    settings['lim_ang'] = lim_ang
    settings['alg_name'] = 'FBP'
    settings['iterations'] = 500
    settings['det_sub_samp'] = det_sub_samp
    
    num_offsets = 5
    base_offset = 300
    slice_range = np.arange(1,202)
    for index in slice_range:
        for offset_ind in np.arange(num_offsets):
            settings['offset'] = offset_ind * base_offset
            print(f'Reconstructing: {index}, offset: {offset_ind}')
            # Get datapath
            index_str = number_to_text(index)
            settings['data_dir'] = data_dir_str + index_str + '/'

            # Reconstruct
            rec = reconstruct_func.reconstruct(settings)

            # Save image
            save_file = save_dir_str + f"{offset_ind}rec{index_str}.tiff"
            tf.imsave(save_file, rec.astype(np.double))