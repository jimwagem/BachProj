# BachProj
This github repository contains data and programs used in the thesis
"Deep Learning for X-ray Tomography"
By Jim Wagemans (supervisor Felix Lucka) at the Centrum voor Wiskunde en Informatica (CWI).

The contents are organized into several directories.
enhancement_programs:
- fbpmodels: Saved neural network models trained on data generated by the FBP algorithm.
- sirtmodels: Saved neural network models trained on data generated by the SIRT algorithm.
- random_data: Example neural network for overfitting on random data.
- enhancer.py: Neural network that saves model parameters
- use_enhancer.py: Uses model parameter to generate an image.
- extend_enhancer.py: extend network training

reconstruction_programs:
- reconstruct_func.py: Contains functions for making reconstructions from data. Can be run to plot reconstruct and plot a single image.
- gen_data.py: Uses reconstruct_func to generate a dataset.
- gen_example: Generate example reconstructions

data:
- rawdata*: Raw scanning data. Each slice corresponds to 1 cross-section. Used as input for reconstruct_func/gen_data.
- fbp_data*: Images reconstructed using the FBP algorithm. Subfolder indicated reconstruction constraint.
- sirt_data*: Images reconstructed using the SIRT algorithm. Subfolder indicated reconstruction constraint.
- result_images: Reconstructions and neural network enhancements.
- training_losses: Loss per epoch for networks trained on different input.

* Too large, moved to zenodo
