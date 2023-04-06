import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from pathlib import Path
from importlib import reload
from IPython.display import clear_output
import time
import copy

from scipy import interpolate

import poppy

from . import utils
import misc_funs as misc

def calibrate(sysi, probe_amplitude, probe_modes, calibration_amplitude, calibration_modes, start_mode=0):
    print('Calibrating I-EFC...')
    slopes = [] # this becomes the response cube
    images = [] # this becomes the calibration cube
    # Loop through all modes that you want to control
    start = time.time()
    for ci, calibration_mode in enumerate(calibration_modes[start_mode::]):
        try:
            slope = 0
            # We need a + and - probe to estimate the jacobian
            for s in [-1, 1]:
                # Set the DM to the correct state
                sysi.add_dm(s * calibration_amplitude * calibration_mode.reshape(sysi.Nact, sysi.Nact))
                differential_images, single_images = take_measurement(sysi, probe_modes, probe_amplitude, return_all=True)
                sysi.add_dm(-s * calibration_amplitude * calibration_mode.reshape(sysi.Nact, sysi.Nact))
                
                slope += s * differential_images / (2 * calibration_amplitude)
                images.append(single_images)
            print("\tCalibrated mode {:d} / {:d} in {:.3f}s".format(ci+1+start_mode, calibration_modes.shape[0], 
                                                                    time.time()-start))
            slopes.append(slope)
        except KeyboardInterrupt: 
            print('Calibration interrupted.')
            break
    print('Calibration complete.')
    
    return np.array(slopes), np.array(images)

def construct_control_matrix(response_matrix, weight_map, nprobes=2, rcond=1e-2, WLS=True, pca_modes=None):
    weight_mask = weight_map>0
    
    # Invert the matrix with an SVD and Tikhonov regularization
    masked_matrix = response_matrix[:, :, weight_mask].reshape((response_matrix.shape[0], -1)).T
    
    # Add the extra PCA modes that are fitted
    if pca_modes is not None:
        double_pca_modes = np.concatenate( (pca_modes[:, weight_mask], pca_modes[:, weight_mask]), axis=1).T
        masked_matrix = np.hstack((masked_matrix, double_pca_modes))
    
    nmodes = int(response_matrix.shape[0])
    if WLS:  
        print('Using Weighted Least Squares ')
        if nprobes==2:
            Wmatrix = np.diag(np.concatenate((weight_map[weight_mask], weight_map[weight_mask])))
        elif nprobes==3:
            Wmatrix = np.diag(np.concatenate((weight_map[weight_mask], weight_map[weight_mask], weight_map[weight_mask])))
        control_matrix = utils.WeightedLeastSquares(masked_matrix[:,:nmodes], Wmatrix, rcond=rcond)
    else: 
        print('Using Tikhonov Inverse')
        control_matrix = utils.TikhonovInverse(masked_matrix[:,:nmodes], rcond=rcond)
    
    if pca_modes is not None:
        # Return the control matrix minus the pca_mode coefficients
        return control_matrix[0:-pca_modes.shape[0]]
    else:
        return control_matrix

def single_iteration(sysi, probe_cube, probe_amplitude, control_matrix, pixel_mask_dark_hole):
    # Take a measurement
    differential_images = take_measurement(sysi, probe_cube, probe_amplitude)
    
    # Choose which pixels we want to control
    measurement_vector = differential_images[:, pixel_mask_dark_hole].ravel()

    # Calculate the control signal in modal coefficients
    reconstructed_coefficients = control_matrix.dot( measurement_vector )
    
    return reconstructed_coefficients

# def take_measurement(system_interface, probe_cube, probe_amplitude, return_all=False, pca_modes=None):
def take_measurement(sysi, probe_cube, probe_amplitude, return_all=False, pca_modes=None, display=False):

    if probe_cube.shape[0]==2:
        differential_operator = np.array([[-1,1,0,0],
                                          [0,0,-1,1]]) / (2 * probe_amplitude * sysi.texp)
    elif probe_cube.shape[0]==3:
        differential_operator = np.array([[-1,1,0,0,0,0],
                                          [0,0,-1,1,0,0],
                                          [0,0,0,0,-1,1]]) / (2 * probe_amplitude * sysi.texp)
    
    amps = np.linspace(-probe_amplitude, probe_amplitude, 2)
    images = []
    for probe in probe_cube: 
        for amp in amps:
            sysi.add_dm(amp*probe)
            image = sysi.snap()
            images.append(image.flatten())
            sysi.add_dm(-amp*probe)
            
    images = np.array(images)
    
    differential_images = differential_operator.dot(images)
    
    if pca_modes is not None:
        differential_images = differential_images - (pca_modes.T.dot( pca_modes.dot(differential_images.T) )).T
        
    if display:
        if probe_cube.shape[0]==2:
            misc.imshow2(differential_images[0].reshape(sysi.npsf, sysi.npsf),
                           differential_images[1].reshape(sysi.npsf, sysi.npsf))
        elif probe_cube.shape[0]==3:
            misc.imshow3(differential_images[0].reshape(sysi.npsf, sysi.npsf),
                           differential_images[1].reshape(sysi.npsf, sysi.npsf),
                           differential_images[2].reshape(sysi.npsf, sysi.npsf))
    if return_all:
        return differential_images, images
    else:
        return differential_images
    
def run(sysi, 
        reg_fun, reg_conds, 
        response_matrix, 
        probe_modes, probe_amplitude,
        calibration_modes, 
        weights,
        num_iterations=10, 
        loop_gain=0.5, 
        leakage=0.0,
        display_current=True,
        display_all=False):
    print('Running I-EFC...')
    start = time.time()
    
    metric_images = []
    dm_commands = []
    
    dm_ref = sysi.get_dm()
    command = 0.0
    for i in range(num_iterations):
        print("\tClosed-loop iteration {:d} / {:d}".format(i+1, num_iterations))
        if i==0 or i in reg_conds[0]:
            reg_cond_ind = np.argwhere(i==reg_conds[0])[0][0]
            reg_cond = reg_conds[1, reg_cond_ind]
            print('\tComputing EFC matrix via ' + reg_fun.__name__ + ' with condition value {:.2e}'.format(reg_cond))
        
            control_matrix = reg_fun(response_matrix, 
                                       np.array(weights.flatten()), 
                                       nprobes=len(probe_modes),
                                       rcond=reg_cond, 
                                       pca_modes=None)
            
        delta_coefficients = -single_iteration(sysi, probe_modes, probe_amplitude, control_matrix, weights.flatten()>0)
        command = (1.0-leakage)*command + loop_gain*delta_coefficients
        
        # Reconstruct the full phase from the Fourier modes
        dm_command = calibration_modes.T.dot(command).reshape(sysi.Nact,sysi.Nact)

        # Set the current DM state
        sysi.set_dm(dm_ref + dm_command)

        # Take an image to estimate the metrics
        image = sysi.snap()
        
        metric_images.append(copy.copy(image))
        dm_commands.append(sysi.get_dm())
        
        if display_current: 
            if not display_all: clear_output(wait=True)
            misc.imshow2(dm_commands[i], image, 
                           'DM','Image: Iteration {:d}'.format(i+1),
                           lognorm2=True, vmin2=image.max()/1e7,
                           pxscl2=sysi.psf_pixelscale_lamD)
    print('I-EFC loop completed in {:.3f}s.'.format(time.time()-start))
    return metric_images, dm_commands


