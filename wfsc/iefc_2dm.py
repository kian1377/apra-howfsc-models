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

from importlib import reload
reload(utils)

from cgi_phasec_poppy import misc

# def take_measurement(system_interface, probe_cube, probe_amplitude, return_all=False, pca_modes=None):
def take_measurement(sysi, probe_cube, probe_amplitude, DM=1, return_all=False, pca_modes=None, display=False):

    if probe_cube.shape[0]==2:
        differential_operator = np.array([[-1,1,0,0],
                                          [0,0,-1,1]]) / (2 * probe_amplitude)
    elif probe_cube.shape[0]==3:
        differential_operator = np.array([[-1,1,0,0,0,0],
                                          [0,0,-1,1,0,0],
                                          [0,0,0,0,-1,1]]) / (2 * probe_amplitude)
    
    amps = np.linspace(-probe_amplitude, probe_amplitude, 2)
    images = []
    for probe in probe_cube: 
        for amp in amps:
            if DM==1:
                sysi.add_dm1(amp*probe)
                psf = sysi.snap()
                images.append(psf.flatten())
                sysi.add_dm1(-amp*probe)
            elif DM==2:
                sysi.add_dm2(amp*probe)
                psf = sysi.snap()
                images.append(psf.flatten())
                sysi.add_dm2(-amp*probe)
            
    images = np.array(images)
    
    differential_images = differential_operator.dot(images)
    
    if pca_modes is not None:
        differential_images = differential_images - (pca_modes.T.dot( pca_modes.dot(differential_images.T) )).T
        
    if display:
        if probe_cube.shape[0]==2:
            misc.myimshow2(differential_images[0].reshape(sysi.npsf, sysi.npsf),
                           differential_images[1].reshape(sysi.npsf, sysi.npsf))
        elif probe_cube.shape[0]==3:
            misc.myimshow3(differential_images[0].reshape(sysi.npsf, sysi.npsf),
                           differential_images[1].reshape(sysi.npsf, sysi.npsf),
                           differential_images[2].reshape(sysi.npsf, sysi.npsf))
    if return_all:
        return differential_images, images
    else:
        return differential_images
    
def calibrate(sysi, probe_amplitude, probe_modes, calibration_amplitude, calibration_modes, start_mode=0):
    print('Calibrating I-EFC...')
    
    slopes_1 = []
    slopes_2 = []
    images_1 = []
    images_2 = []
    
    # Loop through all modes that you want to control
    start = time.time()
    for ci, calibration_mode in enumerate(calibration_modes[start_mode::]):
        try:
            slope1, slope2 = (0, 0)
            for s in [-1, 1]: # We need a + and - probe to estimate the jacobian
                # DM1: Set the DM to the correct state
                sysi.add_dm1(s * calibration_amplitude * calibration_mode.reshape(sysi.Nact, sysi.Nact))
                differential_images_1, single_images_1 = take_measurement(sysi, probe_modes, probe_amplitude, DM=1,
                                                                          return_all=True)
                
                images_1.append(single_images_1)
                slope1 += s * differential_images_1 / (2 * calibration_amplitude)
                
                sysi.add_dm1(-s * calibration_amplitude * calibration_mode.reshape(sysi.Nact, sysi.Nact)) # remove the mode
                
                # DM2: Set the DM to the correct state
                sysi.add_dm2(s * calibration_amplitude * calibration_mode.reshape(sysi.Nact, sysi.Nact))
                differential_images_2, single_images_2 = take_measurement(sysi, probe_modes, probe_amplitude, DM=1,
                                                                          return_all=True)
                
                images_2.append(single_images_2)
                slope2 += s * differential_images_2 / (2 * calibration_amplitude)
                
                sysi.add_dm2(-s * calibration_amplitude * calibration_mode.reshape(sysi.Nact, sysi.Nact)) 
                
            print("\tCalibrated mode {:d} / {:d} in {:.3f}s".format(ci+1+start_mode, calibration_modes.shape[0], 
                                                                    time.time()-start))
            slopes_1.append(slope1)
            slopes_2.append(slope2)
        except KeyboardInterrupt: 
            print('Calibration interrupted.')
            break
    
    slopes_1 = np.array(slopes_1)
    slopes_2 = np.array(slopes_2)
    images_1 = np.array(images_1)
    images_2 = np.array(images_2)

    slopes = np.concatenate((slopes_1,slopes_2), axis=0) # this is the response cube
    images = np.concatenate((images_1,images_2), axis=0) # this is the calibration cube
    
    print('Calibration complete.')
    return slopes, images

def construct_control_matrix(response_matrix, weight_map, nprobes=2, rcond1=1e-2, rcond2=1e-2, WLS=True, pca_modes=None):
    weight_mask = weight_map>0
    
    # Invert the matrix with an SVD and Tikhonov regularization
    masked_matrix = response_matrix[:, :, weight_mask].reshape((response_matrix.shape[0], -1)).T
    
    # Add the extra PCA modes that are fitted
    if pca_modes is not None:
        double_pca_modes = np.concatenate( (pca_modes[:, weight_mask], pca_modes[:, weight_mask]), axis=1).T
        masked_matrix = np.hstack((masked_matrix, double_pca_modes))
    
    nmodes = int(response_matrix.shape[0]/2)
    if WLS:
        print('Using Weighted Least Squares ')
        if nprobes==2:
            Wmatrix = np.diag(np.concatenate((weight_map[weight_mask], weight_map[weight_mask])))
        elif nprobes==3:
            Wmatrix = np.diag(np.concatenate((weight_map[weight_mask], weight_map[weight_mask], weight_map[weight_mask])))
        control_matrix_1 = utils.WeightedLeastSquares(masked_matrix[:,:nmodes], Wmatrix, rcond=rcond1)
        control_matrix_2 = utils.WeightedLeastSquares(masked_matrix[:,nmodes:], Wmatrix, rcond=rcond2)
    else: 
        print('Using Tikhonov Inverse')
        control_matrix_1 = utils.TikhonovInverse(masked_matrix[:,:nmodes], rcond=rcond1)
        control_matrix_2 = utils.TikhonovInverse(masked_matrix[:,nmodes:], rcond=rcond2)
    control_matrix = np.concatenate((control_matrix_1, control_matrix_2), axis=0)
    
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

def run(sysi, 
        reg_fun, reg_conds, response_matrix, 
#         control_matrx,
        probe_modes, probe_amplitude, 
        calibration_modes, weights,
        num_iterations=10, 
        loop_gain=0.5, 
        leakage=0.0,
        display_current=True,
        display_all=False):
    
    print('Running I-EFC...')
    start = time.time()
    
    nmodes = calibration_modes.shape[0]
    
    # The metric
    metric_images = []
    dm1_commands = []
    dm2_commands = []
    
    dm1_ref = sysi.get_dm1()
    dm2_ref = sysi.get_dm2()
    commands = 0.0
    for i in range(num_iterations):
        print("\tClosed-loop iteration {:d} / {:d}".format(i+1, num_iterations))
        if i==0 or i in reg_conds[0]:
            reg_cond_ind = np.argwhere(i==np.array(reg_conds[0]))[0][0]
            reg_cond = reg_conds[1][reg_cond_ind]
            print('\tComputing EFC matrix via ' + reg_fun.__name__ + ' with condition values ' + str(reg_cond))
        
            control_matrix = reg_fun(response_matrix, 
                                     weights.flatten(), 
                                     rcond1=reg_cond[0],
                                     rcond2=reg_cond[1],
                                     nprobes=probe_modes.shape[0], pca_modes=None)
            
        delta_coefficients = single_iteration(sysi, probe_modes, probe_amplitude, control_matrix, weights.flatten()>0)
        commands = (1.0-leakage) * commands - loop_gain * delta_coefficients
        
        # Reconstruct the full phase from the Fourier modes
        dm1_command = calibration_modes.T.dot(commands[:nmodes]).reshape(sysi.Nact,sysi.Nact)
        dm2_command = calibration_modes.T.dot(commands[nmodes:]).reshape(sysi.Nact,sysi.Nact)
        
        # Set the current DM state
        sysi.set_dm1(dm1_ref + dm1_command)
        sysi.set_dm2(dm2_ref + dm2_command)
        
        # Take an image to estimate the metrics
        image = sysi.snap()
        metric_images.append(image)
        dm1_commands.append(sysi.get_dm1())
        dm2_commands.append(sysi.get_dm2())
        
        if display_current: 
            misc.myimshow3(dm1_commands[i], dm2_commands[i], image, 
                           'DM1', 'DM2', 'Image: Iteration {:d}'.format(i+1),
                           lognorm3=True, pxscl3=sysi.psf_pixelscale.to(u.mm/u.pix))
    print('I-EFC loop completed in {:.3f}s.'.format(time.time()-start))
    return metric_images, dm1_commands, dm2_commands





