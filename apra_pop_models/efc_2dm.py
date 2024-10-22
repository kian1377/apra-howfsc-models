from .math_module import xp, _scipy, ensure_np_array
from . import imshows
from . import utils

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

from pathlib import Path
# iefc_data_dir = Path('/groups/douglase/kians-data-files/roman-cgi-iefc-data')
iefc_data_dir = Path('/home/kianmilani/Projects/roman-cgi-iefc-data')

def compute_jacobian(M, 
                   calib_amp, calib_modes, 
                   control_mask, 
                   plot_responses=False,
                  ):
    start = time.time()
    
    Nmodes = calib_modes.shape[0]
    Nmask = int(control_mask.sum())

    response_matrix = xp.zeros((2*Nmask, Nmodes), dtype=xp.float64)
    print('Calculating Jacobian: ')
    for i, calibration_mode in enumerate(calib_modes):
        # reshape calibration mode into the DM1 and DM2 components
        dm1_mode = calibration_mode[:M.Nact**2].reshape(M.Nact, M.Nact)
        dm2_mode = calibration_mode[M.Nact**2:].reshape(M.Nact, M.Nact)
        
        # Add the mode to the DMs
        M.add_dm1(calib_amp * dm1_mode)
        M.add_dm2(calib_amp * dm2_mode)
        E_pos = M.calc_wf()
        M.add_dm1(-2 * calib_amp * dm1_mode) # remove the mode
        M.add_dm2(-2 * calib_amp * dm2_mode)
        E_neg = M.calc_wf()
        M.add_dm1(calib_amp * dm1_mode)
        M.add_dm2(calib_amp * dm2_mode)

        response = ( E_pos - E_neg ) / (2*calib_amp)

        response_matrix[::2,i] = response[control_mask].real
        response_matrix[1::2,i] = response[control_mask].imag

        print('\tCalculated response for mode {:d}/{:d}. Elapsed time={:.3f} sec.'.format(ci+1, Nmodes, time.time()-start), end='')
        print("\r", end="")
    
    print()
    print('Jacobian built in {:.3f} sec'.format(time.time()-start))
    
    if plot_responses:
        responses = response_matrix[::2] + 1j*response_matrix[1::2]
        dm_rms = xp.sqrt( xp.mean( xp.square(xp.abs(responses.dot(xp.array(calib_modes)))), axis=0) ) 
        dm1_rms = dm_rms[:M.Nact**2].reshape(M.Nact, M.Nact)
        dm2_rms = dm_rms[M.Nact**2:].reshape(M.Nact, M.Nact)
        imshows.imshow2(dm1_rms, dm2_rms, 
                        'DM1 RMS Actuator Responses', 'DM2 RMS Actuator Responses')

    return response_matrix

def run(I, 
        response_matrix,
        reg_fun, reg_cond,
        control_mask, 
        est_fun=None, 
        est_params=None, 
        gain=0.5, 
        iterations=3, 
        plot_all=False, 
        plot=True,
        all_ims=[], 
        all_efs=[],
        all_commands=[],
        ):
    
    print('Beginning closed-loop EFC simulation.')    

    Nmask = int(control_mask.sum())
    control_matrix = reg_fun(response_matrix, reg_cond)
    
    starting_itr = len(all_ims)
    if len(all_commands)>0:
        total_dm1 = copy.copy(all_commands[-1][0])
        total_dm2 = copy.copy(all_commands[-1][1])
    else:
        total_dm1 = xp.zeros((I.Nact,I.Nact))
        total_dm2 = xp.zeros((I.Nact,I.Nact))

    del_dm1 = xp.zeros((I.Nact,I.Nact))
    del_dm2 = xp.zeros((I.Nact,I.Nact))
    for i in range(iterations):
        print(f'\tRunning iteration {i+1+starting_itr}/{iterations+starting_itr}.')
        
        if est_fun is None:
            E_ab = I.calc_wf() # no PWP, just use model
        else:
            E_ab = est_fun(I, **est_params)
        # efields.append([copy.copy(electric_field)])
        efield_ri = xp.zeros(2*Nmask)
        efield_ri[::2] = E_ab[control_mask].real
        efield_ri[1::2] = E_ab[control_mask].imag

        del_acts = gain * control_matrix.dot(efield_ri)
        del_dm1[I.dm_mask] = del_acts[:I.Nacts]
        del_dm2[I.dm_mask] = del_acts[I.Nacts:]
        total_dm1 += del_dm1
        total_dm2 += del_dm2

        I.add_dm1(del_dm1)
        I.add_dm2(del_dm2)
        
        image_ni = I.snap()

        all_ims.append(copy.copy(image_ni))
        all_efs.append(copy.copy(E_ab))
        all_commands.append(xp.array([total_dm1, total_dm2]))
        
        mean_ni = xp.mean(image_ni[control_mask])
        print(f'\tMean NI of this iteration: {mean_ni:.3e}')

        if plot or plot_all:

            imshows.imshow3(all_commands[-1][0], all_commands[-1][1], all_ims[-1], 
                            'DM1', 'DM2', f'Image: Iteration {i+starting_itr+1}\nMean NI: {mean_ni:.3e}',
                            cmap1='viridis', cmap2='viridis',
                            lognorm3=True, vmin3=1e-10, pxscl3=I.psf_pixelscale_lamD, xlabel3='$\lambda/D$')
            if not plot_all: clear_output(wait=True)

    return all_ims, all_efs, all_commands



