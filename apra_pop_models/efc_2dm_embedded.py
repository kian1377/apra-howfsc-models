from .math_module import xp, xcipy, ensure_np_array
from . import imshows
from . import utils

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

from pathlib import Path

def make_new_wfe(time, beta = 1000e-12, size=(1000, 1000)):
    #beta is in units of meters per sqrt(second)
    #time is in seconds
    #OPD map is in meters
    return xp.random.normal(loc=0.0, scale=beta*(time**0.5), size=size)

def wait(time):
    return time

def do_fun(fun, I, efc_data, *args,):
    result = fun(*args)
    if type(result) is not tuple: result = (result,)

    time_taken = result[-1]
    action = fun.__name__
    current_time = efc_data['time_stamps'][-1]
    new_time = current_time + time_taken

    new_wfe = make_new_wfe(time_taken)
    I.WFE.opd += new_wfe
    image = I.snap()
    contrast = xp.mean(image[efc_data['control_mask']])

    imshows.imshow3(new_wfe, I.WFE.opd, image, title3=f'{contrast:.3e}', lognorm3=1)

    efc_data['images'].append(image)
    efc_data['contrasts'].append(contrast)
    efc_data['time_stamps'].append(new_time)
    efc_data['actions'].append(action)

    return result[0]

def run_with_decomp(
        I, 
        EC, 
        efc_data,
        est_fun=None, 
        est_params=None, 
        gain=0.5, 
        iterations=3, 
        use_update_wfe=False,
        plot_all=False, 
        plot=True,
    ):
    
    print('Beginning closed-loop EFC simulation.')    

    control_mask = efc_data['control_mask']

    Nmask = int(control_mask.sum())
    
    starting_itr = len(efc_data['dm1_commands'])
    if len(efc_data['dm1_commands'])>0:
        total_dm1 = copy.copy(efc_data['dm1_commands'][-1])
        total_dm2 = copy.copy(efc_data['dm2_commands'][-1])
    else:
        total_dm1, total_dm2 = ( xp.zeros((I.Nact,I.Nact)), xp.zeros((I.Nact,I.Nact)) ) 

    del_dm1 = xp.zeros((I.Nact, I.Nact))
    del_dm2 = xp.zeros((I.Nact, I.Nact))
    Nacts = int(2 * I.Nacts)
    Nmask = int(control_mask.sum())
    E_ab_vec = xp.zeros(2*Nmask)
    for i in range(iterations):
        print(f'\tRunning iteration {i+1+starting_itr}/{iterations+starting_itr}.')
        
        # before_efc_im = I.snap()
        # before_efc_contrast = xp.mean(before_efc_im[control_mask])

        if est_fun is None:
            E_ab = I.calc_wf() # no PWP, just use model
        else:
            E_ab = est_fun(I, **est_params)

        E_ab_vec[::2] = E_ab[control_mask].real
        E_ab_vec[1::2] = E_ab[control_mask].imag

        # del_acts, time_taken = EC.do_EFC_with_decomp( ensure_np_array(E_ab_vec) )
        # del_acts *=  -gain

        del_acts = do_fun(EC.do_EFC_with_decomp, I, efc_data, ensure_np_array(E_ab_vec))
        del_acts *=  -gain

        # if use_update_wfe: 
        #     new_wfe = make_new_wfe(time_taken)
        #     I.WFE.opd += new_wfe

        del_dm1[I.dm_mask] = del_acts[:I.Nacts]
        del_dm2[I.dm_mask] = del_acts[I.Nacts:]
        total_dm1 += del_dm1
        total_dm2 += del_dm2

        I.add_dm1(del_dm1)
        I.add_dm2(del_dm2)

        # after_efc_im = I.snap()
        # after_efc_contrast = xp.mean(after_efc_im[control_mask])

        do_fun(wait, I, efc_data, 5)

        # efc_data['before_images'].append(copy.copy(before_efc_im))
        # efc_data['before_contrasts'].append(copy.copy(before_efc_contrast))
        # efc_data['after_efc_images'].append(copy.copy(after_efc_im))
        # efc_data['after_contrasts'].append(copy.copy(after_efc_contrast))
        # efc_data['images'].append(copy.copy(after_efc_im))
        # efc_data['contrasts'].append(copy.copy(after_efc_contrast))
        # efc_data['time_stamps'].append(efc_data['time_stamps'][-1] )
        efc_data['efields'].append(copy.copy(E_ab))
        efc_data['dm1_commands'].append(copy.copy(total_dm1))
        efc_data['del_dm1_commands'].append(copy.copy(del_dm1))
        efc_data['dm2_commands'].append(copy.copy(total_dm2))
        efc_data['del_dm2_commands'].append(copy.copy(del_dm2))

        if plot or plot_all:
            imshows.imshow3(
                del_dm1, del_dm2, efc_data['images'][-1], 
                f'$\delta$DM1', f'$\delta$DM2', 
                f'Iteration {starting_itr + i:d} Image\nContrast = {efc_data["contrasts"][-1]:.3e}',
                cmap1='viridis', cmap2='viridis', 
                pxscl3=I.psf_pixelscale_lamDc, lognorm3=True, vmin3=1e-10,
            )

            if not plot_all: clear_output(wait=True)

    return efc_data


