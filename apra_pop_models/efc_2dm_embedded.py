from .math_module import xp, _scipy, ensure_np_array
from . import imshows
from . import utils

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output

from pathlib import Path

def run(I, EC, 
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

        del_acts = gain * EC.do_EFC_with_decomp(ensure_np_array(efield_ri))
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



