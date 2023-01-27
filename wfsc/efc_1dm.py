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
from importlib import reload

from . import pwp_1dm as pwp
from . import utils
reload(pwp)
reload(utils)

from cgi_phasec_poppy import misc

def build_jacobian(sysi, epsilon, dark_mask, display=False):
    start = time.time()
    print('Building Jacobian.')
    
    responses = []
    amps = np.linspace(-epsilon, epsilon, 2) # for generating a negative and positive actuator poke
    
    dm_mask = sysi.dm_mask.flatten()
    if hasattr(sysi, 'bad_acts'):
        dm_mask[sysi.bad_acts] = False
    
    num_modes = sysi.Nact**2
    modes = np.eye(num_modes) # each column in this matrix represents a vectorized DM shape where one actuator has been poked
    
    count = 1
    for i in range(num_modes):
        if dm_mask[i]:
            response = 0
            for amp in amps:
                mode = modes[i].reshape(sysi.Nact,sysi.Nact)

                sysi.add_dm1(amp*mode)
                wavefront = sysi.calc_psf()
                response += amp*wavefront/np.var(amps)
                sysi.add_dm1(-amp*mode)

            if display:
                misc.myimshow2(np.abs(response), np.angle(response))
                
            response = response.flatten()[dark_mask.flatten()]

            responses.append(np.concatenate((response.real, response.imag)))
        
            print('\tCalculated response for mode {:d}/{:d}. Elapsed time={:.3f} sec.'.format(count,round(dm_mask.sum()),
                                                                                              time.time()-start))
            count += 1
        else:
            pass
    jacobian = np.array(responses).T
    print('Jacobian built in {:.3f} sec'.format(time.time()-start))
    
    return jacobian

def run_efc_perfect(sysi, 
                    jac, 
                    reg_fun,
                    reg_conds,
                    dark_mask, 
                    Imax_unocc=1,
                    efc_loop_gain=0.5, 
                    iterations=5, 
                    display_all=False, 
                    display_current=True,
                    plot_sms=True):
    # This function is only for running EFC simulations
    print('Beginning closed-loop EFC simulation.')    
    commands = []
    efields = []
    
    start = time.time()
    
    U, s, V = np.linalg.svd(jac, full_matrices=False)
    alpha2 = np.max( np.diag( np.real( jac.conj().T @ jac ) ) )
    print('Max singular value squared:\t', s.max()**2)
    print('alpha^2:\t\t\t', alpha2) 
    
    N_DH = dark_mask.sum()
    
    dm_mask = sysi.dm_mask.flatten()
    if hasattr(sysi, 'bad_acts'):
        dm_mask[sysi.bad_acts] = False
    
    dm_ref = sysi.get_dm1()
    dm_command = np.zeros((sysi.Nact, sysi.Nact)) 
    print()
    for i in range(iterations+1):
        try:
            print('\tRunning iteration {:d}/{:d}.'.format(i+1, iterations))

            if i==0 or i in reg_conds[0]:
                reg_cond_ind = np.argwhere(i==reg_conds[0])[0][0]
                reg_cond = reg_conds[1, reg_cond_ind]
                print('\tComputing EFC matrix via ' + reg_fun.__name__ + ' with condition value {:.2e}'.format(reg_cond))
                efc_matrix = reg_fun(jac, reg_cond)

            sysi.set_dm1(dm_ref + dm_command)

            electric_field = sysi.calc_psf()

            commands.append(sysi.get_dm1())
            efields.append(copy.copy(electric_field))

            efield_ri = np.concatenate( (electric_field[dark_mask].real, electric_field[dark_mask].imag) )
            del_dm = -efc_matrix.dot(efield_ri)

            del_dm = utils.map_acts_to_dm(del_dm, dm_mask)
            dm_command += efc_loop_gain * del_dm
            
            if display_current or display_all:
                if not display_all: clear_output(wait=True)
                    
                fig,ax = misc.myimshow2(commands[i], np.abs(electric_field)**2, 
                                        'DM', 'Image: Iteration {:d}'.format(i),
                                        cmap1='viridis',
                                        lognorm2=True, vmin2=(np.abs(electric_field)**2).max()/1e7,
                                        pxscl2=sysi.psf_pixelscale_lamD,
                                        return_fig=True, display_fig=True)
                if plot_sms:
                    sms_fig = utils.sms(U, s, alpha2, efield_ri, N_DH, Imax_unocc, i)
        except KeyboardInterrupt:
            print('EFC interrupted.')
            break
        
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return commands, efields

def run_efc_pwp(sysi, 
                pwp_fun,
                pwp_kwargs,
                jac, 
                reg_fun,
                reg_conds,
                dark_mask, 
                Imax_unocc=1,
                efc_loop_gain=0.5, 
                iterations=5, 
                display_all=False, 
                display_current=True,
                plot_sms=True):
    print('Beginning closed-loop EFC simulation.')
    
    commands = []
    efields = []
    images = []
    
    start=time.time()
    
    U, s, V = np.linalg.svd(jac, full_matrices=False)
    alpha2 = np.max( np.diag( np.real( jac.conj().T @ jac ) ) )
    print('Max singular value squared:\t', s.max()**2)
    print('alpha^2:\t\t\t', alpha2) 
    
    N_DH = dark_mask.sum()
    
    dm_mask = sysi.dm_mask.flatten()
    if hasattr(sysi, 'bad_acts'):
        dm_mask[sysi.bad_acts] = False
    
    dm_ref = sysi.get_dm1()
    dm_command = np.zeros((sysi.Nact, sysi.Nact)) 
    for i in range(iterations+1):
        try:
            print('\tRunning iteration {:d}/{:d}.'.format(i, iterations))
            
            if i==0 or i in reg_conds[0]:
                reg_cond_ind = np.argwhere(i==reg_conds[0])[0][0]
                reg_cond = reg_conds[1, reg_cond_ind]
                print('\tComputing EFC matrix via ' + reg_fun.__name__ + ' with condition value {:.2e}'.format(reg_cond))
                efc_matrix = reg_fun(jac, reg_cond)
                
            sysi.set_dm1(dm_ref + dm_command)
            E_est = pwp_fun(sysi, dark_mask, **pwp_kwargs)
            I_exact = sysi.snap()
            
            commands.append(sysi.get_dm1())
            efields.append(copy.copy(E_est))
            images.append(copy.copy(I_exact))

            efield_ri = np.concatenate( (E_est[dark_mask].real, E_est[dark_mask].imag) )
            del_dm = -efc_matrix.dot(efield_ri)
            
            del_dm = utils.map_acts_to_dm(del_dm, dm_mask)
            dm_command += efc_loop_gain * del_dm
            
            if display_current or display_all:
                if not display_all: clear_output(wait=True)

                fig,ax = misc.myimshow3(commands[i], np.abs(E_est)**2, I_exact, 
                                        'DM', 'Estimated Intensity', 'Image: Iteration {:d}'.format(i),
                                        cmap1='viridis',
                                        lognorm2=True, vmin2=I_exact.max()/1e6, vmax2=I_exact.max(),
                                        lognorm3=True, vmin3=I_exact.max()/1e6, vmax3=I_exact.max(),
                                        pxscl2=sysi.psf_pixelscale_lamD, pxscl3=sysi.psf_pixelscale_lamD,
                                        return_fig=True, display_fig=True)
                if plot_sms:
                    sms_fig = utils.sms(U, s, alpha2, efield_ri, N_DH, Imax_unocc, i)
        except KeyboardInterrupt:
            print('EFC interrupted.')
            break
        
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return commands, efields, images


