import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from pathlib import Path
from importlib import reload
from IPython.display import clear_output, display
import time
import copy
from importlib import reload

from . import utils
reload(utils)

from cgi_phasec_poppy import misc

def build_jacobian(sysi, epsilon, dark_mask, display=False):
    start = time.time()
    print('Building Jacobian.')
    
    dm_mask = sysi.dm_mask.flatten()
    if hasattr(sysi, 'bad_acts'):
        dm_mask[sysi.bad_acts] = False
    
    responses_1 = []
    responses_2 = []
    amps = np.linspace(-epsilon, epsilon, 2) # for generating a negative and positive actuator poke
    
    dm_mask = sysi.dm_mask.flatten()
    
    num_modes = sysi.Nact**2
    modes = np.eye(num_modes) # each column in this matrix represents a vectorized DM shape where one actuator has been poked
    
    count = 1
    for i, mode in enumerate(modes):
        if dm_mask[i]:
            response1 = 0
            response2 = 0
            for amp in amps:
                mode = mode.reshape(sysi.Nact,sysi.Nact)
                
                sysi.add_dm1(amp*mode)
                wavefront = sysi.calc_psf()
                response1 += amp*wavefront/np.var(amps)
                sysi.add_dm1(-amp*mode)
                
                sysi.add_dm2(amp*mode)
                wavefront = sysi.calc_psf()
                response2 += amp*wavefront/np.var(amps)
                sysi.add_dm2(-amp*mode)

            if display:
                misc.myimshow2(cp.abs(response), cp.angle(response))
            
            response1 = response1.flatten()[dark_mask]
            response2 = response2.flatten()[dark_mask]
            
            responses_1.append(np.concatenate((response1.real, response1.imag)))
            responses_2.append(np.concatenate((response2.real, response2.imag)))
            
            print('\tCalculated response for mode {:d}/{:d}. Elapsed time={:.3f} sec.'.format(count,round(dm_mask.sum()),
                                                                                              time.time()-start))
            count += 1
        else:
            pass
        
    responses_1 = np.array(responses_1)
    responses_2 = np.array(responses_2)
    print(responses_1.shape, responses_2.shape)
    
    responses = np.concatenate( ( responses_1, responses_2 ), axis=0 )
    print(responses.shape)
    jacobian = responses.T
    print('Jacobian built in {:.3f} sec'.format(time.time()-start))
    
    return jacobian

def run_efc_perfect(sysi, 
                    jac, 
                    reg_fun,
                    reg_conds,
                    dark_mask, 
                    Imax_unocc,
                    efc_loop_gain=0.5, 
                    iterations=5, 
                    display_all=False, 
                    display_current=True,
                    plot_sms=True):
    # This function is only for running EFC simulations
    print('Beginning closed-loop EFC simulation.')
    dm1_commands = []
    dm2_commands = []
    efields = []
    
    start = time.time()
    
    U, s, V = np.linalg.svd(jac, full_matrices=False)
    alpha2 = np.max( np.diag( np.real( jac.conj().T @ jac ) ) )
    print('Max singular value squared:\t', s.max()**2)
    print('alpha^2:\t\t\t', alpha2) 
    
    dm_mask = sysi.dm_mask.flatten()
    if hasattr(sysi, 'bad_acts'):
        dm_mask[sysi.bad_acts] = False
    
    N_DH = dark_mask.sum()
    
    dm1_ref = sysi.get_dm1()
    dm2_ref = sysi.get_dm2()
    
    dm1_command = 0.0
    dm2_command = 0.0
    print()
    for i in range(iterations+1):
        try:
            print('\tRunning iteration {:d}/{:d}.'.format(i, iterations))

            if i==0 or i in reg_conds[0]:
                reg_cond_ind = np.argwhere(i==reg_conds[0])[0][0]
                reg_cond = reg_conds[1, reg_cond_ind]
                print('\tComputing EFC matrix via ' + reg_fun.__name__ + ' with condition value {:.2e}'.format(reg_cond))
                efc_matrix = reg_fun(jac, reg_cond)

            sysi.set_dm1(dm1_ref + dm1_command) 
            sysi.set_dm2(dm2_ref + dm2_command) 

            electric_field = sysi.calc_psf()

            dm1_commands.append(sysi.get_dm1())
            dm2_commands.append(sysi.get_dm2())
            efields.append(copy.copy(electric_field))

            efield_ri = np.concatenate( (electric_field[dark_mask].real, electric_field[dark_mask].imag) )
            del_dms = -efc_matrix.dot(efield_ri)
            print(del_dms.shape)
            
            del_dm1 = utils.map_acts_to_dm(del_dms[:del_dms.shape[0]//2], dm_mask)
            del_dm2 = utils.map_acts_to_dm(del_dms[del_dms.shape[0]//2:], dm_mask)
            
            dm1_command += efc_loop_gain * del_dm1
            dm2_command += efc_loop_gain * del_dm2

            if display_current or display_all:
                if not display_all: clear_output(wait=True)

                fig,ax = misc.myimshow3(dm1_commands[i], dm2_commands[i], np.abs(electric_field)**2, 
                                        'DM1', 'DM2', 'Image: Iteration {:d}'.format(i),
                                        cmap1='viridis', cmap2='viridis',
                                        lognorm3=True, vmin3=(np.abs(electric_field)**2).max()/1e7,
                                        pxscl3=sysi.psf_pixelscale_lamD,
                                        return_fig=True, display_fig=True)
                if plot_sms:
                    sms_fig = utils.sms(U, s, alpha2, efield_ri, N_DH, Imax_unocc, i)
        except KeyboardInterrupt:
            print('EFC interrupted.')
            break
            
            
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return dm1_commands, dm2_commands, efields

def run_efc_pwp(sysi, efc_matrix, jac, probes, dark_mask, efc_loop_gain=0.5, iterations=5, display=False):
    print('Beginning closed-loop EFC simulation.')
    
    commands = []
    efields = []
    images = []
    
    start=time.time()
    
    dm_ref = sysi.get_dm1()
    dm_command = np.zeros((sysi.Nact, sysi.Nact)) 
    for i in range(iterations):
        print('\tRunning iteration {:d}/{:d}.'.format(i+1, iterations))
        
        sysi.set_dm1(dm_ref + dm_command)
        E_est = run_pwp(sysi, probes, jac, dark_mask)
        I_exact = sysi.snap()
        
        commands.append(sysi.get_dm1())
        efields.append(copy.copy(E_est))
        images.append(copy.copy(I_exact))
        
        if display:
            misc.myimshow2(dm1_commands[i], dm2_commands[i], 'DM1', 'DM2')
            misc.myimshow2(E_est[i], I_exact[i], 'DM1', 'DM2')
        
        x = np.concatenate( (E_est[dark_mask].imag, E_est[dark_mask].real) )
        y = efc_matrix.dot(x)
        
        dm_command -= efc_loop_gain * y.reshape(sysi.Nact,sysi.Nact)
        
        
    print('EFC completed in {:.3f} sec.'.format(time.time()-start))
    
    return commands, efields, images

