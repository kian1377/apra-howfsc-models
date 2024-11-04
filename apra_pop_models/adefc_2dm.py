from .math_module import xp, _scipy, ensure_np_array
from . import utils
from .imshows import imshow1, imshow2, imshow3

import numpy as np
import astropy.units as u
from scipy.optimize import minimize
import time
import copy

def calc_wfs(I, waves, control_mask, plot=False):
    Nwaves = len(waves)
    E_abs = xp.zeros((Nwaves, I.npsf, I.npsf), dtype=xp.complex128)
    for i in range(Nwaves):
        I.wavelength = waves[i]
        E_abs[i] = I.calc_wf() * control_mask
        if plot: imshow2(xp.abs(E_abs[i])**2, xp.angle(E_abs[i])*control_mask, lognorm1=True, cmap2='twilight')

    return E_abs

def pwp(I, 
        M, 
        current_acts, 
        control_mask, 
        probes, 
        probe_amp, 
        reg_cond=1e-3, 
        plot=False,
        plot_est=False,
        ):
    
    Nmask = int(control_mask.sum())
    Nprobes = probes.shape[0]

    Ip = []
    In = []
    for i in range(Nprobes):
        for s in [-1, 1]:
            I.add_dm(s*probe_amp*probes[i])
            coro_im = I.snap()
            I.add_dm(-s*probe_amp*probes[i]) # remove probe from DM

            if s==-1: 
                In.append(coro_im)
            else: 
                Ip.append(coro_im)
            
    E_probes = xp.zeros((probes.shape[0], 2*Nmask))
    I_diff = xp.zeros((probes.shape[0], Nmask))
    for i in range(Nprobes):
        if i==0: 
            E_nom = M.forward(current_acts, use_vortex=True, use_wfe=True)
        E_with_probe = M.forward(xp.array(current_acts) + xp.array(probe_amp*probes[i])[M.dm_mask], use_vortex=True, use_wfe=True)
        E_probe = E_with_probe - E_nom

        if plot:
            imshow3(xp.abs(E_probe), xp.angle(E_probe), Ip[i]-In[i], 
                    f'Probe {i+1}: '+'$|E_{probe}|$', f'Probe {i+1}: '+r'$\angle E_{probe}$', 'Difference Image', 
                    cmap2='twilight')
            
        E_probes[i, ::2] = E_probe[control_mask].real
        E_probes[i, 1::2] = E_probe[control_mask].imag

        # I_diff[i:(i+1), :] = (Ip[i] - In[i])[control_mask]
        I_diff[i, :] = (Ip[i] - In[i])[control_mask]
    
    # Use batch process to estimate each pixel individually
    E_est = xp.zeros(Nmask, dtype=xp.complex128)
    for i in range(Nmask):
        delI = I_diff[:, i]
        H = 4*xp.array([E_probes[:,2*i], E_probes[:,2*i + 1]]).T
        Hinv = xp.linalg.pinv(H.T@H, reg_cond)@H.T
    
        est = Hinv.dot(delI)

        E_est[i] = est[0] + 1j*est[1]
        
    E_est_2d = xp.zeros((I.npsf,I.npsf), dtype=xp.complex128)
    E_est_2d[control_mask] = E_est

    if plot or plot_est:
        I_est = xp.abs(E_est_2d)**2
        P_est = xp.angle(E_est_2d)
        imshow2(I_est, P_est, 
                'Estimated Intensity', 'Estimated Phase',
                lognorm1=True, vmin1=xp.max(I_est)/1e4, 
                cmap2='twilight',
                pxscl=M.psf_pixelscale_lamD)
    return E_est_2d

def pwp_bb():
    return

def run(I, 
        M, 
        val_and_grad,
        control_mask,
        data,
        pwp_params=None,
        Nitr=3, 
        reg_cond=1e-2,
        bfgs_tol=1e-3,
        bfgs_opts=None,
        gain=0.5, 
        ):
    
    starting_itr = len(data['images'])
    if len(data['dm1_commands'])>0:
        total_dm1 = copy.copy(data['dm1_commands'][-1])
        total_dm2 = copy.copy(data['dm2_commands'][-1])
    else:
        total_dm1, total_dm2 = ( xp.zeros((M.Nact,M.Nact)), xp.zeros((M.Nact,M.Nact)) ) 

    del_dm1 = xp.zeros((M.Nact,M.Nact))
    del_dm2 = xp.zeros((M.Nact,M.Nact))
    del_acts0 = np.zeros(M.Nacts)
    for i in range(Nitr):
        print('Running estimation algorithm ...')
        I.subtract_dark = False
        
        if pwp_params is not None: 
            E_ab = pwp(I, M, ensure_np_array(total_dm1[M.dm_mask]), ensure_np_array(total_dm2[M.dm_mask]), **pwp_params)
        else:
            E_ab = I.calc_wf()
        
        print('Computing EFC command with L-BFGS')
        total_acts = ensure_np_array(xp.concatenate([total_dm1[M.dm_mask], total_dm2[M.dm_mask]]))
        res = minimize(val_and_grad, 
                       jac=True, 
                       x0=del_acts0,
                       args=(M, total_acts, E_ab, control_mask, I.wavelength.to_value(u.m), reg_cond), 
                       method='L-BFGS-B',
                       tol=bfgs_tol,
                       options=bfgs_opts,
                       )

        del_acts = gain * res.x
        del_dm1[M.dm_mask] = del_acts[:M.Nacts//2]
        del_dm2[M.dm_mask] = del_acts[M.Nacts//2:]
        total_dm1 += del_dm1
        total_dm2 += del_dm2

        I.add_dm1(del_dm1)
        I.add_dm2(del_dm2)
        image_ni = I.snap()
        mean_ni = xp.mean(image_ni[control_mask])

        data['images'].append(copy.copy(image_ni))
        data['efields'].append(copy.copy(E_ab))
        data['dm1_commands'].append(copy.copy(total_dm1))
        data['del_dm1_commands'].append(copy.copy(del_dm1))
        data['dm2_commands'].append(copy.copy(total_dm2))
        data['del_dm2_commands'].append(copy.copy(del_dm2))
        data['bfgs_tols'].append(bfgs_tol)
        data['reg_conds'].append(reg_cond)
        
        imshow3(del_dm1, del_dm2, image_ni, 
                f'$\delta$DM1', f'$\delta$DM2', 
                f'Iteration {starting_itr + i + 1:d} Image\nMean NI = {mean_ni:.3e}',
                cmap1='viridis', cmap2='viridis', 
                pxscl3=I.psf_pixelscale_lamD, lognorm3=True, vmin3=1e-10)

    return data


