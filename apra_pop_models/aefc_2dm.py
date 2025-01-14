from .math_module import xp, xcipy, ensure_np_array
from adefc_vortex import utils
from adefc_vortex.imshows import imshow1, imshow2, imshow3
import adefc_vortex.pwp as pwp

import numpy as np
from scipy.optimize import minimize
import time
import copy

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
        
        if pwp_params is not None: 
            print('Running PWP ...')
            E_ab = pwp.run(I, M, **pwp_params)
        else:
            print('Computing E-field with model ...')
            E_ab = I.calc_wf()
        
        print('Computing EFC command with L-BFGS')
        current_acts = xp.concatenate([total_dm1[M.dm_mask], total_dm2[M.dm_mask]])
        E_FP_NOM, E_EP, E_DM2P, DM1_PHASOR, DM2_PHASOR = M.forward(current_acts, I.wavelength, use_vortex=True, return_ints=True,)

        rmad_vars = { 
            'current_acts':current_acts,
            'E_ab':E_ab, 
            'E_FP_NOM':E_FP_NOM,
            'E_EP':E_EP,
            'E_DM2P':E_DM2P,
            'DM1_PHASOR':DM1_PHASOR,
            'DM2_PHASOR':DM2_PHASOR,
            'control_mask':control_mask,
            'wavelength':I.wavelength,
            'r_cond':reg_cond,
        }

        res = minimize(
            val_and_grad, 
            jac=True, 
            x0=del_acts0,
            args=(M, rmad_vars), 
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
                f'Iteration {starting_itr + i:d} Image\nMean NI = {mean_ni:.3e}',
                cmap1='viridis', cmap2='viridis', 
                pxscl3=I.psf_pixelscale_lamDc, lognorm3=True, vmin3=1e-10)

    return data

def calc_wfs(I, waves, control_mask, plot=False):
    Nwaves = len(waves)
    E_abs = xp.zeros((Nwaves, I.npsf, I.npsf), dtype=xp.complex128)
    for i in range(Nwaves):
        I.wavelength = waves[i]
        E_abs[i] = I.calc_wf() * control_mask
        if plot: imshow2(xp.abs(E_abs[i])**2, xp.angle(E_abs[i])*control_mask, lognorm1=True, cmap2='twilight')

    return E_abs

def get_forward_vars(M, current_acts, est_waves):
    Nwaves = est_waves.shape[0]
    E_FP_NOMs = []
    E_EPs = []
    E_DM2Ps = []
    DM1_PHASORs = []
    DM2_PHASORs = []
    for i in range(Nwaves):
        E_FP_NOM, E_EP, E_DM2P, DM1_PHASOR, DM2_PHASOR = M.forward(current_acts, est_waves[i], use_vortex=True, return_ints=True,)
        E_FP_NOMs.append(E_FP_NOM)
        E_EPs.append(E_EP)
        E_DM2Ps.append(E_DM2P)
        DM1_PHASORs.append(DM1_PHASOR)
        DM2_PHASORs.append(DM2_PHASOR)
    return xp.array(E_FP_NOMs), xp.array(E_EPs), xp.array(E_DM2Ps), xp.array(DM1_PHASORs), xp.array(DM2_PHASORs)

def run_bb(
        I, 
        M, 
        val_and_grad,
        control_mask,
        data,
        pwp_params=None,
        Nitr=3, 
        reg_cond=1e-2,
        weights=None, 
        bfgs_tol=1e-3,
        bfgs_opts=None,
        gain=0.5, 
    ):
    
    Nbps = I.bandpasses.shape[0]
    Nwaves_per_bp = I.bandpasses.shape[1]
    control_waves = I.bandpasses[:, Nwaves_per_bp//2]

    starting_itr = len(data['images'])
    if len(data['dm1_commands'])>0:
        total_dm1, total_dm2 = ( copy.copy(data['dm1_commands'][-1]), copy.copy(data['dm2_commands'][-1]) )
    else:
        total_dm1, total_dm2 = ( xp.zeros((M.Nact,M.Nact)), xp.zeros((M.Nact,M.Nact)) ) 

    del_dm1 = xp.zeros((M.Nact,M.Nact))
    del_dm2 = xp.zeros((M.Nact,M.Nact))
    del_acts0 = np.zeros(M.Nacts)
    for i in range(Nitr):
        if pwp_params is not None: 
            print('Running PWP ...')
            E_abs = pwp.run_bb(I, M, **pwp_params)
        else:
            print('Computing E-field with model ...')
            E_abs = calc_wfs(I, control_waves, control_mask)
        
        print('Computing EFC command with L-BFGS')
        current_acts = xp.concatenate([total_dm1[M.dm_mask], total_dm2[M.dm_mask]])
        E_FP_NOMs, E_EPs, E_DM2Ps, DM1_PHASORs, DM2_PHASORs = get_forward_vars(M, current_acts, control_waves)
        rmad_vars = { 
            'current_acts':current_acts,
            'E_abs':E_abs, 
            'E_FP_NOMs':E_FP_NOMs,
            'E_EPs':E_EPs,
            'E_DM2Ps':E_DM2Ps,
            'DM1_PHASORs':DM1_PHASORs,
            'DM2_PHASORs':DM2_PHASORs,
            'control_mask':control_mask,
            'control_waves':control_waves,
            'r_cond':reg_cond,
            'weights':weights,
        }

        res = minimize(
            val_and_grad, 
            jac=True, 
            x0=del_acts0,
            args=(M, rmad_vars), 
            method='L-BFGS-B',
            tol=bfgs_tol,
            options=bfgs_opts,
        )

        del_acts = gain * res.x
        del_dm1[M.dm_mask] = del_acts[:M.Nacts//2]
        del_dm2[M.dm_mask] = del_acts[M.Nacts//2:]
        I.add_dm1(del_dm1)
        I.add_dm2(del_dm2)
        total_dm1 = total_dm1 + del_dm1
        total_dm2 = total_dm2 + del_dm2

        image_ni = I.snap()
        mean_ni = xp.mean(image_ni[control_mask])

        data['images'].append(copy.copy(image_ni))
        data['efields'].append(copy.copy(E_abs))
        data['dm1_commands'].append(copy.copy(total_dm1))
        data['del_dm1_commands'].append(copy.copy(del_dm1))
        data['dm2_commands'].append(copy.copy(total_dm2))
        data['del_dm2_commands'].append(copy.copy(del_dm2))
        data['bfgs_tols'].append(bfgs_tol)
        data['reg_conds'].append(reg_cond)
        
        imshow3(del_dm1, del_dm2, image_ni, 
                f'$\delta$DM1', f'$\delta$DM2', 
                f'Iteration {starting_itr + i:d} Image\nMean NI = {mean_ni:.3e}',
                cmap1='viridis', cmap2='viridis', 
                pxscl3=I.psf_pixelscale_lamDc, lognorm3=True, vmin3=1e-10)

    return data


