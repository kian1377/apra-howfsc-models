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

from . import utils
reload(utils)

from cgi_phasec_poppy import misc

def create_sinc_probe(Nacts, amp, probe_radius, probe_phase=0, offset=(0,0), bad_axis='x'):
    print('Generating probe with amplitude={:.3e}, radius={:.1f}, phase={:.3f}, offset=({:.1f},{:.1f}), with discontinuity along '.format(amp, probe_radius, probe_phase, offset[0], offset[1]) + bad_axis + ' axis.')
    
    xacts = np.arange( -(Nacts-1)/2, (Nacts+1)/2 )/Nacts - np.round(offset[0])/Nacts
    yacts = np.arange( -(Nacts-1)/2, (Nacts+1)/2 )/Nacts - np.round(offset[1])/Nacts
    Xacts,Yacts = np.meshgrid(xacts,yacts)
    if bad_axis=='x': 
        fX = 2*probe_radius
        fY = probe_radius
        omegaY = probe_radius/2
        probe_commands = amp * np.sinc(fX*Xacts)*np.sinc(fY*Yacts) * np.cos(2*np.pi*omegaY*Yacts + probe_phase)
    elif bad_axis=='y': 
        fX = probe_radius
        fY = 2*probe_radius
        omegaX = probe_radius/2
        probe_commands = amp * np.sinc(fX*Xacts)*np.sinc(fY*Yacts) * np.cos(2*np.pi*omegaX*Xacts + probe_phase) 
    if probe_phase == 0:
        f = 2*probe_radius
        probe_commands = amp * np.sinc(f*Xacts)*np.sinc(f*Yacts)
    return probe_commands

def create_sinc_probes(Npairs, Nacts, dm_mask, probe_amplitude, probe_radius=10, probe_offset=(0,0), display=False):
    
    probe_phases = np.linspace(0, np.pi*(Npairs-1)/Npairs, Npairs)
    
    probes = []
    for i in range(Npairs):
        if i%2==0:
            axis = 'x'
        else:
            axis = 'y'
            
        probe = create_sinc_probe(Nacts, probe_amplitude, probe_radius, probe_phases[i], offset=probe_offset, bad_axis=axis)
            
        probes.append(probe*dm_mask)
    
    if display:
        if Npairs==2:
            misc.myimshow2(probes[0], probes[1])
        elif Npairs==3:
            misc.myimshow3(probes[0], probes[1], probes[2])
    
    return np.array(probes)


def run_pwp_bp(sysi, dark_mask, 
               probes,
               use, jacobian=None, model=None, 
               display=False, display_probe_field=False):
    nmask = dark_mask.sum()
    
    dm_mask = sysi.dm_mask.flatten()
    if hasattr(sysi, 'bad_acts'):
        dm_mask[sysi.bad_acts] = False
    act_inds = np.where(dm_mask.astype(int))[0]
    
    dm_ref = sysi.get_dm1()
    amps = np.linspace(-1, 1, 2) # for generating a negative and positive probe
    
    Ip = []
    In = []
    for i,probe in enumerate(probes):
        for amp in amps:
            sysi.add_dm1(amp*probe)
            psf = sysi.snap()
                
            if amp==-1: 
                In.append(psf)
            else: 
                Ip.append(psf)
                
            sysi.add_dm1(-amp*probe) # remove probe from DM
            
        if display:
            misc.myimshow3(Ip[i], In[i], Ip[i]-In[i],
                           'Probe {:d} Positive Image'.format(i+1), 'Probe {:d} Negative Image'.format(i+1),
                           'Intensity Difference',
                           lognorm1=True, lognorm2=True, vmin1=Ip[i].max()/1e6, vmin2=In[i].max()/1e6,
                          )
        
    E_probes = np.zeros((probes.shape[0], 2*nmask))
    I_diff = np.zeros((probes.shape[0], nmask))
    for i in range(len(probes)):
        if (use=='jacobian' or use=='j') and jacobian is not None:
#             E_probe = 1j*jacobian.dot(np.array(probes[i].flatten())) # Use jacobian to model probe E-field at the focal plane
            E_probe = 1j*jacobian.dot(np.array(probes[i].flatten()[act_inds])) 
            E_probe = E_probe[:nmask] + 1j*E_probe[nmask:]
        elif (use=='model' or use=='m') and model is not None:
            if i==0: E_full = model.calc_psf().wavefront.get()[dark_mask]
                
            model.add_dm1(probes[i])
            E_full_probe = model.calc_psf().wavefront.get()[dark_mask]
            model.add_dm1(-probes[i])
            
            E_probe = E_full_probe - E_full
            
        if display_probe_field:
            E_probe_2d = np.zeros((sysi.npsf,sysi.npsf), dtype=np.complex128)
            np.place(E_probe_2d, mask=dark_mask, vals=E_probe)
            misc.myimshow2(np.abs(E_probe_2d), np.angle(E_probe_2d), 'E_probe Amp', 'E_probe Phase')
        
        E_probes[i, :nmask] = E_probe.real
        E_probes[i, nmask:] = E_probe.imag

        I_diff[i:(i+1), :] = (Ip[i] - In[i])[dark_mask]
    
    # Use batch process to estimate each pixel individually
    E_est = np.zeros((nmask,), dtype=cp.complex128)
    for i in range(nmask):
        delI = I_diff[:, i]
        M = 4*np.array([E_probes[:,i], E_probes[:,i+nmask]]).T
        Minv = np.linalg.pinv(M.T@M, 1e-1)@M.T
    
        est = Minv.dot(delI)

        E_est[i] = est[0] + 1j*est[1]
        
    E_est_2d = np.zeros((sysi.npsf,sysi.npsf), dtype=np.complex128)
    np.place(E_est_2d, mask=dark_mask, vals=E_est)
    
    return E_est_2d

def run_pwp_redmond(sysi, dark_mask, 
                probes,
                use, jacobian=None, model=None, 
                rcond=1e-15,
                use_noise=False, display=False):
    nmask = dark_mask.sum()
    nprobes = probes.shape[0]
    
    dm_ref = sysi.get_dm1()
    amps = np.linspace(-1, 1, 2) # for generating a negative and positive probe
    
    Ip = []
    In = []
    for i,probe in enumerate(probes):
        for amp in amps:
            sysi.add_dm1(amp*probe)
            
            im = sysi.snap()
                
            if amp==-1: 
                In.append(im)
            else: 
                Ip.append(im)
                
            sysi.add_dm1(-amp*probe) # remove probe from DM
            
        if display:
            misc.myimshow3(Ip[i], In[i], Ip[i]-In[i],
                           'Probe {:d} Positive Image'.format(i+1), 'Probe {:d} Negative Image'.format(i+1),
                           'Intensity Difference',
                           lognorm1=True, lognorm2=True, vmin1=Ip[i].max()/1e6, vmin2=In[i].max()/1e6,
                          )

    E_probes = np.zeros((2*nmask*nprobes,))
    I_diff = np.zeros((nmask*nprobes,))
    for i in range(nprobes):
        I_diff[ i*nmask : (i+1)*nmask ] = (Ip[i] - In[i])[dark_mask]

        if (use=='jacobian' or use=='j') and jacobian is not None:
            E_probe = jacobian.dot(np.array(probes[i].flatten())) # Use jacobian to model probe E-field at the focal plane
        elif (use=='model' or use=='m') and model is not None:
            if i==0: E_full = model.calc_psf().wavefront.get()[dark_mask]
                
            model.add_dm1(probes[i])
            E_full_probe = model.calc_psf().wavefront.get()[dark_mask]
            model.add_dm1(-probes[i])
            
            E_probe = E_full_probe - E_full
            E_probe = np.concatenate((E_probe.real, E_probe.imag))
            
        E_probes[ i*2*nmask : (i+1)*2*nmask ] = E_probe
        
        E_probe_2d = np.zeros((sysi.npsf,sysi.npsf), dtype=np.complex128)
        np.place(E_probe_2d, mask=dark_mask, 
                 vals=E_probes[i*2*nmask : (i+1)*2*nmask ][:nmask] + 1j*E_probes[i*2*nmask : (i+1)*2*nmask ][nmask:])
        misc.myimshow2(np.abs(E_probe_2d), np.angle(E_probe_2d), 'E_probe Amp', 'E_probe Phase')
        
    B = np.diag(np.ones((nmask,2*nmask))[0], k=0)[:nmask,:2*nmask] + np.diag(np.ones((nmask,2*nmask))[0], k=nmask)[:nmask,:2*nmask]
    misc.myimshow(B, figsize=(10,4))
    print('B.shape', B.shape)
    
    for i in range(nprobes):
        h = 4 * B @ np.diag( E_probes[ i*2*nmask : (i+1)*2*nmask ] )
        Hinv = h if i==0 else np.vstack((Hinv,h))
    
    print('Hinv.shape', Hinv.shape)
    
    H = np.linalg.pinv(Hinv.T@Hinv, rcond)@Hinv.T
    print('H.shape', H.shape)
    
    E_est = H.dot(I_diff)
        
    E_est_2d = np.zeros((sysi.npsf,sysi.npsf), dtype=np.complex128)
    np.place(E_est_2d, mask=dark_mask, vals=E_est)
    
    return E_est_2d

def run_pwp_2011(sysi, dark_mask, 
                probes,
                use, jacobian=None, model=None, 
                rcond=1e-15,
                use_noise=False, display=False):
    nmask = dark_mask.sum()
    nprobes = probes.shape[0]
    
    I0 = sysi.snap()
    
    amps = np.linspace(-1, 1, 2) # for generating a negative and positive probe
    
    Ip = []
    In = []
    for i,probe in enumerate(probes):
        for amp in amps:
            sysi.add_dm1(amp*probe)
            
            im = sysi.snap()
                
            if amp==-1: 
                In.append(im)
            else: 
                Ip.append(im)
                
            sysi.add_dm1(-amp*probe) # remove probe from DM
            
        if display:
            misc.myimshow3(Ip[i], In[i], Ip[i]-In[i],
                           'Probe {:d} Positive Image'.format(i+1), 'Probe {:d} Negative Image'.format(i+1),
                           'Intensity Difference',
                           lognorm1=True, lognorm2=True, vmin1=Ip[i].max()/1e6, vmin2=In[i].max()/1e6,
                          )
    delI = np.zeros((nprobes*nmask,))
    delp = np.zeros((nprobes*nmask,), dtype=np.complex128) 
    for i in range(nprobes):
        delI[i*nmask:(i+1)*nmask] = (Ip[i]-In[i])[dark_mask]/2
        
        delp_amp = np.sqrt( (Ip[i]+In[i])[dark_mask]/2 - I0[dark_mask] )
        delp_amp[np.isnan(delp_amp)] = 0 # set the bad pixels to 0

        if (use=='jacobian' or use=='j') and jacobian is not None:
            del_p = jacobian.dot(np.array(probes[i].flatten())) 
            del_p = del_p[:nmask] + 1j*del_p[nmask:]
            
        delp_phs = np.angle(del_p)
        
        delp[i*nmask:(i+1)*nmask] = delp_amp * np.exp(1j*delp_phs)
        
    E_est = np.zeros((nmask,), dtype=cp.complex128)
    for i in range(nmask):
        
        M = 2*np.array([[-delp[i].imag, delp[i].real],
                        [-delp[i+nmask].imag, delp[i+nmask].real]])
        Minv = np.linalg.pinv(M.T@M, 1e-5)@M.T
    
        est = Minv.dot( [delI[i], delI[i+nmask]])

        E_est[i] = est[0] + 1j*est[1]
        
    E_est_2d = np.zeros((sysi.npsf,sysi.npsf), dtype=np.complex128)
    np.place(E_est_2d, mask=dark_mask, vals=E_est)
        
    return E_est_2d


