from .math_module import xp, xcipy, ensure_np_array
from adefc_vortex import utils
from adefc_vortex.imshows import imshow1, imshow2, imshow3
from adefc_vortex import dm
from adefc_vortex import props

import numpy as np
import astropy.units as u
from astropy.io import fits

import os
from pathlib import Path
import time
import copy

import poppy

import matplotlib.pyplot as plt
plt.rcParams['image.origin']='lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec

from scipy.signal import windows
from scipy.optimize import minimize

def acts_to_command(acts, dm_mask):
    Nact = dm_mask.shape[0]
    command = xp.zeros((Nact,Nact))
    command[dm_mask] = xp.array(acts)
    return command

class MODEL():
    def __init__(
            self,
            npix=1000,
            d_dm1_dm2=700*u.mm,
            Nact=96,
            npsf=256,
        ):

        # initialize physical parameters
        self.wavelength_c = 650e-9

        self.dm_beam_diam = 47*u.mm
        self.d_dm1_dm2 = d_dm1_dm2
        self.lyot_pupil_diam = 4/5*self.dm_beam_diam
        self.lyot_stop_diam = 0.9 * self.lyot_pupil_diam
        self.lyot_ratio = 0.9
        self.control_rad = Nact/2 * 47/48 * self.lyot_ratio
        self.psf_pixelscale_lamDc = 0.347
        self.psf_pixelscale_lamD = self.psf_pixelscale_lamDc
        self.npsf = npsf

        self.Imax_ref = 1

        # initialize sampling parameters and load masks
        self.npix = npix
        self.oversample = 4.096
        self.N = int(self.npix*self.oversample)

        pwf = poppy.FresnelWavefront(beam_radius=self.dm_beam_diam/2, npix=self.npix, oversample=1) # pupil wavefront
        self.APERTURE = poppy.CircularAperture(radius=self.dm_beam_diam/2).get_transmission(pwf)
        self.APMASK = self.APERTURE>0
        self.LYOT = poppy.CircularAperture(radius=self.lyot_ratio*self.dm_beam_diam/2).get_transmission(pwf)
        self.AMP = xp.ones((self.npix,self.npix))
        self.OPD = xp.zeros((self.npix,self.npix))

        self.Nact = Nact
        if self.Nact==34:
            self.act_spacing = 1.4117*u.mm
        if self.Nact==48:
            self.act_spacing = 1.00*u.mm
        elif self.Nact==64:
            self.act_spacing = 0.750*u.mm
        elif self.Nact==96:
            self.act_spacing = 0.500*u.mm
        self.dm_pxscl = self.dm_beam_diam/(self.npix * u.pix)
        self.inf_sampling = self.act_spacing.to_value(u.m)/self.dm_pxscl.to_value(u.m/u.pix)
        self.inf_fun = dm.make_gaussian_inf_fun(act_spacing=self.act_spacing, sampling=self.inf_sampling, coupling=0.15, Nact=self.Nact+2)
        self.Nsurf = self.inf_fun.shape[0]

        y,x = (xp.indices((self.Nact, self.Nact)) - self.Nact//2 + 1/2)
        r = xp.sqrt(x**2 + y**2)
        self.dm_mask = r<(self.Nact/2 + 1/2)
        self.Nacts = int(2*self.dm_mask.sum())

        self.inf_fun_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(self.inf_fun,)))
        # DM command coordinates
        xc = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2)
        yc = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2)

        # Influence function frequncy sampling
        fx = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf))
        fy = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf))

        # forward DM model MFT matrices
        self.Mx = xp.exp(-1j*2*np.pi*xp.outer(fx,xc))
        self.My = xp.exp(-1j*2*np.pi*xp.outer(yc,fy))

        self.Mx_back = xp.exp(1j*2*np.pi*xp.outer(xc,fx))
        self.My_back = xp.exp(1j*2*np.pi*xp.outer(fy,yc))

        # Vortex model parameters
        self.oversample_vortex = 4.096
        self.N_vortex_lres = int(self.npix*self.oversample_vortex)
        self.lres_sampling = 1/self.oversample_vortex # low resolution sampling in lam/D per pixel
        self.lres_win_size = int(30/self.lres_sampling)
        w1d = xp.array(windows.tukey(self.lres_win_size, 1, False))
        self.lres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_lres)
        self.vortex_lres = props.make_vortex_phase_mask(self.N_vortex_lres)

        self.hres_sampling = 0.025 # lam/D per pixel; this value is chosen empirically
        self.N_vortex_hres = int(np.round(30/self.hres_sampling))
        self.hres_win_size = int(30/self.hres_sampling)
        w1d = xp.array(windows.tukey(self.hres_win_size, 1, False))
        self.hres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_hres)
        self.vortex_hres = props.make_vortex_phase_mask(self.N_vortex_hres)

        y,x = (xp.indices((self.N_vortex_hres, self.N_vortex_hres)) - self.N_vortex_hres//2)*self.hres_sampling
        r = xp.sqrt(x**2 + y**2)
        self.hres_dot_mask = r>=0.15

        self.det_rotation = 0
        self.flip_dm = False
        self.flip_lyot = False

        self.use_vortex = True

        self.dm1_command = xp.zeros((self.Nact, self.Nact))
        self.dm2_command = xp.zeros((self.Nact, self.Nact))

    def forward(self, actuators, wavelength, use_vortex=True, return_ints=False, plot=False, fancy_plot=False):
        dm1_command = xp.zeros((self.Nact,self.Nact))
        dm1_command[self.dm_mask] = xp.array(actuators[:self.Nacts//2])
        dm1_mft = self.Mx@dm1_command@self.My
        dm1_surf_fft = self.inf_fun_fft * dm1_mft
        dm1_surf = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(dm1_surf_fft,))).real
        dm1_surf = utils.pad_or_crop(dm1_surf, self.N)
        DM1_PHASOR = xp.exp(1j * 4*xp.pi/wavelength * dm1_surf)

        dm2_command = xp.zeros((self.Nact,self.Nact))
        dm2_command[self.dm_mask] = xp.array(actuators[self.Nacts//2:])
        dm2_mft = self.Mx@dm2_command@self.My
        dm2_surf_fft = self.inf_fun_fft * dm2_mft
        dm2_surf = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(dm2_surf_fft,))).real
        dm2_surf = utils.pad_or_crop(dm2_surf, self.N)
        DM2_PHASOR = xp.exp(1j * 4*xp.pi/wavelength * dm2_surf)

        if self.flip_dm: 
            DM1_PHASOR = xp.rot90(xp.rot90(DM1_PHASOR))
            DM2_PHASOR = xp.rot90(xp.rot90(DM2_PHASOR))

        # Initialize the wavefront
        WFE = utils.pad_or_crop(self.AMP, self.N) * xp.exp(1j * 2*xp.pi/wavelength * utils.pad_or_crop(self.OPD, self.N))
        E_EP = utils.pad_or_crop(self.APERTURE.astype(xp.complex128), self.N) * WFE / xp.sqrt(self.Imax_ref)
        if plot: imshow2(xp.abs(E_EP), xp.angle(E_EP), 'Entrance Pupil WF', npix=1.5*self.npix)

        E_DM1 = E_EP * utils.pad_or_crop(DM1_PHASOR, self.N)
        if plot: imshow2(xp.abs(E_DM1), xp.angle(E_DM1), 'After DM1 WF', npix=1.5*self.npix)

        E_DM2P = props.ang_spec(E_DM1, wavelength*u.m, self.d_dm1_dm2, self.dm_pxscl)
        if plot: imshow2(xp.abs(E_DM2P), xp.angle(E_DM2P), 'At DM2 WF', npix=1.5*self.npix)

        E_DM2 = E_DM2P * utils.pad_or_crop(DM2_PHASOR, self.N)
        if plot: imshow2(xp.abs(E_DM2), xp.angle(E_DM2), 'After DM2 WF', npix=1.5*self.npix)

        E_PUP = props.ang_spec(E_DM2, wavelength*u.m, -self.d_dm1_dm2, self.dm_pxscl)
        if plot: imshow2(xp.abs(E_PUP), xp.angle(E_PUP), 'Back to Pupil WF', npix=1.5*self.npix)

        if use_vortex:
            lres_wf = utils.pad_or_crop(E_PUP, self.N_vortex_lres) # pad to the larger array for the low res propagation
            fp_wf_lres = props.fft(lres_wf)
            fp_wf_lres *= self.vortex_lres * (1 - self.lres_window) # apply low res FPM and inverse Tukey window
            pupil_wf_lres = props.ifft(fp_wf_lres)
            pupil_wf_lres = utils.pad_or_crop(pupil_wf_lres, self.N)
            if plot: imshow2(xp.abs(pupil_wf_lres), xp.angle(pupil_wf_lres), 'Vortex FFT WF', npix=1.5*self.npix)

            fp_wf_hres = props.mft_forward(E_PUP, self.npix, self.N_vortex_hres, self.hres_sampling, convention='-')
            fp_wf_hres *= self.vortex_hres * self.hres_window * self.hres_dot_mask # apply high res FPM, window, and dot mask
            pupil_wf_hres = props.mft_reverse(fp_wf_hres, self.hres_sampling, self.npix, self.N, convention='+')
            if plot: imshow2(xp.abs(pupil_wf_hres), xp.angle(pupil_wf_hres), 'Vortex MFT WF', npix=1.5*self.npix)

            E_LP = (pupil_wf_lres + pupil_wf_hres)
            if plot: imshow2(xp.abs(E_LP), xp.angle(E_LP), 'Post Vortex Relay Pupil WF', npix=1.5*self.npix)
        else:
            E_LP = E_PUP
        
        if self.flip_lyot: 
            E_LP = xp.rot90(xp.rot90(E_LP))

        E_LS = E_LP * utils.pad_or_crop(self.LYOT, self.N)
        if plot: imshow2(xp.abs(E_LS), xp.angle(E_LS), 'After Lyot Stop WF', npix=1.5*self.npix)
        
        psf_pixelscale_lamD = self.psf_pixelscale_lamDc * self.wavelength_c/wavelength
        E_FP = props.mft_forward(E_LS, self.npix * self.lyot_ratio, self.npsf, psf_pixelscale_lamD)
        if plot: imshow2(xp.abs(E_FP)**2, xp.angle(E_FP), lognorm1=True)

        if fancy_plot: 
            fancy_plot_forward(dm1_command, dm2_command, DM1_PHASOR, DM2_PHASOR, E_PUP, E_LP, E_FP, npix=self.npix, wavelength=wavelength)

        if return_ints:
            return E_FP, E_EP, E_DM2P, DM1_PHASOR, DM2_PHASOR
        else:
            return E_FP
        
    def getattr(self, attr):
        return getattr(self, attr)
    
    def setattr(self, attr, val):
        setattr(self, attr, val)
        
    # def zero_dms(self):
    #     self.dm1_command = xp.zeros((self.Nact,self.Nact))
    #     self.dm2_command = xp.zeros((self.Nact,self.Nact))

    # def add_dm1(self, del_dm):
    #     self.dm1_command += del_dm
    
    # def set_dm1(self, dm_command):
    #     self.dm1_command = dm_command

    # def get_dm1(self,):
    #     return copy.copy(self.dm1_command)

    # def add_dm2(self, del_dm):
    #     self.dm2_command += del_dm
    
    # def set_dm2(self, dm_command):
    #     self.dm2_command = dm_command

    # def get_dm2(self,):
    #     return copy.copy(self.dm2_command)
        
    # def calc_wf(self, wavelength=650e-9):
    #     actuators = xp.concatenate([self.dm1_command[self.dm_mask], self.dm2_command[self.dm_mask]])
    #     fpwf = self.forward(actuators, wavelength, use_vortex=self.use_vortex, )
    #     return fpwf
        
    # def snap(self):
    #     actuators = xp.concatenate([self.dm1_command[self.dm_mask], self.dm2_command[self.dm_mask]])
    #     im = xp.abs(self.forward(actuators, wavelength, use_vortex=self.use_vortex,))**2
    #     return im

def val_and_grad(
        del_acts, 
        M, 
        rmad_vars,
        verbose=False, 
        plot=False, 
        fancy_plot=False,
    ):
    # Convert array arguments into correct types
    del_acts = xp.array(del_acts)
    del_acts_waves = del_acts/M.wavelength_c
    
    current_acts = rmad_vars['current_acts']
    E_ab = rmad_vars['E_ab']
    E_FP_NOM = rmad_vars['E_FP_NOM']
    E_EP = rmad_vars['E_EP']
    E_DM2P = rmad_vars['E_DM2P']
    DM1_PHASOR = rmad_vars['DM1_PHASOR']
    DM2_PHASOR = rmad_vars['DM2_PHASOR']
    wavelength = rmad_vars['wavelength']
    control_mask = rmad_vars['control_mask']
    r_cond = rmad_vars['r_cond']
    
    E_ab_l2norm = E_ab[control_mask].dot(E_ab[control_mask].conjugate()).real

    # Compute E_dm using the forward DM model
    E_FP_delDMs = M.forward(current_acts+del_acts, wavelength, use_vortex=True) # make sure to do the array indexing
    E_DMs = E_FP_delDMs - E_FP_NOM

    # compute the cost function
    delE = E_ab + E_DMs
    delE_vec = delE[control_mask] # make sure to do array indexing
    J_delE = delE_vec.dot(delE_vec.conjugate()).real
    J_c = r_cond * del_acts_waves.dot(del_acts_waves)
    J = (J_delE + J_c) / E_ab_l2norm
    if verbose: 
        print(f'\tCost-function J_delE: {J_delE:.3f}')
        print(f'\tCost-function J_c: {J_c:.3f}')
        print(f'\tCost-function normalization factor: {E_ab_l2norm:.3f}')
        print(f'\tTotal cost-function value: {J:.3f}\n')

    # Compute the gradient with the adjoint model
    delE_masked = control_mask * delE # still a 2D array
    dJ_dE_DMs = 2 * delE_masked / E_ab_l2norm

    psf_pixelscale_lamD = M.psf_pixelscale_lamDc * M.wavelength_c/wavelength
    dJ_dE_LS = props.mft_reverse(dJ_dE_DMs, psf_pixelscale_lamD, M.npix * M.lyot_ratio, M.N, convention='+')
    if plot: imshow2(xp.abs(dJ_dE_LS), xp.angle(dJ_dE_LS), 'RMAD Lyot Stop', npix=1.5*M.npix)

    dJ_dE_LP = dJ_dE_LS * utils.pad_or_crop(M.LYOT, M.N)
    if M.flip_lyot: 
        dJ_dE_LP = xp.rot90(xp.rot90(dJ_dE_LP))
    if plot: imshow2(xp.abs(dJ_dE_LP), xp.angle(dJ_dE_LP), 'RMAD Lyot Pupil', npix=1.5*M.npix)

    # Now we have to split and back-propagate the gradient along the two branches used to model 
    # the vortex. So one branch for the FFT vortex procedure and one for the MFT vortex procedure. 
    dJ_dE_LP_fft = utils.pad_or_crop(copy.copy(dJ_dE_LP), M.N_vortex_lres)
    dJ_dE_FPM_fft = props.fft(dJ_dE_LP_fft)
    dJ_dE_FP_fft = M.vortex_lres.conjugate() * (1 - M.lres_window) * dJ_dE_FPM_fft
    dJ_dE_PUP_fft = props.ifft(dJ_dE_FP_fft)
    dJ_dE_PUP_fft = utils.pad_or_crop(dJ_dE_PUP_fft, M.N)
    if plot: imshow2(xp.abs(dJ_dE_PUP_fft), xp.angle(dJ_dE_PUP_fft), 'RMAD FFT Pupil', npix=1.5*M.npix)

    dJ_dE_LP_mft = utils.pad_or_crop(copy.copy(dJ_dE_LP), M.N)
    dJ_dE_FPM_mft = props.mft_forward(dJ_dE_LP_mft,  M.npix, M.N_vortex_hres, M.hres_sampling, convention='-')
    dJ_dE_FP_mft = M.vortex_hres.conjugate() * M.hres_window * M.hres_dot_mask * dJ_dE_FPM_mft
    dJ_dE_PUP_mft = props.mft_reverse(dJ_dE_FP_mft, M.hres_sampling, M.npix, M.N, convention='+')
    if plot: imshow2(xp.abs(dJ_dE_PUP_mft), xp.angle(dJ_dE_PUP_mft), 'RMAD MFT Pupil', npix=1.5*M.npix)

    dJ_dE_PUP = dJ_dE_PUP_fft + dJ_dE_PUP_mft
    if plot: imshow2(xp.abs(dJ_dE_PUP), xp.angle(dJ_dE_PUP), 'RMAD Total Pupil', npix=1.5*M.npix)

    dJ_dE_DM2 = props.ang_spec(dJ_dE_PUP, wavelength*u.m, M.d_dm1_dm2, M.dm_pxscl)
    if plot: imshow2(xp.abs(dJ_dE_DM2), xp.angle(dJ_dE_DM2), 'RMAD DM2 WF', npix=1.5*M.npix)

    dJ_dE_DM2P = dJ_dE_DM2 * DM2_PHASOR.conj()
    if plot: imshow2(xp.abs(dJ_dE_DM2P), xp.angle(dJ_dE_DM2P), 'RMAD DM2 Plane WF', npix=1.5*M.npix)

    dJ_dE_DM1 = props.ang_spec(dJ_dE_DM2P, wavelength*u.m, -M.d_dm1_dm2, M.dm_pxscl)
    if plot: imshow2(xp.abs(dJ_dE_DM1), xp.angle(dJ_dE_DM1), 'RMAD DM1 WF', npix=1.5*M.npix)

    dJ_dS_DM2 = 4*xp.pi/wavelength * xp.imag(dJ_dE_DM2 * E_DM2P.conj() * DM2_PHASOR.conj())
    dJ_dS_DM1 = 4*xp.pi/wavelength * xp.imag(dJ_dE_DM1 * E_EP.conj() * DM1_PHASOR.conj())
    if M.flip_dm: 
        dJ_dS_DM1 = xp.rot90(xp.rot90(dJ_dS_DM1))
        dJ_dS_DM2 = xp.rot90(xp.rot90(dJ_dS_DM2))
    if plot: imshow2(xp.real(dJ_dS_DM2), xp.imag(dJ_dS_DM2), 'RMAD DM2 Surface', npix=1.5*M.npix)
    if plot: imshow2(xp.real(dJ_dS_DM1), xp.imag(dJ_dS_DM1), 'RMAD DM1 Surface', npix=1.5*M.npix)

    # Now pad back to the array size fo the DM surface to back propagate through the adjoint DM model
    dJ_dS_DM2 = utils.pad_or_crop(dJ_dS_DM2, M.Nsurf)
    x2_bar = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(dJ_dS_DM2)))
    x1_bar = x2_bar * M.inf_fun_fft.conj()
    dJ_dA2 = M.Mx_back@x1_bar@M.My_back / ( M.Nsurf * M.Nact * M.Nact ) # why I have to divide by this constant is beyond me
    if plot: imshow2(dJ_dA2.real, dJ_dA2.imag, 'RMAD DM2 Actuators')

    dJ_dS_DM1 = utils.pad_or_crop(dJ_dS_DM1, M.Nsurf)
    x2_bar = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(dJ_dS_DM1)))
    x1_bar = x2_bar * M.inf_fun_fft.conj()
    dJ_dA1 = M.Mx_back@x1_bar@M.My_back / ( M.Nsurf * M.Nact * M.Nact ) # why I have to divide by this constant is beyond me
    if plot: imshow2(dJ_dA1.real, dJ_dA1.imag, 'RMAD DM1 Actuators')

    dJ_dA = xp.concatenate([dJ_dA1[M.dm_mask].real, dJ_dA2[M.dm_mask].real]) + xp.array( r_cond * 2*del_acts_waves )

    if fancy_plot: 
        fancy_plot_adjoint(dJ_dE_DMs, dJ_dE_LP, dJ_dE_PUP, dJ_dS_DM1, dJ_dS_DM2, dJ_dA1, dJ_dA2, control_mask)

    return ensure_np_array(J), ensure_np_array(dJ_dA)

def val_and_grad_bb(
        del_acts, 
        M, 
        rmad_vars,
        verbose=False, 
        plot=False, 
        fancy_plot=False,
    ):

    del_acts = xp.array(del_acts)
    del_acts_waves = del_acts/M.wavelength_c
    
    current_acts = rmad_vars['current_acts']
    E_abs = rmad_vars['E_abs']
    E_FP_NOMs = rmad_vars['E_FP_NOMs']
    E_EPs = rmad_vars['E_EPs']
    E_DM2Ps = rmad_vars['E_DM2Ps']
    DM1_PHASORs = rmad_vars['DM1_PHASORs']
    DM2_PHASORs = rmad_vars['DM2_PHASORs']
    control_waves = rmad_vars['control_waves']
    control_mask = rmad_vars['control_mask']
    r_cond = rmad_vars['r_cond']
    weights = rmad_vars['weights']

    Nwaves = len(control_waves)
    mono_Js = np.zeros(Nwaves)
    mono_dJ_dAs = np.zeros((Nwaves, M.Nacts))
    mono_rmad_vars = {
        'current_acts':current_acts,
        'control_mask':control_mask,
        # 'r_cond':0,
        'r_cond':r_cond,
    }
    for i in range(Nwaves):
        mono_rmad_vars.update({'E_ab':copy.copy(E_abs[i])})
        mono_rmad_vars.update({'E_FP_NOM':copy.copy(E_FP_NOMs[i])})
        mono_rmad_vars.update({'E_EP':copy.copy(E_EPs[i])})
        mono_rmad_vars.update({'E_DM2P':copy.copy(E_DM2Ps[i])})
        mono_rmad_vars.update({'DM1_PHASOR':copy.copy(DM1_PHASORs[i])})
        mono_rmad_vars.update({'DM2_PHASOR':copy.copy(DM2_PHASORs[i])})
        mono_rmad_vars.update({'wavelength':copy.copy(control_waves[i])})

        J_mono, dJ_dA_mono = val_and_grad(del_acts, M, mono_rmad_vars, verbose=verbose, plot=plot, fancy_plot=fancy_plot)

        mono_Js[i] = J_mono
        mono_dJ_dAs[i] = dJ_dA_mono

    # if weights is None: 
    #     J_bb = np.sum(mono_Js)/Nwaves + ensure_np_array( r_cond * del_acts_waves.dot(del_acts_waves) )
    #     dJ_dA_bb = np.sum(mono_dJ_dAs, axis=0)/Nwaves + ensure_np_array( r_cond * 2*del_acts_waves )
    # else: 
    #     J_bb = np.sum(weights * mono_Js) / np.sum(weights) + ensure_np_array( r_cond * del_acts_waves.dot(del_acts_waves) )
    #     dJ_dA_bb = np.sum(weights[:, None] * mono_dJ_dAs, axis=0) / np.sum(weights) + ensure_np_array( r_cond * 2*del_acts_waves )
    
    if weights is None: 
        J_bb = np.sum(mono_Js)/Nwaves
        dJ_dA_bb = np.sum(mono_dJ_dAs, axis=0)/Nwaves
    else: 
        J_bb = np.sum(weights * mono_Js) / np.sum(weights)
        dJ_dA_bb = np.sum(weights[:, None] * mono_dJ_dAs, axis=0) / np.sum(weights)
    
    # # Testing beta regularization
    # dJ_dA_bb = np.sum(mono_dJ_dAs, axis=0)
    # alpha2 = dJ_dA_bb.dot(dJ_dA_bb)
    # E_abs_norm = E_abs[:,control_mask].ravel().dot(E_abs[:,control_mask].ravel().conjugate()).real
    # J_acts = alpha2 * 10**(r_cond) * ensure_np_array( del_acts.dot(del_acts) )
    # print(J_E)
    # print(J_acts)
    # dJ_dA_bb = dJ_dA_bb + alpha2 * 10**(r_cond) * 2 * ensure_np_array( del_acts ) 
    
    # J_bb = (J_E + J_acts)

    return J_bb, dJ_dA_bb


def fancy_plot_forward(dm1_command, dm2_command, DM1_PHASOR, DM2_PHASOR, E_PUP, E_LP, E_FP, npix, wavelength):
    DM1_SURF = ensure_np_array(wavelength/(4*xp.pi) * utils.pad_or_crop(xp.angle(DM1_PHASOR), 1.5*npix) )
    DM2_SURF = ensure_np_array(wavelength/(4*xp.pi) * utils.pad_or_crop(xp.angle(DM2_PHASOR), 1.5*npix) )
    E_PUP = ensure_np_array(utils.pad_or_crop(E_PUP, 1.5*npix))
    E_LP = ensure_np_array(utils.pad_or_crop(E_LP, 1.5*npix))
    E_FP = ensure_np_array(E_FP)

    fig = plt.figure(figsize=(20,10), dpi=125)
    gs = GridSpec(2, 6, figure=fig)

    title_fz = 16

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(ensure_np_array(dm1_command), cmap='viridis')
    ax.set_title('DM1 Command', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(ensure_np_array(dm2_command), cmap='viridis')
    ax.set_title('DM2 Command', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(DM1_SURF, cmap='viridis',)
    ax.set_title('DM1 Surface', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(DM2_SURF, cmap='viridis',)
    ax.set_title('DM2 Surface', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(np.abs(E_PUP), cmap='plasma')
    ax.set_title('Total Pupil Amplitude', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(np.angle(E_PUP), cmap='twilight')
    ax.set_title('Total Pupil Phase', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(np.abs(E_LP), cmap='plasma')
    ax.set_title('Lyot Pupil Amplitude', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(np.angle(E_LP), cmap='twilight')
    ax.set_title('Lyot Pupil Phase', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(np.abs(E_FP)**2, cmap='magma', norm=LogNorm(vmin=1e-8))
    ax.set_title('Focal Plane Intensity', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 4])
    ax.imshow(np.angle(E_FP), cmap='twilight')
    ax.set_title('Focal Plane Phase', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.subplots_adjust(hspace=-0.3)

def fancy_plot_adjoint(dJ_dE_DMs, dJ_dE_LP, dJ_dE_PUP, dJ_dS_DM1, dJ_dS_DM2, dJ_dA1, dJ_dA2, control_mask, npix=1000):

    control_mask = ensure_np_array(control_mask)
    dJ_dE_DMs = ensure_np_array(dJ_dE_DMs)
    dJ_dE_LP = ensure_np_array(utils.pad_or_crop(dJ_dE_LP, 1.5*npix))
    dJ_dE_PUP = ensure_np_array(utils.pad_or_crop(dJ_dE_PUP, 1.5*npix))
    dJ_dS_DM1 = ensure_np_array(utils.pad_or_crop(dJ_dS_DM1, int(1.5*npix))).real
    dJ_dS_DM2 = ensure_np_array(utils.pad_or_crop(dJ_dS_DM2, int(1.5*npix))).real
    dJ_dA1 = ensure_np_array(dJ_dA1).real
    dJ_dA2 = ensure_np_array(dJ_dA2).real

    fig = plt.figure(figsize=(20,10), dpi=125)
    gs = GridSpec(2, 5, figure=fig)

    title_fz = 26

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(control_mask * np.abs(dJ_dE_DMs)**2, cmap='magma', norm=LogNorm(vmin=1e-5))
    ax.set_title(r'$| \frac{\partial J}{\partial E_{DM}} |^2$', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(control_mask * np.angle(dJ_dE_DMs), cmap='twilight',)
    ax.set_title(r'$\angle \frac{\partial J}{\partial E_{DM}} $', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(np.abs(dJ_dE_LP), cmap='plasma')
    ax.set_title(r'$| \frac{\partial J}{\partial E_{LP}} |$', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(np.angle(dJ_dE_LP), cmap='twilight')
    ax.set_title(r'$\angle \frac{\partial J}{\partial E_{LP}} $', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(np.abs(dJ_dE_PUP), cmap='plasma')
    ax.set_title(r'$| \frac{\partial J}{\partial E_{PUP}} |$', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(np.angle(dJ_dE_PUP), cmap='twilight')
    ax.set_title(r'$\angle \frac{\partial J}{\partial E_{PUP}} $', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(dJ_dS_DM1, cmap='viridis')
    ax.set_title(r'$ \frac{\partial J}{\partial S_{DM1}} $', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(dJ_dS_DM2, cmap='viridis')
    ax.set_title(r'$ \frac{\partial J}{\partial S_{DM2}} $', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(dJ_dA1, cmap='viridis')
    ax.set_title(r'$ \frac{\partial J}{\partial A_1} $', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 4])
    ax.imshow(dJ_dA2, cmap='viridis')
    ax.set_title(r'$ \frac{\partial J}{\partial A_2} $', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.subplots_adjust(hspace=-0.2)


