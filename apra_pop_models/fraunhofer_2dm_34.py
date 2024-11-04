from .math_module import xp,_scipy, ensure_np_array
from . import utils
from . import imshows
from . import dm
from . import props

import numpy as np
import astropy.units as u
from astropy.io import fits

import os
from pathlib import Path
import time
import copy

import poppy

from scipy.signal import windows
from scipy.optimize import minimize

def acts_to_command(acts, dm_mask):
    Nact = dm_mask.shape[0]
    command = xp.zeros((Nact,Nact))
    command[dm_mask] = xp.array(acts)
    return command

class MODEL():
    def __init__(self):

        # initialize physical parameters
        self.wavelength_c = 650e-9
        self.waves = np.array([self.wavelength_c])

        self.pupil_diam = 6.5*u.m
        self.dm_beam_diam = 9.6*u.mm
        self.d_dm1_dm2 = 283.569*u.mm
        # self.d_dm1_dm2 = 0*u.mm
        self.lyot_pupil_diam = 7.680*u.mm
        self.lyot_stop_diam = 0.9 * self.lyot_pupil_diam
        self.lyot_ratio = (self.lyot_stop_diam / self.lyot_pupil_diam).decompose().value
        self.control_rad = 34/2 * 9.6/10.2 * self.lyot_ratio
        self.psf_pixelscale_lamDc = 0.354
        self.psf_pixelscale_lamD = self.psf_pixelscale_lamDc
        self.npsf = 100

        self.wavelength = 650e-9

        self.Imax_ref = 1

        # initialize sampling parameters and load masks
        self.npix = 1000
        self.oversample = 4.096
        self.N = int(self.npix*self.oversample)

        pwf = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, npix=self.npix, oversample=1) # pupil wavefront
        self.APERTURE = poppy.CircularAperture(radius=self.pupil_diam/2).get_transmission(pwf)
        self.APMASK = self.APERTURE>0
        self.LYOT = poppy.CircularAperture(radius=self.lyot_ratio*self.pupil_diam/2).get_transmission(pwf)
        self.AMP = xp.ones((self.npix,self.npix))
        self.OPD = xp.zeros((self.npix,self.npix))

        self.Nact = 34
        self.act_spacing = 300e-6*u.m
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
        DM1_PHASOR = xp.exp(1j * 4*xp.pi/wavelength * utils.pad_or_crop(dm1_surf, self.N))

        dm2_command = xp.zeros((self.Nact,self.Nact))
        dm2_command[self.dm_mask] = xp.array(actuators[self.Nacts//2:])
        dm2_mft = self.Mx@dm2_command@self.My
        dm2_surf_fft = self.inf_fun_fft * dm2_mft
        dm2_surf = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(dm2_surf_fft,))).real
        DM2_PHASOR = xp.exp(1j * 4*xp.pi/wavelength * utils.pad_or_crop(dm2_surf, self.N))

        if self.flip_dm: 
            DM1_PHASOR = xp.rot90(xp.rot90(DM1_PHASOR))
            DM2_PHASOR = xp.rot90(xp.rot90(DM2_PHASOR))

        # Initialize the wavefront
        WFE = utils.pad_or_crop(self.AMP, self.N) * xp.exp(1j * 2*xp.pi/wavelength * utils.pad_or_crop(self.OPD, self.N))
        E_EP = utils.pad_or_crop(self.APERTURE.astype(xp.complex128), self.N) * WFE / xp.sqrt(self.Imax_ref)
        if plot: imshows.imshow2(xp.abs(E_EP), xp.angle(E_EP), 'Entrance Pupil WF', npix=1.5*self.npix)

        E_DM1 = E_EP * utils.pad_or_crop(DM1_PHASOR, self.N)
        if plot: imshows.imshow2(xp.abs(E_DM1), xp.angle(E_DM1), 'After DM1 WF', npix=1.5*self.npix)

        E_DM2P = props.ang_spec(E_DM1, self.wavelength*u.m, self.d_dm1_dm2, self.dm_pxscl)
        if plot: imshows.imshow2(xp.abs(E_DM2P), xp.angle(E_DM2P), 'At DM2 WF', npix=1.5*self.npix)

        E_DM2 = E_DM2P * utils.pad_or_crop(DM2_PHASOR, self.N)
        if plot: imshows.imshow2(xp.abs(E_DM2), xp.angle(E_DM2), 'After DM2 WF', npix=1.5*self.npix)

        E_PUP = props.ang_spec(E_DM2, self.wavelength*u.m, -self.d_dm1_dm2, self.dm_pxscl)
        if plot: imshows.imshow2(xp.abs(E_PUP), xp.angle(E_PUP), 'Back to Pupil WF', npix=1.5*self.npix)

        if use_vortex:
            lres_wf = utils.pad_or_crop(E_PUP, self.N_vortex_lres) # pad to the larger array for the low res propagation
            fp_wf_lres = props.fft(lres_wf)
            fp_wf_lres *= self.vortex_lres * (1 - self.lres_window) # apply low res FPM and inverse Tukey window
            pupil_wf_lres = props.ifft(fp_wf_lres)
            pupil_wf_lres = utils.pad_or_crop(pupil_wf_lres, self.N)
            if plot: imshows.imshow2(xp.abs(pupil_wf_lres), xp.angle(pupil_wf_lres), 'Vortex FFT WF', npix=1.5*self.npix)

            fp_wf_hres = props.mft_forward(E_PUP, self.npix, self.N_vortex_hres, self.hres_sampling, convention='-')
            fp_wf_hres *= self.vortex_hres * self.hres_window * self.hres_dot_mask # apply high res FPM, window, and dot mask
            pupil_wf_hres = props.mft_reverse(fp_wf_hres, self.hres_sampling, self.npix, self.N, convention='+')
            if plot: imshows.imshow2(xp.abs(pupil_wf_hres), xp.angle(pupil_wf_hres), 'Vortex MFT WF', npix=1.5*self.npix)

            E_LP = (pupil_wf_lres + pupil_wf_hres)
            if plot: imshows.imshow2(xp.abs(E_LP), xp.angle(E_LP), 'Post Vortex Relay Pupil WF', npix=1.5*self.npix)
        else:
            E_LP = E_PUP
        
        if self.flip_lyot: 
            E_LP = xp.rot90(xp.rot90(E_LP))

        E_LS = E_LP * utils.pad_or_crop(self.LYOT, self.N)
        if plot: imshows.imshow2(xp.abs(E_LS), xp.angle(E_LS), 'After Lyot Stop WF', npix=1.5*self.npix)
        
        psf_pixelscale_lamD = self.psf_pixelscale_lamDc * self.wavelength_c/wavelength
        E_FP = props.mft_forward(E_LS, self.npix * self.lyot_ratio, self.npsf, psf_pixelscale_lamD)
        if plot: imshows.imshow2(xp.abs(E_FP)**2, xp.angle(E_FP), lognorm1=True)

        if fancy_plot: 
            imshows.fancy_plot_forward(dm1_command, dm2_command, DM1_PHASOR, DM2_PHASOR, E_PUP, E_LP, E_FP, npix=self.npix, wavelength=wavelength)

        if return_ints:
            return E_FP, E_EP, E_DM2P, DM1_PHASOR, DM2_PHASOR
        else:
            return E_FP
        
    def getattr(self, attr):
        return getattr(self, attr)
    
    def setattr(self, attr, val):
        setattr(self, attr, val)
        
    def zero_dms(self):
        self.dm1_command = xp.zeros((self.Nact,self.Nact))
        self.dm2_command = xp.zeros((self.Nact,self.Nact))

    def add_dm1(self, del_dm):
        self.dm1_command += del_dm
    
    def set_dm1(self, dm_command):
        self.dm1_command = dm_command

    def get_dm1(self,):
        return copy.copy(self.dm1_command)

    def add_dm2(self, del_dm):
        self.dm2_command += del_dm
    
    def set_dm2(self, dm_command):
        self.dm2_command = dm_command

    def get_dm2(self,):
        return copy.copy(self.dm2_command)
        
    def calc_wf(self, wavelength=650e-9):
        actuators = xp.concatenate([self.dm1_command[self.dm_mask], self.dm2_command[self.dm_mask]])
        fpwf = self.forward(actuators, wavelength, use_vortex=self.use_vortex, )
        return fpwf
        
    # def snap(self):
    #     actuators = xp.concatenate([self.dm1_command[self.dm_mask], self.dm2_command[self.dm_mask]])
    #     im = xp.abs(self.forward(actuators, wavelength, use_vortex=self.use_vortex,))**2
    #     return im

def val_and_grad(del_acts, M, actuators, E_ab, control_mask, wavelength, r_cond, verbose=False, plot=False, fancy_plot=False):
    # Convert array arguments into correct types
    del_acts_waves = del_acts/M.wavelength_c
    actuators = ensure_np_array(actuators)
    E_ab = xp.array(E_ab)
    
    E_ab_l2norm = E_ab[control_mask].dot(E_ab[control_mask].conjugate()).real

    # Compute E_dm using the forward DM model
    E_FP_nom, E_EP, E_DM2P, DM1_PHASOR, DM2_PHASOR = M.forward(actuators, wavelength, use_vortex=True, return_ints=True,) # make sure to do the array indexing
    E_DMs = M.forward(actuators+del_acts, wavelength, use_vortex=True) # make sure to do the array indexing
    E_DMs = E_DMs - E_FP_nom

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
    if plot: imshows.imshow2(xp.abs(dJ_dE_LS), xp.angle(dJ_dE_LS), 'RMAD Lyot Stop', npix=1.5*M.npix)

    dJ_dE_LP = dJ_dE_LS * utils.pad_or_crop(M.LYOT, M.N)
    if M.flip_lyot: dJ_dE_LP = xp.rot90(xp.rot90(dJ_dE_LP))
    if plot: imshows.imshow2(xp.abs(dJ_dE_LP), xp.angle(dJ_dE_LP), 'RMAD Lyot Pupil', npix=1.5*M.npix)

    # Now we have to split and back-propagate the gradient along the two branches used to model 
    # the vortex. So one branch for the FFT vortex procedure and one for the MFT vortex procedure. 
    dJ_dE_LP_fft = utils.pad_or_crop(copy.copy(dJ_dE_LP), M.N_vortex_lres)
    dJ_dE_FPM_fft = props.fft(dJ_dE_LP_fft)
    dJ_dE_FP_fft = M.vortex_lres.conjugate() * (1 - M.lres_window) * dJ_dE_FPM_fft
    dJ_dE_PUP_fft = props.ifft(dJ_dE_FP_fft)
    dJ_dE_PUP_fft = utils.pad_or_crop(dJ_dE_PUP_fft, M.N)
    if plot: imshows.imshow2(xp.abs(dJ_dE_PUP_fft), xp.angle(dJ_dE_PUP_fft), 'RMAD FFT Pupil', npix=1.5*M.npix)

    dJ_dE_LP_mft = utils.pad_or_crop(copy.copy(dJ_dE_LP), M.N)
    dJ_dE_FPM_mft = props.mft_forward(dJ_dE_LP_mft,  M.npix, M.N_vortex_hres, M.hres_sampling, convention='-')
    dJ_dE_FP_mft = M.vortex_hres.conjugate() * M.hres_window * M.hres_dot_mask * dJ_dE_FPM_mft
    dJ_dE_PUP_mft = props.mft_reverse(dJ_dE_FP_mft, M.hres_sampling, M.npix, M.N, convention='+')
    if plot: imshows.imshow2(xp.abs(dJ_dE_PUP_mft), xp.angle(dJ_dE_PUP_mft), 'RMAD MFT Pupil', npix=1.5*M.npix)

    dJ_dE_PUP = dJ_dE_PUP_fft + dJ_dE_PUP_mft
    if plot: imshows.imshow2(xp.abs(dJ_dE_PUP), xp.angle(dJ_dE_PUP), 'RMAD Total Pupil', npix=1.5*M.npix)

    dJ_dE_DM2 = props.ang_spec(dJ_dE_PUP, M.wavelength*u.m, M.d_dm1_dm2, M.dm_pxscl)
    if plot: imshows.imshow2(xp.abs(dJ_dE_DM2), xp.angle(dJ_dE_DM2), 'RMAD DM2 WF', npix=1.5*M.npix)

    dJ_dE_DM2P = dJ_dE_DM2 * DM2_PHASOR.conj()
    if plot: imshows.imshow2(xp.abs(dJ_dE_DM2P), xp.angle(dJ_dE_DM2P), 'RMAD DM2 Plane WF', npix=1.5*M.npix)

    dJ_dE_DM1 = props.ang_spec(dJ_dE_DM2P, M.wavelength*u.m, -M.d_dm1_dm2, M.dm_pxscl)
    if plot: imshows.imshow2(xp.abs(dJ_dE_DM1), xp.angle(dJ_dE_DM1), 'RMAD DM1 WF', npix=1.5*M.npix)

    dJ_dS_DM2 = 4*xp.pi/M.wavelength * xp.imag(dJ_dE_DM2 * E_DM2P.conj() * DM2_PHASOR.conj())
    dJ_dS_DM1 = 4*xp.pi/M.wavelength * xp.imag(dJ_dE_DM1 * E_EP.conj() * DM1_PHASOR.conj())
    if M.flip_dm: 
        dJ_dS_DM1 = xp.rot90(xp.rot90(dJ_dS_DM1))
        dJ_dS_DM2 = xp.rot90(xp.rot90(dJ_dS_DM2))
    if plot: imshows.imshow2(xp.real(dJ_dS_DM2), xp.imag(dJ_dS_DM2), 'RMAD DM2 Surface', npix=1.5*M.npix)
    if plot: imshows.imshow2(xp.real(dJ_dS_DM1), xp.imag(dJ_dS_DM1), 'RMAD DM1 Surface', npix=1.5*M.npix)

    # Now pad back to the array size fo the DM surface to back propagate through the adjoint DM model
    dJ_dS_DM2 = utils.pad_or_crop(dJ_dS_DM2, M.Nsurf)
    x2_bar = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(dJ_dS_DM2)))
    x1_bar = x2_bar * M.inf_fun_fft.conj()
    dJ_dA2 = M.Mx_back@x1_bar@M.My_back / ( M.Nsurf * M.Nact * M.Nact ) # why I have to divide by this constant is beyond me
    if plot: imshows.imshow2(dJ_dA2.real, dJ_dA2.imag, 'RMAD DM2 Actuators')

    dJ_dS_DM1 = utils.pad_or_crop(dJ_dS_DM1, M.Nsurf)
    x2_bar = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(dJ_dS_DM1)))
    x1_bar = x2_bar * M.inf_fun_fft.conj()
    dJ_dA1 = M.Mx_back@x1_bar@M.My_back / ( M.Nsurf * M.Nact * M.Nact ) # why I have to divide by this constant is beyond me
    if plot: imshows.imshow2(dJ_dA1.real, dJ_dA1.imag, 'RMAD DM1 Actuators')

    dJ_dA = xp.concatenate([dJ_dA1[M.dm_mask].real, dJ_dA2[M.dm_mask].real]) + xp.array( r_cond * 2*del_acts_waves )

    if fancy_plot: 
        imshows.fancy_plot_adjoint(dJ_dE_DMs, dJ_dE_LP, dJ_dE_PUP, dJ_dS_DM1, dJ_dS_DM2, dJ_dA1, dJ_dA2, control_mask)

    return ensure_np_array(J), ensure_np_array(dJ_dA)

def val_and_grad_bb(del_acts, M, actuators, E_abs, control_mask, waves, r_cond, weights=None, verbose=False, plot=False, fancy_plot=False):
    # del_acts, M, actuators, E_ab, control_mask, wavelength, r_cond,
    Nwaves = len(waves)
    E_abs = xp.array(E_abs)
    del_acts_waves = del_acts/M.wavelength_c

    r_cond_mono = 0
    J_monos = np.zeros(Nwaves)
    dJ_dA_monos = np.zeros((Nwaves, M.Nacts))
    for i in range(Nwaves):
        wavelength = waves[i]
        E_ab = E_abs[i]
        J_mono, dJ_dA_mono = val_and_grad(del_acts, M, actuators, E_ab, control_mask, wavelength, r_cond_mono, verbose=verbose, plot=plot, fancy_plot=fancy_plot)
        # print(J_mono, dJ_dA_mono.shape)
        J_monos[i] = J_mono
        dJ_dA_monos[i] = dJ_dA_mono

    # imshows.imshow1(acts_to_command(dJ_dA_monos[2] - dJ_dA_monos[0], M.dm_mask))

    if weights is None: 
        weights = np.array(Nwaves*[1])
        # TODO: implement weights for each wavelength correctly

    J_bb = np.sum(J_monos)/Nwaves + r_cond * del_acts_waves.dot(del_acts_waves)
    dJ_dA_bb = np.sum(dJ_dA_monos, axis=0) + ensure_np_array( r_cond * 2*del_acts_waves )
    
    return J_bb, dJ_dA_bb



