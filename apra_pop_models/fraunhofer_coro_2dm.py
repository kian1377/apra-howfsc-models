from .math_module import xp, _scipy, ensure_np_array
from . import utils
from . import imshows
from . import dm

# from scoobpsf import dm

import numpy as np
import astropy.units as u
from astropy.io import fits
import time
import os
from pathlib import Path
import copy

import poppy

from scipy.signal import windows

def make_vortex_phase_mask(npix, charge=6, 
                           grid='odd', 
                           singularity=None, 
                           focal_length=500*u.mm, pupil_diam=9.7*u.mm, wavelength=632.8*u.nm):
    if grid=='odd':
        x = xp.linspace(-npix//2, npix//2-1, npix)
    elif grid=='even':
        x = xp.linspace(-npix//2, npix//2-1, npix) + 1/2
    x,y = xp.meshgrid(x,x)
    th = xp.arctan2(y,x)

    phasor = xp.exp(1j*charge*th)
    
    if singularity is not None:
#         sing*D/(focal_length*lam)
        r = xp.sqrt((x-1/2)**2 + (y-1/2)**2)
        mask = r>(singularity*pupil_diam/(focal_length*wavelength)).decompose().value
        phasor *= mask
    
    return phasor

def prop_as(wavefront, wavelength, distance, pixelscale):
    n = wavefront.shape[0]

    delkx = 2*np.pi/(n*pixelscale.to_value(u.m/u.pix))
    kxy = (xp.linspace(-n/2, n/2-1, n) + 1/2)*delkx
    k = 2*np.pi/wavelength.to_value(u.m)
    kx, ky = xp.meshgrid(kxy,kxy)

    wf_as = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(wavefront)))
    
    kz = xp.sqrt(k**2 - kx**2 - ky**2 + 0j)
    tf = xp.exp(1j*kz*distance.to_value(u.m))

    prop_wf = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(wf_as * tf)))
    kz = 0.0
    tf = 0.0

    return prop_wf

# def AS(Ein, z):
#     E_hat = fft2d(Ein) # I need an amplitude factor
#     delkx = 2*np.pi/(n*delx)
    
#     kxy = np.linspace(-n/2, n/2-1, n)*delkx
# #     print(kxy/delkx)
#     kx, ky = np.meshgrid(kxy,kxy)
    
#     kz = np.sqrt(k**2 - kx**2 - ky**2 + 0j)
    
#     E_as = ifft2d(E_hat * np.exp(1j*kz*z)) # I need an amplitude factor
#     return E_as


class CORO():

    def __init__(self, 
                 wavelength=None, 
                 pupil_diam=9.5*u.mm,
                 lyot_diam=6.5*u.mm, 
                 npsf=128,
                 psf_pixelscale=5e-6*u.m/u.pix,
                 psf_pixelscale_lamD=None, 
                 detector_rotation=0, 
                 dm1_ref=np.zeros((34,34)),
                 dm2_ref=np.zeros((34,34)),
                 dm_inf=None, # defaults to inf.fits
                 d_dm1_dm2=277*u.mm, 
                 Imax_ref=1,
                 WFE=None,
                 FPM=None,
                 ):
        
        self.wavelength_c = 650e-9*u.m
        self.pupil_diam = pupil_diam
        
        self.wavelength = self.wavelength_c if wavelength is None else wavelength
        
        self.npix = 1000
        self.oversample = 2.048
        # self.npix = 512
        # self.oversample = 2
        self.N = int(self.npix*self.oversample)
        self.Nfpm = 4096

        self.WFE = xp.ones((self.N,self.N), dtype=complex) if WFE is None else WFE
        self.FPM = xp.ones((self.Nfpm,self.Nfpm), dtype=complex) if FPM is None else FPM

        self.pupil_apodizer_ratio = 1
        self.pupil_lyot_mag = 400/500 # pupil size ratios derived from focal lengths of relay OAPs

        self.fpm_fl = 500*u.mm
        self.imaging_fl = 200*u.mm

        self.lyot_diam = lyot_diam
        self.um_per_lamD = (self.wavelength_c*self.imaging_fl/(self.lyot_diam)).to(u.um)

        self.npsf = npsf
        if psf_pixelscale_lamD is None: # overrides psf_pixelscale this way
            self.psf_pixelscale = psf_pixelscale
            self.psf_pixelscale_lamD = self.psf_pixelscale.to_value(u.um/u.pix)/self.um_per_lamD.value
        else:
            self.psf_pixelscale_lamD = psf_pixelscale_lamD
            self.psf_pixelscale = self.psf_pixelscale_lamD * self.um_per_lamD/u.pix
        
        self.as_per_lamD = ((self.wavelength_c/self.pupil_diam)*u.radian).to(u.arcsec)
        self.psf_pixelscale_as = self.psf_pixelscale_lamD * self.as_per_lamD * self.oversample

        self.dm_inf = os.path.dirname(__file__)+'/inf.fits' if dm_inf is None else dm_inf
        self.dm1_ref = dm1_ref
        self.dm2_ref = dm2_ref
        self.d_dm1_dm2 = d_dm1_dm2
        self.init_dms()
        self.reset_dms()

        self.Imax_ref = Imax_ref
        self.reverse_parity = False

        self.det_rotation = detector_rotation
    
    

    def getattr(self, attr):
        return getattr(self, attr)
    
    @property
    def psf_pixelscale(self):
        return self._psf_pixelscale
    
    @psf_pixelscale.setter
    def psf_pixelscale(self, value):
        self._psf_pixelscale = value.to(u.m/u.pix)
        self.psf_pixelscale_lamD = (self._psf_pixelscale / self.um_per_lamD).decompose().value

    # def init_dms(self):
    #     self.Nact = 34
    #     self.Nacts = 952
    #     self.act_spacing = 300e-6*u.m
    #     self.dm_active_diam = 10.2*u.mm
    #     self.dm_full_diam = 11.1*u.mm
        
    #     self.full_stroke = 1.5e-6*u.m
        
    #     self.dm_mask = np.ones((self.Nact,self.Nact), dtype=bool)
    #     xx = (np.linspace(0, self.Nact-1, self.Nact) - self.Nact/2 + 1/2) * self.act_spacing.to(u.mm).value*2
    #     x,y = np.meshgrid(xx,xx)
    #     r = np.sqrt(x**2 + y**2)
    #     self.dm_mask[r>10.5] = 0 # had to set the threshold to 10.5 instead of 10.2 to include edge actuators
        
    #     self.dm_zernikes = ensure_np_array(poppy.zernike.arbitrary_basis(xp.array(self.dm_mask), nterms=15, outside=0))
        
    #     self.DM1 = poppy.ContinuousDeformableMirror(dm_shape=(self.Nact,self.Nact), name='DM1', 
    #                                                 actuator_spacing=self.act_spacing, 
    #                                                 influence_func=self.dm_inf,
    #                                                 include_factor_of_two=True, 
    #                                                 radius=self.dm_full_diam,
    #                                                )
        
    #     self.DM2 = poppy.ContinuousDeformableMirror(dm_shape=(self.Nact,self.Nact), name='DM2', 
    #                                                 actuator_spacing=self.act_spacing, 
    #                                                 influence_func=self.dm_inf,
    #                                                 include_factor_of_two=True, 
    #                                                 radius=self.dm_full_diam,
    #                                                )

        
    # def reset_dms(self):
    #     self.set_dm1(self.dm1_ref)
    #     self.set_dm2(self.dm2_ref)
    
    # def zero_dm1(self):
    #     self.set_dm(np.zeros((self.Nact,self.Nact)))
        
    # def set_dm1(self, dm_command):
    #     self.DM1.set_surface(ensure_np_array(dm_command))
        
    # def add_dm1(self, dm_command):
    #     self.DM1.set_surface(ensure_np_array(self.get_dm1()) + ensure_np_array(dm_command))
        
    # def get_dm1(self):
    #     return self.DM1.surface

    # def zero_dm2(self):
    #     self.set_dm(np.zeros((self.Nact,self.Nact)))
        
    # def set_dm2(self, dm_command):
    #     self.DM2.set_surface(ensure_np_array(dm_command))
        
    # def add_dm2(self, dm_command):
    #     self.DM2.set_surface(ensure_np_array(self.get_dm1()) + ensure_np_array(dm_command))
        
    # def get_dm2(self):
    #     return self.DM2.surface

    def init_dms(self):
        pupil_pxscl = self.pupil_diam.to_value(u.um)/self.npix
        sampling = 300/pupil_pxscl
        # sampling = int(np.round(300/pupil_pxscl))
        inf, inf_sampling = dm.make_gaussian_inf_fun(act_spacing=300e-6*u.m, sampling=sampling, coupling=0.15,)
        self.DM1 = dm.DeformableMirror(inf_fun=inf, inf_sampling=sampling, name='DM1')
        self.DM2 = dm.DeformableMirror(inf_fun=inf, inf_sampling=sampling, name='DM2')

        self.Nact = self.DM1.Nact
        self.Nacts = self.DM1.Nacts
        self.act_spacing = self.DM1.act_spacing
        self.dm_active_diam = self.DM1.active_diam
        self.dm_full_diam = self.DM1.pupil_diam
        
        self.full_stroke = self.DM1.full_stroke
        
        self.dm_mask = self.DM1.dm_mask

    def reset_dms(self):
        self.set_dm1(self.dm1_ref)
        self.set_dm2(self.dm2_ref)

    def zero_dms(self):
        self.set_dm1(xp.zeros((self.Nact,self.Nact)))
        self.set_dm2(xp.zeros((self.Nact,self.Nact)))
    
    def set_dm1(self, command):
        if command.shape[0]==self.Nacts:
            dm_command = self.DM1.map_actuators_to_command(xp.asarray(command))
        else: 
            dm_command = xp.asarray(command)
        self.DM1.command = dm_command
        
    def add_dm1(self, command):
        if command.shape[0]==self.Nacts:
            dm_command = self.DM1.map_actuators_to_command(xp.asarray(command))
        else: 
            dm_command = xp.asarray(command)
        self.DM1.command += dm_command
        
    def get_dm1(self):
        return self.DM1.command
    
    def set_dm2(self, command):
        if command.shape[0]==self.Nacts:
            dm_command = self.DM2.map_actuators_to_command(xp.asarray(command))
        else: 
            dm_command = xp.asarray(command)
        self.DM2.command = dm_command
        
    def add_dm2(self, command):
        if command.shape[0]==self.Nacts:
            dm_command = self.DM2.map_actuators_to_command(xp.asarray(command))
        else: 
            dm_command = xp.asarray(command)
        self.DM2.command += dm_command
        
    def get_dm2(self):
        return self.DM2.command
    
    def map_actuators_to_command(self, act_vector):
        command = np.zeros((self.Nact, self.Nact))
        command.ravel()[self.dm_mask.ravel()] = ensure_np_array(act_vector)
        return command
    
    def calc_wfs(self, save_wfs=True, quiet=True): # method for getting the PSF in photons
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        wfs = []
        self.wf = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength,
                                         npix=self.npix, oversample=self.oversample)
        self.wf *= poppy.CircularAperture(radius=self.pupil_diam/2, name='Coronagraph Pupil')
        if save_wfs: wfs.append(copy.copy(self.wf))

        WFE = poppy.ScalarTransmission(name='WFE Place-holder') if self.WFE is None else self.WFE
        self.wf = self.wf*WFE
        if save_wfs: wfs.append(copy.copy(self.wf))

        self.wf = self.wf*self.DM1
        if save_wfs: wfs.append(copy.copy(self.wf))

        self.wf.propagate_fresnel(self.d_dm1_dm2)
        self.wf = self.wf*self.DM2
        if save_wfs: wfs.append(copy.copy(self.wf))

        self.wf.propagate_fresnel(-self.d_dm1_dm2)
        self.wf *= poppy.ScalarTransmission('DM2 at Pupil')
        if save_wfs: wfs.append(copy.copy(self.wf))

        # WFE = poppy.ScalarTransmission(name='WFE Place-holder') if self.RETRIEVED is None else self.RETRIEVED
        # self.wf *= WFE
        # wfs.append(copy.copy(self.wf))

        self.wf.wavefront = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(self.wf.wavefront)))
        FPM = xp.ones((self.N, self.N)) if self.FPM is None else self.FPM
        self.wf.wavefront *= FPM
        self.wf *= poppy.ScalarTransmission(name='FPM')
        if save_wfs: wfs.append(copy.copy(self.wf))

        self.wf.wavefront = xp.fft.ifftshift(xp.fft.ifft2(xp.fft.fftshift(self.wf.wavefront)))
        self.wf *= poppy.ScalarTransmission('Lyot Pupil')
        if save_wfs: wfs.append(copy.copy(self.wf))

        self.wf = self.wf*self.LYOT
        if save_wfs: wfs.append(copy.copy(self.wf))

        nlamD = self.psf_pixelscale_lamD * self.npsf
        self.wf.wavefront = poppy.matrixDFT.matrix_dft(utils.pad_or_crop(self.wf.wavefront, self.npix), 
                                                       nlamD, self.npsf, inverse=False, centering='FFTSTYLE')
        self.wf *= poppy.ScalarTransmission('Detector')
        if self.reverse_parity:
            self.wf.wavefront = xp.rot90(xp.rot90(self.wf.wavefront))
        if save_wfs: wfs.append(copy.copy(self.wf))

        if save_wfs:
            return wfs
        else:
            return self.wf
    
    def apply_vortex(self, pupil_wf):
        # course FPM first
        vortex_mask = make_vortex_phase_mask(self.Nfpm)

        window_size = int(30/ (self.npix/self.Nfpm))
        print(window_size)
        w1d = xp.array(windows.tukey(window_size, 1, False))
        low_res_window = 1 - utils.pad_or_crop(xp.outer(w1d, w1d), self.Nfpm)
        imshows.imshow1(low_res_window, npix=128, pxscl=self.npix/self.Nfpm)

        fp_wf_low_res = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(utils.pad_or_crop(pupil_wf, self.Nfpm)))) # to FPM
        fp_wf_low_res *= vortex_mask * low_res_window # apply FPM
        pupil_wf_low_res = utils.pad_or_crop(xp.fft.ifftshift(xp.fft.ifft2(xp.fft.fftshift(fp_wf_low_res))), self.N) # to Lyot Pupil

        # high res FPM second
        high_res_sampling = 0.025 # lam/D per pixel
        # Nmft = self.Nfpm
        Nmft = int(np.round(30/high_res_sampling))
        # vortex_mask = utils.pad_or_crop(vortex_mask, Nmft)
        vortex_mask = make_vortex_phase_mask(Nmft)
        window_size = int(30/high_res_sampling)
        print(window_size)
        w1d = xp.array(windows.tukey(window_size, 1, False))
        high_res_window = utils.pad_or_crop(xp.outer(w1d, w1d), Nmft)

        # x = xp.linspace(-self.Nfpm//2, self.Nfpm//2-1, self.Nfpm) * high_res_sampling
        x = (xp.linspace(-Nmft//2, Nmft//2-1, Nmft)) * high_res_sampling
        x,y = xp.meshgrid(x,x)
        r = xp.sqrt(x**2 + y**2)
        sing_mask = r>0.15
        high_res_window *= sing_mask

        imshows.imshow1(high_res_window, npix=int(np.round(128*9.765625)), pxscl=high_res_sampling)

        nlamD = high_res_sampling * Nmft
        centering = 'SYMMETRIC'
        centering = 'FFTSTYLE'
        fp_wf_high_res = poppy.matrixDFT.matrix_dft(utils.pad_or_crop(pupil_wf, self.npix), 
                                                    nlamD, Nmft, inverse=False, centering=centering)
        fp_wf_high_res *= vortex_mask * high_res_window # apply FPM
        pupil_wf_high_res = poppy.matrixDFT.matrix_dft(fp_wf_high_res,
                                                       nlamD, self.npix, inverse=True, centering=centering)
        pupil_wf_high_res = utils.pad_or_crop(pupil_wf_high_res, self.N)

        # fp_wf_high_res = poppy.matrixDFT.matrix_dft(pupil_wf, 
        #                                             nlamD, Nmft, inverse=False, centering=centering)
        # fp_wf_high_res *= vortex_mask * high_res_window # apply FPM
        # pupil_wf_high_res = poppy.matrixDFT.matrix_dft(fp_wf_high_res,
        #                                                nlamD, self.N, inverse=True, centering=centering)
        # pupil_wf_high_res = utils.pad_or_crop(pupil_wf_high_res, self.N)

        post_fpm_pupil = pupil_wf_low_res + pupil_wf_high_res

        return post_fpm_pupil
    
    # def apply_vortex(self, pupil_wf):
    #     from scipy.signal import windows

    #     # course FPM first
    #     vortex_mask = make_vortex_phase_mask(self.Nfpm)

    #     window_size_lamD = 30
    #     window_size = int(window_size_lamD/ (self.npix/self.Nfpm)) 
    #     wx = xp.array(windows.tukey(window_size, 1, False))
    #     wy = xp.array(windows.tukey(window_size, 1, False))
    #     low_res_window = 1 - utils.pad_or_crop(xp.outer(wy, wx), self.Nfpm)
    #     ndisp = 256
    #     imshows.imshow1(low_res_window, npix=ndisp, pxscl=self.npix/self.Nfpm)

    #     fp_wf_low_res = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(utils.pad_or_crop(pupil_wf, self.Nfpm)))) # to FPM
    #     fp_wf_low_res *= vortex_mask * low_res_window # apply FPM
    #     pupil_wf_low_res = utils.pad_or_crop(xp.fft.ifftshift(xp.fft.ifft2(xp.fft.fftshift(fp_wf_low_res))), self.N) # to Lyot Pupil
        
    #     post_fpm_pupil = pupil_wf_low_res

    #     # high res FPM second
    #     centering = 'FFTSTYLE'
    #     # centering = 'SYMMETRIC'
    #     high_res_sampling = 0.025 # lam/D per pixel
    #     window_size = int(window_size_lamD/high_res_sampling)
    #     wx = xp.array(windows.tukey(window_size, 1, False))
    #     wy = xp.array(windows.tukey(window_size, 1, False))
    #     high_res_window_big = utils.pad_or_crop(xp.outer(wy, wx), self.Nfpm)

    #     window_size_lamD = 0.3
    #     window_size = int(window_size_lamD/high_res_sampling)
    #     wx = xp.array(windows.tukey(window_size, 1, False))
    #     wy = xp.array(windows.tukey(window_size, 1, False))
    #     high_res_window_small = utils.pad_or_crop(xp.outer(wy, wx), self.Nfpm)
    #     high_res_window = high_res_window_big - high_res_window_small
    #     imshows.imshow1(high_res_window, npix=int(np.round(ndisp*self.npix/self.Nfpm/high_res_sampling)), pxscl=high_res_sampling)
    #     imshows.imshow1(high_res_window, npix=128, pxscl=high_res_sampling)

    #     nlamD = high_res_sampling * self.Nfpm
    #     fp_wf_high_res = poppy.matrixDFT.matrix_dft(utils.pad_or_crop(pupil_wf, self.npix), 
    #                                                 nlamD, self.Nfpm, inverse=False, centering=centering)
    #     fp_wf_high_res *= vortex_mask * high_res_window # apply FPM
    #     pupil_wf_high_res = poppy.matrixDFT.matrix_dft(utils.pad_or_crop(fp_wf_high_res, self.npix), 
    #                                                    nlamD, self.Nfpm, inverse=True, centering=centering)
    #     pupil_wf_high_res = utils.pad_or_crop(pupil_wf_high_res, self.N)

    #     post_fpm_pupil += pupil_wf_high_res

    #     # high res FPM third
    #     high_res_sampling = 0.0025 # lam/D per pixel
    #     window_size = int(window_size_lamD/high_res_sampling)
    #     wx = xp.array(windows.tukey(window_size, 1, False))
    #     wy = xp.array(windows.tukey(window_size, 1, False))
    #     high_res_window = utils.pad_or_crop(xp.outer(wy, wx), self.Nfpm)

    #     # x = xp.linspace(-self.Nfpm//2, self.Nfpm//2-1, self.Nfpm) * high_res_sampling
    #     # x,y = xp.meshgrid(x,x)
    #     # r = xp.sqrt(x**2 + y**2)
    #     # sing_mask = r>0.3

    #     imshows.imshow1(high_res_window, npix=int(np.round(128*10)), pxscl=high_res_sampling)

    #     nlamD = high_res_sampling * self.Nfpm
    #     fp_wf_high_res = poppy.matrixDFT.matrix_dft(utils.pad_or_crop(pupil_wf, self.npix), 
    #                                                 nlamD, self.Nfpm, inverse=False, centering=centering)
    #     fp_wf_high_res *= vortex_mask * high_res_window # apply FPM
    #     pupil_wf_high_res = poppy.matrixDFT.matrix_dft(utils.pad_or_crop(fp_wf_high_res, self.npix), 
    #                                                    nlamD, self.Nfpm, inverse=True, centering=centering)
    #     pupil_wf_high_res = utils.pad_or_crop(pupil_wf_high_res, self.N)

    #     post_fpm_pupil += pupil_wf_high_res

    #     return post_fpm_pupil
    
    # def calc_wfs(self, save_wfs=True, quiet=True): # method for getting the PSF in photons
    #     start = time.time()
    #     if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
    #     wfs = []
    #     self.fwf = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength,
    #                                       npix=self.npix, oversample=self.oversample)
    #     self.wf = xp.ones((self.N, self.N), dtype=complex)
    #     self.wf *= poppy.CircularAperture(radius=self.pupil_diam/2).get_transmission(self.fwf)
    #     if save_wfs: wfs.append(copy.copy(self.wf))

    #     self.wf *= self.WFE
    #     if save_wfs: wfs.append(copy.copy(self.wf))

    #     dm1_surf = utils.pad_or_crop(self.DM1.get_surface(), self.N)
    #     self.wf *= xp.exp(1j*2*np.pi/self.wavelength.to_value(u.m) * 2*dm1_surf)
    #     if save_wfs: wfs.append(copy.copy(self.wf))

    #     dm2_surf = utils.pad_or_crop(self.DM2.get_surface(), self.N)
    #     self.wf = prop_as(self.wf, self.wavelength, self.d_dm1_dm2, self.pupil_diam/(self.npix*u.pix))
    #     self.wf *= xp.exp(1j*2*np.pi/self.wavelength.to_value(u.m) * 2*dm2_surf)
    #     if save_wfs: wfs.append(copy.copy(self.wf))

    #     self.wf = prop_as(self.wf, self.wavelength, -self.d_dm1_dm2, self.pupil_diam/(self.npix*u.pix)) # back to pupil
    #     if save_wfs: wfs.append(copy.copy(self.wf))

    #     self.wf = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(utils.pad_or_crop(self.wf, self.Nfpm)))) # to FPM
    #     if save_wfs: wfs.append(copy.copy(self.wf))
        
    #     self.wf *= self.FPM # apply FPM
    #     if save_wfs: wfs.append(copy.copy(self.wf))

    #     self.wf = utils.pad_or_crop(xp.fft.ifftshift(xp.fft.ifft2(xp.fft.fftshift(self.wf))), self.N) # to Lyot Pupil
    #     if save_wfs: wfs.append(copy.copy(self.wf)) 

    #     # self.LYOT = poppy.CircularAperture(radius=self.lyot_diam/2/self.pupil_lyot_mag, name='Lyot Stop')

    #     self.wf *= poppy.CircularAperture(radius=self.lyot_diam/2).get_transmission(self.fwf) # apply Lyot stop
    #     if save_wfs: wfs.append(copy.copy(self.wf))

    #     nlamD = self.psf_pixelscale_lamD * self.npsf
    #     self.wf = poppy.matrixDFT.matrix_dft(utils.pad_or_crop(self.wf, self.npix), 
    #                                          nlamD, self.npsf, inverse=False, centering='FFTSTYLE')
    #     if self.reverse_parity:
    #         self.wf = xp.rot90(xp.rot90(self.wf))
    #     if save_wfs: wfs.append(copy.copy(self.wf))

    #     if save_wfs:
    #         return wfs
    #     else:
    #         return self.wf
    
    def calc_psf(self):

        # fpwf = self.calc_wfs(save_wfs=False, quiet=True).wavefront
        fpwf = self.calc_wfs(save_wfs=False, quiet=True)

        return fpwf/xp.sqrt(self.Imax_ref)
    
    def snap(self): # method for getting the PSF in photons
        
        # image = self.calc_wfs(save_wfs=False, quiet=True).intensity
        image = xp.abs(self.calc_wfs(save_wfs=False, quiet=True))**2

        return image/self.Imax_ref
    


