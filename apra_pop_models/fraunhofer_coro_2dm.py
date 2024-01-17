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

def make_vortex_phase_mask(npix, charge=6, 
                           singularity=None, 
                           focal_length=500*u.mm, pupil_diam=9.7*u.mm, wavelength=632.8*u.nm):
    
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

class CORO():

    def __init__(self, 
                 wavelength=None, 
                 pupil_diam=9.5*u.mm,
                 lyot_diam=6.5*u.mm, 
                 npix=256, 
                 oversample=16,
                 npsf=100,
                 psf_pixelscale=5e-6*u.m/u.pix,
                 psf_pixelscale_lamD=None, 
                 detector_rotation=0, 
                 dm1_ref=np.zeros((34,34)),
                 dm2_ref=np.zeros((34,34)),
                 dm_inf=None, # defaults to inf.fits
                 d_dm1_dm2=277*u.mm, 
                 Imax_ref=1,
                 RETRIEVED=None,
                 FPM=None, 
                 use_lyot_stop=True):
        
        self.wavelength_c = 650e-9*u.m
        self.pupil_diam = pupil_diam
        
        self.wavelength = self.wavelength_c if wavelength is None else wavelength
        
        self.npix = npix
        self.oversample = oversample
        self.N = int(self.npix*self.oversample)

        self.RETRIEVED = RETRIEVED
        self.FPM = FPM
        self.use_lyot_stop = use_lyot_stop

        self.pupil_apodizer_ratio = 1
        self.pupil_lyot_ratio = 400/500 # pupil size ratios derived from focal lengths of relay OAPs

        self.fpm_fl = 500*u.mm
        self.imaging_fl = 200*u.mm

        self.lyot_diam = lyot_diam
        if self.use_lyot_stop:
            self.um_per_lamD = (self.wavelength_c*self.imaging_fl/(self.lyot_diam)).to(u.um)
            self.LYOT = poppy.CircularAperture(radius=1/self.pupil_lyot_ratio * self.lyot_diam/2, name='Lyot Stop')
        else:
            self.um_per_lamD = (self.wavelength_c*self.imaging_fl/(self.pupil_diam*self.pupil_lyot_ratio)).to(u.um)
            self.LYOT = poppy.ScalarTransmission('Lyot Pupil')

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
        sampling = int(np.round(300/pupil_pxscl))
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

        WFE = poppy.ScalarTransmission(name='WFE Place-holder') if self.RETRIEVED is None else self.RETRIEVED
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
        # if self.reverse_parity:
        #     fpwf = xp.rot90(xp.rot90(fpwf))
        if save_wfs: wfs.append(copy.copy(self.wf))

        if save_wfs:
            return wfs
        else:
            return self.wf
        
    def calc_psf(self):

        fpwf = self.calc_wfs(save_wfs=False, quiet=True).wavefront

        return fpwf/xp.sqrt(self.Imax_ref)
    
    def snap(self): # method for getting the PSF in photons
        
        image = self.calc_wfs(save_wfs=False, quiet=True).intensity

        return image/self.Imax_ref
    


