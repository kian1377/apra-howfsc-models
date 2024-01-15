from .math_module import xp,_scipy, ensure_np_array
from .import dm, utils
from . import imshows

import numpy as np
import astropy.units as u
from astropy.io import fits
import time
import os
from pathlib import Path
import copy

import poppy

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
                 d_dm1_dm2=320*u.mm, 
                 wf_norm='none',
                 Imax_ref=1,
                 RETRIEVED=None,
                 FPM=None, 
                 use_lyot_stop=False):
        
        self.wavelength_c = 650e-9*u.m
        self.pupil_diam = pupil_diam
        
        self.wavelength = self.wavelength_c if wavelength is None else wavelength
        
        self.npix = npix
        self.oversample = oversample

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
        
        self.wf_norm = wf_norm
        self.Imax_ref = 1
        
        self.dm1_ref = dm1_ref
        self.dm2_ref = dm2_ref
        self.d_dm1_dm2 = d_dm1_dm2
        self.init_dms()
        self.reset_dms()

        self.reverse_parity = False

        self.det_rotation = detector_rotation
        
    def getattr(self, attr):
        return getattr(self, attr)
    
    def init_dms(self):
        self.Nact = 34
        self.Nacts = 952
        self.act_spacing = 300e-6*u.m
        self.dm_active_diam = 10.2*u.mm
        self.dm_full_diam = 11.1*u.mm
        
        self.full_stroke = 1.5e-6*u.m
        
        self.dm_mask = np.ones((self.Nact,self.Nact), dtype=bool)
        xx = (np.linspace(0, self.Nact-1, self.Nact) - self.Nact/2 + 1/2) * self.act_spacing.to(u.mm).value*2
        x,y = np.meshgrid(xx,xx)
        r = np.sqrt(x**2 + y**2)
        self.dm_mask[r>10.5] = 0 # had to set the threshold to 10.5 instead of 10.2 to include edge actuators
        
        self.dm_zernikes = ensure_np_array(poppy.zernike.arbitrary_basis(xp.array(self.dm_mask), nterms=15, outside=0))
        
        self.DM1 = poppy.ContinuousDeformableMirror(dm_shape=(self.Nact,self.Nact), name='DM1', 
                                                    actuator_spacing=self.act_spacing, 
                                                    influence_func=self.dm_inf,
                                                    include_factor_of_two=True, 
                                                   )
        
        self.DM2 = poppy.ContinuousDeformableMirror(dm_shape=(self.Nact,self.Nact), name='DM2', 
                                                    actuator_spacing=self.act_spacing, 
                                                    influence_func=self.dm_inf,
                                                    include_factor_of_two=True, 
                                                   )

        
    def reset_dms(self):
        self.set_dm1(self.dm1_ref)
        self.set_dm2(self.dm2_ref)
    
    def zero_dm1(self):
        self.set_dm(np.zeros((self.Nact,self.Nact)))
        
    def set_dm1(self, dm_command):
        self.DM1.set_surface(ensure_np_array(dm_command))
        
    def add_dm1(self, dm_command):
        self.DM1.set_surface(ensure_np_array(self.get_dm()) + ensure_np_array(dm_command))
        
    def get_dm1(self):
        return self.DM1.surface

    def zero_dm2(self):
        self.set_dm(np.zeros((self.Nact,self.Nact)))
        
    def set_dm2(self, dm_command):
        self.DM2.set_surface(ensure_np_array(dm_command))
        
    def add_dm2(self, dm_command):
        self.DM2.set_surface(ensure_np_array(self.get_dm()) + ensure_np_array(dm_command))
        
    def get_dm2(self):
        return self.DM2.surface
    
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
        self.wf *= poppy.CircularAperture(radius=self.pupil_diam/2)
        self.wf *= self.DM1
        if save_wfs: wfs.append(copy.copy(self.wf))

        self.wf.propagate_fresnel(self.d_dm1_dm2)
        self.wf *= self.DM2
        wfs.append(copy.copy(self.wf))

        self.wf.propagate_fresnel(-self.d_dm1_dm2)
        self.wf *= poppy.ScalarTransmission('DM2 at Pupil')
        wfs.append(copy.copy(self.wf))

        WFE = poppy.ScalarTransmission(name='WFE Place-holder') if self.RETRIEVED is None else self.RETRIEVED
        self.wf *= WFE
        wfs.append(copy.copy(self.wf))

        # self.wf *= poppy.QuadraticLens(f_lens=self.fpm_fl)
        # self.wf.propagate_fresnel(self.fpm_fl)
        self.wf.wavefront = poppy.accel_math.fft_2d(self.wf.wavefront, forward=True, fftshift=False)
        FPM = poppy.ScalarTransmission(name='FPM Place-holder') if self.FPM is None else self.FPM
        self.wf *= FPM
        wfs.append(copy.copy(self.wf))

        # self.wf.propagate_fresnel(-self.fpm_fl/2)
        # self.wf *= poppy.QuadraticLens(f_lens=self.fpm_fl)
        self.wf.wavefront = poppy.accel_math.fft_2d(self.wf.wavefront, forward=False, fftshift=False)
        self.wf *= poppy.ScalarTransmission('Lyot Pupil')
        wfs.append(copy.copy(self.wf))

        self.wf = copy.copy(self.wf)*self.LYOT
        wfs.append(copy.copy(self.wf))

        # if self.reverse_parity:
        #     fpwf = xp.rot90(xp.rot90(fpwf))
        if save_wfs:
            return wfs
        else:
            return self.wf
    
    def snap(self): # method for getting the PSF in photons
        
        image = xp.abs(self.calc_psf())**2

        return image
    


