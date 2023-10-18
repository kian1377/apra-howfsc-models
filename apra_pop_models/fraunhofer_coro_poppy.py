import numpy as np
import scipy
import astropy.units as u
from astropy.io import fits
from pathlib import Path
import pickle
import time
import copy

import poppy

from poppy.poppy_core import PlaneType
pupil = PlaneType.pupil
inter = PlaneType.intermediate
image = PlaneType.image

if poppy.accel_math._USE_CUPY:
    import cupy as cp
    import cupyx.scipy as _scipy
    xp = cp
else: 
    cp = None
    xp = np
    _scipy = scipy
    
def ensure_np_array(arr):
    if isinstance(arr, np.ndarray):
        return arr
    else:
        return arr.get()

class CORO():

    def __init__(self, 
                 wavelength=None, 
                 npix=256, 
                 oversample=16,
                 npsf=100,
                 psf_pixelscale=5e-6*u.m/u.pix,
                 psf_pixelscale_lamD=None, 
                 detector_rotation=0, 
                 dm_ref=np.zeros((34,34)),
                 dm_inf=None, # defaults to inf.fits
                 wf_norm='none',
                 im_norm=1,
                 RETRIEVED=None,
                 APODIZER=None,
                 FPM=None,
                 LYOT=None):
        
        self.wavelength_c = 650e-9*u.m
        self.pupil_diam = 10.2*u.mm
        
        self.wavelength = self.wavelength_c if wavelength is None else wavelength
        
        self.npix = npix
        self.oversample = oversample
        
        self.npsf = npsf
        if psf_pixelscale_lamD is None: # overrides psf_pixelscale this way
            self.psf_pixelscale = psf_pixelscale
            self.psf_pixelscale_lamD = (1/3.6) * self.psf_pixelscale.to(u.m/u.pix).value/5e-6
        else:
            self.psf_pixelscale_lamD = psf_pixelscale_lamD
            self.psf_pixelscale = 5e-6*u.m/u.pix / self.psf_pixelscale_lamD/(1/4.2)
        
        self.as_per_lamD = ((self.wavelength_c/self.pupil_diam)*u.radian).to(u.arcsec)
        self.psf_pixelscale_as = self.psf_pixelscale_lamD * self.as_per_lamD * self.oversample
        
        self.dm_inf = 'inf.fits' if dm_inf is None else dm_inf
        
        self.wf_norm = wf_norm
        self.im_norm = 1
        
        self.APODIZER = APODIZER
        self.RETRIEVED = RETRIEVED
        self.FPM = FPM
        self.LYOT = LYOT
        
        self.dm_ref = dm_ref
        self.init_dm()
        
        self.det_rotation = detector_rotation
        
        self.pupil_apodizer_ratio = 1 
        self.pupil_lyot_ratio = 350/500 # pupil size ratios derived from focal lengths of relay OAPs
        
        self.init_osys()
        
    def getattr(self, attr):
        return getattr(self, attr)

    def init_dm(self):
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
        
        self.DM = poppy.ContinuousDeformableMirror(dm_shape=(self.Nact,self.Nact), name='DM', 
                                                   actuator_spacing=self.act_spacing, 
                                                   influence_func=self.dm_inf,
                                                  )
        
    def reset_dm(self):
        self.set_dm(self.dm_ref)
    
    def zero_dm(self):
        self.set_dm(np.zeros((self.Nact,self.Nact)))
        
    def set_dm(self, dm_command):
        self.DM.set_surface(ensure_np_array(dm_command))
        
    def add_dm(self, dm_command):
        self.DM.set_surface(self.get_dm() + ensure_np_array(dm_command))
        
    def get_dm(self):
        return ensure_np_array(self.DM.surface)
    
    def map_actuators_to_command(self, act_vector):
        command = np.zeros((self.Nact, self.Nact))
        command.ravel()[self.dm_mask.ravel()] = ensure_np_array(act_vector)
        return command
    
    def init_osys(self):
        RETRIEVED = poppy.ScalarTransmission(name='Retrieved WFE Place-holder') if self.RETRIEVED is None else self.RETRIEVED
        APODIZER = poppy.ScalarTransmission(name='Apodizer Place-holder') if self.APODIZER is None else self.APODIZER
        FPM = poppy.ScalarTransmission(name='FPM Place-holder') if self.FPM is None else self.FPM
        LYOT = poppy.ScalarTransmission(name='Lyot Stop Place-holder') if self.LYOT is None else self.LYOT
        
        # define FresnelOpticalSystem and add optics
        osys = poppy.OpticalSystem(pupil_diameter=self.pupil_diam, npix=self.npix, oversample=self.oversample)
        
        osys.add_pupil(poppy.CircularAperture(radius=self.pupil_diam/2))
        osys.add_pupil(self.DM)
        osys.add_image(poppy.ScalarTransmission('Intermediate Image Plane'))
        osys.add_pupil(RETRIEVED)
        osys.add_pupil(APODIZER)
        osys.add_image(FPM)
        osys.add_pupil(LYOT)
        
        self.psf_pixelscale_as = self.psf_pixelscale_lamD * self.as_per_lamD * self.oversample
        osys.add_detector(pixelscale=self.psf_pixelscale_as.value, fov_pixels=self.npsf/self.oversample)
        
        self.osys = osys
        
    def init_inwave(self):
        inwave = poppy.Wavefront(diam=self.pupil_diam, wavelength=self.wavelength,
                                 npix=self.npix, oversample=self.oversample)
        self.inwave = inwave
    
    def calc_wfs(self, quiet=False):
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        self.init_osys()
        self.init_inwave()
        _, wfs = self.osys.calc_psf(inwave=self.inwave, normalize=self.wf_norm, return_intermediates=True)
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
        wfs[-1].wavefront /= np.sqrt(self.im_norm)
        return wfs
    
    def calc_psf(self, quiet=True): # method for getting the PSF in photons
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        self.init_osys()
        self.init_inwave()
        _, wf = self.osys.calc_psf(inwave=self.inwave, normalize=self.wf_norm, return_final=True, return_intermediates=False)
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
        return wf[0].wavefront/np.sqrt(self.im_norm)
    
    def snap(self): # method for getting the PSF in photons
        self.init_osys()
        self.init_inwave()
        _, wf = self.osys.calc_psf(inwave=self.inwave, normalize=self.wf_norm, return_intermediates=False, return_final=True)
        image = wf[0].intensity
        image /= self.im_norm
        return image
    


