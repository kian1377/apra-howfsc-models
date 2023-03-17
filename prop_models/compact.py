import numpy as np
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

import cupy as cp
import cupyx.scipy.ndimage

import misc_funs as misc

class CORO():

    def __init__(self, 
                 wavelength=None, 
                 npix=128, 
                 oversample=16,
                 npsf=100,
                 psf_pixelscale=5e-6*u.m/u.pix,
                 psf_pixelscale_lamD=None, 
                 detector_rotation=0, 
                 dm_ref=np.zeros((34,34)),
                 dm_inf=None, # defaults to inf.fits
                 im_norm=None,
                 RETRIEVED=None,
                 APODIZER=None,
                 FPM=None,
                 LYOT=None):
        
        poppy.accel_math.update_math_settings()
        
        self.is_model = True
        
        self.wavelength_c = 750e-9*u.m
        if wavelength is None: 
            self.wavelength = self.wavelength_c
        else: 
            self.wavelength = wavelength
        
        self.pupil_diam = 10.2*u.mm
        self.npix = npix
        self.oversample = oversample
        
        self.npsf = npsf
        if psf_pixelscale_lamD is None: # overrides psf_pixelscale this way
            self.psf_pixelscale = psf_pixelscale
            self.psf_pixelscale_lamD = (1/4.2) * self.psf_pixelscale.to(u.m/u.pix).value/5e-6
        else:
            self.psf_pixelscale_lamD = psf_pixelscale_lamD
            self.psf_pixelscale = 5e-6*u.m/u.pix / self.psf_pixelscale_lamD/(1/4.25)
        
        self.as_per_lamD = ((self.wavelength_c/self.pupil_diam)*u.radian).to(u.arcsec)
        self.psf_pixelscale_as = self.psf_pixelscale_lamD * self.as_per_lamD * self.oversample
        
        self.dm_inf = 'inf.fits' if dm_inf is None else dm_inf
        
        self.RETRIEVED = poppy.ScalarTransmission(name='Retrieved WFE Place-holder') if RETRIEVED is None else RETRIEVED
        self.APODIZER = poppy.ScalarTransmission(name='Apodizer Place-holder') if APODIZER is None else APODIZER
        self.FPM = poppy.ScalarTransmission(name='FPM Place-holder') if FPM is None else FPM
        self.LYOT = poppy.ScalarTransmission(name='Lyot Stop Place-holder') if LYOT is None else LYOT
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
        
        self.dm_zernikes = poppy.zernike.arbitrary_basis(cp.array(self.dm_mask), nterms=15, outside=0).get()
        
        self.DM = poppy.ContinuousDeformableMirror(dm_shape=(self.Nact,self.Nact), name='DM', 
                                                   actuator_spacing=self.act_spacing, 
                                                   influence_func=self.dm_inf,
                                                  )
        
    def reset_dm(self):
        self.set_dm(np.zeros((self.Nact,self.Nact)))
        
    def set_dm(self, dm_command):
        self.DM.set_surface(dm_command)
        
    def add_dm(self, dm_command):
        self.DM.set_surface(self.get_dm() + dm_command)
        
    def get_dm(self):
        return self.DM.surface.get()
    
    def show_dm(self):
        wf = poppy.FresnelWavefront(beam_radius=self.dm_active_diam/2, npix=self.npix, oversample=1)
        misc.imshow2(self.get_dm(), self.DM.get_opd(wf), 'DM Command', 'DM Surface',
                     pxscl2=wf.pixelscale.to(u.mm/u.pix))
    
    def init_osys(self):
        RETRIEVED = poppy.ScalarTransmission(name='Retrieved WFE Place-holder') if self.RETRIEVED is None else self.RETRIEVED
        APODIZER = poppy.ScalarTransmission(name='Apodizer Place-holder') if self.APODIZER is None else self.APODIZER
        FPM = poppy.ScalarTransmission(name='FPM Place-holder') if self.FPM is None else self.FPM
        LYOT = poppy.ScalarTransmission(name='Lyot Stop Place-holder') if self.LYOT is None else self.LYOT
        
        # define FresnelOpticalSystem and add optics
        osys = poppy.OpticalSystem(pupil_diameter=self.pupil_diam, npix=self.npix, oversample=self.oversample)
        
        osys.add_pupil(poppy.CircularAperture(radius=self.pupil_diam/2))
        osys.add_pupil(RETRIEVED)
        osys.add_pupil(self.DM)
        osys.add_pupil(APODIZER)
        osys.add_image(FPM)
        osys.add_pupil(LYOT)
        osys.add_detector(pixelscale=self.psf_pixelscale_as.value, fov_pixels=self.npsf/self.oversample,)
        
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
        _, wfs = self.osys.calc_psf(inwave=self.inwave, return_intermediates=True)
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
        
        return wfs
    
    def calc_psf(self, quiet=True): # method for getting the PSF in photons
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        self.init_osys()
        self.init_inwave()
        _, wf = self.osys.calc_psf(inwave=self.inwave, return_final=True, return_intermediates=False)
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
#         resamped_wf = self.rotate_and_interp_image(wf[0]).get()
        return wf[0].wavefront.get()
    
    def snap(self): # method for getting the PSF in photons
        self.init_osys()
        self.init_inwave()
        _, wf = self.osys.calc_psf(inwave=self.inwave, return_intermediates=False, return_final=True)
#         image = (cp.abs(self.rotate_and_interp_image(wf[0]))**2).get()
        image = wf[0].intensity.get()
        return image
    
    def rotate_and_interp_image(self, im_wf):
        wavefront = im_wf.wavefront
        wavefront_r = cupyx.scipy.ndimage.rotate(cp.real(wavefront), angle=-self.det_rotation, reshape=False, order=0)
        wavefront_i = cupyx.scipy.ndimage.rotate(cp.imag(wavefront), angle=-self.det_rotation, reshape=False, order=0)
        
        im_wf.wavefront = wavefront_r + 1j*wavefront_i
        
        resamped_wf = self.interp_wf(im_wf)
        return resamped_wf
    
    def interp_wf(self, wave): # this will interpolate the FresnelWavefront data to match the desired pixelscale
        n = wave.wavefront.shape[0]
        xs = (cp.linspace(0, n-1, n))*wave.pixelscale.to(u.m/u.pix).value
        
        extent = self.npsf*self.psf_pixelscale.to(u.m/u.pix).value
        
        for i in range(n):
            if xs[i+1]>extent:
                newn = i
                break
        newn += 2
        cropped_wf = misc.pad_or_crop(wave.wavefront, newn)

        wf_xmax = wave.pixelscale.to(u.m/u.pix).value * newn/2
        x,y = cp.ogrid[-wf_xmax:wf_xmax:cropped_wf.shape[0]*1j,
                       -wf_xmax:wf_xmax:cropped_wf.shape[1]*1j]

        det_xmax = extent/2
        newx,newy = cp.mgrid[-det_xmax:det_xmax:self.npsf*1j,
                             -det_xmax:det_xmax:self.npsf*1j]
        x0 = x[0,0]
        y0 = y[0,0]
        dx = x[1,0] - x0
        dy = y[0,1] - y0

        ivals = (newx - x0)/dx
        jvals = (newy - y0)/dy

        coords = cp.array([ivals, jvals])
        
        resamped_wf = cupyx.scipy.ndimage.map_coordinates(cropped_wf, coords, order=3)
        
        m = (wave.pixelscale.to(u.m/u.pix)/self.psf_pixelscale.to(u.m/u.pix)).value
        resamped_wf /= m
        
        return resamped_wf
    
    



