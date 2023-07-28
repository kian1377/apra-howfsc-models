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

    
import misc

def ensure_np_array(arr):
    if isinstance(arr, np.ndarray):
        return arr
    else:
        return arr.get()
    
class OTE():

    def __init__(self, 
                 wavelength=None, 
                 npix=1024, 
                 oversample=2,
                 npsf=100,
                 norm='none',
                 use_opds=True,
                 m2_shift=None,
                 m3_shift=None,
                 index=3.0,
                ):
        
        self.pupil_diam = 6.5*u.m
        self.wavelength_c = 650e-9*u.m
        if wavelength is None: 
            self.wavelength = self.wavelength_c
        else: 
            self.wavelength = wavelength
        
        self.npix = npix
        self.oversample = oversample
        
        self.norm = norm
        
        self.m1_diam = 6.5*u.m
        self.m2_diam = 700*u.mm
        self.m3_diam = 800*u.mm # using the maximum dimension of M3 to define the surface opdm, dims are [800*u.mm, 700*u.mm]
        self.m4_diam = 100*u.mm
        
        self.m2_shift = m2_shift
        self.m3_shift = m3_shift
        
        self.index = index
        
        self.defocus = 0.0*u.mm
        
        self.use_opds = use_opds
        self.init_opds()
        
    def getattr(self, attr):
        return getattr(self, attr)
    
    def init_opds(self, seeds=None):
        
        seed1, seed2, seed3, seed4 = (1,2,3,4)
        
        m1wf = poppy.FresnelWavefront(beam_radius=self.m1_diam/2, wavelength=self.wavelength_c, npix=self.npix, oversample=1)
        m1_opd = poppy.StatisticalPSDWFE(index=self.index, wfe=10*u.nm, radius=self.m1_diam/2, seed=seed1).get_opd(m1wf)
        self.m1_opd = poppy.ArrayOpticalElement(opd=m1_opd, pixelscale=m1wf.pixelscale)
        
        m2wf = poppy.FresnelWavefront(beam_radius=self.m2_diam/2, wavelength=self.wavelength_c, npix=self.npix, oversample=1)
        m2_opd = poppy.StatisticalPSDWFE(index=self.index, wfe=10*u.nm, radius=self.m2_diam/2, seed=seed2).get_opd(m2wf)
        self.m2_opd = poppy.ArrayOpticalElement(opd=m2_opd, pixelscale=m2wf.pixelscale)
        
        m3wf = poppy.FresnelWavefront(beam_radius=self.m3_diam/2, wavelength=self.wavelength_c, npix=self.npix, oversample=1)
        m3_opd = poppy.StatisticalPSDWFE(index=self.index, wfe=10*u.nm, radius=self.m3_diam/2, seed=seed3).get_opd(m3wf)
        self.m3_opd = poppy.ArrayOpticalElement(opd=m3_opd, pixelscale=m3wf.pixelscale)
        
        m4wf = poppy.FresnelWavefront(beam_radius=self.m4_diam/2, wavelength=self.wavelength_c, npix=self.npix, oversample=1)
        m4_opd = poppy.StatisticalPSDWFE(index=self.index, wfe=10*u.nm, radius=self.m4_diam/2, seed=seed4).get_opd(m4wf)
        self.m4_opd = poppy.ArrayOpticalElement(opd=m4_opd, pixelscale=m4wf.pixelscale)
        
    def shift_surface_errors(surf_element, shift=np.array([0,0])*u.m):
        surf_opd = surf_element.opd

        x_shift_pix = shift[0].to_value(u.m)/wf.pixelscale.to_value(u.m/u.pix)
        y_shift_pix = shift[1].to_value(u.m)/wf.pixelscale.to_value(u.m/u.pix)
        print('Pixels to shift in x and y:', x_shift_pix, y_shift_pix)
        
        shifted_surf_opd = _scipy.ndimage.shift(surf_opd, (y_shift_pix, x_shift_pix))
        new_surf_element = poppy.ArrayOpticalElement(opd=shifted_surf_opd, pixelscale=surf_element.pixelscale)
        return new_surf_element
    
    def init_fosys(self):
        # hard-coded distances from zemax
        d_m1_m2 = 16638.12910134875*u.mm
        d_m2_m3 = 18500.0*u.mm
        d_m3_m4 = 1895.0*u.mm
        d_m4_fp = 2091.997751264193*u.mm - 0.0062856074*u.m
        
        fl_m1 = 3.652962023674745E+004/2*u.mm
        fl_m2 = -3.636649801410836E+003/2*u.mm
        fl_m3 = 3.463978665836946E+003/2*u.mm
        
        PUPIL = poppy.CircularAperture(radius=self.pupil_diam/2)
        m1 = poppy.QuadraticLens(fl_m1, name='M1')
        m2 = poppy.QuadraticLens(fl_m2, name='M2')
        m3 = poppy.QuadraticLens(fl_m3, name='M3')
        m4 = poppy.ScalarTransmission()
        
        m2_opd = self.m2_opd if self.m2_shift is None else self.shift_surface_errors(self.m2_opd, self.m2_shift)
        m3_opd = self.m3_opd if self.m3_shift is None else self.shift_surface_errors(self.m3_opd, self.m3_shift)
        
        fosys = poppy.FresnelOpticalSystem(pupil_diameter=self.pupil_diam, npix=self.npix, beam_ratio=1/self.oversample)
        
        fosys.add_optic(PUPIL)
        fosys.add_optic(m1)
        if self.use_opds: fosys.add_optic(self.m1_opd)
        fosys.add_optic(m2, distance=d_m1_m2)
        if self.use_opds: fosys.add_optic(self.m2_opd)
        fosys.add_optic(m3, distance=d_m2_m3)
        if self.use_opds: fosys.add_optic(self.m3_opd)
        fosys.add_optic(m4, distance=d_m3_m4)
        if self.use_opds: fosys.add_optic(self.m4_opd)
        fosys.add_optic(poppy.ScalarTransmission('Image'), distance=d_m4_fp + self.defocus)
        if self.propagate_to_pupil:
            fosys.add_optic(poppy.QuadraticLens(224.99998119573664*u.m), distance=224.99998119573664*u.m)
            fosys.add_optic(poppy.ScalarTransmission('Pupil Plane'), distance=224.99998119573664*u.m)
        
        self.fosys = fosys
        
    @property
    def pupil_mask(self):
        PUPIL = poppy.CircularAperture(radius=self.pupil_diam/2)
        wf = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength,
                                        npix=self.npix, oversample=1)
        self._pupil_mask = PUPIL.get_transmission(wf)
        return self._pupil_mask
    
    def init_inwave(self):
        inwave = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength,
                                        npix=self.npix, oversample=self.oversample)
        self.inwave = inwave
    
    def calc_wfs(self, quiet=False):
        self.propagate_to_pupil = False
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        self.init_fosys()
        self.init_inwave()
        _, wfs = self.fosys.calc_psf(inwave=self.inwave, normalize=self.norm, return_intermediates=True)
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
        
        return wfs
    
    def calc_psf(self): # method for getting the PSF in photons
        self.propagate_to_pupil = False
        start = time.time()
        self.init_fosys()
        self.init_inwave()
        _, wf = self.fosys.calc_psf(inwave=self.inwave, normalize=self.norm, return_final=True, return_intermediates=False)
        return wf[0].wavefront
    
#     def calc_pupil(self, return_amp_opd=True):
#         self.propagate_to_pupil = True
#         self.init_fosys()
#         self.init_inwave()
#         _, wf = self.fosys.calc_psf(inwave=self.inwave, normalize=self.norm, return_final=True, return_intermediates=False)
#         pupil_wf = misc.pad_or_crop(wf[0].wavefront, self.npix)
        
#         pupil_amp = xp.abs(pupil_wf)*self.pupil_mask
#         pupil_opd = xp.angle(pupil_wf)*self.pupil_mask * self.wavelength_c.to_value(u.m)/(2*np.pi)
        
#         if return_amp_opd:
#             return pupil_amp, pupil_opd
#         else:
#             pupil_wf = pupil_amp *xp.exp(1j*2*np.pi/self.wavelength_c.to_value(u.m) * pupil_opd)
#             return pupil_wf
    
    def calc_pupil(self, return_amp_opd=True):
        psf = self.calc_psf()
        pupil_wf = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(psf))) / psf.shape[0]
        
        pupil_wf = misc.pad_or_crop(pupil_wf, self.npix)
        
        pupil_amp = xp.abs(pupil_wf)*self.pupil_mask
        pupil_opd = xp.angle(pupil_wf)*self.pupil_mask * self.wavelength_c.to_value(u.m)/(2*np.pi)
        
        if return_amp_opd:
            return pupil_amp, pupil_opd
        else:
            pupil_wf = pupil_amp *xp.exp(1j*2*np.pi/self.wavelength_c.to_value(u.m) * pupil_opd)
            return pupil_wf
    
    
    
    