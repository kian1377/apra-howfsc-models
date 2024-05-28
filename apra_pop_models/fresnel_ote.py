from .math_module import xp, _scipy, ensure_np_array
from . import utils
from . import imshows
from . import dm
from . import props

import numpy as np
import astropy.units as u
from astropy.io import fits
import time
import os
from pathlib import Path
import copy

import poppy
from poppy.poppy_core import PlaneType
pupil = PlaneType.pupil
inter = PlaneType.intermediate
image = PlaneType.image
    
class OTE():

    def __init__(self, 
                 wavelength=None, 
                 npix=1000, 
                 oversample=4,
                 npsf=100,
                 norm='none',
                 use_opds=False,
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
        self.m2_diam = 650*u.mm
        # self.m2_footprint_diam = 2*291.3362*u.mm
        self.m2_footprint_diam = 585*u.mm
        self.m3_diam = 700*u.mm # using the maximum dimension of M3 to define the surface opdm, dims are [800*u.mm, 700*u.mm]
        # self.m3_footprint_diam = 2*39.192*u.mm
        self.m3_footprint_diam = 80*u.mm
        self.m4_diam = 100*u.mm

        self.m2_shift_per_mas = (286.8754*u.mm - 286.8838*u.mm)/(100*u.mas)
        self.m3_shift_per_mas = (175.5335*u.mm - 175.4427*u.mm)/(100*u.mas)
        
        self.index = index
        
        self.defocus = 0.0*u.mm
        
        self.calc_pupil = False
        self.use_opds = use_opds
        self.init_opds()

        self.m4_corr = 0.0*u.mm
        
    def getattr(self, attr):
        return getattr(self, attr)
    
    def init_opds(self, seeds=None):
        seed1, seed2, seed3, seed4 = (10,20,30,40)
        
        m1wf = poppy.FresnelWavefront(beam_radius=self.m1_diam/2, wavelength=self.wavelength_c, npix=self.npix, oversample=1)
        m1_opd = poppy.StatisticalPSDWFE(index=self.index, wfe=40*u.nm, radius=self.m1_diam/2, seed=seed1).get_opd(m1wf)
        self.m1_opd = poppy.ArrayOpticalElement(opd=m1_opd, pixelscale=m1wf.pixelscale, name='M1 OPD')
        
        m2wf = poppy.FresnelWavefront(beam_radius=self.m2_footprint_diam/2, wavelength=self.wavelength_c, npix=4096, oversample=1)
        m2_opd = poppy.StatisticalPSDWFE(index=self.index, wfe=20*u.nm, radius=self.m2_diam/2, seed=seed2).get_opd(m2wf)
        self.m2_opd = poppy.ArrayOpticalElement(opd=m2_opd, pixelscale=m2wf.pixelscale, name='M2 OPD')
        
        m3wf = poppy.FresnelWavefront(beam_radius=self.m3_footprint_diam/2, wavelength=self.wavelength_c, npix=4096, oversample=1)
        m3_opd = poppy.StatisticalPSDWFE(index=self.index, wfe=20*u.nm, radius=self.m3_diam/2, seed=seed3).get_opd(m3wf)
        self.m3_opd = poppy.ArrayOpticalElement(opd=m3_opd, pixelscale=m3wf.pixelscale, name='M3 OPD')
        
        m4wf = poppy.FresnelWavefront(beam_radius=self.m4_diam/2, wavelength=self.wavelength_c, npix=self.npix, oversample=1)
        m4_opd = poppy.StatisticalPSDWFE(index=self.index, wfe=15*u.nm, radius=self.m4_diam/2, seed=seed4).get_opd(m4wf)
        self.m4_opd = poppy.ArrayOpticalElement(opd=m4_opd, pixelscale=m4wf.pixelscale, name='M4 OPD')
        
    def shift_surface_errors(self, pointing, quiet=True, plot=False):
        m2_x_shift = np.abs(self.m2_shift_per_mas) * pointing[0]
        m2_y_shift = np.abs(self.m2_shift_per_mas) * pointing[1]

        m2_x_shift_pix = (m2_x_shift/self.m2_opd.pixelscale).decompose()
        m2_y_shift_pix = (m2_y_shift/self.m2_opd.pixelscale).decompose()
        if not quiet: print(m2_x_shift_pix, m2_y_shift_pix)

        m3_x_shift = np.abs(self.m3_shift_per_mas) * pointing[0]
        m3_y_shift = np.abs(self.m3_shift_per_mas) * pointing[1]

        m3_x_shift_pix = (m3_x_shift/self.m3_opd.pixelscale).decompose()
        m3_y_shift_pix = (m3_y_shift/self.m3_opd.pixelscale).decompose()
        if not quiet: print(m3_x_shift_pix, m3_y_shift_pix)

        m2_shifted_opd = _scipy.ndimage.shift(self.m2_opd.opd, (m2_y_shift_pix.value, m2_x_shift_pix.value))
        m3_shifted_opd = _scipy.ndimage.shift(self.m3_opd.opd, (m3_y_shift_pix.value, m3_x_shift_pix.value))
        if plot: imshows.imshow2(m2_shifted_opd - self.m2_opd.opd, m3_shifted_opd - self.m3_opd.opd, 
                                 'M2 Difference', 'M3 Difference', 
                                 npix=128)

        self.m2_opd.opd = copy.copy(m2_shifted_opd)
        self.m3_opd.opd = copy.copy(m3_shifted_opd)
        return
    
    def init_fosys(self):
        # hard-coded distances from zemax
        # d_m1_m2 = 16638.12910134875*u.mm
        # d_m2_m3 = 18500.0*u.mm
        # d_m3_m4 = 1895.0*u.mm
        # d_m4_fp = 2091.997751264193*u.mm

        # fl_m1 = 3.652962023674745E+004/2*u.mm
        # fl_m2 = -3.636649801410836E+003/2*u.mm
        # fl_m3 = 3.463978665836946E+003/2*u.mm

        d_m1_m2 = 1.683033905847026E+004*u.mm
        d_m2_m3 = 1.693225578996068E+004*u.mm
        d_m3_m4 = 1.254178665278960E+003*u.mm
        d_m4_fp = 1.481868319651047E+003*u.mm

        # fl_m1 = 1.827106897189599E+004*u.mm
        # fl_m2 = -1.836523655927925E+003*u.mm
        # fl_m3 = 1.159801322082377E+003*u.mm
        fl_m1 = 18485.075*u.mm
        fl_m2 = -1861.4048*u.mm
        # fl_m2 = -1.836523655927925E+003*u.mm
        fl_m3 = 1165.2982*u.mm
        
        PUPIL = poppy.CircularAperture(radius=self.pupil_diam/2)
        m1 = poppy.QuadraticLens(fl_m1, name='M1')
        m2 = poppy.QuadraticLens(fl_m2, name='M2')
        m3 = poppy.QuadraticLens(fl_m3, name='M3')
        m4 = poppy.CircularAperture(radius=self.m4_diam/2, name='M4/FSM')
        
        fosys = poppy.FresnelOpticalSystem(pupil_diameter=self.pupil_diam, npix=self.npix, beam_ratio=1/self.oversample)
        
        fosys.add_optic(PUPIL)
        fosys.add_optic(m1)
        if self.use_opds: fosys.add_optic(self.m1_opd)
        fosys.add_optic(m2, distance=d_m1_m2)
        if self.use_opds: fosys.add_optic(self.m2_opd)
        fosys.add_optic(m3, distance=d_m2_m3)
        if self.use_opds: fosys.add_optic(self.m3_opd)
        fosys.add_optic(m4, distance=d_m3_m4 + self.m4_corr)
        if self.use_opds: fosys.add_optic(self.m4_opd)
        if not self.calc_pupil:
            fosys.add_optic(poppy.ScalarTransmission('Image'), distance=d_m4_fp - self.m4_corr + self.defocus)
        
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
    
    def calc_wf(self): 
        self.init_fosys()
        self.init_inwave()
        _, wf = self.fosys.calc_psf(inwave=self.inwave, normalize=self.norm, return_final=True, return_intermediates=False)
        if self.calc_pupil:
            pupil_wf = utils.pad_or_crop(wf[0].wavefront, self.npix)
            amp = xp.abs(pupil_wf) * self.pupil_mask
            opd = xp.angle(pupil_wf)* self.wavelength.to_value(u.m)/(2*np.pi) * self.pupil_mask
            return amp, opd
        else:
            return wf[0].wavefront
    
    
    
    