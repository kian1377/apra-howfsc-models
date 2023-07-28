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
                 npix=512, 
                 oversample=8,
                 npsf=100,
                 psf_pixelscale=5e-6*u.m/u.pix,
                 psf_pixelscale_lamD=None, 
                 detector_rotation=0, 
                 dm_ref=np.zeros((34,34)),
                 dm_inf=None, # defaults to inf.fits
                 wf_norm='none',
                 im_norm=1,
                 use_opds=False,
                 OTEWFE=None,
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
        
        self.dm_inf = 'inf.fits' if dm_inf is None else dm_inf
        
        self.wf_norm = 'none'
        self.im_norm = im_norm
        
        self.defocus = 0*u.nm
        self.use_opds = use_opds
        
        self.OTEWFE = poppy.ScalarTransmission(name='OTE WFE Place-holder') if OTEWFE is None else OTEWFE
        self.APODIZER = poppy.ScalarTransmission(name='Apodizer Place-holder') if APODIZER is None else APODIZER
        self.FPM = poppy.ScalarTransmission(name='FPM Place-holder') if FPM is None else FPM
        self.LYOT = poppy.ScalarTransmission(name='Lyot Stop Place-holder') if LYOT is None else LYOT
        
        self.dm_ref = dm_ref
        self.init_dm()
        
        self.oap1_diam = 12.7*u.mm
        self.oap2_diam = 12.7*u.mm
        self.oap3_diam = 12.7*u.mm
        self.oap4_diam = 12.7*u.mm
        self.oap5_diam = 12.7*u.mm
        
        self.fl_oap1 = 200*u.mm
        self.fl_oap2 = 200*u.mm
        self.fl_oap3 = 500*u.mm
        self.fl_oap4 = 350*u.mm
        self.fl_oap5 = 200*u.mm
        
        self.det_rotation = detector_rotation
        
        self.PUPIL = poppy.CircularAperture(radius=self.pupil_diam/2)
        wf = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength_c,
                                    npix=self.npix, oversample=1)
        self.pupil_mask = self.PUPIL.get_transmission(wf)
        
        self.init_opds()
        
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
        
        if poppy.accel_math._USE_CUPY:
            self.dm_zernikes = poppy.zernike.arbitrary_basis(xp.array(self.dm_mask), nterms=15, outside=0).get()
        else: 
            self.dm_zernikes = poppy.zernike.arbitrary_basis(xp.array(self.dm_mask), nterms=15, outside=0)
            
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
    
    def oaefl(self, roc, oad, k=-1):
        """
        roc: float
            parent parabola radius of curvature
        angle: float
            off-axis angle in radians
        oad: float
            off-axis distance
        """
        # compute parabolic sag
        sag = (1/roc)*oad**2 /(1 + np.sqrt(1-(k+1)*(1/roc)**2 * oad**2))
        return roc/2 + sag
    
    def init_fosys(self):
        oap1 = poppy.QuadraticLens(self.fl_oap1, name='OAP1')
        oap2 = poppy.QuadraticLens(self.fl_oap2, name='OAP2')
        oap3 = poppy.QuadraticLens(self.fl_oap3, name='OAP3')
        oap4 = poppy.QuadraticLens(self.fl_oap4, name='OAP4')
        oap5 = poppy.QuadraticLens(self.fl_oap5, name='OAP5')
        
        oap1_ap = poppy.CircularAperture(radius=self.oap1_diam/2)
        oap2_ap = poppy.CircularAperture(radius=self.oap2_diam/2)
        oap3_ap = poppy.CircularAperture(radius=self.oap3_diam/2)
        oap4_ap = poppy.CircularAperture(radius=self.oap4_diam/2)
        oap5_ap = poppy.CircularAperture(radius=self.oap5_diam/2)
        
        OTEWFE = poppy.ScalarTransmission(name='OTE WFE Place-holder') if self.OTEWFE is None else self.OTEWFE
        APODIZER = poppy.ScalarTransmission(name='Apodizer Place-holder') if self.APODIZER is None else self.APODIZER
        FPM = poppy.ScalarTransmission(name='FPM Place-holder') if self.FPM is None else self.FPM
        LYOT = poppy.ScalarTransmission(name='Lyot Stop Place-holder') if self.LYOT is None else self.LYOT
        
        # define FresnelOpticalSystem and add optics
        fosys = poppy.FresnelOpticalSystem(pupil_diameter=self.pupil_diam, npix=self.npix, beam_ratio=1/self.oversample)
        
        
        fosys.add_optic(self.PUPIL) 
        if self.use_opds: fosys.add_optic(OTEWFE)
        fosys.add_optic(self.DM)
        fosys.add_optic(oap1, distance=self.fl_oap1)
        fosys.add_optic(oap1_ap)
        if self.use_opds: fosys.add_optic(self.oap1_opd)
        fosys.add_optic(poppy.ScalarTransmission('Int Focal Plane'), distance=self.fl_oap1)
        fosys.add_optic(oap2, distance=self.fl_oap2)
        fosys.add_optic(oap2_ap)
        if self.use_opds: fosys.add_optic(self.oap2_opd)
        fosys.add_optic(APODIZER, distance=self.fl_oap2)
        fosys.add_optic(oap3, distance=self.fl_oap3)
        fosys.add_optic(oap3_ap)
        if self.use_opds: fosys.add_optic(self.oap3_opd)
            
        fosys.add_optic(FPM, distance=self.fl_oap3)
        
        fosys.add_optic(oap4, distance=self.fl_oap4)
        fosys.add_optic(oap4_ap)
        if self.use_opds: fosys.add_optic(self.oap4_opd)
        fosys.add_optic(poppy.ScalarTransmission('Lyot Stop Plane'), distance=self.fl_oap4)
        if self.image_pupil:
            self.fosys = fosys
            return
        else:
            fosys.add_optic(LYOT)
            fosys.add_optic(oap5, distance=self.fl_oap5)
            fosys.add_optic(oap5_ap)
            if self.use_opds: fosys.add_optic(self.oap5_opd)
            fosys.add_optic(poppy.Detector(pixelscale=self.psf_pixelscale, fov_pixels=self.npsf, interp_order=3),
                            distance=self.fl_oap5 + self.defocus)

            self.inter_fp_index = 5 if self.use_opds else 4
            self.fpm_index = 13 if self.use_opds else 10
            self.image_index = 22 if self.use_opds else 17
        
            self.fosys = fosys
            return
        
    def init_opds(self, seeds=None):
        
        seed1, seed2, seed3, seed4, seed5 = (1,2,3,4,5)
        self.oap1_opd = poppy.StatisticalPSDWFE('OAP1 OPD', index=3.0, wfe=20*u.nm, radius=self.oap1_diam/2, seed=seed1)
        self.oap2_opd = poppy.StatisticalPSDWFE('OAP2 OPD', index=3.0, wfe=20*u.nm, radius=self.oap2_diam/2, seed=seed2)
        self.oap3_opd = poppy.StatisticalPSDWFE('OAP3 OPD', index=3.0, wfe=20*u.nm, radius=self.oap3_diam/2, seed=seed3)
        self.oap4_opd = poppy.StatisticalPSDWFE('OAP4 OPD', index=3.0, wfe=20*u.nm, radius=self.oap4_diam/2, seed=seed4)
        self.oap5_opd = poppy.StatisticalPSDWFE('OAP5 OPD', index=3.0, wfe=20*u.nm, radius=self.oap5_diam/2, seed=seed5)
        
    def init_inwave(self):
        inwave = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength,
                                        npix=self.npix, oversample=self.oversample)
        self.inwave = inwave
    
    def calc_wfs(self, quiet=False):
        self.image_pupil = False
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        self.init_fosys()
        self.init_inwave()
        self.pupil_mask = self.PUPIL.get_transmission(self.inwave)>0
        _, wfs = self.fosys.calc_psf(inwave=self.inwave, normalize=self.wf_norm, return_intermediates=True)
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
        
        wfs[-1].wavefront /= np.sqrt(self.im_norm)
        
        return wfs
    
    def calc_psf(self, quiet=True): # method for getting the PSF in photons
        self.image_pupil = False
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        self.init_fosys()
        self.init_inwave()
        self.pupil_mask = self.PUPIL.get_transmission(self.inwave)>0
        _, wf = self.fosys.calc_psf(inwave=self.inwave, normalize=self.wf_norm, return_final=True, return_intermediates=False)
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
            
        psf = wf[0].wavefront
        psf /= np.sqrt(self.im_norm)
        return psf
    
    
    def snap(self): # method for getting the PSF in photons
        self.image_pupil = False
        self.init_fosys()
        self.init_inwave()
        self.pupil_mask = self.PUPIL.get_transmission(self.inwave)>0
        _, wf = self.fosys.calc_psf(inwave=self.inwave, normalize=self.wf_norm, return_intermediates=False, return_final=True)
        image = wf[0].intensity
        image /= self.im_norm
        return image
    
    def calc_pupil(self):
        self.image_pupil = True
        self.init_fosys()
        self.init_inwave()
        self.pupil_mask = self.PUPIL.get_transmission(self.inwave)>0
        _, wf = self.fosys.calc_psf(inwave=self.inwave, normalize=self.wf_norm, return_intermediates=False, return_final=True)
        pupil_wf = wf[0].wavefront
        
        return pupil_wf
        