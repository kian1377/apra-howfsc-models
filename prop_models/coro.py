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

import misc

class SCOOBM():

    def __init__(self, 
                 wavelength=None, 
                 npix=128, 
                 oversample=2048/128,
                 npsf=400,
                 psf_pixelscale=4.63e-6*u.m/u.pix,
                 psf_pixelscale_lamD=None,
                 texp=0.00001, 
                 normalization=1,
                 interp_order=3,
                 det_rotation=0,
                 offset=(0,0),  
                 use_opds=False,
                 use_aps=False,
                 fpm_defocus=0,
                 dm_ref=np.zeros((34,34)),
                 dm_inf=None, # defaults to bmc_inf.fits
                 im_norm=None,
                 OPD=None,
                 RETRIEVED=None,
                 FPM=None,
                 LYOT=None):
        
        poppy.accel_math.update_math_settings()
        
        self.is_model = True
        
        self.wavelength_c = 632.8e-9*u.m
        if wavelength is None: 
            self.wavelength = self.wavelength_c
        else: 
            self.wavelength = wavelength
        
        self.npix = npix
        self.oversample = oversample
        
        self.npsf = npsf
        if psf_pixelscale_lamD is None: # overrides psf_pixelscale this way
            self.psf_pixelscale = psf_pixelscale
#             self.psf_pixelscale_lamD = (1/2.75) * self.psf_pixelscale.to(u.m/u.pix).value/4.63e-6
            self.psf_pixelscale_lamD = (1/(5)) * self.psf_pixelscale.to(u.m/u.pix).value/4.63e-6
        else:
            self.psf_pixelscale_lamD = psf_pixelscale_lamD
            self.psf_pixelscale = 4.63e-6*u.m/u.pix / self.psf_pixelscale_lamD/(1/2.75)
            
        self.interp_order = interp_order
        self.det_rotation = det_rotation
        self.normalization = normalization
        
        self.dm_inf = str(esc_coro_suite.data_dir/'bmc_inf.fits') if dm_inf is None else dm_inf
        
        self.offset = offset
        self.use_opds = use_opds
        self.use_aps = use_aps
        self.fpm_defocus = fpm_defocus
        
        self.OPD = poppy.ScalarTransmission(name='OPD Place-holder') if OPD is None else OPD
        self.RETRIEVED = poppy.ScalarTransmission(name='Phase Retrieval Place-holder') if RETRIEVED is None else RETRIEVED
        self.FPM = poppy.ScalarTransmission(name='FPM Place-holder') if FPM is None else FPM
        self.LYOT = poppy.ScalarTransmission(name='Lyot Stop Place-holder') if LYOT is None else LYOT
        
        self.texp = texp # between 0.1ms (0.0001s) and 0.01s
        
        self.im_norm = im_norm
        
        self.init_dm()
        self.dm_ref = dm_ref
        if self.use_opds:
            self.init_opds()
        
    def getattr(self, attr):
        return getattr(self, attr)
    
    def copy_model_settings(self, nactors=1):
        settings = []
        for i in range(nactors):
            settings.append({'wavelength':self.wavelength, 
                              'npix':self.npix, 
                                'oversample':self.oversample, 
                                'npsf':self.npsf, 
                                'psf_pixelscale':self.psf_pixelscale,   
                                'use_opds':self.use_opds, 
                                'fpm_defocus':self.fpm_defocus,
                                'FPM':copy.copy(self.FPM), 
                                'LYOT':copy.copy(self.LYOT)})
        return settings

    def init_dm(self):
        self.Nact = 34
        self.act_spacing = 300e-6*u.m
        self.dm_active_diam = 10.2*u.mm
        self.dm_full_diam = 11.1*u.mm
        
        self.full_stroke = 1.5e-6*u.m
        
        self.dm_mask = np.ones((self.Nact,self.Nact))
        xx = (np.linspace(0, self.Nact-1, self.Nact) - self.Nact/2 + 1/2) * self.act_spacing.to(u.mm).value*2
        x,y = np.meshgrid(xx,xx)
        r = np.sqrt(x**2 + y**2)
        self.dm_mask[r>10.5] = 0 # had to set the threshold to 10.5 instead of 10.2 to include edge actuators
        
        self.dm_zernikes = poppy.zernike.arbitrary_basis(cp.array(self.dm_mask), nterms=15, outside=0).get()
        
        bad_acts = [(21,25)]
        self.bad_acts = []
        for act in bad_acts:
            self.bad_acts.append(act[1]*self.Nact + act[0])

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
        wf = poppy.FresnelWavefront(beam_radius=self.dm_active_diam/2, npix=256, oversample=1)
        misc.myimshow2(self.get_dm(), self.DM.get_opd(wf), 'DM Command', 'DM Surface',
                       pxscl2=wf.pixelscale.to(u.mm/u.pix))
    
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
        
        oap0 = poppy.QuadraticLens(fl_oap0, name='OAP0')
        
        
        # define FresnelOpticalSystem and add optics
        self.pupil_diam = 6.75*u.mm
        fosys = poppy.FresnelOpticalSystem(pupil_diameter=self.pupil_diam, npix=self.npix, beam_ratio=1/self.oversample)
        

        
        self.fosys = fosys
        
    def init_opds(self):
        seed = 123456

        self.oap0_opd = poppy.StatisticalPSDWFE('OAP0 OPD', index=3.0, wfe=10*u.nm, radius=oap0_diam/2, see=seed)
    
    def init_inwave(self):
        self.pupil_diam = 6.8*u.mm
        
        inwave = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength,
                                        npix=self.npix, oversample=self.oversample)
        self.inwave = inwave
    
    def calc_wfs(self, quiet=False):
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        self.init_fosys()
        self.init_inwave()
        psf, wfs = self.fosys.calc_psf(inwave=self.inwave, return_intermediates=True)

        if not self.use_opds and not self.use_aps:
            self.fosys_names = ['Pupil Stop', 'Injected OPD',
                                'Flat 1', 'OAP1', 'OAP2',
                                'DM', 'Retrieval Data', 
                                'OAP3',
                                'FPM', 
                                # 'Singularity', 
                                'Flat 2', 'Lens',  
                                'Lyot Plane', 'Lyot Stop',
                                'SciCam Lens', 'Image Plane']
        elif self.use_opds and not self.use_aps:
            self.fosys_names = ['Pupil Stop', 'Flat 1 OPD', 'Injected OPD', 'Flat 1', 'Flat 2 OPD', 
                                'OAP1', 'OAP1 OPD', 'OAP2', 'OAP2 OPD',
                                'DM', 'OAP3', 'OAP3 OPD',
                                'FPM', 'Flat 2', 'Lens',  
                                'Lyot Plane', 'Lyot Stop',
                                'SciCam Lens', 'Image Plane']
            
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
        
        return wfs
    
    def calc_psf(self, quiet=True): # method for getting the PSF in photons
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        self.init_fosys()
        self.init_inwave()
        psf, wfs = self.fosys.calc_psf(inwave=self.inwave, return_final=True, return_intermediates=False)

        if self.im_norm is not None:
            wfs[-1].wavefront *= np.sqrt(self.im_norm)/abs(wfs[-1].wavefront).max()
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
        
        wavefront = wfs[-1].wavefront
        wavefront_r = cupyx.scipy.ndimage.rotate(cp.real(wavefront), angle=-self.det_rotation, reshape=False, order=0)
        wavefront_i = cupyx.scipy.ndimage.rotate(cp.imag(wavefront), angle=-self.det_rotation, reshape=False, order=0)
        
        wfs[-1].wavefront = wavefront_r + 1j*wavefront_i
        
        resamped_wf = self.interp_wf(wfs[-1])
        
        resamped_wf /= np.sqrt(self.normalization)

        return resamped_wf.get()
    
    def snap(self): # method for getting the PSF in photons
        self.init_fosys()
        self.init_inwave()
        psf, wfs = self.fosys.calc_psf(inwave=self.inwave, return_intermediates=False, return_final=True)
        
        wavefront = wfs[-1].wavefront
        wavefront_r = cupyx.scipy.ndimage.rotate(cp.real(wavefront), angle=-self.det_rotation, reshape=False, order=0)
        wavefront_i = cupyx.scipy.ndimage.rotate(cp.imag(wavefront), angle=-self.det_rotation, reshape=False, order=0)
        
        wfs[-1].wavefront = wavefront_r + 1j*wavefront_i
        
        resamped_wf = self.interp_wf(wfs[-1])
        
        image = (cp.abs(resamped_wf)**2).get()
        
        image /= self.normalization
        return image
    
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