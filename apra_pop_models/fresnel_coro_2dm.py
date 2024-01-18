from .math_module import xp, _scipy, ensure_np_array
from . import utils
from . import imshows
from . import dm

import numpy as np
import scipy
import astropy.units as u
from astropy.io import fits
from pathlib import Path
import time

import poppy
from poppy.poppy_core import PlaneType
pupil = PlaneType.pupil
inter = PlaneType.intermediate
image = PlaneType.image

import os
print(os.path.dirname(__file__))
    
class CORO():

    def __init__(self,
                 wavelength=None, 
                 pupil_diam=9.5*u.mm,
                 lyot_diam=6.5*u.mm, 
                 npix=512, 
                 oversample=4,
                 npsf=100,
                 psf_pixelscale=5e-6*u.m/u.pix,
                 psf_pixelscale_lamD=None, 
                 detector_rotation=0, 
                 dm1_ref=np.zeros((34,34)),
                 dm2_ref=np.zeros((34,34)),
                 d_dm1_dm2=277*u.mm, 
                 Imax_ref=1,
                 TELEWFE=None,
                 FPM=None, 
                 use_lyot_stop=True,
                 use_opds=False,
                 use_aps=False,
                ):
        
        self.wavelength_c = 650e-9*u.m
        self.wavelength = self.wavelength_c if wavelength is None else wavelength
        self.pupil_diam = pupil_diam

        self.npix = npix
        self.oversample = oversample
        
        self.use_opds = use_opds
        self.use_aps = use_aps
        self.oap_diams = 15*u.mm
        # self.init_opds()

        self.fl_oap1 = 200*u.mm
        self.fl_oap2 = 200*u.mm
        self.fl_oap3 = 500*u.mm
        self.fl_oap4 = 400*u.mm
        self.fl_oap5 = 400*u.mm
        self.fl_oap6 = 400*u.mm
        self.fl_oap7 = 200*u.mm
        self.fl_oap8 = 200*u.mm
        self.fl_oap9 = 200*u.mm

        self.d_pupil_oap1 = self.fl_oap1
        self.d_oap1_ifp1 = self.fl_oap1
        self.d_ifp1_oap2 = self.fl_oap2
        self.d_oap2_dm1 = self.fl_oap2
        self.d_dm1_dm2 = d_dm1_dm2
        self.d_dm2_oap3 = self.fl_oap3 - d_dm1_dm2
        self.d_oap3_ifp2 = self.fl_oap3
        self.d_ifp2_oap4 = self.fl_oap4
        self.d_oap4_apodizer = self.fl_oap4
        self.d_apodizer_oap5 = self.fl_oap5
        self.d_oap5_fpm = self.fl_oap5
        self.d_fpm_oap6 = self.fl_oap6
        self.d_oap6_lyot = self.fl_oap6
        self.d_lyot_oap7 = self.fl_oap7
        self.d_oap7_fieldstop = self.fl_oap7
        self.d_fieldstop_oap8 = self.fl_oap8
        self.d_oap8_filter = self.fl_oap8
        self.d_filter_oap9 = self.fl_oap9
        self.d_oap9_image = self.fl_oap9
        
        # self.det_rotation = detector_rotation
        
        self.PUPIL = poppy.CircularAperture(radius=self.pupil_diam/2)
        wf = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength_c,
                                    npix=self.npix, oversample=1)
        self.pupil_mask = self.PUPIL.get_transmission(wf)>0
        
        self.TELEWFE = poppy.ScalarTransmission('WFE from Telescope') if TELEWFE is None else TELEWFE 

        self.FPM = poppy.ScalarTransmission('FPM') if FPM is None else FPM

        self.lyot_diam = lyot_diam
        self.use_lyot_stop = use_lyot_stop
        self.pupil_lyot_ratio = self.fl_oap4.to_value(u.mm)/self.fl_oap3.to_value(u.mm)
        if self.use_lyot_stop:
            self.um_per_lamD = (self.wavelength_c*self.fl_oap9/(self.lyot_diam)).to(u.um)
            self.LYOT = poppy.CircularAperture(radius=self.lyot_diam/2, name='Lyot Stop')
        else:
            self.um_per_lamD = (self.wavelength_c*self.fl_oap9/(self.pupil_diam*self.pupil_lyot_ratio)).to(u.um)
            self.LYOT = poppy.ScalarTransmission('Lyot Pupil')

        self.npsf = npsf
        if psf_pixelscale_lamD is None: # overrides psf_pixelscale this way
            self.psf_pixelscale = psf_pixelscale
            self.psf_pixelscale_lamD = self.psf_pixelscale.to_value(u.um/u.pix)/self.um_per_lamD.value
        else:
            self.psf_pixelscale_lamD = psf_pixelscale_lamD
            self.psf_pixelscale = self.psf_pixelscale_lamD * self.um_per_lamD/u.pix
        
        self.Imax_ref = Imax_ref

        self.init_dms()
        self.init_fosys()

        
    def getattr(self, attr):
        return getattr(self, attr)

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
    
    def init_fosys(self):
        oap1 = poppy.QuadraticLens(self.fl_oap1, name='OAP1')
        oap2 = poppy.QuadraticLens(self.fl_oap2, name='OAP2')
        oap3 = poppy.QuadraticLens(self.fl_oap3, name='OAP3')
        oap4 = poppy.QuadraticLens(self.fl_oap4, name='OAP4')
        oap5 = poppy.QuadraticLens(self.fl_oap5, name='OAP5')
        oap6 = poppy.QuadraticLens(self.fl_oap6, name='OAP6')
        oap7 = poppy.QuadraticLens(self.fl_oap7, name='OAP7')
        oap8 = poppy.QuadraticLens(self.fl_oap8, name='OAP8')
        oap9 = poppy.QuadraticLens(self.fl_oap9, name='OAP9')
        
        # define FresnelOpticalSystem and add optics
        fosys = poppy.FresnelOpticalSystem(pupil_diameter=self.pupil_diam, npix=self.npix, beam_ratio=1/self.oversample)
        
        fosys.add_optic(self.PUPIL)
        fosys.add_optic(self.TELEWFE)
        fosys.add_optic(oap1, self.d_pupil_oap1)
        fosys.add_optic(poppy.ScalarTransmission('IFP1'), self.d_oap1_ifp1)
        fosys.add_optic(oap2, self.d_ifp1_oap2)
        fosys.add_optic(self.DM1, self.d_oap2_dm1)
        fosys.add_optic(self.DM2, self.d_dm1_dm2)
        fosys.add_optic(oap3, self.d_dm2_oap3)
        fosys.add_optic(poppy.ScalarTransmission('IFP2'), self.d_oap3_ifp2)
        fosys.add_optic(oap4, self.d_ifp2_oap4)
        fosys.add_optic(poppy.ScalarTransmission('Apodizer Plane'), self.d_oap4_apodizer)
        fosys.add_optic(oap5, self.d_apodizer_oap5)
        fosys.add_optic(self.FPM, self.d_oap5_fpm)
        fosys.add_optic(oap6, self.d_fpm_oap6)
        fosys.add_optic(poppy.ScalarTransmission('Lyot Pupil'), self.d_oap6_lyot)
        fosys.add_optic(self.LYOT,)
        fosys.add_optic(oap7, self.d_lyot_oap7)
        fosys.add_optic(poppy.ScalarTransmission('Field Stop'), self.d_oap7_fieldstop)
        fosys.add_optic(oap8, self.d_fieldstop_oap8)
        fosys.add_optic(poppy.ScalarTransmission('Filter'), self.d_oap8_filter)
        fosys.add_optic(oap9, self.d_filter_oap9)
        fosys.add_optic(poppy.Detector(pixelscale=self.psf_pixelscale, fov_pixels=self.npsf, interp_order=3),
                        distance=self.fl_oap9)
        
        self.fosys = fosys

        return
        
    def init_opds(self, seeds=None):
        
        seed1, seed2, seed3, seed4, seed5 = (1,2,3,4,5)
        np.random.seed(1)
        rms_wfes = np.random.randn(6)*2*u.nm + 10*u.nm

        self.oap0_opd = poppy.StatisticalPSDWFE('OAP0 OPD', index=2.8, wfe=rms_wfes[0], radius=self.oap0_diam/2, seed=seed1)
        self.oap1_opd = poppy.StatisticalPSDWFE('OAP1 OPD', index=3.0, wfe=rms_wfes[1], radius=self.oap1_diam/2, seed=seed1)
        self.oap2_opd = poppy.StatisticalPSDWFE('OAP2 OPD', index=3.0, wfe=rms_wfes[2], radius=self.oap2_diam/2, seed=seed2)
        self.oap3_opd = poppy.StatisticalPSDWFE('OAP3 OPD', index=3.0, wfe=rms_wfes[3], radius=self.oap3_diam/2, seed=seed3)
        self.oap4_opd = poppy.StatisticalPSDWFE('OAP4 OPD', index=3.0, wfe=rms_wfes[4], radius=self.oap4_diam/2, seed=seed4)
        self.oap5_opd = poppy.StatisticalPSDWFE('OAP5 OPD', index=3.0, wfe=rms_wfes[5], radius=self.oap5_diam/2, seed=seed5)
        
    def init_inwave(self):
        inwave = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength,
                                        npix=self.npix, oversample=self.oversample)
        self.inwave = inwave
    
    def calc_wfs(self, quiet=False):
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        self.init_inwave()
        _, wfs = self.fosys.calc_psf(inwave=self.inwave, return_intermediates=True)
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
        
        return wfs
    
    def calc_psf(self, quiet=True): # method for getting the PSF in photons
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        self.init_fosys()
        self.init_inwave()
        self.pupil_mask = self.PUPIL.get_transmission(self.inwave)>0
        _, wf = self.fosys.calc_psf(inwave=self.inwave, return_final=True, return_intermediates=False)
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
            
        psf = wf[0].wavefront
        psf /= np.sqrt(self.im_norm)
        return psf
    
    def snap(self): # method for getting the PSF in photons
        fpwf = self.calc_psf()
        image = xp.abs(fpwf)**2

        return image
    
        