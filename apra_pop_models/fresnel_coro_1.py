from .math_module import xp, _scipy, ensure_np_array
from . import utils
from . import imshows
from . import dm
from . import props

import numpy as np
import scipy
import astropy.units as u
from astropy.io import fits
from pathlib import Path
import time
import copy

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
                 npsf=128, 
                 psf_pixelscale=5e-6*u.m/u.pix,
                 psf_pixelscale_lamD=None, 
                 dm1_ref=np.zeros((34,34)),
                 dm2_ref=np.zeros((34,34)),
                 d_dm1_dm2=283.569*u.mm, 
                 Imax_ref=1,
                 WFE=None,
                 use_opds=False,
                 use_aps=False,
                ):
        
        self.wavelength_c = 650e-9*u.m
        self.total_pupil_diam = 6.5*u.m
        self.pupil_diam = 9.6*u.mm
        self.lyot_pupil_diam = 400/500 * self.pupil_diam
        self.lyot_diam = 400/500 * 0.9 * self.pupil_diam
        
        self.wavelength = self.wavelength_c if wavelength is None else wavelength

        self.npix = 1000
        self.oversample = 2.048
        self.N = int(self.npix*self.oversample)
        self.Nfpm = 4096
        
        self.use_opds = use_opds
        self.use_aps = use_aps
        self.oap_diams = 25.4*u.mm
        self.init_opds()

        self.fl_oap1 = 250*u.mm
        self.fl_oap2 = 250*u.mm
        self.fl_oap3 = 500*u.mm
        self.fl_oap4 = 400*u.mm
        self.fl_oap5 = 400*u.mm
        self.fl_oap6 = 400*u.mm
        self.fl_oap7 = 250*u.mm
        self.fl_oap8 = 250*u.mm
        self.fl_oap9 = 150*u.mm

        self.d_pupil_oap1 = self.fl_oap1
        self.d_oap1_ifp1 = self.fl_oap1
        self.d_ifp1_oap2 = self.fl_oap2
        self.d_oap2_dm1 = self.fl_oap2
        self.d_dm1_dm2 = d_dm1_dm2
        self.d_dm2_oap3 = self.fl_oap3 - d_dm1_dm2
        self.d_oap3_ifp2 = self.fl_oap3
        self.d_ifp2_oap4 = self.fl_oap4
        self.d_oap4_lp1 = 200*u.mm
        self.d_lp1_qwp1 = 10*u.mm
        self.d_qwp1_apodizer = self.fl_oap4 - (self.d_lp1_qwp1 + self.d_oap4_lp1)
        self.d_apodizer_oap5 = self.fl_oap5
        self.d_oap5_fpm = self.fl_oap5
        self.d_fpm_oap6 = self.fl_oap6
        self.d_oap6_lyot = self.fl_oap6
        self.d_lyot_oap7 = self.fl_oap7
        self.d_oap7_fieldstop = self.fl_oap7
        self.d_fieldstop_oap8 = self.fl_oap8
        self.d_oap8_qwp2 = 100*u.mm
        self.d_qwp2_lp2 = 10*u.mm
        self.d_lp2_filter = self.fl_oap8 - (self.d_oap8_qwp2+self.d_qwp2_lp2)
        self.d_filter_oap9 = self.fl_oap9
        self.d_oap9_image = self.fl_oap9
        
        # self.det_rotation = detector_rotation
        
        APERTURE = xp.array(fits.getdata('aperture_gray_1000.fits'))
        self.APMASK = APERTURE>0
        WFE = xp.ones((self.npix,self.npix), dtype=complex) if WFE is None else WFE
        LYOT = xp.array(fits.getdata('lyot_90_gray_1000.fits'))
        
        self.APERTURE = poppy.ArrayOpticalElement(transmission=APERTURE, 
                                                  pixelscale=self.pupil_diam/(self.npix*u.pix),
                                                  name='Pupil')
        self.WFE = poppy.ArrayOpticalElement(transmission=xp.abs(WFE), 
                                             opd=xp.angle(WFE)*self.wavelength_c.to_value(u.m)/(2*np.pi),
                                             pixelscale=self.pupil_diam/(self.npix*u.pix),
                                             name='Pupil WFE')
        self.LYOT = poppy.ArrayOpticalElement(transmission=LYOT, 
                                              pixelscale=self.lyot_pupil_diam/(self.npix*u.pix),
                                              name='Lyot Stop (Pupil)')

        self.um_per_lamD = (self.wavelength_c*self.fl_oap9/(self.lyot_diam)).to(u.um)

        self.npsf = npsf
        self.psf_pixelscale = 5e-6*u.m/u.pix
        self.psf_pixelscale_lamD = self.psf_pixelscale.to_value(u.um/u.pix)/self.um_per_lamD.value
        if psf_pixelscale_lamD is None: # overrides psf_pixelscale this way
            self.psf_pixelscale = psf_pixelscale
            self.psf_pixelscale_lamD = self.psf_pixelscale.to_value(u.um/u.pix)/self.um_per_lamD.value
        else:
            self.psf_pixelscale_lamD = psf_pixelscale_lamD
            self.psf_pixelscale = self.psf_pixelscale_lamD * self.um_per_lamD/u.pix
        
        self.Imax_ref = Imax_ref

        self.use_fpm = False

        self.init_dms()
        self.dm1_ref = dm1_ref
        self.dm2_ref = dm2_ref
        self.set_dm1(dm1_ref)
        self.set_dm2(dm2_ref)
        self.init_fosys()

        self.return_lyot = False

    
    def getattr(self, attr):
        return getattr(self, attr)

    def init_dms(self):
        act_spacing = 300e-6*u.m
        pupil_pxscl = self.pupil_diam.to_value(u.m)/self.npix
        sampling = act_spacing.to_value(u.m)/pupil_pxscl
        print('influence function sampling', sampling)
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
    
    def init_opds(self, seeds=None):
        np.random.seed(1)

        self.opd_index = 2.75
        self.pol_opd_index = 3.0
        rms_wfes = np.random.randn(20)*2*u.nm + 8*u.nm
        seeds = np.linspace(1,20,20).astype(int)

        self.oap1_opd = poppy.StatisticalPSDWFE('OAP1 OPD', index=self.opd_index, wfe=rms_wfes[1], radius=self.oap_diams/2, seed=seeds[1])
        self.oap2_opd = poppy.StatisticalPSDWFE('OAP2 OPD', index=self.opd_index, wfe=rms_wfes[2], radius=self.oap_diams/2, seed=seeds[2])
        self.oap3_opd = poppy.StatisticalPSDWFE('OAP3 OPD', index=self.opd_index, wfe=rms_wfes[3], radius=self.oap_diams/2, seed=seeds[3])
        self.oap4_opd = poppy.StatisticalPSDWFE('OAP4 OPD', index=self.opd_index, wfe=rms_wfes[4], radius=self.oap_diams/2, seed=seeds[4])
        self.oap5_opd = poppy.StatisticalPSDWFE('OAP5 OPD', index=self.opd_index, wfe=rms_wfes[5], radius=self.oap_diams/2, seed=seeds[5])
        self.oap6_opd = poppy.StatisticalPSDWFE('OAP6 OPD', index=self.opd_index, wfe=rms_wfes[6], radius=self.oap_diams/2, seed=seeds[6])
        self.oap7_opd = poppy.StatisticalPSDWFE('OAP7 OPD', index=self.opd_index, wfe=rms_wfes[7], radius=self.oap_diams/2, seed=seeds[7])
        self.oap8_opd = poppy.StatisticalPSDWFE('OAP8 OPD', index=self.opd_index, wfe=rms_wfes[8], radius=self.oap_diams/2, seed=seeds[8])
        self.oap9_opd = poppy.StatisticalPSDWFE('OAP9 OPD', index=self.opd_index, wfe=rms_wfes[9], radius=self.oap_diams/2, seed=seeds[9])

        self.lp1_opd = poppy.StatisticalPSDWFE('LP1 OPD', index=self.pol_opd_index, wfe=rms_wfes[10], radius=self.oap_diams/2, seed=seeds[10])
        self.qwp1_opd = poppy.StatisticalPSDWFE('QWP1 OPD', index=self.pol_opd_index, wfe=rms_wfes[11], radius=self.oap_diams/2, seed=seeds[11])
        self.qwp2_opd = poppy.StatisticalPSDWFE('QWP2 OPD', index=self.pol_opd_index, wfe=rms_wfes[12], radius=self.oap_diams/2, seed=seeds[12])
        self.lp2_opd = poppy.StatisticalPSDWFE('LP2 OPD', index=self.pol_opd_index, wfe=rms_wfes[13], radius=self.oap_diams/2, seed=seeds[13])
        self.filter_opd = poppy.StatisticalPSDWFE('Filter OPD', index=self.pol_opd_index, wfe=rms_wfes[14], radius=self.oap_diams/2, seed=seeds[14])

    def init_fosys(self):
        oap1 = poppy.QuadraticLens(self.fl_oap1, name='OAP1')
        oap2 = poppy.QuadraticLens(self.fl_oap2, name='OAP2')
        oap3 = poppy.QuadraticLens(self.fl_oap3, name='OAP3')
        oap4 = poppy.QuadraticLens(self.fl_oap4, name='OAP4')
        oap5 = poppy.QuadraticLens(self.fl_oap5, name='OAP5')
        # oap6 = poppy.QuadraticLens(self.fl_oap6, name='OAP6')
        oap7 = poppy.QuadraticLens(self.fl_oap7, name='OAP7')
        oap8 = poppy.QuadraticLens(self.fl_oap8, name='OAP8')
        oap9 = poppy.QuadraticLens(self.fl_oap9, name='OAP9')
        
        # define FresnelOpticalSystem and add optics
        fosys = poppy.FresnelOpticalSystem(pupil_diameter=self.pupil_diam, npix=self.npix, beam_ratio=1/self.oversample)
        
        fosys.add_optic(self.APERTURE)
        fosys.add_optic(self.WFE)
        fosys.add_optic(oap1, self.d_pupil_oap1)
        if self.use_opds: fosys.add_optic(self.oap1_opd)
        fosys.add_optic(poppy.ScalarTransmission('IFP1'), self.d_oap1_ifp1)
        fosys.add_optic(oap2, self.d_ifp1_oap2)
        if self.use_opds: fosys.add_optic(self.oap2_opd)
        fosys.add_optic(self.DM1, self.d_oap2_dm1)
        fosys.add_optic(self.DM2, self.d_dm1_dm2)
        fosys.add_optic(oap3, self.d_dm2_oap3)
        if self.use_opds: fosys.add_optic(self.oap3_opd)
        fosys.add_optic(poppy.ScalarTransmission('IFP2'), self.d_oap3_ifp2)
        fosys.add_optic(oap4, self.d_ifp2_oap4)
        if self.use_opds: fosys.add_optic(self.oap4_opd)
        if self.use_opds: 
            fosys.add_optic(self.lp1_opd, self.d_oap4_lp1)
        else: 
            fosys.add_optic(poppy.ScalarTransmission('LP1'), self.d_oap4_lp1)
        if self.use_opds:
            fosys.add_optic(self.qwp1_opd, self.d_lp1_qwp1)
        else: 
            fosys.add_optic(poppy.ScalarTransmission('QWP1'), self.d_lp1_qwp1)
        fosys.add_optic(poppy.ScalarTransmission('Apodizer Pupil Plane'), self.d_qwp1_apodizer)
        fosys.add_optic(oap5, self.d_apodizer_oap5)
        if self.use_opds: fosys.add_optic(self.oap5_opd)
        fosys.add_optic(poppy.ScalarTransmission(name='FPM place-holder'), self.d_oap5_fpm)
        self.fosys_pupil_fpm = fosys

        fosys2 = poppy.FresnelOpticalSystem(pupil_diameter=self.lyot_pupil_diam, npix=self.npix, beam_ratio=1/self.oversample)

        fosys2.add_optic(poppy.ScalarTransmission('Lyot Pupil'))
        fosys2.add_optic(self.LYOT,)
        fosys2.add_optic(oap7, self.d_lyot_oap7)
        if self.use_opds: fosys.add_optic(self.oap7_opd)
        fosys2.add_optic(poppy.ScalarTransmission('Field Stop'), self.d_oap7_fieldstop)
        fosys2.add_optic(oap8, self.d_fieldstop_oap8)
        if self.use_opds: fosys.add_optic(self.oap8_opd)
        if self.use_opds: 
            fosys2.add_optic(self.qwp2_opd, self.d_oap8_qwp2)
        else:
            fosys2.add_optic(poppy.ScalarTransmission('QWP2'), self.d_oap8_qwp2)
        if self.use_opds:
            fosys2.add_optic(self.lp2_opd, self.d_qwp2_lp2)
        else:
            fosys2.add_optic(poppy.ScalarTransmission('LP2'), self.d_qwp2_lp2)
        fosys2.add_optic(poppy.ScalarTransmission('Filter'), self.d_lp2_filter)
        if self.use_opds: fosys2.add_optic(self.filter_opd)
        fosys2.add_optic(oap9, self.d_filter_oap9)
        if self.use_opds: fosys.add_optic(self.oap9_opd)
        fosys2.add_optic(poppy.Detector(pixelscale=self.psf_pixelscale, fov_pixels=self.npsf, interp_order=3),
                         distance=self.fl_oap9)

        self.fosys_lyot_image = fosys2

        return
        
    def init_inwave(self):
        inwave = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength,
                                        npix=self.npix, oversample=self.oversample)
        self.inwave = inwave
    
    def calc_wfs(self, quiet=False):
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        self.init_fosys()
        self.init_inwave()
        _, wfs1 = self.fosys_pupil_fpm.calc_psf(inwave=self.inwave, normalize='none', return_intermediates=True)
        fpm_wf = copy.deepcopy(wfs1[-1])

        lyot_inwave = poppy.FresnelWavefront(beam_radius=self.lyot_pupil_diam/2, wavelength=self.wavelength,
                                             npix=self.npix, oversample=self.oversample,)
        lyot_wfarr = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(fpm_wf.wavefront))) * self.npix * 2
        if self.use_fpm:
            lyot_wfarr = utils.pad_or_crop(lyot_wfarr, self.npix)
            lyot_wfarr = props.apply_vortex(lyot_wfarr, Nfpm=self.Nfpm, N=self.N, plot=False) # apply the vortex mask if using the FPM

        if self.use_opds:
            # back propagate the lyot pupil wavefront to the OAP6 plane
            lyot_wfarr = props.ang_spec(lyot_wfarr, self.wavelength, -self.d_oap6_lyot, self.lyot_pupil_diam/(self.npix*u.pix))
            # apply the OAP6 surface roughness phasor
            lyot_wfarr *= self.oap6_opd.get_phasor(lyot_inwave)
            # propagate back to the Lyot pupil plane
            lyot_wfarr = props.ang_spec(lyot_wfarr, self.wavelength, self.d_oap6_lyot, self.lyot_pupil_diam/(self.npix*u.pix))

        lyot_inwave.wavefront = copy.deepcopy(lyot_wfarr)

        # imshows.imshow2(lyot_inwave.amplitude, lyot_inwave.phase, pxscl=lyot_inwave.pixelscale.to(u.mm/u.pix), npix=self.npix)
        _, wfs2 = self.fosys_lyot_image.calc_psf(normalize='none', return_intermediates=True,
                                                 inwave=lyot_inwave,
                                                 )
        wfs2[-1].wavefront /= xp.sqrt(self.Imax_ref)
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))

        return wfs1+wfs2
    
    def calc_wf(self): # method for getting the PSF in photons
        self.init_fosys()
        self.init_inwave()
        _, wfs1 = self.fosys_pupil_fpm.calc_psf(inwave=self.inwave, normalize='none', return_final=True, return_intermediates=False)
        fpm_wf = copy.deepcopy(wfs1[-1])

        lyot_inwave = poppy.FresnelWavefront(beam_radius=self.lyot_pupil_diam/2, wavelength=self.wavelength,
                                             npix=self.npix, oversample=self.oversample,)
        lyot_wfarr = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(fpm_wf.wavefront))) * self.npix * 2

        if self.return_lyot:
            lyot_wfarr = utils.pad_or_crop(lyot_wfarr, self.npix)
            lyot_amp = xp.abs(lyot_wfarr) * self.APERTURE.amplitude
            lyot_opd = xp.angle(lyot_wfarr) * self.wavelength.to_value(u.m)/(2*np.pi) * self.APERTURE.amplitude
            return lyot_amp, lyot_opd
        
        if self.use_fpm:
            lyot_wfarr = props.apply_vortex(lyot_wfarr, Nfpm=self.Nfpm, N=self.N, plot=False) # apply the vortex mask if using the FPM

        if self.use_opds:
            # back propagate the lyot pupil wavefront to the OAP6 plane
            lyot_wfarr = props.ang_spec(lyot_wfarr, self.wavelength, -self.d_oap6_lyot, self.lyot_pupil_diam/(self.npix*u.pix))
            # apply the OAP6 surface roughness phasor
            lyot_wfarr *= self.oap6_opd.get_phasor(lyot_inwave)
            # propagate back to the Lyot pupil plane
            lyot_wfarr = props.ang_spec(lyot_wfarr, self.wavelength, self.d_oap6_lyot, self.lyot_pupil_diam/(self.npix*u.pix))
        
        lyot_inwave.wavefront = copy.deepcopy(lyot_wfarr)

        _, wfs2 = self.fosys_lyot_image.calc_psf(normalize='none', return_final=True, return_intermediates=False,
                                                 inwave=lyot_inwave,
                                                 )
        
        return wfs2[-1].wavefront/xp.sqrt(self.Imax_ref)
    
    def snap(self): # method for getting the final image
        return xp.abs(self.calc_wf())**2
    
        