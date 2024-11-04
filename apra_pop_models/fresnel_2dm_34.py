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

from scipy.signal import windows

import os
data_path = Path(os.path.dirname(__file__))
print(os.path.dirname(__file__))

class CORO():

    def __init__(self,
                 dm1_ref=xp.zeros((34,34)),
                 dm2_ref=xp.zeros((34,34)),
                 d_dm1_dm2=283.569*u.mm, 
                 WFE=None,
                 use_opds=False,
                 use_aps=False,
                ):
        
        self.wavelength_c = 650e-9
        self.wavelength = self.wavelength_c
        self.bandpasses = np.array([self.wavelength_c])

        self.total_pupil_diam = 6.5*u.m
        self.pupil_diam = 9.6*u.mm
        self.lyot_pupil_diam = 400/500 * self.pupil_diam
        self.lyot_diam = 400/500 * 0.9 * self.pupil_diam
        # self.exit_pupil_diam = 250/400 * self.lyot_diam

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
        
        self.npix = 1000
        self.oversample = 2.048
        self.N = int(self.npix*self.oversample)
        self.npsf = 100
        self.psf_pixelscale = 5e-6*u.m/u.pix
        self.um_per_lamD = (self.wavelength_c*u.m * self.fl_oap9/(self.lyot_diam)).to(u.um)
        self.psf_pixelscale_lamDc = self.psf_pixelscale.to_value(u.um/u.pix)/self.um_per_lamD.value

        APERTURE = xp.array(fits.getdata(data_path/'aperture_gray_1000.fits'))
        self.APMASK = APERTURE>0
        WFE = xp.ones((self.npix,self.npix), dtype=complex) if WFE is None else WFE
        LYOT = xp.array(fits.getdata(data_path/'lyot_90_gray_1000.fits'))
        
        self.APERTURE = poppy.ArrayOpticalElement(transmission=APERTURE, 
                                                  pixelscale=self.pupil_diam/(self.npix*u.pix),
                                                  name='Pupil')
        self.WFE = poppy.ArrayOpticalElement(transmission=xp.abs(WFE), 
                                             opd=xp.angle(WFE)*self.wavelength_c/(2*np.pi),
                                             pixelscale=self.pupil_diam/(self.npix*u.pix),
                                             name='Pupil WFE')
        self.LYOT = poppy.ArrayOpticalElement(transmission=LYOT, 
                                              pixelscale=self.lyot_pupil_diam/(self.npix*u.pix),
                                              name='Lyot Stop (Pupil)')
        self.DETECTOR = poppy.Detector(pixelscale=self.psf_pixelscale, fov_pixels=self.npsf, interp_order=5, name='Scicam (FP)')
        self.Imax_ref = 1

        self.return_pupil = False
        self.use_opds = use_opds
        self.use_aps = use_aps
        self.oap_diams = 25.4*u.mm
        self.init_opds()

        self.use_vortex = False
        self.plot_vortex = False
        self.oversample_vortex = 4.096
        self.N_vortex_lres = int(self.npix*self.oversample_vortex)
        self.vortex_win_diam = 30 # diameter of the window to apply with the vortex model
        self.lres_sampling = 1/self.oversample_vortex # low resolution sampling in lam/D per pixel
        self.lres_win_size = int(self.vortex_win_diam/self.lres_sampling)
        w1d = xp.array(windows.tukey(self.lres_win_size, 1, False))
        self.lres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_lres)
        self.vortex_lres = props.make_vortex_phase_mask(self.N_vortex_lres)

        y,x = (xp.indices((self.N_vortex_lres, self.N_vortex_lres)) - self.N_vortex_lres//2)*self.lres_sampling
        r = xp.sqrt(x**2 + y**2)

        self.hres_sampling = 0.025 # lam/D per pixel; this value is chosen empirically
        self.N_vortex_hres = int(np.round(self.vortex_win_diam/self.hres_sampling))
        self.hres_win_size = int(self.vortex_win_diam/self.hres_sampling)
        w1d = xp.array(windows.tukey(self.hres_win_size, 1, False))
        self.hres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_hres)
        self.vortex_hres = props.make_vortex_phase_mask(self.N_vortex_hres)

        y,x = (xp.indices((self.N_vortex_hres, self.N_vortex_hres)) - self.N_vortex_hres//2)*self.hres_sampling
        r = xp.sqrt(x**2 + y**2)
        self.hres_dot_mask = r>=0.3/2

        self.Nact = 34
        self.act_spacing = 300e-6*u.m
        pupil_pxscl = self.pupil_diam.to_value(u.m)/self.npix
        sampling = self.act_spacing.to_value(u.m)/pupil_pxscl
        inf_fun = dm.make_gaussian_inf_fun(act_spacing=300e-6*u.m, sampling=sampling, coupling=0.15,  Nact=self.Nact + 2)
        self.DM1 = dm.DeformableMirror(inf_fun=inf_fun, inf_sampling=sampling, name='DM1 (pupil)',)
        self.DM2 = dm.DeformableMirror(inf_fun=inf_fun, inf_sampling=sampling, name='DM2 ',)
        self.Nacts = self.DM1.Nacts
        self.act_spacing = self.DM1.act_spacing
        self.dm_active_diam = self.DM1.active_diam
        self.dm_mask = self.DM1.dm_mask
        self.dm1_ref = dm1_ref
        self.set_dm1(dm1_ref)
        self.dm2_ref = dm2_ref
        self.set_dm2(dm2_ref)
    
    def getattr(self, attr):
        return getattr(self, attr)
    
    def reset_dms(self):
        self.set_dm1(self.dm1_ref)
        self.set_dm2(self.dm2_ref)

    def zero_dms(self):
        self.set_dm1(xp.zeros((self.Nact,self.Nact)))
        self.set_dm2(xp.zeros((self.Nact,self.Nact)))
    
    def set_dm1(self, command):
        self.DM1.command = command
        
    def add_dm1(self, command):
        self.DM1.command += command
        
    def get_dm1(self):
        return self.DM1.command
    
    def set_dm2(self, command):
        self.DM2.command = command
        
    def add_dm2(self, command):
        self.DM2.command += command
        
    def get_dm2(self):
        return self.DM2.command
    
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
        oap6 = poppy.QuadraticLens(self.fl_oap6, name='OAP6')
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
        fosys.add_optic(poppy.ScalarTransmission('LP1'), self.d_oap4_lp1)
        if self.use_opds: fosys.add_optic(self.lp1_opd)
        fosys.add_optic(poppy.ScalarTransmission('QWP1'), self.d_lp1_qwp1)
        if self.use_opds: fosys.add_optic(self.qwp1_opd)
        fosys.add_optic(poppy.ScalarTransmission('Apodizer Pupil Plane'), self.d_qwp1_apodizer)
        if self.return_pupil: 
            fosys.add_optic(self.oap5_opd, self.d_apodizer_oap5)
            fosys.add_optic(poppy.ScalarTransmission(name='Pre-FPM Pupil'), distance= -self.d_apodizer_oap5)
            return fosys
        fosys.add_optic(oap5, self.d_apodizer_oap5)
        if self.use_opds: fosys.add_optic(self.oap5_opd)
        fosys.add_optic(poppy.ScalarTransmission(name='FPM place-holder'), self.d_oap5_fpm)
        fosys_to_fpm = fosys

        fosys2 = poppy.FresnelOpticalSystem(npix=self.npix, beam_ratio=1/self.oversample)
        fosys2.add_optic(poppy.ScalarTransmission(name='Post-FPM'))
        fosys2.add_optic(oap6, distance=self.d_fpm_oap6)
        fosys2.add_optic(poppy.ScalarTransmission('Lyot Pupil'), distance=self.d_oap6_lyot)
        fosys2.add_optic(self.LYOT,)
        fosys2.add_optic(oap7, self.d_lyot_oap7)
        if self.use_opds: fosys.add_optic(self.oap7_opd)
        fosys2.add_optic(poppy.ScalarTransmission('IFP3: Field Stop'), self.d_oap7_fieldstop)
        fosys2.add_optic(oap8, self.d_fieldstop_oap8)
        if self.use_opds: fosys.add_optic(self.oap8_opd)
        fosys2.add_optic(poppy.ScalarTransmission('QWP2'), self.d_oap8_qwp2)
        if self.use_opds: fosys2.add_optic(self.qwp2_opd)
        fosys2.add_optic(poppy.ScalarTransmission('LP2'), self.d_qwp2_lp2)
        if self.use_opds: fosys2.add_optic(self.lp2_opd)
        fosys2.add_optic(poppy.ScalarTransmission('Filter (pupil)'), self.d_lp2_filter)
        if self.use_opds: fosys2.add_optic(self.filter_opd)
        fosys2.add_optic(oap9, self.d_filter_oap9)
        if self.use_opds: fosys.add_optic(self.oap9_opd)
        fosys2.add_optic(self.DETECTOR, distance=self.fl_oap9)
        fosys_to_image = fosys2

        return fosys_to_fpm, fosys_to_image
        
    def init_inwave(self):
        inwave = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, wavelength=self.wavelength*u.m,
                                        npix=self.npix, oversample=self.oversample)
        return inwave
    
    def apply_vortex(self, fpwf):
        vpup_wf = props.ifft(fpwf)
        if self.plot_vortex: 
            imshows.imshow2(xp.abs(vpup_wf), xp.angle(vpup_wf), 'Virtual Pupil Amplitude', 'Virtual Pupil Phase',
                            npix=1.5*self.npix,
                            )

        lres_wf = utils.pad_or_crop(vpup_wf, self.N_vortex_lres) # pad to the larger array for the low res propagation
        fp_wf_lres = props.fft(lres_wf)
        fp_wf_lres *= self.vortex_lres * (1 - self.lres_window) # apply low res (windowed) FPM
        pupil_wf_lres = props.ifft(fp_wf_lres)
        pupil_wf_lres = utils.pad_or_crop(pupil_wf_lres, self.N,)
        if self.plot_vortex: 
            imshows.imshow2(xp.abs(pupil_wf_lres), xp.angle(pupil_wf_lres), 'FFT Pupil Amplitude', 'FFT Pupil Phase', 
                            npix=1.5*self.npix,
                            )

        fp_wf_hres = props.mft_forward(vpup_wf, self.npix, self.N_vortex_hres, self.hres_sampling, convention='-', fp_centering='odd')
        fp_wf_hres *= self.vortex_hres * self.hres_window * self.hres_dot_mask # apply high res (windowed) FPM
        pupil_wf_hres = props.mft_reverse(fp_wf_hres, self.hres_sampling, self.npix, self.N, convention='+', fp_centering='odd')
        if self.plot_vortex: 
            imshows.imshow2(xp.abs(pupil_wf_hres), xp.angle(pupil_wf_hres), 'MFT Pupil Amplitude', 'MFT Pupil Phase',
                            npix=1.5*self.npix,
                            )

        post_vortex_vpup_wf = (pupil_wf_lres + pupil_wf_hres)
        if self.plot_vortex: 
            imshows.imshow2(xp.abs(post_vortex_vpup_wf), xp.angle(post_vortex_vpup_wf), 'Total Pupil Amplitude', 'Total Pupil Phase',
                            npix=1.5*self.npix,
                            )
        post_vortex_fpwf = props.fft(post_vortex_vpup_wf)

        return post_vortex_fpwf

    def calc_wfs(self, quiet=False):
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}m.'.format(self.wavelength))
        self.return_pupil = False
        fosys_to_fpm, fosys_to_scicam = self.init_fosys()
        ep_inwave = self.init_inwave()
        _, wfs_to_fpm = fosys_to_fpm.calc_psf(inwave=ep_inwave, normalize='none', return_intermediates=True)
        fpm_inwave = copy.copy(wfs_to_fpm[-1])
        if self.use_vortex: 
            post_fpm_wf = self.apply_vortex(copy.copy(fpm_inwave.wavefront))
            fpm_inwave.wavefront = copy.copy(post_fpm_wf)
        _, wfs_to_scicam = fosys_to_scicam.calc_psf(inwave=fpm_inwave, normalize='none', return_intermediates=True)
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))

        all_wfs = wfs_to_fpm + wfs_to_scicam
        return all_wfs
    
    def calc_wf(self): 
        self.return_pupil=False
        fosys_to_fpm, fosys_to_scicam = self.init_fosys()
        ep_inwave = self.init_inwave()
        _, fpm_wf = fosys_to_fpm.calc_psf(inwave=ep_inwave, normalize='none', return_final=True)
        fpm_inwave = copy.copy(fpm_wf[0])
        # print(fpm_inwave.wavelength)
        if self.use_vortex: 
            post_fpm_wf = self.apply_vortex(copy.copy(fpm_inwave.wavefront))
            fpm_inwave.wavefront = copy.copy(post_fpm_wf)
        _, final_wf = fosys_to_scicam.calc_psf(inwave=fpm_inwave, normalize='none', return_final=True)
        return final_wf[0].wavefront/xp.sqrt(self.Imax_ref)
    
    def snap(self, bp=0): # method for getting the final image
        if bp==0:
            waves = self.bandpasses.flatten()
        else:
            Nwaves_per_bp = self.bandpasses.shape[1]
            waves = self.bandpasses[bp-1, :]

        Nwaves = len(waves)
        im = 0.0
        for i in range(Nwaves):
            self.wavelength = waves[i]
            fpwf = self.calc_wf()
            im += xp.abs( fpwf )**2 / Nwaves
        return im
    
    def calc_pupil(self):
        self.wavelength = self.wavelength_c
        self.return_pupil = True
        fosys_to_pupil = self.init_fosys()
        ep_inwave = self.init_inwave()
        _, pupil_wf = fosys_to_pupil.calc_psf(inwave=ep_inwave, normalize='none', return_final=True)
        pupil = utils.pad_or_crop(pupil_wf[-1].wavefront, self.npix)
        # print(pupil_wf[0].wavelength)
        amp = xp.abs(pupil) * self.APERTURE.amplitude
        phs = xp.angle(pupil) * self.APERTURE.amplitude
        opd = phs * self.wavelength_c / (2*xp.pi)
        return amp, opd
    
        