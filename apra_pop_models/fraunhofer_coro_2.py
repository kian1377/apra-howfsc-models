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

class CORO():

    def __init__(self, 
                 wavelength=None, 
                 npsf=128,
                 psf_pixelscale=5e-6*u.m/u.pix,
                 psf_pixelscale_lamD=None, 
                 dm1_ref=np.zeros((34,34)),
                 dm2_ref=np.zeros((34,34)),
                 dm_inf=None, # defaults to inf.fits
                 d_dm1_dm2=277*u.mm, 
                 Imax_ref=1,
                 WFE=None,
                 ):
        
        self.wavelength_c = 650e-9*u.m
        self.total_pupil_diam = 6.5*u.m
        self.pupil_diam = 20*u.mm
        
        self.wavelength = self.wavelength_c if wavelength is None else wavelength
        
        self.npix = 1000
        self.oversample = 2.048
        self.N = int(self.npix*self.oversample)
        self.Nfpm = 4096

        self.APERTURE = xp.array(fits.getdata('aperture_gray_1000.fits'))
        self.APMASK = self.APERTURE>0
        self.WFE = xp.ones((self.npix,self.npix), dtype=complex) if WFE is None else WFE
        self.LYOT = xp.array(fits.getdata('lyot_90_gray_1000.fits'))

        self.pupil_apodizer_ratio = 1
        self.pupil_lyot_mag = 400/500 # pupil size ratios derived from focal lengths of relay OAPs

        self.fpm_fl = 500*u.mm
        self.imaging_fl = 500*u.mm

        self.lyot_diam = self.pupil_lyot_mag * 0.9 * self.pupil_diam
        self.um_per_lamD = (self.wavelength_c*self.imaging_fl/(self.lyot_diam)).to(u.um)
        self.as_per_lamD = ((self.wavelength_c/self.total_pupil_diam)*u.radian).to(u.arcsec)

        self.npsf = npsf
        if psf_pixelscale_lamD is None: # overrides psf_pixelscale this way
            self.psf_pixelscale = psf_pixelscale
            self.psf_pixelscale_lamD = self.psf_pixelscale.to_value(u.um/u.pix)/self.um_per_lamD.value
        else:
            self.psf_pixelscale_lamD = psf_pixelscale_lamD
            self.psf_pixelscale = self.psf_pixelscale_lamD * self.um_per_lamD/u.pix
        
        self.dm_inf = os.path.dirname(__file__)+'/inf.fits' if dm_inf is None else dm_inf
        self.dm1_ref = dm1_ref
        self.dm2_ref = dm2_ref
        self.d_dm1_dm2 = d_dm1_dm2
        self.init_dms()
        self.reset_dms()

        self.use_fpm = False

        self.Imax_ref = Imax_ref
        self.reverse_parity = False

    def getattr(self, attr):
        return getattr(self, attr)
    
    @property
    def psf_pixelscale(self):
        return self._psf_pixelscale
    
    @psf_pixelscale.setter
    def psf_pixelscale(self, value):
        self._psf_pixelscale = value.to(u.m/u.pix)
        self.psf_pixelscale_lamD = (self._psf_pixelscale / self.um_per_lamD).decompose().value

    def init_dms(self):
        act_spacing = 300e-6*u.m
        pupil_pxscl = self.pupil_diam.to_value(u.m)/self.npix
        sampling = act_spacing.to_value(u.m)/pupil_pxscl
        print('influence function sampling', sampling)
        inf, inf_sampling = dm.make_gaussian_inf_fun(act_spacing=act_spacing, sampling=sampling, coupling=0.15,)
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
    
    def calc_wfs(self, save_wfs=True, quiet=True): # method for getting the PSF in photons
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
        wfs = []
        self.wf = utils.pad_or_crop(self.APERTURE, self.N).astype(complex)
        if save_wfs: wfs.append(copy.copy(self.wf))

        self.wf *= utils.pad_or_crop(self.WFE, self.N)
        if save_wfs: wfs.append(copy.copy(self.wf))

        dm1_surf = utils.pad_or_crop(self.DM1.get_surface(), self.N)
        self.wf *= xp.exp(1j*4*np.pi*dm1_surf/self.wavelength.to_value(u.m))
        if save_wfs: wfs.append(copy.copy(self.wf))

        self.wf = props.ang_spec(self.wf, self.wavelength, self.d_dm1_dm2, self.pupil_diam/(self.npix*u.pix))
        dm2_surf = utils.pad_or_crop(self.DM2.get_surface(), self.N)
        # imshows.imshow1(dm2_surf)

        self.wf *= xp.exp(1j*4*np.pi*dm2_surf/self.wavelength.to_value(u.m))
        if save_wfs: wfs.append(copy.copy(self.wf))

        self.wf = props.ang_spec(self.wf, self.wavelength, -self.d_dm1_dm2, self.pupil_diam/(self.npix*u.pix))
        if save_wfs: wfs.append(copy.copy(self.wf))

        if self.use_fpm:
            self.wf = props.apply_vortex(self.wf, Nfpm=self.Nfpm, N=self.N, plot=False)
        if save_wfs: wfs.append(copy.copy(self.wf))

        self.wf *= utils.pad_or_crop(self.LYOT, self.N).astype(complex)
        if save_wfs: wfs.append(copy.copy(self.wf))

        if self.reverse_parity: self.wf = xp.rot90(xp.rot90(self.wf))

        Nlyot = int(np.round(0.9 * self.npix))
        self.wf = props.mft_forward(utils.pad_or_crop(self.wf, Nlyot), self.psf_pixelscale_lamD, self.npsf)/xp.sqrt(self.Imax_ref)
        if save_wfs: wfs.append(copy.copy(self.wf))

        if save_wfs:
            return wfs
        else:
            return self.wf
    
    def calc_wf(self):
        fpwf = self.calc_wfs(save_wfs=False, quiet=True)
        return fpwf
    
    def snap(self):
        image = xp.abs(self.calc_wfs(save_wfs=False, quiet=True))**2
        return image
    


