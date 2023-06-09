import numpy as np
import scipy

try:
    import cupy as xp
    import cupyx.scipy as _scipy
except ImportError:
    import numpy as xp
    import scipy as _scipy
    
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

import cupyx.scipy.ndimage

import misc_funs as misc

def ensure_np_array(arr):
    if isinstance(arr,np.ndarray):
        return arr
    else: 
        return arr.get()

class CORO():

    def __init__(self, 
                 wavelength=None, 
                 npix=256, 
                 oversample=4,
                 npsf=100,
#                  psf_pixelscale=5e-6*u.m/u.pix,
                 psf_pixelscale_lamD=1/6, 
                 detector_rotation=0, 
                 dm_ref=np.zeros((34,34)),
                 dm_inf=None, # defaults to inf.fits
                 norm=None,
                 WFE=None,
                 USE_FPM=False,
                 CHARGE=6,
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
        self.N = int(npix*oversample)
        self.npsf = npsf
        self.psf_pixelscale_lamD = psf_pixelscale_lamD
        
        self.norm = norm
        
        self.dm_inf = 'inf.fits' if dm_inf is None else dm_inf
        
        self.WFE = WFE
        self.LYOT = LYOT
        
        self.init_dm()
        self.init_grids()
        
        self.det_rotation = detector_rotation
        
        self.pupil_apodizer_ratio = 1 
        self.pupil_lyot_ratio = 350/500 # pupil size ratios derived from focal lengths of relay OAPs
        
        self.USE_FPM = USE_FPM
        self.CHARGE = CHARGE
        
        
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
    
    def init_grids(self):
        self.pupil_pixelscale = self.pupil_diam.to_value(u.m) / self.npix
        self.N = int(self.npix*self.oversample)
        x_p = ( xp.linspace(-self.N/2, self.N/2-1, self.N) + 1/2 ) * self.pupil_pixelscale
        self.ppx, self.ppy = xp.meshgrid(x_p, x_p)
        self.ppr = xp.sqrt(self.ppx**2 + self.ppy**2)
        
        self.PUPIL = self.ppr < self.pupil_diam.to_value(u.m)/2
        
        self.focal_pixelscale_lamD = 1/self.oversample
        x_f = ( xp.linspace(-self.N/2, self.N/2-1, self.N) + 1/2 ) * self.focal_pixelscale_lamD
        self.fpx, self.fpy = xp.meshgrid(x_f, x_f)
        self.fpr = xp.sqrt(self.fpx**2 + self.fpy**2)
        self.fpth = xp.arctan2(self.fpy,self.fpx)
        
        x_im = ( xp.linspace(-self.npsf/2, self.npsf/2-1, self.npsf) + 1/2 ) * self.psf_pixelscale_lamD
        self.imx, self.imy = xp.meshgrid(x_im, x_im)
        self.imr = xp.sqrt(self.imx**2 + self.imy**2)
    
    def apply_dm(self, wavefront):
        fwf = poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, npix=self.npix, oversample=self.oversample)
        dm_opd = self.DM.get_opd(fwf)
        wf_opd = xp.angle(wavefront)*self.wavelength.to_value(u.m)/(2*np.pi)
        wf_opd += dm_opd
        wavefront = xp.abs(wavefront) * xp.exp(1j*2*np.pi/self.wavelength.to_value(u.m) * wf_opd)
        return wavefront
    
    def fft(self, wavefront, forward=True):
        if forward:
            wavefront = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(wavefront)))
        else:
            wavefront = xp.fft.ifftshift(xp.fft.ifft2(xp.fft.fftshift(wavefront)))
            
        return wavefront
    
    def mft(self, wavefront, nlamD, npix, forward=True,):
        # this code was duplicated from POPPY's MFT method
        npupY, npupX = wavefront.shape
        nlamDX, nlamDY = nlamD, nlamD
        npixY, npixX = npix, npix
        
        if forward:
            dU = nlamDX / float(npixX)
            dV = nlamDY / float(npixY)
            dX = 1.0 / float(npupX)
            dY = 1.0 / float(npupY)
        else:
            dX = nlamDX / float(npupX)
            dY = nlamDY / float(npupY)
            dU = 1.0 / float(npixX)
            dV = 1.0 / float(npixY)
        
        offsetY, offsetX = 0.0, 0.0
        
        Xs = (xp.arange(npupX, dtype=float) - float(npupX) / 2.0 - offsetX + 0.5) * dX
        Ys = (xp.arange(npupY, dtype=float) - float(npupY) / 2.0 - offsetY + 0.5) * dY

        Us = (xp.arange(npixX, dtype=float) - float(npixX) / 2.0 - offsetX + 0.5) * dU
        Vs = (xp.arange(npixY, dtype=float) - float(npixY) / 2.0 - offsetY + 0.5) * dV
        
        XU = xp.outer(Xs, Us)
        YV = xp.outer(Ys, Vs)
        
        if forward:
            expXU = xp.exp(-2.0 * np.pi * -1j * XU)
            expYV = xp.exp(-2.0 * np.pi * -1j * YV).T
            t1 = xp.dot(expYV, wavefront)
            t2 = xp.dot(t1, expXU)
        else:
            expYV = xp.exp(-2.0 * np.pi * 1j * YV).T
            expXU = xp.exp(-2.0 * np.pi * 1j * XU)
            t1 = xp.dot(expYV, wavefront)
            t2 = xp.dot(t1, expXU)

        norm_coeff = np.sqrt((nlamDY * nlamDX) / (npupY * npupX * npixY * npixX))
        
        return norm_coeff * t2
    
    def propagate(self):
        self.init_grids()
        
        WFE = xp.ones((self.N, self.N), dtype=xp.complex128) if self.WFE is None else self.WFE
        LYOT = xp.ones((self.N, self.N), dtype=xp.complex128) if self.LYOT is None else self.LYOT
        
        self.wavefront = xp.ones((self.N,self.N), dtype=xp.complex128)
        self.wavefront *= self.PUPIL # apply the pupil
        self.wavefront /= np.float64(xp.sqrt(xp.sum(self.PUPIL))) if self.norm is None else self.norm
        self.wavefront = self.apply_dm(self.wavefront)# apply the DM
        
        # propagate to intermediate focal plane
        self.wavefront = self.fft(self.wavefront)
        
        # propagate to the pre-FPM pupil plane
        self.wavefront = self.fft(self.wavefront, forward=False)
        self.wavefront *= WFE # apply WFE data
        
        if self.USE_FPM: 
            self.wavefront = self.apply_vortex(self.wavefront)
        
        self.wavefront *= LYOT # apply the Lyot stop
        
        # propagate to image plane with MFT
        self.nlamD = self.npsf * self.psf_pixelscale_lamD * self.oversample
        self.wavefront = self.mft(self.wavefront, self.nlamD, self.npsf)
        
        return self.wavefront
    
    def apply_vortex(self, wavefront):
        # using FALCO's implementation of a scalar vortex
        beam_radius_pix = int(self.npix/2)
        pix_per_lamD = 4
        D = 2.0*beam_radius_pix
        
        NA = wavefront.shape[1]
        NB = int(round(pix_per_lamD*D)) 
        if NB%2==1: NB +=1
        print(NA,NB)
        
        x = xp.arange(-NB/2, NB/2)
        x,y = xp.meshgrid(x,x)
        rho = xp.sqrt(x**2 + y**2)
        th = xp.arctan2(y,x)
        
        vortex_mask = xp.exp(1j * self.CHARGE * th)
#         misc.imshow2(xp.abs(vortex_mask), xp.angle(vortex_mask))
        
        in_val, out_val = (0.3, 5)
#         in_val, out_val = (1.2, 20)
        window_knee = 1 - in_val/out_val
        
        window_mask_1 = self.gen_tukey_for_vortex(2*out_val*pix_per_lamD, rho, window_knee)
        window_mask_2 = self.gen_tukey_for_vortex(NB, rho, window_knee)
        
        # DFT vectors
        x =xp.arange(-NA/2,NA/2,dtype=float)/D   #(-NA/2:NA/2-1)/D
        u1 = xp.arange(-NB/2,NB/2,dtype=float)/pix_per_lamD #(-NB/2:NB/2-1)/lambdaOverD
        u2 = xp.arange(-NB/2,NB/2,dtype=float)*2*out_val/NB
        
        # Low-sampled DFT of entire region
        FP1 = 1/(1*D*pix_per_lamD)*xp.exp(-1j*2*np.pi*xp.outer(u1,x)) @ wavefront @ xp.exp(-1j*2*np.pi*np.outer(x,u1))
        misc.imshow2(xp.abs(FP1)**2, xp.angle(FP1), npix=self.npix, lognorm1=True)
        FP1 *= vortex_mask * (1-window_mask_1)
        
        LP1 = 1/(1*D*pix_per_lamD)*xp.exp(-1j*2*np.pi*xp.outer(x,u1)) @ FP1 @ xp.exp(-1j*2*np.pi*xp.outer(u1,x))
        misc.imshow2(xp.abs(LP1), xp.angle(LP1), npix=self.npix, lognorm1=True)
        
        # Fine sampled DFT of innter region
        FP2 = 2*out_val/(1*D*NB)*xp.exp(-1j*2*np.pi*xp.outer(u2,x)) @ wavefront @ xp.exp(-1j*2*np.pi*xp.outer(x,u2))
        misc.imshow2(xp.abs(FP2)**2, xp.angle(FP2), npix=self.npix, lognorm1=True)
        
        FP2 *= vortex_mask * window_mask_2
        LP2 = 2.0*out_val/(1*D*NB)*xp.exp(-1j*2*np.pi*xp.outer(x,u2)) @ FP2 @ xp.exp(-1j*2*np.pi*xp.outer(u2,x)) 
        misc.imshow2(xp.abs(LP2), xp.angle(LP2), npix=self.npix, lognorm1=True)
        
        post_pupil_wavefront = LP1 + LP2;
        
        return post_pupil_wavefront
    
    def gen_tukey_for_vortex(self, Nwindow, RHO, alpha):
        Nlut = int(10*Nwindow)
        rhos0 = xp.linspace(-Nwindow/2, Nwindow/2, Nlut)
        lut = xp.array(scipy.signal.tukey(Nlut, alpha))  #,left=0,right=0)
        
        windowTukey = xp.interp(RHO, rhos0, lut)
        
        return windowTukey
    
    #     def apply_vortex(self, wavefront, q=1024, scaling_factor=4, window_size=32):
#         pupil_diameter = wavefront.shape[0] * self.pupil_pixelscale
#         print('pupil', pupil_diameter)
#         levels = int(np.ceil(np.log(q / 2) / np.log(scaling_factor))) + 1
#         qs = [2 * scaling_factor**i for i in range(levels)]
#         num_airys = [np.array([self.N,self.N])/2]
#         print(levels)
#         print(qs)
#         print(num_airys)
        
#         focal_grids = []
#         focal_masks = []
#         props = []
        
#         for i in range(1, levels):
#             num_airys.append(num_airys[i - 1] * window_size / (2 * qs[i - 1] * num_airys[i - 1]))
#         print(num_airys)
        
#         for i in range(levels):
#             q = qs[i] # the number of pixels per resolution element (pixels/(f lambda/D))
#             num_airy = num_airys[i] # radial extent of the grid in resolution elements (f lambda/D)
#             nfp = int(2*q*num_airy[0]) # total extent of the focal plane in pixels
# #             print(nfp)
# #             focal_grid = make_focal_grid(q, num_airy, pupil_diameter=pupil_diameter, reference_wavelength=1, focal_length=1)
# #             focal_mask = Field(np.exp(1j * charge * focal_grid.as_('polar').theta), focal_grid)
# #             focal_mask *= 1 - make_circular_aperture(1e-9)(focal_grid)
            
#             x_f = ( np.linspace(-nfp/2, nfp/2-1, nfp) + 1/2 ) /q*pupil_diameter * 2
#             fpx, fpy = np.meshgrid(x_f, x_f)
#             misc.imshow1(fpx)
#             fpr = np.sqrt(fpx**2 + fpy**2)
#             fpth = np.arctan2(fpy,fpx)
#             circ_mask = fpr < 1
#             print(fpr.max())
#             focal_mask = np.exp(1j * self.CHARGE * fpth)
# #             focal_mask *= 1 - circ_mask
#             print(focal_mask.shape)
# #             misc.imshow2(np.abs(focal_mask), np.angle(focal_mask))
            
#             if i != levels - 1:
#                 wx = scipy.signal.windows.tukey(window_size, 1, False)
#                 wy = scipy.signal.windows.tukey(window_size, 1, False)
#                 w = np.outer(wy, wx)

#                 w = np.pad(w, (np.array([nfp,nfp]) - w.shape) // 2, 'constant')
#                 focal_mask *= 1 - w

#             for j in range(i):
# #                 fft = FastFourierTransform(focal_grids[j])
# #                 mft = MatrixFourierTransform(focal_grid, fft.output_grid)
# #                 focal_mask -= mft.backward(fft.forward(self.focal_masks[j]))
#                 temp = self.fft(xp.array(focal_masks[j]), forward=False)
#                 focal_mask -= ensure_np_array(self.mft(temp, fpr.max(), nfp, forward=False))
                
#             focal_masks.append(focal_mask)
            
# #             misc.imshow2(np.abs(focal_mask), np.angle(focal_mask))

#         return wavefront
    
    
    
    