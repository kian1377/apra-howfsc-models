from .math_module import xp, _scipy, ensure_np_array
from . import utils
from . import imshows
from . import dm

# from scoobpsf import dm

import numpy as np
import astropy.units as u
from astropy.io import fits
import time
import os
from pathlib import Path
import copy

import poppy

from scipy.signal import windows

def make_vortex_phase_mask(npix, charge=6, 
                           grid='odd', 
                           singularity=None, 
                           focal_length=500*u.mm, pupil_diam=9.5*u.mm, wavelength=650*u.nm):
    if grid=='odd':
        x = xp.linspace(-npix//2, npix//2-1, npix)
    elif grid=='even':
        x = xp.linspace(-npix//2, npix//2-1, npix) + 1/2
    x,y = xp.meshgrid(x,x)
    th = xp.arctan2(y,x)

    phasor = xp.exp(1j*charge*th)
    
    if singularity is not None:
#         sing*D/(focal_length*lam)
        r = xp.sqrt((x-1/2)**2 + (y-1/2)**2)
        mask = r>(singularity*pupil_diam/(focal_length*wavelength)).decompose().value
        phasor *= mask
    
    return phasor

def ang_spec(wavefront, wavelength, distance, pixelscale):
    """Propagate a wavefront a given distance via the angular spectrum method. 

    Parameters
    ----------
    wavefront : complex 2D array
        the input wavefront
    wavelength : astropy quantity
        the wavelength of the wavefront
    distance : astropy quantity
        distance to propagate wavefront
    pixelscale : astropy quantity
        pixelscale in physical units of the wavefront

    Returns
    -------
    complex 2D array
        the propagated wavefront
    """
    n = wavefront.shape[0]

    delkx = 2*np.pi/(n*pixelscale.to_value(u.m/u.pix))
    kxy = (xp.linspace(-n/2, n/2-1, n) + 1/2)*delkx
    k = 2*np.pi/wavelength.to_value(u.m)
    kx, ky = xp.meshgrid(kxy,kxy)

    wf_as = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(wavefront)))
    
    kz = xp.sqrt(k**2 - kx**2 - ky**2 + 0j)
    tf = xp.exp(1j*kz*distance.to_value(u.m))

    prop_wf = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(wf_as * tf)))
    # kz = 0.0
    # tf = 0.0

    return prop_wf

def mft_forward(pupil, psf_pixelscale_lamD, npsf):
    """_summary_

    Parameters
    ----------
    pupil : complex 2D array
        the pupil plane wavefront
    psf_pixelscale_lamD : scalar
        the pixelscale of the desired focal plane wavefront in terms
        of lambda/D
    npsf : integer
        the size of the desired focal plane in pixels

    Returns
    -------
    complex 2D array
        the complex wavefront at the focal plane
    """

    npix = pupil.shape[0]
    dx = 1.0 / npix
    Xs = (xp.arange(npix, dtype=float) - (npix / 2)) * dx

    du = psf_pixelscale_lamD
    Us = (xp.arange(npsf, dtype=float) - npsf / 2) * du

    xu = xp.outer(Us, Xs)
    vy = xp.outer(Xs, Us)

    My = xp.exp(-1j*2*np.pi*vy) 
    Mx = xp.exp(-1j*2*np.pi*xu) 

    norm_coeff = psf_pixelscale_lamD/npix

    # return Mx, My

    return Mx@pupil@My * norm_coeff

def mft_reverse(fpwf, psf_pixelscale_lamD, npix):
    """_summary_

    Parameters
    ----------
    fpwf : complex 2D array
        the focal plane wavefront
    psf_pixelscale_lamD : scalar
        the pixelscale of the given focal plane wavefront in terms
        of lambda/D
    npix : integer
        number of pixels across the pupil plane we are 
        performing the MFT to

    Returns
    -------
    complex 2D array
        the complex wavefront at the pupil plane
    """

    npsf = fpwf.shape[0]
    du = psf_pixelscale_lamD
    Us = (xp.arange(npsf, dtype=float) - npsf / 2) * du

    dx = 1.0 / npix
    Xs = (xp.arange(npix, dtype=float) - (npix / 2)) * dx

    ux = xp.outer(Xs, Us)
    yv = xp.outer(Us, Xs)

    My = xp.exp(-1j*2*np.pi*yv) 
    Mx = xp.exp(-1j*2*np.pi*ux) 

    norm_coeff = psf_pixelscale_lamD/npix 

    return Mx@fpwf@My * norm_coeff

def apply_vortex(pupil_wf, Nfpm, N, plot=False, return_all=False):
    # course FPM first
    if plot: imshows.imshow1(xp.abs(pupil_wf))

    npix = pupil_wf.shape[0]
    # print(npix)
    vortex_mask = make_vortex_phase_mask(Nfpm)

    window_size = int(30/ (npix/Nfpm))
    w1d = xp.array(windows.tukey(window_size, 1, False))
    low_res_window = 1 - utils.pad_or_crop(xp.outer(w1d, w1d), Nfpm)

    fp_wf_low_res = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(utils.pad_or_crop(pupil_wf, Nfpm)))) # to FPM
    fp_wf_low_res *= vortex_mask * low_res_window # apply FPM
    pupil_wf_low_res = utils.pad_or_crop(xp.fft.ifftshift(xp.fft.ifft2(xp.fft.fftshift(fp_wf_low_res))), N) # to Lyot Pupil
    if plot: imshows.imshow2(xp.abs(pupil_wf_low_res), low_res_window, npix2=128, pxscl2=npix/Nfpm)

    # high res FPM second
    high_res_sampling = 0.025 # lam/D per pixel
    # Nmft = self.Nfpm
    Nmft = int(np.round(30/high_res_sampling))
    # vortex_mask = utils.pad_or_crop(vortex_mask, Nmft)
    vortex_mask = make_vortex_phase_mask(Nmft)
    window_size = int(30/high_res_sampling)
    w1d = xp.array(windows.tukey(window_size, 1, False))
    high_res_window = utils.pad_or_crop(xp.outer(w1d, w1d), Nmft)

    # x = xp.linspace(-self.Nfpm//2, self.Nfpm//2-1, self.Nfpm) * high_res_sampling
    x = (xp.linspace(-Nmft//2, Nmft//2-1, Nmft)) * high_res_sampling
    x,y = xp.meshgrid(x,x)
    r = xp.sqrt(x**2 + y**2)
    sing_mask = r>0.15
    high_res_window *= sing_mask

    fp_wf_high_res = mft_forward(utils.pad_or_crop(pupil_wf, npix), high_res_sampling, Nmft)
    fp_wf_high_res *= vortex_mask * high_res_window # apply FPM
    pupil_wf_high_res = mft_reverse(fp_wf_high_res, high_res_sampling, npix,)
    pupil_wf_high_res = utils.pad_or_crop(pupil_wf_high_res, N)

    if plot: imshows.imshow2(xp.abs(pupil_wf_high_res), high_res_window, npix2=int(np.round(128*9.765625)), pxscl2=high_res_sampling)

    post_fpm_pupil = pupil_wf_low_res + pupil_wf_high_res

    if plot: imshows.imshow1(xp.abs(post_fpm_pupil))

    if return_all:
        return post_fpm_pupil, pupil_wf_low_res, pupil_wf_high_res, low_res_window, high_res_window
    else:
        return post_fpm_pupil

# def apply_vortex(self, pupil_wf):
#     from scipy.signal import windows

#     # course FPM first
#     vortex_mask = make_vortex_phase_mask(self.Nfpm)

#     window_size_lamD = 30
#     window_size = int(window_size_lamD/ (self.npix/self.Nfpm)) 
#     wx = xp.array(windows.tukey(window_size, 1, False))
#     wy = xp.array(windows.tukey(window_size, 1, False))
#     low_res_window = 1 - utils.pad_or_crop(xp.outer(wy, wx), self.Nfpm)
#     ndisp = 256
#     imshows.imshow1(low_res_window, npix=ndisp, pxscl=self.npix/self.Nfpm)

#     fp_wf_low_res = xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(utils.pad_or_crop(pupil_wf, self.Nfpm)))) # to FPM
#     fp_wf_low_res *= vortex_mask * low_res_window # apply FPM
#     pupil_wf_low_res = utils.pad_or_crop(xp.fft.ifftshift(xp.fft.ifft2(xp.fft.fftshift(fp_wf_low_res))), self.N) # to Lyot Pupil
    
#     post_fpm_pupil = pupil_wf_low_res

#     # high res FPM second
#     centering = 'FFTSTYLE'
#     # centering = 'SYMMETRIC'
#     high_res_sampling = 0.025 # lam/D per pixel
#     window_size = int(window_size_lamD/high_res_sampling)
#     wx = xp.array(windows.tukey(window_size, 1, False))
#     wy = xp.array(windows.tukey(window_size, 1, False))
#     high_res_window_big = utils.pad_or_crop(xp.outer(wy, wx), self.Nfpm)

#     window_size_lamD = 0.3
#     window_size = int(window_size_lamD/high_res_sampling)
#     wx = xp.array(windows.tukey(window_size, 1, False))
#     wy = xp.array(windows.tukey(window_size, 1, False))
#     high_res_window_small = utils.pad_or_crop(xp.outer(wy, wx), self.Nfpm)
#     high_res_window = high_res_window_big - high_res_window_small
#     imshows.imshow1(high_res_window, npix=int(np.round(ndisp*self.npix/self.Nfpm/high_res_sampling)), pxscl=high_res_sampling)
#     imshows.imshow1(high_res_window, npix=128, pxscl=high_res_sampling)

#     nlamD = high_res_sampling * self.Nfpm
#     fp_wf_high_res = poppy.matrixDFT.matrix_dft(utils.pad_or_crop(pupil_wf, self.npix), 
#                                                 nlamD, self.Nfpm, inverse=False, centering=centering)
#     fp_wf_high_res *= vortex_mask * high_res_window # apply FPM
#     pupil_wf_high_res = poppy.matrixDFT.matrix_dft(utils.pad_or_crop(fp_wf_high_res, self.npix), 
#                                                    nlamD, self.Nfpm, inverse=True, centering=centering)
#     pupil_wf_high_res = utils.pad_or_crop(pupil_wf_high_res, self.N)

#     post_fpm_pupil += pupil_wf_high_res

#     # high res FPM third
#     high_res_sampling = 0.0025 # lam/D per pixel
#     window_size = int(window_size_lamD/high_res_sampling)
#     wx = xp.array(windows.tukey(window_size, 1, False))
#     wy = xp.array(windows.tukey(window_size, 1, False))
#     high_res_window = utils.pad_or_crop(xp.outer(wy, wx), self.Nfpm)

#     # x = xp.linspace(-self.Nfpm//2, self.Nfpm//2-1, self.Nfpm) * high_res_sampling
#     # x,y = xp.meshgrid(x,x)
#     # r = xp.sqrt(x**2 + y**2)
#     # sing_mask = r>0.3

#     imshows.imshow1(high_res_window, npix=int(np.round(128*10)), pxscl=high_res_sampling)

#     nlamD = high_res_sampling * self.Nfpm
#     fp_wf_high_res = poppy.matrixDFT.matrix_dft(utils.pad_or_crop(pupil_wf, self.npix), 
#                                                 nlamD, self.Nfpm, inverse=False, centering=centering)
#     fp_wf_high_res *= vortex_mask * high_res_window # apply FPM
#     pupil_wf_high_res = poppy.matrixDFT.matrix_dft(utils.pad_or_crop(fp_wf_high_res, self.npix), 
#                                                    nlamD, self.Nfpm, inverse=True, centering=centering)
#     pupil_wf_high_res = utils.pad_or_crop(pupil_wf_high_res, self.N)

#     post_fpm_pupil += pupil_wf_high_res

#     return post_fpm_pupil

