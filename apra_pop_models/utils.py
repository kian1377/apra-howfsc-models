from .math_module import xp, xcipy, ensure_np_array
from adefc_vortex.imshows import imshow1, imshow2, imshow3

import numpy as np
import scipy
import astropy.units as u
from astropy.io import fits
import pickle

import poppy

def make_grid(npix, pixelscale=1, half_shift=False):
    if half_shift:
        y,x = (xp.indices((npix, npix)) - npix//2 + 1/2)*pixelscale
    else:
        y,x = (xp.indices((npix, npix)) - npix//2)*pixelscale
    return x,y

def pad_or_crop( arr_in, npix ):
    n_arr_in = arr_in.shape[0]
    if n_arr_in == npix:
        return arr_in
    elif npix < n_arr_in:
        x1 = n_arr_in // 2 - npix // 2
        x2 = x1 + npix
        arr_out = arr_in[x1:x2,x1:x2].copy()
    else:
        arr_out = xp.zeros((npix,npix), dtype=arr_in.dtype)
        x1 = npix // 2 - n_arr_in // 2
        x2 = x1 + n_arr_in
        arr_out[x1:x2,x1:x2] = arr_in
    return arr_out

def save_fits(fpath, data, header=None, ow=True, quiet=False):
    data = ensure_np_array(data)
    if header is not None:
        keys = list(header.keys())
        hdr = fits.Header()
        for i in range(len(header)):
            hdr[keys[i]] = header[keys[i]]
    else: 
        hdr = None
    hdu = fits.PrimaryHDU(data=data, header=hdr)
    hdu.writeto(str(fpath), overwrite=ow) 
    if not quiet: print('Saved data to: ', str(fpath))

def load_fits(fpath, header=False):
    data = xp.array(fits.getdata(fpath))
    if header:
        hdr = fits.getheader(fpath)
        return data, hdr
    else:
        return data

# functions for saving python objects
def save_pickle(fpath, data, quiet=False):
    out = open(str(fpath), 'wb')
    pickle.dump(data, out)
    out.close()
    if not quiet: print('Saved data to: ', str(fpath))

def load_pickle(fpath):
    infile = open(str(fpath),'rb')
    pkl_data = pickle.load(infile)
    infile.close()
    return pkl_data  

def rms(data, mask=None):
    if mask is None:
        return xp.sqrt(xp.mean(xp.square(data)))
    else:
        return xp.sqrt(xp.mean(xp.square(data[mask])))

def rotate_arr(arr, rotation, reshape=False, order=3):
    if arr.dtype == complex:
        arr_r = xcipy.ndimage.rotate(xp.real(arr), angle=rotation, reshape=reshape, order=order)
        arr_i = xcipy.ndimage.rotate(xp.imag(arr), angle=rotation, reshape=reshape, order=order)
        
        rotated_arr = arr_r + 1j*arr_i
    else:
        rotated_arr = xcipy.ndimage.rotate(arr, angle=rotation, reshape=reshape, order=order)
    return rotated_arr

def interp_arr(arr, pixelscale, new_pixelscale, order=3):
        Nold = arr.shape[0]
        old_xmax = pixelscale * Nold/2

        x,y = xp.ogrid[-old_xmax:old_xmax-pixelscale:Nold*1j,
                       -old_xmax:old_xmax-pixelscale:Nold*1j]

        Nnew = int(np.ceil(2*old_xmax/new_pixelscale)) - 1
        new_xmax = new_pixelscale * Nnew/2

        newx,newy = xp.mgrid[-new_xmax:new_xmax-new_pixelscale:Nnew*1j,
                             -new_xmax:new_xmax-new_pixelscale:Nnew*1j]

        x0 = x[0,0]
        y0 = y[0,0]
        dx = x[1,0] - x0
        dy = y[0,1] - y0

        ivals = (newx - x0)/dx
        jvals = (newy - y0)/dy

        coords = xp.array([ivals, jvals])

        interped_arr = xcipy.ndimage.map_coordinates(arr, coords, order=order)
        return interped_arr

def lstsq(modes, data):
    """Least-Squares fit of modes to data.

    Parameters
    ----------
    modes : iterable
        modes to fit; sequence of ndarray of shape (m, n)
    data : numpy.ndarray
        data to fit, of shape (m, n)
        place NaN values in data for points to ignore

    Returns
    -------
    numpy.ndarray
        fit coefficients
    """
    mask = xp.isfinite(data)
    data = data[mask]
    modes = xp.asarray(modes)
    modes = modes.reshape((modes.shape[0], -1))  # flatten second dim
    modes = modes[:, mask.ravel()].T  # transpose moves modes to columns, as needed for least squares fit
    c, *_ = xp.linalg.lstsq(modes, data, rcond=None)
    return c

def generate_wfe(diam, 
                 npix=256, oversample=4, 
                 wavelength=500*u.nm,
                 opd_index=2.5, amp_index=2, 
                 opd_seed=1234, amp_seed=12345,
                 opd_rms=10*u.nm, amp_rms=0.05,
                 ):
    amp_rms *= u.nm
    wf = poppy.FresnelWavefront(beam_radius=diam/2, npix=npix, oversample=oversample, wavelength=wavelength)
    wfe_opd = poppy.StatisticalPSDWFE(index=opd_index, wfe=opd_rms, radius=diam/2, seed=opd_seed).get_opd(wf)
    wfe_amp = poppy.StatisticalPSDWFE(index=amp_index, wfe=amp_rms, radius=diam/2, seed=amp_seed).get_opd(wf)
    # print(wfe_amp)
    wfe_amp /= amp_rms.unit.to(u.m)
    
    wfe_amp = xp.asarray(wfe_amp)
    wfe_opd = xp.asarray(wfe_opd)

    mask = poppy.CircularAperture(radius=diam/2).get_transmission(wf)>0
    Zs = poppy.zernike.arbitrary_basis(mask, nterms=3, outside=0)
    
    Zc_amp = lstsq(Zs, wfe_amp)
    Zc_opd = lstsq(Zs, wfe_opd)
    for i in range(3):
        wfe_amp -= Zc_amp[i] * Zs[i]
        wfe_opd -= Zc_opd[i] * Zs[i]
    wfe_amp += 1

    mask = poppy.CircularAperture(radius=diam/2).get_transmission(wf)>0
    wfe_rms = xp.sqrt(xp.mean(xp.square(wfe_opd[mask])))
    wfe_opd *= opd_rms.to_value(u.m)/wfe_rms

    wfe = wfe_amp * xp.exp(1j*2*np.pi/wavelength.to_value(u.m) * wfe_opd)
    wfe *= poppy.CircularAperture(radius=diam/2).get_transmission(wf)

    return wfe

def centroid(arr, rounded=False):
    weighted_sum_x = 0
    total_sum_x = 0
    for i in range(arr.shape[1]):
        weighted_sum_x += np.sum(arr[:,i])*i
        total_sum_x += np.sum(arr[:,i])
    xc = round(weighted_sum_x/total_sum_x) if rounded else weighted_sum_x/total_sum_x
    
    weighted_sum_y = 0
    total_sum_y = 0
    for i in range(arr.shape[0]):
        weighted_sum_y += np.sum(arr[i,:])*i
        total_sum_y += np.sum(arr[i,:])
        
    yc = round(weighted_sum_y/total_sum_y) if rounded else weighted_sum_y/total_sum_y
    return (yc, xc)

def create_zernike_modes(pupil_mask, nmodes=15, remove_modes=0):
    if remove_modes>0:
        nmodes += remove_modes
    zernikes = poppy.zernike.arbitrary_basis(pupil_mask, nterms=nmodes, outside=0)[remove_modes:]

    return zernikes

def make_f(h=10, w=6, shift=(0,0), Nact=34):
    f_command = xp.zeros((Nact, Nact))

    top_row = Nact//2 + h//2 + shift[1]
    mid_row = Nact//2 + shift[1]
    row0 = Nact//2 - h//2 + shift[1]

    col0 = Nact//2 - w//2 + shift[0] + 1
    right_col = Nact//2 + w//2 + shift[0] + 1

    rows = xp.arange(row0, top_row)
    cols = xp.arange(col0, right_col)

    f_command[rows, col0] = 1
    f_command[top_row,cols] = 1
    f_command[mid_row,cols] = 1
    return f_command

def make_ring(rad=15, Nact=34, thresh=1/2):
    y,x = (xp.indices((Nact, Nact)) - Nact//2 + 1/2)
    r = xp.sqrt(x**2 + y**2)
    ring = (rad-thresh<r) * (r < rad+thresh)
    ring = ring.astype(float)
    return ring

def make_fourier_command(x_cpa=10, y_cpa=10, Nact=34):
    # cpa = cycles per aperture
    # max cpa must be Nact/2
    if x_cpa>Nact/2 or y_cpa>Nact/2:
        raise ValueError('The cycles per aperture is too high for the specified number of actuators.')
    y,x = xp.indices((Nact, Nact)) - Nact//2
    fourier_command = xp.cos(2*np.pi*(x_cpa*x + y_cpa*y)/Nact)
    return fourier_command

def make_cross_command(xc=[0], yc=[0], Nact=34):
    y,x = (xp.indices((Nact, Nact)) - Nact//2 + 1/2)
    cross = xp.zeros((Nact,Nact))
    for i in range(len(xc)):
        cross[(xc[i]-0.5<=x) & (x<xc[i]+0.5)] = 1
        cross[(yc[i]-0.5<=y) & (y<yc[i]+0.5)] = 1
    # cross
    return cross

# Creating focal plane masks
def create_annular_focal_plane_mask(npsf, psf_pixelscale, 
                                    irad, orad,  
                                    edge=None,
                                    shift=(0,0), 
                                    rotation=0,
                                    plot=False):
    x = (xp.linspace(-npsf/2, npsf/2-1, npsf) + 1/2)*psf_pixelscale
    x,y = xp.meshgrid(x,x)
    r = xp.hypot(x, y)
    mask = (r > irad) * (r < orad)
    if edge is not None: mask *= (x > edge)
    
    mask = xcipy.ndimage.rotate(mask, rotation, reshape=False, order=0)
    mask = xcipy.ndimage.shift(mask, (shift[1], shift[0]), order=0)
    
    if plot:
        imshow1(mask)
        
    return mask

def create_hadamard_modes(dm_mask): 
    Nacts = dm_mask.sum().astype(int)
    Nact = dm_mask.shape[0]
    np2 = 2**int(xp.ceil(xp.log2(Nacts)))
    hmodes = xp.array(scipy.linalg.hadamard(np2))
    
    had_modes = []

    inds = xp.where(dm_mask.flatten().astype(int))
    for hmode in hmodes:
        hmode = hmode[:Nacts]
        mode = xp.zeros((dm_mask.shape[0]**2))
        mode[inds] = hmode
        had_modes.append(mode)
    had_modes = xp.array(had_modes).reshape(np2, Nact, Nact)
    
    return had_modes
    
def create_fourier_modes(dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, 
                         edge=None,
                         rotation=0, 
                         fourier_sampling=0.75, 
                         which='both', 
                         return_fs=False,
                         plot=False,
                         ):
    Nact = dm_mask.shape[0]
    nfg = int(xp.round(npsf * psf_pixelscale_lamD/fourier_sampling))
    if nfg%2==1: nfg += 1
    yf, xf = (xp.indices((nfg, nfg)) - nfg//2 + 1/2) * fourier_sampling
    fourier_cm = create_annular_focal_plane_mask(nfg, fourier_sampling, iwa-fourier_sampling, owa+fourier_sampling, 
                                                 edge=edge, rotation=rotation)
    ypp, xpp = (xp.indices((Nact, Nact)) - Nact//2 + 1/2)

    sampled_fs = xp.array([xf[fourier_cm], yf[fourier_cm]]).T
    if plot: imshow1(fourier_cm, pxscl=fourier_sampling, grid=True)
    
    fourier_modes = []
    for i in range(len(sampled_fs)):
        fx = sampled_fs[i,0]
        fy = sampled_fs[i,1]
        if which=='both' or which=='cos':
            fourier_modes.append( dm_mask * xp.cos(2 * np.pi * (fx*xpp + fy*ypp)/Nact) )
        if which=='both' or which=='sin':
            fourier_modes.append( dm_mask * xp.sin(2 * np.pi * (fx*xpp + fy*ypp)/Nact) )
    
    if return_fs:
        return xp.array(fourier_modes), sampled_fs
    else:
        return xp.array(fourier_modes)

def create_fourier_probes(dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, 
                          edge=None, 
                          rotation=0, 
                          fourier_sampling=0.75, 
                          shifts=None, nprobes=2,
                          use_weighting=False, 
                          plot=False,
                          ): 
    Nact = dm_mask.shape[0]
    cos_modes, fs = create_fourier_modes(
        dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, rotation,
        fourier_sampling=fourier_sampling, 
        return_fs=True,
        which='cos',
    )
    sin_modes = create_fourier_modes(
        dm_mask, npsf, psf_pixelscale_lamD, iwa, owa, rotation,
        fourier_sampling=fourier_sampling, 
        which='sin',
    )
    nfs = fs.shape[0]

    probes = xp.zeros((nprobes, Nact, Nact))
    if use_weighting:
        fmax = xp.max(np.sqrt(fs[:,0]**2 + fs[:,1]**2))
        sum_cos = 0
        sum_sin = 0
        for i in range(nfs):
            f = np.sqrt(fs[i][0]**2 + fs[i][1]**2)
            weight = f/fmax
            sum_cos += weight*cos_modes[i]
            sum_sin += weight*sin_modes[i]
        sum_cos = sum_cos
        sum_sin = sum_sin
    else:
        sum_cos = cos_modes.sum(axis=0)
        sum_sin = sin_modes.sum(axis=0)
    
    # nprobes=2 will give one probe that is purely the sum of cos and another that is the sum of sin
    cos_weights = np.linspace(1,0,nprobes)
    sin_weights = np.linspace(0,1,nprobes)
    
    shifts = [(0,0)]*nprobes if shifts is None else shifts

    for i in range(nprobes):
        probe = cos_weights[i]*sum_cos + sin_weights[i]*sum_sin
        probe = xcipy.ndimage.shift(probe, (shifts[i][1], shifts[i][0]))
        probes[i] = probe/xp.max(probe)

        if plot: 
            probe_response = xp.abs(xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(pad_or_crop(probes[i], 4*Nact)))))
            imshow2(probes[i], probe_response, cmap1='viridis', pxscl2=1/4)

    return probes

def create_all_poke_modes(dm_mask, Ndms=1):
    Nact = dm_mask.shape[0]
    Nacts = int(np.sum(dm_mask))
    poke_modes = xp.zeros((Nacts, Nact, Nact))
    count=0
    for i in range(Nact):
        for j in range(Nact):
            if dm_mask[i,j]:
                poke_modes[count, i,j] = 1
                count+=1

    poke_modes = poke_modes[:,:].reshape(Nacts, Nact**2)
    print(poke_modes.shape)
    if Ndms==2:
        poke_modes_dm1 = xp.concatenate([poke_modes, xp.zeros_like(poke_modes)])
        poke_modes_dm2 = xp.concatenate([xp.zeros_like(poke_modes), poke_modes])
        poke_modes = xp.concatenate([poke_modes_dm1, poke_modes_dm2], axis=1)
    
    return poke_modes

def beta_reg(J, beta=-1):
    # J is the Jacobian
    JTJ = xp.matmul(J.T, J)
    rho = xp.diag(JTJ)
    alpha2 = rho.max()

    # control_matrix = xp.matmul( xp.linalg.inv( JTJ + alpha2*10.0**(beta) * xp.eye(sts.shape[0]) ), S.T)
    control_matrix = xp.matmul( xp.linalg.inv( JTJ + alpha2*10.0**(beta) * xp.eye(JTJ.shape[0]) ), J.T)
    return control_matrix

from matplotlib.patches import Circle
import skimage

def measure_center_and_angle(waffle_im, psf_pixelscale_lamD, im_thresh=1e-4, r_thresh=12,
                           verbose=True, 
                           plot=True):
    npsf = waffle_im.shape[0]
    y,x = (xp.indices((npsf, npsf)) - npsf//2)*psf_pixelscale_lamD
    r = xp.sqrt(x**2 + y**2)
    waffle_mask = (waffle_im >im_thresh) * (r>r_thresh)

    centroids = []
    for i in [0,1]:
        for j in [0,1]:
            arr = waffle_im[j*npsf//2:(j+1)*npsf//2, i*npsf//2:(i+1)*npsf//2]
            mask = waffle_mask[j*npsf//2:(j+1)*npsf//2, i*npsf//2:(i+1)*npsf//2]
            cent = np.flip(skimage.measure.centroid(ensure_np_array(mask*arr)))
            cent[0] += i*npsf//2
            cent[1] += j*npsf//2
            centroids.append(cent)
            # print(cent)
            # imshow3(mask, arr, mask*arr, lognorm2=True,
            #         patches1=[Circle(cent, 1, fill=True, color='cyan')])
    centroids.append(centroids[0])
    centroids = np.array(centroids)
    centroids[[2,3]] = centroids[[3,2]]
    if verbose: print('Centroids:\n', centroids)

    if plot: 
        patches = []
        for i in range(4):
            patches.append(Circle(centroids[i], 1, fill=False, color='black'))
        imshow3(waffle_mask, waffle_im, waffle_mask*waffle_im, lognorm2=True, vmin2=1e-5, patches1=patches)

    mean_angle = 0.0
    for i in range(4):
        angle = np.arctan2(centroids[i+1][1] - centroids[i][1], centroids[i+1][0] - centroids[i][0]) * 180/np.pi
        if angle<0:
            angle += 360
        if 0<angle<90:
            angle = 90-angle
        elif 90<angle<180:
            angle = 180-angle
        elif 180<angle<270:
            angle = 270-angle
        elif 270<angle<360:
            angle = 360-angle
        mean_angle += angle/4
    if verbose: print('Angle: ', mean_angle)

    m1 = (centroids[0][1] - centroids[2][1])/(centroids[0][0] - centroids[2][0])
    m2 = (centroids[1][1] - centroids[3][1])/(centroids[1][0] - centroids[3][0])
    # print(m1,m2)
    b1 = -m1*centroids[0][0] + centroids[0][1]
    b2 =  -m2*centroids[1][0] + centroids[1][1]
    # print(b1,b2)

    # m1*x + b1 = m2*x + b2
    # (m1-m2) * x = b2 - b1
    xc = (b2 - b1) / (m1 - m2)
    yc = m1*xc + b1
    print('Measured center in X: ', xc)
    print('Measured center in Y: ', yc)

    xshift = np.round(npsf/2 - xc)
    yshift = np.round(npsf/2 - yc)
    print('Required shift in X: ', xshift)
    print('Required shift in Y: ', yshift)

    return xshift,yshift,mean_angle

def measure_pixelscale(sin_im, cpa, 
                       dm_diam=10.2, dm_lyot_mag=9.4/9.4, lyot_diam=8.6, 
                       im_thresh=1e-4, r_thresh=20, 
                       verbose=True, plot=True,):
    npsf = sin_im.shape[0]
    y,x = (xp.indices((npsf, npsf)) - npsf//2)
    r = xp.sqrt(x**2 + y**2)
    sin_mask = (sin_im >im_thresh) * (r>r_thresh)
    imshow2(sin_mask, sin_mask*sin_im)

    centroids = []
    for i in [0,1]:
        arr = sin_im[:, i*npsf//2:(i+1)*npsf//2]
        mask = sin_mask[:, i*npsf//2:(i+1)*npsf//2]
        cent = np.flip(skimage.measure.centroid(ensure_np_array(mask*arr)))
        cent[0] += i*npsf//2
        centroids.append(cent)
        # print(cent)
        # imshow3(mask, arr, mask*arr, lognorm2=True,
        #         patches1=[Circle(cent, 1, fill=True, color='cyan')])
    centroids = np.array(centroids)
    if verbose: print('Centroids:\n', centroids)

    if plot: 
        patches = []
        for i in range(2):
            patches.append(Circle(centroids[i], 1, fill=True, color='black'))
        imshow3(sin_mask, sin_im, sin_mask*sin_im, lognorm2=True, vmin2=1e-5, patches1=patches)

    sep_pix = np.sqrt((centroids[1][0] - centroids[0][0])**2 + (centroids[1][1] - centroids[0][1])**2)
    pixelscale_lamD = (2*cpa) / sep_pix * lyot_diam/(dm_diam * dm_lyot_mag)
    if verbose: print('Pixelscale:\n', pixelscale_lamD)

    return pixelscale_lamD





