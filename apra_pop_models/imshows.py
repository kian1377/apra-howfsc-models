from .math_module import xp, xcipy, ensure_np_array
from adefc_vortex import utils

import numpy as np
import scipy
import astropy.units as u
from IPython.display import display

import matplotlib.pyplot as plt
plt.rcParams['image.origin']='lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec

def imshow1(arr, 
            title=None, 
            xlabel=None,
            npix=None,
            lognorm=False, vmin=None, vmax=None,
            cmap='magma',
            pxscl=None,
            axlims=None,
            patches=None,
            grid=False, 
            figsize=(4,4), dpi=125, display_fig=True, return_fig=False):
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)
    
    if npix is not None:
        arr = utils.pad_or_crop(arr, npix)

    arr = ensure_np_array(arr)
    
    if pxscl is not None:
        if isinstance(pxscl, u.Quantity):
            pxscl = pxscl.value
        vext = pxscl * arr.shape[0]/2
        hext = pxscl * arr.shape[1]/2
        extent = [-vext,vext,-hext,hext]
    else:
        extent=None
    
    norm = LogNorm(vmin=vmin,vmax=vmax) if lognorm else Normalize(vmin=vmin,vmax=vmax)
    
    im = ax.imshow(arr, cmap=cmap, norm=norm, extent=extent)
    if axlims is not None:
        ax.set_xlim(axlims1[:2])
        ax.set_ylim(axlims1[2:])
    ax.tick_params(axis='x', labelsize=9, rotation=30)
    ax.tick_params(axis='y', labelsize=9, rotation=30)
    ax.set_xlabel(xlabel)
    if patches: 
        for patch in patches:
            ax.add_patch(patch)
    ax.set_title(title)
    if grid: ax.grid()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    plt.close()
    
    if display_fig: display(fig)
    if return_fig: return fig,ax
    
def imshow2(arr1, arr2, 
            title1=None, title2=None,
            xlabel=None, xlabel1=None, xlabel2=None,
            npix=None, npix1=None, npix2=None,
            pxscl=None, pxscl1=None, pxscl2=None,
            axlims=None, axlims1=None, axlims2=None,
            grid=False, grid1=False, grid2=False,
            cmap1='magma', cmap2='magma',
            lognorm=False, lognorm1=False, lognorm2=False,
            vmin1=None, vmax1=None, vmin2=None, vmax2=None,
            patches1=None, patches2=None,
            display_fig=True, 
            return_fig=False, 
            figsize=(10,4), dpi=125, wspace=0.2):
    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=dpi)
    
    npix1, npix2 = (npix, npix) if npix is not None else (npix1, npix2)
    if npix1 is not None: arr1 = utils.pad_or_crop(arr1, npix1)
    if npix2 is not None: arr2 = utils.pad_or_crop(arr2, npix2)

    arr1 = ensure_np_array(arr1)
    arr2 = ensure_np_array(arr2)
    
    pxscl1, pxscl2 = (pxscl, pxscl) if pxscl is not None else (pxscl1, pxscl2)
    if pxscl1 is not None:
        if isinstance(pxscl1, u.Quantity):
            vext = pxscl1.value * arr1.shape[0]/2
            hext = pxscl1.value * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
        else:
            vext = pxscl1 * arr1.shape[0]/2
            hext = pxscl1 * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
    else:
        extent1=None
        
    if pxscl2 is not None:
        if isinstance(pxscl2, u.Quantity):
            vext = pxscl2.value * arr2.shape[0]/2
            hext = pxscl2.value * arr2.shape[1]/2
            extent2 = [-pxscl2.value *arr2.shape[0]/2,vext,-hext,hext]
        else:
            vext = pxscl2 * arr2.shape[0]/2
            hext = pxscl2 * arr2.shape[1]/2
            extent2 = [-vext,vext,-hext,hext]
    else:
        extent2=None
    
    axlims1, axlims2 = (axlims, axlims) if axlims is not None else (axlims1, axlims2) # overide axlims
    xlabel1, xlabel2 = (xlabel, xlabel) if xlabel is not None else (xlabel1, xlabel2)
    
    norm1 = LogNorm(vmin=vmin1,vmax=vmax1) if lognorm1 or lognorm else Normalize(vmin=vmin1,vmax=vmax1)
    norm2 = LogNorm(vmin=vmin2,vmax=vmax2) if lognorm2 or lognorm else Normalize(vmin=vmin2,vmax=vmax2)
    
    # first plot
    im = ax[0].imshow(arr1, cmap=cmap1, norm=norm1, extent=extent1)
    if axlims1 is not None:
        ax[0].set_xlim(axlims1[:2])
        ax[0].set_ylim(axlims1[2:])
    if grid or grid1: ax[0].grid()
    ax[0].tick_params(axis='x', labelsize=9, rotation=30)
    ax[0].tick_params(axis='y', labelsize=9, rotation=30)
    ax[0].set_xlabel(xlabel1)
    if patches1: 
        for patch1 in patches1:
            ax[0].add_patch(patch1)
    ax[0].set_title(title1)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    
    # second plot
    im = ax[1].imshow(arr2, cmap=cmap2, norm=norm2, extent=extent2)
    if axlims2 is not None:
        ax[1].set_xlim(axlims2[:2])
        ax[1].set_ylim(axlims2[2:])
    if grid or grid2: ax[1].grid()
    ax[1].tick_params(axis='x', labelsize=9, rotation=30)
    ax[1].tick_params(axis='y', labelsize=9, rotation=30)
    ax[1].set_xlabel(xlabel2)
    if patches2: 
        for patch2 in patches2:
            ax[1].add_patch(patch2)
    ax[1].set_title(title2)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
        
    plt.subplots_adjust(wspace=wspace)
    
    plt.close()
    
    if display_fig: display(fig)
    if return_fig: return fig,ax

def imshow3(arr1, arr2, arr3,
            title1=None, title2=None, title3=None, titlesize=12,
            npix=None, npix1=None, npix2=None, npix3=None,
            pxscl=None, pxscl1=None, pxscl2=None, pxscl3=None, 
            axlims=None, axlims1=None, axlims2=None, axlims3=None,
            xlabel=None, xlabel1=None, xlabel2=None, xlabel3=None,
            cmap1='magma', cmap2='magma', cmap3='magma',
            lognorm=False, lognorm1=False, lognorm2=False, lognorm3=False,
            vmin1=None, vmax1=None, vmin2=None, vmax2=None, vmin3=None, vmax3=None, 
            patches1=None, patches2=None, patches3=None,
            grid=False, grid1=False, grid2=False, grid3=False,
            display_fig=True, 
            return_fig=False,
            figsize=(14,7), dpi=125, wspace=0.3):
    fig,ax = plt.subplots(nrows=1, ncols=3, figsize=figsize, dpi=dpi)
    
    npix1, npix2, npix3 = (npix, npix, npix) if npix is not None else (npix1, npix2, npix3)
    if npix1 is not None: arr1 = utils.pad_or_crop(arr1, npix1)
    if npix2 is not None: arr2 = utils.pad_or_crop(arr2, npix2)
    if npix3 is not None: arr3 = utils.pad_or_crop(arr3, npix3)

    arr1 = ensure_np_array(arr1)
    arr2 = ensure_np_array(arr2)
    arr3 = ensure_np_array(arr3)
    
    pxscl1, pxscl2, pxscl3 = (pxscl, pxscl, pxscl) if pxscl is not None else (pxscl1, pxscl2, pxscl3)
    if pxscl1 is not None:
        if isinstance(pxscl1, u.Quantity):
            vext = pxscl1.value * arr1.shape[0]/2
            hext = pxscl1.value * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
        else:
            vext = pxscl1 * arr1.shape[0]/2
            hext = pxscl1 * arr1.shape[1]/2
            extent1 = [-vext,vext,-hext,hext]
    else:
        extent1=None
        
    if pxscl2 is not None:
        if isinstance(pxscl2, u.Quantity):
            vext = pxscl2.value * arr2.shape[0]/2
            hext = pxscl2.value * arr2.shape[1]/2
            extent2 = [-vext,vext,-hext,hext]
        else:
            vext = pxscl2 * arr2.shape[0]/2
            hext = pxscl2 * arr2.shape[1]/2
            extent2 = [-vext,vext,-hext,hext]
    else:
        extent2=None
        
    if pxscl3 is not None:
        if isinstance(pxscl3, u.Quantity):
            vext = pxscl3.value * arr3.shape[0]/2
            hext = pxscl3.value * arr3.shape[1]/2
            extent3 = [-vext,vext,-hext,hext]
        else:
            vext = pxscl3 * arr3.shape[0]/2
            hext = pxscl3 * arr3.shape[1]/2
            extent3 = [-vext,vext,-hext,hext]
    else:
        extent3 = None
    
    axlims1, axlims2, axlims3 = (axlims, axlims, axlims) if axlims is not None else (axlims1, axlims2, axlims3) # overide axlims
    xlabel1, xlabel2, xlabel3 = (xlabel, xlabel, xlabel) if xlabel is not None else (xlabel1, xlabel2, xlabel3)
    
    norm1 = LogNorm(vmin=vmin1,vmax=vmax1) if lognorm1 or lognorm else Normalize(vmin=vmin1,vmax=vmax1)
    norm2 = LogNorm(vmin=vmin2,vmax=vmax2) if lognorm2 or lognorm else Normalize(vmin=vmin2,vmax=vmax2)
    norm3 = LogNorm(vmin=vmin3,vmax=vmax3) if lognorm3 or lognorm else Normalize(vmin=vmin3,vmax=vmax3)
    
    # first plot
    im = ax[0].imshow(arr1, cmap=cmap1, norm=norm1, extent=extent1)
    if axlims1 is not None:
        ax[0].set_xlim(axlims1[:2])
        ax[0].set_ylim(axlims1[2:])
    if grid or grid1: ax[0].grid()
    ax[0].tick_params(axis='x', labelsize=9, rotation=30)
    ax[0].tick_params(axis='y', labelsize=9, rotation=30)
    ax[0].set_xlabel(xlabel1)
    if patches1: 
        for patch1 in patches1:
            ax[0].add_patch(patch1)
    ax[0].set_title(title1)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    
    # second plot
    im = ax[1].imshow(arr2, cmap=cmap2, norm=norm2, extent=extent2)
    if axlims2 is not None:
        ax[1].set_xlim(axlims2[:2])
        ax[1].set_ylim(axlims2[2:])
    if grid or grid2: ax[1].grid()
    ax[1].tick_params(axis='x', labelsize=9, rotation=30)
    ax[1].tick_params(axis='y', labelsize=9, rotation=30)
    ax[1].set_xlabel(xlabel2)
    if patches2: 
        for patch2 in patches2:
            ax[1].add_patch(patch2)
    ax[1].set_title(title2)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
    
    # second plot
    im = ax[2].imshow(arr3, cmap=cmap3, norm=norm3, extent=extent3)
    if axlims3 is not None:
        ax[2].set_xlim(axlims3[:2])
        ax[2].set_ylim(axlims3[2:])
    if grid or grid3: ax[2].grid()
    ax[2].tick_params(axis='x', labelsize=9, rotation=30)
    ax[2].tick_params(axis='y', labelsize=9, rotation=30)
    ax[2].set_xlabel(xlabel3)
    if patches3: 
        for patch3 in patches3:
            ax[2].add_patch(patch3)
    ax[2].set_title(title3)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    fig.colorbar(im, cax=cax)
        
    plt.subplots_adjust(wspace=wspace)
    
    plt.close()
    
    if display_fig: display(fig)
    if return_fig: return fig,ax
    
def plot_data(data, 
              imvmin=1e-9, imvmax=1e-4, 
              vmin=1e-9, vmax=1e-4, 
              xticks=None,
              fname=None,
              ):
    ims = ensure_np_array( xp.array(data['images']) ) 
    control_mask = ensure_np_array( data['control_mask'] )
    # print(type(control_mask))
    Nitr = ims.shape[0]
    npsf = ims.shape[1]
    psf_pixelscale_lamD = data['pixelscale']

    mean_nis = np.mean(ims[:,control_mask], axis=1)
    ibest = np.argmin(mean_nis)
    best_im = ensure_np_array(data['images'][ibest])

    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6), dpi=125)
    ext = psf_pixelscale_lamD*npsf/2
    extent = [-ext, ext, -ext, ext]

    im = ax[0].imshow( best_im, norm=LogNorm(vmax=imvmax, vmin=imvmin), cmap='magma', extent=extent)
    ax[0].set_title(f'Best Iteration:\nMean Contrast = {mean_nis[ibest]:.2e}', fontsize=14)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im, cax=cax,)
    cbar.ax.set_ylabel('NI', rotation=0, labelpad=7)
    ax[0].set_position([0.05, 0, 0.45, 0.45])

    ax[0].set_ylabel('Y [$\lambda/D$]', fontsize=12, labelpad=-5)
    ax[0].set_xlabel('X [$\lambda/D$]', fontsize=12, labelpad=5)

    ax[1].set_title('Mean Contrast per Iteration', fontsize=14)
    ax[1].semilogy(mean_nis, label='3.6% Bandpass')
    ax[1].grid()
    ax[1].set_xlabel('Iteration Number', fontsize=12, )
    ax[1].set_ylabel('Mean Contrast', fontsize=14, labelpad=1)
    ax[1].set_ylim([vmin, vmax])
    xticks = np.arange(0,Nitr,2) if xticks is None else xticks
    ax[1].set_xticks(xticks)
    ax[1].set_position([0.525, 0, 0.45, 0.45])

    if fname is not None: fig.savefig(fname, format='pdf', bbox_inches="tight")

def plot_data_with_ref(
        data, 
        im1vmin=1e-9, im1vmax=1e-4,
        im2vmin=1e-9, im2vmax=1e-4, 
        vmin=1e-9, vmax=1e-4, 
        xticks=None,
        fname=None,
    ):
    ims = ensure_np_array( xp.array(data['images']) ) 
    control_mask = ensure_np_array( data['control_mask'] )
    # print(type(control_mask))
    Nitr = ims.shape[0]
    npsf = ims.shape[1]
    psf_pixelscale_lamD = data['pixelscale']

    mean_nis = np.mean(ims[:,control_mask], axis=1)
    ibest = np.argmin(mean_nis)
    ref_im = ensure_np_array(data['images'][0])
    best_im = ensure_np_array(data['images'][ibest])

    fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(15,10), dpi=125, gridspec_kw={'width_ratios': [1, 1, 1], })
    ext = psf_pixelscale_lamD*npsf/2
    extent = [-ext, ext, -ext, ext]

    w = 0.225
    im1 = ax[0].imshow(ref_im, norm=LogNorm(vmax=im1vmax, vmin=im1vmin), cmap='magma', extent=extent)
    ax[0].set_title(f'Reference Image:\nMean Contrast = {mean_nis[0]:.2e}', fontsize=14)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im1, cax=cax)
    cbar.ax.set_ylabel('NI', rotation=0, labelpad=7)
    ax[0].set_position([0, 0, w, w]) # [left, bottom, width, height]

    im2 = ax[1].imshow( best_im, norm=LogNorm(vmax=im2vmax, vmin=im2vmin), cmap='magma', extent=extent)
    ax[1].set_title(f'Best Iteration:\nMean Contrast = {mean_nis[ibest]:.2e}', fontsize=14)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im2, cax=cax,)
    cbar.ax.set_ylabel('NI', rotation=0, labelpad=7)
    ax[1].set_position([0.23, 0, w, w])

    ax[0].set_ylabel('Y [$\lambda/D$]', fontsize=12, labelpad=-5)
    ax[0].set_xlabel('X [$\lambda/D$]', fontsize=12, labelpad=5)
    ax[1].set_xlabel('X [$\lambda/D$]', fontsize=12, labelpad=5)

    ax[2].set_title('Mean Contrast per Iteration', fontsize=14)
    ax[2].semilogy(mean_nis, label='3.6% Bandpass')
    ax[2].grid()
    ax[2].set_xlabel('Iteration Number', fontsize=12, )
    ax[2].set_ylabel('Mean Contrast', fontsize=14, labelpad=1)
    ax[2].set_ylim([vmin, vmax])
    xticks = np.arange(0,Nitr,2) if xticks is None else xticks
    ax[2].set_xticks(xticks)
    ax[2].set_position([0.525, 0, 0.3, w])

    if fname is not None: fig.savefig(fname, format='pdf', bbox_inches="tight")


def plot_both_data(efc_data, aefc_data,  
                    im1vmin=1e-9, im1vmax=1e-4, 
                    im2vmin=1e-9, im2vmax=1e-4, 
                    vmin=1e-9, vmax=1e-4, 
                    xticks=None,
                    fname=None,
                    ):
    efc_ims = ensure_np_array( xp.array(efc_data['images']) ) 
    aefc_ims = ensure_np_array( xp.array(aefc_data['images']) ) 

    # print(type(control_mask))
    Nitr_efc = efc_ims.shape[0]
    Nitr_aefc = efc_ims.shape[0]

    control_mask = ensure_np_array( efc_data['control_mask'] )
    npsf = efc_ims.shape[1]
    psf_pixelscale_lamD = efc_data['pixelscale']

    mean_nis_efc = np.mean(efc_ims[:,control_mask], axis=1)
    ibest_efc = np.argmin(mean_nis_efc)
    best_efc_im = ensure_np_array(efc_data['images'][ibest_efc])
    mean_nis_aefc = np.mean(aefc_ims[:,control_mask], axis=1)
    ibest_aefc = np.argmin(mean_nis_aefc)
    best_aefc_im = ensure_np_array(aefc_data['images'][ibest_aefc])

    fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(16,9), dpi=125, gridspec_kw={'width_ratios': [1, 1, 1], })
    ext = psf_pixelscale_lamD*npsf/2
    extent = [-ext, ext, -ext, ext]

    im1 = ax[0].imshow(best_efc_im, norm=LogNorm(vmax=im1vmax, vmin=im1vmin), cmap='magma', extent=extent)
    ax[0].set_title(f'Best EFC Iteration:\nMean Contrast = {mean_nis_efc[ibest_efc]:.2e}', fontsize=14)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im1, cax=cax)
    cbar.ax.set_ylabel('NI', rotation=0, labelpad=7)
    ax[0].set_position([0, 0.3, 0.25, 0.25]) # [left, bottom, width, height]

    im2 = ax[1].imshow( best_aefc_im, norm=LogNorm(vmax=im2vmax, vmin=im2vmin), cmap='magma', extent=extent)
    ax[1].set_title(f'Best aEFC Iteration:\nMean Contrast = {mean_nis_aefc[ibest_aefc]:.2e}', fontsize=14)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im2, cax=cax,)
    cbar.ax.set_ylabel('NI', rotation=0, labelpad=7)
    ax[1].set_position([0.225, 0.3, 0.25, 0.25])

    ax[0].set_ylabel('Y [$\lambda/D$]', fontsize=12, labelpad=-5)
    ax[0].set_xlabel('X [$\lambda/D$]', fontsize=12, labelpad=5)
    ax[1].set_xlabel('X [$\lambda/D$]', fontsize=12, labelpad=5)

    ax[2].set_title('Mean Contrast per Iteration', fontsize=14)
    ax[2].semilogy(mean_nis_aefc, label='aEFC')
    ax[2].semilogy(mean_nis_efc, label='EFC')
    ax[2].grid()
    ax[2].set_xlabel('Iteration Number', fontsize=12, )
    ax[2].set_ylabel('Mean Contrast', fontsize=14, labelpad=1)
    ax[2].set_ylim([vmin, vmax])
    xticks = np.arange(0, Nitr_efc,2) if xticks is None else xticks
    ax[2].set_xticks(xticks)
    ax[2].legend(loc='upper right', fontsize=14)
    ax[2].set_position([0.525, 0.3, 0.25, 0.25])

    if fname is not None: fig.savefig(fname, format='pdf', bbox_inches="tight")

def plot_both_with_reg_conds(
        efc_data, aefc_data,  
        im1vmin=1e-9, im1vmax=1e-4, 
        im2vmin=1e-9, im2vmax=1e-4, 
        vmin=1e-9, vmax=1e-4, 
        xticks=None,
        fname=None,
        ):
    efc_ims = ensure_np_array( xp.array(efc_data['images']) ) 
    aefc_ims = ensure_np_array( xp.array(aefc_data['images']) ) 

    # print(type(control_mask))
    Nitr_efc = efc_ims.shape[0]
    Nitr_aefc = efc_ims.shape[0]

    control_mask = ensure_np_array( efc_data['control_mask'] )
    npsf = efc_ims.shape[1]
    psf_pixelscale_lamD = efc_data['pixelscale']

    mean_nis_efc = np.mean(efc_ims[:,control_mask], axis=1)
    ibest_efc = np.argmin(mean_nis_efc)
    best_efc_im = ensure_np_array(efc_data['images'][ibest_efc])
    mean_nis_aefc = np.mean(aefc_ims[:,control_mask], axis=1)
    ibest_aefc = np.argmin(mean_nis_aefc)
    best_aefc_im = ensure_np_array(aefc_data['images'][ibest_aefc])

    aefc_reg_conds = aefc_data['reg_conds']
    efc_reg_conds = efc_data['reg_conds']

    fig,ax = plt.subplots(nrows=2, ncols=2, figsize=(12,9), dpi=125,)
    ext = psf_pixelscale_lamD*npsf/2
    extent = [-ext, ext, -ext, ext]

    im1 = ax[0,0].imshow(best_efc_im, norm=LogNorm(vmax=im1vmax, vmin=im1vmin), cmap='magma', extent=extent)
    ax[0,0].set_title(f'Best EFC Iteration:\nMean Contrast = {mean_nis_efc[ibest_efc]:.2e}', fontsize=14)
    divider = make_axes_locatable(ax[0,0])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im1, cax=cax)
    cbar.ax.set_ylabel('NI', rotation=0, labelpad=7)
    # ax[0,0].set_position([0, 0.3, 0.25, 0.25]) # [left, bottom, width, height]

    im2 = ax[1,0].imshow( best_aefc_im, norm=LogNorm(vmax=im2vmax, vmin=im2vmin), cmap='magma', extent=extent)
    ax[1,0].set_title(f'Best aEFC Iteration:\nMean Contrast = {mean_nis_aefc[ibest_aefc]:.2e}', fontsize=14)
    divider = make_axes_locatable(ax[1,0])
    cax = divider.append_axes("right", size="4%", pad=0.075)
    cbar = fig.colorbar(im2, cax=cax,)
    cbar.ax.set_ylabel('NI', rotation=0, labelpad=7)
    # ax[1,0].set_position([0.225, 0.3, 0.25, 0.25])

    ax[0,0].set_ylabel('Y [$\lambda/D$]', fontsize=12, labelpad=0)
    ax[1,0].set_ylabel('Y [$\lambda/D$]', fontsize=12, labelpad=0)
    ax[1,0].set_xlabel('X [$\lambda/D$]', fontsize=12, labelpad=5)

    xticks = np.arange(0, Nitr_efc,2) if xticks is None else xticks

    ax[0,1].semilogy(mean_nis_aefc, label='aEFC')
    ax[0,1].semilogy(mean_nis_efc, label='EFC')
    ax[0,1].set_title('Mean Contrast per Iteration', fontsize=14)
    ax[0,1].grid()
    ax[0,1].set_xlabel('Iteration Number', fontsize=12, )
    ax[0,1].set_ylabel('Mean Contrast', fontsize=14, labelpad=1)
    ax[0,1].set_ylim([vmin, vmax])
    ax[0,1].set_xticks(xticks)
    ax[0,1].legend(loc='upper right', fontsize=14)
    # ax[2].set_position([0.525, 0.3, 0.25, 0.25])

    ax1 = ax[1,1]
    ax1.semilogy(np.linspace(1, Nitr_aefc-1, Nitr_aefc-1), aefc_reg_conds, '-o', label='aEFC', )
    ax1.set_ylabel('aEFC regularization values', fontsize=14, labelpad=5,)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax2 = ax1.twinx()
    ax2.plot(np.linspace(1, Nitr_efc-1, Nitr_efc-1), efc_reg_conds, '-o', color='#ff7f0e',)
    ax2.set_ylabel('EFC $\\beta$  values', fontsize=14, rotation=-90, labelpad = 25)
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2.set_ylim([-3.5, -0.5])
    ax2.set_xticks(xticks)
    ax[1,1].grid(axis='x')
    # ax[1,1].legend()

    ax1.set_xlabel('Iteration Number', fontsize=12, )

    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    if fname is not None: fig.savefig(fname, format='pdf', bbox_inches="tight")

def get_radial_dist(shape, scaleyx=(1.0, 1.0), cenyx=None):
    '''
    Compute the radial separation of each pixel
    from the center of a 2D array, and optionally 
    scale in x and y.
    '''
    indices = np.indices(shape)
    if cenyx is None:
        cenyx = ( (shape[0] - 1) / 2., (shape[1] - 1)  / 2.)
    radial = np.sqrt( (scaleyx[0]*(indices[0] - cenyx[0]))**2 + (scaleyx[1]*(indices[1] - cenyx[1]))**2 )
    return radial

def get_radial_contrast(im, mask, nbins=50, cenyx=None):
    im = ensure_np_array(im)
    mask = ensure_np_array(mask)
    radial = get_radial_dist(im.shape, cenyx=cenyx)
    bins = np.linspace(0, radial.max(), num=nbins, endpoint=True)
    digrad = np.digitize(radial, bins)
    profile = np.asarray([np.mean(im[ (digrad == i) & mask]) for i in np.unique(digrad)])
    return bins, profile
    
def plot_radial_contrast(im, mask, pixelscale, nbins=30, cenyx=None, xlims=None, ylims=None):
    bins, contrast = get_radial_contrast(im, mask, nbins=nbins, cenyx=cenyx)
    r = bins * pixelscale

    fig,ax = plt.subplots(nrows=1, ncols=1, dpi=125, figsize=(6,4))
    ax.semilogy(r,contrast)
    ax.set_xlabel('radial position [$\lambda/D$]')
    ax.set_ylabel('Contrast')
    ax.grid()
    if xlims is not None: ax.set_xlim(xlims[0], xlims[1])
    if ylims is not None: ax.set_ylim(ylims[0], ylims[1])
    plt.close()
    display(fig)




