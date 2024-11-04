from .math_module import xp, _scipy, ensure_np_array
from .utils import pad_or_crop

import numpy as np
import scipy
import astropy.units as u

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
        arr = pad_or_crop(arr, npix)

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
    if npix1 is not None: arr1 = pad_or_crop(arr1, npix1)
    if npix2 is not None: arr2 = pad_or_crop(arr2, npix2)

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
    if npix1 is not None: arr1 = pad_or_crop(arr1, npix1)
    if npix2 is not None: arr2 = pad_or_crop(arr2, npix2)
    if npix3 is not None: arr3 = pad_or_crop(arr3, npix3)

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
    
    
def fancy_plot_forward(dm1_command, dm2_command, DM1_PHASOR, DM2_PHASOR, E_PUP, E_LP, E_FP, npix, wavelength):
    DM1_SURF = ensure_np_array(wavelength/(4*xp.pi) * pad_or_crop(xp.angle(DM1_PHASOR), 1.5*npix) )
    DM2_SURF = ensure_np_array(wavelength/(4*xp.pi) * pad_or_crop(xp.angle(DM2_PHASOR), 1.5*npix) )
    E_PUP = ensure_np_array(pad_or_crop(E_PUP, 1.5*npix))
    E_LP = ensure_np_array(pad_or_crop(E_LP, 1.5*npix))
    E_FP = ensure_np_array(E_FP)

    fig = plt.figure(figsize=(20,10), dpi=125)
    gs = GridSpec(2, 6, figure=fig)

    title_fz = 16

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(ensure_np_array(dm1_command), cmap='viridis')
    ax.set_title('DM1 Command', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(ensure_np_array(dm2_command), cmap='viridis')
    ax.set_title('DM2 Command', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(DM1_SURF, cmap='viridis',)
    ax.set_title('DM1 Surface', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(DM2_SURF, cmap='viridis',)
    ax.set_title('DM2 Surface', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(np.abs(E_PUP), cmap='plasma')
    ax.set_title('Total Pupil Amplitude', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(np.angle(E_PUP), cmap='twilight')
    ax.set_title('Total Pupil Phase', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(np.abs(E_LP), cmap='plasma')
    ax.set_title('Lyot Pupil Amplitude', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(np.angle(E_LP), cmap='twilight')
    ax.set_title('Lyot Pupil Phase', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(np.abs(E_FP)**2, cmap='magma', norm=LogNorm(vmin=1e-8))
    ax.set_title('Focal Plane Intensity', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 4])
    ax.imshow(np.angle(E_FP), cmap='twilight')
    ax.set_title('Focal Plane Phase', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.subplots_adjust(hspace=-0.3)

def fancy_plot_adjoint(dJ_dE_DMs, dJ_dE_LP, dJ_dE_PUP, dJ_dS_DM1, dJ_dS_DM2, dJ_dA1, dJ_dA2, control_mask, npix=1000):

    control_mask = ensure_np_array(control_mask)
    dJ_dE_DMs = ensure_np_array(dJ_dE_DMs)
    dJ_dE_LP = ensure_np_array(pad_or_crop(dJ_dE_LP, 1.5*npix))
    dJ_dE_PUP = ensure_np_array(pad_or_crop(dJ_dE_PUP, 1.5*npix))
    dJ_dS_DM1 = ensure_np_array(pad_or_crop(dJ_dS_DM1, int(1.5*npix))).real
    dJ_dS_DM2 = ensure_np_array(pad_or_crop(dJ_dS_DM2, int(1.5*npix))).real
    dJ_dA1 = ensure_np_array(dJ_dA1).real
    dJ_dA2 = ensure_np_array(dJ_dA2).real

    fig = plt.figure(figsize=(20,10), dpi=125)
    gs = GridSpec(2, 5, figure=fig)

    title_fz = 26

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(control_mask * np.abs(dJ_dE_DMs)**2, cmap='magma', norm=LogNorm(vmin=1e-5))
    ax.set_title(r'$| \frac{\partial J}{\partial E_{DM}} |^2$', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(control_mask * np.angle(dJ_dE_DMs), cmap='twilight',)
    ax.set_title(r'$\angle \frac{\partial J}{\partial E_{DM}} $', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(np.abs(dJ_dE_LP), cmap='plasma')
    ax.set_title(r'$| \frac{\partial J}{\partial E_{LP}} |$', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(np.angle(dJ_dE_LP), cmap='twilight')
    ax.set_title(r'$\angle \frac{\partial J}{\partial E_{LP}} $', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(np.abs(dJ_dE_PUP), cmap='plasma')
    ax.set_title(r'$| \frac{\partial J}{\partial E_{PUP}} |$', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(np.angle(dJ_dE_PUP), cmap='twilight')
    ax.set_title(r'$\angle \frac{\partial J}{\partial E_{PUP}} $', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(dJ_dS_DM1, cmap='viridis')
    ax.set_title(r'$ \frac{\partial J}{\partial S_{DM1}} $', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(dJ_dS_DM2, cmap='viridis')
    ax.set_title(r'$ \frac{\partial J}{\partial S_{DM2}} $', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(dJ_dA1, cmap='viridis')
    ax.set_title(r'$ \frac{\partial J}{\partial A_1} $', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 4])
    ax.imshow(dJ_dA2, cmap='viridis')
    ax.set_title(r'$ \frac{\partial J}{\partial A_2} $', fontsize=title_fz)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.subplots_adjust(hspace=-0.2)

