
   
'''
Basic implementation of an Ideal AGPM
following the POPPY nomenclature (v0.5.1)
@author Gilles Orban de Xivry (ULg)
@date 12 / 02 / 2017
'''
from __future__ import division
import poppy
from poppy.poppy_core import Wavefront, PlaneType
from poppy.fresnel import FresnelWavefront
from poppy import utils

import numpy as np

poppy.accel_math.update_math_settings()
global _ncp
from poppy.accel_math import _ncp

import astropy.units as u


class IdealAGPM(poppy.AnalyticOpticalElement):
    """ Defines an ideal vortex phase mask coronagraph.
    Parameters
    ----------
    name : string
        Descriptive name
    wavelength : float
        Wavelength in meters.
    charge : int
        Charge of the vortex
    """
    @utils.quantity_input(wavelength=u.meter)
    def __init__(self, name="unnamed AGPM ",
                 wavelength=1e-6 * u.meter,
                 charge=2,
                 singularity=None,
                 **kwargs):
        
        poppy.accel_math.update_math_settings()
        global _ncp
        from poppy.accel_math import _ncp
        
        poppy.AnalyticOpticalElement.__init__(self, planetype=PlaneType.intermediate, **kwargs)
        self.name = name
        self.lp = charge
        self.singularity = singularity
        self.central_wavelength = wavelength
        
    def get_phasor(self, wave):
        """
        Compute the amplitude transmission appropriate for a vortex for
        some given pixel spacing corresponding to the supplied Wavefront
        """

        if not isinstance(wave, Wavefront) and not isinstance(wave, FresnelWavefront):  # pragma: no cover
            raise ValueError("AGPM get_phasor must be called with a Wavefront "
                             "to define the spacing")
        assert (wave.planetype != PlaneType.image)

        y, x= self.get_coordinates(wave)
        phase = _ncp.arctan2(y, x)

        AGPM_phasor = _ncp.exp(1.j * self.lp * phase) * self.get_transmission(wave)

        idx = _ncp.where(x==0)[0][0]
        idy = _ncp.where(y==0)[0][0]
        AGPM_phasor[idx, idy] = 0
        return AGPM_phasor

    def get_opd(self, wave):
        y, x = self.get_coordinates(wave)
        phase = _ncp.arctan2(y, x)
        return self.lp * phase * self.central_wavelength.to(u.meter).value / (2 * np.pi)

    def get_transmission(self, wave):
        y, x = self.get_coordinates(wave)
        
        if self.singularity is None:
            trans = _ncp.ones(y.shape)
        else:
            circ = poppy.InverseTransmission(poppy.CircularAperture(radius=self.singularity/2))
            trans = circ.get_transmission(wave)
        return trans
    
    