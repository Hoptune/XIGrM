"""
Tools for calculating X-ray properties of the IGrM using pynbody. 
Assume any necessary quantity is already prepared as the derived 
arrays of the snapshot.
"""

import array
import pynbody as pnb
import astropy.constants as astroc
import astropy.units as astrou
import numpy as np

from pyatomdb.atomic import Z_to_mass
from pyatomdb.atomdb import get_abundance

from . import prepare_pyatomdb as ppat
from .prepare_pyatomdb import get_index
from . import calculate_R as cR
import pytspec as pt

# some constants
m_p = astroc.m_p.cgs.value
k_B = astroc.k_B.cgs.value
# Metal abundance in AG89
element_numbers = np.sort(ppat.atomic_in_atomdb)
element_abundances = ppat.AG89_abundances(element_numbers)
element_masses = ppat.get_atomic_masses(element_numbers)
Z_ag89 = (element_abundances*element_masses)[2:].sum()/(element_abundances*element_masses).sum()

def cal_tweight(halogas, weight_type='Lx'):
    """
    Calculate luminosity weighted or mass weighted temperatures.

    Parameters
    -----------
    halogas : pynbody.snapshot.SubSnap
        Gas subsnap to calculate.
    weight_type : str
        Type of weight to take. Related to the available properties
        of the gas. Now available: luminousity weighted (starts 
        with 'l') and mass weighted (starts with 'm')
    """
    with halogas.immediate_mode:
        try:
            if weight_type[0].lower() == 'l':
                weight_units = 'erg s**-1'
                T, weight_sum = np.average(halogas['temp'].in_units('K').view(np.ndarray), \
                                weights=halogas[weight_type].in_units(weight_units).view(np.ndarray), returned=True)
            elif weight_type[0].lower() == 'm':
                weight_units = 'Msol'
                T, weight_sum= np.average(halogas['temp'].in_units('K').view(np.ndarray), \
                                weights=halogas[weight_type].in_units(weight_units).view(np.ndarray), returned=True)
            else:
                raise Exception("weight_type Error!")
        except ZeroDivisionError:
            T = 0
            weight_sum = 0

    T = pnb.array.SimArray(T*k_B, units='erg')
    T.convert_units('keV')
    weight_sum = pnb.array.SimArray(weight_sum, units=weight_units)
    return T, weight_sum

def cal_tspec(hdgas, cal_f, datatype):
    """
    Calculate the Tspec of hot diffuse gas particles.

    Parameters
    -----------
    hdgas : pynbody.snapshot.SubSnap
        SubSnap of the hot diffuse gas.
    cal_f : str
        Calibration file.
    datatype : str
        Simulation data type.
    """

    cal_f = cal_f.encode('utf-8')
    k_B_with_units = pnb.units.Unit('cm**2 g s**-2 K**-1') * k_B
    with hdgas.immediate_mode:
    #hdgas = gas#[pnb.filt.HighPass('temp', '5e5 K')&pnb.filt.LowPass('nh', '0.13 cm**-3')] # short for hot diffuse
        tTemp_in_kev = (hdgas['temp'] * k_B_with_units).in_units('keV')
        if 'X_H' in hdgas.keys() and 'X_He' in hdgas.keys():
            tZ_in_solar = (1 - hdgas['X_H']- hdgas['X_He'])/Z_ag89
        elif datatype[:5] == 'gizmo':
            tZ_in_solar = hdgas['metals'][:, 0]/Z_ag89
        elif datatype[:5] == 'tipsy':
            tZ_in_solar = hdgas['metals']/Z_ag89
        else:
            raise Exception("Simulation Datatype Not Accepted!")
        temission_meassure = (hdgas['mass'].in_units('Msol') * \
                            hdgas['rho'].in_units('Msol kpc**-3')).in_units('g**2 cm**-3')
    
    Temp_in_kev = array.array('f', tTemp_in_kev)
    Z_in_solar = array.array('f', tZ_in_solar)
    emission_meassure = array.array('f', temission_meassure)
    
    if len(Temp_in_kev) == 0:
        return 0
    return pt.calculate(cal_f, Temp_in_kev, Z_in_solar, emission_meassure)

