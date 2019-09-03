"""
Calculate cosmological parameters.
"""
import numpy as np

def Delta_vir(sim):
    '''
    Calculate the virial overdensity factor according
    to eq. 3 in Liang et al. (2016) and hereafter.

    Parameters
    ----------
    sim : pynbody.snapshot.SimSnap
    '''

    prop = sim.properties
    z = prop['z']
    omegaM0 = prop['omegaM0']
    omegaMz = omegaM0*(1+z)**3/(1 - omegaM0 + omegaM0*(1+z)**3)
    return round(49 + 96*omegaMz + 200*omegaMz/(1 + 5*omegaMz), 2)

def Ez(sim):
    '''
    Calculate E(z)=H(z)/H_0.

    Parameters
    ----------
    sim : pynbody.snapshot.SimSnap
    '''
    prop = sim.properties
    omegaM0 = prop['omegaM0']
    z = prop['z']
    return np.sqrt(1-omegaM0 + (1 + z)**3*omegaM0)