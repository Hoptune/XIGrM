'''
Tools to generate basic information of gas particles 
required by following analysing.
'''
import pynbody as pnb
import astropy.constants as astroc
import numpy as np

from . import prepare_pyatomdb as ppat

# some constants
m_p = astroc.m_p.cgs.value # mass of proton in cgs units
k_B = astroc.k_B.cgs.value # Boltzmann constant in cgs units

default_elements = ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ca', 'Fe']

def nh(sim):
    '''
    Calculating hydrogen number density from density.
    '''
    #X_H = 1-sim.gas['metals'][:,0]-sim.gas['metals'][:,1] #0 for metal and 1 for He
    result = sim.gas['rho'].in_units('g cm**-3').view(np.ndarray)/m_p*sim.gas['X_H']
    result = pnb.array.SimArray(result)
    result.units = 'cm**-3'
    return result

def n_X(density, mass_fraction, element):
    '''
    Convert mass fractions of other elements to number densities.
    '''
    atomicNumber = ppat.elsymbs_to_z0s([element])
    aMass = ppat.get_atomic_masses(atomicNumber)
    result = density.in_units('g cm**-3').view(np.ndarray) * mass_fraction/(aMass*m_p)
    result = pnb.array.SimArray(result)
    result.units = 'cm**-3'
    return result


def temp(sim):
    '''
    Convert internal energy to temperature, following 
    the instructions from `GIZMO documentation <http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html#snaps>`_
    '''
    #XH = 1-sim.gas['metals'][:,0]-sim.gas['metals'][:,1]
    GAMMA = 5.0/3
    # If ne = 0 gas is unionized
    mu = 4.0 / (3.0 * sim.gas['X_H'] + 4.0 * sim.gas['X_H'] * sim.gas['ElectronAbundance'] + 1.0)
    # mu = 0.6
    result = pnb.array.SimArray(sim.gas['u'].in_units('cm**2 s**-2').view(np.ndarray) * (GAMMA - 1) *\
                                 mu * (m_p / k_B))
    result.units = 'K'
    return result

def abundance_to_solar(mass_fraction, elements=default_elements):
    '''
    Convert elements mass fraction to abundance 
    relative to AG89 which is accepted by pyatomdb.
    (AG89: Anders, E. and Grevesse, N. 1989, 
    Geochimica et Cosmochimica Acta, 53, 197)

    Parameters
    ----------------
    mass_fraction
        Mass fractions of different elements. In the 
        shape of (n_particles, n_elements). The order 
        of element list must be sorted as atomic number 
        from small to large. Hydrogen must be included.
    elements
        List of elements symbols included.
    
    Returns
    -------
    numpy.ndarrays
        Abundance with respect to AG89 results. In the 
        shape of (n_particles, n_elements).
    '''

    atomicNumbers = ppat.elsymbs_to_z0s(elements)
    ag89 = ppat.AG89_abundances(atomicNumbers)
    aMass = ppat.get_atomic_masses(atomicNumbers)
    # relative_abundance = np.zeros(mass_fraction.shape)
    # for n in range(0, len(aMass)):
    #     print('{:2}'.format(n), end='\r')
    #     relative_abundance[:, n] = mass_fraction[:, n] * \
    #             (aMass[0] / (aMass[n] * mass_fraction[:, 0] * ag89[n]))
    relative_massfraction = mass_fraction / mass_fraction[:, 0].reshape(-1,1) # mass fraction relative to hydrogen
    relative_abundance = relative_massfraction * (aMass[0] / (aMass*ag89))
    return relative_abundance

def calcu_luminosity(gas, filename, mode='total', elements=default_elements, band=[0.5, 2.0], bins=1000):
    '''
    Calculate X-ray luminosity of gas particles.

    Parameters
    ----------
    gas : pynbody.snapshot.SimSnap
        SubSnap of gas particles to be calculated.
    filename : emissivity file
    mode : str
        If set to 'total', both continuum and line emission 
        will be taken into account. If set to 'cont', only 
        continuum emission will be considered.
    elements, band, bins
        Required by prepare_pyatomdb.load_emissivity_file(). 
        See load_emissivity_file() docmentation for details.
    
    Returns
    -------
    list
        List of luminosities.
    '''
    atomicNumbers = ppat.elsymbs_to_z0s(elements)

    emission = ppat.load_emissivity_file(filename, specific_elements=elements, \
                    energy_band=band, n_bins=bins)
    if mode == 'cont':
        emissivity = emission['cont']
    elif mode == 'total':
        emissivity = emission['cont'] + emission['line']
    emission_atomic_numbers = emission['atomic_numbers']

    elements_idx = np.array([], dtype=np.int)
    for i in atomicNumbers:
        a_pos, = np.where(emission_atomic_numbers == i)
        if len(a_pos) == 0:
            raise Exception('Privided emissivity file is incomplete! Lacking Z={}'.format(i))
        elements_idx = np.append(elements_idx, a_pos)

    specific_emissivity = emissivity[elements_idx, :]
    T_index = ppat.get_index(gas['temp'].in_units('K').view(np.ndarray))
    result = (specific_emissivity[:,T_index].T*gas['abundance'].view(np.ndarray)).sum(axis=1)
    result *= gas['ElectronAbundance'] * (gas['nh'].in_units('cm**-3').view(np.ndarray))**2*\
                gas['volume'].in_units('cm**3').view(np.ndarray)
    result = pnb.array.SimArray(result)
    result.units='erg s**-1'
    return result