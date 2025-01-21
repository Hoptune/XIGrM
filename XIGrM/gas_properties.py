'''
Tools to generate basic information of gas particles 
required by following analysing.
'''
import pynbody as pnb
import astropy.constants as astroc
import numpy as np
import h5py
import os
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

    elements_idx = np.array([], dtype=int)
    for i in atomicNumbers:
        a_pos, = np.where(emission_atomic_numbers == i)
        if len(a_pos) == 0:
            raise Exception('Provided emissivity file is incomplete! Lacking Z={}'.format(i))
        elements_idx = np.append(elements_idx, a_pos)

    specific_emissivity = emissivity[elements_idx, :]
    T_index = ppat.get_index(gas['temp'].in_units('K').view(np.ndarray))
    result = (specific_emissivity[:,T_index].T*gas['abundance'].view(np.ndarray)).sum(axis=1)
    result *= gas['ElectronAbundance'] * ((gas['nh'])**2*gas['volume']).in_units('cm**-3').view(np.ndarray)
    result = pnb.array.SimArray(result)
    result.units='erg s**-1'
    return result

def load_gas_properties(gasfile, s, hotdiffusefilter, props_to_load=['temp', 'nh', 'ne', 'volume'],
                        Lx_to_load=['Lx', 'Lxb', 'Lx_cont', 'Lxb_cont'],
                        elements_to_load=['H', 'He', 'O', 'Si', 'Fe']):
    props_units = {'temp': 'K', 'nh': 'cm**-3', 'ne': 'cm**-3', 'volume': 'kpc**3'}
    with h5py.File(gasfile, "r") as f:
    # print(f.keys())
        if 'iord' not in s.keys():
            _ = s['iord']
        assert np.abs(f['iord'][()] - s.gas['iord']).max() < 1, "Orders of particles doesn't agree"
        
        for propkey in props_to_load:
            s.gas[propkey] = pnb.array.SimArray(f[propkey][()], units=props_units[propkey])
        
        igrm_filter = hotdiffusefilter
        for Lxkey in Lx_to_load:
            s.gas[Lxkey] = pnb.array.SimArray(f['luminosity'][Lxkey][()], units='erg s**-1')
            s.gas[~igrm_filter][Lxkey] = 0.
        
        for metal in elements_to_load:
            s.gas[f'X_{metal}'] = f['mass_fraction'][metal][()]

def prepare_gas_properties(basename, s, Lxfiles,
                           Lx_narrowband = [.5, 2.0], Lx_broadband = [.1, 2.4],
                           elements=default_elements, elements_idx=None, fixElectronAbundance=False):
    if elements_idx is None:
        elements_idx = {'H': 0, 'He': 1, 'C': 2, 'N': 3, 'O': 4, 'Ne': 5, 'Mg': 6, 'Si': 7, 'S': 8, 'Ca': 9, 'Fe': 10}
    gasfile = f"{basename}_gasproperties_Lx={Lx_narrowband[0]:.1f}-{Lx_narrowband[1]:.1f}_" + \
                f"Lxb={Lx_broadband[0]:.1f}-{Lx_broadband[1]:.1f}_" + \
                f"fixElectronAbundance={fixElectronAbundance}.hdf5"
    if not os.path.isfile(gasfile):
        s.physical_units()
        s.gas['X_H'] = 1-s.gas['metals'][:,0]-s.gas['metals'][:,1] # hydrogen mass fraction
        s.gas['nh'] = nh(s) # Hydrogen number density
        s.gas['temp'] = temp(s) # From internal energy to temperature
        if fixElectronAbundance:
            s.gas['ElectronAbundance'] = 1.14
        s.gas['ElectronAbundance'].units = '1'
        s.gas['ne'] = s.gas['ElectronAbundance'] * s.gas['nh'].in_units('cm**-3') # eletron number density
        s.gas['volume'] = s.gas['mass']/s.gas['rho'] # volume of gas particles

        s.gas['mass_fraction'] = np.zeros(s.gas['metals'].shape)
        s.gas['mass_fraction'][:, 0] = s.gas['X_H']
        s.gas['mass_fraction'][:, 1:] = s.gas['metals'][:, 1:]
        s.gas['abundance'] = abundance_to_solar(s.gas['mass_fraction'], elements=elements)
        _ = s['iord']
        
        narrow_band_file = Lxfiles['narrow']
        broad_band_file = Lxfiles['broad']

        Emission_type = ['Lx', 'Lxb', 'Lx_cont', 'Lxb_cont']
        for i in Emission_type:
            s.gas[i] = 0
            s.gas[i].units = 'erg s**-1'
        
        s.gas['Lx'] = calcu_luminosity(gas=s.gas, filename=narrow_band_file, \
                                        mode='total', band=Lx_narrowband)
        s.gas['Lxb'] = calcu_luminosity(gas=s.gas, \
                                    filename=broad_band_file, mode='total', band=Lx_broadband)
        s.gas['Lx_cont'] = calcu_luminosity(gas=s.gas, filename=narrow_band_file, \
                                                    mode='cont', band=Lx_narrowband)
        s.gas['Lxb_cont'] = calcu_luminosity(gas=s.gas, \
                                            filename=broad_band_file, mode='cont', band=Lx_broadband)

        attrs = {'Lx': f'{Lx_narrowband[0]:.1f}-{Lx_narrowband[1]:.1f} keV',
                'Lxb': f'{Lx_broadband[0]:.1f}-{Lx_broadband[1]:.1f} keV',
                'Lx_cont': f'{Lx_narrowband[0]:.1f}-{Lx_narrowband[1]:.1f} keV, w/o line emission',
                'Lxb_cont': f'{Lx_broadband[0]:.1f}-{Lx_broadband[1]:.1f} keV, w/o line emission',
                'iord': 'Gas particle ids',
                'nh': 'hydrogen number density in cm^-3',
                'ne': 'electron number density in cm^-3',
                'temp': 'temperature in K',
                'volume': 'volume of the gas particle derived from mass/rho; in kpc**3'
                }
        with h5py.File(gasfile, "w") as f:
            for key in ['iord', 'nh', 'ne', 'temp', 'volume']:
                dataset = f.create_dataset(key, data=s.gas[key])
                dataset.attrs['Description'] = attrs[key]

            grp = f.create_group("luminosity")
            for key in ['Lx', 'Lxb', 'Lx_cont', 'Lxb_cont']:
                dataset = grp.create_dataset(key, data=s.gas[key])
                dataset.attrs['Description'] = attrs[key]

            grp = f.create_group("mass_fraction")
            grp.attrs['Description'] = "Mass fractions of different elements (summed to 1)."
            for i in range(len(elements)):
                dataset = grp.create_dataset(elements[i], data=s.gas['mass_fraction'][:, elements_idx[elements[i]]])

            grp = f.create_group("abundance_to_solar")
            grp.attrs['Description'] = "Abundances of different elements relative to AG89 solar."
            for i in range(len(elements)):
                dataset = grp.create_dataset(elements[i], data=s.gas['abundance'][:, elements_idx[elements[i]]])
    else:
        print("File exists!")
    return gasfile