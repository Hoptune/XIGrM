'''
A series of codes to deal with atomdb data.

Currently used atomdb version: 3.0.9

The temporary version only provides atomic data
within [0.01, 100] keV. To check the details, 
see 2.0.2 release notes in 
http://www.atomdb.org/download.php
'''

import pyatomdb
import numpy as np
import astropy.io.fits as fits
from astropy import units as astrou
from astropy import constants as astroc
import time
import h5py
from scipy.spatial import KDTree
import os

def elsymbs_to_z0s(elements): # Get atomic numbers for given element lists.
    '''
    Convert element symbols to atomic numbers.
    Based on pyatomdb.atomic.elsymb_to_z0().
    '''
    aNumbers = []
    for element in elements:
        aNumbers += [pyatomdb.atomic.elsymb_to_z0(element)]
    aNumbers = np.array(aNumbers)
    return np.sort(aNumbers)

try:
    ATOMDB = os.environ['ATOMDB']
except KeyError:
    HOME = os.environ['HOME']
    ATOMDB = HOME + '/atomdb'
    if 'atomdb' not in os.listdir(HOME):
        os.mkdir(ATOMDB)
    os.environ['ATOMDB'] = ATOMDB

# Change the following into your own path.

if ATOMDB[-1] == '/':
    line_file = ATOMDB + 'apec_line.fits'
    coco_file = ATOMDB + 'apec_coco.fits'
else:
    line_file = ATOMDB + '/apec_line.fits'
    coco_file = ATOMDB + '/apec_coco.fits'

try:
    line = fits.open(line_file)
    continuum = fits.open(coco_file)
except FileNotFoundError:
    _version = '3.0.9'
    userid = '00000000'
    userprefs={}
    userprefs['USERID'] = userid
    userprefs['LASTVERSIONCHECK'] = time.time()
    pyatomdb.util.write_user_prefs(userprefs, adbroot=ATOMDB)
    pyatomdb.util.download_atomdb_emissivity_files(ATOMDB, userid, _version)
    line = fits.open(line_file)
    continuum = fits.open(coco_file)

# ergperkev
erg_per_kev = (1*astrou.keV/astrou.erg).cgs.value#1.60205062e-9

# Modify this to make it compatible with your element list
Included_Elements = ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ca', 'Fe']
atomicNumbers = elsymbs_to_z0s(Included_Elements)

# atomic = [1, 2, 6, 8, 14, 26]
atomic_in_atomdb = np.sort(continuum[2].data['Z'])

# keV_per_kboltz = 11604.505 * 1.0e3
# line_cut = line[1].data['kT'] * keV_per_kboltz
# cut = continuum[1].data['kT'] * keV_per_kboltz

line_cut = ((line[1].data['kT']) * astrou.keV/astroc.k_B).to('K').value
line_cut_kev = line[1].data['kT']
cut = (continuum[1].data['kT'] * astrou.keV/astroc.k_B).to('K').value
cut_kev = continuum[1].data['kT']
def calculate_continuum_emission(energy_bins, specific_elements = atomicNumbers, return_spectra = False):
    """
    Calculate continuum emissions and cooling rates 
    for individual atoms in atomdb.

    Parameters
    ----------
    energy_bins
        Energy_bins to calculate cooling rates and 
        generate spectra on, must be in keV in the 
        range of [0.01, 100].
    specfic_elements
        Atomic numbers of elements to be individually 
        listed in the result. All the other elements 
        in atomdb will also be calculated but will be 
        added together as the last element of the result.
    return_spectra
        Whether to return generated spectra.

    Returns
    -------
    dict
        A dictionary consists of Cooling rates (key: 'CoolingRate') 
        and spectra (key: 'Emissivity') if chosen contributed 
        by continuum for different elements at different temperatures.
    """

    atomic = specific_elements
    avg_energy_in_bins = (energy_bins[1:] + energy_bins[:-1]) / 2.0

    cont_emission = np.zeros((len(atomic) + 1, len(cut)))
    if return_spectra:
        cont_spectra = np.zeros((len(atomic) + 1, len(cut), len(energy_bins)-1))
		
    cie = pyatomdb.spectrum.CIESession(linefile = line_file, cocofile = coco_file)
    cie.set_response(energy_bins, raw=True)
    cie.set_eebrems(False)

    for i in range(0, len(cut)):
        print('Cont. Temperature: %g K' % cut[i])
        k = 0
        for a in atomic_in_atomdb:
            # spec = pyatomdb.spectrum.make_spectrum(energy_bins, i + 2, dolines = False, dopseudo = False, \
            #                                        elements = [a],\
            #                                        linefile = line_file,\
            #                                        cocofile = coco_file)
            cie.set_abund(atomic_in_atomdb, 0.)
            cie.set_abund(a, 1.)
            spec = cie.return_spectrum(cut_kev[i], nearest=True, dolines = False, dopseudo = False)
            
            if a in atomic:
                real_idx = k
                print('Atomic number: %d' % atomic[k])
                k += 1
            else:
                real_idx = -1

            print('Adding to index: %d' % real_idx)
            cont_emission[real_idx][i] += np.sum(spec * avg_energy_in_bins)
            if return_spectra:
                cont_spectra[real_idx, i, :] += spec
    if return_spectra:
        return {'CoolingRate': cont_emission * erg_per_kev, 'Emissivity': cont_spectra}
    else:
        return {'CoolingRate': cont_emission * erg_per_kev}

def calculate_line_emission(energy_bins, specific_elements = atomicNumbers, return_spectra = False):
    """
    Calculate line emissions and cooling rates for 
    individual atoms in atomdb.

    Parameters
    ----------
    energy_bins
        Energy_bins to calculate cooling rates and 
        generate spectra on, must be in keV in the 
        range of [0.01, 100].
    specfic_elements
        Atomic numbers of elements to be individually 
        listed in the result. All the other elements 
        in atomdb will also be calculated but will be 
        added together as the last element of the result.
    return_spectra
        Whether to return generated spectra.

    Returns
    -------
    dict
        A dictionary consists of Cooling rates (key: 'CoolingRate') 
        and spectra (key: 'Emissivity') if chosen contributed by emission 
        lines for different elements at different temperatures.
    """

    atomic = specific_elements
    avg_energy_in_bins = (energy_bins[1:] + energy_bins[:-1]) / 2.0

    line_emission = np.zeros((len(atomic) + 1, len(line_cut)))
    if return_spectra:
        line_spectra = np.zeros((len(atomic) + 1, len(line_cut), len(energy_bins)-1))
		
    cie = pyatomdb.spectrum.CIESession(linefile = line_file, cocofile = coco_file)
    cie.set_response(energy_bins, raw=True)
    cie.set_eebrems(False)
		
    for i in range(0, len(line_cut)):
        print('Line Temperature: %g K' % line_cut[i])
        k = 0
        for a in atomic_in_atomdb:
            # spec = pyatomdb.spectrum.make_spectrum(energy_bins, i + 2, dolines = True, docont = False, dopseudo = True,\
            #                                        elements = [a],\
            #                                        linefile = line_file,\
            #                                        cocofile = coco_file)
            cie.set_abund(atomic_in_atomdb, 0.)
            cie.set_abund(a, 1.)
            spec = cie.return_spectrum(line_cut_kev[i], nearest=True, dolines = True, docont = False, dopseudo = True)
            
            if a in atomic:
                real_idx = k
                print('Atomic number: %d' % atomic[k])
                k += 1
            else:
                real_idx = -1

            print('Adding to index: %d' % real_idx)
            line_emission[real_idx][i] += np.sum(spec * avg_energy_in_bins)
            if return_spectra:
                line_spectra[real_idx, i , :] += spec
    if return_spectra:
        return {'CoolingRate': line_emission * erg_per_kev, 'Emissivity': line_spectra}
    else:
        return {'CoolingRate':line_emission * erg_per_kev}

def get_index(te, teunits='K', logscale=False):
    """
    Finds indexes in the calculated table with kT closest ro desired kT.

    Parameters
    ----------
    te : numpy.ndarray
        Temperatures in keV or K
    teunits : {'keV' , 'K'}
        Units of te (kev or K, default keV)
    logscale : bool
        Search on a log scale for nearest temperature if set.

    Returns
    -------
    numpy.adarray
        Indexes in the Temperature list.

    #  History
    #  -------
    #  Version 0.1 - initial release
    #    Adam Foster July 17th 2015
    #
    #
    #  Version 0.2 - fixed bug so teunits = l works properly
    #    Adam Foster Jan 26th 2016
    #
    #
    #  Version 0.3 - allow array as input and use KDTree to boost calculations
    #    Zhiwei Shao July 20 2019
    """

    if teunits.lower() == 'k':
        teval = te.reshape(-1,1) # For convenience of KDTree
    elif teunits.lower() == 'kev':
        teval = ((te*astrou.keV/astroc.k_B).to('K')).value.reshape(-1,1)
    else:
        print(f"*** ERROR: unknown temeprature unit {teunits}. Must be keV or K. Exiting ***")
            
    if logscale:
        line_cut_tree = KDTree(np.log10(line_cut).reshape(-1, 1), leafsize=1)
        _, i = line_cut_tree.query(np.log10(teval), workers=-1)
    else:
        line_cut_tree = KDTree(line_cut.reshape(-1, 1), leafsize=1)
        _, i = line_cut_tree.query(teval, workers=-1)
    return i

def get_atomic_masses(atomic_numbers):
    '''
    Get atomic masses of the input atomic numbers.
    Based on pyatomdb.atomic.Z_to_mass().
    '''
    aMass = []
    for num in atomic_numbers:
        aMass.append(pyatomdb.atomic.Z_to_mass(num))
    aMass = np.array(aMass)
    return aMass

def AG89_abundances(atomic_numbers):
    '''
    Get AG89 abundances of the given input atomic numbers.
    Based on pyatomdb.atomdb.get_abundance().
    '''
    ag89 = []
    abundance = pyatomdb.atomdb.get_abundance()
    for num in atomic_numbers:
        ag89.append(abundance[num])
    ag89 = np.array(ag89)
    return ag89

def load_emissivity_file(filename, specific_elements=None, energy_band=[0.5, 2.0], n_bins=10000):
    '''
    Load the emissivity file calculated based on pyatomdb.
    If filename can't be loaded, then will calculate and save
    the emissivity information based on supplied specific_elements.

    Parameters
	----------
    filename : str
        File name of the emissivity file to load.
    specific_elements
        List of element symbols to include in calculation. 
        If set to None, will automatically calculate all 
        elements included in pyatomdb.
    energy_band
        [min, max] energy range in keV to calculate emissivity within.
    n_bins : int
        Number of bins when calculating emissivity.
    '''
    try:
        with h5py.File(filename, 'r') as f:
            continuum_emission = f['continuum_emission'][:]
            lines_emission = f['line_emission'][:]
            atom_list = f['atomic_numbers'][:]
            return {'cont': continuum_emission, 'line': lines_emission, 'atomic_numbers': atom_list}
    except OSError:
        print('Can\'t find emissivity file! Generating one now...')
        energy_bins = np.linspace(energy_band[0], energy_band[1], n_bins + 1)
        if specific_elements == None:
            elements = atomic_in_atomdb
        else:
            elements = elsymbs_to_z0s(specific_elements)
        continuum_emission = calculate_continuum_emission(energy_bins, specific_elements=elements)['CoolingRate']
        lines_emission = calculate_line_emission(energy_bins, specific_elements=elements)['CoolingRate']
        with h5py.File(filename, 'w') as f:
            h = f.create_group('Units')
            h.attrs['Emissivity'] = 'erg*cm**3/s'
            h.attrs['Band'] = '{}keV-{}keV'.format(energy_band[0], energy_band[1])
            f.create_dataset('line_emission', data = lines_emission)
            f.create_dataset('continuum_emission', data = continuum_emission)
            f.create_dataset('atomic_numbers', data=elements)
        return {'cont': continuum_emission, 'line': lines_emission, 'atomic_numbers': elements}
