import pynbody as pnb
import astropy.constants as c
import astropy.units as u
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from astropy.table import Table
import modules.cosmology as cos
import modules.gas_properties as g_p
import modules.halo_analysis as h_a
import modules.prepare_pyatomdb as ppat

# Load snapshot
data_file = 'data_file'
param_file = 'param_file'
s = pnb.load(data_file, paramfile = param_file)
s.physical_units()

# To correct the bug in the units of internal energy in the
# temperary version of pynbody, check before you apply. pynbody still
# use comoving units to interprete GIZMO internal energy while GIZMO
# actually uses physical units by default. See 
# http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html#snaps 
# for detail. And this line only applys to GIZMO result, for tipsy, you can safely skip this line.
s.gas['u'] /= pnb.array.SimArray(s.properties['a'], units='1')

s.gas['X_H'] = 1-s.gas['metals'][:,0]-s.gas['metals'][:,1] # hydrogen mass fraction
s.gas['nh'] = g_p.nh(s) # Hydrogen number density

# From internal energy to temperature. Again, skip this line for tipsy.
s.gas['temp'] = g_p.temp(s)

s.gas['ElectronAbundance'].units = '1'
s.gas['ne'] = s.gas['ElectronAbundance'] * s.gas['nh'].in_units('cm**-3') # eletron number density
s.gas['volume'] = s.gas['mass']/s.gas['rho'] # volume of gas particles
# Filtering hot diffuse gas particles
hotdiffusegas = s.gas[pnb.filt.HighPass('temp', '5e5 K')&pnb.filt.LowPass('nh', '0.13 cm**-3')]

# Calculate mass fraction of the included elements. The first column must be Hydrogen mass fraction.
# And you will need to change the codes for tipsy outputs.
s.gas['mass_fraction'] = np.zeros(s.gas['metals'].shape)
s.gas['mass_fraction'][:, 0] = s.gas['X_H']
s.gas['mass_fraction'][:, 1:] = s.gas['metals'][:, 1:]

# Abundance relative to solar abundance, don't have to be in gizmo format. 
# See function documentation for details.
hotdiffusegas['abundance'] = g_p.abundance_to_solar(hotdiffusegas['mass_fraction'])

# Find proper pytspec calibration file. See pytspec documentation for details.
cal_dat_dir = 'cal_dat_dir'
cal_dat_files = os.listdir(cal_dat_dir)
cal_redshift = []
for calfile in cal_dat_files:
    cal_redshift += [eval(calfile[10:17])]
calfile_idx = np.abs(np.array(cal_redshift) - s.properties['z']).argmin()
cal_file = (cal_dat_dir + cal_dat_files[calfile_idx])

# Calculate luminosity

# Recommend uncomment the following lines for the first time to generate 
# the emissivity files containing all available elements in pyatomdb.

# ----------start-------------
# e_band = [0.5, 2.0]
# e_filename = 'your_dir/{:.1f}-{:.1f}keV_emissivity_all_elements.hdf5'.format(e_band[0], e_band[1])
# ppat.load_emissivity_file(filename=e_filename, energy_band=e_band)
# -----------end--------------

narrow_band_file = 'your_dir/0.5-2.0keV_emissivity_all_elements.hdf5'
broad_band_file =  'your_dir/0.1-2.4keV_emissivity_all_elements.hdf5'

# gas property initialization
Emission_type = ['Lx', 'Lxb', 'Lx_cont', 'Lxb_cont']
for i in Emission_type:
    s.gas[i] = 0
    s.gas[i].units = 'erg s**-1'

# Total X-ray luminosity
hotdiffusegas['Lx'] = g_p.calcu_luminosity(gas=hotdiffusegas, filename=narrow_band_file, \
                                           mode='total', band=[0.5, 2.0])
hotdiffusegas['Lxb'] = g_p.calcu_luminosity(gas=hotdiffusegas, \
                            filename=broad_band_file, mode='total', band=[0.1, 2.4])

# Only continuum emission are taken into account
hotdiffusegas['Lx_cont'] = g_p.calcu_luminosity(gas=hotdiffusegas, filename=narrow_band_file, \
                                                mode='cont', band=[0.5, 2.0])
hotdiffusegas['Lxb_cont'] = g_p.calcu_luminosity(gas=hotdiffusegas, \
                                    filename=broad_band_file, mode='cont', band=[0.1, 2.4])


# Halo Analysis
h = s.halos() # Load halo catalogue
# Initialize the class halo_props. The datatype here means 
# we use gizmo outputs and amiga halo finder
halo = h_a.halo_props(h, datatype='gizmo_ahf')

galaxy_low_limit = 64 * s.star['mass'].mean() # Limits above which galaxies are considered as luminous
halo.init_relationship(galaxy_low_limit=galaxy_low_limit)

# To check if there is any unusual halo_ids and host_ids. 
# For ahf results of Liang's data, there are some halos whose 
# "hostHalo" ID is not recorded in "#ID". I choose to ignore these halos.
print("Error List:", halo.errorlist)

# Calculate halo radii and masses
halo.calcu_radii_masses(halo_id_list=halo.host_list)
# Calculate some specific masses, like starX, gasX, etc.
halo.calcu_specific_masses(halo_id_list=halo.host_list)
# Calculate temperatures and luminosities. Calculate these quantities 
# together will save some time. But individual methods are also provided 
# in halo_props if you want to calculate separately.
halo.calcu_temp_lumi(cal_file=cal_file, halo_id_list=halo.host_list)
# Calculate entropy.
halo.calcu_entropy(cal_file=cal_file, halo_id_list=halo.host_list)
# Save data.
halo.savedata('file_name_whatever_you_like.hdf5', halo_id_list=halo.host_list)