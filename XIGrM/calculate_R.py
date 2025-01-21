'''
This module contains codes for calculating radii R200, 
R500, etc (and corresponding masses) of halos.

Based on Douglas Rennehan's code on HaloAnalysis.
'''

import numpy as np
import pynbody as pnb

def get_radius(halo, overdensities = np.array([]), rho_crit=None, \
        precision=1e-2, rmax=None, cen=np.array([]), prop=None, ncpu=1):
    """
    Calculate different radii of a given halo with a decreasing sphere method.

    Parameters
    ----------
    
    halo
        Halo to be calculated, SimSnap in pynbody. 
        Paramaters need to be in physical_units.

    overdensity
        Overdensity factor $\Delta$s. Must be a 
        list!

    rho_crit
        Critical density of the universe at the 
        redshift of current SimSnap. Must be in 
        units of Msol/kpc**3.

    precision
        Precision for calculating radius.

    rmax
        The radius at which start using decreasing sphere 
        method. If 0, then use 2 * distance of 
        farthest particle. If set, must be in units of 
        kpc or convertible to kpc via pynbody.

    cen
        center coordinates of calculated halo. If not provided, 
        then will load from property dictionary or calculate 
        via pynbody.analysis.halo.center().

    prop
        halo.properties dictionary. If set to 0 then will 
        automatically load it. Only used for getting boxsize 
        and redshift.

	Returns
	-------

	mass
        Dict of masses of the halo within calculated radii.

    radius
        Dict of radii of the halo at given overdensities.

    """


    if prop == None:
        prop = halo.properties
    # All positions in prop are in comoving coordinates.
    boxsize = prop['boxsize'].in_units('kpc')
    if len(cen) == 0:
        center = pnb.analysis.halo.center(halo, mode='pot', return_cen=True, with_velocity=False).in_units('kpc')
    else:
        center = cen.in_units('kpc')
    #if prop['mass'] > 1e12: # For massive halos which spend most of time loading particle information, use numpy.
    with halo.immediate_mode:
        halopos = halo['pos'].in_units('kpc').view(np.ndarray) - center.view(np.ndarray)
        halomass = halo['mass'].in_units('Msol').view(np.ndarray)
   
    for i in range(3): # Correct the position of patricles crossing the box periodical boundary.
        index1, = np.nonzero(halopos[:, i] < -boxsize/2)
        halopos[index1, i] += boxsize
        index2, = np.nonzero(halopos[:, i] > boxsize/2)
        halopos[index2, i] -= boxsize
    halor = np.linalg.norm(halopos, axis=1)
    radius = 2 * halor.max()
    mass = halomass.sum()
    if rho_crit == None:
        rho_crit = pnb.analysis.cosmology.rho_crit(halo, z=prop["z"], unit='Msol kpc**-3')
    if rmax != None:
        radius  = rmax
        particles_within_r, = np.nonzero(halor <= radius)
        _halomass = halomass[particles_within_r]
        mass = _halomass.sum()
    overdensities = np.array(overdensities)
    if len(overdensities) > 1:
        overdensities.sort() # From low density to high density
    densities = overdensities * rho_crit
#         densities.units = munits/lunits**3
#         temphalo = halo
    masses = {}
    radii = {}

    for i in range(len(overdensities)):
        if i > 0: # Continue calculating based on the last result.
            radius= radii[overdensities[i-1]].view(np.ndarray)
            mass = masses[overdensities[i-1]].view(np.ndarray)
        density = densities[i]
        while True:
            temp_radius = radius
            radius = ((3 / (4 * np.pi)) * mass / density)**(1/3)
            particles_within_r, = np.nonzero(halor <= radius)
            halor = halor[particles_within_r]
            halomass = halomass[particles_within_r]
            mass = halomass.sum()
            radial_difference = np.abs(temp_radius - radius) / radius
            if mass == 0:
                radii[overdensities[i]] = pnb.array.SimArray(.0 ,units='kpc')
                masses[overdensities[i]] = pnb.array.SimArray(.0 ,units='Msol')
                break
            if radial_difference <= precision:
                radii[overdensities[i]] = pnb.array.SimArray(radius)
                masses[overdensities[i]] = pnb.array.SimArray(mass)
                radii[overdensities[i]].units = 'kpc'
                masses[overdensities[i]].units = 'Msol'
                break
    return  masses, radii
    #densities.units = halo.dm[0]['mass'].units/halo.dm['pos'].units**3

def get_radius_bisection(halo, overdensities=np.array([]), rho_crit=0, precision=1e-2, prop=0):
    """
    Calculate different radii of a given halo with a 
    bisection method. This is a prototype without much 
    optimization. And you will need to modify the source 
    code to make it compatible with the module.

    Parameters
    ----------
    halo
        Halo to be calculated, SimSnap in pynbody.
		Paramaters need to be in physical_units.
    overdensity
        Overdensity factor $\Delta$s.
    rho_crit
        Critical density of the universe at the
        redshift of current SimSnap. Must be in
        units of halo['mass'].units / 
        halo['pos'].units**3.
    precision
        Precision within which radii is calculated.
        
	Returns
	-------
    mass
        Dict of masses of the halo within calculated radii.
    radius
        Dict of radii of the halo at given overdensities.
    """

    if True:
        raise Exception('This function is now deprived! You will need to modify the source code.')
    if prop == 0:
        prop = halo.properties
	# All positions in prop are in comoving coordinates.
    center = np.array([prop['Xc'], prop['Yc'], prop['Zc']])*prop['a']/prop['h']
    tx = pnb.transformation.inverse_translate(halo, center)
    
    with tx:
        radius = 2 * halo['r'].max()
        mass = halo['mass'].sum()
        if rho_crit == 0:
            rho_crit = pnb.analysis.cosmology.rho_crit(halo, z=prop["z"], unit=mass.units/radius.units**3)

        overdensities = np.array(overdensities)
        overdensities.sort()
        densities = pnb.array.SimArray(overdensities * rho_crit)
        densities.units = mass.units/radius.units**3
        temphalo = halo
        masses = {}
        radii = {}
        
        left = 0
        right = radius

        for i in range(len(overdensities)):
            if i > 0:
                left = 0
                right = radii[overdensities[i-1]]
            end_condition = False
            while not end_condition:
                middle = (right + left) / 2
                dr = precision * middle
                temphalo = halo[pnb.filt.Sphere(middle)]
                mass_in_r = temphalo['mass'].sum()
                if mass_in_r == 0:
                    radii[overdensities[i]] = 0
                    masses[overdensities[i]] = 0
                    end_condition = True
                if np.abs(left - right) < dr:
                    radii[overdensities[i]] = middle
                    masses[overdensities[i]] = mass
                    end_condition = True
                volume_in_r = (4 * np.pi / 3) * middle**3
                density_in_r = mass_in_r/volume_in_r
                if density_in_r < densities[i]:
                    right = middle
                else:
                    left = middle
                mass = mass_in_r
    
    return masses, radii
