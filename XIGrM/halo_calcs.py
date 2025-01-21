"""
Tools for analysing IGrM and halo properties. 
Assume any necessary basic quantities is already 
prepared as the derived arrays of the snapshot.
"""

from astropy.table import Table
import pynbody as pnb
import astropy.constants as astroc
import astropy.units as astrou
import numpy as np
import tqdm
from multiprocess import Pool
import time
import h5py
import os.path
from . import calculate_R as cR
from . import cosmology
from .X_properties import cal_tspec, cal_tweight

def get_center(h, haloid, fmt='gizmo_ahf'):
    if fmt.endswith('ahf'):
        prop = h.get_properties_one_halo(haloid)
        cen = pnb.array.SimArray([prop['Xc'], prop['Yc'], prop['Zc']], units='kpc') * \
                prop['a']/prop['h']
        if fmt == 'tipsy_ahf':
            cen -= prop['boxsize'].in_units('kpc')/2
    else:
        if 'phi' in h[haloid].ancestor.loadable_keys():
            cen_mode = 'pot'
        else:
            cen_mode = 'com'
        cen = pnb.analysis.halo.center(h[haloid], \
                    mode=cen_mode, return_cen=True, with_velocity=False)
    return cen

def correct_pos(pos, boxsize):
    '''
    Correct the position of patricles crossing the box periodical boundary.
    '''
    for dim in range(3):
        index1, = np.where(pos[:, dim] < -boxsize/2)
        pos[index1, dim] += boxsize
        index2, = np.where(pos[:, dim] > boxsize/2)
        pos[index2, dim] -= boxsize
    return pos

def center_location(pos, cen, boxsize):
    shifted_pos = pos - cen
    new_pos = correct_pos(shifted_pos, boxsize)
    return new_pos, np.linalg.norm(new_pos, axis=1)

def calcu_initmasses(h, haloid):
    '''
    Calculate total stellar and star-forming gas mass which are used to define galaxies.
    Properties loaded during the calculation:
        gas: temp, mass
        star: mass
    Parameters
    -----------
    h : pynbody.snapshot.subsnap
        Subsnap to perform calculations.
    haloid
        Halo id to calculate radii and masses. Halo will be queried via h[haloid].
    '''
    results = {'M': {}}
    results['M']['total_star'] = h[haloid].star['mass'].sum()
    sf_gas = h[haloid].gas[pnb.filt.LowPass('temp', '3e4 K')]
    results['M']['total_sfgas'] = sf_gas['mass'].sum()
    return haloid, results

def calcu_radii_masses(h, haloid, center=None, rho_crit=None, rdict=None, precision=1e-2, rmax=None):
    '''
    Calculate radii (Rvir, R200, etc) and corresponding masses.
    Properties loaded during the calculation: pos, mass

    Parameters
    -----------
    h : pynbody.snapshot.subsnap
        Subsnap to perform calculations.
    haloid
        Halo id to calculate radii and masses. Halo will be queried via h[haloid].
    rdict : dict
        names and values for overdensity factors. Default is: 
        {'vir': cosmology.Delta_vir(h[haloid]), '200': 200, '500': 500, '2500': 2500}
    precision : float
        Precision for calculate radius. See get_index() in 
        calculate_R.py documentation for detail.
    rmax
        Maximum value for the shrinking sphere method. See 
        get_index() in calculate_R.py documentation for detail.
    '''
    if rdict is None:
        virovden = cosmology.Delta_vir(h[haloid])
        rdict = {'vir': virovden, '200': 200, '500': 500, '2500': 2500}
    if center is None:
        center = get_center(h, haloid)
    MassRadii = cR.get_radius(h[haloid], \
                        overdensities=list(rdict.values()), rho_crit=rho_crit, \
                        prop=None, precision=precision, cen=center, rmax=rmax)
    masses = {key: MassRadii[0][rdict[key]] for key in rdict.keys()}
    radii = {key: MassRadii[1][rdict[key]] for key in rdict.keys()}
    return haloid, dict(M=masses, R=radii)

def calcu_specific_masses(h, haloid, radii, center=None,
                          temp_cut=pnb.units.Unit('5e5 K'),
                          nh_cut=pnb.units.Unit('0.13 cm**-3')):
    '''
    Calculate some specific masses with given radii, such as baryon, IGrM, etc.
    Properties loaded during the calculation:
        i.) stars: mass, pos
        ii.) gas: mass, pos, temp, nh

    Parameters
    -----------
    h : pynbody.snapshot.subsnap
        Subsnap to perform calculations.
    haloid
        Halo id to calculate radii and masses. Halo will be queried via h[haloid].
    radii : dict
        {name: radii} pairs (in units of kpc) to calculate specific masses within.
    temp_cut
        Temperature limit above which gas particles are 
        considered as hot.
    nh_cut
        nh limit above which gas particles are considered 
        as star forming.
    '''
    if center is None:
        center = get_center(h, haloid)
    boxsize = h[haloid].ancestor.properties['boxsize']
    
    _, halor = center_location(h[haloid]['pos'], center, boxsize)
    results = {'M': dict()}

    for rkey, rval in radii.items():
        thalo = h[haloid][halor < rval]
        results['M']['star'+rkey] = thalo.s['mass'].sum()
        results['M']['gas'+rkey] = thalo.g['mass'].sum()
        results['M']['bar'+rkey] = results['M']['star'+rkey] + results['M']['gas'+rkey]

        ism = thalo.g[thalo.g['nh'] > nh_cut]
        results['M']['ism'+rkey] = ism['mass'].sum()

        cold_diffuse = thalo.g[(thalo.g['temp'] < temp_cut) & (thalo.g['nh'] < nh_cut)]
        results['M']['cold'+rkey] = cold_diffuse['mass'].sum() + results['M']['ism'+rkey]

        igrm = thalo.g[(thalo.g['temp'] > temp_cut) & (thalo.g['nh'] < nh_cut)]
        results['M']['igrm'+rkey] = igrm['mass'].sum()
    return haloid, results

def calcu_temp_lumi(h, haloid, radii, cal_file, center=None, datatype='gizmo_ahf',
                    core_corr_factor=.15,
                    temp_cut=pnb.units.Unit('5e5 K'),
                    nh_cut=pnb.units.Unit('0.13 cm**-3'), \
                    additional_filt=None):
    '''
    Calculate all the temperatures and luminosities. 
    Properties loaded during the calculation:
        gas: temp, nh, Lx, Lxb, Lx_cont, Lxb_cont, metals, rho

    Parameters
    -----------
    h : pynbody.snapshot.subsnap
        Subsnap to perform calculations.
    haloid
        Halo id to calculate radii and masses. Halo will be queried via h[haloid].
    radii : dict
        {name: radii} pairs (in units of kpc) to calculate specific masses within.
    cal_file
        Calibration file used for calculating Tspec.
    datatype : str
        gizmo_ahf or tipsy_ahf, will only how to query the total metallicity.
    center : np.array
        center of the halo
    core_corr_factor
        Inner radius for calculating core-corrected 
        temperatures. Gas particles within 
        (core_corr_factor*R, R) will be used for calculation.
    temp_cut
        Temperature limit above which gas particles are 
        considered as hot.
    nh_cut
        nh limit above which gas particles are considered 
        as star forming.
    additional_filt
        Any additional filter used to constrain the hot diffuse 
        gas we are investigating.
    '''
    results = {'L': dict(), 'T': dict()}
    if center is None:
        center = get_center(h, haloid)
    boxsize = h[haloid].ancestor.properties['boxsize']
    if additional_filt is None:
        hot_diffuse_filt = pnb.filt.HighPass('temp', temp_cut) & \
                pnb.filt.LowPass('nh', nh_cut)
    else:
        hot_diffuse_filt = pnb.filt.HighPass('temp', temp_cut) & \
                pnb.filt.LowPass('nh', nh_cut) & additional_filt
        
    hotgas = h[haloid].gas[hot_diffuse_filt]
    _, hotr = center_location(hotgas['pos'], center, boxsize)

    for rkey, rval in radii.items():
        gas_idx = (hotr < rval)
        tmphot = hotgas[gas_idx]
        with tmphot.immediate_mode:
            results['T']['x'], results['L']['x'] = cal_tweight(tmphot, weight_type='Lx')
            results['T']['x_cont'], results['L']['x_cont'] = \
                cal_tweight(tmphot, weight_type='Lx_cont')
            results['T']['mass'], _ = cal_tweight(tmphot, weight_type='mass')
            results['T']['spec'] = pnb.array.SimArray(cal_tspec(tmphot, \
                                    cal_f=cal_file, datatype=datatype), units='keV')
            results['T']['xb'], results['L']['xb'] = \
                        cal_tweight(tmphot, weight_type='Lxb')
            results['L']['xb_cont'] = tmphot['Lxb_cont'].sum()

        # Core-corrected temperatures:
        # Filter:
        core_idx = (hotr < core_corr_factor*rval)
        tmpcore = hotgas[gas_idx & (~core_idx)]
        with tmpcore.immediate_mode:
            results['T']['spec_corr'] = pnb.array.SimArray(cal_tspec(tmpcore, \
                                    cal_f=cal_file, datatype=datatype), units='keV')
            results['T']['x_corr'], results['L']['x_corr'] = cal_tweight(tmpcore, weight_type='Lx')
            results['T']['xb_corr'], results['L']['xb_corr'] = \
                        cal_tweight(tmpcore, weight_type='Lxb')
            results['T']['x_corr_cont'], _ = cal_tweight(tmpcore, weight_type='Lx_cont')
            results['T']['mass_corr'], _ = cal_tweight(tmpcore, weight_type='mass')
    return haloid, results

def calcu_entropy(h, haloid, radii, cal_file, center=None, n_par=9, datatype='gizmo_ahf', \
                  thickness=0.05, volume_type='full', \
                  temp_cut=pnb.units.Unit('5e5 K'),
                  nh_cut=pnb.units.Unit('0.13 cm**-3'), additional_filt=None):
    '''
    Calculate all entropy within a thin spherical shell 
    centered at halo.
    Properties loaded during the calculation:
        gas: temp, nh, ne, metals, rho

    Parameters
    -----------
    h : pynbody.snapshot.subsnap
        Subsnap to perform calculations.
    haloid
        Halo id to calculate radii and masses. Halo will be queried via h[haloid].
    radii : dict
        {name: radii} pairs (in units of kpc) to calculate specific masses within.
    cal_file
        Calibration file used for calculating Tspec.
    n_par : int
        Number of particles the shell must contain, 
        below which entropy will not be calculated.
    thickness : float
        Thickness Devided by radius of the spherical shell, i.e., 
        the shell will be R~(1+thickness)*R. 
    volume : str
        Volume used for calculating average electron number 
        density. 'gas' means only using the sum over the volumes 
        of all hot diffuse gas particles. 'full' means to use 
        4*pi*R^2*dR.
    temp_cut
        Temperature limit above which gas particles are 
        considered as hot.
    nh_cut
        nh limit above which gas particles are considered 
        as star forming.
    additional_filt
        Any additional filter used to constrain the hot diffuse 
        gas we are investigating.
    '''
    results = {'S': dict(), 'T': dict(), 'ne': dict(), 'nh': dict()}
    if center is None:
        center = get_center(h, haloid)
    boxsize = h[haloid].ancestor.properties['boxsize']
    if additional_filt is None:
        hot_diffuse_filt = pnb.filt.HighPass('temp', temp_cut) & \
                pnb.filt.LowPass('nh', nh_cut)
    else:
        hot_diffuse_filt = pnb.filt.HighPass('temp', temp_cut) & \
                pnb.filt.LowPass('nh', nh_cut) & additional_filt
    hotgas = h[haloid].gas[hot_diffuse_filt]
    _, hotr = center_location(hotgas['pos'], center, boxsize)
    for rkey, rval in radii.items():
        gas_idx = ((hotr >= rval) & (hotr < rval*(1+thickness)))
        tmphot = hotgas[gas_idx]
        if len(tmphot) < n_par:
            results['S'][rkey] = np.nan
            results['T']['spec'+rkey] = np.nan
        else:
            tempTspec = pnb.array.SimArray(cal_tspec(tmphot, \
                                cal_f=cal_file, datatype=datatype), units='keV')
            if volume_type == 'gas':
                temp_volume = tmphot['volume'].sum()
            elif volume_type == 'full':
                temp_volume = 4/3*np.pi*(((thickness + 1) * rval)**3 - rval**3)
            elif volume_type == 'full_approx':
                temp_volume = 4*np.pi*rval**2*thickness*rval
            else:
                raise Exception("volume_type is not accepted!")
            avg_ne = ((tmphot['ne'] * tmphot['volume']).sum() / temp_volume).in_units('cm**-3')
            avg_nh = ((tmphot['nh'] * tmphot['volume']).sum() / temp_volume).in_units('cm**-3')

            results['T']['spec'+rkey] = tempTspec
            results['S'][rkey] = tempTspec/avg_ne**(2,3)
            results['ne'][rkey] = avg_ne
            results['nh'][rkey] = avg_nh
    return haloid, results

def calcu_metallicity(h, haloid, radii, center=None, elements=['H', 'O', 'Si', 'Fe'],
                      temp_cut=pnb.units.Unit('5e5 K'),
                      nh_cut=pnb.units.Unit('0.13 cm**-3'), additional_filt=None,
                      weight_types=['mass', 'Lx']):
    '''
    Calculate average metallicity of a halo. Only applicable to GIZMO runs.
    Properties loaded during the calculation:
        gas: temp, nh, metals

    Parameters
    -----------
    h : pynbody.snapshot.subsnap
        Subsnap to perform calculations.
    haloid
        Halo id to calculate radii and masses. Halo will be queried via h[haloid].
    radii : dict
        {name: radii} pairs (in units of kpc) to calculate specific masses within.
    elements
        Elements to calculate abundance.
    temp_cut
        Temperature limit above which gas particles are 
        considered as hot.
    nh_cut
        nh limit above which gas particles are considered 
        as star forming.
    additional_filt
        Any additional filter used to constrain the hot diffuse 
        gas we are investigating.
    weight_types
        Types of weights to use when calculate average metallicity.
    '''
    
    results = {'metals': dict()}
    if center is None:
        center = get_center(h, haloid)
    boxsize = h[haloid].ancestor.properties['boxsize']
    if additional_filt is None:
        hot_diffuse_filt = pnb.filt.HighPass('temp', temp_cut) & \
                pnb.filt.LowPass('nh', nh_cut)
    else:
        hot_diffuse_filt = pnb.filt.HighPass('temp', temp_cut) & \
                pnb.filt.LowPass('nh', nh_cut) & additional_filt
    hotgas = h[haloid].gas[hot_diffuse_filt]
    _, hotr = center_location(hotgas['pos'], center, boxsize)

    for rkey, rval in radii.items():
        gas_idx = (hotr < rval)
        tmphot = hotgas[gas_idx]
        for weight_type in weight_types:
            weight_sum = tmphot[weight_type].sum()
            for ele in elements:
                totZx = (tmphot[f'X_{ele}'] * tmphot[weight_type]).sum()
                # if ele != 'H':
                #     totZx = (tmphot['metals'][:, metal_idx[ele]] * tmphot[weight_type]).sum()
                # else:
                #     totZx = (tmphot['X_H'] * tmphot[weight_type]).sum()
                results['metals'][f'Z_{ele}{rkey}{weight_type}'] = (totZx/weight_sum)
    return haloid, results
