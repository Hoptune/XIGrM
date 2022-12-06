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
import time
import h5py
from multiprocessing import Pool

from . import prepare_pyatomdb as ppat
from . import calculate_R as cR
from . import cosmology
from . import gas_properties as g_p
from .X_properties import cal_tspec, cal_tweight
import pyatomdb

# The field here is only for initialization. If you 
# want to change the calculated quantities, you will 
# need to modify the calculating functions.

radius_field = ['vir', '200', '500', '2500']
masstype = ['star', 'gas', 'bar', 'ism', 'cold', 'igrm'] # used in Figure 8 of Liang et al.
radii_to_cal_sepcific_mass = ['200', '500']
specific_mass_field = []
for ra in radii_to_cal_sepcific_mass:
    specific_mass_field += [_masstype + ra for _masstype in masstype]
mass_field = radius_field + specific_mass_field + ['total_star', 'self_star', 'total_sfgas', 'self_sfgas']
temp_field = ['x', 'x_cont', 'xb', 'xb_corr', 'mass', 'spec', 'spec_corr', \
                'x_corr', 'x_corr_cont', 'mass_corr', 'spec500', 'spec2500']
entropy_field = ['500', '2500']
luminosity_field = ['x', 'x_cont', 'xb', 'xb_cont', 'x_corr', 'xb_corr']
ne_field = ['500', '2500']
nh_field = ['500', '2500']
# _cont for only considering continuum emission, _corr for core-corrected (0.15*R500-R500)
# xb for broad band X-ray (0.1-2.4 keV) in our case.

default_field = {'R': radius_field, 'M': mass_field, 'T': temp_field,\
            'S': entropy_field, 'L': luminosity_field, 'ne': ne_field, 'nh': nh_field}
default_units = {'T': 'keV', 'L': 'erg s**-1', 'R': 'kpc', 'M': 'Msol', \
            'S': 'keV cm**2', 'ne': 'cm**-3', 'nh': 'cm**-3'}

class halo_props:
    '''
    Systematically analyse the halo X-ray properties based 
    on other modules.

    Attributes
    -----------
    datatype : str
        A copy of the input type of simulation data.
    catalogue_original : pynbody.halo.HaloCatalogue
        The input halo catalogue.
    length : length of the input catalogue.
    host_id_of_top_level
        How catalogue record "hostHalo" for those halos 
        without a host halo. Default is 0.
    errorlist : list
        Record when the host halo ID of a certain subhalo is not 
        recorded in the catalogue (weird but will happen in
        ahf sometimes).
    rho_crit
        Critical density of the current snapshot in Msol kpc**-3.
    ovdens
        Virial overdensity factor :math:`\Delta_{vir}` of the current snapshot.
    dict : astropy.table.Table
        A copy of the halo.properties dictionary but in a table form
        to make future reference more convenient.
    haloid
        List of halo_id given by property dictionary.
    IDlist
        Table of halo_id and corresponding #ID given in the property 
        dictionary.
    hostid
        List of the halo_id of the host halo of each halo (originally 
        recorded in the property dictionary in the form of #ID).
    new_catalogue : dict
        The new catalogue which includes all the subhalo particles 
        in its host halo. The keys of the dictionary are the indexes of 
        halos in `catalogue_original`.
    prop
        Table of quantities corresponding to input field.
    host_list
        List of host halos.
    tophost
        halo_ids of the top-level host halo for each halo.
    children : list of sets
        Each set corresponds to the one-level down children of each halo.
    galaxy_list
        List of all galaxies (as long as n_star > 0).
    lumi_galaxy_list
        List of all luminous galaxies (self_m_star > galaxy_low_limit).
    galaxies : list of sets
        Each set corresponds to the embeded galaxies of each halo. All 
        the subhalos will not be considered and will have an empty set. 
        And for host halos it will include all the galaxies within it, 
        including the galaxies actually embedded in the subhalo (i.e., 
        the children of subhalo).
    lumi_galaxies
        Each set corresponds to the embeded luminous galaxies of each 
        halo. Same as `galaxies`, only care about host halo and include 
        all the luminous galaxies within.
    n_lgal
        Number of total luminous galaxies embedded in each halo. Again, 
        only care about host halos and the galaxies within subhalos 
        (i.e., subhalos themselves) will also be taken into account.
    group_list
        halo_id of the halo identified as group in the catalogue.
    '''
    def __init__(self, halocatalogue, datatype, field=default_field,
                    host_id_of_top_level=0, verbose=True, nthreads=1):
        '''
        Initialization routine.

        Input
        -----
        halocatalogue : pynbody.halo.HaloCatalogue
            Only has been tested for pynbody.halo.AHFCatalogue
        field
            Quantities to calculate. When changing specific_mass_field, 
            luminosity_field and temp_field, source codes must be modified.
        datatype : str
            What kind of simulation data you are dealing with. 
            Accepted datatype for now: 'gizmo_ahf' and 'tipsy_ahf'.
        host_id_of_top_level
            How catalogue record "hostHalo" for those halos 
            without a host halo. Default is 0.
        nthreads : int
            Default number of cpus to be used during massive calculations.
        '''
        self.datatype=datatype
        self.catalogue_original = halocatalogue
        self.length = len(self.catalogue_original)
        init_zeros = np.zeros(self.length)
        self.host_id_of_top_level = host_id_of_top_level
        self.errorlist = [{}, {}, {}]
        self.verbose = verbose

        self.rho_crit = pnb.analysis.cosmology.rho_crit(f=self.catalogue_original[1], unit='Msol kpc**-3')
        self.ovdens = cosmology.Delta_vir(self.catalogue_original[1])
        self.nthreads = nthreads
        self.dict = []
        k = 0
        for j in range(self.length):
            i = j + 1
            prop = self.catalogue_original[i].properties
            # currently pynbody can not load the *substructure files 
            # generated by MPI AHF correctly, so here we remove 
            # substructure info even if it can be successfully loaded 
            # with the none-MPI AHF files.
            prop.pop('children', None)
            prop.pop('parentid', None)

            hid = prop['halo_id']
            if i != hid:
                raise Exception('Attention! halo_id doesn\'t equal i !!!')
            self.dict.append(prop)

            if ((i // 100) != (k // 100)) and self.verbose:
                print('Loading properties... {:7} / {}'.format(i, self.length), end='\r')
            k = i

        self.dict = Table(self.dict)
        self.haloid = self.dict['halo_id']
        IDs = self.dict['#ID']
        self.ID_list = Table([IDs, self.haloid], names=['#ID', 'halo_id'])
        self.ID_list.add_row([host_id_of_top_level, host_id_of_top_level])
        
        self.ID_list.add_index('#ID')

        host_in_IDlist = np.isin(self.dict['hostHalo'], self.ID_list['#ID'])
        # Some hostHalo id will not be listed in #ID list, this is probably due to AHF algorithm
        in_idx, = np.where(host_in_IDlist)
        _not_in_ = np.invert(host_in_IDlist)
        not_in_idx, = np.where(_not_in_)
        self.hostid = np.zeros(self.length, dtype=np.int)
        self.hostid[in_idx] = self.ID_list.loc[self.dict['hostHalo'][in_idx]]['halo_id']
        self.hostid = np.ma.array(self.hostid, dtype=np.int, mask=_not_in_)
        # loc method enables using #ID as index
        if len(not_in_idx) > 0:
            for error in not_in_idx:
                self.errorlist[0][self.haloid[error]] = self.dict['hostHalo'][error]

        # prop initialization
        self.prop = {}
        for field_type in default_units:
            init_prop_table = [init_zeros for _ in range(len(field[field_type]))]
            self.prop[field_type] = Table(init_prop_table, names=field[field_type])
            # astropy.table.Table is only used for generating a structured array more conveniently
            self.prop[field_type] = pnb.array.SimArray(self.prop[field_type], units=default_units[field_type])
        if field is None:
            self.field = default_field
        else:
            self.field = field
        self.field_units = default_units

        self._have_children = False
        self._have_galaxy = False
        self._have_group = False
        self._have_radii = False
        self._have_temp = False
        self._have_new_catalogue = False
        self._have_center = False

    def init_relationship(self, galaxy_low_limit, include_sub=False, galaxy_mode='only stellar', N_galaxy=3):
        '''
        Get basic information regarding groups, hosts, children, etc.

        Parameters
        ------------
        galaxy_low_limit : pynbody.array.SimArray
            Required by get_galaxy(). Limit above which galaxies will 
            be identified as luminous galaxies.
        include_sub
            Whether or not to include all the subhalo particles when 
            generating the new catalogue. See get_new_catalogue() for 
            details.
        N_galaxy : int
            Required by get_group_list(). Number of luminous galaxies 
            above which host halos are considered as groups.
        '''
        self.get_children()
        self.get_new_catalogue(include_ = include_sub)
        self.get_galaxy(g_low_limit = galaxy_low_limit, mode=galaxy_mode)
        self.get_group_list(N_galaxy)
        self.get_center()
    
    def calcu_radii_masses(self, halo_id_list=[], rdict=None, precision=1e-2, rmax=None, nthreads=None):
        '''
        Calculate radii (Rvir, R200, etc) and corresponding masses.

        Parameters
        -----------
        halo_id_list
            List of halo_ids to calculate radii and masses. 
            If set to empty list, then will use self.group_list.
        rdict : dict
            names and values for overdensity factors. Default is: 
            {'vir': self.ovdens, '200': 200, '500': 500, '2500': 2500}
        precision : float
            Precision for calculate radius. See get_index() in 
            calculate_R.py documentation for detail.
        rmax
            Maximum value for the shrinking sphere method. See 
            get_radius() in calculate_R.py documentation for detail.
        nthreads : int
            Number of cpus to use when doing the calculation.
        '''
        halo_id_list = np.array(halo_id_list, dtype=np.int).reshape(-1)
        if len(halo_id_list) == 0:
            if not self._have_group:
                raise Exception('Must get_group_list (or init_relationship) first!')
            halo_id_list = self.group_list
        if not self._have_center:
            raise Exception('Must get_center first!')
        
        if rdict == None:
            rdict = {'vir': self.ovdens, '200': 200, '500': 500, '2500': 2500}

        if nthreads is None:
            nthreads = self.nthreads
        
        def func_to_pool(j):
            i = j - 1
            
        
        t1 = 0; t2 = 0
        list_length = np.array(list(halo_id_list)).max()
        k = 0
        for j in halo_id_list:
            i = j - 1
            prop = self.dict[i]
            t1 = time.time()
            MassRadii = cR.get_radius(self.new_catalogue[j], \
                    overdensities=list(rdict.values()), rho_crit=self.rho_crit, \
                        prop=prop, precision=precision, cen=self.center[i], rmax=rmax)
            for key in rdict:
                self.prop['R'][key][i] = MassRadii[1][rdict[key]]
                self.prop['M'][key][i] = MassRadii[0][rdict[key]]
            t2 = time.time()
            if ((i // 100) != (k // 100)) and self.verbose:
                print('Calculating radii and masses... {:7} / {}, time: \
                        {:.5f}s'.format(j, list_length, t2 - t1), end='\r')
            k = i
        self._have_radii = True
    
    def calcu_specific_masses(self, halo_id_list=[], \
                calcu_field=radii_to_cal_sepcific_mass, \
                    temp_cut='5e5 K', nh_cut='0.13 cm**-3'):
        '''
        Calculate some specific masses, such as baryon, IGrM, etc.

        Parameters
        -----------
        halo_id_list
            List of halo_ids to calculate masses. 
            If set to empty list, then will use self.group_list.
        calcu_field
            Radii to calculate specific masses within.
        temp_cut
            Temperature limit above which gas particles are 
            considered as hot.
        nh_cut
            nh limit above which gas particles are considered 
            as star forming.
        '''
        halo_id_list = np.array(halo_id_list, dtype=np.int).reshape(-1)
        if len(halo_id_list) == 0:
            if not self._have_group:
                raise Exception('Must get_group_list (or init_relationship) first!')
            halo_id_list = self.group_list
        if not self._have_radii:
            raise Exception('Must get_radii_masses first!')
        
        list_length = np.array(list(halo_id_list)).max()
        k = 0
        for j in halo_id_list:
            i = j - 1

            prop = self.dict[i]
            center = self.center[i]
            halo = self.new_catalogue[j]
            tx = pnb.transformation.inverse_translate(halo, center)
            with tx:
                boxsize = halo.properties['boxsize'].in_units('kpc')
                original_pos = halo['pos'].copy()
                halo['pos'] = correct_pos(halo['pos'], boxsize)

                for r in calcu_field:
                    # Apply filters
                    subsim = halo[pnb.filt.Sphere(self.prop['R'][i:i+1][r].in_units('kpc'))]
                    cold_diffuse_gas = subsim.gas[pnb.filt.LowPass('temp', temp_cut) & \
                            pnb.filt.LowPass('nh', nh_cut)]
                    ISM = subsim.gas[pnb.filt.HighPass('nh', nh_cut)]
                    hot_diffuse_gas_ = subsim.gas[pnb.filt.HighPass('temp', temp_cut) & \
                            pnb.filt.LowPass('nh', nh_cut)]
                    
                    # Calculate masses
                    self.prop['M']['star' + r][i] = subsim.star['mass'].sum()
                    self.prop['M']['gas' + r][i] = subsim.gas['mass'].sum()
                    self.prop['M']['bar' + r][i] = self.prop['M']['star' + r][i] \
                                + self.prop['M']['gas' + r][i]
                    self.prop['M']['ism' + r][i] = ISM['mass'].sum()
                    self.prop['M']['cold' + r][i] = cold_diffuse_gas['mass'].sum() \
                                + self.prop['M']['ism' + r][i]
                    self.prop['M']['igrm' + r][i] = hot_diffuse_gas_['mass'].sum()
                
                halo['pos'] = original_pos
            if ((i // 100) != (k // 100)) and self.verbose:
                print('Calculating specific masses... {:7} / {}'.format(j, list_length), end='\r')
            k = i

    def calcu_temp_lumi(self, cal_file, halo_id_list=[], \
                    core_corr_factor=0.15, calcu_field='500', \
                    temp_cut='5e5 K', nh_cut='0.13 cm**-3', \
                    additional_filt=None):
        '''
        Calculate all the temperatures and luminosities listed in
        temp_field and luminosity_field. 

        Parameters
        -----------
        cal_file
            Calibration file used for calculating Tspec.
        halo_id_list
            List of halo_ids to calculate temperatures 
            and luminosities. If set to empty list, then will use 
            self.group_list.
        core_corr_factor
            Inner radius for calculating core-corrected 
            temperatures. Gas particles within 
            (core_corr_factor*R, R) will be used for calculation.
        calcu_field
            Radius to calculate temperatures and luminosities 
            within. Must be in radius_field. Default: R_500.
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
        halo_id_list = np.array(halo_id_list, dtype=np.int).reshape(-1)
        if len(halo_id_list) == 0:
            if not self._have_group:
                raise Exception('Must get_group_list (or init_relationship) first!')
            halo_id_list = self.group_list
        if not self._have_radii:
            raise Exception('Must get_radii_masses first!')
        
        list_length = np.array(list(halo_id_list)).max()
        k = 0
        for j in halo_id_list:
            i = j - 1
            center = self.center[i]
            halo = self.new_catalogue[j]
            R = self.prop['R'][i:i+1][calcu_field].in_units('kpc')
            tx = pnb.transformation.inverse_translate(halo, center)
            with tx:
                boxsize = halo.properties['boxsize'].in_units('kpc')
                original_pos = halo['pos'].copy()
                halo['pos'] = correct_pos(halo['pos'], boxsize)

                subsim = halo[pnb.filt.Sphere(R)]
                if additional_filt is None:
                    hot_diffuse_filt = pnb.filt.HighPass('temp', temp_cut) & \
                            pnb.filt.LowPass('nh', nh_cut)
                else:
                    hot_diffuse_filt = pnb.filt.HighPass('temp', temp_cut) & \
                            pnb.filt.LowPass('nh', nh_cut) & additional_filt
                hot_diffuse_gas_ = subsim.gas[hot_diffuse_filt]
                # cal_tweight can return the sum of weight_type at the same time.
                self.prop['T']['x'][i], self.prop['L']['x'][i] = \
                        cal_tweight(hot_diffuse_gas_, weight_type='Lx')
                self.prop['T']['x_cont'][i], self.prop['L']['x_cont'][i] = \
                        cal_tweight(hot_diffuse_gas_, weight_type='Lx_cont')
                self.prop['T']['mass'][i], _= cal_tweight(hot_diffuse_gas_, weight_type='mass')
                self.prop['T']['spec'][i] = pnb.array.SimArray(cal_tspec(hot_diffuse_gas_, \
                                cal_f=cal_file, datatype=self.datatype), units='keV')
                self.prop['T']['xb'][i], self.prop['L']['xb'][i] = \
                        cal_tweight(hot_diffuse_gas_, weight_type='Lxb')
                self.prop['L']['xb_cont'][i] = hot_diffuse_gas_['Lxb_cont'].sum()

                # Core-corrected temperatures:
                # Filter:
                corr_hot_ = hot_diffuse_gas_[~pnb.filt.Sphere(core_corr_factor*R)]

                self.prop['T']['spec_corr'][i] = pnb.array.SimArray(cal_tspec(corr_hot_, \
                                cal_f=cal_file, datatype=self.datatype), units='keV')
                self.prop['T']['x_corr'][i], self.prop['L']['x_corr'][i] = cal_tweight(corr_hot_, weight_type='Lx')
                self.prop['T']['xb_corr'][i], self.prop['L']['xb_corr'][i] = cal_tweight(corr_hot_, weight_type='Lxb')
                self.prop['T']['x_corr_cont'][i], _ = \
                                        cal_tweight(corr_hot_, weight_type='Lx_cont')
                self.prop['T']['mass_corr'][i], _ = cal_tweight(corr_hot_, weight_type='mass')
            
                halo['pos'] = original_pos
            if ((i // 100) != (k // 100)) and self.verbose:
                print('Calculating temperatures and luminosities... {:7} / {}'\
                            .format(j, list_length), end='\r')
            k = i

        self._have_temp = True

    def calcu_entropy(self, cal_file, n_par=9, halo_id_list=[], \
                calcu_field=entropy_field, thickness=0.05, volume_type='full', \
                temp_cut='5e5 K', nh_cut='0.13 cm**-3', additional_filt=None):
        '''
        Calculate all entropy within a thin spherical shell 
        centered at halo.

        Parameters
        -----------
        cal_file
            Calibration file used for calculating Tspec.
        n_par : int
            Number of particles the shell must contain, 
            below which entropy will not be calculated.
        halo_id_list
            List of halo_ids to calculate entropies. 
            If set to empty list, then will use self.group_list.
        calcu_field
            Radii of the thin shell to calculate entropies.
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
        # thickness = pnb.array.SimArray(thickness, 'kpc')
        halo_id_list = np.array(halo_id_list, dtype=np.int).reshape(-1)
        if len(halo_id_list) == 0:
            if not self._have_group:
                raise Exception('Must get_group_list (or init_relationship) first!')
            halo_id_list = self.group_list
        if not self._have_radii:
            raise Exception('Must get_radii_masses first!')

        list_length = np.array(list(halo_id_list)).max()
        k = 0
        for j in halo_id_list:
            i = j - 1
            center = self.center[i]
            halo = self.new_catalogue[j]
            tx = pnb.transformation.inverse_translate(halo, center)
            with tx:
                boxsize = halo.properties['boxsize'].in_units('kpc')
                original_pos = halo['pos'].copy()
                halo['pos'] = correct_pos(halo['pos'], boxsize)

                for r in calcu_field:
                    R = self.prop['R'][i:i+1][r].in_units('kpc')
                    subgas = halo.gas[pnb.filt.Annulus(R, (thickness + 1) * R)]
                    if additional_filt is None:
                        hot_diffuse_filt = pnb.filt.HighPass('temp', temp_cut) & \
                                pnb.filt.LowPass('nh', nh_cut)
                    else:
                        hot_diffuse_filt = pnb.filt.HighPass('temp', temp_cut) & \
                                pnb.filt.LowPass('nh', nh_cut) & additional_filt
                    hot_diffuse_gas_ = subgas[hot_diffuse_filt]
                    if len(hot_diffuse_gas_) < n_par:
                        self.prop['S'][r][i] = np.nan
                        self.prop['T']['spec' + r][i] = np.nan
                    else:
                        tempTspec = pnb.array.SimArray(cal_tspec(hot_diffuse_gas_, \
                                cal_f=cal_file, datatype=self.datatype), units='keV')
                        if volume_type == 'gas':
                            temp_volume = hot_diffuse_gas_['volume'].sum()
                        elif volume_type == 'full':
                            temp_volume = 4/3*np.pi*(((thickness + 1) * R)**3 - R**3)
                        elif volume_type == 'full_approx':
                            temp_volume = 4*np.pi*R**2*thickness*R
                        else:
                            raise Exception("volume_type is not accepted!")
                        avg_ne = ((hot_diffuse_gas_['ne'] * hot_diffuse_gas_['volume']).sum() \
                                / temp_volume).in_units('cm**-3')
                        avg_nh = ((hot_diffuse_gas_['nh'] * hot_diffuse_gas_['volume']).sum() \
                                / temp_volume).in_units('cm**-3')
                        self.prop['T']['spec' + r][i] = tempTspec
                        self.prop['ne'][r][i] = avg_ne
                        self.prop['nh'][r][i] = avg_nh
                        self.prop['S'][r][i] = tempTspec/(avg_ne)**(2, 3)
                
                halo['pos'] = original_pos
            if ((i // 100) != (k // 100)) and self.verbose:
                print('            Calculating entropies... {:7} / {}'\
                            .format(j, list_length), end='\r')
            k = i

    def calcu_metallicity(self, halo_id_list=[], elements=['H', 'O', 'Si', 'Fe'], \
                radii=['500'], temp_cut='5e5 K', nh_cut='0.13 cm**-3', \
                additional_filt=None, weight_types=['mass', 'Lx']):

        if self.datatype[:5] == 'gizmo':
            self.metal_idx = {'He': 1, 'C': 2, 'N': 3, 'O': 4, \
                'Ne': 5, 'Mg': 6, 'Si': 7, 'S': 8, 'Ca': 9, 'Fe': 10}
        else:
            raise Exception('Currently only support GIZMO.')
        init_zeros = np.zeros(self.length)
        field_names = []
        for ele in elements:
            for rad in radii:
                for weight_type in weight_types:
                    field_names.append('Z_' + ele + rad + weight_type)
        init_prop_table = Table([init_zeros for _ in range(len(field_names))])
        self.prop['metals'] = Table(init_prop_table, names=field_names)
        self.prop['metals'] = pnb.array.SimArray(self.prop['metals'], units='cm**-3')
        self.field['metals'] = field_names
        self.field_units['metals'] = 'cm**-3'

        halo_id_list = np.array(halo_id_list, dtype=np.int).reshape(-1)
        if len(halo_id_list) == 0:
            if not self._have_group:
                raise Exception('Must get_group_list (or init_relationship) first!')
            halo_id_list = self.group_list
        if not self._have_radii:
            raise Exception('Must get_radii_masses first!')

        list_length = np.array(list(halo_id_list)).max()
        k = 0
        for j in halo_id_list:
            i = j - 1
            center = self.center[i]
            halo = self.new_catalogue[j]
            tx = pnb.transformation.inverse_translate(halo, center)
            with tx:
                boxsize = halo.properties['boxsize'].in_units('kpc')
                original_pos = halo['pos'].copy()
                halo['pos'] = correct_pos(halo['pos'], boxsize)

                for r in radii:
                    R = self.prop['R'][i:i+1][r].in_units('kpc')
                    subgas = halo.gas[pnb.filt.Sphere(R)]
                    if additional_filt is None:
                        hot_diffuse_filt = pnb.filt.HighPass('temp', temp_cut) & \
                                pnb.filt.LowPass('nh', nh_cut)
                    else:
                        hot_diffuse_filt = pnb.filt.HighPass('temp', temp_cut) & \
                                pnb.filt.LowPass('nh', nh_cut) & additional_filt
                    igrm = subgas[hot_diffuse_filt]
                    for weight_type in weight_types:
                        weight_sum = igrm[weight_type].sum()

                        for ele in elements:
                            if ele != 'H':
                                # gas_nx = g_p.n_X(igrm['rho'], igrm['metals'][:, self.metal_idx[ele]], ele)
                                totZx = (igrm['metals'][:, self.metal_idx[ele]] * igrm[weight_type]).sum()
                            else:
                                # totNx = (igrm['nh'] * igrm[weight_type]).sum()
                                totZx = (igrm['X_H'] * igrm[weight_type]).sum()
                            self.prop['metals']['Z_' + ele + r + weight_type][i] = \
                                            (totZx/weight_sum)
                halo['pos'] = original_pos
            if ((i // 100) != (k // 100)) and self.verbose:
                print('            Calculating metallicities... {:7} / {}'\
                            .format(j, list_length), end='\r')
            k = i

    def savedata(self, filename, field = None, halo_id_list=[], units=None):
        '''
        Save the data in hdf5 format. Will save halo_id_list 
        (key: 'halo_id') and the quantities listed in field.

        Parameters
        -----------
        filename
            Filename of the hdf5 file.
        field
            Type of information to save.
        halo_id_list
            List of halo_ids to save.If set to empty list, 
            then will use self.group_list.
        units
            Convert the data into specified inits and save.
        '''
        if field is None:
            field = self.field
        if units is None:
            field_units = self.field_units
        halo_id_list = np.array(halo_id_list, dtype=np.int).reshape(-1)
        if len(halo_id_list) == 0:
            halo_id_list = self.group_list
        with h5py.File(filename, "w") as f:
            dataset = f.create_dataset("halo_id", data = halo_id_list)
            dataset.attrs['Description'] = 'halo_ids of halos saved in this file.'
            dataset2 = f.create_dataset("N_lgal", data = self.n_lgal[halo_id_list - 1])
            dataset2.attrs['Description'] = 'Number of luminous galaxies'
            for attr in field:
                grp = f.create_group(attr)
                infos = field[attr]
                for info in infos:
                    data_to_save = self.prop[attr][info][halo_id_list - 1]
                    data_to_save.convert_units(field_units[attr])
                    dset = grp.create_dataset(info, data = data_to_save)
                    dset.attrs['units'] = str(data_to_save.units)

    def get_children(self):
        '''
        Generate list of children (subhalos) for each halo.
        Subhalo itself can also have children. And this list 
        will not contain "grandchildren" (i.e., the children 
        of children).
        '''
        self.host_list = []
        self.tophost = np.zeros(self.length).astype(np.int)
        self.children = [set() for _ in range(self.length)]
        k = 0
        for i in range(self.length):
            j = self.haloid[i]#j = i + 1
            if ((j // 100) != (k // 100)) and self.verbose:
                print('Generating children list... Halo: {:7} / {}'.format(j, self.length), end='\r')
            k = j
            prop = self.dict[i]
            hostID = prop['hostHalo']
            if j in self.errorlist[0]:
                self.errorlist[1][j] = hostID
                continue
            try:
                if hostID == self.host_id_of_top_level:
                    self.host_list.append(j)
                    self.tophost[i] = j
                else:
                    if hostID < 0:
                        print('Make sure you\'ve used the correct host ID of the top-level halos!')
                    host_haloid = self.ID_list.loc[hostID]['halo_id']
                    self.children[host_haloid - 1].add(j)
                    temphost = j
                    while temphost != self.host_id_of_top_level:
                        temphost2 = temphost
                        temphost = self.hostid[temphost - 1]
                    self.tophost[i] = temphost2
            except IndexError:
                self.errorlist[1][j] = hostID
        self._have_children = True
    
    def get_new_catalogue(self, include_):
        '''
        Generate a new catalogue based on catalogue_original, 
        the new catalogue will include all the subhalo particles 
        in its host halo.
        
        Parameters
        -------------
        include_ : bool
            If True, then will include all the subhalo particles. 
            Otherwise will just be a copy of catalogue_original.
        '''
        if not self._have_children:
            raise Exception('Must get_children first!')
        if include_:
            self.new_catalogue = {}
            k = 0
            for i in range(self.length):
                j = self.haloid[i]
                if ((i // 100) != (k // 100)) and self.verbose:
                    print('Generating new catalogue... Halo: {:7} / {}'.format(j, self.length), end='\r')
                    k = i
                if len(self.children[i]) == 0:
                    self.new_catalogue[j] = self.catalogue_original[j]
                else:
                    union_list = [j] + list(self.children[i])
                    self.new_catalogue[j] = get_union(self.catalogue_original, union_list)
        else:
            self.new_catalogue = self.catalogue_original
        self._have_new_catalogue = True

    def get_galaxy(self, g_low_limit, mode='only stellar'):
        '''
        Generate list of galaxies for each host halo. The subsubhalo 
        will also be included in the hosthalo galaxy list. And it won't 
        generate list for the subhalos even if there are galaxies within.

        Parameters
        -------------
        g_low_limit : pynbody.array.SimArray
            Limit above which galaxies will be identified as luminous 
            galaxies.
        '''
        if not self._have_children:
            raise Exception('Must get_children first!')
        if not self._have_new_catalogue:
            raise Exception('Must get_new_catalogue first!')

        self.galaxy_list = [] # List of all galaxies (as long as n_star > 0).
        self.lumi_galaxy_list = [] # List of all luminous galaxies (self_m_star > galaxy_low_limit).
        self.galaxies = [set() for _ in range(self.length)]
        self.lumi_galaxies = [set() for _ in range(self.length)]
        self.n_lgal = np.zeros(self.length) # Number of total luminous galaxies embedded in each host halo.
        # The galaxies within subhalos (i.e., subhalos themselves) will also be taken into account.
        
        k = 0
        for i in range(self.length):
            j = self.haloid[i]
            if ((i // 100) != (k // 100)) and self.verbose:
                print('Calculating total stellar masses... Halo: {:7} / {}'.format(j, self.length), end='\r')
                k = i
            self.prop['M']['total_star'][i] = self.new_catalogue[j].star['mass'].sum()
            sf_gas = self.new_catalogue[j].gas[pnb.filt.LowPass('temp', '3e4 K')]
            # sf_gas = self.new_catalogue[j].gas[pnb.filt.HighPass('nh', '0.13 cm**-3')]
            self.prop['M']['total_sfgas'][i] = sf_gas['mass'].sum()
            # sf_gas, i.e., star forming gas, is used in the definition of resolved galaxies in Liang's Figure2.
            # But seems that Liang didn't plot Figure 2 using the concept of resolved galaxies.
        low_limit = g_low_limit.in_units(self.prop['M']['total_star'].units)
        k = 0
        for i in range(self.length):
            j = self.haloid[i]
            if ((i // 100) != (k // 100)) and self.verbose:
                print('            Identifying galaxies... Halo: {:7} / {}'.format(j, self.length), end='\r')
                k = i
            children_list = np.array(list(self.children[i]))
            if len(children_list) == 0:
                self_Mstar = self.prop['M']['total_star'][i]
                # if mode == 'include cold gas':
                self_Msfgas = self.prop['M']['total_sfgas'][i]
            else:
                children_union = get_union(self.new_catalogue, list(children_list))
                children_union_within_ = self.new_catalogue[j].intersect(children_union)
                self_Mstar = self.prop['M']['total_star'][i] - children_union_within_.star['mass'].sum()
                # if mode == 'include cold gas':
                sf_gas_union = children_union_within_.gas[pnb.filt.LowPass('temp', '3e4 K')]
                # sf_gas_union = children_union.gas[pnb.filt.HighPass('nh', '0.13 cm**-3')]
                self_Msfgas = self.prop['M']['total_sfgas'][i] - sf_gas_union['mass'].sum()
            self.prop['M']['self_star'][i] = self_Mstar
            self.prop['M']['self_sfgas'][i] = self_Msfgas
            try:
                if mode == 'only stellar':
                    condition = (self_Mstar > 0)
                elif mode == 'include cold gas':
                    condition = (self_Mstar + self_Msfgas > low_limit)
                if condition:
                    self.galaxy_list.append(j)
                    temp_tophost = self.tophost[i]
                    self.galaxies[temp_tophost-1].add(j)
        
                    if self_Mstar > low_limit:
                        self.lumi_galaxy_list.append(j)
                        self.n_lgal[temp_tophost-1] += 1
                        self.lumi_galaxies[temp_tophost-1].add(j)
            except KeyError:
                self.errorlist[2][j] = self.dict['hostHalo'][i]
        self._have_galaxy = True

    def get_group_list(self, N_galaxy):
        '''
        halo_id of the halo identified as group in the catalogue. 

        Parameters
        -----------
        N_galaxy : int
            Number of luminous galaxies above which host halos 
            are considered as groups.
        '''
        if not self._have_galaxy:
            raise Exception('Must get_galaxy first!')
        self.group_list, = np.where(self.n_lgal >= N_galaxy)
        self.group_list += 1
        self._have_group = True
    
    def calcu_tx_lx(self, halo_id_list=[], \
                    core_corr_factor=0.15, calcu_field='500', \
                    temp_cut='5e5 K', nh_cut='0.13 cm**-3', additional_filt=None):
        '''
        Calculate X-ray luminosities and emission weighted 
        temperatures listed in temp_field and luminosity_field. 

        Parameters
        -----------
        halo_id_list
            List of halo_ids to calculate temperatures on. 
            If set to empty list, then will use self.group_list.
        core_corr_factor
            Inner radius for calculating core-corrected 
            temperatures. Gas particles within 
            (core_corr_factor*R, R) will be used for calculation.
        calcu_field
            Radius to calculate temperatures and luminosities 
            within. Must be in radius_field. Default: R_500.
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
        halo_id_list = np.array(halo_id_list, dtype=np.int).reshape(-1)
        if len(halo_id_list) == 0:
            if not self._have_group:
                raise Exception('Must get_group_list (or init_relationship) first!')
            halo_id_list = self.group_list
        if not self._have_radii:
            raise Exception('Must get_radii_masses first!')
        if not self._have_new_catalogue:
            raise Exception('Must get_new_catalogue first!')
        
        list_length = np.array(list(halo_id_list)).max()
        for j in halo_id_list:
            i = j - 1
            if self.verbose:
                print('Calculating temperatures and luminosities... {:7} / {}'\
                            .format(j, list_length), end='\r')
            center = self.center[i]
            halo = self.new_catalogue[j]
            R = self.prop['R'][i:i+1][calcu_field].in_units('kpc')
            tx = pnb.transformation.inverse_translate(halo, center)
            with tx:
                boxsize = halo.properties['boxsize'].in_units('kpc')
                original_pos = halo['pos'].copy()
                halo['pos'] = correct_pos(halo['pos'], boxsize)

                subsim = halo[pnb.filt.Sphere(R)]
                if additional_filt is None:
                    hot_diffuse_filt = pnb.filt.HighPass('temp', temp_cut) & \
                            pnb.filt.LowPass('nh', nh_cut)
                else:
                    hot_diffuse_filt = pnb.filt.HighPass('temp', temp_cut) & \
                            pnb.filt.LowPass('nh', nh_cut) & additional_filt
                hot_diffuse_gas_ = subsim.gas[hot_diffuse_filt]
                # cal_tweight can return the sum of weight_type at the same time.
                self.prop['T']['x'][i], self.prop['L']['x'][i] = \
                        cal_tweight(hot_diffuse_gas_, weight_type='Lx')
                self.prop['T']['x_cont'][i], self.prop['L']['x_cont'][i] = \
                        cal_tweight(hot_diffuse_gas_, weight_type='Lx_cont')

                # Core-corrected temperatures:
                # Filter:
                corr_hot_ = hot_diffuse_gas_[~pnb.filt.Sphere(core_corr_factor*R)]

                self.prop['T']['x_corr'][i], _ = cal_tweight(corr_hot_, weight_type='Lx')
                self.prop['T']['x_corr_cont'][i], _ = \
                                        cal_tweight(corr_hot_, weight_type='Lx_cont')

                halo['pos'] = original_pos

    def calcu_tspec(self, cal_file, halo_id_list=[], \
                    core_corr_factor=0.15, calcu_field='500', temp_cut='5e5 K', \
                    nh_cut='0.13 cm**-3', additional_filt=None):
        '''
        Calculate spectroscopic temperatures based on Douglas's 
        pytspec module.

        Parameters
        -----------
        cal_file
            Calibration file used for calculating Tspec.
        halo_id_list
            List of halo_ids to calculate temperatures and 
            luminosities. If set to empty list, then will use 
            self.group_list.
        core_corr_factor
            Inner radius for calculating core-corrected temperatures. 
            Gas particles within (core_corr_factor*R, R) will be used 
            for calculation.
        calcu_field
            Radius to calculate temperatures and luminosities within. 
            Must be in radius_field. Default: R_500.
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
        halo_id_list = np.array(halo_id_list, dtype=np.int).reshape(-1)
        if len(halo_id_list) == 0:
            if not self._have_group:
                raise Exception('Must get_group_list (or init_relationship) first!')
            halo_id_list = self.group_list
        if not self._have_radii:
            raise Exception('Must get_radii_masses first!')
        if not self._have_new_catalogue:
            raise Exception('Must get_new_catalogue first!')
        
        list_length = np.array(list(halo_id_list)).max()
        for j in halo_id_list:
            i = j - 1
            if self.verbose:
                print('Calculating spectroscopic temperatures... {:7} / {}'\
                            .format(j, list_length), end='\r')
            center = self.center[i]
            halo = self.new_catalogue[j]
            R = self.prop['R'][i:i+1][calcu_field].in_units('kpc')
            tx = pnb.transformation.inverse_translate(halo, center)
            with tx:
                boxsize = halo.properties['boxsize'].in_units('kpc')
                original_pos = halo['pos'].copy()
                halo['pos'] = correct_pos(halo['pos'], boxsize)

                subsim = halo[pnb.filt.Sphere(R)]
                if additional_filt is None:
                    hot_diffuse_filt = pnb.filt.HighPass('temp', temp_cut) & \
                            pnb.filt.LowPass('nh', nh_cut)
                else:
                    hot_diffuse_filt = pnb.filt.HighPass('temp', temp_cut) & \
                            pnb.filt.LowPass('nh', nh_cut) & additional_filt
                hot_diffuse_gas_ = subsim.gas[hot_diffuse_filt]
                self.prop['T']['spec'][i] = pnb.array.SimArray(cal_tspec(hot_diffuse_gas_, \
                                cal_f=cal_file, datatype=self.datatype), units='keV')
                # Core-corrected temperatures:
                # Filter:
                corr_hot_ = hot_diffuse_gas_[~pnb.filt.Sphere(core_corr_factor*R)]
                self.prop['T']['spec_corr'][i] = pnb.array.SimArray(cal_tspec(corr_hot_, \
                                cal_f=cal_file, datatype=self.datatype), units='keV')

                halo['pos'] = original_pos
    def get_center(self):
        '''
        Calculate the center of the halos if an ahfcatalogue is 
        provided, then will automatically load the results in ahf. 
        Otherwise it will try to calculate the center coordinates 
        via gravitional potential or center of mass.

        Notes
        ------
        Due to a bug in pynbody, calculating center of mass will 
        lead to an incorrect result for the halos crossing the 
        periodical boundary of the simulation box. Make sure pynbody 
        has fixed it before you use.
        '''
        if self.datatype[-4:] == '_ahf':
            axes = ['Xc', 'Yc', 'Zc']
            tempcen = {}
            for axis in axes:
                tempcen[axis] = np.asarray(self.dict[axis], dtype=float).reshape(-1, 1)
            self.center = np.concatenate((tempcen['Xc'], tempcen['Yc'], tempcen['Zc']), axis=1)
            self.center = pnb.array.SimArray(self.center, units='kpc') * self.dict['a'][0] / self.dict['h'][0]
            if self.datatype == 'tipsy_ahf':
                self.center -= self.dict['boxsize'][0].in_units('kpc')/2
        else:
            self.center = pnb.array.SimArray(np.zeros((self.length, 3)), units='kpc')
            if 'phi' in self.new_catalogue[1].loadable_keys():
                center_mode = 'pot'
            else:
                center_mode = 'com'
            for i in range(self.length):
                j = self.haloid[i]
                print('Calculating center... {:7} / {}'.format(j, self.length), end='\r')
                self.center[i] = pnb.analysis.halo.center(self.new_catalogue[j], \
                    mode=center_mode, retcen=True, vel=False)
        self._have_center = True
    
def get_union(catalogue, list):
    '''
    Calculate the union of the particles listed in the list.

    Parameters
    -----------
    catalogue : pynbody.halo.HaloCatalogue
    list
        List of halos in the catalogue to get union.
    '''
    temp_halo = catalogue[list[0]]
    for i in list[1:]:
        temp_halo = temp_halo.union(catalogue[i])
    return temp_halo

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
