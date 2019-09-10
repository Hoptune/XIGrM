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

from . import prepare_pyatomdb as ppat
from . import calculate_R as cR
from . import cosmology
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
mass_field = radius_field + specific_mass_field + ['total_star', 'self_star']#, 'total_sfgas', 'self_sfgas']
temp_field = ['x', 'x_cont', 'mass', 'spec', 'spec_corr', \
                'x_corr', 'x_corr_cont', 'mass_corr', 'x500', 'x2500']
entropy_field = ['500', '2500']
luminosity_field = ['x', 'x_cont', 'xb', 'xb_cont']
# _cont for only considering continuum emission, _corr for core-corrected (0.15*R500-R500)
# xb for broad band X-ray (0.1-2.4 keV) in our case.

default_field = {'R': radius_field, 'M': mass_field, 'T': temp_field, 'S': entropy_field, 'L': luminosity_field}
default_units = {'T': 'keV', 'L': 'erg s**-1', 'R': 'kpc', 'M': 'Msol', 'S': 'keV cm**2'}

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
    def __init__(self, halocatalogue, datatype, field=default_field, host_id_of_top_level=0):
        '''
        Initialization routine.

        Input
        -----
        ahfcatalogue : pynbody.halo.HaloCatalogue
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
        '''
        self.datatype=datatype
        self.catalogue_original = halocatalogue
        self.length = len(self.catalogue_original)
        init_zeros = np.zeros(self.length)
        self.host_id_of_top_level = host_id_of_top_level
        self.errorlist = [{}, {}, {}]

        self.rho_crit = pnb.analysis.cosmology.rho_crit(f=self.catalogue_original[1], unit='Msol kpc**-3')
        self.ovdens = cosmology.Delta_vir(self.catalogue_original[1])

        self.dict = []
        for j in range(self.length):
            i = j + 1
            print('Loading properties... {:7} / {}'.format(i, self.length), end='\r')
            prop = self.catalogue_original[i].properties
            hid = prop['halo_id']
            if i != hid:
                raise Exception('Attention! halo_id doesn\'t equal i !!!')
            self.dict.append(prop)

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
        
        self._have_children = False
        self._have_galaxy = False
        self._have_group = False
        self._have_radii = False
        self._have_temp = False
        self._have_new_catalogue = False
        self._have_center = False

    def init_relationship(self, galaxy_low_limit, include_sub=False, N_galaxy=3):
        '''
        Get basic information regarding groups, hosts, children, etc.

        Parameters
        ------------
        galaxy_low_limit : pynbody.array.SimArray
            Required by get_galaxyh(). Limit above which galaxies will 
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
        self.get_galaxy(g_low_limit = galaxy_low_limit)
        self.get_group_list(N_galaxy)
        self.get_center()
    
    def calcu_radii_masses(self, halo_id_list=np.array([]), rdict=None, precision=1e-2, rmax=None):
        '''
        Calculate radii (Rvir, R200, etc) and corresponding masses.

        Parameters
        -----------
        halo_id_list
            List of halo_ids to calculate radii and masses. 
            If set to None, then will use self.group_list.
        rdict
            names and values for overdensity factors. Default is: 
            {'vir': self.ovdens, '200': 200, '500': 500, '2500': 2500}
        precision
            Precision for calculate radius. See get_index() in 
            calculate_R.py documentation for detail.
        rmax
            Maximum value for the shrinking sphere method. See 
            get_index() in calculate_R.py documentation for detail.
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
        t1 = 0; t2 = 0
        list_length = np.array(list(halo_id_list)).max()
        for j in halo_id_list:
            i = j - 1
            print('Calculating radii and masses... {:7} / {}, time: \
                        {:.5f}s'.format(j, list_length, t2 - t1), end='\r')
            prop = self.dict[i]
            t1 = time.time()
            MassRadii = cR.get_radius(self.new_catalogue[j], \
                    overdensities=list(rdict.values()), rho_crit=self.rho_crit, \
                        prop=prop, precision=precision, cen=self.center[i], rmax=rmax)
            for key in rdict:
                self.prop['R'][key][i] = MassRadii[1][rdict[key]]
                self.prop['M'][key][i] = MassRadii[0][rdict[key]]
            t2 = time.time()
        self._have_radii = True
    
    def calcu_specific_masses(self, halo_id_list=np.array([]), \
                calcu_field=radii_to_cal_sepcific_mass):
        '''
        Calculate some specific masses, such as baryon, IGrM, etc.

        Parameters
        -----------
        halo_id_list
            List of halo_ids to calculate masses. 
            If set to None, then will use self.group_list.
        calcu_field
            Radii to calculate specific masses within.
        '''
        halo_id_list = np.array(halo_id_list, dtype=np.int).reshape(-1)
        if len(halo_id_list) == 0:
            if not self._have_group:
                raise Exception('Must get_group_list (or init_relationship) first!')
            halo_id_list = self.group_list
        if not self._have_radii:
            raise Exception('Must get_radii_masses first!')
        
        list_length = np.array(list(halo_id_list)).max()
        for j in halo_id_list:
            i = j - 1
            print('Calculating specific masses... {:7} / {}'.format(j, list_length), end='\r')
            prop = self.dict[i]
            center = self.center[i]
            halo = self.new_catalogue[j]
            tx = pnb.transformation.inverse_translate(halo, center)
            with tx:
                for r in calcu_field:
                    # Apply filters
                    subsim = halo[pnb.filt.Sphere(self.prop['R'][i:i+1][r].in_units('kpc'))]
                    cold_diffuse_gas = subsim.gas[pnb.filt.LowPass('temp', '5e5 K') & \
                            pnb.filt.LowPass('nh', '0.13 cm**-3')]
                    ISM = subsim.gas[pnb.filt.HighPass('nh', '0.13 cm**-3')]
                    hot_diffuse_gas_ = subsim.gas[pnb.filt.HighPass('temp', '5e5 K') & \
                            pnb.filt.LowPass('nh', '0.13 cm**-3')]
                    
                    # Calculate masses
                    self.prop['M']['star' + r][i] = subsim.star['mass'].sum()
                    self.prop['M']['gas' + r][i] = subsim.gas['mass'].sum()
                    self.prop['M']['bar' + r][i] = self.prop['M']['star' + r][i] \
                                + self.prop['M']['gas' + r][i]
                    self.prop['M']['ism' + r][i] = ISM['mass'].sum()
                    self.prop['M']['cold' + r][i] = cold_diffuse_gas['mass'].sum() \
                                + self.prop['M']['ism' + r][i]
                    self.prop['M']['igrm' + r][i] = hot_diffuse_gas_['mass'].sum()

    def calcu_temp_lumi(self, cal_file, halo_id_list=np.array([]), \
                    core_corr_factor=0.15, calcu_field='500'):
        '''
        Calculate all the temperatures and luminosities listed in
        temp_field and luminosity_field. 

        Parameters
        -----------
        cal_file
            Calibration file used for calculating Tspec.
        halo_id_list
            List of halo_ids to calculate temperatures 
            and luminosities. If set to None, then will use 
            self.group_list.
        core_corr_factor
            Inner radius for calculating core-corrected 
            temperatures. Gas particles within 
            (core_corr_factor*R, R) will be used for calculation.
        calcu_field
            Radius to calculate temperatures and luminosities 
            within. Must be in radius_field. Default: R_500.
        '''
        halo_id_list = np.array(halo_id_list, dtype=np.int).reshape(-1)
        if len(halo_id_list) == 0:
            if not self._have_group:
                raise Exception('Must get_group_list (or init_relationship) first!')
            halo_id_list = self.group_list
        if not self._have_radii:
            raise Exception('Must get_radii_masses first!')
        
        list_length = np.array(list(halo_id_list)).max()
        for j in halo_id_list:
            i = j - 1
            print('Calculating temperatures and luminosities... {:7} / {}'\
                            .format(j, list_length), end='\r')
            center = self.center[i]
            halo = self.new_catalogue[j]
            R = self.prop['R'][i:i+1][calcu_field].in_units('kpc')
            tx = pnb.transformation.inverse_translate(halo, center)
            with tx:
                subsim = halo[pnb.filt.Sphere(R)]
                hot_diffuse_gas_ = subsim.gas[pnb.filt.HighPass('temp', '5e5 K') & \
                            pnb.filt.LowPass('nh', '0.13 cm**-3')]
                # cal_tweight can return the sum of weight_type at the same time.
                self.prop['T']['x'][i], self.prop['L']['x'][i] = \
                        cal_tweight(hot_diffuse_gas_, weight_type='Lx')
                self.prop['T']['x_cont'][i], self.prop['L']['x_cont'][i] = \
                        cal_tweight(hot_diffuse_gas_, weight_type='Lx_cont')
                self.prop['T']['mass'][i], _= cal_tweight(hot_diffuse_gas_, weight_type='mass')
                self.prop['T']['spec'][i] = pnb.array.SimArray(cal_tspec(hot_diffuse_gas_, \
                                cal_f=cal_file, datatype=self.datatype), units='keV')
                self.prop['L']['xb'][i] = hot_diffuse_gas_['Lxb'].sum()
                self.prop['L']['xb_cont'][i] = hot_diffuse_gas_['Lxb_cont'].sum()

                # Core-corrected temperatures:
                # Filter:
                corr_hot_ = hot_diffuse_gas_[~pnb.filt.Sphere(core_corr_factor*R)]

                self.prop['T']['spec_corr'][i] = pnb.array.SimArray(cal_tspec(corr_hot_, \
                                cal_f=cal_file, datatype=self.datatype), units='keV')
                self.prop['T']['x_corr'][i], _ = cal_tweight(corr_hot_, weight_type='Lx')
                self.prop['T']['x_corr_cont'][i], _ = \
                                        cal_tweight(corr_hot_, weight_type='Lx_cont')
                self.prop['T']['mass_corr'][i], _ = cal_tweight(corr_hot_, weight_type='mass')

        self._have_temp = True

    def calcu_entropy(self, cal_file, n_par=9, halo_id_list=np.array([]), \
                calcu_field=entropy_field, thickness=pnb.array.SimArray(1, 'kpc')):
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
            If set to None, then will use self.group_list.
        calcu_field
            Radii of the thin shell to calculate entropies.
        thickness
            Thickness of the spherical shell.
        '''
        halo_id_list = np.array(halo_id_list, dtype=np.int).reshape(-1)
        if len(halo_id_list) == 0:
            if not self._have_group:
                raise Exception('Must get_group_list (or init_relationship) first!')
            halo_id_list = self.group_list
        if not self._have_radii:
            raise Exception('Must get_radii_masses first!')

        list_length = np.array(list(halo_id_list)).max()
        for j in halo_id_list:
            i = j - 1
            print('            Calculating entropies... {:7} / {}'\
                            .format(j, list_length), end='\r')
            center = self.center[i]
            halo = self.new_catalogue[j]
            tx = pnb.transformation.inverse_translate(halo, center)
            with tx:
                for r in calcu_field:
                    R = self.prop['R'][i:i+1][r].in_units('kpc')
                    subgas = halo.gas[pnb.filt.Annulus(R, thickness + R)]
                    hot_diffuse_gas_ = subgas[pnb.filt.HighPass('temp', '5e5 K') \
                            & pnb.filt.LowPass('nh', '0.13 cm**-3')]
                    if len(hot_diffuse_gas_) < n_par:
                        self.prop['S'][r][i] = np.nan
                        self.prop['T']['x' + r][i] = np.nan
                    else:
                        tempTspec = pnb.array.SimArray(cal_tspec(hot_diffuse_gas_, \
                                cal_f=cal_file, datatype=self.datatype), units='keV')
                        avg_ne = (hot_diffuse_gas_['ne'] * hot_diffuse_gas_['volume']).sum() \
                                / hot_diffuse_gas_['volume'].sum()
                        self.prop['T']['x' + r][i] = tempTspec
                        self.prop['S'][r][i] = tempTspec/(avg_ne.in_units('cm**-3'))**(2, 3)

    def savedata(self, filename, field = default_field, halo_id_list=np.array([]), units=default_units):
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
            List of halo_ids to save.If set to None, 
            then will use self.group_list.
        units
            Convert the data into specified inits and save.
        '''
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
                    data_to_save.convert_units(default_units[attr])
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
        for i in range(self.length):
            j = self.haloid[i]#j = i + 1
            print('Generating children list... Halo: {:7} / {}'.format(j, self.length), end='\r')
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
            for i in range(self.length):
                j = self.haloid[i]
                print('Generating new catalogue... Halo: {:7} / {}'.format(j, self.length), end='\r')
                if len(self.children[i]) == 0:
                    self.new_catalogue[j] = self.catalogue_original[j]
                else:
                    union_list = [j] + list(self.children[i])
                    self.new_catalogue[j] = get_union(self.catalogue_original, union_list)
        else:
            self.new_catalogue = self.catalogue_original
        self._have_new_catalogue = True

    def get_galaxy(self, g_low_limit):
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
        

        for i in range(self.length):
            j = self.haloid[i]
            print('Calculating total stellar masses... Halo: {:7} / {}'.format(j, self.length), end='\r')
            self.prop['M']['total_star'][i] = self.new_catalogue[j].star['mass'].sum()
            #sf_gas = self.new_catalogue[j].gas[pnb.filt.LowPass('temp', '3e4 K')]
            # sf_gas = self.new_catalogue[j].gas[pnb.filt.HighPass('nh', '0.13 cm**-3')]
            #self.prop['M']['total_sfgas'][i] = sf_gas['mass'].sum()
            # sf_gas, i.e., star forming gas, is used in the definition of resolved galaxies in Liang's Figure2.
            # But seems that Liang didn't plot Figure 2 using the concept of resolved galaxies.
        low_limit = g_low_limit.in_units(self.prop['M']['total_star'].units)
        for i in range(self.length):
            j = self.haloid[i]
            print('            Identifying galaxies... Halo: {:7} / {}'.format(j, self.length), end='\r')
            children_list = np.array(list(self.children[i]))
            if len(children_list) == 0:
                self_Mstar = self.prop['M']['total_star'][i]
                #self_Msfgas = self.prop['M']['total_sfgas'][i]
            else:
                children_union = get_union(self.new_catalogue, list(children_list))
                children_union_within_ = self.new_catalogue[j].intersect(children_union)
                #sf_gas_union = children_union.gas[pnb.filt.LowPass('temp', '3e4 K')]
                # sf_gas_union = children_union.gas[pnb.filt.HighPass('nh', '0.13 cm**-3')]
                self_Mstar = self.prop['M']['total_star'][i] - children_union_within_.star['mass'].sum()
                #self_Msfgas = self.prop['M']['total_sfgas'][i] - sf_gas_union['mass'].sum()
            self.prop['M']['self_star'][i] = self_Mstar
            # self.prop['M']['self_sfgas'][i] = self_Msfgas
            try:
                if self_Mstar > 0:# + self_Msfgas > low_limit:
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
    
    def calcu_tx_lx(self, halo_id_list=np.array([]), \
                    core_corr_factor=0.15, calcu_field='500'):
        '''
        Calculate X-ray luminosities and emission weighted 
        temperatures listed in temp_field and luminosity_field. 

        Parameters
        -----------
        halo_id_list
            List of halo_ids to calculate temperatures on. 
            If set to None, then will use self.group_list.
        core_corr_factor
            Inner radius for calculating core-corrected 
            temperatures. Gas particles within 
            (core_corr_factor*R, R) will be used for calculation.
        calcu_field
            Radius to calculate temperatures and luminosities 
            within. Must be in radius_field. Default: R_500.
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
            print('Calculating temperatures and luminosities... {:7} / {}'\
                            .format(j, list_length), end='\r')
            center = self.center[i]
            halo = self.new_catalogue[j]
            R = self.prop['R'][i:i+1][calcu_field].in_units('kpc')
            tx = pnb.transformation.inverse_translate(halo, center)
            with tx:
                subsim = halo[pnb.filt.Sphere(R)]
                hot_diffuse_gas_ = subsim.gas[pnb.filt.HighPass('temp', '5e5 K') & \
                            pnb.filt.LowPass('nh', '0.13 cm**-3')]
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
    
    def calcu_tspec(self, cal_file, halo_id_list=np.array([]), \
                    core_corr_factor=0.15, calcu_field='500'):
        '''
        Calculate spectroscopic temperatures based on Douglas's 
        pytspec module.

        Parameters
        -----------
        cal_file
            Calibration file used for calculating Tspec.
        halo_id_list
            List of halo_ids to calculate temperatures and 
            luminosities. If set to None, then will use self.group_list.
        core_corr_factor
            Inner radius for calculating core-corrected temperatures. 
            Gas particles within (core_corr_factor*R, R) will be used 
            for calculation.
        calcu_field
            Radius to calculate temperatures and luminosities within. 
            Must be in radius_field. Default: R_500.
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
            print('Calculating spectroscopic temperatures... {:7} / {}'\
                            .format(j, list_length), end='\r')
            center = self.center[i]
            halo = self.new_catalogue[j]
            R = self.prop['R'][i:i+1][calcu_field].in_units('kpc')
            tx = pnb.transformation.inverse_translate(halo, center)
            with tx:
                subsim = halo[pnb.filt.Sphere(R)]
                hot_diffuse_gas_ = subsim.gas[pnb.filt.HighPass('temp', '5e5 K') & \
                            pnb.filt.LowPass('nh', '0.13 cm**-3')]

                self.prop['T']['spec'][i] = pnb.array.SimArray(cal_tspec(hot_diffuse_gas_, \
                                cal_f=cal_file, datatype=self.datatype), units='keV')
                # Core-corrected temperatures:
                # Filter:
                corr_hot_ = hot_diffuse_gas_[~pnb.filt.Sphere(core_corr_factor*R)]
                self.prop['T']['spec_corr'][i] = pnb.array.SimArray(cal_tspec(corr_hot_, \
                                cal_f=cal_file, datatype=self.datatype), units='keV')
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
                self.center[i] = pnb.analysis.halo.center(self.new_catalogue[j], mode=center_mode, retcen=True, vel=False)
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