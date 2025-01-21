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
        List of halo ids which are used to query halos via h[haloid].
    IDlist
        Table of haloid and corresponding ahfidname (directly given by AHF) given in the property 
        dictionary.
    hostid
        List of the haloid of the host halo of each halo (originally 
        recorded in the property dictionary in the form of ahfidname).
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
    def __init__(self, halocatalogue, datatype, attrpath=None, field=default_field,
            field_units=default_units,
            host_id_of_top_level=0):
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

        if attrpath is None:
            self.datatype=datatype
            self.catalogue_original = halocatalogue
            self.number_mapper = halocatalogue.number_mapper
            self.length = len(self.catalogue_original)
            init_zeros = np.zeros(self.length)
            self.host_id_of_top_level = host_id_of_top_level
            self.errorlist = [{}, {}, {}]
            # self.verbose = verbose

            self.rho_crit = pnb.analysis.cosmology.rho_crit(
                f=self.catalogue_original[self.number_mapper.index_to_number(0)],
                                                            unit='Msol kpc**-3')
            self.ovdens = cosmology.Delta_vir(
                        self.catalogue_original[self.number_mapper.index_to_number(0)])
            self.a_sim = self.catalogue_original[self.number_mapper.index_to_number(0)].properties['a']
            self.h_sim = self.catalogue_original[self.number_mapper.index_to_number(0)].properties['h']
            self.z_sim = self.catalogue_original[self.number_mapper.index_to_number(0)].properties['z']
            self.boxsize = self.catalogue_original[self.number_mapper.index_to_number(0)].properties['boxsize'].in_units('kpc')

            halodicts = halocatalogue.get_properties_all_halos().copy()
            del halodicts['children'], halodicts['parent']
            halodicts = Table(halodicts)
            halodicts['halo_number'] = self.number_mapper.index_to_number(np.arange(len(halocatalogue)))
            halodicts['boxsize'] = self.boxsize
            halodicts.add_column(self.z_sim, name='z')
            halodicts.add_column(self.a_sim, name='a')
            halodicts.add_column(self.h_sim, name='h')
            self.dict = halodicts
            self.haloid = self.dict['halo_number']

            # Some hostHalo id will not be listed in ahfidname list, this is probably due to AHF algorithm
            host_in_IDlist = (np.isin(self.dict['hostHalo'], self.dict['halo_number'])) | \
                            (self.dict['hostHalo'] == self.host_id_of_top_level)
            in_idx, = np.where(host_in_IDlist)
            _not_in_ = np.invert(host_in_IDlist)
            not_in_idx, = np.nonzero(_not_in_)
            self.hostid = np.zeros(self.length, dtype=int) + host_id_of_top_level
            self.hostid[in_idx] = self.dict['hostHalo'][in_idx]
            self.hostid = np.ma.array(self.hostid, dtype=int, mask=_not_in_)

            if len(not_in_idx) > 0:
                for error in not_in_idx:
                    self.errorlist[0][self.haloid[error]] = self.dict['hostHalo'][error]

            # prop initialization
            self.prop = {}
            for field_type in field_units:
                init_prop_table = [init_zeros for _ in range(len(field[field_type]))]
                self.prop[field_type] = Table(init_prop_table, names=field[field_type])
                # astropy.table.Table is only used for generating a structured array more conveniently
                self.prop[field_type] = pnb.array.SimArray(self.prop[field_type], units=field_units[field_type])
            if field is None:
                self.field = default_field
            else:
                self.field = field
            self.field_units = field_units

            self._have_children = False
            self._have_galaxy = False
            self._have_group = False
            self._have_precalcumass = False

        else:
            thalo = np.load(attrpath, allow_pickle=True)[()]
            self.__dict__.update(thalo)
            self.catalogue_original = halocatalogue
            if self._have_new_catalogue and 'new_include_sub' in self.__dict__.keys():
                self.get_new_catalogue(self.new_include_sub)
            else:
                print("new_catalogue is the same as catalogue_original.")
                self.get_new_catalogue(False)

    def init_relationship(self, include_sub=False):
        '''
        Get basic information regarding hosts, children, etc.

        Parameters
        ------------
        include_sub
            Whether or not to include all the subhalo particles when 
            generating the new catalogue. See get_new_catalogue() for 
            details.
        '''
        self.get_children()
        self.get_new_catalogue(include_sub=include_sub)
        self.get_center()

    def identify_galaxyAndGroups(self, galaxy_low_limit, galaxy_mode='only stellar', N_galaxy=3):
        """
        Parameters
        ------------
        galaxy_low_limit : pynbody.array.SimArray
            Required by get_galaxy(). Limit above which galaxies will 
            be identified as luminous galaxies.
        N_galaxy : int
            Required by get_group_list(). Number of luminous galaxies 
            above which host halos are considered as groups.
        """

        self.get_galaxy(g_low_limit=galaxy_low_limit, mode=galaxy_mode)
        self.get_group_list(N_galaxy)
    
    def init_metallicity(self, elements, rkeys, weight_types):
        init_zeros = np.zeros(self.length)
        field_names = []
        for ele in elements:
            for rad in rkeys:
                for weight_type in weight_types:
                    field_names.append('Z_' + ele + rad + weight_type)
        init_prop_table = Table([init_zeros for _ in range(len(field_names))])
        self.prop['metals'] = Table(init_prop_table, names=field_names)
        self.prop['metals'] = pnb.array.SimArray(self.prop['metals'], units='cm**-3')
        self.field['metals'] = field_names
        self.field_units['metals'] = 'cm**-3'
    
    def loadprop(self, props):
        for i in range(len(props)):
            haloid, tprops = props[i]
            haloidx = self.number_mapper.number_to_index(haloid)
            for proptype in tprops.keys():
                subprop = tprops[proptype]
                for key in subprop.keys():
                    self.prop[proptype][key][haloidx] = subprop[key]

    def saveattrs(self, path):
        '''
        Save the object as a .npy file.

        Parameters
        -----------
        path
            Path to save the object.
        '''
        if not os.path.isfile(path):
            savedict = self.__dict__.copy()
            _ = savedict.pop("new_catalogue", None)
            _ = savedict.pop("catalogue_original", None)
            np.save(path, {**savedict})
        else:
            raise Exception('File already exists!')

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
        halo_id_list = np.array(halo_id_list, dtype=int).reshape(-1)
        if len(halo_id_list) == 0:
            halo_id_list = self.group_list
        with h5py.File(filename, "w") as f:
            dataset = f.create_dataset("halo_id", data = halo_id_list)
            dataset.attrs['Description'] = 'halo_ids of halos saved in this file.'
            dataset2 = f.create_dataset("N_lgal", data = self.n_lgal[self.number_mapper.number_to_index(halo_id_list)])
            dataset2.attrs['Description'] = 'Number of luminous galaxies'
            for attr in field:
                grp = f.create_group(attr)
                infos = field[attr]
                for info in infos:
                    data_to_save = self.prop[attr][info][self.number_mapper.number_to_index(halo_id_list)]
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
        self.tophost = np.zeros(self.length).astype(int)
        self.children = [set() for _ in range(self.length)]
        # k = 0
        print('Generating children list...')
        for i in tqdm.tqdm(range(self.length)):
            j = self.number_mapper.index_to_number(i)
            # if ((j // 100) != (k // 100)) and self.verbose:
            #     print('Generating children list... Halo: {:7} / {}'.format(j, self.length), end='\r')
            # k = j
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
                    host_haloid = hostID
                    self.children[self.number_mapper.number_to_index(host_haloid)].add(j)
                    temphost = j
                    while temphost != self.host_id_of_top_level:
                        temphost2 = temphost
                        temphost = self.hostid[self.number_mapper.number_to_index(temphost)]
                    self.tophost[i] = temphost2
            except IndexError:
                self.errorlist[1][j] = hostID
        self._have_children = True
    
    def get_new_catalogue(self, include_sub):
        '''
        Generate a new catalogue based on catalogue_original, 
        the new catalogue will include all the subhalo particles 
        in its host halo.
        
        Parameters
        -------------
        include_sub : bool
            If True, then will include all the subhalo particles. 
            Otherwise will just be a copy of catalogue_original.
        '''
        if not self._have_children:
            raise Exception('Must get_children first!')
        if include_sub:
            self.new_include_sub = True
            self.new_catalogue = {}
            # k = 0
            print('Generating new catalogue...')
            for i in tqdm.tqdm(range(self.length)):
                j = self.number_mapper.index_to_number(i)
                # if ((i // 100) != (k // 100)) and self.verbose:
                #     print('Generating new catalogue... Halo: {:7} / {}'.format(j, self.length), end='\r')
                #     k = i
                if len(self.children[i]) == 0:
                    self.new_catalogue[j] = self.catalogue_original[j]
                else:
                    union_list = [j] + list(self.children[i])
                    self.new_catalogue[j] = get_union(self.catalogue_original, union_list)
        else:
            self.new_include_sub = False
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
        if not self._have_precalcumass:
            raise Exception('Must run halo_calcs.init_precalcumass first to get some masses.')

        self.galaxy_list = [] # List of all galaxies (as long as n_star > 0).
        self.lumi_galaxy_list = [] # List of all luminous galaxies (self_m_star > galaxy_low_limit).
        self.galaxies = [set() for _ in range(self.length)]
        self.lumi_galaxies = [set() for _ in range(self.length)]
        self.n_lgal = np.zeros(self.length) # Number of total luminous galaxies embedded in each host halo.
        # The galaxies within subhalos (i.e., subhalos themselves) will also be taken into account.
    

        # k = 0
        # print('Calculating total stellar masses...')
        # for i in tqdm.tqdm(range(self.length)):
        #     j = self.number_mapper.index_to_number(i)
        #     # if ((i // 100) != (k // 100)) and self.verbose:
        #     #     print('Calculating total stellar masses... Halo: {:7} / {}'.format(j, self.length), end='\r')
        #     #     k = i
        #     self.prop['M']['total_star'][i] = self.new_catalogue[j].star['mass'].sum()
        #     sf_gas = self.new_catalogue[j].gas[pnb.filt.LowPass('temp', '3e4 K')]
        #     # sf_gas = self.new_catalogue[j].gas[pnb.filt.HighPass('nh', '0.13 cm**-3')]
        #     self.prop['M']['total_sfgas'][i] = sf_gas['mass'].sum()
        #     # sf_gas, i.e., star forming gas, is used in the definition of resolved galaxies in Liang's Figure2.
        #     # But seems that Liang didn't plot Figure 2 using the concept of resolved galaxies.

        low_limit = g_low_limit.in_units(self.prop['M']['total_star'].units)
        # k = 0
        print('Identifying galaxies...')
        for i in tqdm.tqdm(range(self.length)):
            j = self.number_mapper.index_to_number(i)
            # if ((i // 100) != (k // 100)) and self.verbose:
            #     print('            Identifying galaxies... Halo: {:7} / {}'.format(j, self.length), end='\r')
            #     k = i
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
            self_Mstar = self.prop['M']['self_star'][i]
            self_Msfgas = self.prop['M']['self_sfgas'][i]
            try:
                if mode == 'only stellar':
                    condition = (self_Mstar > 0)
                elif mode == 'include cold gas':
                    condition = (self_Mstar + self_Msfgas > low_limit)
                if condition:
                    self.galaxy_list.append(j)
                    temp_tophost = self.tophost[i]
                    self.galaxies[self.number_mapper.number_to_index(temp_tophost)].add(j)
        
                    if self_Mstar > low_limit:
                        self.lumi_galaxy_list.append(j)
                        self.n_lgal[self.number_mapper.number_to_index(temp_tophost)] += 1
                        self.lumi_galaxies[self.number_mapper.number_to_index(temp_tophost)].add(j)
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
        self.group_list = self.number_mapper.index_to_number(np.nonzero(self.n_lgal >= N_galaxy)[0])
        # self.group_list += 1
        self._have_group = True
    
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
            self.center = pnb.array.SimArray(self.center, units='kpc') * self.a_sim / self.h_sim
            if self.datatype == 'tipsy_ahf':
                self.center -= self.boxsize.in_units('kpc')/2
        else:
            self.center = pnb.array.SimArray(np.zeros((self.length, 3)), units='kpc')
            if 'phi' in self.new_catalogue[self.number_mapper.index_to_number(0)].loadable_keys():
                center_mode = 'pot'
            else:
                center_mode = 'com'
            for i in range(self.length):
                j = self.number_mapper.index_to_number(i)
                print('Calculating center... {:7} / {}'.format(j, self.length), end='\r')
                self.center[i] = pnb.analysis.halo.center(self.new_catalogue[j], \
                    mode=center_mode, return_cen=True, with_velocity=False)
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
