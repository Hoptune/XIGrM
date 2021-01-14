from pynbody.analysis.profile import Profile
import pynbody as pnb
import numpy as np
from .X_properties import gas_properties as g_p
from .prepare_pyatomdb import Included_Elements as elems

@Profile.profile_property
def avg_mass_fraction(self):
    pmass = self.sim['mass'].in_units('Msol').view(np.ndarray)
    pmf = self.sim['mass_fraction'].view(np.ndarray).reshape(-1, 1)

    _, n_elements = pmf.shape
    metalmass = pnb.array.SimArray(np.zeros((self.nbins, n_elements)), \
                                                        units='Msol')
    for i in range(self.nbins):
        metalmass[i] = (pmass[self.binind[i]] * \
                    pmf[self.binind[i]]).sum(axis=0)
    mf = metalmass/self['mass'].reshape(-1, 1)
    return mf

@Profile.profile_property
def Ne(self):
    ne = pnb.array.SimArray(np.zeros(self.nbins), units='1')
    pvol = self.sim['volume'].in_units('cm**3').view(np.ndarray)
    pne = self.sim['ne'].in_units('cm**-3').view(np.ndarray)
    for i in range(self.nbins):
        ne[i] = (pvol[self.binind[i]] * pne[self.binind[i]]).sum()
    return ne

@Profile.profile_property
def Nh(self):
    nh = pnb.array.SimArray(np.zeros(self.nbins), units='1')
    pvol = self.sim['volume'].in_units('cm**3').view(np.ndarray)
    pnh = self.sim['nh'].in_units('cm**-3').view(np.ndarray)
    for i in range(self.nbins):
        nh[i] = (pvol[self.binind[i]] * pnh[self.binind[i]]).sum()
    return nh

@Profile.profile_property
def ne(self):
    return (self['Ne']/self._binsize).in_units('cm**-3')

@Profile.profile_property
def nh(self):
    return (self['Nh']/self._binsize).in_units('cm**-3')

@Profile.profile_property
def ElectronAbundance(self):
    return self['ne']/self['nh']

@Profile.profile_property
def volume(self):
    return self._binsize

@Profile.profile_property
def abundance(self, *elements):
    '''
    Covnert average mass fractions to solar abundances. If 
    elements are not default elements in prepare_pyatomdb, use 
    this as profile['abundance,A,B'] to specify elements A, B 
    (e.g., profile['abundance,H,He,O,Fe']). 

    Parameters
    ------------
    elements : Symbols of elements considered in 
        avg_mass_fraction.
    '''
    if len(elements) == 0:
        elements=elems
    return g_p.abundance_to_solar(self['avg_mass_fraction'], \
                                            elements=elements)

@Profile.profile_property
def temp(self):
    T = pnb.array.SimArray(np.zeros(self.nbins), units='K')
    pT = self.sim['temp'].in_units('K').view(np.ndarray)
    pmass = self.sim['mass'].in_units('Msol').view(np.ndarray)
    for i in range(self.nbins):
        T[i] = np.average(pT[self.binind[i]], weights=pmass[self.binind[i]])
    return T

@Profile.profile_property
def Lx(self):
    '''
    Luminosity within each bin, calculated via dirrectly summing.
    '''
    Lx = pnb.array.SimArray(np.zeros(self.nbins), units='erg s**-1')
    pLx = self.sim['Lx'].in_units('erg s**-1').view(np.ndarray)
    for i in range(self.nbins):
        Lx[i] = pLx[self.binind[i]].sum()
    return Lx
