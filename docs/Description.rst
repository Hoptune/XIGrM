How to Use
================================

Data Structure
----------------

Most useful halo information is stored in the `prop` 
attribute of the `class halo_props`. And to modify what 
to be calculated, one will need to provide a new field 
dictionary similarly organized as the default field 
dictionary. Note that quantities related to temperature 
and luminosity are mostly hard coded, so you may need to 
modify the source code to meet your own requirements.

Generally there are four types of properties stored in 
the `prop` attribute: R (radius), M (mass), T (temperature) 
and S (entropy). And the following list explains the physical 
meaning of the terms in the default field dictionary.

R:

========  ===========
Key name  Description
========  ===========
vir       Virial radius :math:`R_{vir}`
200       :math:`R_{200}`
500       :math:`R_{500}`
2500      :math:`R_{2500}`
========  ===========

M (:math:`X = 200, 500`):

============  ===========
Key name      Description
============  ===========
vir           Mass enclosed within :math:`R_{vir}`
200           Mass enclosed within :math:`R_{200}`
500           Mass enclosed within :math:`R_{500}`
2500          Mass enclosed within :math:`R_{2500}`
starX         Stellar mass enclosed within :math:`R_{X}`
gasX          Gas mass enclosed within :math:`R_{X}`
barX          Baryon mass enclosed within :math:`R_{X}`
ismX          ISM mass enclosed within :math:`R_{X}`
coldX         Cold gas mass enclosed within :math:`R_{X}`
igrmX         IGrM mass enclosed within :math:`R_{X}`
total_star    Stellar mass within halo, including contribution from subhalo
self_star     Stellar mass within halo, excluding contribution from subhalo
============  ===========

Before introducing temperature and entropy, I will first talk about the 
luminosities used during calculation:

==========================  =========  ===========
Name                        Key name   Description
==========================  =========  ===========
:math:`L_X`                 Lx         0.5-2.0keV X-ray luminosity, including both continuum and line emission
:math:`L_{X, cont}`         Lx_cont    0.5-2.0keV X-ray luminosity, only including continuum emission
:math:`L_{X, broad}`        Lxb        0.1-2.4keV X-ray luminosity, including both continuum and line emission
:math:`L_{X, broad, cont}`  Lxb_cont   0.1-2.4keV X-ray luminosity, only including continuum emission
==========================  =========  ===========

T (only hot diffuse gas (IGrM) is taken into account when calculating temperature):

============  ===========
Key name      Description
============  ===========
x             Emission weighted temperature using :math:`L_X`
x_cont        Emission weighted temperature using :math:`L_{X, cont}`
mass          Mass weighted temperature
spec          Spectroscopic temperature (See `Vikhlinin, 2006 <https://iopscience.iop.org/article/10.1086/500121>`_)
spec_corr     Spectroscopic (core-corrected) temperature
x_corr        Emission weighted (core-corrected) temperature using :math:`L_X`
x_corr_cont   Emission weighted (core-corrected) temperature using :math:`L_{X, cont}`
mass_corr     Mass weighted (core-corrected) temperature
spec500       Spectroscopic temperature at :math:`R_{500}` (used when calculating entropy)
spec2500      Spectroscopic temperature at :math:`R_{2500}` (used when calculating entropy)
============  ===========

S (calculated within a thin spherical shell with default thickness = 1 kpc):

============  ===========
Key name      Description
============  ===========
500           Entropy of the shell at :math:`R_{500}`
2500          Entropy of the shell at :math:`R_{2500}`
============  ===========

Besides, there are other attributes of `class halo_props` which might be 
useful, see Halo Analysis Module for details.

Examples
----------
Though the sricpts below is written in a .py file, Jupyter Notebook is 
strongly recommand for doing the following analysis, just like the one 
done in `Final_Tutorial.ipynb <https://github.com/Hoptune/XIGrM/blob/master/Final_Tutorial.ipynb>`_ .

.. literalinclude:: ../example/example.py

The main calculation part is now done. If you want to see what plots you 
can get based on above calculation, see 
`Plots.ipynb <https://github.com/Hoptune/XIGrM/blob/master/Plots.ipynb>`_.