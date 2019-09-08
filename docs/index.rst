.. XIGrM documentation master file, created by
   sphinx-quickstart on Thu Aug 29 17:03:12 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to XIGrM's documentation!
==================================

=============
Introduction
=============
This is a package for systematically analysing the X-ray properties 
of IGrM (Intragroup Medium) in cosmological simulations based on `pynbody <http://pynbody.github.io/pynbody/>`_, 
`pyatomdb <https://atomdb.readthedocs.io/en/master/>`_, 
`pytspec <https://github.com/rennehan/pytspec>`_ and `Liang et al., 2016 
<https://academic.oup.com/mnras/article/456/4/4266/2892203>`_. The project github page is 
`here <https://github.com/Hoptune/XIGrM>`_.

The package consists of six parts:

- :doc:`pyatomdb </prepare_pyatomdb>` : A series of codes to deal with atomdb data.
- :doc:`cosmology </cosmology>` : Calculate some necessary cosmological parameters.
- :doc:`gas_properties </gas_properties>` : A series of codes to generate basic information of gas particles required by following analysing.
- :doc:`X_properties </X_properties>` : Tools for analysing X-ray properties of the IGrM.
- :doc:`calculate_R </calculate_R>` : Codes for caluclating radii R200, R500, etc (and corresponding masses) of halos.
- :doc:`halo_analysis </halo_analysis>` : Tools for analysing IGrM and halo properties.

=========
Contents
=========

.. toctree::
   :maxdepth: 2

   prepare_pyatomdb
   cosmology
   gas_properties
   X_properties
   calculate_R
   halo_analysis
   Description

==================  
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
