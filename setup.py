from setuptools import setup, find_packages


setup(name = 'XIGrMtest',
      version = '0.2',
      description = 'Analysis tools for the intragroup medium.',
      url = 'https://github.com/Hoptunes/XIGrM',
      author = 'Zhiwei Shao',
      author_email = 'zwshao@sjtu.edu.cn',
      license = '',
      packages = find_packages(),
      zip_safe = False,
      install_requires = ['numpy', 'scipy', 'h5py', 'pynbody', 'astropy', 'pyatomdb'])