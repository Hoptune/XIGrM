from setuptools import setup, find_packages


setup(name = 'XIGrM',
      version = '0.1',
      description = 'Analysis tools for the intragroup medium.',
      url = 'https://github.com/Hoptunes/XIGrM',
      author = 'Zhiwei Shao',
      author_email = 'zws@smail.nju.edu.cn',
      license = '',
      packages = find_packages(),
      zip_safe = False,
      install_requires = ['numpy', 'scipy', 'h5py', 'pynbody', 'astropy', 'pyatomdb>=0.10.0'])
