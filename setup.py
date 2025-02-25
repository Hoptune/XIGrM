from setuptools import setup, find_packages


setup(name = 'XIGrM',
      version = '0.3',
      description = 'Analysis tools for the intragroup medium.',
      url = 'https://github.com/Hoptunes/XIGrM',
      author = 'Zhiwei Shao',
      author_email = 'zwshao@sjtu.edu.cn',
      license = '',
      packages = find_packages(),
      zip_safe = False,
      install_requires = ['numpy', 'scipy', 'h5py', 'pynbody>=2.0', 'tqdm',
                          'multiprocess', 'astropy', 'pyatomdb>=0.10.0'],
      python_requires='>=3.10')
