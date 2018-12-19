from setuptools import setup

setup(
    name='hsi_atk',

    version='0.0.4a',

    package_dir = {'hsi_atk': 'hsi_atk'},

    packages=['hsi_atk',
              'hsi_atk.utils',
              'hsi_atk.Pentara',
              'hsi_atk.Sasatari',
              'hsi_atk.Ganita',
              'hsi_atk.Moratikara'],

    python_requires='>=3.6.0',

    url='https://github.com/tensor-strings/hsi_atk',

    download_url='https://github.com/tensor-strings/hsi_atk/archive/v0.0.4.tar.gz',

    license='BSD 3-Clause',

    author='David Ruddell',

    author_email='dlruddell@gmail.com',

    install_requires=['numpy>=1.14.2',
                      'scipy>=1.1.0',
                      'scikit-learn>=0.19.1',
                      'scikit-image>=0.14.0',
                      'scikit-hyper>=0.0.2',
                      'tensorflow>=1.8.0',
                      'rasterio>=0.36.0',
                      'h5py>=2.8.0']
)
