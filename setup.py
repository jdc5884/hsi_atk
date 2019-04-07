from setuptools import setup

setup(
    name='hsi_atk',

    version='0.0.7',

    package_dir = {'hsi_atk': 'hsi_atk'},

    packages=['hsi_atk',
              'hsi_atk.augmentation',
              'hsi_atk.exploratory',
              # 'hsi_atk.model',
              'hsi_atk.pipeline',
              'hsi_atk.preprocessing',
              'hsi_atk.simulation',
              'hsi_atk.utils'],

    python_requires='>=3.6.0',

    url='https://github.com/tensor-strings/hsi_atk',

    # download_url='https://github.com/tensor-strings/hsi_atk/archive/v0.0.5.tar.gz',

    license='BSD 3-Clause',

    author='David Ruddell',

    author_email='dlruddell@gmail.com',

    install_requires=['numpy>=1.14.2',
                      'pandas>=0.23.0',
                      # 'keras>=2.2.0',
                      'scipy>=1.1.0',
                      'scikit-learn>=0.19.1',
                      'scikit-image>=0.14.0',
                      # 'tensorflow>=1.8.0',
                      'rasterio>=0.36.0',
                      'h5py>=2.8.0']
)
