# hsi_atk - Hyper-Spectral Image Analysis Toolkit

*** This repository is still under development ***

Repo for analysis on hyper spectral image data using machine learning. 
Includes simulated image generators with accompanying evaluation models, and ensembling methods.
As well as, various image processing tools, dataset I/O methods for creating and accessing datasets, and multiple visualization methods.

# Getting Started
These instructions are for installing, and running a simple simulation

### Prerequisites
So most prerequisities can be obtained by using a standard anaconda install of python >=3.6.0 except a package called rasterio
that is being used for some I/O of the dataset currently being worked on.

The explicit requirements are:
```'numpy>=1.14.2',
   'scipy>=1.1.0',
   'scikit-learn>=0.19.1',
   'scikit-image>=0.14.0',
   'scikit-hyper>=0.0.2',
   'tensorflow>=1.8.0',
   'rasterio>=0.36.0',
   'h5py>=2.8.0'
```
As you will see in setup.py of this repo.

### Installing
```
git clone --single-branch --branch rebuild-kspec https://github.com/tensor-strings/hsi_atk
cd hsi_atk
python setup.py --install
```