***UNDER DEVELOPMENT***

hsi_atk is a hyperspectral image (hsi) analysis toolkit that features several resources for hsi analysis. The key
purpose, above all, is to enable analysts to fit models to arbitrary image analysis problems. Along with some basic 
file I/O and basic image array manipulation, the toolkit features analysis pipelines and synthetic image/label 
generators.

The synthetic image/label generators key features enable users to define explicitly a range of images that should exist
by user defined functions for the shape and brightness scaling for n-band images. Similarly, the labels for each 
image component can be explicitly defined by the user to create labels based off of image attributes. This is mostly 
useful for regression/density problems, but can also be used for categorical labeling. However, the user may often find
that these categorical labels will be inherently tied to the functions decided to describe the brightness.

The toolkit is also intended to enable creation of well-curated explicit synthetic datasets for training models. The
toolkit utilizes h5py for dataset creation and accessing. Further development in this area is underway.

Analysis pipelines will be for automated selection and testing of models based on high-level inputs given by the user
that define the input problem. This setup is meant to go hand-in-hand with the synthetic data generation as the 
problem to be solved will be understood. In-between, file I/O, image manipulation, extra statistics and metrics, as
well as visualization will be at finger tips as well.

The HSI toolkit utilizes scikit-learn, scikit-hyper, scikit-image, numpy, scipy, h5py, ml-ensemble, tensorflow,
and custom functions and classes.

kspec branch code will focus on analysis of hyperspectral images of corn kernels for current work.
