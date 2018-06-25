#!/usr/bin/env bash

# rebuild all custom base libraries

cd ~/hsi_atk/scikit-hyper/
$HSI setup.py install
cd ~/hsi_atk/scikit-image/
$HSI setup.py install
cd ~/hsi_atk/scikit-learn/
$HSI setup.py install
cd ~/hsi_atk/mlens/
$HSI setup.py install