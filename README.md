# Main Library
This repository contains the main modules which are required for the artificial intelligence (AI)-based surrogate reservoir modelling. 

The following modules are contained in this repository: 
 - DNN_models**: contain functions and derived classes, which are used to build different types of custom AI-based deep neural network (DNN) architectures including the three-module (or four-module) neural network, 
 convolutional-based encoder-decoder (CEDNN), fully connected deep neural network (FCDNN) and the FCDNN with residula connections (ResNet).
 - batch_loss: contains functions, which are used in computing the physics-based regularization terms and the training data losses (if any). The functions are optimized to work in static (or graph) mode for faster computations.   
 - training_m: contains functions and derived classes, which are used to batch the datasets, reinitialize models and configure the optimizer prior to training. The optimizer configurations include: multi-optimizer settings, training callbacks, initial learning hyperparameters - learning rate, learning rate decay type. 
 - PVT_models: contains functions used to read the pressure-volume-temperature (PVT) dataset from a MS Excel file. The PVT dataset contains fluid property fields, which are required for physics-based semi-supervised learning. 

## How to build
Building requires an integrated development enviroment (IDE) running the Python 3+ interpreter and following libraries installed.
 - Numpy 1.20+
 - Tensorflow 2.9+
 - Pandas 1.4+
 - Tensorflow addons 0.17+
 - Matplotlib 3.5+
 
The modules are downloaded to a local folder. Its path is then added to the local System path. 

