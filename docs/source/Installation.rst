.. SpatialGlue documentation master file, created by
   sphinx-quickstart on Thu Sep 16 19:43:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation
============

Our SpatialEx and SpatialEx+ are developed using the PyTorch framework. Below, we provide instructions for creating a conda environment capable of running our model.

Note that the installation of all packages included in file 'requirement.txt' are needed.

.. code-block:: python

   1) Create conda environment
   
   #Create an environment called SpatialEx

   conda create -n SpatialEx python=3.8

   #Activate your environment

   conda activate SpatialEx

   #All the packages are included in the requirements.txt file

   pip install requirements.txt

   #Several important packages are listed below in case you prefer not to install too many packages.

   anndata=0.8.0
   scanpy==1.9.3
   numpy==1.23.5
   pandas==2.0.3
   cellpose==3.0.10
   scikit-image==0.21.0
   scikit-learn==1.3.2
   scikit-mise==0.2.0
   torch==2.3.1
   huggingface-hub==0.24.6
   timm==1.0.8
   torchvision==0.18.1
 
   2) To use the environment in jupyter notebook, add python kernel for this environment.

   pip install ipykernel

   python -m ipykernel install --user --name=SpatialEx

Optionally, we have developed an easy-to-use Python package, which can be installed using the following command:

.. code-block:: python

   pip install SpatialEx

We provide concrete examples in Tutorials 1 and 2 to illustrate this in detail.
