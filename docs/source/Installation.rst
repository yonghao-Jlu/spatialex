.. SpatialGlue documentation master file, created by
   sphinx-quickstart on Thu Sep 16 19:43:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation
============

The SpatialGlue package has been developed based on the pytorch framework and can be implemented with both GPU and CPU. Using GPU acceleration can significantly speed up the integration progress. So we recommend running the package with GPU first.

To ensure smooth installation and execution of the package, it's important to have PyTorch and CUDA (including CUDNN) installed correctly. These dependencies are necessary for GPU acceleration. Make sure you have the appropriate GPU drivers installed on your system as well.

To run the SpatialGlue package, two methods are provided. Note that the installation of all packages included in file 'requirement.txt' are needed.

1. Installation from PyPI (Python Package Index):
---------------------

.. code-block:: python

   Open a terminal or command prompt

   Run the following command: 'pip install SpatialGlue' or 'pip install SpatialGlue_3M'

   This command will download and install the SpatialGlue package from PyPI along with its dependencies. If any of the dependencies listed in the 'requirement.txt' file are missing, they will be installed automatically.

   Note: please ensure the pip version is python3.
   

2. Anaconda
------------
For convenience, we recommend using a separate conda environment for running SpatialGlue. Please make ensure annaconda3 is installed on your system. If you haven't installed it yet, you can download and install Anaconda3 from the official Anaconda website (https://www.anaconda.com/products/individual).

.. code-block:: python

   1) Create conda environment and install SpatialGlue package
   
   #Create an environment called SpatialGlue

   conda create -n SpatialGlue python=3.8

   #Activate your environment

   conda activate SpatialGlue

   #Install SpatialGlue (SpatialGlue_3M) package

   pip install SpatialGlue or pip install SpatialGlue_3M

 
   2) To use the environment in jupyter notebook, add python kernel for this environment.

   pip install ipykernel

   python -m ipykernel install --user --name=SpatialGlue
   
