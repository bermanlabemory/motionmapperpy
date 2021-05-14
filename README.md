# motionmapperpy : Modified Python 3.0 implementation of [MotionMapper](https://github.com/gordonberman/MotionMapper)

This package is a GPU accelerated implementation of the MotionMapper pipeline for creating low dimensional density maps using tSNE or UMAP. Some methodologies 
may differ from the original implementation, please refer to the source code for a detailed look.  

Package functions are:
- Subsampling training points by running mini-tSNEs on a group of datasets. 
- Re-embedding new points on a learned tSNE map. 
- Watershed segmentation and grouping. 

## Installation:
1. (OPTIONAL) Create a new conda environment <code>conda create -n mmenv python=3.6</code>
2. Activate desired conda environemnt <code>conda activate mmenv</code> 
3. Download the repository and unzip contents. Open terminal and navigate to unzipped folder containing setup.py.
4. Run 
```
pip install -U h5py==2.1 
pip install numpy scikit-image hdf5storage
python setup.py install
```

Additionally, install cupy (if GPU present on system) by following the instructions [here](https://docs.cupy.dev/en/stable/install.html).  


## Demo.
After installation, run "python3 demo/demo.py". 

## Issues:
Please post any code related issues at https://github.com/bermanlabemory/motionmapperpy/issues with a complete error 
trace where possible. 