# motionmapperpy : Modified Python 3.0 implementation of MotionMapper (https://github.com/gordonberman/MotionMapper)

This package is a GPU accelerated implementation of the MotionMapper pipeline for creating tSNE maps. Some methodologies 
may differ from the original implementation, please refer to the source code for a detailed look.  

Package functionalities include:
- Subsampling training points by running mini-tSNEs on a group of datasets. 
- Re-embedding new points on a learned tSNE map. 
- Watershed segmentation and grouping. 

To see the demo run, please call "python3 demo/run.py" after installing necessary package dependencies.
    