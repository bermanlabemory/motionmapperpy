import glob, os, pickle
from datetime import datetime

import numpy as np
from scipy.io import loadmat,savemat
import hdf5storage
import sys
sys.path.append('/mnt/HFSP_Data/scripts/motionmapperpy')
import motionmapperpy as mmpy

"""All of this code can be run in a Jupyter notebook. To use GPUs, please install cupy (https://docs.cupy.dev/en/stable/install.html)."""


"""1. Lets first create mock data to embed with the package."""

# Create a project folder which contains all the data that you want to embed in a single map.
projectPath = '../data/TestProject'
mmpy.createProjectDirectory(projectPath)


# Now add some mock projections to Projections folder. Please note the identifier "pcaModes.mat" for projections.
for i in range(5):
    projs = np.concatenate([np.random.normal(loc=(np.random.rand()-0.5)*2, scale=0.5, size=(2000,1)) for j in range(15)],
                           axis=1)
    print(projs.shape)
    savemat('../data/TestProject/Projections/dataset_%i_pcaModes.mat'%(i+1), {'projections':projs})


"""2. Setup run parameters for MotionMapper."""

parameters = mmpy.setRunParameters()

# %%%%%%% PARAMETERS TO CHANGE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters.projectPath = projectPath
parameters.method = 'UMAP'

parameters.waveletDecomp = True #% Whether to do wavelet decomposition. If False, PCA projections are used for
                                #% tSNE embedding.

parameters.minF = 0.5 #% Minimum frequency for Morlet Wavelet Transform

parameters.maxF = 30                    #% Maximum frequency for Morlet Wavelet Transform,
                                        #% equal to Nyquist frequency for your measurements.

parameters.samplingFreq = 60            #% Sampling frequency (or FPS) of data.

parameters.numPeriods = 25              #% No. of frequencies between minF and maxF.

parameters.numProcessors = -1           #% No. of processor to use when parallel
                                        #% processing (for wavelets, if not using GPU). -1 to use all cores.

parameters.useGPU = -1                   # GPU to use, set to -1 if GPU not present

parameters.training_numPoints=1000      #% Number of points in mini-tSNEs.

# %%%%% NO NEED TO CHANGE THESE UNLESS RAM (NOT GPU) MEMORY ERRORS RAISED%%%%%%%%%%
parameters.trainingSetSize=3000        #% Total number of representative points to find. Increase or decrease based on
                                        #% available RAM. For reference, 36k is a good number with 64GB RAM.

parameters.embedding_batchSize = 30000  #% Lower this if you get a memory error when re-embedding points on learned
                                        #% tSNE map.

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


projectionFiles = glob.glob(parameters.projectPath+'/Projections/*pcaModes.mat')
for i in projectionFiles:
    print(i)

m = loadmat(projectionFiles[0], variable_names=['projections'])['projections']

# %%%%%
parameters.pcaModes = m.shape[1] #%Number of PCA projections in saved files.
parameters.numProjections = parameters.pcaModes
# %%%%%
del m

print(datetime.now().strftime('%m-%d-%Y_%H-%M'))
print('tsneStarted')

if parameters.method == 'TSNE':
    if parameters.waveletDecomp:
        tsnefolder = parameters.projectPath+'/TSNE/'
    else:
        tsnefolder = parameters.projectPath + '/TSNE_Projections/'
elif parameters.method == 'UMAP':
    tsnefolder = parameters.projectPath+'/UMAP/'

if not os.path.exists(tsnefolder +'training_tsne_embedding.mat'):
    print('Running minitSNE')
    mmpy.subsampled_tsne_from_projections(parameters, parameters.projectPath)
    print('minitSNE done, finding embeddings now.')
    print(datetime.now().strftime('%m-%d-%Y_%H-%M'))

import h5py
with h5py.File(tsnefolder + 'training_data.mat', 'r') as hfile:
    trainingSetData = hfile['trainingSetData'][:].T

with h5py.File(tsnefolder+ 'training_embedding.mat', 'r') as hfile:
    trainingEmbedding= hfile['trainingEmbedding'][:].T

if parameters.method == 'TSNE':
    zValstr = 'zVals' if parameters.waveletDecomp else 'zValsProjs'
else:
    zValstr = 'uVals'

for i in range(len(projectionFiles)):
    print('Finding Embeddings')
    print('%i/%i : %s'%(i+1,len(projectionFiles), projectionFiles[i]))
    if os.path.exists(projectionFiles[i][:-4] +'_%s.mat'%(zValstr)):
        print('Already done. Skipping.\n')
        continue


    projections = loadmat(projectionFiles[i])['projections']
    zValues, outputStatistics = mmpy.findEmbeddings(projections,trainingSetData,trainingEmbedding,parameters)

    hdf5storage.write(data = {'zValues':zValues}, path = '/', truncate_existing = True,
                    filename = projectionFiles[i][:-4]+'_%s.mat'%(zValstr), store_python_metadata = False,
                      matlab_compatible = True)
    with open(projectionFiles[i][:-4] + '_%s_outputStatistics.pkl'%(zValstr), 'wb') as hfile:
        pickle.dump(outputStatistics, hfile)

    print('Embeddings saved.\n')
    del zValues,projections,outputStatistics

print('All Embeddings Saved!')

mmpy.findWatershedRegions(parameters, minimum_regions=150, startsigma=0.3, pThreshold=[0.33, 0.67],
                     saveplot=True, endident = '*_pcaModes.mat')