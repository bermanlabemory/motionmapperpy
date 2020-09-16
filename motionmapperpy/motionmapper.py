import os, time, glob, shutil
import multiprocessing as mp

import matplotlib
matplotlib.use('Agg')

import numpy as np
from scipy.io import savemat, loadmat
from sklearn.manifold import TSNE
import hdf5storage
from sklearn.neighbors import NearestNeighbors
from skimage.segmentation import watershed
import h5py
from easydict import EasyDict as edict
from scipy.spatial import Delaunay
from scipy.optimize import fmin
import matplotlib.pyplot as plt
from skimage.filters import roberts

from .wavelet import findWavelets
from .mmutils import findPointDensity, gencmap
from .setrunparameters import setRunParameters
"""Core t-SNE MotionMapper functions."""

def findKLDivergences(data):
    N = len(data)
    logData = np.log(data)
    logData[~np.isfinite(logData)] = 0

    entropies = -np.sum(np.multiply(data, logData), 1)

    D = - np.dot(data, logData.T)

    D = D - entropies[:,None]

    D = D / np.log(2)
    np.fill_diagonal(D, 0)
    return D, entropies


def run_tSne(data, parameters=None):
    """
    run_tSne runs the t-SNE algorithm on an array of normalized wavelet amplitudes
    :param data: Nxd array of wavelet amplitudes (will normalize if unnormalized) containing N data points
    :param parameters: motionmapperpy Parameters dictionary.
    :return:
            yData -> N x 2 array of embedding results
    """
    parameters = setRunParameters(parameters)

    vals = np.sum(data, 1)
    if ~np.all(vals == 1):
        data = data / vals[:, None]

    print('Finding Distances')
    D, _ = findKLDivergences(data)
    D[~np.isfinite(D)] = 0.0
    D = np.square(D)

    print('Computing t-SNE')
    tsne = TSNE(perplexity=parameters.perplexity, metric='precomputed', verbose=1, n_jobs=-1,
                method=parameters.tSNE_method)
    yData = tsne.fit_transform(D)
    return yData


"""Training-set Generation"""


def returnTemplates(yData, signalData, minTemplateLength=10, kdNeighbors=10):
    maxY = np.ceil(np.max(np.abs(yData[:]))) + 1
    d = signalData.shape[1]

    nn = NearestNeighbors(n_neighbors=kdNeighbors + 1, n_jobs=-1)
    nn.fit(yData)
    D, _ = nn.kneighbors(yData)
    sigma = np.median(D[:, -1])

    _, xx, density = findPointDensity(yData, sigma, 501, [-maxY, maxY])

    L = watershed(-density, connectivity=10)

    # savemat('/mnt/HFSP_Data/scripts/LIDAR/testdata.mat', {'ydata':yData, 'D':D, 'density':density, 'L':L})

    watershedValues = np.digitize(yData, xx)
    watershedValues = L[watershedValues[:, 1], watershedValues[:, 0]]

    maxL = np.max(L)

    templates = []
    for i in range(1, maxL + 1):
        templates.append(signalData[watershedValues == i])
    lengths = np.array([len(i) for i in templates])
    templates = np.array(templates, dtype=object)

    idx = np.where(lengths >= minTemplateLength)[0]
    vals2 = np.zeros(watershedValues.shape)
    for i in range(len(idx)):
        vals2[watershedValues == idx[i]+1] = i + 1

    templates = templates[lengths >= minTemplateLength]
    lengths = lengths[lengths >= minTemplateLength]

    return templates, xx, density, sigma, lengths, L, vals2


def findTemplatesFromData(signalData, yData, signalAmps, numPerDataSet, parameters,projectionFile):
    kdNeighbors = parameters.kdNeighbors
    minTemplateLength = parameters.minTemplateLength

    print('Finding Templates.')
    templates, _, density, _, templateLengths, L, vals = returnTemplates(yData, signalData, minTemplateLength, kdNeighbors)

    ####################################################
    wbounds = np.where(roberts(L).astype('bool'))
    wbounds = (wbounds[1], wbounds[0])
    fig, ax = plt.subplots()
    ax.imshow(density, origin='lower', cmap=gencmap())
    ax.scatter(wbounds[0], wbounds[1], color='k', s=0.1)
    fig.savefig(projectionFile[:-4]+'_trainingtSNE.png')
    plt.close()
    ####################################################

    N = len(templates)
    d = len(signalData[1, :])
    selectedData = np.zeros((numPerDataSet, d))
    selectedAmps = np.zeros((numPerDataSet, 1))

    numInGroup = np.round(numPerDataSet * templateLengths / np.sum(templateLengths))
    numInGroup[numInGroup == 0] = 1
    sumVal = np.sum(numInGroup)
    if sumVal < numPerDataSet:
        q = int(numPerDataSet - sumVal)
        idx = np.random.permutation(N)[:min(q, N)]
        numInGroup[idx] = numInGroup[idx] + 1
    else:
        if sumVal > numPerDataSet:
            q = int(sumVal - numPerDataSet)
            idx2 = np.where(numInGroup > 1)[0]
            Lq = len(idx2)
            if Lq < q:
                idx2 = np.arange(len(numInGroup))
            idx = np.random.permutation(len(idx2))[:q]
            numInGroup[idx2[idx]] = numInGroup[idx2[idx]] - 1
    idx = numInGroup > templateLengths
    numInGroup[idx] = templateLengths[idx]
    cumSumGroupVals = [0] + np.cumsum(numInGroup).astype(int).tolist()

    for j in range(N):

        if cumSumGroupVals[j + 1] > cumSumGroupVals[j]:
            amps = signalAmps[vals == j+1]
            idx2 = np.random.permutation(len(templates[j][:, 1]))[:int(numInGroup[j])].astype(int)
            selectedData[cumSumGroupVals[j]:cumSumGroupVals[j + 1], :] = templates[j][idx2, :]
            selectedAmps[cumSumGroupVals[j]:cumSumGroupVals[j + 1], 0] = amps[idx2]

    signalData = selectedData
    signalAmps = selectedAmps

    return signalData, signalAmps

def mm_findWavelets(projections, numModes, parameters):

    amplitudes, f = findWavelets(projections, numModes, parameters.omega0, parameters.numPeriods,
                                 parameters.samplingFreq, parameters.maxF, parameters.minF, parameters.numProcessors,
                                 parameters.useGPU)
    return amplitudes, f

def file_embeddingSubSampling(projectionFile, parameters):
    perplexity = parameters.training_perplexity
    numPoints = parameters.training_numPoints

    print('\t Loading Projections')
    try:
        projections = np.array(loadmat(projectionFile, variable_names=['projections'])['projections'])
    except:
        with h5py.File(projectionFile, 'r') as hfile:
            projections = hfile['projections'][:].T
        projections = np.array(projections)


    N = len(projections)
    numModes = parameters.pcaModes
    skipLength = np.floor(N / numPoints).astype(int)
    if skipLength == 0:
        skipLength = 1
        numPoints = N

    firstFrame = (N%numPoints)


    print('\t Calculating Wavelets')
    data, _ = mm_findWavelets(projections, numModes, parameters)
    signalIdx = np.indices((data.shape[0],))[0]
    signalIdx = signalIdx[firstFrame:int(firstFrame + (numPoints) * skipLength): skipLength]


    if parameters.useGPU >= 0:
        data2 = data[signalIdx].copy()
        signalData = data2.get()
        del data, data2
    else:
        signalData = data[signalIdx]

    signalAmps = np.sum(signalData, axis=1)

    signalData = signalData/signalAmps[:,None]

    print('\t Calculating Distances')
    D, _ = findKLDivergences(signalData)
    D[~np.isfinite(D)] = 0.0
    D = np.square(D)

    print('\t Running t-SNE')
    parameters.perplexity = perplexity
    tsne = TSNE(perplexity=parameters.perplexity, metric='precomputed', verbose=1, n_jobs=-1,
                method=parameters.tSNE_method)
    yData = tsne.fit_transform(D)


    return yData,signalData,signalIdx,signalAmps

def runEmbeddingSubSampling(projectionDirectory, parameters):
    """
    runEmbeddingSubSampling generates a training set given a set of .mat files.

    :param projectionDirectory: directory path containing .mat projection files.
    Each of these files should contain an N x pcaModes variable, 'projections'.
    :param parameters: motionmapperpy Parameters dictionary.
    :return:
        trainingSetData -> normalized wavelet training set
                           (N x (pcaModes*numPeriods) )
        trainingSetAmps -> Nx1 array of training set wavelet amplitudes
        projectionFiles -> list of files in 'projectionDirectory'
    """
    parameters = setRunParameters(parameters)
    projectionFiles = glob.glob(projectionDirectory+'/*pcaModes.mat')
    
    N = parameters.trainingSetSize
    L = len(projectionFiles)
    numPerDataSet = round(N / L)
    numModes = parameters.pcaModes
    numPeriods = parameters.numPeriods

    trainingSetData = np.zeros((numPerDataSet * L, numModes * numPeriods))
    trainingSetAmps = np.zeros((numPerDataSet * L, 1))
    useIdx = np.ones((numPerDataSet * L), dtype='bool')

    for i in range(L):

        print('Finding training set contributions from data set %i/%i : \n%s'%(i+1, L, projectionFiles[i]))

        currentIdx = np.arange(numPerDataSet) + (i * numPerDataSet)

        yData, signalData, _, signalAmps = file_embeddingSubSampling(projectionFiles[i], parameters)

        trainingSetData[currentIdx,:], trainingSetAmps[currentIdx] = findTemplatesFromData(signalData, yData,
                                                                                           signalAmps, numPerDataSet,
                                                                                        parameters,projectionFiles[i])

        a = (np.sum(trainingSetData[currentIdx,:], 1) == 0)
        useIdx[currentIdx[a]] = False

    trainingSetData = trainingSetData[useIdx,:]
    trainingSetAmps = trainingSetAmps[useIdx]

    return trainingSetData,trainingSetAmps,projectionFiles

def subsampled_tsne_from_projections(parameters,results_directory):
    """
    Wrapper function for training set subsampling and mapping.
    """
    projection_directory = results_directory+'/Projections/'

    tsne_directory= results_directory+'/TSNE/'

    parameters.tsne_directory = tsne_directory

    parameters.tsne_readout = 50

    print('Finding Training Set')
    if not os.path.exists(tsne_directory+'training_data.mat'):
        trainingSetData,trainingSetAmps,_ = runEmbeddingSubSampling(projection_directory,parameters)
        if os.path.exists(tsne_directory):
            shutil.rmtree(tsne_directory)
            os.mkdir(tsne_directory)
        else:
            os.mkdir(tsne_directory)

        hdf5storage.write(data={'trainingSetData': trainingSetData}, path='/', truncate_existing=True,
                          filename=tsne_directory+'/training_data.mat', store_python_metadata=False,
                          matlab_compatible=True)

        hdf5storage.write(data={'trainingSetAmps': trainingSetAmps}, path='/', truncate_existing=True,
                          filename=tsne_directory + '/training_amps.mat', store_python_metadata=False,
                          matlab_compatible=True)


        del trainingSetAmps
    else:
        print('Subsampled trainingSetData found, skipping minitSNE and running training tSNE')
        with h5py.File(tsne_directory + '/training_data.mat', 'r') as hfile:
            trainingSetData = hfile['trainingSetData'][:].T


    # %% Run t-SNE on training set

    parameters.tsne_readout = 5
    print('Finding t-SNE Embedding for Training Set')

    trainingEmbedding= run_tSne(trainingSetData,parameters)

    hdf5storage.write(data={'trainingEmbedding': trainingEmbedding}, path='/', truncate_existing=True,
                      filename=tsne_directory + '/training_tsne_embedding.mat', store_python_metadata=False,
                      matlab_compatible=True)

"""Re-Embedding Code"""


def returnCorrectSigma_sparse(ds, perplexity, tol,maxNeighbors):

    highGuess = np.max(ds)
    lowGuess = 1e-10

    sigma = .5*(highGuess + lowGuess)

    dsize = ds.shape
    sortIdx = np.argsort(ds)
    ds = ds[sortIdx[:maxNeighbors]]
    p = np.exp(-0.5*np.square(ds)/sigma**2)
    p = p/np.sum(p)
    idx = p>0
    H = np.sum(-np.multiply(p[idx],np.log(p[idx]))/np.log(2))
    P = 2**H

    if abs(P-perplexity) < tol:
        test = False
    else:
        test = True

    count = 0
    if ~np.isfinite(sigma):
        raise ValueError('Starting sigma is %0.02f, highGuess is %0.02f '
                'and lowGuess is %0.02f'%(sigma, highGuess, lowGuess))
    while test:

        if P > perplexity:
            highGuess = sigma
        else:
            lowGuess = sigma

        sigma = .5*(highGuess + lowGuess)


        p = np.exp(-.5*np.square(ds)/sigma**2)
        p = p/np.sum(p)
        idx = p>0
        H = np.sum(-np.multiply(p[idx],np.log(p[idx]))/np.log(2))
        P = 2**H

        if np.abs(P-perplexity) < tol:
            test = False

    out = np.zeros((dsize[0],))
    out[sortIdx[:maxNeighbors]] = p
    return sigma,out


def findListKLDivergences(data, data2):
    logData = np.log(data)

    entropies = -np.sum(np.multiply(data,logData), 1)
    del logData

    logData2 = np.log(data2)

    D = - np.dot(data,logData2.T)

    D = D - entropies[:,None]

    D = D / np.log(2)
    return D,entropies


def calculateKLCost(x,ydata,ps):
    d = np.sum(np.square(ydata-x),1).T
    out = np.log(np.sum(1/(1+d))) + np.sum(np.multiply(ps,np.log(1+d)))
    return out


def TDistProjs(i, q, perplexity, sigmaTolerance, maxNeighbors, trainingEmbedding, readout):
    if (i+1)%readout == 0:
        t1 = time.time()
        print('\t\t Calculating Sigma Image #%5i'% (i+1))
    _, p = returnCorrectSigma_sparse(q, perplexity, sigmaTolerance, maxNeighbors)

    if (i+1)%readout == 0:
        print('\t\t Calculated Sigma Image #%5i'%(i+1))

    idx2 = p>0
    z = trainingEmbedding[idx2,:]
    maxIdx = np.argmax(p)
    a = np.sum(z*(p[idx2].T)[:,None],axis=0)

    guesses = [a, trainingEmbedding[maxIdx,:]]

    q = Delaunay(z)

    if (i+1)%readout == 0:
        print('\t\t FminSearch Image #%5i'%(i+1))

    b = np.zeros((2, 2))
    c = np.zeros((2,))
    flags = np.zeros((2,))


    b[0,:],c[0],_,_,flags[0] = fmin(calculateKLCost,x0=guesses[0],args=(z,p[idx2]),disp=False,full_output=True,maxiter=100)
    b[1,:],c[1],_,_,flags[1] = fmin(calculateKLCost,x0=guesses[1],args=(z,p[idx2]),disp=False,full_output=True,maxiter=100)

    if (i+1)%readout == 0:
        print('\t\t FminSearch Done Image #%5i %0.02fseconds'%(i+1, time.time()-t1))

    polyIn = q.find_simplex(b)>=0

    if np.sum(polyIn) > 0:
        pp = np.where(polyIn)[0]
        mI = np.argmin(c[polyIn])
        mI = pp[mI]
        current_poly = True
    else:
        mI = np.argmin(c)
        current_poly = False
    if (i+1)%readout == 0:
        print('\t\t Simplex search done Image #%5i %0.02fseconds'%(i+1, time.time()-t1))
    exitFlags = flags[mI]
    current_guesses = guesses[mI]
    current = b[mI]
    tCosts = c[mI]
    current_meanMax = mI
    return current_guesses, current, tCosts, current_poly, current_meanMax, exitFlags


def findTDistributedProjections_fmin(data, trainingData, trainingEmbedding, parameters):
    readout = 20000
    sigmaTolerance = 1e-5
    perplexity = parameters.perplexity
    maxNeighbors = parameters.maxNeighbors
    batchSize = parameters.embedding_batchSize



    N = len(data)
    zValues = np.zeros((N,2))
    zGuesses = np.zeros((N,2))
    zCosts = np.zeros((N,))
    batches = np.ceil(N/batchSize).astype(int)
    inConvHull = np.zeros((N,), dtype=bool)
    meanMax = np.zeros((N,))
    exitFlags = np.zeros((N,))

    if parameters.numProcessors < 0:
        numProcessors = mp.cpu_count()
    else:
        numProcessors = parameters.numProcessors
    # ctx = mp.get_context('spawn')

    for j in range(batches):
        print('\t Processing batch #%4i out of %4i'%(j+1,batches))
        idx = np.arange(batchSize) + j*batchSize
        idx = idx[idx < N]
        currentData = data[idx,:]
        if np.sum(currentData==0):
            print('Zeros found in wavelet data at following positions. Will replace then with 1e-12.')
            currentData[currentData==0] = 1e-12

        print('\t Calculating distances for batch %4i'%(j+1))
        t1 = time.time()
        D2,_ = findListKLDivergences(currentData,trainingData)
        print('\t Calculated distances for batch %4i %0.02fseconds.'%(j+1, time.time()-t1))

        # triter = tqdm(range(len(idx)), )
        print('\t Calculating fminProjections for batch %4i' % (j + 1))
        t1 = time.time()
        pool = mp.Pool(numProcessors)
        outs = pool.starmap(TDistProjs, [(i, D2[i,:], perplexity, sigmaTolerance, maxNeighbors, trainingEmbedding, readout)
                            for i in range(len(idx))])

        zGuesses[idx,:] = np.concatenate([out[0][:,None] for out in outs], axis=1).T
        zValues[idx,:] = np.concatenate([out[1][:,None] for out in outs], axis=1).T
        zCosts[idx] = np.array([out[2] for out in outs])
        inConvHull[idx] = np.array([out[3] for out in outs])
        meanMax[idx] = np.array([out[4] for out in outs])
        exitFlags[idx] = np.array([out[5] for out in outs])
        pool.close()
        pool.join()
        print('\t Processed batch #%4i out of %4i in %0.02fseconds.\n'%(j+1, batches, time.time()-t1))

    zValues[~inConvHull,:] = zGuesses[~inConvHull,:]

    return zValues,zCosts,zGuesses,inConvHull,meanMax,exitFlags


def findEmbeddings(projections, trainingData, trainingEmbedding, parameters):
    """
    findEmbeddings finds the optimal embedding of a data set into a previously
    found t-SNE embedding.
    :param projections:  N x (pcaModes x numPeriods) array of projection values.
    :param trainingData: Nt x (pcaModes x numPeriods) array of wavelet amplitudes containing Nt data points.
    :param trainingEmbedding: Nt x 2 array of embeddings.
    :param parameters: motionmapperpy Parameters dictionary.
    :return: zValues : N x 2 array of embedding results, outputStatistics : dictionary containing other parametric
    outputs.
    """
    d = projections.shape[1]
    numModes = parameters.pcaModes
    numPeriods = parameters.numPeriods


    print('Finding Wavelets')
    data, f = mm_findWavelets(projections, numModes, parameters)
    if parameters.useGPU >= 0:
        data = data.get()



    data = data / np.sum(data, 1)[:,None]


    print('Finding Embeddings')
    t1 = time.time()
    zValues, zCosts, zGuesses, inConvHull, meanMax, exitFlags = findTDistributedProjections_fmin(data,
                                                                            trainingData, trainingEmbedding, parameters)
    del data
    print('Embeddings found in %0.02f seconds.'%(time.time()-t1))

    outputStatistics = edict()
    outputStatistics.zCosts = zCosts
    outputStatistics.f = f
    outputStatistics.numModes = numModes
    outputStatistics.zGuesses = zGuesses
    outputStatistics.inConvHull = inConvHull
    outputStatistics.meanMax = meanMax
    outputStatistics.exitFlags = exitFlags


    return zValues,outputStatistics

