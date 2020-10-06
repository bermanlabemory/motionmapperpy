from easydict import EasyDict as edict

def setRunParameters(parameters=None):
    """
    Get parameter dictionary for running motionmapperpy.
    :param parameters: Existing parameter dictionary, defaults will be filled for missing keys.
    :return: Parameter dictionary.
    """
    if isinstance(parameters, dict):
        parameters = edict(parameters)
    else:
        parameters = edict()


    """"# %%%%%%%% General Parameters %%%%%%%%"""

    # %number of processors to use in parallel code
    numProcessors = 12

    useGPU = -1


    # %%%%%%%% Wavelet Parameters %%%%%%%%

    # %number of wavelet frequencies to use
    numPeriods = 25

    # dimensionless Morlet wavelet parameter
    omega0 = 5

    # sampling frequency (Hz)
    samplingFreq = 100

    # minimum frequency for wavelet transform (Hz)
    minF = 1

    # maximum frequency for wavelet transform (Hz)
    maxF = 50


    """%%%%%%%% t-SNE Parameters %%%%%%%%"""
    # Global tSNE method - 'barnes_hut' or 'exact'
    tSNE_method = 'barnes_hut'

    # %2^H (H is the transition entropy)
    perplexity = 32

    # %embedding batchsize
    embedding_batchSize = 20000

    # %maximum number of iterations for the Nelder-Mead algorithm
    maxOptimIter = 100

    # %number of points in the training set
    trainingSetSize = 35000

    # %number of neigbors to use when re-embedding
    maxNeighbors = 200

    # %local neighborhood definition in training set creation
    kdNeighbors = 5

    # %t-SNE training set perplexity
    training_perplexity = 20

    # %number of points to evaluate in each training set file
    training_numPoints = 10000

    # %minimum training set template length
    minTemplateLength = 1



    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""


    if not 'numProcessors' in parameters.keys():
        parameters.numProcessors = numProcessors

    if not 'numPeriods' in parameters.keys():
        parameters.numPeriods = numPeriods

    if not 'omega0' in parameters.keys():
        parameters.omega0 = omega0



    if not 'samplingFreq' in parameters.keys():
        parameters.samplingFreq = samplingFreq

    if not 'minF' in parameters.keys():
        parameters.minF = minF

    if not 'maxF' in parameters.keys():
        parameters.maxF = maxF


    if not 'tSNE_method' in parameters.keys():
        parameters.tSNE_method = tSNE_method

    if not 'perplexity' in parameters.keys():
        parameters.perplexity = perplexity

    if not 'embedding_batchSize' in parameters.keys():
        parameters.embedding_batchSize = embedding_batchSize

    if not 'maxOptimIter' in parameters.keys():
        parameters.maxOptimIter = maxOptimIter

    if not 'trainingSetSize' in parameters.keys():
        parameters.trainingSetSize = trainingSetSize

    if not 'maxNeighbors' in parameters.keys():
        parameters.maxNeighbors = maxNeighbors

    if not 'kdNeighbors' in parameters.keys():
        parameters.kdNeighbors = kdNeighbors

    if not 'training_perplexity' in parameters.keys():
        parameters.training_perplexity = training_perplexity

    if not 'training_numPoints' in parameters.keys():
        parameters.training_numPoints = training_numPoints

    if not 'minTemplateLength' in parameters.keys():
        parameters.minTemplateLength = minTemplateLength
    
    return parameters