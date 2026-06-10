import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import copy
import matplotlib as mpl
from scipy.io import loadmat
import h5py
import hdf5storage
import matplotlib.colors
import os

def gencmap():
    """
    Get behavioral map colormap as a matplotlib colormap instance.
    :return: Matplotlib colormap instance.
    """
    colors = np.zeros((64, 3))
    colors[:21, 0] = np.linspace(1, 0, 21)
    colors[20:43, 0] = np.linspace(0, 1, 23)
    colors[42:, 0] = 1.0

    colors[:21, 1] = np.linspace(1, 0, 21)
    colors[20:43, 1] = np.linspace(0, 1, 23)
    colors[42:, 1] = np.linspace(1, 0, 22)

    colors[:21, 2] = 1.0
    colors[20:43, 2] = np.linspace(1, 0, 23)
    colors[42:, 2] = 0.0
    return mpl.colors.ListedColormap(colors)


def createProjectDirectory(pathToProject):
    _dirs = [pathToProject, '%s/Projections'%pathToProject,
             '%s/TSNE_Projections'%pathToProject,
             '%s/TSNE'%pathToProject, '%s/UMAP'%pathToProject]
    for d in _dirs:
        if not os.path.exists(d):
            print('Creating : %s'%d)
            os.makedirs(d, exist_ok=True)
        else:
            print('Skipping, path already exists : %s'%d)
    return

def getDensityBounds(density, thresh=1e-6):
    """
    Get the outline for density maps.
    :param density: m by n density image.
    :param thresh: Density threshold for boundaries. Default 1e-6.
    :return: (p by 2) points outlining density map.
    """
    x_w, y_w = np.where(density > thresh)
    x, inv_inds = np.unique(x_w, return_inverse=True)
    bounds = np.zeros((x.shape[0] * 2 + 1, 2))
    for i in range(x.shape[0]):
        bounds[i, 0] = x[i]
        bounds[i, 1] = np.min(y_w[x_w == bounds[i, 0]])
        bounds[x.shape[0] + i, 0] = x[-i - 1]
        bounds[x.shape[0] + i, 1] = np.max(y_w[x_w == bounds[x.shape[0] + i, 0]])
    bounds[-1] = bounds[0]
    bounds[:, [0, 1]] = bounds[:, [1, 0]]
    return bounds.astype(int)


def findPointDensity(zValues, sigma, numPoints, rangeVals):
    """
    findPointDensity finds a Kernel-estimated PDF from a set of 2D data points
    through convolving with a gaussian function.
    :param zValues: 2d points of shape (m by 2).
    :param sigma: standard deviation of smoothing gaussian.
    :param numPoints: Output density map dimension (n x n).
    :param rangeVals: 1 x 2 array giving the extrema of the observed range
    :return:
        bounds -> Outline of the density map (k x 2).
        xx -> 1 x numPoints array giving the x and y axis evaluation points.
%       density -> numPoints x numPoints array giving the PDF values (n by n) density map.
    """
    xx = np.linspace(rangeVals[0], rangeVals[1], numPoints)
    yy = copy.copy(xx)
    [XX, YY] = np.meshgrid(xx, yy)
    G = np.exp(-0.5 * (np.square(XX) + np.square(YY)) / np.square(sigma))
    Z = np.histogramdd(zValues, bins=[xx, yy])[0]
    Z = Z / np.sum(Z)
    Z = np.pad(Z, ((0, 1), (0, 1)), mode='constant', constant_values=((0, 0), (0, 0)))
    density = fftshift(np.real(ifft2(np.multiply(fft2(G), fft2(Z))))).T
    density[density < 0] = 0
    bounds = getDensityBounds(density)
    return bounds, xx, density


def randomizewshed(wshed):
    outwshed = np.zeros_like(wshed)
    regs = np.unique(wshed)[1:]
    np.random.shuffle(regs)
    for i, wreg in enumerate(regs):
        outwshed[wshed==wreg] = i
    return outwshed


def getPDF(x, mu, sigma, p):
    return (p / np.sqrt(2 * np.pi * (sigma ** 2))) * np.exp(-.5 * (x - mu) ** 2 / sigma ** 2)


def conV2matV7(matfile):
    try:
        a = loadmat(matfile)
        _ = [a.pop(i) for i in list(a.keys()) if '__' in i]
        hdf5storage.write(data=a, path='/', truncate_existing=True, filename=matfile, store_python_metadata=False,
                          matlab_compatible=True)
        print('File converted to MATLAB -v7.3')
        return
    except Exception as E:
        try:
            h5file = h5py.File(matfile, 'r')
            print('File already to MATLAB -v7.3')
            return
        except Exception as F:
            print('Something bad happened. File could be lost.')
            print(E)
            print(F)
            return

def checkParams(parameters, projections):
    if np.any([p.shape[0]<parameters.training_numPoints  for p in projections]):
        plens = [p.shape[0] for p in projections]
        print(plens)
        print('Training number of points for miniTSNE is greater than # samples in some files. Adjust it to '
              'smallest # samples : %i'%(np.min(plens)))


"""Project I/O helpers -- hide the loadmat / h5py / hdf5storage + naming-convention boilerplate."""


def embeddingFolder(parameters):
    """Subfolder holding the training set + embeddings for the current method
    ('/TSNE/', '/TSNE_Projections/' or '/UMAP/')."""
    p = parameters['projectPath']
    if parameters['method'] == 'TSNE':
        return p + ('/TSNE/' if parameters.get('waveletDecomp', True) else '/TSNE_Projections/')
    elif parameters['method'] == 'UMAP':
        return p + '/UMAP/'
    raise ValueError("parameters.method must be 'TSNE' or 'UMAP'.")


def embeddingSuffix(parameters):
    """Filename suffix for a per-dataset embedding: 'zVals', 'zValsProjs' or 'uVals'."""
    if parameters['method'] == 'TSNE':
        return 'zVals' if parameters.get('waveletDecomp', True) else 'zValsProjs'
    return 'uVals'


def readMat(path, variable_names=None):
    """Robustly load a .mat file regardless of version: scipy.io for v7, h5py (transposed) for
    MATLAB v7.3 / hdf5storage files. Returns a dict of arrays."""
    try:
        d = loadmat(path, variable_names=variable_names)
        return {k: v for k, v in d.items() if not k.startswith('__')}
    except (NotImplementedError, ValueError):
        out = {}
        with h5py.File(path, 'r') as f:
            keys = list(f.keys()) if variable_names is None else variable_names
            for k in keys:
                if k in f:
                    out[k] = np.array(f[k]).T
        return out


def loadProjections(projectionFile, variable='projections'):
    """Load a projections array from a Projections/*_pcaModes.mat file."""
    return readMat(projectionFile, [variable])[variable]


def saveProjections(parameters, name, projections):
    """Save a (T, pcaModes) projections array to Projections/<name>_pcaModes.mat."""
    path = '%s/Projections/%s_pcaModes.mat' % (parameters['projectPath'], name)
    hdf5storage.savemat(path, {'projections': np.asarray(projections)})
    return path


def loadTrainingSet(parameters):
    """Return (trainingSetData, trainingEmbedding) for the current method."""
    folder = embeddingFolder(parameters)
    data = readMat(folder + 'training_data.mat', ['trainingSetData'])['trainingSetData']
    emb = readMat(folder + 'training_embedding.mat', ['trainingEmbedding'])['trainingEmbedding']
    return data, emb


def loadEmbeddings(parameters, name):
    """Load the 2-D embedding (zValues) of one dataset by name."""
    f = '%s/Projections/%s_%s.mat' % (parameters['projectPath'], name, embeddingSuffix(parameters))
    return readMat(f, ['zValues'])['zValues']


def loadWatershed(parameters):
    """Load the full zVals_wShed_groups dict for the current method."""
    return readMat(embeddingFolder(parameters) + 'zVals_wShed_groups.mat')
