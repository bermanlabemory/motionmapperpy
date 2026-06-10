from .setrunparameters import setRunParameters
from .mmutils import (findPointDensity, gencmap, createProjectDirectory, readMat, embeddingFolder,
                      embeddingSuffix, loadProjections, saveProjections, loadTrainingSet,
                      loadEmbeddings, loadWatershed)
from .motionmapper import (run_tSne, run_UMAP, runEmbeddingSubSampling, subsampled_tsne_from_projections,
                           findEmbeddings, embedProject, mm_findWavelets, waveletTransform)
from .wavelet import findWavelets
from .wshed import findWatershedRegions, makeGroupsAndSegments, velocityStereotypy, makeRegionVideo
from .preprocessing import egocenter, pairwiseDistances
from . import dynamics
from .demoutils import makeregionvideo_flies, getTransitions, makeTransitionMatrix, doTheShannonShuffle, \
    plotLaggedEigenvalues, makeregionvideos_mice
