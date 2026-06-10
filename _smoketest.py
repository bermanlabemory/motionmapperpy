import os, glob, shutil, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
import motionmapperpy as mmpy

print('backend after import (should NOT be forced to agg):', matplotlib.get_backend())
need = ['embedProject', 'waveletTransform', 'loadWatershed', 'loadEmbeddings', 'egocenter',
        'pairwiseDistances', 'velocityStereotypy', 'makeRegionVideo', 'readMat', 'saveProjections']
print('new exports present:', all(hasattr(mmpy, x) for x in need), '/', hasattr(mmpy, 'dynamics'))

proj = '/tmp/mm_smoketest_proj'
if os.path.exists(proj):
    shutil.rmtree(proj)
mmpy.createProjectDirectory(proj)

rng = np.random.default_rng(0)
for i in range(5):
    centers = rng.normal(size=(3, 5))
    lab = rng.integers(0, 3, size=2000)
    X = np.cumsum((centers[lab] + 0.3 * rng.normal(size=(2000, 5))) * 0.05, axis=0)
    mmpy.saveProjections({'projectPath': proj}, 'dset_%d' % i, X)
print('saved projections:', len(glob.glob(proj + '/Projections/*_pcaModes.mat')))

p = mmpy.setRunParameters()
p.projectPath = proj; p.method = 'UMAP'; p.pcaModes = 5; p.samplingFreq = 30
p.minF = 1; p.maxF = 14; p.numPeriods = 10; p.useGPU = -1; p.verbose = False
p.training_numPoints = 1000; p.trainingSetSize = 2000

amps, freqs = mmpy.waveletTransform(np.cumsum(rng.normal(size=(500, 5)), 0), p)
print('waveletTransform ->', amps.shape, 'freqs', freqs.shape)

mmpy.subsampled_tsne_from_projections(p, p.projectPath)
written = mmpy.embedProject(p)
print('embedProject wrote', len(written), 'files')

mmpy.findWatershedRegions(p, minimum_regions=8, startsigma=1.0, saveplot=False)
w = mmpy.loadWatershed(p)
print('loadWatershed -> #regions', int(np.asarray(w['watershedRegions']).max()),
      '| single .mat write only:', len(glob.glob(proj + '/UMAP/zVals_wShed_groups.mat')) == 1)
z = mmpy.loadEmbeddings(p, 'dset_0')
print('loadEmbeddings dset_0 ->', z.shape)

wregs = np.asarray(w['watershedRegions']).flatten()
lens = np.asarray(w['zValLens']).flatten()
seqs = np.split(wregs, np.cumsum(lens)[:-1])
trans = [mmpy.dynamics.getTransitions(s) for s in seqs]
T = mmpy.dynamics.makeTransitionMatrix(trans, lag=1)
rs = T.sum(1)
print('dynamics.makeTransitionMatrix(list) ->', T.shape, '| rows stochastic:',
      np.allclose(rs[rs > 0], 1))
T1 = mmpy.dynamics.makeTransitionMatrix(trans[0], lag=1, n_states=T.shape[0])
print('dynamics single + fixed n_states ->', T1.shape, '(matches:', T1.shape == T.shape, ')')

kp = rng.normal(size=(200, 8, 3))
ego = mmpy.egocenter(kp, center_idx=4, head_idx=0)
egof = mmpy.egocenter(kp, center_idx=4, head_idx=0, tail_idx=7, flatten=True)
pw = mmpy.pairwiseDistances(kp)
print('egocenter ->', ego.shape, '| flat', egof.shape, '| pairwise', pw.shape)

pRest, sreg = mmpy.velocityStereotypy(w, p, saveplot=False)
print('velocityStereotypy -> pRest', np.asarray(pRest).shape,
      '| stereotyped-frame frac %.2f' % (sreg > 0).mean())

print('\nALL SMOKE TESTS PASSED')
