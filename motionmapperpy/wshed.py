import numpy as np
import glob
import h5py
import hdf5storage
import time
from skimage.segmentation import watershed
from skimage.filters import roberts
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from .mmutils import *

bmapcmap = gencmap()


def wshedTransform(zValues, min_regions, sigma, tsnefolder, saveplot=True):
    print('Starting watershed transform...')

    bounds, xx, density = findPointDensity(zValues, sigma, 610,
                                                   rangeVals=[-np.abs(zValues).max() - 15, np.abs(zValues).max() + 15])
    wshed = watershed(-density, connectivity=10)
    wshed[density < 1e-5] = 0
    numRegs = len(np.unique(wshed)) - 1

    if numRegs < min_regions - 10:
        raise ValueError('\t Starting sigma %0.1f too high, maximum # wshed regions possible is %i.' %
                         (sigma, numRegs))

    while numRegs > min_regions:
        sigma += 0.05
        _, xx, density = findPointDensity(zValues, sigma, 610,
                                                  rangeVals=[-np.abs(zValues).max() - 15, np.abs(zValues).max() + 15])
        wshed = watershed(-density, connectivity=10)
        wshed[density < 1e-5] = 0

        numRegs = len(np.unique(wshed)) - 1
        print('\t Sigma %0.2f, Regions %i' % (sigma, numRegs), )
    for i, wreg in enumerate(np.unique(wshed)):
        wshed[wshed == wreg] = i
    wbounds = np.where(roberts(wshed).astype('bool'))
    wbounds = (wbounds[1], wbounds[0])
    if saveplot:
        bend = plt.get_backend()
        plt.switch_backend('Agg')

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        ax = axes[0]
        ax.imshow(randomizewshed(wshed), origin='lower', cmap=bmapcmap)
        for i in np.unique(wshed)[1:]:
            fontsize = 8
            xinds, yinds = np.where(wshed == i)
            ax.text(np.mean(yinds) - fontsize, np.mean(xinds) - fontsize, str(i), fontsize=fontsize, fontweight='bold')
        ax.axis('off')

        ax = axes[1]
        ax.imshow(density, origin='lower', cmap=bmapcmap)
        ax.scatter(wbounds[0], wbounds[1], color='k', s=0.1)
        ax.axis('off')

        fig.savefig(tsnefolder + 'zWshed%i.png' % numRegs)
        plt.close()
        plt.switch_backend(bend)
    return wshed, wbounds, sigma, xx, density


def velGMM(ampV, parameters, projectPath, saveplot=True):
    if parameters.method == 'TSNE':
        if parameters.waveletDecomp:
            tsnefolder = projectPath + '/TSNE/'
        else:
            tsnefolder = projectPath + '/TSNE_Projections/'
    else:
        tsnefolder = projectPath+'/UMAP/'
    ampVels = ampV * parameters['samplingFreq']
    vellog10all = np.log10(ampVels[ampVels > 0])
    npoints = min(50000, len(vellog10all))

    vellog10 = np.random.choice(vellog10all, size=npoints, replace=False)


    gm = GaussianMixture(n_components=2, verbose=1, tol=1e-5, max_iter=2000, n_init=1, reg_covar=1e-3)
    inds = np.random.randint(0, vellog10.shape[0], size=npoints)
    gm = gm.fit(vellog10[inds, None])
    minind = np.argmin(gm.means_.squeeze())

    if saveplot:
        bend = plt.get_backend()
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(8, 8))
        bins = ax.hist(vellog10, bins=200, density=True, color='k', alpha=0.5)
        bins = bins[1]
        p_score = np.exp(gm.score_samples(bins[:, None]))
        ax.plot(bins, p_score, color='k', alpha=0.5)

        for (c, compno, mu, sigma, p) in \
                zip(['royalblue', 'firebrick'], [1, 2], gm.means_.squeeze(), np.sqrt(gm.covariances_.squeeze()),
                    gm.weights_):
            ax.plot(bins, getPDF(bins, mu, sigma, p), label='Component %i' % compno, color=c, alpha=0.5)

        ax.plot(bins, gm.predict_proba(bins[:, None])[:, minind], label='pRest')
        ax.axvline(bins[np.where(gm.predict_proba(bins[:, None])[:, minind] < 0.33)[0][0]], color='firebrick',
                   label='pRest=0.33')
        ax.legend()
        ax.set_xlabel(r'$log_{10}$ Velocity')
        ax.set_ylabel('PDF')

        fig.savefig(tsnefolder + 'zVelocity.png', )
        plt.close()
        plt.switch_backend(bend)

    pRest = np.zeros_like(ampVels)
    pRest[ampVels == 0] = 0.0
    pRest[ampVels > 0] = gm.predict_proba(vellog10all[:, None])[:, minind]
    return ampV, pRest


def makeGroupsAndSegments(watershedRegions, zValLens):
    min_length = 60

    inds = np.zeros_like(watershedRegions)
    start = 0
    for l in zValLens:
        inds[start:start + l] = np.arange(l)
        start += l
    vinds = np.digitize(np.arange(watershedRegions.shape[0]), bins=np.concatenate([[0], np.cumsum(zValLens)]))

    splitinds = np.where(np.diff(watershedRegions, axis=0) != 0)[0] + 1
    inds = [i for i in np.split(inds, splitinds) if len(i) > min_length]
    wregs = [i[0] for i in np.split(watershedRegions, splitinds) if len(i) > min_length]

    vinds = [i for i in np.split(vinds, splitinds) if len(i) > min_length]
    groups = [np.empty((0, 3), dtype=int)] * watershedRegions.max()

    for wreg, tind, vind in zip(wregs, inds, vinds):
        if np.all(vind == vind[0]):
            groups[wreg - 1] = np.concatenate(
                [groups[wreg - 1], np.array([vind[0], tind[0] + 1, tind[-1] + 1])[None, :]])
    groups = np.array([[g] for g in groups])
    return groups


def findWatershedRegions(parameters, minimum_regions=150, startsigma=0.1, pThreshold=None,saveplot=True, endident = '*_pcaModes.mat'):
    projectionfolder = parameters.projectPath + '/Projections/'
    if parameters.method == 'TSNE':
        if parameters.waveletDecomp:
            tsnefolder = parameters.projectPath + '/TSNE/'
        else:
            tsnefolder = parameters.projectPath + '/TSNE_Projections/'
    elif parameters.method == 'UMAP':
        tsnefolder = parameters.projectPath+ '/UMAP/'
    else:
        raise ValueError('parameters.method can only take values \'TSNE\' or \'UMAP\'')

    if pThreshold is None:
        pThreshold = [0.33, 0.67]

    zValues = []
    projfiles = glob.glob(projectionfolder + '/'+endident)
    t1 = time.time()

    zValNames = []
    zValLens = []
    ampVels = []
    for pi, projfile in enumerate(projfiles):
        fname = projfile.split('/')[-1].split('.')[0]
        zValNames.append(fname)
        print('%i/%i Loading embedding for %s %0.02f seconds.' % (pi + 1, len(projfiles), fname, time.time() - t1))
        if parameters.method == 'TSNE':
            zValident = 'zVals' if parameters.waveletDecomp else 'zValsProjs'
        else:
            zValident = 'uVals'
        with h5py.File(projectionfolder + fname + '_%s.mat'%zValident, 'r') as h5file:
            zValues.append(h5file['zValues'][:].T)
        ampVels.append(np.concatenate(([0], np.linalg.norm(np.diff(zValues[-1], axis=0), axis=1)), axis=0))
        # with h5py.File(projectionfolder + fname + '_zAmps_vel.mat', 'r') as h5file:
        #     ampVels.append(h5file['ampvel'][:].T.squeeze())

        assert zValues[-1].shape[0] == ampVels[-1].shape[0]
        zValLens.append(zValues[-1].shape[0])

    zValues = np.concatenate(zValues, 0)
    ampVels = np.concatenate(ampVels, 0)
    # print(zValLens)
    zValLens = np.array(zValLens)
    # print(zValNames)
    zValNames = np.array(zValNames, dtype=object)
    LL, wbounds, sigma, xx, density = wshedTransform(zValues, minimum_regions, startsigma, tsnefolder, saveplot=True)

    print('Assigning watershed regions...')
    watershedRegions = np.digitize(zValues, xx)
    watershedRegions = LL[watershedRegions[:, 1], watershedRegions[:, 0]]

    if parameters.method == 'TSNE':
        print('Calculating velocity distributions...')
        ampVels, pRest = velGMM(ampVels, parameters, parameters.projectPath, saveplot=saveplot)

        outdict = {'zValues': zValues, 'zValNames': zValNames, 'zValLens': zValLens, 'sigma': sigma, 'xx': xx,
                   'density': density, 'LL': LL, 'watershedRegions': watershedRegions, 'v': ampVels, 'pRest': pRest,
                   'wbounds': wbounds}
        hdf5storage.write(data=outdict, path='/', truncate_existing=True,
                          filename=tsnefolder + 'zVals_wShed_groups.mat', store_python_metadata=False,
                          matlab_compatible=True)

        print('\t tempsave done.')

        t1 = time.time()
        print('Adjusting non-stereotypic regions to 0...')
        bwconn = np.convolve((np.diff(watershedRegions) == 0).astype(bool), np.array([True, True]))
        pGoodRest = pRest > np.min(pThreshold)
        badinds = ~np.bitwise_and(bwconn, pGoodRest)
        watershedRegions[badinds] = 0
        print('\t Done. %0.02f seconds'%(time.time()-t1))
    else:
        pRest = 1.0
    outdict = {'zValues':zValues, 'zValNames':zValNames, 'zValLens':zValLens, 'sigma':sigma, 'xx':xx,
               'density':density, 'LL':LL, 'watershedRegions':watershedRegions, 'v':ampVels, 'pRest':pRest,
               'wbounds':wbounds}
    hdf5storage.write(data=outdict, path='/', truncate_existing=True,
                          filename=tsnefolder + 'zVals_wShed_groups.mat', store_python_metadata=False,
                          matlab_compatible=True)
    print('\t tempsave done.')

    groups = makeGroupsAndSegments(watershedRegions, zValLens)
    outdict = {'zValues': zValues, 'zValNames': zValNames, 'zValLens': zValLens, 'sigma': sigma, 'xx': xx,
               'density': density, 'LL': LL, 'watershedRegions': watershedRegions, 'v': ampVels, #'pRest': pRest,
               'wbounds': wbounds, 'groups': groups}
    hdf5storage.write(data=outdict, path='/', truncate_existing=True,
                      filename=tsnefolder + 'zVals_wShed_groups.mat', store_python_metadata=False,
                      matlab_compatible=True)

    print('All data saved in %s.'%(tsnefolder.split('/')[-2]+'/zVals_wShed_groups.mat'))


