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


def wshedTransform(zValues, min_regions, sigma, tsnefolder, saveplot=True, num_regions=-1):
    # The region count falls as sigma grows (a smoother density merges neighbouring peaks). Two modes:
    #   num_regions >= 0 : sweep sigma and keep the count *closest* to num_regions (a target count).
    #   num_regions <  0 : treat min_regions as a true floor -- coarsen as far as possible while
    #                      keeping AT LEAST min_regions regions.
    print('Starting watershed transform...')
    rangeVals = [-np.abs(zValues).max() - 15, np.abs(zValues).max() + 15]

    def _wshed_at(s):
        _, xx_, dens_ = findPointDensity(zValues, s, 611, rangeVals=rangeVals)
        w_ = watershed(-dens_, connectivity=10)
        w_[dens_ < 1e-5] = 0
        return w_, len(np.unique(w_)) - 1, xx_, dens_

    exact = num_regions is not None and num_regions >= 0
    target = num_regions if exact else min_regions
    tname = 'num_regions' if exact else 'minimum_regions'

    wshed, numRegs, xx, density = _wshed_at(sigma)
    if numRegs < target:
        raise ValueError('\t Starting sigma %0.2f is too high: it already yields only %i regions, fewer '
                         'than the requested %s=%i. Lower startsigma and try again.'
                         % (sigma, numRegs, tname, target))

    if exact:
        best = (wshed, numRegs, xx, density, sigma)        # closest-to-target seen so far
        while numRegs > target and sigma < 20.0:
            sigma += 0.05
            wshed, numRegs, xx, density = _wshed_at(sigma)
            print('\t Sigma %0.2f, Regions %i' % (sigma, numRegs), end='\r')
            if abs(numRegs - target) < abs(best[1] - target):
                best = (wshed, numRegs, xx, density, sigma)
        wshed, numRegs, xx, density, sigma = best
    else:
        while sigma < 20.0:                                # stop just before we'd drop below the floor
            w2, n2, xx2, dens2 = _wshed_at(sigma + 0.05)
            print('\t Sigma %0.2f, Regions %i' % (sigma + 0.05, n2), end='\r')
            if n2 < target:
                break
            wshed, numRegs, xx, density, sigma = w2, n2, xx2, dens2, sigma + 0.05
    print('\n\t Using sigma %0.2f -> %i regions.' % (sigma, numRegs))

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
        ax.axvline(bins[np.where(gm.predict_proba(bins[:, None])[:, minind] < np.min(parameters.pThreshold))[0][0]], color='firebrick',
                   label='pRest=%0.2f'%np.min(parameters.pThreshold))
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


def makeGroupsAndSegments(watershedRegions, zValLens, min_length=10, max_length=100):

    inds = np.zeros_like(watershedRegions)
    start = 0
    for l in zValLens:
        inds[start:start + l] = np.arange(l)
        start += l
    vinds = np.digitize(np.arange(watershedRegions.shape[0]), bins=np.concatenate([[0], np.cumsum(zValLens)]))

    splitinds = np.where(np.diff(watershedRegions, axis=0) != 0)[0] + 1
    inds = [i for i in np.split(inds, splitinds) if len(i) > min_length and len(i) < max_length]
    wregs = [i[0] for i in np.split(watershedRegions, splitinds) if len(i) > min_length and len(i) < max_length]

    vinds = [i for i in np.split(vinds, splitinds) if len(i) > min_length and len(i) < max_length]
    groups = [np.empty((0, 3), dtype=int)] * watershedRegions.max()

    for wreg, tind, vind in zip(wregs, inds, vinds):
        if wreg == 0:
            continue
        if np.all(vind == vind[0]):
            groups[wreg - 1] = np.concatenate(
                [groups[wreg - 1], np.array([vind[0], tind[0] + 1, tind[-1] + 1])[None, :]])
    groups = np.array([[g] for g in groups], dtype=object)
    return groups


def findWatershedRegions(parameters, minimum_regions=150, startsigma=0.1, pThreshold=None, saveplot=True,
                         endident='*_pcaModes.mat', min_length_videos=10, stereotypyFiltering=False,
                         num_regions=-1):
    """Watershed-segment the behavioral density map into discrete regions.

    The map is smoothed with a Gaussian of width ``startsigma`` and then progressively coarsened
    (larger sigma merges peaks, so the region count falls). How the stopping count is chosen:

    - ``num_regions`` (>= 0): aim for this *exact* number of regions -- the sweep keeps the sigma whose
      region count is closest to it.
    - ``num_regions`` left at the default -1: ``minimum_regions`` acts as a true *minimum* -- coarsen as
      far as possible while keeping **at least** ``minimum_regions`` regions. (Note: historically
      ``minimum_regions`` behaved as a ceiling; it is now a genuine floor.)

    ``startsigma`` must be small enough to begin with MORE regions than the target, since smoothing can
    only reduce the count; lower it if you hit the "starting sigma too high" error.
    """
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
        parameters.pThreshold = [0.33, 0.67]
    else:
        parameters.pThreshold = pThreshold

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
    LL, wbounds, sigma, xx, density = wshedTransform(zValues, minimum_regions, startsigma, tsnefolder,
                                                     saveplot=saveplot, num_regions=num_regions)

    print('Assigning watershed regions...')
    watershedRegions = np.digitize(zValues, xx)
    watershedRegions = LL[watershedRegions[:, 1], watershedRegions[:, 0]]

    # Velocity-based stereotypy filtering: always for t-SNE (original behavior), and optionally
    # for UMAP when stereotypyFiltering=True (previously the UMAP path skipped this, pRest = 1.0).
    if parameters.method == 'TSNE' or stereotypyFiltering:
        print('Calculating velocity distributions...')
        ampVels, pRest = velGMM(ampVels, parameters, parameters.projectPath, saveplot=saveplot)
        t1 = time.time()
        print('Adjusting non-stereotypic regions to 0...')
        bwconn = np.convolve((np.diff(watershedRegions) == 0).astype(bool), np.array([True, True]))
        pGoodRest = pRest > np.min(parameters.pThreshold)
        watershedRegions[~np.bitwise_and(bwconn, pGoodRest)] = 0
        print('\t Done. %0.02f seconds' % (time.time() - t1))
    else:
        pRest = 1.0

    # Single write of the complete result (previously written three times as intermediate saves).
    groups = makeGroupsAndSegments(watershedRegions, zValLens, min_length=min_length_videos)
    outdict = {'zValues': zValues, 'zValNames': zValNames, 'zValLens': zValLens, 'sigma': sigma, 'xx': xx,
               'density': density, 'LL': LL, 'watershedRegions': watershedRegions, 'v': ampVels, 'pRest': pRest,
               'wbounds': wbounds, 'groups': groups}
    hdf5storage.write(data=outdict, path='/', truncate_existing=True,
                      filename=tsnefolder + 'zVals_wShed_groups.mat', store_python_metadata=False,
                      matlab_compatible=True)
    print('All data saved in %s.' % (tsnefolder.split('/')[-2] + '/zVals_wShed_groups.mat'))


def velocityStereotypy(wshedfile, parameters, saveplot=False):
    """Per-frame stereotypy from the bimodal log-velocity GMM (Berman et al. 2014).

    Works for any map, including UMAP (where findWatershedRegions skips this by default). Fits a
    2-component Gaussian mixture to log map-velocities and returns the probability that each frame
    is in the slow / stereotyped mode, plus a copy of the watershed-region labels with the
    non-stereotyped frames set to 0.

    :param wshedfile: a loaded zVals_wShed_groups dict (see mmutils.loadWatershed).
    :param parameters: motionmapperpy parameters dict.
    :return: (pRest (N,), stereotypedRegions (N,)).
    """
    if parameters.get('pThreshold', None) is None:
        parameters.pThreshold = [0.33, 0.67]
    ampVels = np.asarray(wshedfile['v']).flatten()
    _, pRest = velGMM(ampVels, parameters, parameters['projectPath'], saveplot=saveplot)
    wregs = np.asarray(wshedfile['watershedRegions']).flatten().astype(int).copy()
    bwconn = np.convolve((np.diff(wregs) == 0).astype(bool), np.array([True, True]))
    pGoodRest = pRest > np.min(parameters.pThreshold)
    wregs[~np.bitwise_and(bwconn[:len(wregs)], pGoodRest[:len(wregs)])] = 0
    return pRest, wregs


def makeRegionVideo(region, parameters, wshedfile, render_frame, n_examples=4, animfps=30.0,
                    minLength=10, maxLength=100, submaxframes=500, outdir=None):
    """Make an exemplar grid video for one watershed region (animal-agnostic).

    Generalizes the species-specific makeregionvideo_* helpers: rather than hard-coding how a frame
    is drawn, pass a callback ``render_frame(dataset_index, frame_index, ax)`` that draws a single
    frame onto a matplotlib Axes (e.g. show the video frame and overlay the tracked skeleton). The
    function picks representative segments of the region, tiles an NxN grid, and writes an mp4.

    :param region: 1-based watershed region number.
    :param parameters: motionmapperpy parameters dict.
    :param wshedfile: loaded zVals_wShed_groups dict.
    :param render_frame: callable ``(dataset_index, frame_index, matplotlib_axes) -> None``.
    :param n_examples: maximum number of example segments to tile.
    :return: path to the written mp4, or None if the region has no usable segments.
    """
    try:
        from moviepy.editor import VideoClip                       # moviepy 1.x
    except Exception:
        from moviepy import VideoClip                              # moviepy 2.x
    from moviepy.video.io.bindings import mplfig_to_npimage
    groups = makeGroupsAndSegments(np.asarray(wshedfile['watershedRegions']).flatten(),
                                   np.asarray(wshedfile['zValLens']).flatten(),
                                   min_length=minLength, max_length=maxLength)
    nregs = len(groups)
    if outdir is None:
        outdir = '%s/%s/region_videos_%i/' % (parameters['projectPath'], parameters['method'], nregs)
    os.makedirs(outdir, exist_ok=True)
    outfile = outdir + 'regions_%.3i.mp4' % region
    g = (groups - 1)[region - 1][0]                       # 0-based dataset + frame indices
    if g.shape[0] <= 1:
        print('[Region %i] no segments.' % region)
        return None
    nframes = np.atleast_1d(np.diff(g[:, 1:], axis=1).squeeze())
    valid = np.where(nframes < submaxframes)[0]
    if len(valid) == 0:
        print('[Region %i] all segments longer than %i frames.' % (region, submaxframes))
        return None
    nplots = int(min(n_examples, len(valid)))
    sel = valid[np.argsort(nframes[valid])[::-1]][:nplots]
    vidinds = g[sel, 0]
    framelists = [np.arange(g[i, 1], g[i, 2]) for i in sel]
    maxsize = max(len(fr) for fr in framelists)
    subx = max(1, int(np.ceil(np.sqrt(nplots))))

    bend = plt.get_backend()
    plt.switch_backend('Agg')
    fig, axes = plt.subplots(subx, subx, figsize=(10, 10), squeeze=False)
    fig.subplots_adjust(0, 0, 1, 1, 0, 0)

    def make_frame(t):
        j_ = int(t * animfps)
        for i in range(subx * subx):
            ax = axes[i // subx, i % subx]
            ax.clear(); ax.axis('off')
            if i >= nplots:
                continue
            j = j_ % len(framelists[i])
            render_frame(int(vidinds[i]), int(framelists[i][j]), ax)
        return mplfig_to_npimage(fig)

    VideoClip(make_frame, duration=maxsize / animfps).write_videofile(
        outfile, fps=animfps, audio=False, threads=1, logger=None)
    plt.close(fig)
    plt.switch_backend(bend)
    return outfile


