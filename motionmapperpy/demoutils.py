import os, easydict, time

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import hdf5storage
from .wshed import makeGroupsAndSegments

def egoh5(h5, bindcenter=8, align=True, b1=2, b2=11, silent=False):
    """
    Return h5 in the center of frame of body part index. If align is true, rotate data so that vector from b2 to b1
    always points East.
    """

    ginds = np.setdiff1d(np.arange(h5.shape[1]), bindcenter)
    egoh5 = h5[:, :, :2] - h5[:, [bindcenter for i in range(h5.shape[1])], :2]
    egoh5 = egoh5[:, ginds]
    if not align:
        return egoh5
    dir_arr = egoh5[:, b1] - egoh5[:, b2 - 1]
    dir_arr = dir_arr / np.linalg.norm(dir_arr, axis=1)[:, np.newaxis]
    if not silent:
        for t in tqdm(range(egoh5.shape[0])):
            rot_mat = np.array([[dir_arr[t, 0], dir_arr[t, 1]], [-dir_arr[t, 1], dir_arr[t, 0]]])
            egoh5[t] = np.array(np.dot(egoh5[t], rot_mat.T))
    elif silent:
        for t in range(egoh5.shape[0]):
            rot_mat = np.array([[dir_arr[t, 0], dir_arr[t, 1]], [-dir_arr[t, 1], dir_arr[t, 0]]])
            egoh5[t] = np.array(np.dot(egoh5[t], rot_mat.T))
    return egoh5




def getTransitions(wregs):
    w = wregs[wregs > 0]
    return w[np.where(np.diff(w, axis=0) != 0)[0]]


def makeTransitionMatrix(states, lag):
    # states = np.flatten(states)
    states = states[states>0]
    a = np.unique(states)

    a = np.unique(states).tolist()
    a = a + [a[-1]+1]
    F = np.histogram2d(states[:-lag], states[lag:], bins=[a, a])
    sums = np.sum(F[0], axis=1)
    sums[sums == 0] = 1
    F = F[0] / sums[:, None]
    return F


def doTheShannonShuffle(states, N=None):
    L = len(states)

    if N is None:
        N = L

    vals = np.unique(states)
    M = len(vals)
    positions = []
    numTimes = np.zeros((M,))
    for i in range(M):
        positions.append(np.setdiff1d(np.where(states == vals[i])[0], L - 1))
        numTimes[i] = len(positions[i])

    outStates = np.zeros((N,))
    outStates[0] = states[np.random.randint(0, high=L - 2)]

    for i in range(1, N):
        a = np.where(vals == outStates[i - 1])[0][0]
        idx = positions[a][np.random.randint(0, high=numTimes[a])]
        outStates[i] = states[idx + 1];

    return outStates




def plotLaggedEigenvalues(transitions, lags=None, numModes=5):
    if lags is None:
        lags = np.arange(1, 11).tolist() + np.arange(15, 2000, step=5).tolist()
    cs = 'brmgck'
    N = len(transitions)
    print('Calculating Eigenvalues')
    eigs = np.zeros((len(lags), numModes + 1, N))
    for i in range(N):
        for j in range(len(lags)):
            T = makeTransitionMatrix(transitions[i], lags[j])
            vals = np.linalg.eig(T)[0]
            vals = np.sort(np.abs(vals))[::-1]
            eigs[j, :, i] = vals[:numModes + 1]
    meanEigs = np.mean(eigs, axis=2)
    semEigs = np.std(eigs, axis=2) / np.sqrt(N - 1)

    print('Calculating Markov Eignevalues')
    markov_eigs = np.zeros((len(lags), numModes + 1, N))

    for i in range(N):
        print(i)
        temp = doTheShannonShuffle(transitions[i])
        for j in range(len(lags)):
            T = makeTransitionMatrix(temp, lags[j])
            vals = np.linalg.eig(T)[0]
            vals = np.sort(np.abs(vals))[::-1]
            markov_eigs[j, :, i] = vals[:numModes + 1]
    meanEigs_markov = np.mean(markov_eigs, axis=2)
    semEigs_markov = np.std(markov_eigs, axis=2) / np.sqrt(N - 1)

    outputStruct = easydict.EasyDict({})
    outputStruct.eigs = eigs
    outputStruct.lags = lags
    outputStruct.semEigs = semEigs
    outputStruct.meanEigs = meanEigs
    outputStruct.semEigs_markov = semEigs_markov
    outputStruct.meanEigs_markov = meanEigs_markov
    outputStruct.N = N
    outputStruct.numModes = numModes

    print('Generating Plot')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]

    for i in range(numModes):
        j = i % 6
        ax.fill_between(lags, meanEigs[:, i + 1] - semEigs[:, i + 1], meanEigs[:, i + 1] + semEigs[:, i + 1],
                        color=cs[j], edgecolor=None, alpha=0.4)

        ax.plot(lags, meanEigs[:, i + 1], color=cs[j])

    for i in range(numModes):
        j = i % 6

        ax.plot(lags, meanEigs_markov[:, i + 1], color=cs[j], linestyle='--', linewidth=1)

    ax.set_xscale('log')
    ax.set_xlim([0, 2e3])

    ax.set_xlabel('# of Transitions', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'|$\lambda_i$|', fontsize=16, fontweight='bold')

    ax.legend([ax.plot([], [], '-', color=cs[i], label='i = %i' % i)[0] for i in range(numModes)],
              ['i = %i' % i for i in range(numModes)])

    ax = axes[1]
    for i in range(numModes):
        j = i % 6
        z = np.sqrt(semEigs[:, i + 1] ** 2 + semEigs_markov[:, i + 1] ** 2)

        ax.fill_between(lags, meanEigs[:, i + 1] - meanEigs_markov[:, i + 1] - z,
                        meanEigs[:, i + 1] - meanEigs_markov[:, i + 1] + z, color=cs[j], edgecolor=None, alpha=0.4)
        ax.plot(lags, meanEigs[:, i + 1] - meanEigs_markov[:, i + 1], color=cs[j])

    ax.plot([1, np.max(lags)], [0, 0], 'k--', linewidth=2)

    ax.set_xscale('log')
    ax.set_xlim([0, 2e3])

    ax.set_xlabel('# of Transitions', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'|$\lambda_i$| - |$\lambda^{Markov}_i$|', fontsize=16, fontweight='bold')
    ax.legend([ax.plot([], [], '-', color=cs[i], label='i = %i' % i)[0] for i in range(numModes)],
              ['i = %i' % i for i in range(numModes)])
    return fig, axes, outputStruct

def makeregionvideos_mice(parameters, h5s, clips, dsetnames, minLength=10, maxlength=90, subs=2):
    animfps = 30.0
    pad = 100

    print('Loading .mat file with groups.')
    wshedfile = hdf5storage.loadmat(parameters.projectPath + '/%s/zVals_wShed_groups.mat'%parameters.method)
    groups = wshedfile['groups'] - 1

    indcheck = [np.argwhere(np.array(dsetnames)==wzname[0][0].split('_pcaModes')[0])[0][0] for wzname in wshedfile['zValNames'][0]]
    h5s = [h5s[i] for i in indcheck]
    clips = [clips[i] for i in indcheck]
    dsetnames = [dsetnames[i] for i in indcheck]

    for h5, l in zip(h5s, wshedfile['zValLens'][0]):
        assert (h5.shape[0] == l)

    for wzname, dname in zip(wshedfile['zValNames'][0], dsetnames):
        print(wzname[0][0].split('_pcaModes')[0], dname)
        print('the above pairs should match')

    connections = [[0, 4, 7, 10, 13], [0, 3, 6, 9, 12, 13], [0, 5, 8, 11, 13], [1, 2, 3], [4, 5, 6], [7, 8, 9],
                   [10, 11, 12]]
    wshedregs = np.max(wshedfile['watershedRegions'])
    outputdir = parameters.projectPath + '/%s/RegionVids%i/' % (parameters.method, wshedregs)
    try:
        os.mkdir(outputdir)
    except:
        pass
    for region in range(wshedregs):
        outfile = outputdir + 'regions_' + '%.3i' % (region + 1) + '.mp4'
        if os.path.exists(outfile):
            print('Video %s already exists. Skipping.' % outfile)
            continue
        else:
            print('Making video at %s.' % outfile)
        try:
            tqdm._instances.clear()
        except:
            pass
        if not groups[region][0].shape[0] or groups[region][0].shape[0] == 1:
            print('Region %i has no videos.' % (region))
            continue
        nframes = np.atleast_1d(np.diff(groups[region][0][:, 1:], axis=1).squeeze())
        nplots = min(subs * subs, np.sum((nframes < maxlength) & (nframes > minLength)))
        longinds = np.where((nframes < maxlength) & (nframes > minLength))[0]
        if not longinds.shape[0]:
            print('No videos for region %i' % (region + 1))
            continue
        # nplots = min(subs * subs, np.sum(nframes<600))
        # if not nplots:
        #     print('Shortest video segments is %i frames long. Skipping to next.'%np.min(nframes))
        #     continue
        # longinds = np.where(nframes<600)[0]

        selectedclips = longinds[np.argsort(nframes[longinds])[::-1]][:nplots]

        vidindslist = groups[region][0][selectedclips, 0]
        framestoplot = np.array([np.arange(groups[region][0][i, 1], groups[region][0][i, 2]) for i in selectedclips])
        maxsize = max([i.shape[0] for i in framestoplot])

        print(region + 1, ' starting')

        dnames = [dsetnames[0][v].split('/')[-1].split('.')[0].split('session')[0][:-1] for v in vidindslist]
        framestoplot = framestoplot[np.argsort(dnames)]
        vidindslist = vidindslist[np.argsort(dnames)]
        dnames = [dnames[i] for i in np.argsort(dnames)]

        frames = []

        print('Reading maximum %i frames from %i videos' % (maxsize, nplots))
        print([(f, i.shape[0]) for i, f in zip(framestoplot, vidindslist)])
        for i, v in tqdm(enumerate(vidindslist)):
            fr = np.zeros((len(framestoplot[i]), 2 * pad, 2 * pad))
            for j, f in enumerate(framestoplot[i]):
                frame_region = clips[v].get_frame(f / clips[v].fps)[:, :, 0]
                xmin = np.round(h5s[v][f, 8, 0]).astype('int') - pad
                ymin = np.round(h5s[v][f, 8, 1]).astype('int') - pad
                ymax = ymin + 2 * pad
                xmax = xmin + 2 * pad

                if xmin < 0 or ymin < 0 or xmax > frame_region.shape[1] - 1 or ymax > frame_region.shape[0] - 1:
                    if xmax > frame_region.shape[1] - 1 + pad or ymax > frame_region.shape[0] - 1 + pad:
                        excess = max(frame_region.shape[1] - 1 + pad - xmax, frame_region.shape[0] - 1 + pad - ymax)
                        frame_region = np.pad(frame_region,
                                              ((pad + excess, pad + excess), (pad + excess, pad + excess)), 'minimum')
                    else:
                        frame_region = np.pad(frame_region, ((pad, pad), (pad, pad)), 'minimum')
                    frame_region = frame_region[ymin + pad:ymax + pad, xmin + pad:xmax + pad]

                else:
                    frame_region = frame_region[ymin:ymax, xmin:xmax]
                fr[j, :, :] = frame_region
            frames.append(fr)
        frames = np.array(frames)
        subx = max(2, int(np.ceil(np.sqrt(nplots))))
        fig, axes = plt.subplots(subx, subx, figsize=(12, 12))
        fig.subplots_adjust(0, 0, 1.0, 1.0, 0.0, 0.0)

        def make_frame(t):
            j_ = int(t * animfps)
            for i in range(subx * subx):

                ax = axes[i // subx, i % subx]
                ax.clear()
                ax.axis('off')
                if i >= nplots:
                    continue

                j = j_ % len(framestoplot[i])
                xmin = np.round(h5s[vidindslist[i]][framestoplot[i][j], 8, 0]).astype('int') - pad
                ymin = np.round(h5s[vidindslist[i]][framestoplot[i][j], 8, 1]).astype('int') - pad
                ax.imshow(frames[i][j], cmap='Greys_r')
                for conn in connections:
                    ax.plot(h5s[vidindslist[i]][framestoplot[i][j], conn, 0] - (xmin),
                            h5s[vidindslist[i]][framestoplot[i][j], conn, 1] - (ymin))

                # ax.text(50, 20, dnames[i], color='red', fontsize=14)
            return mplfig_to_npimage(fig)

        try:
            tqdm._instances.clear()
        except:
            pass
        animation = VideoClip(make_frame, duration=maxsize / animfps)
        animation.write_videofile(outfile, fps=animfps, audio=False,
                                  threads=1)
        print('Video saved at %s.' % outfile)
        plt.close()


def makeregionvideo_mice(region, parameters, h5s, clips, dsetnames, minLength=10, maxLength=90, subs=2):
    wshedfile = hdf5storage.loadmat(parameters.projectPath + '/%s/zVals_wShed_groups.mat'%parameters.method)

    wshedregs = np.max(wshedfile['watershedRegions'])

    outputdir = parameters.projectPath + '/%s/RegionVids%i/' % (parameters.method, wshedregs)
    outfile = outputdir + 'regions_' + '%.3i' % (region + 1) + '.mp4'
    if os.path.exists(outfile):
        print('Video %s already exists. Skipping.' % outfile)
        return

    animfps = 30.0
    pad = 100

    print('Loading .mat file with groups.')
    groups = wshedfile['groups'] - 1

    indcheck = [np.argwhere(np.array(dsetnames)==wzname[0][0].split('_pcaModes')[0])[0][0] for wzname in wshedfile['zValNames'][0]]
    h5s = [h5s[i] for i in indcheck]
    clips = [clips[i] for i in indcheck]
    dsetnames = [dsetnames[i] for i in indcheck]

    for h5, l in zip(h5s, wshedfile['zValLens'][0]):
        assert (h5.shape[0] == l)

    for wzname, dname in zip(wshedfile['zValNames'][0], dsetnames):
        print(wzname[0][0].split('_pcaModes')[0], dname)
        print('the above pairs should match')

    connections = [[0, 4, 7, 10, 13], [0, 3, 6, 9, 12, 13], [0, 5, 8, 11, 13], [1, 2, 3], [4, 5, 6], [7, 8, 9],
                   [10, 11, 12]]
    try:
        os.mkdir(outputdir)
    except:
        pass

    outfile = outputdir + 'regions_' + '%.3i' % (region + 1) + '.mp4'
    if os.path.exists(outfile):
        print('Video %s already exists. Skipping.' % outfile)
        return
    else:
        print('Making video at %s.' % outfile)
    try:
        tqdm._instances.clear()
    except:
        pass
    if not groups[region][0].shape[0] or groups[region][0].shape[0] == 1:
        print('Region %i has no videos.' % (region))
        return
    nframes = np.atleast_1d(np.diff(groups[region][0][:, 1:], axis=1).squeeze())
    nplots = min(subs * subs, np.sum((nframes < maxLength) & (nframes > minLength)))
    longinds = np.where((nframes < maxLength) & (nframes > minLength))[0]
    if not longinds.shape[0]:
        print('No videos for region %i' % (region + 1))
        return
    # nplots = min(subs * subs, np.sum(nframes<600))
    # if not nplots:
    #     print('Shortest video segments is %i frames long. Skipping to next.'%np.min(nframes))
    #     continue
    # longinds = np.where(nframes<600)[0]

    selectedclips = longinds[np.argsort(nframes[longinds])[::-1]][:nplots]

    vidindslist = groups[region][0][selectedclips, 0]
    framestoplot = np.array([np.arange(groups[region][0][i, 1], groups[region][0][i, 2]) for i in selectedclips])
    maxsize = max([i.shape[0] for i in framestoplot])

    print(region + 1, ' starting')

    dnames = [dsetnames[0][v].split('/')[-1].split('.')[0].split('session')[0][:-1] for v in vidindslist]
    framestoplot = framestoplot[np.argsort(dnames)]
    vidindslist = vidindslist[np.argsort(dnames)]
    dnames = [dnames[i] for i in np.argsort(dnames)]

    frames = []

    print('Reading maximum %i frames from %i videos' % (maxsize, nplots))
    print([(f, i.shape[0]) for i, f in zip(framestoplot, vidindslist)])
    for i, v in tqdm(enumerate(vidindslist)):
        fr = np.zeros((len(framestoplot[i]), 2 * pad, 2 * pad))
        for j, f in enumerate(framestoplot[i]):
            frame_region = clips[v].get_frame(f / clips[v].fps)[:, :, 0]
            xmin = np.round(h5s[v][f, 8, 0]).astype('int') - pad
            ymin = np.round(h5s[v][f, 8, 1]).astype('int') - pad
            ymax = ymin + 2 * pad
            xmax = xmin + 2 * pad

            if xmin < 0 or ymin < 0 or xmax > frame_region.shape[1] - 1 or ymax > frame_region.shape[0] - 1:
                if xmax > frame_region.shape[1] - 1 + pad or ymax > frame_region.shape[0] - 1 + pad:
                    excess = max(frame_region.shape[1] - 1 + pad - xmax, frame_region.shape[0] - 1 + pad - ymax)
                    frame_region = np.pad(frame_region,
                                          ((pad + excess, pad + excess), (pad + excess, pad + excess)), 'minimum')
                else:
                    frame_region = np.pad(frame_region, ((pad, pad), (pad, pad)), 'minimum')
                frame_region = frame_region[ymin + pad:ymax + pad, xmin + pad:xmax + pad]

            else:
                frame_region = frame_region[ymin:ymax, xmin:xmax]
            fr[j, :, :] = frame_region
        frames.append(fr)
    frames = np.array(frames)
    subx = max(2, int(np.ceil(np.sqrt(nplots))))
    fig, axes = plt.subplots(subx, subx, figsize=(12, 12))
    fig.subplots_adjust(0, 0, 1.0, 1.0, 0.0, 0.0)

    def make_frame(t):
        j_ = int(t * animfps)
        for i in range(subx * subx):

            ax = axes[i // subx, i % subx]
            ax.clear()
            ax.axis('off')
            if i >= nplots:
                continue

            j = j_ % len(framestoplot[i])
            xmin = np.round(h5s[vidindslist[i]][framestoplot[i][j], 8, 0]).astype('int') - pad
            ymin = np.round(h5s[vidindslist[i]][framestoplot[i][j], 8, 1]).astype('int') - pad
            ax.imshow(frames[i][j], cmap='Greys_r')
            for conn in connections:
                ax.plot(h5s[vidindslist[i]][framestoplot[i][j], conn, 0] - (xmin),
                        h5s[vidindslist[i]][framestoplot[i][j], conn, 1] - (ymin))

            # ax.text(50, 20, dnames[i], color='red', fontsize=14)
        return mplfig_to_npimage(fig)

    try:
        tqdm._instances.clear()
    except:
        pass
    animation = VideoClip(make_frame, duration=maxsize / animfps)
    animation.write_videofile(outfile, fps=animfps, audio=False,
                              threads=1)
    print('Video saved at %s.' % outfile)
    plt.close()

def makeregionvideo_flies(region, parameters, wshedfile, clips, subs = 2, minLength=10, maxLength=100):
    animfps = 50.0
    submaxframes = 500

    groups = makeGroupsAndSegments(wshedfile['watershedRegions'][0], wshedfile['zValLens'][0],
                                   min_length=minLength, max_length=maxLength)
    nregs = len(groups)

    region = region - 1

    outputdir = '%s/%s/region_vidoes_%i/' % (parameters.projectPath, parameters.method, nregs)
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    groups = groups - 1
    print('[Region %i] Starting' % (region + 1))

    if os.path.isfile(outputdir + 'regions_' + '%.3i' % (region + 1) + '.mp4'):
        print('[Region %i] Already present. ' % (region + 1))
        return

    tqdm._instances.clear()

    if not groups[region][0].shape[0] or groups[region][0].shape[0] == 1:
        print('[Region %i] No frames in groups.' % (region + 1))
        return

    nframes = np.atleast_1d(np.diff(groups[region][0][:, 1:], axis=1).squeeze())
    if np.sum(nframes < submaxframes) == 0:
        print('[Region %i] All frames sequences more than length %i.' %
              (region + 1, submaxframes))
        return

    nplots = min(subs * subs, np.sum(nframes < submaxframes))
    longinds = np.where(nframes < submaxframes)[0]
    selectedclips = longinds[np.argsort(nframes[longinds])[::-1]][:nplots]

    vidindslist = groups[region][0][selectedclips, 0]
    framestoplot = np.array([np.arange(groups[region][0][i, 1], groups[region][0][i, 2]) for i in selectedclips])
    maxsize = max([i.shape[0] for i in framestoplot])

    print('[Region %i] Making region video...' % (region + 1))

    subx = max(2, int(np.ceil(np.sqrt(nplots))))
    fig, axes = plt.subplots(subx, subx, figsize=(12, 12))
    fig.subplots_adjust(0, 0, 1.0, 1.0, 0.0, 0.0)

    def make_frame(t):
        j_ = int(t * animfps)
        for i in range(subx * subx):

            ax = axes[i // subx, i % subx]
            ax.clear()
            ax.axis('off')
            if i >= nplots:
                continue
            j = j_ % len(framestoplot[i])
            clip = clips[vidindslist[i]]
            ax.imshow(clip.get_frame(framestoplot[i][j] / clip.fps),
                      cmap='Greys_r', origin='lower')
        return mplfig_to_npimage(fig)

    try:
        tqdm._instances.clear()
    except:
        pass

    t1 = time.time()
    animation = VideoClip(make_frame, duration=maxsize / animfps)

    animation.write_videofile(outputdir + 'regions_' + '%.3i' % (region + 1) + '.mp4', fps=animfps, audio=False,
                              threads=1)

    print('[Region %i] %i seconds, Saved at %s' % (
    region + 1, time.time() - t1, outputdir + 'regions_' + '%.3i' % (region + 1) + '.mp4'))