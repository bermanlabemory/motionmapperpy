"""
Exact copy of wavelet calculation in Gordon Berman's MotionMapper code.

Written by :
Kanishk Jain
kanishkbjain@gmail.com
"""

import time
import warnings
def findWavelets(projections, pcaModes, omega0, numPeriods, samplingFreq, maxF, minF, numProcessors, useGPU):
    """
    findWavelets finds the wavelet transforms resulting from a time series.
    :param projections: N x d array of projection values.
    :param pcaModes: # of transforms to find.
    :param omega0: Dimensionless morlet wavelet parameter.
    :param numPeriods: number of wavelet frequencies to use.
    :param samplingFreq: sampling frequency (Hz).
    :param maxF: maximum frequency for wavelet transform (Hz).
    :param minF: minimum frequency for wavelet transform (Hz).
    :param numProcessors: number of processors to use in parallel code.
    :param useGPU: GPU to use.
    :return:
            amplitudes -> wavelet amplitudes (N x (pcaModes*numPeriods) )
            f -> frequencies used in wavelet transforms (Hz)

    """
    t1 = time.time()
    print('\t Calculating wavelets, clock starting.')

    if useGPU>=0:
        try:
            import cupy as np
        except ModuleNotFoundError as E:
            warnings.warn("Trying to use GPU but cupy is not installed. Install cupy or set parameters.useGPU = -1. "
                  "https://docs.cupy.dev/en/stable/install.html")
            raise E

        np.cuda.Device(useGPU).use()
        print('\t Using GPU #%i'%useGPU)
    else:
        import numpy as np
        import multiprocessing as mp
        if numProcessors<0:
            numProcessors = mp.cpu_count()
        print('\t Using #%i CPUs.' % numProcessors)

    projections = np.array(projections)
    t1 = time.time()

    dt = 1.0 / samplingFreq
    minT = 1.0 / maxF
    maxT = 1.0 / minF
    Ts = minT * (2 ** ((np.arange(numPeriods) * np.log(maxT / minT)) / (np.log(2) * (numPeriods - 1))))
    f = (1.0 / Ts)[::-1]
    N = projections.shape[0]

    if useGPU>=0:
        amplitudes = np.zeros((numPeriods*pcaModes,N))
        for i in range(pcaModes):
            amplitudes[i*numPeriods:(i+1)*numPeriods] = fastWavelet_morlet_convolution_parallel(i, projections[:, i], f, omega0, dt, useGPU)
    else:
        try:
            pool = mp.Pool(numProcessors)
            amplitudes = pool.starmap(fastWavelet_morlet_convolution_parallel,
                                      [(i, projections[:, i], f, omega0, dt, useGPU) for i in range(pcaModes)])
            amplitudes = np.concatenate(amplitudes, 0)
            pool.close()
            pool.join()
        except Exception as E:
            pool.close()
            pool.join()
            raise E
    print('\t Done at %0.02f seconds.'%(time.time()-t1))
    return amplitudes.T, f


def fastWavelet_morlet_convolution_parallel(modeno, x, f, omega0, dt, useGPU):
    if useGPU>=0:
        import cupy as np
        np.cuda.Device(useGPU).use()
    else:
        import numpy as np
    N = len(x)
    L = len(f)
    amp = np.zeros((L, N))

    if not N // 2:
        x = np.concatenate((x, [0]), axis=0)
        N = len(x)
        wasodd = True
    else:
        wasodd = False

    x = np.concatenate([np.zeros(int(N / 2)), x, np.zeros(int(N / 2))], axis=0)
    M = N
    N = len(x)
    scales = (omega0 + np.sqrt(2 + omega0 ** 2)) / (4 * np.pi * f)
    Omegavals = 2 * np.pi * np.arange(-N / 2, N / 2) / (N * dt)

    xHat = np.fft.fft(x)
    xHat = np.fft.fftshift(xHat)

    if wasodd:
        idx = np.arange((M / 2), (M / 2 + M - 2)).astype(int)
    else:
        idx = np.arange((M / 2), (M / 2 + M)).astype(int)

    for i in range(L):
        m = (np.pi ** (-0.25)) * np.exp(-0.5 * (-Omegavals * scales[i] - omega0) ** 2)
        q = np.fft.ifft(m * xHat) * np.sqrt(scales[i])

        q = q[idx]
        amp[i, :] = np.abs(q) * (np.pi ** -0.25) * np.exp(0.25 * (omega0 - np.sqrt(omega0 ** 2 + 2)) ** 2) / np.sqrt(
            2 * scales[i])
    # print('Mode %i done.'%(modeno))
    return amp
