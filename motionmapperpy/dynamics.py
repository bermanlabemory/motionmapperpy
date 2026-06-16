"""
Behavioral transition / timescale analysis  (Berman, Bialek & Shaevitz, PNAS 2016).

Clean, public versions of the transition tools that previously lived only in ``demoutils``.
Two improvements over the ``demoutils`` versions (which are kept, unchanged, for backward
compatibility):

* :func:`makeTransitionMatrix` accepts a single state array **or a list of arrays** (one per
  recording) and counts transitions *within* each sequence only -- never across the splice
  between recordings.
* the matrix is a fixed ``n_states x n_states`` indexed by state value, so matrices built from
  different datasets are directly comparable (rather than being sized to whichever states
  happen to be present).
"""
import numpy as np
import easydict
import matplotlib.pyplot as plt


def getTransitions(states):
    """Collapse a per-frame state sequence to the sequence of *distinct* states visited.

    Drops state 0 (the non-stereotyped / unassigned label) and consecutive repeats, so the
    result is the sequence of behavioral transitions used throughout the 2016 analysis.
    """
    states = np.asarray(states)
    w = states[states > 0]
    return w[np.where(np.diff(w) != 0)[0]]


def _accumulate(F, states, lag):
    s = np.asarray(states)
    s = s[s > 0].astype(int)
    if len(s) > lag:
        np.add.at(F, (s[:-lag], s[lag:]), 1.0)


def makeTransitionMatrix(states, lag, n_states=None, normalize=True):
    """Row-stochastic transition matrix ``T[i, j] = P(state j at t+lag | state i at t)``.

    :param states: 1-D integer state array, OR a list/tuple of such arrays. When a list is
        given, transitions are counted within each sequence only (no cross-recording counts).
    :param lag: transition lag (in frames / transitions).
    :param n_states: matrix dimension; defaults to ``max(state) + 1`` across all sequences, so
        the row/column index equals the state value (index 0 is the unused state-0 slot).
    :param normalize: if True (default) rows are normalized to sum to 1.
    :return: ``(n_states, n_states)`` array.
    """
    seqs = list(states) if isinstance(states, (list, tuple)) else [states]
    seqs = [np.asarray(s) for s in seqs if len(np.asarray(s)) > 0]
    if not seqs:
        raise ValueError('No non-empty state sequences provided.')
    if n_states is None:
        n_states = int(max(s[s > 0].max() if np.any(s > 0) else 0 for s in seqs)) + 1
    F = np.zeros((n_states, n_states))
    for s in seqs:
        _accumulate(F, s, lag)
    if normalize:
        sums = F.sum(1)
        sums[sums == 0] = 1
        F = F / sums[:, None]
    return F


def doTheShannonShuffle(states, N=None):
    """Order-1 Markov surrogate of a state sequence.

    Resamples a sequence that preserves only one-step transition statistics, destroying any
    longer-timescale structure -- the null distribution for :func:`laggedEigenvalues`.
    """
    states = np.asarray(states)
    L = len(states)
    if N is None:
        N = L
    vals = np.unique(states)
    M = len(vals)
    positions, numTimes = [], np.zeros(M)
    for i in range(M):
        positions.append(np.setdiff1d(np.where(states == vals[i])[0], L - 1))
        numTimes[i] = len(positions[i])
    out = np.zeros(N)
    out[0] = states[np.random.randint(0, L - 1)]
    for i in range(1, N):
        a = np.where(vals == out[i - 1])[0][0]
        idx = positions[a][np.random.randint(0, int(numTimes[a]))]
        out[i] = states[idx + 1]
    return out


def laggedEigenvalues(transitions, lags=None, numModes=5, markovNull=True, n_states=None):
    """Eigenvalue magnitudes ``|lambda_i|`` of ``T(lag)`` as a function of lag.

    :param transitions: list of per-recording state sequences (e.g. ``[getTransitions(s) for s
        in states_list]``).
    :param markovNull: if True, also compute the order-1 Markov (Shannon-shuffle) null.
    :return: EasyDict with ``lags, eigs, meanEigs, semEigs`` (and ``*_markov`` if requested).
        No plotting -- see :func:`plotLaggedEigenvalues`.
    """
    if lags is None:
        lags = np.arange(1, 11).tolist() + np.arange(15, 2000, 5).tolist()
    transitions = [np.asarray(t) for t in transitions if len(np.asarray(t)) > 0]
    N = len(transitions)
    if n_states is None:
        n_states = int(max(t[t > 0].max() for t in transitions)) + 1

    def _spectrum(seq):
        e = np.zeros((len(lags), numModes + 1))
        for j, lag in enumerate(lags):
            T = makeTransitionMatrix(seq, lag, n_states=n_states)
            vals = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
            e[j] = vals[:numModes + 1]
        return e

    eigs = np.stack([_spectrum(t) for t in transitions], axis=2)
    out = easydict.EasyDict({})
    out.lags = lags
    out.eigs = eigs
    out.meanEigs = eigs.mean(2)
    out.semEigs = eigs.std(2) / np.sqrt(max(N - 1, 1))
    out.N = N
    out.numModes = numModes
    if markovNull:
        meigs = np.stack([_spectrum(doTheShannonShuffle(t)) for t in transitions], axis=2)
        out.meanEigs_markov = meigs.mean(2)
        out.semEigs_markov = meigs.std(2) / np.sqrt(max(N - 1, 1))
    return out


def plotLaggedEigenvalues(transitions, lags=None, numModes=5, n_states=None):
    """Compute (:func:`laggedEigenvalues`) and plot ``|lambda_i|`` vs lag with the Markov null
    overlaid. Returns ``(fig, axes, outputStruct)``."""
    out = laggedEigenvalues(transitions, lags, numModes, markovNull=True, n_states=n_states)
    lags = out.lags
    me, se, mm, sm = out.meanEigs, out.semEigs, out.meanEigs_markov, out.semEigs_markov
    cs = 'brmgck'
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for i in range(numModes):
        j = i % 6
        ax.fill_between(lags, me[:, i + 1] - se[:, i + 1], me[:, i + 1] + se[:, i + 1],
                        color=cs[j], edgecolor=None, alpha=0.4)
        ax.plot(lags, me[:, i + 1], color=cs[j], label='i = %i' % (i + 1))
    for i in range(numModes):
        ax.plot(lags, mm[:, i + 1], color=cs[i % 6], linestyle='--', linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('# of Transitions', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'|$\lambda_i$|', fontsize=14, fontweight='bold')
    ax.legend()
    ax = axes[1]
    for i in range(numModes):
        j = i % 6
        z = np.sqrt(se[:, i + 1] ** 2 + sm[:, i + 1] ** 2)
        d = me[:, i + 1] - mm[:, i + 1]
        ax.fill_between(lags, d - z, d + z, color=cs[j], edgecolor=None, alpha=0.4)
        ax.plot(lags, d, color=cs[j])
    ax.axhline(0, color='k', linestyle='--', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('# of Transitions', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'|$\lambda_i$| - |$\lambda^{Markov}_i$|', fontsize=14, fontweight='bold')
    return fig, axes, out
