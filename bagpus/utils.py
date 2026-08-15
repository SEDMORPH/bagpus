""" Generic utilities for bagpus: SFH-derived timescales, PCA compression,
population sampling and prior construction.

Everything in this module is agnostic to the choice of SFH and dust model —
model-specific code lives in bagpus.models.
"""

import copy
import os

import numpy as np
import scipy.stats as stats
import sklearn.decomposition as deco
import torch


# ---------------------------------------------------------------------------
# Quenching timescales and other properties of a bagpipes SFH object
# ---------------------------------------------------------------------------

def func_tauquench(sfh):
    """ Numerical function to use SFH to calculate the time it takes to decrease from peak SFR to half of peak. Input bagpipes sfh. """

    ind_maxsfh = np.argmax(sfh.sfh)
    age_peak = sfh.ages[ind_maxsfh]

    ind = np.where((sfh.sfh < 0.5 * sfh.sfh[ind_maxsfh]) & (sfh.ages < age_peak))[0]
    if len(ind) > 0:
        tauquench = (age_peak - sfh.ages[ind[-1]]) / 1e9
    else:
        tauquench = -99

    return tauquench, age_peak


def func_tauquench_full(sfh):
    """ Numerical function to calculate when SFR drops from peak to t_quench as defined by bagpipes"""

    ind_maxsfh = np.argmax(sfh.sfh)
    age_peak = sfh.ages[ind_maxsfh]
    age_quench = sfh.age_of_universe - sfh.tquench * 1e9

    if sfh.tquench > 0:
        tauquench = (age_peak - age_quench) / 1e9
    else:
        tauquench = -99

    return tauquench, age_peak


def func_tauquench_0p2tH(sfh):
    """ Numerical function to calculate when SFR drops from peak to below 0.2/t_H. This is very similar to tauquench_full."""

    # sfh starts at age=0. Need to reverse to get it to start at t=0. Then reverse again to get cummass to match other age arrays.
    cummass = (np.cumsum(sfh.sfh[::-1] * sfh.age_widths[::-1]))[::-1]  # Msol/yr * yr -> Msol

    ssfr = sfh.sfh / cummass  # Msol/yr / Msol -> /yr
    ind_maxsfh = np.argmax(sfh.sfh)
    age_peak = sfh.ages[ind_maxsfh]

    # find all indices where ssfr meets criterium and ages more recent than peak SFH
    # sfh.age_of_universe is in years and assumes bagpipes cosmology
    ind = np.where((ssfr < 0.2 / (sfh.age_of_universe - sfh.ages)) & (sfh.ages < age_peak))[0]

    if len(ind) > 0:
        # ages goes from zero back in time (up in age), so need the last index
        tquench = (sfh.age_of_universe - sfh.ages[ind[-1]]) / 1e9
        tauquench = (age_peak - sfh.ages[ind[-1]]) / 1e9
    else:
        tquench = -99
        tauquench = -99

    return tauquench, tquench, age_peak


def func_tauquench_simba(sfh):
    """ Numerical function to calculate when SFR drops from 1/tH to 0.2/t_H
    This will pick up first time it drops below 1 and last time it is above 0.2 i.e. 
    may not be indicative of quenching time in anything other than unimodal SFH"""

    # sfh starts at age=0. Need to reverse to get it to start at t=0. Then reverse again to get cummass to match other age arrays.
    cummass = (np.cumsum(sfh.sfh[::-1] * sfh.age_widths[::-1]))[::-1]

    ssfr = sfh.sfh / cummass
    ind_maxsfh = np.argmax(sfh.sfh)
    age_peak = sfh.ages[ind_maxsfh]

    # find all indices where ssfr meets criterium and ages more recent than peak SFH
    # sfh.age_of_universe is in years and assumes bagpipes cosmology
    ind = np.where((ssfr > 0.2 / (sfh.age_of_universe - sfh.ages))
                   & (ssfr < 1 / (sfh.age_of_universe - sfh.ages))
                   & (sfh.ages < age_peak))[0]

    if len(ind) > 0:
        # ages goes from zero back in time (up in age), so need the last index
        tauquench = (sfh.ages[ind[-1]] - sfh.ages[ind[0]]) / 1e9
    else:
        tauquench = -99

    return tauquench


def func_tauquench_ST(sfh):
    """ Numerical function to calculate quenching timescale following Tachella et al. 2022"""

    cummass = (np.cumsum(sfh.sfh[::-1] * sfh.age_widths[::-1]))[::-1]

    ssfr = sfh.sfh / cummass
    ind_maxsfh = np.argmax(sfh.sfh)

    ind = np.where((ssfr < 1 / (3 * (sfh.age_of_universe - sfh.ages)))
                   & (ssfr > 1 / (20 * (sfh.age_of_universe - sfh.ages)))
                   & (sfh.ages < sfh.ages[ind_maxsfh]))[0]

    if len(ind) > 0:
        tauquench = (sfh.ages[ind[-1]] - sfh.ages[ind[0]]) / 1e9
    else:
        tauquench = -99

    return tauquench

#def func_t90(sfh):
#    """ The lookback time (age) at which 90% of the mass is formed """
#    cummass = (np.cumsum(sfh.sfh[::-1] * sfh.age_widths[::-1]))[::-1]
#    ind = np.argmin(abs(cummass - 0.9 * 10**sfh.formed_mass))
#    t90 = sfh.ages[ind] / 1e9
#    return t90

def func_tXX(sfh,XX):
    """ The cosmic time in Gyr at which fraction XX of the mass is formed e.g. set XX=0.9 for t90."""
    """ Input bagpipes sfh structure, XX is fraction of mass formed. """

    cummass = (np.cumsum(sfh.sfh[::-1]*sfh.age_widths[::-1]))[::-1]
    ind = np.argmin(abs(cummass-XX*np.max(cummass)))
    t_cum = (sfh.age_of_universe-sfh.ages[ind])/1e9
    
    return t_cum

def props_from_sfh(sfh):
    """ Extract the standard set of derived properties from a single bagpipes
    star_formation_history object. Works for any SFH parameterisation.

    Returns a dict of scalars with keys: unphysical, ssfr, tquench, tform,
    t90, tauquench_ST, tauquench_simba, tauquench_init, age_peak,
    tauquench_full.
    """
    props = {}
    props['unphysical'] = sfh.unphysical
    if sfh.sfr != 0:
        props['ssfr'] = sfh.ssfr
    else:
        props['ssfr'] = -24  # set to arbitrary low number, otherwise dust becomes undefined
    props['tquench'] = sfh.tquench
    props['tform'] = sfh.tform
    props['t10']=func_tXX(sfh,0.1)
    props['t50']=func_tXX(sfh,0.5)
    props['t90']=func_tXX(sfh,0.9)
    props['tauquench_ST'] = func_tauquench_ST(sfh)
    props['tauquench_simba'] = func_tauquench_simba(sfh)
    props['tauquench_init'], props['age_peak'] = func_tauquench(sfh)
    props['tauquench_full'], _ = func_tauquench_full(sfh)

    return props


DERIVED_PROP_NAMES = ['unphysical', 'ssfr', 'tquench', 'tform', 't10', 't50', 't90',
                      'tauquench_ST', 'tauquench_simba', 'tauquench_init',
                      'age_peak', 'tauquench_full']


# ---------------------------------------------------------------------------
# PCA compression of the 2D SC probability distributions
# ---------------------------------------------------------------------------

def func_compress_pca(x, pdf_range, floor=0.0001, n_components=20, diagnostic_plot=False):
    """ Fit a PCA basis to a stack of simulated 2D SC histograms and return the
    compressed representation used as the SBI summary statistic. """

    nsims = x.shape[0]
    nbins0 = x.shape[1]
    nbins1 = x.shape[2]

    # work on log of histogram
    x2 = copy.deepcopy(x)
    x2[x2 == 0] = floor  # remove infinite values in log
    x2 = np.log10(x2)

    # calculate and subtract mean array.
    # This is actually done in the PCA implementation below, but not sure how to
    # recover it from the code to use it on other datasets, so do it explicitly here.
    meanarr = x2.mean(0)
    x3 = (x2 - meanarr)

    x4 = np.zeros((nsims, nbins0 * nbins1))
    for i in range(nsims):
        x4[i, :] = x3[i, :, :].flatten()

    pca = deco.PCA(n_components)
    x_r = pca.fit(x4).transform(x4)

    print('explained variance (first %d components): %.2f' % (n_components, sum(pca.explained_variance_ratio_)))

    if diagnostic_plot == True:
        import matplotlib.pyplot as plt
        nrow = 1
        ncol = 3
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, sharey=True, figsize=(2 * (ncol + 1), 2 * (nrow + 1)), layout='compressed')

        im0 = axes[0].imshow(meanarr[:, :].T, cmap='GnBu', extent=[pdf_range[0][0], pdf_range[0][1], pdf_range[1][1], pdf_range[1][0]])
        axes[0].set_xlabel('SC1')
        axes[0].set_ylabel('SC2')
        axes[0].set_title('mean array')

        im1 = axes[1].imshow(x2[50, :, :].T, cmap='GnBu', extent=[pdf_range[0][0], pdf_range[0][1], pdf_range[1][1], pdf_range[1][0]])
        axes[1].set_xlabel('SC1')
        axes[1].set_title('example simulation')

        im2 = axes[2].imshow(x3[50, :, :].T, cmap='GnBu', extent=[pdf_range[0][0], pdf_range[0][1], pdf_range[1][1], pdf_range[1][0]])
        axes[2].set_xlabel('SC1')
        axes[2].set_title('mean subtracted')

        for ax in axes.flat:
            ax.set_aspect(3)
            ax.set_ylim((pdf_range[1][0], pdf_range[1][1]))

        plt.show()

        plt.plot(np.log10(pca.explained_variance_ratio_))
        plt.ylabel('log10 explained variance ratio')
        plt.xlabel('component number')
        plt.show()

    return meanarr, pca, x_r


def func_project_pca(pdf_2d, meanarr, pca, floor=0.0001):
    """Project data onto PCA eigenbasis"""

    # Prepare for PCA in the same way as we prepared the PCA input
    xsize = pdf_2d.shape[0]
    ysize = pdf_2d.shape[1]

    pdf2_2d = copy.deepcopy(pdf_2d)
    pdf2_2d[pdf2_2d == 0] = floor  # remove infinite values in log
    pdf2_2d = np.log10(pdf2_2d)
    pdf3_2d = (pdf2_2d - meanarr)  # remove the eigen-basis mean
    pdf4_2d = pdf3_2d.flatten()

    pca_amps = np.dot(pdf4_2d, pca.components_.T)
    pdf_recon = np.dot(pca_amps, pca.components_)

    # put back into 2D shape, add the mean arr and undo the log to resemble the input data
    pdf_recon = 10**(np.add(pdf_recon.reshape(xsize, ysize), meanarr))

    return pdf_recon, pca_amps


# ---------------------------------------------------------------------------
# Population sampling and SBI prior
# ---------------------------------------------------------------------------

def draw_rvs(theta, model, ngal=1000):
    """Draw ngal random variables from the hyperparameters (theta) of model.

    model can be a PopulationModel or a plain dict; it must provide
    model['param_names'] and, for each name, model[name] = (lower, upper)
    truncation limits. theta is ordered as (mean, sd) pairs following
    param_names order.
    """
    param_rvs = {}

    ii = 0
    for name in model['param_names']:
        lower = model[name][0]
        upper = model[name][1]
        mean = theta[ii]
        sd = theta[ii + 1]
        param = stats.truncnorm((lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd)
        param_rvs[name] = param.rvs(ngal)
        ii += 2

    return param_rvs


def def_prior(model, return_limits=False):
    """ Set SBI's prior object for the model"""

    from sbi.utils import BoxUniform

    # set uniform priors on all parameters with upper and lower limits.
    # note these are not priors on the input parameters to bagpipes (alpha, beta etc)
    # but priors on their distributions, here assumed to be a Gaussian with mean and sigma
    lower = []
    for name in model['param_names']:
        lower.append(model[name + ':mean'][0])
        lower.append(model[name + ':sd'][0])

    upper = []
    for name in model['param_names']:
        upper.append(model[name + ':mean'][1])
        upper.append(model[name + ':sd'][1])

    lower_bound = torch.as_tensor(lower)
    upper_bound = torch.as_tensor(upper)
    prior = BoxUniform(low=lower_bound, high=upper_bound)

    if return_limits == True:
        return prior, lower_bound, upper_bound
    else:
        return prior


# ---------------------------------------------------------------------------
# Run directory management
# ---------------------------------------------------------------------------

def make_run_dirs(runID, working_dir='runs/'):
    """ Make local bagpus directory structure for a named run.

    Returns (dir_figs, dir_training, dir_test, dir_posterior). """

    os.makedirs(working_dir, exist_ok=True)

    dirs = []
    for sub in ['plots', 'training', 'test', 'posterior']:
        d = os.path.join(working_dir, sub, runID) + '/'
        os.makedirs(d, exist_ok=True)
        dirs.append(d)

    return tuple(dirs)
