""" Plotting functions for bagpus: population parameter distributions, derived
property distributions, posterior predictive SC maps and residuals.

All functions take a PopulationModel (dict-style access is used for ranges and
labels; plot_derived also needs its derived_props method). Figures are
returned so they display inline in notebooks; pass fname/figname to save.
"""

import copy
import string

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec

from . import utils

alphabet = list(string.ascii_lowercase)


def plot_params(theta_samples, popmodel, nbins=20, ngal=1000, title=None, label=None,
                axes=None, color='orange', pdf_extra=None, label_extra=None,
                color_extra='blue', fname=None):
    """ Visualise the impact of the hyperparameters (theta) on the model parameters (sfh, dust)"""

    ### set up arrays
    ndraws = theta_samples.shape[0]  # number of draws from the prior
    nparams = len(popmodel['param_names'])  # number of model parameters that are determined by the population parameters

    pdf_model_rvs = np.zeros((ndraws, nparams, nbins))  # PDFs of model parameters
    pdf_edges = np.zeros((nparams, nbins + 1))  # bin edges for plotting

    ### for each draw from the theta distribution, calculate the model parameters, and store the probability distribution function
    for i in range(ndraws):
        theta = theta_samples[i, :]

        model_rvs = utils.draw_rvs(theta, popmodel, ngal=ngal)

        for j, name in enumerate(popmodel['param_names']):
            pdf_model_rvs[i, j, :], pdf_edges[j, :] = np.histogram(
                model_rvs[name], range=[popmodel[name][0], popmodel[name][1]], bins=nbins, density=True)

    # calculate percentiles for each bin in the distribution
    perc = np.percentile(pdf_model_rvs, [16, 50, 84], axis=0)
    if pdf_extra is not None:
        perc_extra = np.percentile(pdf_extra['pdf'], [16, 50, 84], axis=0)

    #### plot distribution of model parameters
    if axes is None:
        ncol = 3
        nrow = int(np.ceil(nparams / ncol))
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(2 * (ncol + 1), 2 * (nrow + 1)), layout='compressed')
        if title is not None:
            fig.suptitle(title)
    else:
        fig = axes.flat[0].figure

    cmap = mpl.colormaps['Oranges']  # for individual draws
    n_lines = 10
    colors = cmap(np.linspace(0, 1, n_lines))

    for j, ax in enumerate(axes.flat):
        if j >= nparams:  # remove axes where there are no more parameters
            ax.set_visible(False)
            continue

        ax.stairs(perc[1, j, :], pdf_edges[j, :], label=label, color=color, linewidth=2)  # median of distribution
        ax.stairs(perc[2, j, :], pdf_edges[j, :], baseline=perc[0, j, :], fill=True, color=color, alpha=0.2)

        if pdf_extra is not None:
            ax.stairs(perc_extra[1, j, :], pdf_extra['edges'][j, :], label=label_extra, color=color_extra)
            ax.stairs(perc_extra[2, j, :], pdf_extra['edges'][j, :], baseline=perc_extra[0, j, :], fill=True, color=color_extra, alpha=0.1)

        for ii, cc in enumerate(colors):
            ax.stairs(pdf_model_rvs[ii, j, :], pdf_edges[j, :], color=cc, linewidth=0.5)

        if 'param_labels' in popmodel:
            ax.set_xlabel(popmodel['param_labels'][j])
        else:
            ax.set_xlabel(popmodel['param_names'][j])
        if (label is not None) & (j == 0):
            ax.legend(fontsize=14)

        ax.tick_params(axis='y', which='major', labelsize=12)
        ax.tick_params(axis='x', which='major', labelsize=14)

        ax.text(0.9, 0.9, '(' + alphabet[j] + ')', fontsize=11, transform=ax.transAxes, horizontalalignment='left')

    if fname is not None:
        fig.savefig(fname, bbox_inches='tight')

    return {'pdf': pdf_model_rvs, 'edges': pdf_edges}  # allows us to plot another set of parameters on top


# limits and latex labels for the derived properties we know how to plot.
# entries with limit=None use [0, t_zmin] (times measured up to observation).
_DERIVED_LIMS = {
    'ssfr':    ([-14, -7], r"$\log_{10}({\rm sSFR/yr^{-1}})$"),
    'tform':   (None,      r'$t_{\rm form}/{\rm Gyr}$'),
    'tquench': (None,      r'$t_{\rm quench}/{\rm Gyr}$'),
    't10':     (None,      r'$t_{\rm 10}/{\rm Gyr}$'),
    't50':     (None,      r'$t_{\rm 50}/{\rm Gyr}$'),
    't90':     (None,      r'$t_{\rm 90}/{\rm Gyr}$'),
    'Av':      ([0, 4],    r'${\rm A_{V,ISM}/ mag}$'),
    'fquench': ([0, 1],    'Quenched fraction'),
}


def get_derived_props_lims(name, t_zmin=None):
    """ Limits and label for a derived property.
    t_zmin must be set for times that depend on the observation epoch. """
    if name not in _DERIVED_LIMS:
        raise KeyError(
            f"No plotting limits defined for derived property '{name}'. "
            f"Known properties: {sorted(_DERIVED_LIMS)}. "
            f"Add an entry to bagpus.plotting._DERIVED_LIMS."
        )
    limit, label = _DERIVED_LIMS[name]
    if limit is None:
        limit = [0, t_zmin]
    return limit, label


def plot_derived(theta_samples, popmodel, names=None, nbins=20, ngal=1000, title=None,
                 label=None, axes=None, pdf_extra=None, label_extra=None,
                 color_extra='blue', fname=None):
    """ Visualise the impact of the hyperparameters (theta) on derived properties
    (tform, tquench etc.). popmodel must be a PopulationModel (its derived_props
    method encapsulates the SFH/dust-specific calculations). """

    ### set up arrays
    ndraws = theta_samples.shape[0]

    fquench = np.zeros((ndraws))

    ### for each draw from the theta distribution, calculate the model parameters, and store the probability distribution function
    for i in range(ndraws):

        model_rvs = popmodel.draw_rvs(theta_samples[i, :], ngal=ngal)
        redshift_rvs = popmodel.draw_redshifts(ngal)

        derived_props = popmodel.derived_props(model_rvs, redshift_rvs)
        if names is None:
            names = list(derived_props.keys())
        nderived = len(names)

        # quenched fraction
        ind_q = np.where(derived_props['tquench'] < 99)[0]  # tquench is set to t_obs by bagpipes
        fquench[i] = len(ind_q) / len(derived_props['tquench'])

        if i == 0:
            pdf_derived_rvs = np.zeros((ndraws, nderived, nbins))
            pdf_edges = np.zeros((nderived, nbins + 1))

        derived_labels = []
        for j, name in enumerate(names):
            limit, tmp = get_derived_props_lims(name, popmodel['t_zmin'])
            derived_labels.append(tmp)

            if name == 'tquench':
                # need >10 quiescent galaxies otherwise get a lot of noise on the tquench histograms
                if len(ind_q) > 10:
                    pdf_derived_rvs[i, j, :], pdf_edges[j, :] = np.histogram(derived_props[name][ind_q], range=limit, bins=nbins, density=True)
                if i == 0:
                    print('PLEASE NOTE tquench only includes distributions with >10 quenched galaxies')
            elif name == 'fquench':
                continue  # this is not a distribution, so is dealt with differently below
            else:
                pdf_derived_rvs[i, j, :], pdf_edges[j, :] = np.histogram(derived_props[name], range=limit, bins=nbins, density=True)

    # calculate percentiles for each bin in the distribution
    perc = np.percentile(pdf_derived_rvs, [16, 50, 84], axis=0)

    if pdf_extra is not None:
        perc_extra = np.percentile(pdf_extra['pdf'], [16, 50, 84], axis=0)

    #### plot distribution of derived parameters
    if axes is None:
        ncol = 3
        nrow = int(np.ceil((nderived) / ncol))
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(2 * (ncol + 1), 2 * (nrow + 1)), layout='compressed')
        if title is not None:
            fig.suptitle(title)
    else:
        fig = axes.flat[0].figure

    # colour map for the individual draws from the prior
    color = 'orange'
    cmap = mpl.colormaps['Oranges']
    n_lines = 10
    colors = cmap(np.linspace(0, 1, n_lines))

    # panel letters continue on from the plot_params figure
    nparams = len(popmodel['param_names'])

    for j, ax in enumerate(axes.flat):
        if j >= nderived:  # remove axes where there are no more parameters
            ax.set_visible(False)
            continue
        if names[j] == 'fquench':  # plot quenched fraction distribution
            ax.hist(fquench, bins=nbins, range=[0, 1], histtype='step', density=True, color=color, linewidth=2)
            ax.set_xlabel(derived_labels[j])
            if pdf_extra is not None:
                ax.hist(pdf_extra['fquench'], bins=nbins, range=[0, 1], histtype='step', density=True, color=color_extra, linewidth=2)
        else:
            ax.stairs(perc[1, j, :], pdf_edges[j, :], label=label, color=color, linewidth=2)  # median of distribution
            ax.stairs(perc[2, j, :], pdf_edges[j, :], baseline=perc[0, j, :], fill=True, color=color, alpha=0.2)
            ax.set_xlabel(derived_labels[j])
            # random draws
            for ii, cc in enumerate(colors):
                ax.stairs(pdf_derived_rvs[ii, j, :], pdf_edges[j, :], color=cc, linewidth=0.5)

            if pdf_extra is not None:
                ax.stairs(perc_extra[1, j, :], pdf_extra['edges'][j, :], label=label_extra, color=color_extra)
                ax.stairs(perc_extra[2, j, :], pdf_extra['edges'][j, :], baseline=perc_extra[0, j, :], fill=True, color=color_extra, alpha=0.1)

        ax.tick_params(axis='y', which='major', labelsize=12)
        ax.tick_params(axis='x', which='major', labelsize=14)

        ax.text(0.9, 0.9, '(' + alphabet[j + nparams] + ')', fontsize=11, transform=ax.transAxes, horizontalalignment='left')

        if names[j] == 'tquench':
            ax.text(0.05, 0.9, 'Quenched galaxies only', fontsize=11, transform=ax.transAxes, horizontalalignment='left')
        if (label is not None) & (j == 0):
            ax.legend(fontsize=14)

    if fname is not None:
        fig.savefig(fname, bbox_inches='tight')

    return {'pdf': pdf_derived_rvs, 'edges': pdf_edges, 'fquench': fquench}


def plot_draws(pdf_2d, pdf_recon, pdf_2d_samples, popmodel, savefig=False, figname='tmp.png',
               title='Data', vmin=0.0005, vmax=0.02):
    """ Grid of SC maps: the data, its PCA reconstruction, and forward-simulated
    posterior samples. """

    pdf_range = popmodel['pdf_range']
    nsamples = pdf_2d_samples.shape[0]
    ncol = 4
    nrow = int(np.ceil((nsamples + 2) / ncol))

    fig = plt.figure(figsize=(2 * (ncol + 1), 2 * (nrow + 1)))

    gs = gridspec.GridSpec(nrow, ncol,
                           wspace=0.0, hspace=0.0,
                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

    # plot data in top left
    ax = plt.subplot(gs[0, 0])
    ax.imshow(pdf_2d.T, cmap='GnBu', norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), extent=[pdf_range[0][0], pdf_range[0][1], pdf_range[1][1], pdf_range[1][0]])
    ax.text(pdf_range[0][0] + 5, pdf_range[1][1] - 5, title, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_aspect('auto')
    ax.set_ylim((pdf_range[1][0], pdf_range[1][1]))
    ax.set_xticklabels([])
    ax.set_ylabel('SC2', fontsize=14)

    # next plot PCA reconstruction
    ax = plt.subplot(gs[0, 1])
    ax.imshow(pdf_recon.T, cmap='GnBu', norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), extent=[pdf_range[0][0], pdf_range[0][1], pdf_range[1][1], pdf_range[1][0]])
    ax.text(pdf_range[0][0] + 5, pdf_range[1][1] - 5, 'PCA reconstruction', fontsize=14)
    ax.set_aspect('auto')
    ax.set_ylim((pdf_range[1][0], pdf_range[1][1]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ii = 0
    for i in range(nrow):
        for j in range(ncol):
            if (i == 0) & (j == 0):
                continue
            if (i == 0) & (j == 1):
                continue
            if ii == nsamples:
                break

            ax = plt.subplot(gs[i, j])

            ax.imshow(pdf_2d_samples[ii, :, :].T, cmap='GnBu', norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), extent=[pdf_range[0][0], pdf_range[0][1], pdf_range[1][1], pdf_range[1][0]])
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_aspect('auto')
            ax.set_ylim((pdf_range[1][0], pdf_range[1][1]))
            plt.setp(ax.get_xticklabels()[-1], visible=False)
            ax.text(pdf_range[0][0] + 5, pdf_range[1][1] - 5, 'Posterior sample', fontsize=14)

            if j == 0:
                ax.set_ylabel('SC2', fontsize=14)
            else:
                ax.set_yticklabels([])

            if i == nrow - 1:
                ax.set_xlabel('SC1', fontsize=14)
            else:
                ax.set_xticklabels([])

            ii += 1

    if savefig == True:
        fig.savefig(figname, bbox_inches='tight')

    return gs


def plot_residuals(pdf_data, pdf_recon, pdf_2d_samples, popmodel, savefig=False,
                   figname='tmp.png', vmin=-5, vmax=4, title='Data', nsim=7000):
    """ As plot_draws but showing normalised residuals (data - model)/sigma.
    nsim must match the ngal used in the simulator. """

    ngal = len(popmodel['redshifts'])
    err_poisson = np.sqrt(pdf_data * ngal)  # this is just the data, and useful for the reconstruction plot

    pdf_range = popmodel['pdf_range']

    nsamples = pdf_2d_samples.shape[0]
    ncol = 4
    nrow = int(np.ceil((nsamples + 2) / ncol))

    fig = plt.figure(figsize=(2 * (ncol + 1), 2 * (nrow + 1)))
    width = np.zeros(ncol + 1) + 1
    width[-1] = 0.1

    gs = gridspec.GridSpec(nrow, ncol + 1, figure=fig, width_ratios=width,
                           wspace=0.0, hspace=0.0,
                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

    # plot data in top left
    ax = fig.add_subplot(gs[0, 0])

    ax.imshow(pdf_data.T, cmap='GnBu', norm=mpl.colors.LogNorm(vmin=0.0005, vmax=0.02), extent=[pdf_range[0][0], pdf_range[0][1], pdf_range[1][1], pdf_range[1][0]])
    ax.text(pdf_range[0][0] + 5, pdf_range[1][1] - 5, title, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylim((pdf_range[1][0], pdf_range[1][1]))
    ax.set_xticklabels([])
    ax.set_aspect('auto')
    ax.set_ylabel('SC2', fontsize=14)

    # next plot PCA reconstruction
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow((ngal * (pdf_data - pdf_recon) / err_poisson).T, cmap='rainbow', vmin=vmin, vmax=vmax,
              extent=[pdf_range[0][0], pdf_range[0][1], pdf_range[1][1], pdf_range[1][0]])
    ax.text(pdf_range[0][0] + 5, pdf_range[1][1] - 5, 'PCA reconstruction', fontsize=14)
    ax.set_aspect('auto')
    ax.set_ylim((pdf_range[1][0], pdf_range[1][1]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ii = 0
    for i in range(nrow):
        for j in range(ncol):
            if (i == 0) & (j == 0):
                continue
            if (i == 0) & (j == 1):
                continue
            if ii == nsamples:
                break

            ax = fig.add_subplot(gs[i, j])

            err_poisson_post = np.sqrt((pdf_data * ngal) + ((ngal / nsim)**2) * (pdf_2d_samples[ii, :, :] * nsim))
            image = ngal * (pdf_data - pdf_2d_samples[ii, :, :]) / err_poisson_post
            im = ax.imshow(image.T, cmap='rainbow', vmin=vmin, vmax=vmax,
                           extent=[pdf_range[0][0], pdf_range[0][1], pdf_range[1][1], pdf_range[1][0]])
            chisq = np.nanmean(abs(image))

            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_aspect('auto')
            ax.set_ylim((pdf_range[1][0], pdf_range[1][1]))
            plt.setp(ax.get_xticklabels()[-1], visible=False)
            ax.text(pdf_range[0][0] + 5, pdf_range[1][1] - 5, 'Posterior sample', fontsize=14)
            ax.text(pdf_range[0][0] + 5, pdf_range[1][1] - 10, r"$\chi^2_\nu=$" + str(np.round(np.median(chisq), decimals=2)), fontsize=14)

            if j == 0:
                ax.set_ylabel('SC2', fontsize=14)
            else:
                ax.set_yticklabels([])

            if i == nrow - 1:
                ax.set_xlabel('SC1', fontsize=14)
            else:
                ax.set_xticklabels([])

            ii += 1

    cbar_ax = fig.add_subplot(gs[:, ncol])
    cbar = fig.colorbar(im, cax=cbar_ax, label=r'${\rm (\rho-\rho_{m})/\sigma_{p}}$')
    cbar.ax.tick_params(labelsize=14)

    if savefig == True:
        fig.savefig(figname, bbox_inches='tight')

    return gs


def plot_stack_residuals(pdf_data, pdf_2d_samples, popmodel, savefig=False, figname='tmp.png',
                         vmin=-5, vmax=4, title=None, nsim=7000):
    """ Residuals stacked over many posterior samples, plus the chi^2 distribution.
    nsim must match the ngal used in the simulator. """

    nsamples = pdf_2d_samples.shape[0]
    ngal = len(popmodel['redshifts'])

    chisq = np.zeros(nsamples)

    for i in range(nsamples):
        err_poisson_post = np.sqrt((pdf_data * ngal) + ((ngal / nsim)**2) * (pdf_2d_samples[i, :, :] * nsim))
        hist, hist_edges = np.histogram((ngal * (pdf_data - pdf_2d_samples[i, :, :]) / err_poisson_post).flatten(), range=[-5, 5], bins=20, density=True)
        image = ngal * (pdf_data - pdf_2d_samples[i, :, :]) / err_poisson_post

        chisq[i] = np.nanmean(abs(image))
        if i == 0:
            hist_out = copy.deepcopy(hist)
            image_out = copy.deepcopy(image)
        else:
            hist_out += hist
            image_out += image

    plt.stairs(hist, hist_edges, color='black')
    plt.xlabel(r'$\chi^2_\nu$')
    plt.ylabel('N(posterior)')
    plt.show()

    print('16th, 50th, 84th percentiles of chi^2 distribution:', np.percentile(chisq, [16, 50, 84]))

    print()
    print('number with sigma<1:', len(np.where(abs(image_out / nsamples).flatten() < 5)[0]))
    print('number with sigma<5:', len(np.where(abs(image_out / nsamples).flatten() < 1)[0]))
    print('fraction within 1 sigma: ', len(np.where(abs(image_out / nsamples).flatten() < 1)[0]) / len(np.where(abs(image_out / nsamples).flatten() < 5)[0]))

    fig, ax = plt.subplots()
    plt.imshow((image_out / nsamples).T, cmap='rainbow', vmin=-3, vmax=3, extent=[popmodel['pdf_range'][0][0], popmodel['pdf_range'][0][1], popmodel['pdf_range'][1][1], popmodel['pdf_range'][1][0]])
    plt.colorbar(label=r'${\rm (\rho-\rho_{m})/\sigma_{p}}$')
    plt.ylim((popmodel['pdf_range'][1][0], popmodel['pdf_range'][1][1]))
    plt.gca().set_aspect(2.5)
    plt.ylabel('SC2', fontsize=16)
    plt.xlabel('SC1', fontsize=16)
    plt.gca().tick_params(axis='both', which='major', labelsize=16)
    if title is None:
        plt.text(0.9, 0.9, r"$\chi^2_\nu=$" + str(np.round(np.median(chisq), decimals=2)), fontsize=14, bbox=dict(facecolor='none', edgecolor='black'), transform=ax.transAxes, horizontalalignment='right')
    else:
        plt.text(0.9, 0.9, title + r" $\chi^2_\nu=$" + str(np.round(np.median(chisq), decimals=2)), fontsize=14, bbox=dict(facecolor='none', edgecolor='black'), transform=ax.transAxes, horizontalalignment='right')

    if savefig == True:
        plt.savefig(figname, bbox_inches='tight')

    return chisq


def plot_pdf(pdf, pdf_edges, xlabel, label, figname=None, pos_text=(0.9, 0.9), ha='right', data_extra=None):
    """ A single derived-property distribution with percentile band and draws. """

    perc = np.percentile(pdf, [16, 50, 84], axis=0)

    # colour map for the individual draws from the prior
    cmap = mpl.colormaps['Oranges']
    n_lines = 10
    colors = cmap(np.linspace(0, 1, n_lines))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.stairs(perc[1, :], pdf_edges, color='orange', linewidth=2)  # median of distribution
    ax.stairs(perc[2, :], pdf_edges, baseline=perc[0, :], fill=True, color='orange', alpha=0.2)

    if data_extra is not None:
        ax.hist(data_extra, range=[pdf_edges[0], pdf_edges[-1]], bins=len(pdf_edges) - 1, histtype='step', density=True, linewidth=1.5, color='black')

    # random draws
    for ii, cc in enumerate(colors):
        ax.stairs(pdf[ii, :], pdf_edges, color=cc, linewidth=0.5)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.text(pos_text[0], pos_text[1], label, fontsize=14, bbox=dict(facecolor='none', edgecolor='black'), transform=ax.transAxes, horizontalalignment=ha)
    if figname is not None:
        fig.savefig(figname, bbox_inches='tight')

    return perc
