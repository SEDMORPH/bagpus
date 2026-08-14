""" The Observations class: holds the observed super-colour catalogue, the
eigenbasis used to compute super-colours from photometry, and the empirical
noise model derived from the data.

Loading and selecting the raw catalogue is dataset-specific and deliberately
kept OUT of this class — build your arrays however you like and pass them in.
See the UDS walkthrough notebook for a worked example.
"""

import numpy as np
import torch
from fast_histogram import histogram2d
from scipy.optimize import curve_fit

from . import supercolours


def _linear(x, a, b):
    return a + b * x


class Observations:
    """ Observed galaxy sample for population inference.

    Parameters
    ----------
    sc : (N, 2) array
        Super-colours (SC1, SC2) of the observed galaxies.
    sc_err : (N, 2) array
        Per-galaxy 1-sigma errors on the super-colours.
    redshifts : (N,) array
        Redshift of each galaxy. The simulator draws model redshifts from
        this array, so the model population matches the data's z distribution.
    eigenbasis_file : str
        Path to the super-colour eigenbasis FITS file.
    filter_list_file : str
        Path to the filter list (name, effective wavelength per row).
    filter_dir : str, optional
        Directory prefixed to each filter filename for bagpipes.
    n_eigenvectors : int, optional
        Truncate the eigenbasis to this many components (e.g. 3 for the UDS
        analysis). Default: use all components in the file.
    pdf_range, pdf_bins : the SC1/SC2 range and binning of the 2D histogram
        that serves as the summary statistic.
    filter_mask : array of int, optional
        Indices of filters in the filter list that are absent from the data;
        the simulator zeroes these model fluxes (e.g. [9, 10] for the unused
        HST bands in UDS).
    mag : (N,) array, optional
        A depth proxy (e.g. K-band magnitude). If given, the noise model is
        fitted on the brighter half of the sample to isolate the noise floor.
    zmin, zmax : float, optional
        Redshift limits of the sample; default to min/max of `redshifts`.
    """

    def __init__(self, sc, sc_err, redshifts,
                 eigenbasis_file, filter_list_file, filter_dir='',
                 n_eigenvectors=None,
                 pdf_range=None, pdf_bins=None,
                 filter_mask=None, mag=None,
                 zmin=None, zmax=None, verbose=True):

        self.sc = np.asarray(sc)
        self.sc_err = np.asarray(sc_err)
        self.redshifts = np.asarray(redshifts)
        self.mag = np.asarray(mag) if mag is not None else None

        if not (len(self.sc) == len(self.sc_err) == len(self.redshifts)):
            raise ValueError('sc, sc_err and redshifts must have the same length.')

        self.pdf_range = pdf_range if pdf_range is not None else [[-50, 100], [-25, 35]]
        self.pdf_bins = pdf_bins if pdf_bins is not None else [50, 55]
        self.filter_mask = np.asarray(filter_mask) if filter_mask is not None else None

        self.zmin = zmin if zmin is not None else float(np.min(self.redshifts))
        self.zmax = zmax if zmax is not None else float(np.max(self.redshifts))

        # eigenbasis for projecting model photometry onto super-colours
        self.ebasis = supercolours.read_eigensystem(
            eigenbasis_file, filter_list_file, verbose=verbose)
        if n_eigenvectors is not None:
            self.ebasis['spec'] = self.ebasis['spec'][:n_eigenvectors, :]

        # bagpipes needs full paths to the filter files
        for i in range(len(self.ebasis['filt_list'])):
            if filter_dir and not self.ebasis['filt_list'][i].startswith(filter_dir):
                self.ebasis['filt_list'][i] = filter_dir + self.ebasis['filt_list'][i]

        self.obs_errors = self._fit_error_model()

    def __len__(self):
        return len(self.redshifts)

    # ------------------------------------------------------------------
    # Empirical noise model
    # ------------------------------------------------------------------
    def _fit_error_model(self):
        """ Model the SC errors as a function of SC1: a linear noise floor
        fitted to the (bright half of the) sample, plus the per-SC1-bin mean
        and SD of the observed errors, extrapolated with the linear fit where
        the data are too sparse. """

        sc, sc_err = self.sc, self.sc_err
        nbins = self.pdf_bins[0]
        sc1_lo, sc1_hi = self.pdf_range[0]
        bin_width = (sc1_hi - sc1_lo) / nbins

        # fit the noise floor on the bright half if a depth proxy is available
        if self.mag is not None:
            ind_fit = np.where((self.mag < np.percentile(self.mag, 50))
                               & (sc_err[:, 0] < 4))[0]
        else:
            ind_fit = np.where(sc_err[:, 0] < 4)[0]

        popt_SC1err, _ = curve_fit(_linear, sc[ind_fit, 0], sc_err[ind_fit, 0])
        popt_SC2err, _ = curve_fit(_linear, sc[ind_fit, 0], sc_err[ind_fit, 1])

        obs_errors = {
            'sc1err_mean': np.zeros(nbins),
            'sc1err_sd':   np.zeros(nbins),
            'sc2err_mean': np.zeros(nbins),
            'sc2err_sd':   np.zeros(nbins),
        }

        for i in range(nbins):
            lo = sc1_lo + i * bin_width
            hi = sc1_lo + (i + 1) * bin_width
            ind_bin = np.where((sc[:, 0] > lo) & (sc[:, 0] < hi)
                               & (sc_err[:, 0] < 5))[0]
            if len(ind_bin) > 10:
                obs_errors['sc1err_mean'][i] = np.mean(sc_err[ind_bin, 0])
                obs_errors['sc1err_sd'][i] = np.std(sc_err[ind_bin, 0])
                obs_errors['sc2err_mean'][i] = np.mean(sc_err[ind_bin, 1])
                obs_errors['sc2err_sd'][i] = np.std(sc_err[ind_bin, 1])

        # extrapolate the linear fit into bins with fewer than 10 galaxies
        bin_centres = sc1_lo + (np.arange(nbins) + 0.5) * bin_width
        ind_empty = np.where(obs_errors['sc1err_mean'] == 0)[0]

        obs_errors['sc1err_mean'][ind_empty] = _linear(bin_centres[ind_empty], *popt_SC1err)
        obs_errors['sc2err_mean'][ind_empty] = _linear(bin_centres[ind_empty], *popt_SC2err)

        nonzero_sc1 = obs_errors['sc1err_sd'][obs_errors['sc1err_sd'] != 0]
        nonzero_sc2 = obs_errors['sc2err_sd'][obs_errors['sc2err_sd'] != 0]
        obs_errors['sc1err_sd'][ind_empty] = np.min(nonzero_sc1) if len(nonzero_sc1) else 0.1
        obs_errors['sc2err_sd'][ind_empty] = np.min(nonzero_sc2) if len(nonzero_sc2) else 0.1

        obs_errors['popt_SC1err'] = popt_SC1err
        obs_errors['popt_SC2err'] = popt_SC2err

        return obs_errors

    # ------------------------------------------------------------------
    # Data products
    # ------------------------------------------------------------------
    def histogram2d(self):
        """ The observed SC distribution as a normalised 2D histogram — the
        data statistic the population model is fitted to. """
        pdf = histogram2d(self.sc[:, 0], self.sc[:, 1],
                          range=self.pdf_range, bins=self.pdf_bins) / len(self)
        return torch.as_tensor(np.float32(pdf))

    def plot(self, ax=None, show=True, save=None, vmin=1, vmax=40):
        """ Plot the observed SC1-SC2 distribution. """
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.figure

        h = ax.hist2d(self.sc[:, 0], self.sc[:, 1], bins=50,
                      norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
                      cmap='GnBu', range=self.pdf_range)
        fig.colorbar(h[3], ax=ax, label='Number of galaxies')
        ax.set_xlabel('SC1')
        ax.set_ylabel('SC2')

        if save is not None:
            fig.savefig(save, dpi=400, bbox_inches='tight')
        if show:
            plt.show()
        return fig
