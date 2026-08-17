""" The Fit class: drives the full inference workflow for a PopulationModel.

Typical use:

    import bagpus

    obs = bagpus.Observations(sc, sc_err, redshifts, ...)
    model = bagpus.PopulationModel(pop_instructions, obs)
    fit = bagpus.Fit(model, run='my_run')

    fit.simulate(nsims=5000)      # generate training data (slow)
    fit.train()                   # PCA + neural posterior estimation
    samples = fit.sample(1000)    # posterior for the observed data
    fit.plot_corner()
    fit.plot_posterior_predictive()

All heavy products are cached on disk under runs/<run>/ and reloaded on
subsequent calls, so the workflow can be interrupted and resumed, and repeated
calls in a notebook are cheap.
"""

import glob
import os

import numpy as np
import torch

from . import plotting
from . import utils
from .simulator import parallel_simulate_SC


class Fit:
    """ Fit a PopulationModel to its Observations with SBI.

    Parameters
    ----------
    popmodel : bagpus.PopulationModel
    run : str
        Name of this run; all outputs are stored under working_dir/<subdir>/<run>/.
    working_dir : str
        Root directory for outputs (default 'runs/').
    ngal_sims : int
        Number of galaxies per simulated population.
    ncores : int
        Parallel workers for simulation.
    """

    def __init__(self, popmodel, run, working_dir='runs/', ngal_sims=7000, ncores=8):
        self.popmodel = popmodel
        self.run = run
        self.ngal_sims = ngal_sims
        self.ncores = ncores

        (self.dir_figs, self.dir_training,
         self.dir_test, self.dir_posterior) = utils.make_run_dirs(run, working_dir=working_dir)

        self.prior, self.lower_bound, self.upper_bound = popmodel.prior(return_limits=True)

        # lazily loaded products
        self._meanarr = None
        self._pca = None
        self._pca_floor = None
        self._posterior = None
        self._samples = None

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def simulate(self, nsims=500, batch=1, ncores=None, force=False):
        """ Generate a batch of training simulations (skipped if the batch
        already exists on disk unless force=True). Returns (theta, x). """

        theta_path = self.dir_training + f'theta_{batch}.pt'
        x_path = self.dir_training + f'x_{batch}.pt'

        if os.path.exists(theta_path) and not force:
            print(f'Training batch {batch} already exists — loading. Use force=True to regenerate.')
            return (torch.load(theta_path, weights_only=False),
                    torch.load(x_path, weights_only=False))

        print(f'Simulating training batch {batch} ({nsims} simulations, '
              f'{self.ngal_sims} galaxies each)...')
        theta = self.prior.sample((nsims,))
        x = parallel_simulate_SC(theta, self.popmodel,
                                 num_workers=ncores or self.ncores,
                                 ngal=self.ngal_sims)

        torch.save(theta, theta_path)
        torch.save(x, x_path)
        torch.save(self.popmodel, self.dir_training + 'model.pt')
        return theta, x

    def simulate_test(self, nsims=500, ncores=None, force=False):
        """ Generate the held-out test set used for mock recovery checks. """

        theta_path = self.dir_test + 'theta.pt'
        x_path = self.dir_test + 'x.pt'

        if os.path.exists(theta_path) and not force:
            print('Test set already exists — loading. Use force=True to regenerate.')
            return (torch.load(theta_path, weights_only=False),
                    torch.load(x_path, weights_only=False))

        print(f'Simulating test set ({nsims} simulations)...')
        theta = self.prior.sample((nsims,))
        x = parallel_simulate_SC(theta, self.popmodel,
                                 num_workers=ncores or self.ncores,
                                 ngal=self.ngal_sims)
        torch.save(theta, theta_path)
        torch.save(x, x_path)
        return theta, x

    def load_training(self):
        """ Load and concatenate all training batches on disk. """
        nfiles = len(glob.glob(self.dir_training + 'x_*.pt'))
        if nfiles == 0:
            raise FileNotFoundError(
                f'No training simulations found in {self.dir_training} — '
                f'run fit.simulate() first.')

        thetas, xs = [], []
        for i in range(nfiles):
            tp = self.dir_training + f'theta_{i + 1}.pt'
            xp = self.dir_training + f'x_{i + 1}.pt'
            if os.path.exists(tp) and os.path.exists(xp):
                thetas.append(torch.load(tp, weights_only=False))
                xs.append(torch.load(xp, weights_only=False))

        theta = torch.cat(thetas, dim=0)
        x = torch.cat(xs, dim=0)
        print(f'Loaded {theta.shape[0]} training simulations from {len(thetas)} batch(es).')
        return theta, x

    # ------------------------------------------------------------------
    # Training: PCA compression + neural posterior estimation
    # ------------------------------------------------------------------
    def train(self, n_pca_components=20, pca_floor=0.0001, density_estimator='maf',
              force=False, diagnostic_plot=False):
        """ Compress the training simulations with PCA and train the neural
        posterior estimator. Both products are cached on disk.

        pca_floor replaces empty histogram cells before taking the log for the
        PCA compression (avoids log 0). The default suits populations of a few
        thousand galaxies on a ~50x50 grid; if your cell occupancies are very
        different it may need adjusting. The value is stored with the PCA and
        automatically reused when the observed data are projected, so the
        summary statistics stay consistent. """

        posterior_path = self.dir_posterior + 'posterior.pt'
        if os.path.exists(posterior_path) and not force:
            print('Trained posterior already exists — loading. Use force=True to retrain.')
            self._load_pca()
            self._posterior = torch.load(posterior_path, weights_only=False)
            return self._posterior

        theta, x = self.load_training()

        meanarr, pca, x_r = utils.func_compress_pca(
            np.array(x), self.popmodel['pdf_range'],
            n_components=n_pca_components, floor=pca_floor,
            diagnostic_plot=diagnostic_plot)
        torch.save(meanarr, self.dir_training + 'pca_meanarr.pt')
        torch.save(pca, self.dir_training + 'pca.pt')
        torch.save(pca_floor, self.dir_training + 'pca_floor.pt')
        self._meanarr, self._pca, self._pca_floor = meanarr, pca, pca_floor

        from sbi.neural_nets import posterior_nn
        from sbi.inference import NPE

        print('Training neural posterior estimator...')
        neural_posterior = posterior_nn(model=density_estimator)
        inference = NPE(prior=self.prior, density_estimator=neural_posterior)

        x_r_tensor = torch.as_tensor(np.float32(x_r))
        posterior_net = inference.append_simulations(theta, x_r_tensor).train()
        self._posterior = inference.build_posterior(posterior_net)

        torch.save(self._posterior, posterior_path)
        print(f'Saved trained posterior to {posterior_path}')
        return self._posterior

    def _load_pca(self):
        if self._pca is None:
            self._meanarr = torch.load(self.dir_training + 'pca_meanarr.pt', weights_only=False)
            self._pca = torch.load(self.dir_training + 'pca.pt', weights_only=False)
            floor_path = self.dir_training + 'pca_floor.pt'
            if os.path.exists(floor_path):
                self._pca_floor = torch.load(floor_path, weights_only=False)
            else:
                self._pca_floor = 0.0001  # runs trained before the floor was stored
        return self._meanarr, self._pca

    @property
    def posterior(self):
        if self._posterior is None:
            path = self.dir_posterior + 'posterior.pt'
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f'{path} not found — run fit.train() first.')
            self._posterior = torch.load(path, weights_only=False)
        return self._posterior

    # ------------------------------------------------------------------
    # Inference on the observed data
    # ------------------------------------------------------------------
    def data_pca(self):
        """ The observed SC histogram projected onto the PCA basis, using the
        same floor as the training compression.
        Returns (pdf_2d, pdf_recon, pca_amps). """
        meanarr, pca = self._load_pca()
        pdf_2d = self.popmodel.obs.histogram2d()
        pdf_recon, pca_amps = utils.func_project_pca(pdf_2d, meanarr, pca,
                                                     floor=self._pca_floor)
        return pdf_2d, pdf_recon, pca_amps

    def sample(self, n=1000, force=False):
        """ Draw posterior samples of the population hyperparameters given the
        observed data. Cached on disk. """

        samples_path = self.dir_posterior + 'samples_data.pt'
        if os.path.exists(samples_path) and not force:
            self._samples = torch.load(samples_path, weights_only=False)
            return self._samples

        _, _, pca_amps = self.data_pca()
        print(f'Sampling {n} posterior draws...')
        self._samples = self.posterior.set_default_x(pca_amps).sample((n,))
        torch.save(self._samples, samples_path)
        return self._samples

    def sample_mock(self, x_mock, n=1000):
        """ Posterior samples for a mock 2D SC histogram (e.g. from the test
        set) — used for recovery checks. Not cached. """
        meanarr, pca = self._load_pca()
        _, pca_amps = utils.func_project_pca(torch.as_tensor(np.float32(x_mock)),
                                             meanarr, pca, floor=self._pca_floor)
        return self.posterior.set_default_x(pca_amps).sample((n,))

    @property
    def samples(self):
        if self._samples is None:
            return self.sample()
        return self._samples

    def posterior_predictive(self, n=100, force=False):
        """ Forward-simulate SC maps for n posterior draws (cached). """

        path = self.dir_posterior + 'x_samples_data.pt'
        if os.path.exists(path) and not force:
            return torch.load(path, weights_only=False)

        samples = self.samples
        print(f'Forward-simulating {n} posterior draws...')
        x_samples = parallel_simulate_SC(samples[:n, :], self.popmodel,
                                         num_workers=self.ncores, ngal=self.ngal_sims)
        torch.save(x_samples, path)
        return x_samples

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot_corner(self, samples=None, save=None, figsize=(16, 16)):
        """ Pairplot of the posterior samples over the population hyperparameters. """
        from sbi.analysis import pairplot

        if samples is None:
            samples = self.samples

        limits = torch.as_tensor(np.stack([np.array(self.lower_bound),
                                           np.array(self.upper_bound)], axis=1))
        fig, axes = pairplot(samples, limits=limits,
                             labels=self.popmodel['prior_labels'], figsize=figsize)
        for ax in np.array(axes).reshape(-1):
            ax.tick_params(axis='both', which='major', labelsize=12)
        if save is not None:
            fig.savefig(save)
        return fig

    def plot_params(self, samples=None, prior_reference=True, nbins=20, ngal=1000, save=None):
        """ Distributions of the galaxy-level parameters implied by the
        posterior, optionally with the prior overlaid. """
        pdf_prior = None
        if prior_reference:
            pdf_prior = plotting.plot_params(self.prior.sample((1000,)), self.popmodel,
                                             nbins=nbins, ngal=ngal)
        if samples is None:
            samples = self.samples
        return plotting.plot_params(samples, self.popmodel, nbins=nbins, ngal=ngal,
                                    pdf_extra=pdf_prior, label='Posterior',
                                    label_extra='Prior' if prior_reference else None,
                                    fname=save)

    def plot_derived(self, samples=None, names=('ssfr', 'tform', 'tquench', 'Av', 'fquench'),
                     prior_reference=True, nbins=20, ngal=1000, save=None):
        """ Distributions of derived properties implied by the posterior. """
        pdf_prior = None
        if prior_reference:
            pdf_prior = plotting.plot_derived(self.prior.sample((1000,)), self.popmodel,
                                              list(names), nbins=nbins, ngal=ngal)
        if samples is None:
            samples = self.samples
        return plotting.plot_derived(samples, self.popmodel, list(names), nbins=nbins,
                                     ngal=ngal, pdf_extra=pdf_prior, fname=save)

    def plot_posterior_predictive(self, n_show=6, vmin=0.0001, vmax=0.02, save=None):
        """ The observed SC map next to its PCA reconstruction and
        forward-simulated posterior draws. """
        pdf_2d, pdf_recon, _ = self.data_pca()
        x_samples = self.posterior_predictive()
        return plotting.plot_draws(pdf_2d, pdf_recon, x_samples[:n_show, :, :],
                                   self.popmodel, title='Data', vmin=vmin, vmax=vmax,
                                   savefig=save is not None, figname=save or 'tmp.png')

    def plot_residuals(self, n_show=6, vmin=-4, vmax=4, save=None):
        """ Poisson-normalised residual maps (n_obs - n_exp)/sqrt(n_exp) for
        individual posterior draws, with expected counts from the PCA
        reconstruction of each draw. """
        meanarr, pca = self._load_pca()
        pdf_2d, pdf_recon, _ = self.data_pca()
        x_samples = self.posterior_predictive()
        return plotting.plot_residuals(pdf_2d, pdf_recon, x_samples[:n_show, :, :],
                                       self.popmodel, meanarr, pca,
                                       ngal=len(self.popmodel.obs),
                                       floor=self._pca_floor,
                                       vmin=vmin, vmax=vmax,
                                       savefig=save is not None, figname=save or 'tmp.png')

    def plot_stacked_residuals(self, vmin=-3, vmax=3, save=None):
        """ Poisson-normalised residuals stacked over all forward-simulated
        posterior draws; returns the chi^2_nu of each draw. """
        meanarr, pca = self._load_pca()
        pdf_2d, _, _ = self.data_pca()
        x_samples = self.posterior_predictive()
        return plotting.plot_stack_residuals(pdf_2d, x_samples, self.popmodel,
                                             meanarr, pca,
                                             ngal=len(self.popmodel.obs),
                                             floor=self._pca_floor,
                                             vmin=vmin, vmax=vmax,
                                             savefig=save is not None, figname=save or 'tmp.png')

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    def summary_table(self, latex=False):
        """ Percentiles of the posterior on each population hyperparameter. """
        from astropy.table import Table

        samples = np.array(self.samples)
        tbl = Table()
        tbl['Name'] = self.popmodel['prior_labels']
        tbl['16th'] = np.percentile(samples, 16, axis=0)
        tbl['median'] = np.percentile(samples, 50, axis=0)
        tbl['84th'] = np.percentile(samples, 84, axis=0)

        if latex:
            from astropy.io import ascii
            ascii.write(tbl, format='latex',
                        formats={'median': '%0.2f', '16th': '%0.2f', '84th': '%0.2f'})
        return tbl

    def quenching_statistics(self, omega_deg2=None, zbins=None, ndraws=400,
                             ngal=None, nbins=20):
        """ Quenched fractions, quenching timescale distributions and (if the
        survey solid angle is given) number densities, computed from the
        posterior samples.

        Returns a dict with distributions ('pdf_*' + '*_edges' pairs),
        per-draw summary arrays and, when omega_deg2 is set, 'fquench_zbin' /
        'nquench_zbin' for each requested lookback redshift bin.
        """
        cosmo = self.popmodel.cosmo
        samples = self.samples
        ndraws = min(ndraws, samples.shape[0])
        if ngal is None:
            ngal = len(self.popmodel['redshifts'])

        V = None
        if omega_deg2 is not None:
            Omega = omega_deg2 / 3282.8  # deg^2 -> steradian
            d_lo = cosmo.comoving_distance(self.popmodel['zmin'])
            d_hi = cosmo.comoving_distance(self.popmodel['zmax'])
            V = (Omega / 3 * (d_hi**3 - d_lo**3)).value
        zbins = list(zbins) if zbins is not None else []

        out = {
            'fquench': np.zeros(ndraws),
            'med_tauquench_init': np.zeros(ndraws),
            'med_tauquench_full': np.zeros(ndraws),
            'ssfr_max': np.zeros(ndraws),
            'pdf_tauquench_init': np.zeros((ndraws, nbins)),
            'pdf_tauquench_full': np.zeros((ndraws, nbins)),
            'pdf_tform_quenched': np.zeros((ndraws, nbins)),
            'pdf_ssfr_quenched': np.zeros((ndraws, nbins)),
            'fquench_zbin': np.zeros((ndraws, len(zbins))),
            'nquench_zbin': np.zeros((ndraws, len(zbins))),
            'zbins': zbins,
            'volume': V,
        }

        for i in range(ndraws):
            model_rvs = self.popmodel.draw_rvs(samples[i, :], ngal=ngal)
            redshift_rvs = self.popmodel.draw_redshifts(ngal)
            props = self.popmodel.derived_props(model_rvs, redshift_rvs)

            ind_q = np.where(props['tquench'] < 99)[0]
            out['fquench'][i] = len(ind_q) / ngal
            if len(ind_q) == 0:
                continue

            out['pdf_tform_quenched'][i, :], out['tform_edges'] = np.histogram(
                props['tform'][ind_q], range=[0, self.popmodel['t_zmin']], bins=nbins, density=True)
            out['pdf_ssfr_quenched'][i, :], out['ssfr_edges'] = np.histogram(
                props['ssfr'][ind_q], range=[-15, -8], bins=nbins, density=True)
            out['ssfr_max'][i] = np.max(props['ssfr'][ind_q])

            out['pdf_tauquench_init'][i, :], out['tauquench_init_edges'] = np.histogram(
                props['tauquench_init'][ind_q] * 1e3, range=[0, 1000], bins=nbins, density=True)
            out['med_tauquench_init'][i] = np.median(props['tauquench_init'][ind_q]) * 1e3

            out['pdf_tauquench_full'][i, :], out['tauquench_full_edges'] = np.histogram(
                props['tauquench_full'][ind_q] * 1e3, range=[0, 2000], bins=nbins, density=True)
            out['med_tauquench_full'][i] = np.median(props['tauquench_full'][ind_q]) * 1e3

            for j, z in enumerate(zbins):
                t_z = cosmo.age(z).value
                ind = np.where(props['tquench'] < t_z)[0]
                out['fquench_zbin'][i, j] = len(ind) / ngal
                if V is not None:
                    out['nquench_zbin'][i, j] = 1e5 * len(ind) / V

        return out
