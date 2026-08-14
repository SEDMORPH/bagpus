"""
step4_infer.py — Sample the posterior for the observed data and make the
posterior predictive diagnostic figures.

Usage
-----
    python scripts/step4_infer.py --config config/run_uds_dblplaw_tauhalf.py
"""

import matplotlib
matplotlib.use('Agg')

from common import load_config, set_stellar_grid, build_fit

cfg, args = load_config()
set_stellar_grid(cfg)  # posterior predictive simulations need the right grid

fit = build_fit(cfg)

samples = fit.sample(n=cfg.n_posterior_samples)
print(f'Posterior samples: {tuple(samples.shape)}')

fit.posterior_predictive(n=cfg.n_posterior_predictive)

fit.plot_corner(save=fit.dir_figs + 'corners_data.pdf')
fit.plot_posterior_predictive(save=fit.dir_figs + 'posterior_draws.pdf')
fit.plot_residuals(save=fit.dir_figs + 'posterior_residuals.pdf')
fit.plot_stacked_residuals(save=fit.dir_figs + 'stacked_residuals.pdf')
print(f'Figures saved to {fit.dir_figs}')
