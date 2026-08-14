"""
step5_analyse.py — Population parameter and derived-property analysis:
prior/posterior comparisons, quenching timescales, number densities, tables.

Usage
-----
    python scripts/step5_analyse.py --config config/run_uds_dblplaw_tauhalf.py
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np

from common import load_config, build_fit

import bagpus

cfg, args = load_config()

fit = build_fit(cfg)

# ---------------------------------------------------------------------------
# Prior vs posterior distributions (parameters and derived properties)
# ---------------------------------------------------------------------------
prior_params = bagpus.plotting.plot_params(
    fit.prior.sample((1000,)), fit.popmodel, nbins=20, ngal=1000,
    fname=fit.dir_figs + 'prior_params.pdf')
prior_derived = bagpus.plotting.plot_derived(
    fit.prior.sample((1000,)), fit.popmodel,
    ['ssfr', 'tform', 'tquench', 'Av', 'fquench'], nbins=20, ngal=1000,
    fname=fit.dir_figs + 'prior_derived.pdf')

post_params = bagpus.plotting.plot_params(
    fit.samples, fit.popmodel, nbins=20, ngal=1000,
    pdf_extra=prior_params, label='Posterior', label_extra='Prior',
    fname=fit.dir_figs + 'posterior_params.pdf')
post_derived = bagpus.plotting.plot_derived(
    fit.samples, fit.popmodel,
    ['ssfr', 'tform', 'tquench', 'Av', 'fquench'], nbins=20, ngal=1000,
    pdf_extra=prior_derived,
    fname=fit.dir_figs + 'posterior_derived.pdf')

print('Prior quenched fraction [16,50,84]:',
      np.percentile(prior_derived['fquench'], [16, 50, 84]))
print('Posterior quenched fraction [16,50,84]:',
      np.percentile(post_derived['fquench'], [16, 50, 84]))

# ---------------------------------------------------------------------------
# Hyperparameter table (latex to stdout)
# ---------------------------------------------------------------------------
fit.summary_table(latex=True)

# ---------------------------------------------------------------------------
# Quenching timescales, formation times and number densities
# ---------------------------------------------------------------------------
stats = fit.quenching_statistics(
    omega_deg2=cfg.omega_survey_deg2, zbins=cfg.zbin_comparison, ndraws=400)

print(f"\nComoving volume: {stats['volume']:.4e} Mpc^3")

for label, arr in [('tau_q,full median (Myr)', stats['med_tauquench_full']),
                   ('tau_q,init median (Myr)', stats['med_tauquench_init'])]:
    p = np.percentile(arr, [16, 50, 84])
    print(f'{label}: {p[1]:.0f} +{p[2]-p[1]:.0f} -{p[1]-p[0]:.0f}')

for j, z in enumerate(stats['zbins']):
    pf = np.percentile(stats['fquench_zbin'][:, j], [16, 50, 84])
    pn = np.percentile(stats['nquench_zbin'][:, j], [16, 50, 84])
    print(f'z > {z}: fquench = {pf[1]:.3f} +{pf[2]-pf[1]:.3f} -{pf[1]-pf[0]:.3f}; '
          f'n_q = {pn[1]:.2f} +{pn[2]-pn[1]:.2f} -{pn[1]-pn[0]:.2f} (1e-5 Mpc^-3)')

bagpus.plotting.plot_pdf(
    stats['pdf_tauquench_init'], stats['tauquench_init_edges'],
    r"$\tau_{q,\,init}({\rm t_{peak}\rightarrow t_{0.5peak}})$/Myr",
    'Quenched galaxies only', fit.dir_figs + 'tauquench_init.pdf')
bagpus.plotting.plot_pdf(
    stats['pdf_tauquench_full'], stats['tauquench_full_edges'],
    r"$\tau_{q,\,full}({\rm t_{peak}\rightarrow t_{quench}})$/Myr",
    'Quenched galaxies only', fit.dir_figs + 'tauquench_full.pdf')
bagpus.plotting.plot_pdf(
    stats['pdf_tform_quenched'], stats['tform_edges'],
    r"$t_{\rm form}$/Gyr", 'Quenched\n galaxies\n only',
    fit.dir_figs + 'tform_quenched.pdf', pos_text=(0.1, 0.7), ha='left')
bagpus.plotting.plot_pdf(
    stats['pdf_ssfr_quenched'], stats['ssfr_edges'],
    r"$\log_{10}({\rm sSFR/yr^{-1}})$", 'Quenched galaxies only',
    fit.dir_figs + 'ssfr_quenched.pdf', pos_text=(0.1, 0.9), ha='left')

print(f'\nFigures saved to {fit.dir_figs}')
