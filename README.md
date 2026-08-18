# bagpus

**Bayesian Analysis of Galaxy PopUlations with Spectra**

Bagpus infers the *population-level* distributions of galaxy physical parameters — star formation histories, dust, metallicity — by fitting the observed distribution of an entire survey at once, rather than fitting galaxies one by one. It uses simulation-based inference (SBI) with [bagpipes](https://github.com/ACCarnall/bagpipes) as the forward model of individual galaxies, and super-colour distributions as the summary statistic.

The method and its first application (quenching timescales of massive galaxies at 1.7 < z < 2 in the UDS field) are described in the release paper ([Wild et al. in press](https://ui.adsabs.harvard.edu/abs/2026arXiv260605971W/abstract)).

## Installation

```bash
git clone https://github.com/SEDMORPH/bagpus.git
cd bagpus
pip install -e .
```

bagpipes and its stellar grids must be set up separately — see the [documentation](https://bagpus.readthedocs.io) for details, including how to switch to alternative stellar libraries (e.g. CB19) with `bagpus.grids`.

## Quick example

```python
import bagpus

# your survey: super-colours, errors, redshifts + the eigenbasis files
obs = bagpus.Observations(sc, sc_err, redshifts,
                          eigenbasis_file='UDS/VWSC_eigenbasis_0p5z3_wavemin2500.fits',
                          filter_list_file='FILTERS/vwsc_uds.lis',
                          filter_dir='FILTERS/', n_eigenvectors=3)

# the population model: which SFH/dust model, and hyperpriors on the
# population mean and SD of every parameter
pop_instructions = {
    "sfh":  {"type": "dblplaw",
             "tau":        {"limits": (0.2, 5.2), "mu": (2, 4),   "sigma": (0.5, 2)},
             "logbeta":    {"limits": (-1, 2),    "mu": (-1, 3),  "sigma": (0.5, 2)},
             "logtauhalf": {"limits": (-2, 1),    "mu": (-2, 1),  "sigma": (0.1, 1)}},
    "dust": {"type": "Calzetti",
             "eta":      {"limits": (1, 3),     "mu": (1, 2),     "sigma": (0.2, 2)},
             "logAvint": {"limits": (0.8, 2.0), "mu": (1.2, 1.5), "sigma": (0.1, 0.2)}},
    "metallicity": {"limits": (0.5, 2.5), "mu": (0.5, 2), "sigma": (0.2, 2)},
}

model = bagpus.PopulationModel(pop_instructions, obs)
fit = bagpus.Fit(model, run='my_run')

fit.simulate(nsims=5000)     # training simulations (slow; cached on disk)
fit.train()                  # PCA compression + neural posterior estimation
samples = fit.sample(1000)   # posterior over the population hyperparameters
fit.plot_corner()
fit.plot_posterior_predictive()
```

## Getting started

* [examples/1_quickstart.ipynb](examples/1_quickstart.ipynb) — the full workflow on mock data; runs in minutes, no survey catalogue needed.
* [examples/2_uds_walkthrough.ipynb](examples/2_uds_walkthrough.ipynb) — reproduces the release-paper analysis of the UDS.
* [scripts/](scripts/) — the same pipeline as command-line steps for cluster use, driven by a config file ([config/run_uds_dblplaw_tauhalf.py](config/run_uds_dblplaw_tauhalf.py)).

## Extending bagpus

Version 1 ships with the double power-law SFH (parameterised by quenching half-life) and Calzetti dust with an Av–sSFR relation, validated against the release paper. The API is designed so that any bagpipes SFH or dust model can be added by subclassing `bagpus.models.SFHModel` / `DustModel` and registering the class — see the *Adding a new model* page of the documentation.

## Note on code origins

The original code used for the release paper was written the "old-fashioned" way by Vivienne Wild. This public release version has been tidied up by Claude-AI, and made object-orientated to match *bagpipes* in spirit. The results of *bagpus* have been fully bench-marked against the original human-written code. All credit for the structure of the code goes to Adam Carnall who wrote *bagpipes* into the wonderful user-friendly package that it is. *bagpus* is more complex than *bagpipes*, just by the nature of what it is doing. We are actively working on making it more general  -- please get in contact if there are particular features that would be useful to you. 

## Citation

If you use bagpus, please cite the release paper ([Wild et al. in press](https://ui.adsabs.harvard.edu/abs/2026arXiv260605971W/abstract)) and the underlying tools: the super-colours ([Wild et al. 2014](https://ui.adsabs.harvard.edu/abs/2014MNRAS.440.1880W/abstract)), bagpipes ([Carnall et al. 2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.480.4379C/abstract)) and the sbi package ([Boelts et al. 2025](https://joss.theoj.org/papers/10.21105/joss.07754)) at a minimum. 
