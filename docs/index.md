# bagpus

**Bayesian Analysis of Galaxy PopUlations with Spectra**

bagpus infers the *population-level* distributions of galaxy physical
parameters — star formation histories, dust, metallicity — by fitting the
observed distribution of an entire survey at once, using simulation-based
inference with [bagpipes](https://github.com/ACCarnall/bagpipes) as the
forward model of individual galaxies and super-colour distributions as the
summary statistic.

Instead of asking "what is the SFH of this galaxy?", bagpus asks "what
distribution of SFHs across the population reproduces the observed
colour distribution of the survey?" — including its shape, bimodality
and outliers. This sidesteps the biases of stacking individual noisy
posteriors, and gives direct access to population questions such as the
distribution of quenching timescales and the quenched fraction as a
function of cosmic time.



## Example output

Applied to $1.7<z<2.0$ massive galaxies in the UKIDSS UDS field (the
release-paper analysis, reproduced in the
[UDS walkthrough](examples/2_uds_walkthrough)):

```{figure} _static/13c_data_posterior.png
:width: 600px
:align: center

The observed super-colour distribution (top left) and its PCA
reconstruction (top second from left), compared with six draws from the population
posterior predictive — the fitted population reproduces the observed
colour bimodality, including the post-starburst branch.
```

```{figure} _static/13c_tauquench_full_paper2.png
:width: 450px
:align: center

The inferred distribution of quenching timescales (time from peak star
formation to quenching) for quenched galaxies in the population posterior —
individual thin lines are draws from the posterior, the thick orange line is the
median and the coloured region the 16th-84th percentiles.
```

## How it works

1. **Population model** — every galaxy-level parameter (SFH shape, dust,
   metallicity) is drawn from a truncated Gaussian; the population mean and SD
   of each are the inferred hyperparameters ({class}`bagpus.PopulationModel`).
2. **Forward simulation** — for a draw of the hyperparameters, a mock survey of
   thousands of bagpipes galaxies is generated, projected onto super-colours,
   perturbed with the survey's empirical noise model, and binned into a 2D
   distribution ({mod}`bagpus.simulator`).
3. **Inference** — the simulated distributions are compressed with PCA and a
   neural density estimator (masked autoregressive flow, via the
   [sbi](https://sbi-dev.github.io/sbi/) package) learns the posterior over the
   hyperparameters; applying it to the observed distribution gives the
   population posterior ({class}`bagpus.Fit`).

### Note on code origins

The original code used for the release paper was written the "old-fashioned" way by Vivienne Wild. This Public release version has been tidied up by Claude-AI, and made object-orientated to match bagpipes in spirit. The results of bagpus have been fully bench-marked against the original human-written code. All credit for the structure of the code goes to Adam Carnall who wrote bagpipes into the wonderful user-friendly package that it is. bagpus is more complex than bagpipes, just by the nature of what it is doing. We are actively working on making it more general -- please get in contact if there are particular features that would be useful to you. 

```{toctree}
:maxdepth: 1
:caption: Getting started

self
installation
data
examples/1_quickstart
examples/2_uds_walkthrough
```

```{toctree}
:maxdepth: 1
:caption: Reference

adding_models
api
```
