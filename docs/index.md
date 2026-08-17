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

## Note on code origins

The original code used for the release paper was written the "old-fashioned" way by Vivienne Wild. This Public release version has been tidied up by Claude-AI, and made object-orientated to match [bagpipes] in spirit. The results of [bagpus] have been fully bench-marked against the original human-written code. All credit for the structure of the code goes to Adam Carnall who wrote [bagpipes] into the wonderful user-friendly package that it is. [bagpus] is more complex than [bagpipes], just by the nature of what it is doing. We are actively working on making it more general. 

```{toctree}
:maxdepth: 1
:caption: Getting started

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
