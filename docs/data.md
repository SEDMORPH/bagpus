# Data requirements

## Included in the repository

* `FILTERS/` — filter transmission curves and the filter list
  (`vwsc_uds.lis`: filter filename + effective wavelength per row).
* `UDS/VWSC_eigenbasis_0p5z3_wavemin2500.fits` — the super-colour eigenbasis
  used to project photometry (observed or simulated) onto SC1/SC2/SC3.

## Not included (obtain separately)

* **UDS DR11 catalogues** used in the walkthrough notebook and pipeline
  scripts: the photometry catalogue (`DR11-2arcsec-Jun-30-2019.fits`, available here https://www.nottingham.ac.uk/astronomy/UDS/DR11/) and the
  super-colour catalogue (`TDL_VWSC_0.5_3.024.5_dr11.fits` available here https://www.nottingham.ac.uk/astronomy/UDS/data/sc.html). 
* **Stellar grids** for bagpipes beyond its defaults (e.g. CB19) — see
  [Installation](installation).

## Using your own survey

bagpus needs, per galaxy:

| Quantity | Shape | Notes |
|---|---|---|
| super-colours | (N, 2+) | SC1, SC2 (+ higher components, unused) |
| SC errors | (N, 2) | 1-sigma; drives the empirical noise model |
| redshift | (N,) | the simulator matches this distribution |
| depth proxy (optional) | (N,) | e.g. K magnitude; isolates the noise floor |

plus an eigenbasis + filter list consistent with how your super-colours were
computed. Construct a {class}`bagpus.Observations` from these arrays — the
catalogue selection code is yours (see the dataset-specific block at the top
of `scripts/step1_prepare.py` for the UDS example).

If your survey uses different photometric bands, you need an eigenbasis
computed for those bands (see
[Wild et al. 2014](https://ui.adsabs.harvard.edu/abs/2014MNRAS.440.1880W/abstract)
for the super-colour method, code to do this yourself is in prep. please ask if you are interested).
