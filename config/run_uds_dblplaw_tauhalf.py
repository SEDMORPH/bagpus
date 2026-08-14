# Configuration for: UDS DR11, double power-law SFH, tauhalf parameterisation.
# This reproduces the bagpus release paper run.
#
# To adapt for a different dataset:
#   1. Update the paths in the "Paths" section.
#   2. Adjust the sample selection cuts (used by scripts/step1_prepare.py,
#      whose catalogue-loading block is dataset-specific).
#   3. Review pop_instructions — the sfh tau limits depend on t_zmax, which is
#      computed automatically from zmin/zmax.
#   4. Set a new runID so outputs don't collide with previous runs.
#   5. Update omega_survey_deg2 if your survey footprint differs.

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
dir_project = '/Users/vw8/Research/Projects/bagpus_claude/'
dir_data = dir_project + 'UDS/'
dir_filters = dir_project + 'FILTERS/'
dir_stellar_grids = dir_project + 'Bagpipes_grids/'

photo_file = 'DR11-2arcsec-Jun-30-2019.fits'
sc_file = 'TDL_VWSC_0.5_3.024.5_dr11.fits'
eigenbasis_file = 'VWSC_eigenbasis_0p5z3_wavemin2500.fits'
filter_list_file = dir_filters + 'vwsc_uds.lis'

# Stellar population library (see bagpus.grids for available options)
stellar_grid = 'cb19'

# Root directory for all pipeline outputs
output_dir = 'runs/'

# Run identifier — controls subdirectory names inside output_dir
runID = '1.7_2.0_hyp2'

# ---------------------------------------------------------------------------
# UDS sample selection
# ---------------------------------------------------------------------------
zmin = 1.7      # minimum redshift of galaxies you want to fit
zmax = 2.0      # maximum redshift of galaxies you want to fit
klim = 24.0     # K-band limit of galaxies you want to include
masslim = 10.3  # log10(M*/Msun), stellar mass lower limit

# Number of eigenvectors to use from the UDS eigenbasis file
# (UDS data was fitted with 3 components)
n_eigenvectors = 3

# Filters present in the filter list but not used in the data (zeroed in the
# simulator) — for UDS these are the two HST bands
filter_mask = [9, 10]

# Derived (do not edit — computed from zmin/zmax above)
t_zmin = cosmo.age(zmin).value  # cosmic time at zmin, Gyr
t_zmax = cosmo.age(zmax).value  # cosmic time at zmax, Gyr

# ---------------------------------------------------------------------------
# Super-colour grid
# ---------------------------------------------------------------------------
pdf_bins = [50, 55]                  # number of SC1-SC2 bins
pdf_range = [[-50, 100], [-25, 35]]  # min max SC1 and SC2

# ---------------------------------------------------------------------------
# Survey geometry (for number density calculations in step5)
# ---------------------------------------------------------------------------
omega_survey_deg2 = 0.60

# Redshift bins for fossil-record number densities (median z of comparison samples)
zbin_comparison = [2.33, 3.49, 4.32]

# ---------------------------------------------------------------------------
# The population model
#
# Every varying parameter is drawn per-galaxy from a truncated Gaussian:
#   limits — truncation range for individual galaxies
#   mu     — flat hyperprior on the population mean (inferred)
#   sigma  — flat hyperprior on the population SD (inferred)
# ---------------------------------------------------------------------------
pop_instructions = {
    "sfh": {
        "type": "dblplaw",
        # t_p: time of SFH peak (Gyr). Upper limit 1.5*t_zmax so all galaxies
        # share the same prior regardless of z (bagpipes cuts off at 1.5*t_obs)
        "tau":        {"limits": (0.2, 1.5 * t_zmax), "mu": (2.0, 4.0), "sigma": (0.5, 2.0)},
        # log10(beta): rising slope
        "logbeta":    {"limits": (-1.0, 2.0), "mu": (-1.0, 3.0), "sigma": (0.5, 2.0)},
        # log10(tau_half/Gyr): quenching timescale (log prior)
        "logtauhalf": {"limits": (-2.0, 1.0), "mu": (-2.0, 1.0), "sigma": (0.1, 1.0)},
    },
    "dust": {
        "type": "Calzetti",
        # epsilon: birth-cloud attenuation multiplier
        "eta":      {"limits": (1.0, 3.0), "mu": (1.0, 2.0), "sigma": (0.2, 2.0)},
        # c_dust: intercept of the Av-sSFR relation (log10)
        "logAvint": {"limits": (0.8, 2.0), "mu": (1.2, 1.5), "sigma": (0.1, 0.2)},
    },
    # Z*/Zsun
    "metallicity": {"limits": (0.5, 2.5), "mu": (0.5, 2.0), "sigma": (0.2, 2.0)},
}

# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------
ngal_sims = 7000   # galaxies per simulated population
nsims_train = 5000 # prior draws per training batch; use --batch in step2 for more batches
nsims_test = 500   # prior draws for test set
ncores = 8         # parallel workers for simulation

# ---------------------------------------------------------------------------
# SBI / training settings
# ---------------------------------------------------------------------------
n_pca_components = 20
neural_posterior_model = 'maf'
n_posterior_samples = 1000
n_posterior_predictive = 100
