""" The PopulationModel class: parses the population instructions dict,
resolves the SFH and dust model specs, and provides the prior, per-galaxy
sampling and derived properties.

The instructions dict mirrors bagpipes' fit_instructions in spirit:

    pop_instructions = {
        "sfh": {
            "type": "dblplaw",
            "tau":        {"limits": (0.2, 5.2), "mu": (2, 4),   "sigma": (0.5, 2)},
            "logbeta":    {"limits": (-1, 2),    "mu": (-1, 3),  "sigma": (0.5, 2)},
            "logtauhalf": {"limits": (-2, 1),    "mu": (-2, 1),  "sigma": (0.1, 1)},
        },
        "dust": {
            "type": "Calzetti",
            "eta":      {"limits": (1, 3),     "mu": (1, 2),     "sigma": (0.2, 2)},
            "logAvint": {"limits": (0.8, 2.0), "mu": (1.2, 1.5), "sigma": (0.1, 0.2)},
        },
        "metallicity": {"limits": (0.5, 2.5), "mu": (0.5, 2), "sigma": (0.2, 2)},
        "nebular": {"logU": -3},   # fixed
        "massformed": 10,          # fixed
    }

Every varying parameter is drawn per-galaxy from a truncated Gaussian with
truncation range "limits"; the population mean and SD of that Gaussian are the
inferred hyperparameters, with flat priors over "mu" and "sigma". Giving
"metallicity" as a scalar fixes it instead of inferring its distribution.
"""

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from . import utils
from .models import get_sfh_model, get_dust_model

_METALLICITY_LABELS = (r"$Z^*/Z_\odot$", r"$\mu_{Z^*}$", r"$\sigma_{Z^*}$")


class PopulationModel:
    """ A population model tied to a set of observations.

    Parameters
    ----------
    pop_instructions : dict
        The population model specification (see module docstring).
    obs : bagpus.Observations
        The observed sample; provides redshifts, the noise model, the
        eigenbasis and the SC histogram settings.
    cosmology : astropy cosmology, optional
        Defaults to FlatLambdaCDM(H0=70, Om0=0.3).
    """

    def __init__(self, pop_instructions, obs, cosmology=None):
        self.instructions = pop_instructions
        self.obs = obs
        self.cosmo = cosmology if cosmology is not None else FlatLambdaCDM(H0=70., Om0=0.3)

        for section in ('sfh', 'dust'):
            if section not in pop_instructions:
                raise ValueError(f"pop_instructions must contain a '{section}' section.")

        self.sfh_spec = get_sfh_model(pop_instructions['sfh'])
        self.dust_spec = get_dust_model(pop_instructions['dust'])

        # ------------------------------------------------------------------
        # assemble the flat parameter table consumed by draw_rvs / def_prior
        # ------------------------------------------------------------------
        popdict = {}
        param_names = []
        param_labels = []
        prior_labels = []

        for spec in (self.sfh_spec, self.dust_spec):
            param_names.extend(spec.param_names)
            param_labels.extend(spec.param_labels)
            prior_labels.extend(spec.prior_labels)
            for flat in spec.param_names:
                popdict[flat] = spec.limits[flat]
                popdict[flat + ':mean'] = spec.hyper_mean[flat]
                popdict[flat + ':sd'] = spec.hyper_sd[flat]

        # metallicity: free parameter (dict) or fixed value (scalar)
        met = pop_instructions.get('metallicity', 1.0)
        if isinstance(met, dict):
            param_names.append('Zmet')
            param_labels.append(met.get('label', _METALLICITY_LABELS[0]))
            prior_labels.extend(_METALLICITY_LABELS[1:])
            popdict['Zmet'] = tuple(met['limits'])
            popdict['Zmet:mean'] = tuple(met['mu'])
            popdict['Zmet:sd'] = tuple(met['sigma'])
            self.fixed_metallicity = None
        else:
            self.fixed_metallicity = float(met)

        popdict['param_names'] = param_names
        popdict['param_labels'] = param_labels
        popdict['prior_labels'] = prior_labels

        # fixed model settings
        popdict['massformed'] = pop_instructions.get('massformed', 10)
        popdict['nebular'] = pop_instructions.get('nebular', {'logU': -3})
        popdict['pipes_model'] = self.sfh_spec.pipes_type

        # observation-linked settings
        popdict['redshifts'] = obs.redshifts
        popdict['obs_errors'] = obs.obs_errors
        popdict['ebasis'] = obs.ebasis
        popdict['pdf_range'] = obs.pdf_range
        popdict['pdf_bins'] = obs.pdf_bins
        popdict['filter_mask'] = obs.filter_mask
        popdict['zmin'] = obs.zmin
        popdict['zmax'] = obs.zmax
        popdict['t_zmin'] = self.cosmo.age(obs.zmin).value
        popdict['t_zmax'] = self.cosmo.age(obs.zmax).value

        self.popdict = popdict

    # ------------------------------------------------------------------
    # dict-style access so generic utilities (draw_rvs, def_prior, the
    # simulator and the plotting functions) can treat this like the popmodel
    # dict they have always consumed
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        return self.popdict[key]

    def __contains__(self, key):
        return key in self.popdict

    def get(self, key, default=None):
        return self.popdict.get(key, default)

    @property
    def param_names(self):
        return self.popdict['param_names']

    @property
    def param_labels(self):
        return self.popdict['param_labels']

    @property
    def prior_labels(self):
        return self.popdict['prior_labels']

    @property
    def n_hyper(self):
        return 2 * len(self.param_names)

    # ------------------------------------------------------------------
    # sampling and priors
    # ------------------------------------------------------------------
    def prior(self, return_limits=False):
        """ The SBI prior (BoxUniform over the population hyperparameters). """
        return utils.def_prior(self.popdict, return_limits=return_limits)

    def draw_rvs(self, theta, ngal=1000):
        """ Draw per-galaxy parameters for one hyperparameter vector theta. """
        param_rvs = utils.draw_rvs(theta, self.popdict, ngal=ngal)
        if self.fixed_metallicity is not None:
            param_rvs['Zmet'] = np.full(ngal, self.fixed_metallicity)
        return param_rvs

    def draw_redshifts(self, ngal):
        """ Draw model redshifts matching the observed redshift distribution. """
        ind = np.random.randint(low=0, high=len(self.obs.redshifts), size=ngal)
        return self.obs.redshifts[ind]

    def derived_props(self, param_rvs, redshifts):
        """ Derived properties (ssfr, tform, tquench, quenching timescales,
        Av, unphysical flag) for a population drawn with draw_rvs. """
        derived = self.sfh_spec.derived_props(param_rvs, redshifts)
        derived.update(self.dust_spec.population_arrays(param_rvs, derived))
        return derived
