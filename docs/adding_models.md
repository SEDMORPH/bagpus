# Adding a new SFH or dust model

bagpus v1 ships with the double power-law SFH (`dblplaw`) and Calzetti dust law,
validated against the release paper. The code is designed so that any SFH or
dust model available in bagpipes can be added with a small subclass — but
**think carefully about the population-level implications first** (see the
caveats below). 

**This part of the documentation has been generated entirely by AI, 
and has not yet been human-verified, so please use with extra caution.** 

## A new SFH model

Subclass {class}`bagpus.models.SFHModel` and register it. Using bagpipes'
exponential (`exponential`) model as an example:

```python
import numpy as np
from bagpus.models import SFH_MODELS
from bagpus.models.base import SFHModel

class ExponentialSFH(SFHModel):
    type_name = 'exponential'

    # each inner list is one accepted set of parameter names for the
    # "sfh" section of pop_instructions
    allowed_params = [['age', 'logtau']]

    default_labels = {
        'age':    (r"$t_{\rm age}/{\rm Gyr}$", r"$\mu_{t}$",        r"$\sigma_{t}$"),
        'logtau': (r"$\log_{10}\tau$",         r"$\mu_{\log\tau}$", r"$\sigma_{\log\tau}$"),
    }

    def population_arrays(self, param_rvs):
        # translate the drawn (flat-named) parameters into the keyword
        # values bagpipes expects, applying any reparameterisation
        return {'age': param_rvs['sfh_age'],
                'tau': 10**param_rvs['sfh_logtau']}

    def template_dict(self):
        # placeholder dict with the SAME keys make_sfh_dict will produce
        # (bagpipes' update() requires an identical structure)
        return {'age': 1.0, 'tau': 1.0, 'massformed': 10, 'metallicity': 1}

SFH_MODELS['exponential'] = ExponentialSFH
```

That is all: `make_sfh_dict` and `derived_props` (ssfr, tform, tquench, the
quenching timescales, and the `unphysical` flag) are inherited — they work
numerically from the bagpipes SFH object, for any parameterisation.

Then in the population instructions:

```python
pop_instructions["sfh"] = {
    "type": "exponential",
    "age":    {"limits": (0.1, 5.0), "mu": (0.5, 4.0), "sigma": (0.2, 2.0)},
    "logtau": {"limits": (-1, 1),    "mu": (-1, 1),    "sigma": (0.1, 1.0)},
}
```

## A new dust model

Subclass {class}`bagpus.models.DustModel`; implement `population_arrays`
(any derived per-galaxy quantities — this is where an Av–sSFR-style coupling
to the SFH lives), `make_dust_dict` and `template_dict`, and register in
`bagpus.models.DUST_MODELS`.

## Caveats — read before adding an SFH model

* **The hyperpriors are on YOUR parameterisation.** A truncated Gaussian
  population in `logtau` is a different population model from one in `tau`.
  The choice of parameterisation *is* part of the model, and the release paper
  shows the results can be sensitive to it (linear vs log prior on the
  quenching timescale). Choose parameters whose population distributions are
  plausibly unimodal and roughly Gaussian.
* **Derived properties depend on the observation epoch.** tform/tquench run up
  to the age of the Universe at each galaxy's redshift; the inherited
  `derived_props` handles this, but interpret population distributions
  accordingly.
* **Reparameterisations must be vectorised** in `population_arrays` — it is
  called once per simulated population, not per galaxy.
* **Validate before science.** Run a mock recovery across prior space (see the
  quickstart notebook) before trusting posteriors from a new model.
