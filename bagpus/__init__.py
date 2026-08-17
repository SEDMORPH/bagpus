""" bagpus: Bayesian Analysis of Galaxy PopUlations with Spectra.

Hierarchical population inference for galaxy surveys using simulation-based
inference, with bagpipes as the forward model of individual galaxies and
super-colour distributions as the summary statistic.

Typical workflow:

    import bagpus

    obs = bagpus.Observations(sc, sc_err, redshifts, eigenbasis_file=..., ...)
    model = bagpus.PopulationModel(pop_instructions, obs)
    fit = bagpus.Fit(model, run='my_run')

    fit.simulate(nsims=5000)
    fit.train()
    samples = fit.sample(1000)
    fit.plot_corner()
"""

from .observations import Observations
from .popmodel import PopulationModel
from .fit import Fit

from . import models
from . import plotting
from . import simulator
from . import supercolours
from . import utils

__version__ = '0.1.0'


def __getattr__(name):
    # bagpus.grids imports bagpipes at module level, which is slow — load it
    # lazily on first access so plain `import bagpus` stays fast
    if name == 'grids':
        import importlib
        return importlib.import_module('.grids', __name__)
    raise AttributeError(f"module 'bagpus' has no attribute '{name}'")
