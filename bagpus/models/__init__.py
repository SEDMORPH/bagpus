""" SFH and dust model registries.

To add a new model, subclass bagpus.models.base.SFHModel or DustModel and add
it to the appropriate registry here. The "type" entry of the sfh/dust section
of the population instructions selects the class by registry key.
"""

from .base import SFHModel, DustModel
from .sfh_dblplaw import DblplawSFH
from .dust_calzetti import CalzettiDust

SFH_MODELS = {
    'dblplaw': DblplawSFH,
}

DUST_MODELS = {
    'Calzetti': CalzettiDust,
}


def get_sfh_model(instructions):
    """ Resolve and instantiate the SFH spec from the "sfh" section of the
    population instructions. """
    type_name = instructions.get('type')
    if type_name not in SFH_MODELS:
        raise NotImplementedError(
            f"SFH model '{type_name}' is not implemented. Available models: "
            f"{sorted(SFH_MODELS)}. To add one, subclass bagpus.models.SFHModel "
            f"and register it in bagpus.models.SFH_MODELS — see the "
            f"'Adding a new model' page of the documentation."
        )
    return SFH_MODELS[type_name](instructions)


def get_dust_model(instructions):
    """ Resolve and instantiate the dust spec from the "dust" section of the
    population instructions. """
    type_name = instructions.get('type')
    if type_name not in DUST_MODELS:
        raise NotImplementedError(
            f"Dust model '{type_name}' is not implemented. Available models: "
            f"{sorted(DUST_MODELS)}. To add one, subclass bagpus.models.DustModel "
            f"and register it in bagpus.models.DUST_MODELS — see the "
            f"'Adding a new model' page of the documentation."
        )
    return DUST_MODELS[type_name](instructions)
