""" Calzetti dust attenuation with an Av-sSFR relation.

The attenuation Av of each galaxy is not a free parameter: it follows the
empirical relation log10(Av) = logAvint + 0.13 * log10(sSFR) found from
bagpipes fits to the UDS data, with logAvint (the intercept) and eta (the
extra attenuation towards birth clouds) drawn from the population model.
"""

import numpy as np

from .base import DustModel


def get_AV(dust_logAvint, ssfr):
    """ The Av-sSFR relation: more strongly star-forming galaxies are dustier. """
    logAv = dust_logAvint + 0.13 * ssfr
    Av = 10**logAv
    return Av


class CalzettiDust(DustModel):
    """ Calzetti attenuation curve; population parameters are the Av-sSFR
    intercept (logAvint) and the birth-cloud attenuation multiplier (eta).

    Required parameters in the "dust" instructions section: eta, logAvint.
    """

    type_name = 'Calzetti'

    allowed_params = [
        ['eta', 'logAvint'],
    ]

    default_labels = {
        'eta':      (r"$\epsilon$",
                     r"$\mu_\epsilon$", r"$\sigma_\epsilon$"),
        'logAvint': (r"$c_{\rm dust}$",
                     r"$\mu_{c_{\rm dust}}$", r"$\sigma_{c_{\rm dust}}$"),
    }

    def population_arrays(self, param_rvs, derived):
        return {'Av': get_AV(param_rvs['dust_logAvint'], derived['ssfr'])}

    def make_dust_dict(self, param_rvs, arrays, i):
        return {
            'type': 'Calzetti',
            'Av': arrays['Av'][i],
            'eta': param_rvs['dust_eta'][i],
        }

    def template_dict(self):
        return {'type': 'Calzetti', 'Av': 1, 'eta': 1.0}
