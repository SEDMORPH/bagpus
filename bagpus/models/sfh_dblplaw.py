""" Double power-law SFH, parameterised by the peak time (tau), the rising
slope (beta, in log10) and the quenching half-life (tauhalf, linear or log10).

This is the SFH model used in the bagpus release paper. The bagpipes dblplaw
function takes (tau, alpha, beta); here the falling slope alpha is replaced by
tau_half — the time for the SFR to fall to half its peak value — which is far
more interpretable as a quenching timescale. The conversion is analytic
(func_tauhalf_to_alpha).
"""

import numpy as np

from .base import SFHModel


def tauhalf_func(tauhalf, alpha, beta, tau):
    return (1 + tauhalf / tau)**alpha + (1 + tauhalf / tau)**-beta - 4


def alpha_to_halflife_general(alpha, beta, tau):
    """use scipy root finding to solve for the conversion from alpha and beta and tau to tau half"""
    from scipy.optimize import elementwise
    args = (alpha, beta, tau)
    res_bracket = elementwise.bracket_root(tauhalf_func, 0, xmin=0, args=args)
    res_root = elementwise.find_root(tauhalf_func, res_bracket.bracket, args=args)
    if np.any(res_root.status):
        raise Exception(f"Root finder failed in {np.sum(res_root.status)} of the alpha to halflife calculations.")
    return res_root.x


def func_tauhalf_to_alpha(tauhalf, tau, beta):
    """ Convert tauhalf to alpha in a double powerlaw """
    alpha = np.log10(4 - (1 + tauhalf / tau)**(-beta)) / np.log10(1 + tauhalf / tau)
    return alpha


class DblplawSFH(SFHModel):
    """ Double power-law SFH with the falling slope expressed as a half-life.

    Accepted parameter sets (keys of the "sfh" instructions section):
      tau, logbeta, logtauhalf   — tau_half with a log10 prior (paper default)
      tau, logbeta, tauhalf      — tau_half with a linear prior
    """

    type_name = 'dblplaw'

    allowed_params = [
        ['tau', 'logbeta', 'logtauhalf'],
        ['tau', 'logbeta', 'tauhalf'],
    ]

    default_labels = {
        'tau':        (r"$t_p/{\rm Gyr}$",
                       r"$\mu_{t_p}$", r"$\sigma_{t_p}$"),
        'logbeta':    (r"$\log_{10}\beta$",
                       r"$\mu_{\log\beta}$", r"$\sigma_{\log\beta}$"),
        'logtauhalf': (r"$\log_{10}\gamma$",
                       r"$\mu_{\log\gamma}$", r"$\sigma_{\log\gamma}$"),
        'tauhalf':    (r"$\gamma/{\rm Gyr}$",
                       r"$\mu_{\gamma}$", r"$\sigma_{\gamma}$"),
    }

    def population_arrays(self, param_rvs):
        tau = param_rvs['sfh_tau']
        beta = 10**param_rvs['sfh_logbeta']

        if 'sfh_logtauhalf' in param_rvs:
            tauhalf = 10**param_rvs['sfh_logtauhalf']
        else:
            tauhalf = param_rvs['sfh_tauhalf']

        alpha = func_tauhalf_to_alpha(tauhalf, tau, beta)

        return {'tau': tau, 'alpha': alpha, 'beta': beta}

    def template_dict(self):
        return {
            'tau': 2,        # ~ peak of double powerlaw in Gyr from big bang
            'alpha': 10,     # falling slope, high is quickly falling
            'beta': 10,      # rising slope, high is quickly rising
            'massformed': 10,
            'metallicity': 1,
        }
