""" The forward model: simulate the 2D super-colour distribution of a galaxy
population given a set of population hyperparameters.

The physics (SFH parameterisation, dust law) is injected through the model
specs attached to the PopulationModel — nothing in this module is specific to
a particular SFH or dust model.
"""

import numpy as np
import torch
from fast_histogram import histogram2d
from joblib import Parallel, delayed

from . import supercolours as SC
from . import utils


def simulator_SC(theta, popmodel, return_props=False, ngal=7000):
    """ Simulate the population SC image for hyperparameters theta.

    theta : 1D tensor/array of hyperparameters, or 2D (nsim, nhyper)
    popmodel : PopulationModel (or dict-like with sfh_spec/dust_spec attached)
    ngal : number of galaxies per simulated population
    return_props : also return the per-galaxy model properties (last sim only)
    """
    import bagpipes as pipes

    rng = np.random.default_rng()

    sfh_spec = popmodel.sfh_spec
    dust_spec = popmodel.dust_spec
    massformed = popmodel['massformed']
    filter_mask = popmodel['filter_mask']

    if theta.ndim == 1:
        nsim = 1
        param_rvs = utils.draw_rvs(theta, popmodel, ngal=ngal)
    else:
        nsim = theta.shape[0]
        pdf_2d = np.zeros([nsim, popmodel['pdf_bins'][0], popmodel['pdf_bins'][1]])
        if return_props == True:
            print('SIMULATOR WARNING you have nsim>1 with return_props = True: I will only return properties of final simulation.')

    # extract the observational errors on the parameters for convenience below
    obs_errors = popmodel['obs_errors']

    errbinwidth = (popmodel['pdf_range'][0][1] - popmodel['pdf_range'][0][0]) / popmodel['pdf_bins'][0]
    errbincentres = popmodel['pdf_range'][0][0] + errbinwidth / 2 + np.arange(popmodel['pdf_bins'][0] + 1) * errbinwidth

    # initiate bagpipes - this is "slow" so use the update method below to run
    # large numbers of mocks. The component structure must be identical to the
    # dicts used inside the loop; parameter values don't matter here.
    galmodel_comp = {}
    galmodel_comp['redshift'] = popmodel['zmax']
    galmodel_comp['dust'] = dust_spec.template_dict()
    galmodel_comp['nebular'] = dict(popmodel['nebular'])
    galmodel_comp[sfh_spec.pipes_type] = sfh_spec.template_dict()

    model_mujy = pipes.model_galaxy(galmodel_comp, filt_list=popmodel['ebasis']['filt_list'], phot_units="mujy")

    for jj in range(nsim):

        if nsim != 1:
            param_rvs = utils.draw_rvs(theta[jj, :], popmodel, ngal=ngal)

        # apply the model-specific reparameterisation once, vectorised
        sfh_arrays = sfh_spec.population_arrays(param_rvs)

        # set up output data structure
        output = {}
        output['params'] = param_rvs
        ncomps = popmodel['ebasis']['spec'].shape[0]
        output['SC'] = np.zeros((ngal, ncomps))
        output['SCERR'] = np.zeros((ngal, 2))

        # set redshift rvs of models to match redshift distribution of data
        ind_data = np.random.randint(low=0, high=len(popmodel['redshifts']), size=ngal)
        output['ind_data'] = ind_data
        output['redshift'] = popmodel['redshifts'][ind_data]

        # derived SFH properties (ssfr, quenching times, unphysical flag)
        derived = sfh_spec.derived_props(param_rvs, output['redshift'], arrays=sfh_arrays)
        for key in derived:
            output[key] = derived[key]

        # dust quantities, possibly coupled to the SFH (e.g. Av-sSFR relation)
        dust_arrays = dust_spec.population_arrays(param_rvs, derived)
        for key in dust_arrays:
            output[key] = dust_arrays[key]

        # now generate SCs for all the galaxies in our simulation from the rvs
        for i in range(ngal):

            galmodel_comp['redshift'] = output['redshift'][i]

            metallicity = param_rvs['Zmet'][i] if 'Zmet' in param_rvs else 1.0
            galmodel_comp[sfh_spec.pipes_type] = sfh_spec.make_sfh_dict(
                sfh_arrays, i, metallicity=metallicity, massformed=massformed)

            # something is wrong, skip and the histogram (density) will account
            # for the lost simulation. Always worth investigating why.....
            if derived['unphysical'][i] == True:
                print("ERROR IN SFH!!", galmodel_comp)
                continue

            galmodel_comp['dust'] = dust_spec.make_dust_dict(param_rvs, dust_arrays, i)

            model_mujy.update(galmodel_comp)

            flux = model_mujy.photometry.copy()
            if filter_mask is not None:
                flux[filter_mask] = 0  # filters present in filt_list but not used in the data

            # put the fluxes into the super-sampled eigenbasis array
            flux_super = SC.vwsc_fillflux(flux, galmodel_comp['redshift'], popmodel['ebasis'])

            # we don't use the output errors, so these are just for the code.
            # make sure to give each flux the same weight in the projection
            flux_super_err = np.zeros(len(flux_super))
            flux_super_err[flux_super > 0] = np.mean(0.1 * flux)

            pcs, _ = SC.normgappy(flux_super / 10**5, flux_super_err / 10**5, popmodel['ebasis'])

            output['SC'][i, :] = pcs

            # add errors to SCs based on SC1 to match dataset; scale is the SD.
            # don't worry about models out of range, these don't count anyway
            if (pcs[0] > popmodel['pdf_range'][0][0]) & (pcs[0] < popmodel['pdf_range'][0][1]) \
               & (pcs[1] > popmodel['pdf_range'][1][0]) & (pcs[1] < popmodel['pdf_range'][1][1]):
                ind = np.argmin(abs(errbincentres - pcs[0]))

                output['SCERR'][i, 0] = rng.normal(loc=obs_errors['sc1err_mean'][ind], scale=obs_errors['sc1err_sd'][ind])
                output['SCERR'][i, 1] = rng.normal(loc=obs_errors['sc2err_mean'][ind], scale=obs_errors['sc2err_sd'][ind])

                # error floor from the minimum observed error at each SC1
                if output['SCERR'][i, 0] < obs_errors['popt_SC1err'][0] + pcs[0] * obs_errors['popt_SC1err'][1]:
                    output['SCERR'][i, 0] = obs_errors['popt_SC1err'][0] + pcs[0] * obs_errors['popt_SC1err'][1]
                if output['SCERR'][i, 1] < obs_errors['popt_SC2err'][0] + pcs[0] * obs_errors['popt_SC2err'][1]:
                    output['SCERR'][i, 1] = obs_errors['popt_SC2err'][0] + pcs[0] * obs_errors['popt_SC2err'][1]

                # final check, just in case a rogue simulation has gone below the limit
                if output['SCERR'][i, 0] < 0:
                    output['SCERR'][i, 0] = 0.001
                if output['SCERR'][i, 1] < 0:
                    output['SCERR'][i, 1] = 0.001

                output['SC'][i, 0] += rng.normal(loc=0, scale=output['SCERR'][i, 0])
                output['SC'][i, 1] += rng.normal(loc=0, scale=output['SCERR'][i, 1])

        # turn into PDF (image): histogram2d is faster version of hist2d
        ind = np.where(derived['unphysical'] == False)[0]
        if nsim == 1:
            pdf_2d = histogram2d(output['SC'][ind, 0], output['SC'][ind, 1], range=popmodel['pdf_range'], bins=popmodel['pdf_bins']) / len(ind)
        else:
            pdf_2d[jj, :, :] = histogram2d(output['SC'][ind, 0], output['SC'][ind, 1], range=popmodel['pdf_range'], bins=popmodel['pdf_bins']) / len(ind)

    # turn into format the SBI code likes
    pdf_2d = torch.from_numpy(np.array(np.float32(pdf_2d)))

    if return_props == True:
        return pdf_2d, output

    return pdf_2d


def parallel_simulate_SC(theta, pop_model, num_workers=8, **kwargs):
    """ Run simulator_SC for a batch of theta values in parallel. """

    simulation_outputs = Parallel(n_jobs=num_workers)(
        delayed(simulator_SC)(batch, pop_model, **kwargs)
        for batch in theta
    )
    return torch.from_numpy(np.array(simulation_outputs))
