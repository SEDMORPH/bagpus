""" Abstract base classes for SFH and dust model specifications.

A model spec translates between the three layers of bagpus:

1. the population instructions dict written by the user
   (parameter names -> {limits, mu, sigma} hyperprior specification),
2. the flat parameter arrays drawn per-galaxy by utils.draw_rvs, and
3. the component dicts that bagpipes' model_galaxy expects.

To add a new SFH or dust model: subclass SFHModel or DustModel, fill in the
class attributes and the abstract methods, and register the class in
bagpus.models.SFH_MODELS / DUST_MODELS. See sfh_dblplaw.py for a worked
example.
"""

from abc import ABC, abstractmethod

import numpy as np

from .. import utils


class ModelSpec(ABC):
    """ Shared machinery for parsing a section of the population instructions.

    Class attributes to define in subclasses
    ----------------------------------------
    type_name : str
        Registry key, and the value of "type" in the instructions section.
    prefix : str
        Prefix applied to parameter names to build unique flat names
        (e.g. 'sfh_' turns 'tau' into 'sfh_tau').
    allowed_params : list of lists
        Each inner list is an acceptable complete set of parameter names for
        this model (allows alternative parameterisations, e.g. tauhalf vs
        logtauhalf).
    default_labels : dict
        name -> (param_label, mu_label, sigma_label), latex strings used in
        plots. Users can override per-parameter with a "label" entry.
    """

    type_name = None
    prefix = ''
    allowed_params = []
    default_labels = {}

    def __init__(self, instructions):
        self.instructions = instructions

        given = [k for k in instructions if k not in ('type', 'pipes_type')]

        # find which allowed parameterisation the user supplied
        self.param_set = None
        for pset in self.allowed_params:
            if sorted(given) == sorted(pset):
                self.param_set = pset
                break
        if self.param_set is None:
            raise ValueError(
                f"{self.__class__.__name__}: parameters {sorted(given)} do not "
                f"match any accepted parameter set: {self.allowed_params}"
            )

        # flat names, limits and hyperprior ranges for PopulationModel
        self.param_names = []
        self.limits = {}
        self.hyper_mean = {}
        self.hyper_sd = {}
        self.param_labels = []
        self.prior_labels = []

        for name in self.param_set:
            spec = instructions[name]
            for key in ('limits', 'mu', 'sigma'):
                if key not in spec:
                    raise ValueError(
                        f"{self.__class__.__name__}: parameter '{name}' is missing "
                        f"'{key}'. Each parameter needs limits (truncation range), "
                        f"mu (hyperprior on the mean) and sigma (hyperprior on the SD)."
                    )

            flat = self.prefix + name
            self.param_names.append(flat)
            self.limits[flat] = tuple(spec['limits'])
            self.hyper_mean[flat] = tuple(spec['mu'])
            self.hyper_sd[flat] = tuple(spec['sigma'])

            if 'label' in spec:
                labels = spec['label']
                if isinstance(labels, str):
                    labels = (labels, r"$\mu$(" + labels + ")", r"$\sigma$(" + labels + ")")
            elif name in self.default_labels:
                labels = self.default_labels[name]
            else:
                labels = (name, r"$\mu$(" + name + ")", r"$\sigma$(" + name + ")")

            self.param_labels.append(labels[0])
            self.prior_labels.extend([labels[1], labels[2]])


class SFHModel(ModelSpec):
    """ Base class for star formation history specifications.

    Subclasses must implement population_arrays() and template_dict().
    The bagpipes component key defaults to type_name but can be overridden
    with a "pipes_type" entry in the instructions (for custom bagpipes SFH
    function names).
    """

    prefix = 'sfh_'

    def __init__(self, instructions):
        super().__init__(instructions)
        self.pipes_type = instructions.get('pipes_type', self.type_name)

    @abstractmethod
    def population_arrays(self, param_rvs):
        """ Convert drawn parameter arrays (flat names) into the per-galaxy
        arrays of bagpipes SFH keyword values, applying any reparameterisation.

        Returns a dict of arrays, e.g. {'tau': ..., 'alpha': ..., 'beta': ...}.
        """

    @abstractmethod
    def template_dict(self):
        """ A valid bagpipes SFH component dict with placeholder values, used
        for the one-off model_galaxy instantiation. Must contain the same keys
        as make_sfh_dict produces (bagpipes' update() requires an identical
        structure). """

    def make_sfh_dict(self, arrays, i, metallicity=1.0, massformed=10):
        """ The bagpipes SFH component dict for galaxy i. """
        sfh = {key: arrays[key][i] for key in arrays}
        sfh['massformed'] = massformed
        sfh['metallicity'] = metallicity
        return sfh

    def derived_props(self, param_rvs, redshifts, arrays=None):
        """ Derived SFH properties (ssfr, tform, tquench, quenching
        timescales, unphysical flag) for a population.

        param_rvs : dict of flat-name parameter arrays from draw_rvs
        redshifts : array of model redshifts (one per galaxy)
        arrays : optionally pass pre-computed population_arrays output
        """
        import bagpipes as pipes

        if arrays is None:
            arrays = self.population_arrays(param_rvs)

        ngal = len(redshifts)
        first = arrays[list(arrays.keys())[0]]
        if np.isscalar(first) or len(first) != ngal:
            raise ValueError(
                'derived_props: parameter arrays and redshifts must have the '
                'same length (one entry per model galaxy).'
            )

        output = {name: np.zeros(ngal) for name in utils.DERIVED_PROP_NAMES}
        output['unphysical'] = np.full(ngal, True)

        for i in range(ngal):
            model_comp = {
                self.pipes_type: self.make_sfh_dict(arrays, i),
                'redshift': redshifts[i],
            }
            sfh = pipes.models.star_formation_history(model_comp)
            props = utils.props_from_sfh(sfh)
            for name in props:
                output[name][i] = props[name]

        return output


class DustModel(ModelSpec):
    """ Base class for dust attenuation specifications.

    Subclasses must implement population_arrays() (which may use SFH-derived
    properties, e.g. an Av-sSFR relation), make_dust_dict() and
    template_dict().
    """

    prefix = 'dust_'

    @abstractmethod
    def population_arrays(self, param_rvs, derived):
        """ Per-galaxy arrays of derived dust quantities (e.g. Av), possibly
        coupled to SFH-derived properties in `derived` (e.g. ssfr). """

    @abstractmethod
    def make_dust_dict(self, param_rvs, arrays, i):
        """ The bagpipes dust component dict for galaxy i. """

    @abstractmethod
    def template_dict(self):
        """ A valid bagpipes dust component dict with placeholder values.
        Must have the same keys as make_dust_dict produces. """
