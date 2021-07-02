"""This module contains the CopulaABC methods."""


import logging
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from scipy.stats import norm


import elfi.client
import elfi.methods.mcmc as mcmc
import elfi.visualization.interactive as visin
import elfi.visualization.visualization as vis
from elfi.loader import get_sub_seed
from elfi.methods.bo.acquisition import LCBSC
from elfi.methods.bo.gpy_regression import GPyRegression
from elfi.methods.bo.utils import stochastic_optimization
from elfi.methods.posteriors import BolfiPosterior
from elfi.methods.results import BolfiSample, OptimizationResult, Sample, SmcSample
from elfi.methods.utils import (GMDistribution, ModelPrior, arr2d_to_batch,
                                batch_to_arr2d, ceil_to_batch_size, weighted_var)
from elfi.model.elfi_model import ComputationContext, ElfiModel, NodeReference
from elfi.methods.parameter_inference import Rejection
from elfi.utils import is_array

logger = logging.getLogger(__name__)







class Copula_ABC(Rejection):
    """Copula ABC rejection sampler.

    For a description of the rejection sampler and a general introduction to ABC, see e.g.
    J. Li et al. Compuational Statistics and Data Analysis. 2017.

    """

    def __init__(self, model, discrepancy_name=None, output_names=None, **kwargs):
        """Initialize the Rejection sampler.

        Parameters
        ----------
        model : ElfiModel or NodeReference
        discrepancy_name : str, NodeReference, optional
            Only needed if model is an ElfiModel
        output_names : list, optional
            Additional outputs from the model to be included in the inference result, e.g.
            corresponding summaries to the acquired samples
        kwargs:
            See InferenceMethod

        """
        super(Copula_ABC, self).__init__(model, output_names, **kwargs)



    def update(self, batch, batch_index):
        """Update the inference state with a new batch.

        Parameters
        ----------
        batch : dict
            dict with `self.outputs` as keys and the corresponding outputs for the batch
            as values
        batch_index : int

        """
        # super(Copula_ABC, self).update(batch, batch_index)
        self.state['n_batches'] += 1
        self.state['n_sim'] += self.batch_size
        if self.state['samples'] is None:
            # Lazy initialization of the outputs dict
            self._init_samples_lazy(batch)
        self._merge_batch(batch)

    def infer(self, *args, vis=None, **kwargs):
        """Set the objective and start the iterate loop until the inference is finished.

        See the other arguments from the `set_objective` method.

        Returns
        -------
        result : Sample

        """
        vis_opt = vis if isinstance(vis, dict) else {}

        self.set_objective(*args, **kwargs)

        while not self.finished:
            self.iterate()
            if vis:
                self.plot_state(interactive=True, **vis_opt)

        self.batches.cancel_pending()

        self.copula_cal()

        if vis:
            self.plot_state(close=True, **vis_opt)

        return self.extract_result()

    def extract_result(self):
        """Extract the result from the current state.

        Returns
        -------
        result : Sample

        """
        if self.state['samples'] is None:
            raise ValueError('Nothing to extract')

        # Take out the correct number of samples
        outputs = dict()
        outputs['Lambda'] = self.Lambda
        outputs['Marg'] = self.Marg

        return CopulaABC_Sample(outputs=outputs, **self._extract_result_kwargs())


    def copula_cal(self):
        """Based on the generated samples, calculate the Lambda correlation matrix and the Marg distribution.

        Returns
        -------
        Lambda : the bi-variate correlation matrix
        Marg: the Marginal distribution

        """

        PP = self.state['samples'][self.discrepancy_name].shape[1]

        Lambda = np.ones((PP, PP))
        Marg = np.empty((self.objective['n_samples'], PP))

        quantile_val = self.objective['n_samples']/self.state['n_sim']*100
        for K1 in range(PP-1):
            for K2 in np.arange(K1+1, PP):
                dis_K1K2 = np.sqrt(np.sum(self.state['samples'][self.discrepancy_name][:, [K1, K2]]**2, axis=1))
                indexx = (dis_K1K2<=np.percentile(dis_K1K2, quantile_val))

                val1 = self.state['samples'][self.parameter_names[0]][indexx, [[K1], [K2]]].T

                ri = ss.rankdata(val1[:, 0])
                rj = ss.rankdata(val1[:, 1])

                nn = sum(indexx)
                # nn = len(indexx)
                etaii = norm.ppf(ri/(nn+1))
                etajj = norm.ppf(rj/(nn+1))

                Lambda[K1, K2] = np.corrcoef(etaii, etajj)[0, 1]
                Lambda[K2, K1] = Lambda[K1, K2]

        for kk in range(PP):
            dis_KK = self.state['samples'][self.discrepancy_name][:, kk]
            indexx = (dis_KK <= np.percentile(dis_KK, quantile_val))

            val_adjust_ii = self.state['samples'][self.parameter_names[0]][indexx, kk]

            Marg[:, kk] = val_adjust_ii.reshape((-1))

        self.Lambda = Lambda
        self.Marg = Marg

    def _init_samples_lazy(self, batch):
        """Initialize the outputs dict based on the received batch."""
        samples = {}
        e_noarr = "Node {} output must be in a numpy array of length {} (batch_size)."
        e_len = "Node {} output has array length {}. It should be equal to the batch size {}."

        for node in self.output_names:
            # Check the requested outputs
            if node not in batch:
                raise KeyError("Did not receive outputs for node {}".format(node))

            nbatch = batch[node]
            if not is_array(nbatch):
                raise ValueError(e_noarr.format(node, self.batch_size))
            elif len(nbatch) != self.batch_size:
                raise ValueError(e_len.format(node, len(nbatch), self.batch_size))

            # Prepare samples
            shape = (self.objective['n_samples'] + self.batch_size, ) + nbatch.shape[1:]
            dtype = nbatch.dtype

            samples[node] = np.array([]).reshape(0, nbatch.shape[1])

        self.state['samples'] = samples

    def _merge_batch(self, batch):
        # TODO: add index vector so that you can recover the original order
        samples = self.state['samples']
        # Put the acquired samples to the end
        for node, v in samples.items():
            self.state['samples'][node] = np.vstack((v, batch[node]))
            # v[self.objective['n_samples']:] = batch[node]
