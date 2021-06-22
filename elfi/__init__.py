# -*- coding: utf-8 -*-
# flake8: noqa

"""Engine for Likelihood-Free Inference (ELFI) is a statistical software
package for likelihood-free inference (LFI) such as Approximate Bayesian
Computation (ABC).
"""

import elfi.clients.native

import elfi.methods.mcmc
import elfi.model.tools as tools
from elfi.client import get_client, set_client
from elfi.methods.diagnostics import TwoStageSelection
from elfi.methods.model_selection import *
from elfi.methods.inference.bolfi import *
from elfi.methods.inference.bolfire import *
from elfi.methods.inference.romc import *
from elfi.methods.inference.samplers import *
from elfi.methods.post_processing import adjust_posterior
from elfi.model.elfi_model import *
from elfi.model.extensions import ScipyLikeDistribution as Distribution
from elfi.store import OutputPool, ArrayPool
from elfi.testbench.testbench import Testbench, TestbenchMethod
from elfi.visualization.visualization import nx_draw as draw
from elfi.visualization.visualization import plot_params_vs_node
from elfi.visualization.visualization import plot_predicted_summaries
from elfi.methods.bo.gpy_regression import GPyRegression

__author__ = 'ELFI authors'
__email__ = 'elfi-support@hiit.fi'

# make sure __version_ is on the last non-empty line (read by setup.py)
__version__ = '0.8.0'
