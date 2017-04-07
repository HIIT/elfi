from functools import partial
import copy
import uuid
import numpy as np
import inspect
import re

from elfi.utils import scipy_from_str, observed_name
from elfi.store import OutputPool
from elfi.fn_wrappers import rvs_wrapper, discrepancy_wrapper
from elfi.graphical_model import GraphicalModel
import elfi.client

__all__ = ['ElfiModel', 'ComputationContext', 'Constant', 'Prior', 'Simulator', 'Summary',
           'Discrepancy', 'get_current_model', 'reset_current_model']


_current_model = None


def get_current_model():
    global _current_model
    if _current_model is None:
        _current_model = ElfiModel()
    return _current_model


def reset_current_model(model=None):
    global _current_model
    if model is None:
        model = ElfiModel()
    if not isinstance(model, ElfiModel):
        raise ValueError('{} is not an instance of ElfiModel'.format(ElfiModel))
    _current_model = model


def random_name(length=6):
    return str(uuid.uuid4().hex[0:length])


class ComputationContext:
    def __init__(self, seed=None, batch_size=None, observed=None, output_supply=None):
        """

        Parameters
        ----------
        seed : int, False, None (default)
            - When None, generates a random integer seed.
            - When False, numpy's global random_state will be used in all computations.
              Used for testing.
        batch_size : int
        observed : dict
        output_supply : dict

        """

        # Extract the seed from numpy RandomState. Alternative would be to use
        # os.urandom(4) casted as int.
        self.seed = seed if (seed is not None) \
                    else np.random.RandomState().get_state()[1][0]
        self.batch_size = batch_size or 1
        self.observed = observed or {}
        self._pool = None

    @property
    def pool(self):
        return self._pool

    @pool.setter
    def pool(self, pool):
        if pool is not None:
            pool.init_context(self)
        self._pool = pool

    def callback(self, batch, batch_index):
        if self.pool:
            self.pool.add_batch(batch_index, batch)

    def copy(self):
        return copy.copy(self)


class ElfiModel(GraphicalModel):
    def __init__(self, name=None, source_net=None, parameters=None,
                 computation_context=None):
        self.name = name or "model_{}".format(random_name())
        self.parameters = parameters or []
        self.computation_context = computation_context or ComputationContext()
        super(ElfiModel, self).__init__(source_net)

    def generate(self, batch_size=1, outputs=None, with_values=None):
        """Generates a batch using the global seed. Useful for testing.

        Parameters
        ----------
        batch_size : int
        outputs : list
        with_values : dict

        """

        if outputs is None:
            outputs = self.source_net.nodes()
        elif isinstance(outputs, str):
            outputs = [outputs]
        if not isinstance(outputs, list):
            raise ValueError('Outputs must be a list of node names')

        context = self.computation_context.copy()
        # Use the global random_state
        context.seed = False
        context.batch_size = batch_size
        if with_values is not None:
            pool = OutputPool(with_values.keys())
            pool.add_batch(0, with_values)
            context.pool = pool

        client = elfi.client.get()
        compiled_net = client.compile(self.source_net, outputs)
        loaded_net = client.load_data(compiled_net, context, batch_index=0)
        return client.compute(loaded_net)

    def get_reference(self, name):
        cls = self.get_node(name)['class']
        return cls.reference(name, self)

    @property
    def observed(self):
        return self.computation_context.observed

    def __copy__(self):
        model_copy = super(ElfiModel, self).__copy__()
        model_copy.computation_context = self.computation_context.copy()
        model_copy.parameters = list(self.parameters)
        return model_copy

    def __getitem__(self, node_name):
        return self.get_reference(node_name)


class NodeReference:
    """This is a base class for reference objects to nodes that a user of Elfi will
    typically use, e.g. `elfi.Prior` or `elfi.Simulator`. Each node has a state that
    describes how the node ultimately produces its output. The state is located in the
    ElfiModel so that serializing the model is straightforward. NodeReference and it's
    subclasses are convenience classes that makes it easy to manipulate the state and only
    contain a reference to the corresponding state in the ElfiModel.

    Currently NodeReference objects have two responsibilities:

    1. Provide convenience methods for manipulating and creating the state dictionaries of
       different types of nodes, e.g. creating a simulator node with
       `elfi.Simulator(fn, arg1, ...).
    2. Provide a compiler function that turns a state dictionary into to an Elfi
       callable output function in the computation graph. The output function will receive
       the values of its parents as arguments. The edge names correspond to argument
       names. Integers are interpreted as positional arguments. See computation graph
       for more information.

    The state of a node is a Python dictionary. It describes the type of the node and
    any other relevant state information, such as a user provided function in the case of
    elfi.Simulator.

    There are a few reserved keywords for the state dict that serve as flags for the Elfi
    compiler for specific purposes. Currently these are:

    - stochastic
        Indicates that the node is stochastic. Elfi will provide a random_state argument
        for such nodes, which contains a RandomState object for drawing random quantities.
        This node will appear in the computation graph. Using Elfi provided random states
        makes it possible to have repeatable experiments in Elfi.
    - observable
        Indicates that the true value of the node is observable. Elfi will create a copy
        of the node to the computation graph. When the user provides the observed value
        that will be added as its output. Note that if the parent observed values are
        defined, the child will be able to compute its value automatically.
    - uses_batch_size
        The node requires batch_size as input. A corresponding edge will be added to the
        computation graph.
    - uses_observed
        The node requires the observed data of its parents as input. Elfi will gather
        the observed values of its parents to a tuple and link them to the node as a named
        argument observed.
    """

    def __init__(self, *parents, state=None, model=None, name=None):
        """

        Parameters
        ----------
        parents : variable
        name : string
        state : dict
        model : ElfiModel
        """
        state = state or {}
        state["class"] = self.__class__
        model = model or get_current_model()

        name = name or self._give_name(model)
        model.add_node(name, state)

        self._init_reference(name, model)
        self._add_parents(parents)

    def _add_parents(self, parents):
        for parent in parents:
            if not isinstance(parent, NodeReference):
                parent_name = "_{}_{}".format(self.name, random_name())
                parent = Constant(parent, name=parent_name, model=self.model)
            self.model.add_edge(parent.name, self.name)

    @classmethod
    def reference(cls, name, model):
        """Creates a reference for an existing node

        Parameters
        ----------
        name : string
            name of the node
        model : ElfiModel

        Returns
        -------
        NodePointer instance
        """
        instance = cls.__new__(cls)
        instance._init_reference(name, model)
        return instance

    def _init_reference(self, name, model):
        """Initializes all internal variables of the instance

        Parameters
        ----------
        name : name of the node in the model
        model : ElfiModel

        """
        self.name = name
        self.model = model

    def generate(self, batch_size=1, with_values=None):
        """Generates a batch. Useful for testing.

        Parameters
        ----------
        batch_size : int
        with_values : dict

        """
        result = self.model.generate(batch_size, self.name, with_values=with_values)
        return result[self.name]

    @staticmethod
    def compile_output(state):
        return state['fn']

    def _give_name(self, model):
        # Test if context info is available and try to give the same name as the variable
        # Please note that this is only a convenience methos which is not quaranteed to
        # work in all cases. If you require a specific name, pass the name argument.
        frame = inspect.currentframe()
        if frame:
            # Frames are available
            # Take the callers frame
            frame = frame.f_back.f_back
            info = inspect.getframeinfo(frame, 1)

            # Skip super calls to find the assignment frame
            while re.match('\s*super\(', info.code_context[0]):
                frame = frame.f_back
                info = inspect.getframeinfo(frame, 1)

            # Match simple direct assignment with the class name, no commas or semicolons
            # Also do not accept a name starting with an underscore
            rex = '\s*([^\W_][\w]*)\s*=\s*\w?[\w\.]*{}\('.format(self.__class__.__name__)
            match = re.match(rex, info.code_context[0])
            if match:
                name = match.groups()[0]
                # Return the same name as the assgined reference
                if not model.has_node(name):
                    return name

        # Inspecting the name failed, return a random name
        while True:
            name = "{}_{}".format(self.__class__.__name__.lower(), random_name())
            if not model.has_node(name):
                break

        return name

    def __getitem__(self, item):
        """

        Returns
        -------
        item from the state dict of the node
        """
        return self.model.get_node(self.name)[item]

    def __repr__(self):
        return "{}('{}')".format(self.__class__.__name__, self.name)

    def __str__(self):
        return self.name


class Constant(NodeReference):
    def __init__(self, value, **kwargs):
        state = dict(value=value)
        super(Constant, self).__init__(state=state, **kwargs)

    @staticmethod
    def compile_output(state):
        return state['value']


class StochasticMixin(NodeReference):
    def __init__(self, *args, state, **kwargs):
        # Flag that this node is stochastic
        state['stochastic'] = True
        super(StochasticMixin, self).__init__(*args, state=state, **kwargs)


class ObservableMixin(NodeReference):
    """
    """

    def __init__(self, *args, state, observed=None, **kwargs):
        # Flag that this node can be observed
        state['observable'] = True
        super(ObservableMixin, self).__init__(*args, state=state, **kwargs)

        if observed is not None:
            self.model.computation_context.observed[self.name] = observed

    @property
    def observed(self):
        obs_name = observed_name(self.name)
        result = self.model.generate(0, obs_name)
        return result[obs_name]


class ScipyLikeRV(StochasticMixin, NodeReference):
    def __init__(self, distribution="uniform", *params, size=None, **kwargs):
        """

        Parameters
        ----------
        distribution : str or scipy-like distribution object
        params : params of the distribution
        size : int, tuple or None, optional
            size of a single random draw. None (default) means a scalar.

        """

        state = dict(distribution=distribution, size=size, uses_batch_size=True)
        super(ScipyLikeRV, self).__init__(*params, state=state, **kwargs)

    @staticmethod
    def compile_output(state):
        size = state['size']
        distribution = state['distribution']
        if not (size is None or isinstance(size, tuple)):
            size = (size, )

        # Note: sending the scipy distribution object also pickles the global numpy random
        # state with it. If this needs to be avoided, the object needs to be constructed
        # on the worker.
        if isinstance(distribution, str):
            distribution = scipy_from_str(distribution)

        if not hasattr(distribution, 'rvs'):
            raise ValueError("Distribution {} "
                             "must implement a rvs method".format(distribution))

        output = partial(rvs_wrapper, distribution=distribution, size=size)
        return output

    @property
    def size(self):
        return self['size']

    def __repr__(self):
        d = self['distribution']

        if isinstance(d, str):
            name = "'{}'".format(d)
        elif hasattr(d, 'name'):
            name = "'{}'".format(d.name)
        elif isinstance(d, type):
            name = d.__name__
        else:
            name = d.__class__.__name__

        return super(ScipyLikeRV, self).__repr__()[0:-1] + ", {})".format(name)


class Prior(ScipyLikeRV):
    def __init__(self, *args, **kwargs):
        super(Prior, self).__init__(*args, **kwargs)
        if self.name not in self.model.parameters:
            self.model.parameters.append(self.name)


class Simulator(StochasticMixin, ObservableMixin, NodeReference):
    def __init__(self, fn, *dependencies, **kwargs):
        state = dict(fn=fn, uses_batch_size=True)
        super(Simulator, self).__init__(*dependencies, state=state, **kwargs)


class Summary(ObservableMixin, NodeReference):
    def __init__(self, fn, *dependencies, **kwargs):
        if not dependencies:
            raise ValueError('No dependencies given')
        state = dict(fn=fn)
        super(Summary, self).__init__(*dependencies, state=state, **kwargs)


class Discrepancy(NodeReference):
    def __init__(self, fn, *dependencies, **kwargs):
        if not dependencies:
            raise ValueError('No dependencies given')
        state = dict(fn=fn, uses_observed=True)
        super(Discrepancy, self).__init__(*dependencies, state=state, **kwargs)

    @staticmethod
    def compile_output(state):
        fn = state['fn']
        output_fn = partial(discrepancy_wrapper, fn=fn)
        return output_fn












