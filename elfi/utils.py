import scipy.stats as ss
import networkx as nx
import numpy as np


SCIPY_ALIASES = {
    'normal': 'norm',
    'exponential': 'expon',
    'unif': 'uniform',
    'bin': 'binom',
    'binomial': 'binom'
}


def scipy_from_str(name):
    name = name.lower()
    name = SCIPY_ALIASES.get(name, name)
    return getattr(ss, name)


def observed_name(name):
    return "_{}_observed".format(name)


def args_to_tuple(*args):
    return tuple(args)


def is_array(output):
    # Ducktyping numpy arrays
    return hasattr(output, 'shape')


# NetworkX utils


def nbunch_ancestors(G, nbunch):
    # Resolve output ancestors
    ancestors = set(nbunch)
    for node in nbunch:
        ancestors = ancestors.union(nx.ancestors(G, node))
    return ancestors


def get_sub_seed(random_state, sub_seed_index, high=2**31):
    """Returns a sub seed. The returned sub seed is unique for its index, i.e. no
    two indexes can return the same sub_seed. Same random_state will also always
    produce the same sequence.

    Parameters
    ----------
    random_state : np.random.RandomState
    sub_seed_index : int
    high : int
        upper limit for the range of sub seeds (exclusive)

    Returns
    -------
    int
        from interval [0, high - 1]

    Notes
    -----
    There is no guarantee how close the random_states initialized with sub_seeds may end
    up to each other. Better option is to use PRNG:s that have an advance or jump
    functions available.

    """

    if sub_seed_index >= high:
        raise ValueError("Sub seed index {} is out of range".format(sub_seed_index))

    n_unique = 0
    n_unique_required = sub_seed_index + 1
    sub_seeds = None
    seen = set()
    while n_unique != n_unique_required:
        n_draws = n_unique_required - n_unique
        sub_seeds = random_state.randint(high, size=n_draws, dtype='uint32')
        seen.update(sub_seeds)
        n_unique = len(seen)

    return sub_seeds[-1]


def tabulate(fun, *args):
    """Compute a function on the cartesian product of the arguments.

    Parameters
    ----------
    fun
      function to compute
    *args : array_like
      points along each axis

    Returns
    -------
    (grid, result)
      A meshgrid constructed from the given points and
      the results of the function evaluations.

    Examples
    --------
    >>> from elfi import utils
    >>> arr = np.arange(1, 4)
    >>> grid, res = utils.tabulate(lambda x: x[0] + x[1], arr, arr)
    >>> res
    array([[2, 3, 4],
           [3, 4, 5],
           [4, 5, 6]])
    """
    grid = np.meshgrid(*args)
    stack = np.stack(grid, axis=0)
    return grid, np.apply_along_axis(fun, 0, stack)


def tabulate_n(funs, *args):
    """Compute functions on the cartesian product of the arguments.

    Same as :func:`~elfi.utils.tabulate`, but for multiple functions.

    Parameters
    ----------
    funs
      a list of functions to evaluate
    *args: array_like
      points along each axis

    Returns
    -------
    (grid, [results])
      A meshgrid constructed from the given points and
      a list of results corresponding to each function.

    """
    grid = np.meshgrid(*args)
    stack = np.stack(grid, axis=0)
    return grid, [np.apply_along_axis(fun, 0, stack) for fun in funs]
