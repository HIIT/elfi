"""Common utilities."""

import uuid

import networkx as nx
import numpy as np
import scipy.stats as ss

SCIPY_ALIASES = {
    'normal': 'norm',
    'exponential': 'expon',
    'unif': 'uniform',
    'bin': 'binom',
    'binomial': 'binom'
}


def scipy_from_str(name):
    """Return the scipy.stats distribution corresponding to `name`."""
    name = name.lower()
    name = SCIPY_ALIASES.get(name, name)
    return getattr(ss, name)


def random_seed():
    """Extract the seed from numpy RandomState.

    Alternative would be to use os.urandom(4) cast as int.
    """
    return np.random.RandomState().get_state()[1][0]


def random_name(length=4, prefix=''):
    """Generate a random string.

    Parameters
    ----------
    length : int, optional
    prefix : str, optional

    """
    return prefix + str(uuid.uuid4().hex[0:length])


def observed_name(name):
    """Return `_name_observed`."""
    return "_{}_observed".format(name)


def args_to_tuple(*args):
    """Combine args into a tuple."""
    return tuple(args)


def is_array(output):
    """Check if `output` behaves as np.array (simple)."""
    return hasattr(output, 'shape')


# NetworkX utils


def nbunch_ancestors(G, nbunch):
    """Resolve output ancestors."""
    ancestors = set(nbunch)
    for node in nbunch:
        ancestors = ancestors.union(nx.ancestors(G, node))
    return ancestors


def get_sub_seed(seed, sub_seed_index, high=2**31, cache=None):
    """Return a sub seed.

    The returned sub seed is unique for its index, i.e. no two indexes can
    return the same sub_seed.

    Parameters
    ----------
    seed : int
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
    if isinstance(seed, np.random.RandomState):
        raise ValueError('Seed cannot be a random state')

    random_state = np.random.RandomState(seed)

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

    if cache is None:
        return sub_seeds[-1]
    else:
        return sub_seeds, cache
