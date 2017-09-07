import os

import pytest

import elfi
from elfi.examples import bdm, gauss, gauss_nd, ricker, gnk, bignk


def test_bdm():
    """Currently only works in unix-like systems and with a cloned repository"""
    cpp_path = bdm.get_sources_path()

    do_cleanup = False
    if not os.path.isfile(cpp_path + '/bdm'):
        os.system('make -C {}'.format(cpp_path))
        do_cleanup = True

    assert os.path.isfile(cpp_path + '/bdm')

    # Remove the executable if it already exists
    if os.path.isfile('bdm'):
        os.system('rm bdm')

    with pytest.warns(RuntimeWarning):
        m = bdm.get_model()

    # Copy the file here to run the test
    os.system('cp {}/bdm .'.format(cpp_path))

    # Should no longer warn
    m = bdm.get_model()

    # Test that you can run the inference

    rej = elfi.Rejection(m, 'd', batch_size=100)
    rej.sample(20)

    # TODO: test the correctness of the result

    os.system('rm ./bdm')
    if do_cleanup:
        os.system('rm {}/bdm'.format(cpp_path))

def test_gauss():
    m = gauss.get_model()
    rej = elfi.Rejection(m, m['d'], batch_size=10)
    rej.sample(20)

def test_gauss_1d():
    params_true = [4]
    m = gauss_nd.get_model(true_params=params_true, cov_ii=1)
    rej = elfi.Rejection(m, m['d'], batch_size=10)
    rej.sample(20)


def test_gauss_2d():
    params_true = [4, 4]
    m = gauss_nd.get_model(true_params=params_true, cov_ii=1, cov_ij=.5)
    rej = elfi.Rejection(m, m['d'], batch_size=10)
    rej.sample(20)


def test_Ricker():
    m = ricker.get_model()
    rej = elfi.Rejection(m, m['d'], batch_size=10)
    rej.sample(20)


def test_gnk():
    m = gnk.get_model()
    rej = elfi.Rejection(m, m['d'], batch_size=10)
    rej.sample(20)


def test_bignk(stats_summary=['ss_octile']):
    m = bignk.get_model()
    rej = elfi.Rejection(m, m['d'], batch_size=10)
    rej.sample(20)
