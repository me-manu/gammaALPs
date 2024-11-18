from __future__ import absolute_import, division, print_function
import numpy as np
import os
import gammaALPs
from numpy.testing import assert_allclose
from gammaALPs.bfields import gmf
from astropy.tests.helper import pytest


@pytest.fixture(scope='module')
def jansson_file(request, tmpdir_factory):
    path = tmpdir_factory.mktemp('tmp')

    outfile = os.path.join(path, "jansson_field_test.npy")
    url = 'https://raw.githubusercontent.com/me-manu/gammaALPs/master/data/jansson_field_test.npy'
    os.system('curl -o %s -OL %s' % (outfile, url))
    request.addfinalizer(lambda: path.remove(rec=1))

    return outfile


@pytest.fixture(scope='module')
def pshirkov_file(request, tmpdir_factory):
    path = tmpdir_factory.mktemp('tmp')

    outfile = os.path.join(path, "pshirkov_field_test.npy")
    url = 'https://raw.githubusercontent.com/me-manu/gammaALPs/master/data/pshirkov_field_test.npy'
    os.system('curl -o %s -OL %s' % (outfile, url))
    request.addfinalizer(lambda: path.remove(rec=1))

    return outfile


@pytest.fixture(scope='module')
def uf23_file(request, tmpdir_factory):
    path = tmpdir_factory.mktemp('tmp')

    outfile = os.path.join(path, "uf23_field_test.npy")
    # url = 'https://raw.githubusercontent.com/FriedL12/gammaALPs/gmf-unger-farrar/data/uf23_field_test.npy'
    url = 'https://raw.githubusercontent.com/me-manu/gammaALPs/master/data/uf23_field_test.npy'
    os.system('curl -o %s -OL %s' % (outfile, url))
    request.addfinalizer(lambda: path.remove(rec=1))

    return outfile


def test_jansson(jansson_file):

    # 3D coordinates for galacto centric coordinate sys
    x = np.linspace(-20., 20., 300)
    y = np.linspace(-20., 20., 300)
    z = np.linspace(-20., 20., 300)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # rho and phi component
    rr = np.sqrt(xx ** 2. + yy ** 2.)  # rho component
    pp = np.arctan2(yy, xx)  # phi component

    # index for z = 0 plane
    idz = np.argmin(np.abs(z))

    # index for y = 0 plane
    idy = np.argmin(np.abs(y))

    jansson = gmf.GMF()

    # disk component
    Bdisk = np.zeros_like(rr[..., idz])
    for i, r in enumerate(rr[..., idz]):
        b = jansson.Bdisk(rho=r, phi=pp[:, i, idz], z=zz[:, i, idz])
        Bdisk[:, i] = b[1] * gmf.signum(b[0][1, :])

    # halo component
    Bhalo = np.zeros_like(xx[:, idy, :])
    for i, xi in enumerate(xx[:, idy, :]):
        b = jansson.Bhalo(rho=np.sqrt(xi ** 2. + y[idy] ** 2.),
                          z=zz[i, idy, :])
        Bhalo[:, i] = b[1]

    # X component
    BX = np.zeros_like(xx[:, idy, :])
    for i, xi in enumerate(xx[:, idy, :]):
        b = jansson.BX(rho=np.sqrt(xi ** 2. + y[idy] ** 2.),
                       z=zz[i, idy, :])
        BX[:, i] = b[1]

    # uncomment these lines if you need to regenerate the files
    #jansson_file = os.path.join(os.path.dirname(os.path.dirname(gammaALPs.__file__)),
    #                            "data/jansson_field_test.npy")
    #np.save(jansson_file,
    #        {"X": BX, "halo": Bhalo, "disk": Bdisk})

    compare_fields = np.load(jansson_file, allow_pickle=True).flat[0]

    assert_allclose(BX, compare_fields["X"], rtol=1e-6)
    assert_allclose(Bdisk, compare_fields["disk"], rtol=1e-6)
    assert_allclose(Bhalo, compare_fields["halo"], rtol=1e-6)


def test_pshirkov(pshirkov_file):

    # 3D coordinates for galacto centric coordinate sys
    x = np.linspace(-20., 20., 300)
    y = np.linspace(-20., 20., 300)
    z = np.linspace(-20., 20., 300)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # rho and phi component
    rr = np.sqrt(xx ** 2. + yy ** 2.)  # rho component
    pp = np.arctan2(yy, xx)  # phi component

    # index for z = 0 plane
    idz = np.argmin(np.abs(z))

    # index for y = 0 plane
    idy = np.argmin(np.abs(y))

    pshirkov = gmf.GMFPshirkov(model='ASS')

    # disk component
    Bdisk = np.zeros_like(rr[..., idz])
    for i, r in enumerate(rr[..., idz]):
        b = pshirkov.Bdisk(rho=r, phi=pp[:, i, idz], z=zz[:, i, idz])
        Bdisk[:, i] = b[1] * gmf.signum(b[0][1, :])

    # halo component
    Bhalo = np.zeros_like(xx[:, idy, :])
    for i, xi in enumerate(xx[:, idy, :]):
        b = pshirkov.Bhalo(rho=np.sqrt(xi ** 2. + y[idy] ** 2.),
                          z=zz[i, idy, :])
        Bhalo[:, i] = b[1]

    # uncomment these lines if you need to regenerate the files
    #pshirkov_file = os.path.join(os.path.dirname(os.path.dirname(gammaALPs.__file__)),
                                 #"data/pshirkov_field_test.npy")
    #np.save(pshirkov_file,
            #{"halo": Bhalo, "disk": Bdisk})

    compare_fields = np.load(pshirkov_file, allow_pickle=True).flat[0]

    assert_allclose(Bdisk, compare_fields["disk"], rtol=1e-6)
    assert_allclose(Bhalo, compare_fields["halo"], rtol=1e-6)


def test_uf23(uf23_file):

    # 3D coordinates for galacto centric coordinate sys
    x = np.linspace(-20., 20., 300)
    y = np.linspace(-20., 20., 300)
    z = np.linspace(-20., 20., 300)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # rho and phi component
    rr = np.sqrt(xx ** 2. + yy ** 2.)  # rho component
    pp = np.arctan2(yy, xx)  # phi component

    # index for z = 0 plane
    idz = np.argmin(np.abs(z))

    # index for y = 0 plane
    idy = np.argmin(np.abs(y))

    models = ['base', 'expX', 'neCL', 'twistX', 'nebCor', 'cre10', 'synCG', 'spur']
    uf23 = []
    for model in models:
        uf23.append(gmf.UF23(model=model))

    # disk component
    Bdisk = np.zeros((len(models), *rr[...,idz].shape))
    for k in range(len(models)):
        for i, r in enumerate(rr[..., idz]):
            b = uf23[k].Bdisk(rho=r, phi=pp[:, i, idz], z=zz[:, i, idz])
            Bdisk[k][:, i] = b[1] * gmf.signum(b[0][1, :])

    # halo component
    Bhalo = np.zeros((len(models), *xx[:, idy, :].shape))
    for k in range(len(models)):
        for i, xi in enumerate(xx[:, idy, :]):
            b = uf23[k].Bhalo(rho=np.sqrt(xi ** 2. + y[idy] ** 2.),
                              z=zz[i, idy, :])
            Bhalo[k][:, i] = b[1]

    # uncomment these lines if you need to regenerate the files
    #uf23_file = os.path.join(os.path.dirname(os.path.dirname(gammaALPs.__file__)),
                                 #"data/uf23_field_test.npy")
    #np.save(uf23_file,
            #{"halo": Bhalo, "disk": Bdisk})

    compare_fields = np.load(uf23_file, allow_pickle=True).flat[0]

    assert_allclose(Bdisk, compare_fields["disk"], rtol=1e-6)
    assert_allclose(Bhalo, compare_fields["halo"], rtol=1e-6)