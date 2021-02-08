from __future__ import absolute_import, division, print_function
import numpy as np
import os
from numpy.testing import assert_allclose
from gammaALPs.core import Source, ALP, ModuleList
from astropy.tests.helper import pytest


@pytest.fixture(scope='module')
def conv_ngc1275_file(request, tmpdir_factory):
    path = tmpdir_factory.mktemp('data')

    outfile = "conversion_prob_ngc1275.npy"
    url = 'https://raw.githubusercontent.com/me-manu/gammaALPs/master/data/conversion_prob_ngc1275.npy'
    os.system('curl -o %s -OL %s' % (outfile, url))
    request.addfinalizer(lambda: path.remove(rec=1))

    return outfile


@pytest.fixture(scope='module')
def conv_ngc1275_file_no_ebl(request, tmpdir_factory):
    path = tmpdir_factory.mktemp('data')

    outfile = "conversion_prob_ngc1275_no_ebl.npy"
    url = 'https://raw.githubusercontent.com/me-manu/gammaALPs/master/data/conversion_prob_ngc1275_no_ebl.npy'
    os.system('curl -o %s -OL %s' % (outfile, url))
    request.addfinalizer(lambda: path.remove(rec=1))

    return outfile


@pytest.fixture(scope='module')
def conv_prob_los_file(request, tmpdir_factory):
    path = tmpdir_factory.mktemp('data')

    outfile = "conversion_prob_los.npy"
    url = 'https://raw.githubusercontent.com/me-manu/gammaALPs/master/data/conversion_prob_los.npy'
    os.system('curl -o %s -OL %s' % (outfile, url))
    request.addfinalizer(lambda: path.remove(rec=1))

    return outfile


@pytest.fixture(scope='module')
def conv_prob_los_ebl_file(request, tmpdir_factory):
    path = tmpdir_factory.mktemp('data')

    outfile = "jansson_field_test.npy"
    url = 'https://raw.githubusercontent.com/me-manu/gammaALPs/master/data/conversion_prob_los_ebl.npy'
    os.system('curl -o %s -OL %s' % (outfile, url))
    request.addfinalizer(lambda: path.remove(rec=1))

    return outfile

class TestConversionModules:

    def test_icm_gauss_ebl_gmf(self, conv_ngc1275_file):
        EGeV = np.logspace(1., 3.5, 50)
        pin = np.diag((1., 1., 0.)) * 0.5
        source = Source(z=0.017559, ra='03h19m48.1s', dec='+41d30m42s')
        alp = ALP(1., 1.)
        m = ModuleList(alp, source, pin=pin, EGeV=EGeV)

        # test for Perseus B field
        m.add_propagation("ICMGaussTurb",
                          0,
                          nsim=10,
                          B0=10.,
                          n0=39.,
                          n2=4.05,
                          r_abell=500.,
                          r_core=80.,
                          r_core2=280.,
                          beta=1.2,
                          beta2=0.58,
                          eta=0.5,
                          kL=0.18,
                          kH=9.,
                          q=-2.80,
                          seed=0
                          )
        m.add_propagation("EBL", 1, eblmodel="dominguez", eblnorm=1.)
        m.add_propagation("GMF", 2, model="jansson12")

        # change ALP mass
        m.alp.m = 30.
        m.alp.g = 0.5

        # check conversion prop using multiprocessing
        px, py, pa = m.run(multiprocess=4)

        #np.save("conversion_prob_ngc1275", {"px": px, "py": py, "pa": pa})
        #conv_ngc1275_file = "conversion_prob_ngc1275.npy"

        compare_conv_prob = np.load(conv_ngc1275_file, allow_pickle=True).flat[0]

        assert_allclose(px, compare_conv_prob['px'], rtol=1e-5)
        assert_allclose(py, compare_conv_prob['py'], rtol=1e-5)
        assert_allclose(pa, compare_conv_prob['pa'], rtol=1e-5)

    def test_icm_gauss_no_ebl_gmf(self, conv_ngc1275_file_no_ebl):
        EGeV = np.logspace(1., 3.5, 50)
        pin = np.diag((1., 1., 0.)) * 0.5
        source = Source(z=0.017559, ra='03h19m48.1s', dec='+41d30m42s')
        alp = ALP(1., 1.)
        m = ModuleList(alp, source, pin=pin, EGeV=EGeV)

        # test for Perseus B field
        m.add_propagation("ICMGaussTurb",
                          0,
                          nsim=10,
                          B0=10.,
                          n0=39.,
                          n2=4.05,
                          r_abell=500.,
                          r_core=80.,
                          r_core2=280.,
                          beta=1.2,
                          beta2=0.58,
                          eta=0.5,
                          kL=0.18,
                          kH=9.,
                          q=-2.80,
                          seed=0
                          )
        m.add_propagation("GMF", 1, model="jansson12")

        # change ALP mass
        m.alp.m = 30.
        m.alp.g = 0.5

        # check conversion prop using multiprocessing
        px, py, pa = m.run(multiprocess=4)

        #np.save("conversion_prob_ngc1275_no_ebl", {"px": px, "py": py, "pa": pa})
        #conv_ngc1275_file_no_ebl = "conversion_prob_ngc1275_no_ebl.npy"

        compare_conv_prob = np.load(conv_ngc1275_file_no_ebl, allow_pickle=True).flat[0]

        assert_allclose(px, compare_conv_prob['px'], rtol=1e-5)
        assert_allclose(py, compare_conv_prob['py'], rtol=1e-5)
        assert_allclose(pa, compare_conv_prob['pa'], rtol=1e-5)

    def test_full_los(self, conv_prob_los_file):
        EGeV = np.logspace(1., 3.5, 50)
        pin = np.diag((1., 1., 0.)) * 0.5
        alp = ALP(1., 1.)
        source = Source(z=0.034, ra='16h53m52.2s', dec='+39d45m37s', bLorentz=9.)  # Mrk501

        m = ModuleList(alp, source, pin=pin, EGeV=EGeV)

        m.add_propagation("JetHelicalTangled",
                           0,  # position of module counted from the source.
                           ndom=400,
                           ft=0.7,  # fraction of magnetic field energy density in tangled field
                           Bt_exp=-1.,  # exponent of the transverse component of the helical field
                           r_T=0.3,  # radius at which helical field becomes toroidal in pc
                           r0=0.3,  # radius where B field is equal to b0 in pc
                           B0=0.8,  # Bfield strength in G
                           n0=1.e4,  # electron density at r0 in cm**-3
                           rjet=98.3e+3,  # jet length in pc
                           rvhe=0.3,  # distance of gamma-ray emission region from BH in pc
                           alpha=1.68,  # power-law index of electron energy distribution function
                           l_tcor='jetwidth',  # tangled field coherence average length in pc if a constant, or keyword
                           # jwf = 1.,  # jet width factor used when calculating l_tcor = jwf*jetwidth
                           jwf_dist='Uniform',  # type of distribution for jet width factors (jwf)
                           seed=0
                           )

        m.add_propagation("ICMGaussTurb",
                          1,
                          nsim=1,
                          B0=10.,
                          n0=39.,
                          n2=4.05,
                          r_abell=500.,
                          r_core=80.,
                          r_core2=280.,
                          beta=1.2,
                          beta2=0.58,
                          eta=0.5,
                          kL=0.18,
                          kH=9.,
                          q=-2.80,
                          seed=0
                          )

        m.add_propagation("IGMF",
                          2,  # position of module counted from the source.
                          nsim=1,  # number of random B-field realizations
                          B0=1e-3,  # B field strength in micro Gauss at z = 0
                          n0=1e-7,  # normalization of electron density in cm^-3 at z = 0
                          L0=1e3,  # coherence (cell) length in kpc at z = 0
                          eblmodel='dominguez',  # EBL model
                          seed=0
                          )

        m.add_propagation("GMF", 3, model="jansson12")

        # change ALP mass
        m.alp.m = 30.
        m.alp.g = 0.5

        # check conversion prop using multiprocessing
        px, py, pa = m.run(multiprocess=4)

        #np.save("conversion_prob_los", {"px": px, "py": py, "pa": pa})
        #conv_prob_los_file = "conversion_prob_los.npy"

        compare_conv_prob = np.load(conv_prob_los_file, allow_pickle=True).flat[0]

        assert_allclose(px, compare_conv_prob['px'], rtol=1e-6)
        assert_allclose(py, compare_conv_prob['py'], rtol=1e-6)
        assert_allclose(pa, compare_conv_prob['pa'], rtol=1e-6)

    def test_full_los_ebl(self, conv_prob_los_ebl_file):
        EGeV = np.logspace(1., 3.5, 50)
        pin = np.diag((1., 1., 0.)) * 0.5
        alp = ALP(1., 1.)
        source = Source(z=0.034, ra='16h53m52.2s', dec='+39d45m37s', bLorentz=9.)  # Mrk501

        m = ModuleList(alp, source, pin=pin, EGeV=EGeV)

        m.add_propagation("JetHelicalTangled",
                          0,  # position of module counted from the source.
                          ndom=400,
                          ft=0.7,  # fraction of magnetic field energy density in tangled field
                          Bt_exp=-1.,  # exponent of the transverse component of the helical field
                          r_T=0.3,  # radius at which helical field becomes toroidal in pc
                          r0=0.3,  # radius where B field is equal to b0 in pc
                          B0=0.8,  # Bfield strength in G
                          n0=1.e4,  # electron density at r0 in cm**-3
                          rjet=98.3e+3,  # jet length in pc
                          rvhe=0.3,  # distance of gamma-ray emission region from BH in pc
                          alpha=1.68,  # power-law index of electron energy distribution function
                          l_tcor='jetwidth',  # tangled field coherence average length in pc if a constant, or keyword
                          # jwf = 1.,  # jet width factor used when calculating l_tcor = jwf*jetwidth
                          jwf_dist='Uniform',  # type of distribution for jet width factors (jwf)
                          seed=0
                          )

        m.add_propagation("ICMGaussTurb",
                          1,
                          nsim=1,
                          B0=10.,
                          n0=39.,
                          n2=4.05,
                          r_abell=500.,
                          r_core=80.,
                          r_core2=280.,
                          beta=1.2,
                          beta2=0.58,
                          eta=0.5,
                          kL=0.18,
                          kH=9.,
                          q=-2.80,
                          seed=0
                          )

        m.add_propagation("EBL", 2, eblmodel="dominguez", eblnorm=1.)

        m.add_propagation("GMF", 3, model="jansson12")

        # change ALP mass
        m.alp.m = 30.
        m.alp.g = 0.5

        # check conversion prop using multiprocessing
        px, py, pa = m.run(multiprocess=4)

        #np.save("conversion_prob_los_ebl", {"px": px, "py": py, "pa": pa})
        #conv_prob_los_ebl_file = "conversion_prob_los_ebl.npy"

        compare_conv_prob = np.load(conv_prob_los_ebl_file, allow_pickle=True).flat[0]

        assert_allclose(px, compare_conv_prob['px'], rtol=1e-6)
        assert_allclose(py, compare_conv_prob['py'], rtol=1e-6)
        assert_allclose(pa, compare_conv_prob['pa'], rtol=1e-6)


