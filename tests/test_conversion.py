from __future__ import absolute_import, division, print_function
import numpy as np
from numpy.testing import assert_allclose
from gammaALPs.core import Source, ALP, ModuleList
from astropy import units as u
from ebltable.tau_from_model import OptDepth


class TestConversionModules:

    def test_icm_gauss(self):
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
        # check access to module
        assert m.modules['MixICMGaussTurb'] == m.modules[0]

        # check conversion prop
        px, py, pa = m.run(multiprocess=1)

        assert_allclose(px + py + pa, np.ones_like(px), rtol=1e-6)

        # check conversion prop for multiprocess
        # and check that random seed is working correctly
        px2, py2, pa2 = m.run(multiprocess=2)
        assert_allclose(px2 + py2 + pa2, np.ones_like(px), rtol=1e-6)

        assert_allclose(px, px2, rtol=1e-6)
        assert_allclose(py, py2, rtol=1e-6)
        assert_allclose(pa, pa2, rtol=1e-6)

    def test_jet_helical_tangled(self):
        EGeV = np.logspace(1., 3.5, 50)
        pin = np.diag((1., 1., 0.)) * 0.5
        source = Source(z=0.034, ra='16h53m52.2s', dec='+39d45m37s')
        alp = ALP(1., 1.)
        m = ModuleList(alp, source, pin=pin, EGeV=EGeV)

        m.add_propagation("JetHelicalTangled",
                          0,  # position of module counted from the source.
                          ndom=400,
                          ft=0.7,  # fraction of magnetic field energy density in tangled field
                          Bt_exp=-1.,  # exponent of the transverse component of the helical field
                          r_T=0.3,  # radius at which helical field becomes toroidal in pc
                          r0=0.3,  # radius where B field is equal to b0 in pc
                          B0=0.8,  # Bfield strength in G
                          g0=9.,  # jet lorenz factor at r0
                          n0=1.e4,  # electron density at r0 in cm**-3
                          rjet=98.3e+3,  # jet length in pc
                          rvhe=0.3,  # distance of gamma-ray emission region from BH in pc
                          alpha=1.68,  # power-law index of electron energy distribution function
                          l_tcor='jetwidth',  # tangled field coherence average length in pc if a constant, or keyword
                          jwf_dist='Uniform'  # type of distribution for jet width factors (jwf)
                          )
        # check access to module
        assert m.modules['JetHelicalTangled'] == m.modules[0]

        # check conversion prop
        px, py, pa = m.run(multiprocess=2)

        assert_allclose(px + py + pa, np.ones_like(px), rtol=1e-6)

    def test_jet(self):
        EGeV = np.logspace(1., 3.5, 50)
        pin = np.diag((1., 1., 0.)) * 0.5
        src = Source(z=0.859000, ra='22h53m57.7s', dec='+16d08m54s',
                     bLorentz=15.6, theta_obs=1.3)  # 3C454.3
        alp = ALP(1., 1.)
        m = ModuleList(alp, src, pin=pin, EGeV=EGeV)

        gamma_min = 1.
        m.add_propagation("Jet",
                          0,  # position of module counted from the source.
                          B0=0.32,  # Jet field at r = R0 in G
                          r0=1.,  # distance from BH where B = B0 in pc
                          rgam=3.19e17 * u.cm.to('pc'),  # distance of gamma-ray emitting region to BH
                          alpha=-1,  # exponent of toroidal magnetic field (default: -1.)
                          psi=np.pi / 4.,  # Angle between one photon polarization state and B field.
                          # Assumed constant over entire jet.
                          helical=True,  # if True, use helical magnetic-field model from Clausen-Brown et al. (2011).
                          # In this case, the psi kwarg is treated is as the phi angle
                          # of the photon trajectory in the cylindrical jet coordinate system
                          equipartition=True,  # if true, assume equipartition between electrons and the B field.
                          # This will overwrite the exponent of electron density beta = 2 * alpha
                          # and set n0 given the minimum electron lorentz factor set with gamma_min
                          gamma_min=gamma_min,
                          # minimum lorentz factor of emitting electrons, only used if equipartition = True
                          gamma_max=np.exp(10.) * gamma_min,
                          # maximum lorentz factor of emitting electrons, only used if equipartition = True
                          Rjet=40.,  # maximum jet length in pc (default: 1000.)
                          n0=1e4, beta=-2.
                          )
        # check access to module
        assert m.modules['Jet'] == m.modules[0]

        # check conversion prop
        px, py, pa = m.run(multiprocess=2)

        assert_allclose(px + py + pa, np.ones_like(px), rtol=1e-6)

    def test_icm_cell(self):
        EGeV = np.logspace(1., 3.5, 50)
        pin = np.diag((1., 1., 0.)) * 0.5
        source = Source(z=0.116, ra='21h58m52.0s', dec='-30d13m32s')
        alp = ALP(1., 1.)
        m = ModuleList(alp, source, pin=pin, EGeV=EGeV)

        m.add_propagation("ICMCell",
                          0,
                          nsim=10
                          )
        # check access to module
        assert m.modules['MixICMCell'] == m.modules[0]

        # check conversion prop
        px, py, pa = m.run(multiprocess=2)

        assert_allclose(px + py + pa, np.ones_like(px), rtol=1e-5)

    def test_gmf(self):
        EGeV = np.logspace(1., 3.5, 50)
        pin = np.diag((1., 1., 0.)) * 0.5
        source = Source(z=0.116, ra='21h58m52.0s', dec='-30d13m32s')
        alp = ALP(1., 1.)
        m1 = ModuleList(alp, source, pin=pin, EGeV=EGeV)
        m2 = ModuleList(alp, source, pin=pin, EGeV=EGeV)
        m3 = ModuleList(alp, source, pin=pin, EGeV=EGeV)
        m4 = ModuleList(alp, source, pin=pin, EGeV=EGeV)
        m5 = ModuleList(alp, source, pin=pin, EGeV=EGeV)

        m1.add_propagation("GMF",
                          0,
                          model='jansson12'
                          )
        m2.add_propagation("GMF",
                           0,
                           model='jansson12b'
                           )
        m3.add_propagation("GMF",
                           0,
                           model='jansson12c'
                           )
        m4.add_propagation("GMF",
                           0,
                           model='pshirkov',
                           model_sym='BSS'
                           )
        m5.add_propagation("GMF",
                           0,
                           model='pshirkov',
                           model_sym='ASS'
                           )
        # check access to module
        assert m1.modules['MixGMF'] == m1.modules[0]

        # check conversion prop
        px1, py1, pa1 = m1.run(multiprocess=2)
        px2, py2, pa2 = m2.run(multiprocess=2)
        px3, py3, pa3 = m3.run(multiprocess=2)
        px4, py4, pa4 = m4.run(multiprocess=2)
        px5, py5, pa5 = m5.run(multiprocess=2)

        assert_allclose(px1 + py1 + pa1, np.ones_like(px1), rtol=1e-5)
        assert_allclose(px2 + py2 + pa2, np.ones_like(px2), rtol=1e-5)
        assert_allclose(px3 + py3 + pa3, np.ones_like(px3), rtol=1e-5)
        assert_allclose(px4 + py4 + pa4, np.ones_like(px4), rtol=1e-5)
        assert_allclose(px5 + py5 + pa5, np.ones_like(px5), rtol=1e-5)

    def test_igmf(self):
        EGeV = np.logspace(1., 3.5, 50)
        pin = np.diag((1., 1., 0.)) * 0.5
        source = Source(z=0.116, ra='21h58m52.0s', dec='-30d13m32s')

        alp = ALP(m=1., g=1e-10)
        ebl_model = 'dominguez'

        m = ModuleList(alp, source, pin=pin, EGeV=EGeV)

        m.add_propagation("IGMF",
                          0,
                          nsim=10,
                          B0=1e-10,  # B field in micro Gauss
                          n0=1e-7,
                          L0=1e3,
                          eblmodel=ebl_model
                          )
        # check access to module
        assert m.modules['MixIGMFCell'] == m.modules[0]

        # mixing should be so small, that it's the same as EBL attenuation only
        px, py, pa = m.run(multiprocess=1)

        pgg = px + py

        tau = OptDepth.readmodel(model='dominguez')

        # these numbers are pretty high,
        # this is reported in a github issue
        # and needs to be investigated further
        rtol = 0.2
        atol = 0.03
        for p in pgg:
            assert_allclose(p, np.exp(-tau.opt_depth(source.z, EGeV / 1e3)), rtol=rtol, atol=atol)

# TODO add tests for modules initialized from files / arrays
