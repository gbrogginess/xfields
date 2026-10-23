# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #
from pathlib import Path

import numpy as np
import pytest

import xtrack as xt
import xfields as xf


FOUR_CELL_RING = (
    Path(__file__).parent.parent / 'test_data' / 'four_cell_ring' / 'line.json')

# Interaction points of the tests: two drifts of the ring
IPS = ['d2.1', 'd4.2']

BEAM = dict(nemitt_x=1e-5, nemitt_y=1e-7, sigma_z=4e-3, sigma_delta=1e-3,
            bunch_intensity=4e10, phi=0.0)
# ~0.03 photons per particle: a few hundred particles beyond the cut
GENERATOR = dict(compt_scale=1e10, n_generated=10_000, delta_cut=-0.01)


def _ring_with_markers():
    """Four-cell electron ring with a marker at each interaction point."""
    line = xt.load(FOUR_CELL_RING)
    line.xfields.bhabha.install_markers(at=IPS)
    return line


@pytest.fixture(scope='module')
def ring():
    """
    The ring with markers, configured, and its 4d twiss (the ring has no
    RF): the other tests configure fresh rings with the same layout using
    this twiss.
    """
    line = _ring_with_markers()
    twiss = line.twiss(method='4d')
    line.xfields.bhabha.configure(twiss=twiss, seed=1997, **BEAM, **GENERATOR)
    return line, twiss


def _configured(twiss, **kwargs):
    line = _ring_with_markers()
    config = dict(twiss=twiss, seed=1997, **BEAM, **GENERATOR)
    config.update(kwargs)
    line.xfields.bhabha.configure(**config)
    return line


def _assert_same_particles(part, other):
    for cc in ('x', 'px', 'y', 'py', 'zeta', 'delta', 'weight', 'state'):
        assert np.array_equal(getattr(part, cc), getattr(other, cc))


###########################
# Installation, configure #
###########################

def test_facade_returns_a_generator_bound_to_the_line():
    line = xt.load(FOUR_CELL_RING)
    bhabha = line.xfields.bhabha
    assert isinstance(bhabha, xf.RadiativeBhabhaGenerator)
    assert bhabha.line is line


def test_generator_needs_a_line():
    with pytest.raises(TypeError):
        xf.RadiativeBhabhaGenerator('not a line')


def test_install_markers_at_the_interaction_points():
    ref_line = xt.load(FOUR_CELL_RING)
    ref_tab = ref_line.get_table()
    line = xt.load(FOUR_CELL_RING)
    names = line.xfields.bhabha.install_markers(at=IPS)
    assert names == ['bhabha.0', 'bhabha.1']
    assert line.xfields.bhabha.marker_names == names
    tab = line.get_table()
    for nn, ip in zip(names, IPS):
        assert isinstance(line[nn], xf.RadiativeBhabhaMarker)
        assert np.isclose(tab['s', nn],
                          ref_tab['s', ip] + 0.5*ref_line[ip].length)
    # the markers are transparent for the optics
    twiss = line.twiss(method='4d')
    ref = ref_line.twiss(method='4d')
    assert np.isclose(twiss.qx, ref.qx, rtol=0, atol=1e-12)
    assert np.isclose(twiss.qy, ref.qy, rtol=0, atol=1e-12)


def test_install_markers_names():
    line = xt.load(FOUR_CELL_RING)
    bhabha = line.xfields.bhabha
    assert bhabha.install_markers(at=IPS[0], names='ipa') == ['ipa']
    assert bhabha.install_markers(at=IPS[1], prefix='ip') == ['ip.0']
    with pytest.raises(ValueError):
        bhabha.install_markers(at=IPS[1], names='ipa')
    with pytest.raises(ValueError):
        bhabha.install_markers(at=IPS, names=['a', 'a'])
    with pytest.raises(ValueError):
        bhabha.install_markers(at=[])


def test_configure_stores_the_configuration_on_the_markers(ring):
    line, twiss = ring
    gamma0 = line.particle_ref.gamma0[0]
    beta0 = line.particle_ref.beta0[0]
    for nn in line.xfields.bhabha.marker_names:
        marker = line[nn]
        assert marker.p0c == line.particle_ref.p0c[0]
        for kk in ('alfx', 'betx', 'alfy', 'bety', 'dx', 'dpx', 'dy', 'dpy'):
            assert getattr(marker, kk) == twiss[kk, nn]
        for kk in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
            assert getattr(marker, f'{kk}_co') == twiss[kk, nn]
        assert marker.t_rev0 == twiss.t_rev0
        assert np.isclose(marker.gemitt_x, BEAM['nemitt_x']/(beta0*gamma0))
        assert np.isclose(marker.gemitt_y, BEAM['nemitt_y']/(beta0*gamma0))
        for kk in ('sigma_z', 'sigma_delta', 'bunch_intensity', 'phi'):
            assert getattr(marker, kk) == BEAM[kk]
        for kk, vv in GENERATOR.items():
            assert getattr(marker, kk) == vv
        # electrons collide with positrons by default
        assert marker.other_beam_q0 == -line.particle_ref.q0
        assert marker.seed == 1997


def test_configure_stores_a_drawn_seed(ring):
    _, twiss = ring
    line = _configured(twiss, seed=None)
    seeds = {int(line[nn].seed) for nn in line.xfields.bhabha.marker_names}
    assert len(seeds) == 1
    assert seeds.pop() > 0


def test_configure_a_subset_of_the_markers(ring):
    _, twiss = ring
    line = _configured(twiss, at='bhabha.1', delta_cut=-0.02)
    assert line['bhabha.1'].delta_cut == -0.02
    assert line['bhabha.0'].p0c == 0    # not configured


def test_configure_checks_its_arguments(ring):
    _, twiss = ring
    line = _ring_with_markers()
    bhabha = line.xfields.bhabha
    config = dict(twiss=twiss, **BEAM, **GENERATOR)
    for kk in ('bunch_intensity', 'sigma_z', 'sigma_delta', 'phi',
               'compt_scale', 'n_generated'):
        with pytest.raises(ValueError, match=kk):
            bhabha.configure(**{**config, kk: None})
    with pytest.raises(ValueError):
        bhabha.configure(**{**config, 'compt_scale': 0})
    with pytest.raises(ValueError):
        bhabha.configure(**{**config, 'n_generated': 0})
    with pytest.raises(ValueError):
        bhabha.configure(**config, gemitt_x=1e-9, gemitt_y=1e-11)
    with pytest.raises(TypeError):
        bhabha.configure(**config, at='mqf.1')
    with pytest.raises(ValueError):
        bhabha.configure(**config, at='no_such_element')
