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


##############
# Generation #
##############

def test_generated_particles(ring):
    line, twiss = ring
    part = line.xfields.bhabha.generate(at='bhabha.1')
    marker = line['bhabha.1']
    tab = line.get_table()

    n = np.sum(part.state > xt.particles.LAST_INVALID_STATE)
    assert n > 100
    assert part._capacity == 2*n
    alive = part.state > 0
    assert np.all(part.at_element[:n] == line.element_names.index('bhabha.1'))
    assert np.allclose(part.s[:n], tab['s', 'bhabha.1'])
    assert np.all(part.delta[alive] - marker.delta_co < marker.delta_cut)
    # the rate of the crossings of the bunch particles, scaled down by
    # compt_scale, shared by the generated particles
    assert np.allclose(
        part.weight[:n],
        BEAM['bunch_intensity']/twiss.t_rev0
        / (GENERATOR['compt_scale']*GENERATOR['n_generated']))
    assert marker.luminosity > 0
    assert 0 < marker.mean_photons_per_particle < 0.1


def test_generation_is_reproducible(ring):
    line, _ = ring
    bhabha = line.xfields.bhabha
    part = bhabha.generate(at='bhabha.0')
    _assert_same_particles(part, bhabha.generate(at='bhabha.0'))
    # independent of the other markers generated
    _assert_same_particles(part, bhabha.generate()['bhabha.0'])
    # another seed gives other particles
    other = bhabha.generate(at='bhabha.0', seed=1)
    assert not np.array_equal(part.delta[:10], other.delta[:10])


def test_replicas(ring):
    line, _ = ring
    bhabha = line.xfields.bhabha
    single = bhabha.generate(at='bhabha.0')
    replicas = [bhabha.generate(at='bhabha.0', replica=rr, n_replicas=2)
                for rr in range(2)]
    assert np.allclose(replicas[0].weight[0], single.weight[0]/2)
    assert not np.array_equal(replicas[0].delta[:10], replicas[1].delta[:10])
    with pytest.raises(ValueError):
        bhabha.generate(at='bhabha.0', replica=2, n_replicas=2)


def test_generation_from_a_reloaded_line(ring, tmp_path):
    line, _ = ring
    line.to_json(tmp_path / 'line.json')
    reloaded = xt.load(tmp_path / 'line.json')
    assert reloaded.xfields.bhabha.marker_names == ['bhabha.0', 'bhabha.1']
    _assert_same_particles(line.xfields.bhabha.generate(at='bhabha.1'),
                           reloaded.xfields.bhabha.generate(at='bhabha.1'))


def test_rate_does_not_depend_on_compt_scale(ring):
    """
    compt_scale scales up the emissions and down the weights: the rate of
    the emissions beyond the cut is the same, within the statistical error.
    """
    _, twiss = ring
    rates = []
    for compt_scale, n_generated in ((1e10, 20_000), (2e9, 100_000)):
        line = _configured(twiss, at='bhabha.0', compt_scale=compt_scale,
                           n_generated=n_generated)
        part = line.xfields.bhabha.generate(at='bhabha.0')
        weight = part.weight[part.state > 0]
        rates.append((weight.sum(), weight[0]*np.sqrt(len(weight))))
    (rate_1, error_1), (rate_2, error_2) = rates
    assert abs(rate_1 - rate_2) < 4*np.hypot(error_1, error_2)


def test_pileup_warning(ring):
    _, twiss = ring
    line = _configured(twiss, at='bhabha.0', compt_scale=1e11,
                       n_generated=2_000)
    with pytest.warns(UserWarning, match='photons per particle'):
        line.xfields.bhabha.generate(at='bhabha.0')


def test_generate_checks_its_arguments(ring):
    line, _ = ring
    bhabha = line.xfields.bhabha
    with pytest.raises(ValueError):
        bhabha.generate(n_replicas=0)
    with pytest.raises(TypeError):
        bhabha.generate(at='mqf.1')
    unconfigured = _ring_with_markers()
    with pytest.raises(ValueError, match='not configured'):
        unconfigured.xfields.bhabha.generate()
