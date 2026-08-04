# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #
import csv
import json
from pathlib import Path

import numpy as np
import pytest

import xfields as xf
import xtrack as xt


REFERENCE_DIR = (
    Path(__file__).parent.parent
    / 'test_data' / 'touschek' / 'elegant_reference'
)


def _build_reference_study(reference_rows):
    env = xt.Environment()
    line = env.new_line(components=[
        env.new('qf1', xt.Quadrupole, length=0.3, k1=0.1),
        env.new('d11', xt.Drift, length=1.0),
        env.new('b1', xt.Bend, length=3.0, angle=np.pi / 2),
        env.new('d21', xt.Drift, length=1.0),
        env.new('qd1', xt.Quadrupole, length=0.3, k1=-0.7),
        env.new('d31', xt.Drift, length=1.0),
        env.new('b2', xt.Bend, length=3.0, angle=np.pi / 2),
        env.new('d41', xt.Drift, length=1.0),
        env.new('qf2', xt.Quadrupole, length=0.3, k1=0.1),
        env.new('d12', xt.Drift, length=1.0),
        env.new('b3', xt.Bend, length=3.0, angle=np.pi / 2),
        env.new('d22', xt.Drift, length=1.0),
        env.new('qd2', xt.Quadrupole, length=0.3, k1=-0.7),
        env.new('d32', xt.Drift, length=1.0),
        env.new('b4', xt.Bend, length=3.0, angle=np.pi / 2),
        env.new('d42', xt.Drift, length=1.0),
    ])
    line.set_particle_ref('electron', p0c=1e9)
    line.configure_bend_model(core='full', edge=None)

    placements = []
    for index, row in enumerate(reference_rows):
        env.elements[f'TS{index}'] = xf.TouschekScattering()
        placements.append(env.place(f'TS{index}', at=float(row['s_m'])))
    line.insert(placements)

    twiss = line.twiss(method='4d')
    table = line.get_table()
    names = table.rows.match(element_type='TouschekScattering').name
    xfields_s = np.array([table['s', name] for name in names])
    elegant_s = np.array([float(row['s_m']) for row in reference_rows])
    np.testing.assert_allclose(xfields_s, elegant_s, atol=1e-12)

    lma = xt.Table({
        'name': names,
        's': xfields_s,
        'delta_neg': np.full(len(names), -0.012),
        'delta_pos': np.full(len(names), 0.012),
    })
    study = xf.TouschekStudy(
        line,
        twiss=twiss,
        local_momentum_acceptance=lma,
        local_momentum_acceptance_scale=0.85,
        nemitt_x=1e-5,
        nemitt_y=1e-7,
        sigma_z=4e-3,
        sigma_delta=1e-3,
        bunch_intensity=4e9,
        n_scattering_events=100_000,
        nx=3,
        ny=3,
        nz=3,
        weight_retention_fraction=0.99,
        seed=1997,
        method='4d',
    )
    study.initialise_touschek()
    return line, names


@pytest.fixture(scope='module')
def elegant_comparison():
    with (REFERENCE_DIR / 'reference.csv').open(newline='') as stream:
        reference_rows = list(csv.DictReader(stream))
    with (REFERENCE_DIR / 'reference_summary.json').open() as stream:
        reference_summary = json.load(stream)
    line, names = _build_reference_study(reference_rows)
    return line, names, reference_rows, reference_summary


def test_local_piwinski_rates_against_elegant(elegant_comparison):
    line, names, reference_rows, _ = elegant_comparison
    xfields_rates = np.array([line[name].piwinski_rate for name in names])
    elegant_rates = np.array([
        float(row['piwinski_local_rate_hz']) for row in reference_rows
    ])

    # ELEGANT records zero at the first, zero-length TSCATTER section. The
    # remaining local values agree to a few ppm for this fixed-aperture case.
    np.testing.assert_allclose(
        xfields_rates[1:], elegant_rates[1:], rtol=1e-5)


def test_total_rate_and_lifetime_against_elegant(elegant_comparison):
    line, names, _, reference_summary = elegant_comparison
    total_rate = sum(line[name].integrated_piwinski_rate for name in names)
    lifetime = 4e9 / total_rate

    assert total_rate == pytest.approx(
        reference_summary['total_piwinski_rate_hz'], rel=1e-5)
    assert lifetime == pytest.approx(
        reference_summary['touschek_lifetime_s'], rel=1e-5)
