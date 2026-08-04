# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #
"""Compare xfields with the frozen ELEGANT Touschek reference.

This script only reads committed CSV/JSON reference data. It does not run or
require ELEGANT or SDDS.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import xfields as xf
import xtrack as xt


REFERENCE_DIR = (
    Path(__file__).resolve().parents[2]
    / 'test_data' / 'touschek' / 'elegant_reference'
)

LOCAL_RATE_RTOL = 0.11
TOTAL_RATE_RTOL = 0.05
LIFETIME_RTOL = 0.05

NEMITT_X = 1e-5
NEMITT_Y = 1e-7
SIGMA_Z = 4e-3
SIGMA_DELTA = 1e-3
BUNCH_INTENSITY = 4e9


######################################################
# Load the frozen ELEGANT scalar/table outputs
######################################################
with (REFERENCE_DIR / 'reference.csv').open(newline='') as stream:
    elegant_rows = list(csv.DictReader(stream))

with (REFERENCE_DIR / 'reference_summary.json').open() as stream:
    elegant_summary = json.load(stream)

elegant_s = np.array([float(row['s_m']) for row in elegant_rows])
elegant_local_rates = np.array([
    float(row['piwinski_local_rate_hz']) for row in elegant_rows
])
elegant_total_rate = elegant_summary['total_piwinski_rate_hz']
elegant_lifetime = elegant_summary['touschek_lifetime_s']


######################################################
# Build the same xfields toy ring used by the test
######################################################
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

table = line.get_table()
magnets = table.rows[
    (table.element_type == 'Bend')
    | (table.element_type == 'Quadrupole')
]
placements = []
for index, name in enumerate(magnets.name):
    env.elements[f'TS{index}'] = xf.TouschekScattering()
    placements.append(env.place(f'TS{index}', at=0.0, from_=name))
env.elements['TS8'] = xf.TouschekScattering()
placements.append(env.place('TS8', at=table.s[-1]))
line.insert(placements)

twiss = line.twiss(method='4d')
table = line.get_table()
touschek_names = table.rows.match(element_type='TouschekScattering').name
xfields_s = np.array([table['s', name] for name in touschek_names])

# Use the same fixed +/-1.2% aperture as the frozen ELEGANT case. This avoids
# dynamic-aperture tracking and isolates the Touschek rate comparison.
lma = xt.Table({
    'name': touschek_names,
    's': xfields_s,
    'delta_neg': np.full(len(touschek_names), -0.012),
    'delta_pos': np.full(len(touschek_names), 0.012),
})

touschek = xf.TouschekStudy(
    line,
    twiss=twiss,
    local_momentum_acceptance=lma,
    local_momentum_acceptance_scale=0.85,
    nemitt_x=NEMITT_X,
    nemitt_y=NEMITT_Y,
    sigma_z=SIGMA_Z,
    sigma_delta=SIGMA_DELTA,
    bunch_intensity=BUNCH_INTENSITY,
    n_scattering_events=100_000,
    nx=3,
    ny=3,
    nz=3,
    weight_retention_fraction=0.99,
    seed=1997,
    method='4d',
)
touschek.initialise_touschek()

xfields_local_rates = np.array([
    line[name].piwinski_rate for name in touschek_names
])
xfields_total_rate = sum(
    line[name].integrated_piwinski_rate for name in touschek_names
)
xfields_lifetime = BUNCH_INTENSITY / xfields_total_rate


######################################################
# Apply the same comparisons and tolerances as pytest
######################################################
# ELEGANT reports zero at the first zero-length TSCATTER section, so the local
# comparison starts at TS1. Both codes' TS0 values remain visible in the plot.
np.testing.assert_allclose(
    xfields_local_rates[1:],
    elegant_local_rates[1:],
    rtol=LOCAL_RATE_RTOL,
)
np.testing.assert_allclose(
    xfields_total_rate,
    elegant_total_rate,
    rtol=TOTAL_RATE_RTOL,
)
np.testing.assert_allclose(
    xfields_lifetime,
    elegant_lifetime,
    rtol=LIFETIME_RTOL,
)

print('Frozen ELEGANT comparison passed')
print(f'  total rate: xfields={xfields_total_rate:.6f} Hz, '
      f'ELEGANT={elegant_total_rate:.6f} Hz')
print(f'  lifetime:   xfields={xfields_lifetime:.6f} s, '
      f'ELEGANT={elegant_lifetime:.6f} s')


######################################################
# Plot the quantities asserted above
######################################################
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.plot(
    elegant_s,
    elegant_local_rates * 1e-3,
    'o-',
    label='ELEGANT (frozen)',
)
ax.plot(
    xfields_s,
    xfields_local_rates * 1e-3,
    's--',
    label='xfields',
)
ax.set_xlabel('s [m]')
ax.set_ylabel('Local Piwinski rate [kHz]')
ax.set_title(f'Local rates (asserted at TS1-TS8, rtol={LOCAL_RATE_RTOL:.0%})')
ax.grid(alpha=0.3)
ax.legend()

labels = ['ELEGANT\n(frozen)', 'xfields']
colors = ['tab:blue', 'tab:orange']

ax = axes[1]
ax.bar(
    labels,
    np.array([elegant_total_rate, xfields_total_rate]) * 1e-3,
    color=colors,
)
ax.set_ylabel('Total Piwinski rate [kHz]')
ax.set_title(f'Total rate (rtol={TOTAL_RATE_RTOL:.0%})')
ax.grid(axis='y', alpha=0.3)

ax = axes[2]
ax.bar(
    labels,
    np.array([elegant_lifetime, xfields_lifetime]) / 60,
    color=colors,
)
ax.set_ylabel('Touschek lifetime [min]')
ax.set_title(f'Lifetime (rtol={LIFETIME_RTOL:.0%})')
ax.grid(axis='y', alpha=0.3)

fig.suptitle('xfields comparison with frozen ELEGANT Touschek reference')
fig.tight_layout()
plt.show()
