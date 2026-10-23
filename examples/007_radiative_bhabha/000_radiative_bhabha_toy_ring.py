# copyright ############################### #
# This file is part of the Xfields Package. #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt

######################################################
# Beam parameters
######################################################
nemitt_x = 1e-5
nemitt_y = 1e-7

sigma_z = 4e-3
sigma_delta = 1e-3

bunch_intensity = 4e10

######################################################
# Load a toy ring
######################################################
line = xt.load('../../test_data/four_cell_ring/line.json')
env = line.env

######################################################
# Install radiative Bhabha markers at the interaction points
######################################################
# The toy ring has no interaction point: two drifts play their role
interaction_points = ['d2.1', 'd4.2']
line.xfields.bhabha.install_markers(at=interaction_points)

######################################################
# Install apertures
######################################################
tab = line.get_table()
needs_aperture = tab.rows.match_not(
    element_type='Drift.*|Marker|RadiativeBhabhaMarker|').name

aper_size = 0.040

env.new('aper', xt.LimitRect,
        min_x=-aper_size, max_x=aper_size,
        min_y=-aper_size, max_y=aper_size)

placements = []
for nn in needs_aperture:
    env.new(f'{nn}_aper_entry', 'aper')
    env.new(f'{nn}_aper_exit', 'aper')
    placements.append(env.place(f'{nn}_aper_entry', at=f'{nn}@start'))
    placements.append(env.place(f'{nn}_aper_exit', at=f'{nn}@end'))

line.insert(placements)

######################################################
# Configure the radiative Bhabha markers
######################################################
nturns = 100

# Stores on each marker the local optics and closed orbit, the beam
# parameters, the crossing geometry and the settings of the generator. The
# configured line can be saved here (e.g. with line.to_json): once loaded
# again, it generates the same particles.
line.xfields.bhabha.configure(
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    sigma_delta=sigma_delta,
    bunch_intensity=bunch_intensity,
    phi=0.,                # head-on collisions
    # Bunch particles sent through each interaction point, and the factor
    # scaling up their emission probability: keep the mean number of photons
    # per particle well below 1
    n_generated=100_000,
    compt_scale=1e10,
    # Keep only the particles that lost more than 0.5% of their momentum:
    # well inside the momentum acceptance, the others cannot be lost
    delta_cut=-0.005,
    seed=1997,
    method='4d',
)

######################################################
# Generate and track the scattered particles
######################################################
# For each marker:
#   1. Generate the particles of one bunch that emitted a radiative Bhabha
#      photon at its interaction point, beyond the momentum cut
#   2. Track them around the ring for `nturns` turns, starting and ending at
#      the marker
particles_list = []
for name in line.xfields.bhabha.marker_names:
    particles = line.xfields.bhabha.generate(at=name)
    print(f'{name}: {line[name].mean_photons_per_particle:.3f} photons per '
          f'particle, luminosity {line[name].luminosity*1e-4:.3e} cm^-2 per '
          'bunch crossing')
    line.track(particles, ele_start=name, ele_stop=name, num_turns=nturns,
               with_progress=10)
    particles_list.append(particles)

# The weights are rates [1/s per bunch]: merged, the samples of all the
# markers represent the radiative Bhabha scattering at all the interaction
# points
particles = xt.Particles.merge(particles_list)

######################################################
# Radiative Bhabha loss rate and lifetime
######################################################
result = line.xfields.bhabha.analyse(particles)

# All of the above in one call:
# result = line.xfields.bhabha.run(n_turns=nturns)

print(f'Emission rate beyond the cut: {result.rate_generated*1e-3:.3f} kHz')
print(f'Radiative Bhabha loss rate:   {result.rate_tracking*1e-3:.3f} kHz')
print(f'Radiative Bhabha lifetime:    {result.lifetime_tracking/3600:.2f} h')

######################################################
# Plot: Toy ring radiative Bhabha loss map
######################################################
lost_particles = particles.filter(particles.state == 0) # lost on an aperture

circumference = line.get_length()
binwidth = 0.1

tab = line.get_table()
for name in line.xfields.bhabha.marker_names:
    plt.axvline(tab['s', name], color='k', ls='--')
plt.title(
    f'Toy ring radiative Bhabha loss map '
    f'(lifetime: {result.lifetime_tracking/3600:.2f} h)'
)
plt.hist(
    lost_particles.s,
    bins=np.arange(0, circumference + binwidth, binwidth),
    weights=lost_particles.weight*1e-3,
)
plt.xlabel('s [m]')
plt.ylabel('Loss rate [kHz]')
plt.grid()
plt.show()
