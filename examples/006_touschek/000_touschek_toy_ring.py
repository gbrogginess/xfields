# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #
import numpy as np
import matplotlib.pyplot as plt

import xobjects as xo
import xtrack as xt
import xfields as xf

######################################################
# Beam parameters
######################################################
nemitt_x = 1e-5
nemitt_y = 1e-7

sigma_z = 4e-3
sigma_delta = 1e-3

bunch_intensity = 4e9

######################################################
# Build a toy ring
######################################################
lbend = 3
angle = np.pi / 2

lquad = 0.3
k1qf = 0.1
k1qd = 0.7

# Create environment
env = xt.Environment()

# Define the line (toy ring)
line = env.new_line(components=[
    env.new('mqf.1', xt.Quadrupole, length=lquad, k1=k1qf),
    env.new('d1.1',  xt.Drift, length=1),
    env.new('mb1.1', xt.Bend, length=lbend, angle=angle),
    env.new('d2.1',  xt.Drift, length=1),

    env.new('mqd.1', xt.Quadrupole, length=lquad, k1=-k1qd),
    env.new('d3.1',  xt.Drift, length=1),
    env.new('mb2.1', xt.Bend, length=lbend, angle=angle),
    env.new('d4.1',  xt.Drift, length=1),

    env.new('mqf.2', xt.Quadrupole, length=lquad, k1=k1qf),
    env.new('d1.2',  xt.Drift, length=1),
    env.new('mb1.2', xt.Bend, length=lbend, angle=angle),
    env.new('d2.2',  xt.Drift, length=1),

    env.new('mqd.2', xt.Quadrupole, length=lquad, k1=-k1qd),
    env.new('d3.2',  xt.Drift, length=1),
    env.new('mb2.2', xt.Bend, length=lbend, angle=angle),
    env.new('d4.2',  xt.Drift, length=1),
])

# Set the reference particle
line.set_particle_ref('electron', p0c=1e9)

# Configure the bend model
line.configure_bend_model(core='full', edge=None)

######################################################
# Insert Touschek scattering centers
######################################################
# We insert Touschek scattering centers in the middle of each magnet
# to have good coverage of variations of the optical functions
tab = line.get_table()
tab_bends_quads = tab.rows[(tab.element_type == 'Bend') | (tab.element_type == 'Quadrupole')]

placements = []
for ii, nn in enumerate(tab_bends_quads.name):
    tscatter_name = f'TScatter.{ii}'
    env.elements[tscatter_name] = xf.TouschekScattering()
    placements.append(env.place(tscatter_name, at=0.0, from_=nn))

# The last TouschekScattering element has to be placed at the end of the line
tscatter_name = f'TScatter.{ii+1}'
env.elements[tscatter_name] = xf.TouschekScattering()
placements.append(env.place(tscatter_name, at=tab.s[-1]))

line.insert(placements)

######################################################
# Install apertures
######################################################
tab = line.get_table()
needs_aperture = tab.rows.match_not(element_type='Drift.*|Marker|').name

aper_size = 0.040 # m

placements = []
for nn in needs_aperture:
    env.new(
        f'{nn}_aper_entry', xt.LimitRect,
        min_x=-aper_size, max_x=aper_size,
        min_y=-aper_size, max_y=aper_size
    )
    placements.append(env.place(f'{nn}_aper_entry', at=f'{nn}@start'))

    env.new(
        f'{nn}_aper_exit', xt.LimitRect,
        min_x=-aper_size, max_x=aper_size,
        min_y=-aper_size, max_y=aper_size
    )
    placements.append(env.place(f'{nn}_aper_exit', at=f'{nn}@end'))

line.insert(placements)

######################################################
# Evaluate local momentum acceptance profile
######################################################
# Evaluate local momentum aperture at the touschek scattering centers
tab = line.get_table()
elements = tab.rows.match(element_type="TouschekScattering").name
lma = line.get_local_momentum_acceptance(
    # twiss=tw,
    elements=elements,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    y_offset=1e-9,
    delta_negative_limit=-0.012,
    delta_positive_limit=0.012,
    delta_step_size=1e-4,
    n_turns=1000,
    method="4d"
)

######################################################
# Plot
######################################################
plt.plot(lma.s, lma.deltan*100, c='r')
plt.plot(lma.s, lma.deltap*100, c='r')
plt.title('Toy ring: local momentum acceptance profile')
plt.xlabel('s [m]')
plt.ylabel(r'$\delta$ [%]')
plt.grid()
plt.show()

######################################################
# Touschek simulation
######################################################
# Parameters
local_momentum_acceptance_scale = 0.85 # scaling factor for local momentum acceptance
n_scattering_events = int(5e6) # number of simulated scattering events with delta > delta_min
nturns = 1000 # number of turns to simulate

touschek = xf.TouschekStudy(
    line,
    local_momentum_acceptance=lma,
    local_momentum_acceptance_scale=local_momentum_acceptance_scale,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    sigma_delta=sigma_delta,
    bunch_intensity=bunch_intensity,
    n_scattering_events=n_scattering_events,
    nx=3, ny=3, nz=3,
    weight_retention_fraction=0.99,
    seed=1997,
    method='4d'
)

# Initialise the Touschek simulation.
# Computes the integrated Piwinski scattering rate over each lattice section
# between consecutive TouschekScattering elements (using the trapezoidal rule),
# and configures each element with its local optics, beam parameters, and
# momentum acceptance. The integrated rate is later used to assign the
# correct weights to the Touschek-scattered macro-particles.
touschek.initialise_touschek()

# Build a CPU tracker with OpenMP multithreading to speed up tracking
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

result = touschek.run(
    track=True,
    n_turns=nturns,
    with_progress=1,
)

# Optional: Refine loss location to improve loss map accuracy
loss_loc_refinement = xt.LossLocationRefinement(line,
    n_theta = 360, # Angular resolution in the polygonal approximation of the aperture
    r_max = 0.5,   # Maximum transverse aperture in m
    dr = 50e-6,    # Transverse loss refinement accuracy [m]
    ds = 0.1,      # Longitudinal loss refinement accuracy [m]
    )

loss_loc_refinement.refine_loss_location(result.particles)

######################################################
# Compute Touschek lifetime
######################################################
# Keep lost particles only
particles = result.particles.filter(result.particles.state == 0)
touschek_lifetime = result.lifetime

######################################################
# Plot: Toy ring Touschek loss map
######################################################
circumference = line.get_length()
binwidth = 0.1 # m

plt.title(f'Toy ring Touschek loss map (Touschek lifetime: {touschek_lifetime/60:.2f} min)')
plt.hist(particles.s, bins=np.arange(0, circumference + binwidth, binwidth), weights=particles.weight*1e-3)
plt.xlabel('s [m]')
plt.ylabel('Loss rate [kHz]')
plt.grid()
plt.show()
