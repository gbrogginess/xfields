# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo
import xtrack as xt
import numpy as np

class TouschekScattering(xt.BeamElement):

    _xofields = {
        'p0c': xo.Float64,
        'bunch_population': xo.Float64,
        'gemitt_x': xo.Float64,
        'gemitt_y': xo.Float64,
        'alfx': xo.Float64,
        'betx': xo.Float64,
        'alfy': xo.Float64,
        'bety': xo.Float64,
        'dx': xo.Float64,
        'dpx': xo.Float64,
        'dy': xo.Float64,
        'dpy': xo.Float64,
        'deltaN': xo.Float64,
        'deltaP': xo.Float64,
        'sigma_z': xo.Float64,
        'sigma_delta': xo.Float64,
        'n_simulated': xo.Int64,
        'nx': xo.Float64,
        'ny': xo.Float64,
        'nz': xo.Float64,
        'theta_min': xo.Float64,
        'theta_max': xo.Float64,
        'ignored_portion': xo.Float64,
        'integrated_piwinski_rate': xo.Float64,
        'seed': xo.Int64,
        'inhibit_permute': xo.Int64
    }

    # allow_track = False
    _depends_on = [xt.RandomUniformAccurate]

    _extra_c_sources = [
        '#include "xfields/beam_elements/touschek_src/touschek.h"'
    ]

    _per_particle_kernels = {
        '_scatter': xo.Kernel(
            c_name='TouschekScatter',
            args=[
                xo.Arg(xo.Float64, name='x_out', pointer=True),
                xo.Arg(xo.Float64, name='px_out', pointer=True),
                xo.Arg(xo.Float64, name='y_out', pointer=True),
                xo.Arg(xo.Float64, name='py_out', pointer=True),
                xo.Arg(xo.Float64, name='zeta_out', pointer=True),
                xo.Arg(xo.Float64, name='delta_out', pointer=True),
                xo.Arg(xo.Float64, name='theta_out', pointer=True),
                xo.Arg(xo.Float64, name='weight_out', pointer=True),
                xo.Arg(xo.Float64, name='totalMCRate_out', pointer=True),
                xo.Arg(xo.Int64,   name='n_selected_out', pointer=True),
            ],
        ),
    }

    def __init__(self, s=0.0,
                particle_ref=xt.Particles(),
                element_index=0,
                bunch_population=0.0,
                alfx=0.0, betx=0.0, alfy=0.0, bety=0.0,
                dx=0.0, dpx=0.0, dy=0.0, dpy=0.0,
                x_co=0.0, px_co=0.0, y_co=0.0, py_co=0.0,
                zeta_co=0.0, delta_co=0.0,
                deltaN=0.0, deltaP=0.0,
                gemitt_x=0.0, gemitt_y=0.0,
                sigma_z=0.0, sigma_delta=0.0,
                n_simulated=0, nx=0.0, ny=0.0, nz=0.0,
                theta_min=0.0, theta_max=0.0,
                piwinski_rate=0.0,
                ignored_portion=0.0,
                integrated_piwinski_rate=0.0,
                seed=1997,
                inhibit_permute=0,
                **kwargs):
        
        # This gives AttributeError: 'TouschekScattering' object has no attribute '_xobject'
        # if not isinstance(self._context, xo.ContextCpu) or self._context.openmp_enabled:
        #     raise ValueError('TouschekScattering only enabled on CPU.')

        if '_xobject' in kwargs.keys():
            self.xoinitialize(**kwargs)
            return
        
        super().__init__(**kwargs)

        self.s = s
        self.particle_ref = particle_ref
        self.element_index = element_index
        self.bunch_population = bunch_population
        self.alfx = alfx
        self.betx = betx
        self.alfy = alfy
        self.bety = bety
        self.dx = dx
        self.dpx = dpx
        self.dy = dy
        self.dpy = dpy
        self.x_co = x_co
        self.px_co = px_co
        self.y_co = y_co
        self.py_co = py_co
        self.zeta_co = zeta_co
        self.delta_co = delta_co
        self.deltaN = deltaN
        self.deltaP = deltaP
        self.gemitt_x = gemitt_x
        self.gemitt_y = gemitt_y
        self.sigma_z = sigma_z
        self.sigma_delta = sigma_delta
        self.n_simulated = n_simulated
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.ignored_portion = ignored_portion
        self.integrated_piwinski_rate = integrated_piwinski_rate
        self.piwinski_rate = piwinski_rate
        self.seed = seed
        self.inhibit_permute = inhibit_permute

    def _configure(self, **kwargs):
        config_allowed = {
            "s", "particle_ref", "element_index",
            "bunch_population",
            "gemitt_x", "gemitt_y",
            "alfx", "betx", "alfy", "bety",
            "dx", "dpx", "dy", "dpy",
            "x_co", "px_co", "y_co", "py_co",
            "zeta_co", "delta_co",
            "deltaN", "deltaP",
            "sigma_z", "sigma_delta",
            "n_simulated", "nx", "ny", "nz",
            "theta_min", "theta_max",
            "ignored_portion", "piwinski_rate",
            "integrated_piwinski_rate",
            "seed", "inhibit_permute"
        }

        unknown = set(kwargs) - config_allowed
        if unknown:
            bad = ", ".join(sorted(unknown))
            raise KeyError(f"Unsupported configure() keys: {bad}")
        
        for kk, vv in kwargs.items():
            setattr(self, kk, vv)
            if kk == "particle_ref":
                self.p0c = self.particle_ref.p0c[0]

    def scatter(self):
        context = self._context
        particles = xt.Particles(_context=context)

        if not particles._has_valid_rng_state():
            particles._init_random_number_generator()

        x_out      = context.zeros(shape=(self.n_simulated,), dtype=np.float64)
        px_out     = context.zeros(shape=(self.n_simulated,), dtype=np.float64)
        y_out      = context.zeros(shape=(self.n_simulated,), dtype=np.float64)
        py_out     = context.zeros(shape=(self.n_simulated,), dtype=np.float64)
        zeta_out   = context.zeros(shape=(self.n_simulated,), dtype=np.float64)
        delta_out  = context.zeros(shape=(self.n_simulated,), dtype=np.float64)
        theta_out  = context.zeros(shape=(self.n_simulated,), dtype=np.float64)
        weight_out = context.zeros(shape=(self.n_simulated,), dtype=np.float64)
        totalMCRate_out = context.zeros(shape=(1,), dtype=np.float64)
        n_selected_out  = context.zeros(shape=(1,), dtype=np.int64)

        self._scatter(particles=particles,
                      x_out=x_out, px_out=px_out,
                      y_out=y_out, py_out=py_out,
                      zeta_out=zeta_out, delta_out=delta_out,
                      theta_out=theta_out,
                      weight_out=weight_out,
                      totalMCRate_out=totalMCRate_out,
                      n_selected_out=n_selected_out)
        
        n = n_selected_out[0]
        # Create particle object for tracking
        # TODO: add at_element, start_tracking_at_element, ...
        part = xt.Particles(_capacity=2*n, 
                            p0c=self.p0c,
                            mass0=self.particle_ref.mass0,
                            q0=self.particle_ref.q0, 
                            pdg_id=self.particle_ref.pdg_id,
                            x=x_out[:n], px=px_out[:n],
                            y=y_out[:n], py=py_out[:n],
                            zeta=zeta_out[:n], delta=delta_out[:n],
                            weight=weight_out[:n],
                            s=getattr(self, '_s', 0.0))
        
        # Shift Touschek scattered particles around the closed orbit
        part.x[:n] += self.x_co
        part.px[:n] += self.px_co
        part.y[:n] += self.y_co
        part.py[:n] += self.py_co
        part.zeta[:n] += self.zeta_co

        delta_temp = part.delta.copy()
        delta_temp[:n] += self.delta_co
        part.update_delta(delta_temp)
        
        part.at_element = self.element_index
        
        part_ids = part.filter(part.state == 1).particle_id
        self.theta_log = dict(zip(part_ids.astype(int), theta_out[:n].astype(float)))

        self.total_mc_rate = totalMCRate_out[0]
        self.ignored_rate = self.ignored_portion * self.total_mc_rate

        return part

    def track(self, particles):
        super().track(particles)