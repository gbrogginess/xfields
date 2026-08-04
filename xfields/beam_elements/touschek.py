# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo
import xtrack as xt
import numpy as np
import warnings


def _resolve_weight_retention_fraction(
        *, weight_retention_fraction, ignored_portion, default):
    if ignored_portion is not None:
        if weight_retention_fraction is not None:
            raise ValueError(
                "Provide either `weight_retention_fraction` or "
                "`ignored_portion`, not both.")
        warnings.warn(
            "`ignored_portion` is deprecated. Use "
            "`weight_retention_fraction=1 - ignored_portion` instead.",
            FutureWarning,
            stacklevel=2)
        weight_retention_fraction = 1.0 - ignored_portion

    if weight_retention_fraction is None:
        weight_retention_fraction = default

    if not 0.0 < weight_retention_fraction <= 1.0:
        raise ValueError(
            "`weight_retention_fraction` must be in the interval (0, 1].")

    return float(weight_retention_fraction)


def _resolve_n_scattering_events(
        *, n_scattering_events, n_simulated, default):
    if n_simulated is not None:
        if n_scattering_events is not None:
            raise ValueError(
                "Provide either `n_scattering_events` or `n_simulated`, "
                "not both.")
        warnings.warn(
            "`n_simulated` is deprecated. Use `n_scattering_events` instead.",
            FutureWarning,
            stacklevel=2)
        n_scattering_events = n_simulated

    if n_scattering_events is None:
        n_scattering_events = default

    if n_scattering_events is None:
        raise ValueError("`n_scattering_events` is required.")

    if n_scattering_events < 0:
        raise ValueError("`n_scattering_events` must be non-negative.")

    return int(n_scattering_events)


def _resolve_bunch_intensity(*, bunch_intensity, default):
    if bunch_intensity is None:
        bunch_intensity = default

    if bunch_intensity is None:
        raise ValueError("`bunch_intensity` is required.")

    if bunch_intensity < 0:
        raise ValueError("`bunch_intensity` must be non-negative.")

    return float(bunch_intensity)


class TouschekScattering(xt.BeamElement):
    """
    Beam element that performs a Monte Carlo Touschek scattering simulation
    at a single location in a lattice.

    Each element represents one scattering center along the lattice.  When
    :meth:`scatter` is called it draws macro-particle pairs from the local
    6D phase-space (Gaussian) distribution, applies the Møller cross-section,
    boosts scattered pairs back to the lab frame, and returns the subset of
    macro-particles whose momentum deviation exceeds the local momentum
    acceptance (LMA).

    The element is *passive* during normal tracking (``track`` is a no-op);
    all physics happens inside :meth:`scatter`.

    The Monte Carlo kernel is implemented in C99 and follows the ELEGANT
    algorithm of Xiao & Borland (PRSTAB 13, 074201, 2010).

    Parameters
    ----------
    s : float, optional
        Longitudinal position of the element in the lattice [m].  Default 0.
    particle_ref : xtrack.Particles, optional
        Reference particle carrying.
    element_index : int, optional
        Index of this element in the line.
    bunch_intensity : float, optional
        Number of physical particles in one bunch.
    alfx, betx : float, optional
        Horizontal Twiss parameters at the element.
    alfy, bety : float, optional
        Vertical Twiss parameters at the element.
    dx, dpx : float, optional
        Horizontal dispersion and its derivative at the element.
    dy, dpy : float, optional
        Vertical dispersion and its derivative at the element.
    x_co, px_co : float, optional
        Horizontal closed-orbit position and normalised momentum at the
        element.
    y_co, py_co : float, optional
        Vertical closed-orbit position and normalised momentum at the 
        element.
    zeta_co, delta_co : float, optional
        Longitudinal closed-orbit coordinate and relative momentum
        deviation.
    delta_neg : float, optional
        Negative local momentum acceptance (scaled).
    delta_pos : float, optional
        Positive local momentum acceptance (scaled).
    gemitt_x : float, optional
        Horizontal geometric emittance [m·rad].
    gemitt_y : float, optional
        Vertical geometric emittance [m·rad].
    sigma_z : float, optional
        RMS bunch length [m].
    sigma_delta : float, optional
        RMS relative momentum spread.
    n_scattering_events : int, optional
        Number of Touschek scattering events to generate in the Monte Carlo
        loop. Larger values reduce statistical noise but increase CPU time.
    nx, ny, nz : float, optional
        Truncation of the Gaussian distribution in units of
        :math:`\\sqrt{\\varepsilon}` for the transverse planes and
        :math:`\\sigma` for the longitudinal plane.  The sampling window is
        :math:`\\pm n_x \\sqrt{\\varepsilon_x}`, etc.  ``nz`` may be
        reduced automatically by :class:`TouschekStudy` to prevent
        particles being drawn outside the LMA before scattering.
    theta_min, theta_max : float, optional
        Lower and upper limits of the centre-of-mass scattering angle
        :math:`\\theta^*` [rad].  In practice set to
        :math:`0.00005\\pi` and :math:`0.99995\\pi` to avoid the
        forward/backward divergence of the Møller cross-section.
    piwinski_rate : float, optional
        Local Piwinski scattering rate [Hz] evaluated at this element.
        Stored for diagnostics; not used in the Monte Carlo kernel.
    weight_retention_fraction : float, optional
        Fraction of the generated scattering weight to retain in the returned
        particle sample. The highest-weight particles are retained until their
        cumulative weight reaches approximately this fraction of the total
        generated weight. The default value of ``1.0`` keeps all generated
        particles. Values smaller than one reduce tracking cost by discarding
        the lowest-weight tail, at the price of a controlled downward
        truncation of the represented rate.
    integrated_piwinski_rate : float, optional
        Piwinski rate integrated (trapezoidal rule) over the lattice section
        preceding this element and divided by the line length to give the
        section contribution to the ring-averaged per-bunch rate [1/s].
        Set by :meth:`TouschekStudy.initialise_touschek`; used to weight
        the scattered macro-particles.
    seed : int, optional
        Seed for the ELEGANT-compatible 48-bit LCG random number generator.
        Using the same seed reproduces the ELEGANT Monte Carlo sequence
        exactly.  Default 1997.
    inhibit_permute : int, optional
        If non-zero, the random-order permutation step (``randomizeOrder``)
        is skipped.  Intended for reproducibility testing only.

    Attributes
    ----------
    piwinski_rate : float
        Local Piwinski scattering rate [Hz] at this element.
    total_mc_rate : float
        Total Monte Carlo scattering rate [Hz] returned by the last call to
        :meth:`scatter`.
    ignored_rate : float
        Scattering rate [Hz] associated with the low-weight particles
        discarded by ``pickPart`` in the last call to :meth:`scatter`,
        i.e. approximately ``(1 - weight_retention_fraction)`` times
        ``total_mc_rate``.
    theta_log : dict
        Mapping ``{particle_id: theta}`` of centre-of-mass scattering
        angles [rad] for the particles returned by the last call to
        :meth:`scatter`.

    Notes
    -----
    **Physics summary**

    The Monte Carlo loop follows Xiao & Borland (PRSTAB 13, 074201, 2010):

    1. Two particles are drawn from the local 6-D Gaussian distribution
       using ``selectPartGauss`` with a truncated range of
       :math:`\\pm n \\sqrt{\\varepsilon}`.
    2. The pair is boosted to the centre-of-mass (CM) frame
       (``bunch2cm``).
    3. A scattering angle :math:`\\theta^*` is drawn uniformly in
       :math:`[\\theta_{\\min},\\,\\theta_{\\max}]` and a random azimuthal
       angle :math:`\\phi` is drawn uniformly in :math:`[0,\\,\\pi]`.
    4. The Møller cross-section ``moeller`` is evaluated at
       :math:`\\theta^*`.
    5. Scattered momenta are rotated (``eulertrans``) and boosted back to
       the lab frame (``cm2bunch``).
    6. A particle is selected for tracking only if its resulting
       :math:`\\delta` falls outside the LMA:
       :math:`\\delta < \\delta_N` or :math:`\\delta > \\delta_P`.
    7. ``pickPart`` retains only the highest-weight particles whose
        cumulative weight reaches approximately
        ``weight_retention_fraction`` of the total simulated weight.
        The remaining low-weight events are discarded.

    Macro-particle weights are normalised so that
    :math:`\\sum_i w_i` equals the per-turn loss rate in the
    corresponding lattice section (in particles/turn).

    References
    ----------
    .. [1] A. Xiao and M. Borland, "Monte Carlo simulation of Touschek
       effect", Phys. Rev. ST Accel. Beams **13**, 074201 (2010).
       https://doi.org/10.1103/PhysRevSTAB.13.074201
    .. [2] M. Borland, "elegant: A Flexible SDDS-Compliant Code for
       Accelerator Simulation", APS LS-287 (2000).
    """

    _xofields = {
        'p0c': xo.Float64,
        'bunch_intensity': xo.Float64,
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
        'delta_neg': xo.Float64,
        'delta_pos': xo.Float64,
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
                bunch_intensity=None,
                alfx=0.0, betx=0.0, alfy=0.0, bety=0.0,
                dx=0.0, dpx=0.0, dy=0.0, dpy=0.0,
                x_co=0.0, px_co=0.0, y_co=0.0, py_co=0.0,
                zeta_co=0.0, delta_co=0.0,
                delta_neg=0.0, delta_pos=0.0,
                gemitt_x=0.0, gemitt_y=0.0,
                sigma_z=0.0, sigma_delta=0.0,
                n_scattering_events=None, n_simulated=None,
                nx=0.0, ny=0.0, nz=0.0,
                theta_min=0.0, theta_max=0.0,
                piwinski_rate=0.0,
                weight_retention_fraction=None,
                ignored_portion=None,
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
        self.bunch_intensity = _resolve_bunch_intensity(
            bunch_intensity=bunch_intensity,
            default=0.0)
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
        self.delta_neg = delta_neg
        self.delta_pos = delta_pos
        self.gemitt_x = gemitt_x
        self.gemitt_y = gemitt_y
        self.sigma_z = sigma_z
        self.sigma_delta = sigma_delta
        self.n_scattering_events = _resolve_n_scattering_events(
            n_scattering_events=n_scattering_events,
            n_simulated=n_simulated,
            default=0)
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.weight_retention_fraction = _resolve_weight_retention_fraction(
            weight_retention_fraction=weight_retention_fraction,
            ignored_portion=ignored_portion,
            default=1.0)
        self.integrated_piwinski_rate = integrated_piwinski_rate
        self.piwinski_rate = piwinski_rate
        self.seed = seed
        self.inhibit_permute = inhibit_permute

    @property
    def weight_retention_fraction(self):
        return 1.0 - self.ignored_portion

    @weight_retention_fraction.setter
    def weight_retention_fraction(self, value):
        if not 0.0 < value <= 1.0:
            raise ValueError(
                "`weight_retention_fraction` must be in the interval (0, 1].")
        self.ignored_portion = 1.0 - float(value)

    @property
    def n_scattering_events(self):
        return self.n_simulated

    @n_scattering_events.setter
    def n_scattering_events(self, value):
        if value < 0:
            raise ValueError("`n_scattering_events` must be non-negative.")
        self.n_simulated = int(value)

    def _configure(self, **kwargs):
        config_allowed = {
            "s", "particle_ref", "element_index",
            "bunch_intensity",
            "gemitt_x", "gemitt_y",
            "alfx", "betx", "alfy", "bety",
            "dx", "dpx", "dy", "dpy",
            "x_co", "px_co", "y_co", "py_co",
            "zeta_co", "delta_co",
            "delta_neg", "delta_pos",
            "sigma_z", "sigma_delta",
            "n_scattering_events", "n_simulated", "nx", "ny", "nz",
            "theta_min", "theta_max",
            "weight_retention_fraction", "ignored_portion", "piwinski_rate",
            "integrated_piwinski_rate",
            "seed", "inhibit_permute"
        }

        unknown = set(kwargs) - config_allowed
        if unknown:
            bad = ", ".join(sorted(unknown))
            raise KeyError(f"Unsupported configure() keys: {bad}")
        
        ignored_portion = kwargs.pop("ignored_portion", None)
        weight_retention_fraction = kwargs.pop(
            "weight_retention_fraction", None)
        if (ignored_portion is not None
                or weight_retention_fraction is not None):
            self.weight_retention_fraction = _resolve_weight_retention_fraction(
                weight_retention_fraction=weight_retention_fraction,
                ignored_portion=ignored_portion,
                default=self.weight_retention_fraction)

        n_simulated = kwargs.pop("n_simulated", None)
        n_scattering_events = kwargs.pop("n_scattering_events", None)
        if n_simulated is not None or n_scattering_events is not None:
            self.n_scattering_events = _resolve_n_scattering_events(
                n_scattering_events=n_scattering_events,
                n_simulated=n_simulated,
                default=self.n_scattering_events)

        bunch_intensity = kwargs.pop("bunch_intensity", None)
        if bunch_intensity is not None:
            self.bunch_intensity = _resolve_bunch_intensity(
                bunch_intensity=bunch_intensity,
                default=self.bunch_intensity)

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
                            s=getattr(self, 's', 0.0))
        
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
        self.ignored_rate = (
            1.0 - self.weight_retention_fraction) * self.total_mc_rate

        return part

    def track(self, particles):
        super().track(particles)
