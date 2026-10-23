# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

from dataclasses import dataclass

import xtrack as xt


@dataclass
class RadiativeBhabhaResult:
    """
    Result of a radiative Bhabha simulation.

    It is returned by :meth:`xfields.RadiativeBhabhaGenerator.analyse` and
    :meth:`xfields.RadiativeBhabhaGenerator.run`. The rates are per bunch,
    from the weights of the particles.

    Parameters
    ----------
    element_names : list
        Names of the radiative Bhabha markers included in the result.
    local_rates : xtrack.Table
        Per-marker table. It always contains ``name``, ``s`` and
        ``delta_cut``. When the particles of each marker are known, it also
        contains ``num_particles`` and ``sum_weight`` and, once tracked,
        ``num_lost_particles`` and ``sum_lost_weight``.
    rate_generated : float
        Total weight of the generated particles [1/s]: the rate of the
        photon emissions beyond the momentum cut, at the interaction points
        of the markers. They are not all lost.
    rate_tracking : float or None
        Radiative Bhabha loss rate: weighted rate of the particles lost in
        tracking [1/s]. ``None`` if the particles were not tracked.
    lifetime_tracking : float or None
        Radiative Bhabha lifetime, inferred from the loss rate
        ``rate_tracking`` [s]. ``None`` if the particles were not tracked.
    tracked : bool
        Whether the particles were tracked.
    particles_by_element : dict or None
        Mapping ``{element_name: xtrack.Particles}`` with the particles of
        each marker, when kept.
    particles : xtrack.Particles or None
        Merged particles, when kept.
    lost_particles : xtrack.Particles or None
        Subset of ``particles`` lost in tracking, when kept and if any is lost.
    rate_tracking_error : float or None
        Monte Carlo standard error of ``rate_tracking`` [1/s], from the spread
        between the replicas of each marker. ``None`` unless several replicas
        of some marker are given.
    lifetime_tracking_error : float or None
        Monte Carlo standard error of ``lifetime_tracking`` [s], propagated
        from ``rate_tracking_error`` to first order.
    """
    element_names: list
    local_rates: xt.Table
    rate_generated: float
    rate_tracking: float | None
    lifetime_tracking: float | None
    tracked: bool
    particles_by_element: dict | None = None
    particles: xt.Particles | None = None
    lost_particles: xt.Particles | None = None
    rate_tracking_error: float | None = None
    lifetime_tracking_error: float | None = None
