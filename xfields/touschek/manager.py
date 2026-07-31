# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xtrack as xt
import xfields as xf

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import i0
from scipy.constants import physical_constants

ELECTRON_MASS_EV = xt.ELECTRON_MASS_EV
C_LIGHT_VACUUM = physical_constants['speed of light in vacuum'][0]
CLASSICAL_ELECTRON_RADIUS = physical_constants['classical electron radius'][0]

class TouschekCalculator:
    '''
    Internal helper that evaluates Piwinski scattering rates and integrates
    them along the lattice.

    This class is instantiated and owned by :class:`TouschekManager`; it
    should not be constructed directly by users.

    Parameters
    ----------
    manager : TouschekManager
        The owning manager, from which beam parameters, optics, and the
        local momentum acceptance table are read.

    Attributes
    ----------
    twiss : xtrack.TwissTable or None
        Twiss table for the ring.  Injected by
        :meth:`TouschekManager.initialise_touschek` before any rate
        evaluation is performed.

    References
    ----------
    .. [1] A. Piwinski, "The Touschek Effect in Strong Focusing Storage
       Rings", arXiv:physics/9903034 (1999).
    .. [2] A. Xiao and M. Borland, "Monte Carlo simulation of Touschek
       effect", Phys. Rev. ST Accel. Beams **13**, 074201 (2010).
       https://doi.org/10.1103/PhysRevSTAB.13.074201
    .. [3] M. Borland, "elegant: A Flexible SDDS-Compliant Code for
       Accelerator Simulation", APS LS-287 (2000).
    '''
    def __init__(self, manager):
        self.manager = manager
        self.twiss = None

    def _compute_piwinski_integral(self, tm, B1, B2):
        """
        Compute Piwinski integral for Touschek scattering rate calculation.
        """
        from math import atan, tan, sqrt, exp, log, pi

        km = atan(sqrt(tm))

        def int_piwinski(k):
            t = np.tan(k) ** 2
            fact = (
                (2*t + 1)**2 * (t/tm / (1+t) - 1) / t + t - sqrt(t*tm * (1 + t))
                - (2 + 1 / (2*t)) * log(t/tm / (1+t))
            )
            if B2 * t < 500:
                intp = fact * exp(-B1*t) * i0(B2*t) * sqrt(1+t)
            else:
                intp = (
                    fact
                    * exp(B2*t - B1*t)
                    / sqrt(2*pi * B2*t)
                    * sqrt(1+t)
                )
            return intp

        val, _ =  quad(
            int_piwinski,
            km,
            pi / 2,
            epsabs=1e-16,
            epsrel=1e-12
        )

        return val

    def _compute_piwinski_scattering_rate(self, element):
        """
        Compute Piwinski Touschek scattering rate.
        """
        p0c = self.manager.particle_ref.p0c[0]
        bunch_population = self.manager.bunch_population
        local_momentum_acceptance = self.manager.local_momentum_acceptance
        gemitt_x = self.manager.gemitt_x
        gemitt_y = self.manager.gemitt_y
        twiss = self.twiss
        alfx = twiss['alfx', element]
        betx = twiss['betx', element]
        alfy = twiss['alfy', element]
        bety = twiss['bety', element]
        sigma_z = self.manager.sigma_z
        sigma_delta = self.manager.sigma_delta
        delta = twiss['delta', element]
        dx = twiss['dx', element]
        dpx = twiss['dpx', element]
        dxt = alfx * dx + betx * dpx # dxt: dx tilde
        dy = twiss['dy', element]
        dpy = twiss['dpy', element]
        dyt = alfy * dy + bety * dpy # dyt: dy tilde

        try:
            s = self.twiss.rows[element].s[0]
        except:
            s = self.manager.line.get_s_position(element)

        deltaN = np.interp(s, local_momentum_acceptance.s, local_momentum_acceptance.deltan)
        deltaP = np.interp(s, local_momentum_acceptance.s, local_momentum_acceptance.deltap)

        sigmab_x = np.sqrt(gemitt_x * betx) # Horizontal betatron beam size
        sigma_x = np.sqrt(gemitt_x * betx + dx**2 * sigma_delta**2) # Horizontal beam size

        sigmab_y = np.sqrt(gemitt_y * bety) # Vertical betatron beam size
        sigma_y = np.sqrt(gemitt_y * bety + dy**2 * sigma_delta**2) # Vertical beam size

        sigma_h = (sigma_delta**-2 + (dx**2 + dxt**2)/sigmab_x**2 + (dy**2 + dyt**2)/sigmab_y**2)**(-0.5)

        p = p0c * (1 + delta)
        gamma = np.sqrt(1 + p**2 / ELECTRON_MASS_EV**2)
        beta = np.sqrt(1 - gamma**-2)

        B1 = betx**2 / (2 * beta**2 * gamma**2 * sigmab_x**2) * (1 - sigma_h**2 * dxt**2 / sigmab_x**2) \
             + bety**2 / (2 * beta**2 * gamma**2 * sigmab_y**2) * (1 - sigma_h**2 * dyt**2 / sigmab_y**2)

        B2 = np.sqrt(B1**2 - betx**2 * bety**2 * sigma_h**2 / (beta**4 * gamma**4 * sigmab_x**4 * sigmab_y**4 * sigma_delta**2) \
                             * (sigma_x**2 * sigma_y**2 - sigma_delta**4 * dx**2 * dy**2))

        tmN = beta**2 * (deltaN**2)
        tmP = beta**2 * (deltaP**2)

        piwinski_integralN = self._compute_piwinski_integral(tmN, B1, B2)
        piwinski_integralP = self._compute_piwinski_integral(tmP, B1, B2)

        rateN = CLASSICAL_ELECTRON_RADIUS**2 * C_LIGHT_VACUUM * bunch_population**2 \
                / (8*np.pi * gamma**2 * sigma_z * np.sqrt(sigma_x**2 * sigma_y**2 - sigma_delta**4 * dx**2 * dy**2)) \
                * 2 * np.sqrt(np.pi * (B1**2 - B2**2)) * piwinski_integralN

        rateP = CLASSICAL_ELECTRON_RADIUS**2 * C_LIGHT_VACUUM * bunch_population**2 \
                / (8*np.pi * gamma**2 * sigma_z * np.sqrt(sigma_x**2 * sigma_y**2 - sigma_delta**4 * dx**2 * dy**2)) \
                * 2 * np.sqrt(np.pi * (B1**2 - B2**2)) * piwinski_integralP

        rate = (rateN + rateP) / 2

        return rate

    def _compute_integrated_piwinski_rates(self, element):
        """
        Integrate the Piwinski Touschek scattering rate along the line using
        the trapezoidal rule, between successive TouschekScattering elements.

        For each TouschekScattering element, the method stores the integrated
        rate per bunch over the preceding section of the line. This per-bunch
        rate is later used to assign the correct weights to Touschek-scattered
        particles at the corresponding element.
        """
        def _get_s(name):
            try:
                return tab.rows[name].s[0]
            except (KeyError, AttributeError, IndexError, TypeError):
                return self.manager.line.get_s_position(name)
            
        def _step(name, s_before, rate_before, integrated):
            s = _get_s(name)
            ds = s - s_before
            if ds > 0.0:
                rate = self._compute_piwinski_scattering_rate(name)
                integrated += 0.5 * (rate_before + rate) * ds
                return s, rate, integrated
            else:
                return s_before, rate_before, integrated

        line = self.manager.line
        tab = line.get_table()
        t_rev0 = float(self.twiss.t_rev0)

        # Indexes of the TouschekScatterings
        ii_t = [ii for ii, nn in enumerate(tab.name[:-1]) if isinstance(line[nn], xf.TouschekScattering)]

        integrated = 0.0

        if element is None:
            ii_current = 0
            s0 = 0.0
            r0 = self._compute_piwinski_scattering_rate(tab.name[0])
        else:
            import re
            ii_current = int(re.search(r'\d+', element).group())
            tscatter_before = tab.name[ii_t[ii_current - 1]] if ii_current != 0 else tab.name[0]
            s0 = _get_s(tscatter_before)
            r0 = self._compute_piwinski_scattering_rate(tscatter_before)

        s_before = s0
        rate_before = r0

        if element is None:
            # Configure all the TouschekScattering elements
            for ii, nn in enumerate(tab.name):
                s_before, rate_before, integrated = _step(nn, s_before, rate_before, integrated)

                if ii == ii_t[ii_current]:
                    # divide by c and by t_rev0 --> per-bunch rate
                    integrated_piwinski_rate = integrated / C_LIGHT_VACUUM / t_rev0
                    elem = line[nn] # xf.TouschekScattering
                    # print(f'Integrated Piwinski rate at {nn}: {integrated_piwinski_rate*1e-3} [kHz]')
                    elem._configure(integrated_piwinski_rate=integrated_piwinski_rate)
                    integrated = 0.0
                    ii_current += 1
                    if ii_current == len(ii_t):
                        break
        else:
            # Configure only the TouschekScattering element named `element`
            subtab = tab.rows[tscatter_before:element]
            for nn in subtab.name:
                s_before, rate_before, integrated = _step(nn, s_before, rate_before, integrated)

                if nn == element:
                    # divide by c and by t_rev0 --> per-bunch rate
                    integrated_piwinski_rate = integrated / C_LIGHT_VACUUM / t_rev0
                    elem = line[nn] # xf.TouschekScattering
                    # print(f'Integrated Piwinski rate at {nn}: {integrated_piwinski_rate*1e-3} [kHz]')
                    elem._configure(integrated_piwinski_rate=integrated_piwinski_rate)
                    break


class TouschekManager:
    '''
    High-level manager that orchestrates a full Touschek scattering simulation.

    The manager:

    1. Computes the Piwinski scattering rate at every
       :class:`TouschekScattering` element in the line using the local beam
       optics, emittances, and the local momentum acceptance (LMA).
    2. Integrates those rates over each lattice section between consecutive
       scattering elements (trapezoidal rule) to obtain the per-bunch,
       per-turn loss probability in each section.
    3. Configures each :class:`TouschekScattering` element with all local
       parameters so that it can generate and weight Monte Carlo
       macro-particles.

    Parameters
    ----------
    line : xtrack.Line
        The accelerator lattice.  Must contain at least one
        :class:`TouschekScattering` element and a ``particle_ref``.
    twiss : xtrack.TwissTable or None, optional
        Pre-computed Twiss table.  If ``None``, it is computed internally
        by :meth:`initialise_touschek` using the ``method`` keyword
        argument (default ``"6d"``).
    local_momentum_acceptance : xtrack.Table
        Table returned by ``line.get_local_momentum_acceptance()``.  Must
        contain the columns ``name``, ``s``, ``deltan`` (negative LMA,
        :math:`\\delta_N < 0`), and ``deltap`` (positive LMA,
        :math:`\\delta_P > 0`).  Values are scaled in-place by
        ``local_momentum_acceptance_scale`` upon construction.
    nemitt_x : float, optional
        Horizontal normalised emittance [m·rad].  Mutually exclusive with
        ``gemitt_x``.
    nemitt_y : float, optional
        Vertical normalised emittance [m·rad].  Mutually exclusive with
        ``gemitt_y``.
    gemitt_x : float, optional
        Horizontal geometric emittance [m·rad].  Mutually exclusive with
        ``nemitt_x``.
    gemitt_y : float, optional
        Vertical geometric emittance [m·rad].  Mutually exclusive with
        ``nemitt_y``.
    sigma_z : float
        RMS bunch length [m].
    sigma_delta : float
        RMS relative momentum spread :math:`\\sigma_\\delta`.
    bunch_population : float
        Number of real particles per bunch, :math:`N_b`.
    n_simulated : int
        Number of scattered macro-particle candidates to generate per
        scattering element.  Larger values improve statistics at the cost
        of CPU time.  Values of :math:`10^6`–:math:`10^7` are typical.
    nx : float, optional
        Truncation of the transverse-horizontal Gaussian sampling window in
        units of :math:`\\sqrt{\\varepsilon_x}`.  Default 3.
    ny : float, optional
        Truncation of the transverse-vertical Gaussian sampling window in
        units of :math:`\\sqrt{\\varepsilon_y}`.  Default 3.
    nz : float, optional
        Truncation of the longitudinal Gaussian sampling window in units of
        :math:`\\sigma_\\zeta` and :math:`\\sigma_\\delta`.  May be reduced 
        element-by-element (see Notes) to prevent drawing initial particles 
        outside the LMA.
        Default 3.
    local_momentum_acceptance_scale : float, optional
        Multiplicative safety factor applied to the LMA on construction.
        A value of 0.85 (default) ensures that particles with momentum 
        deviation smaller than the LMA but with nonzero betatron amplitude
        are accepted. This avoids that particles that are eventually lost
        are not considered for tracking
    ignored_portion : float, optional
            Fraction of the total simulated scattering weight that is discarded
            before tracking.  Only the highest-weight particles whose cumulative
            weight reaches ``(1 - ignored_portion)`` of the total are retained
            and tracked; the remaining low-weight, low-probability events are
            dropped.  The default value of ``0.01`` retains 99 % of the total
            weight while significantly reducing the number of particles that
            must be tracked, providing a good accuracy–efficiency trade-off.
            Setting this to ``0`` keeps all simulated particles.
    seed : int, optional
        RNG seed for the ELEGANT-compatible 48-bit LCG generator used
        inside :class:`TouschekScattering`.  Default 1997.
    method : str, optional
        Twiss method forwarded to ``line.twiss()`` when ``twiss`` is
        ``None``.  Accepted values: ``"4d"``, ``"6d"``.  Default ``"6d"``.

    Attributes
    ----------
    line : xtrack.Line
        The accelerator lattice passed at construction.
    particle_ref : xtrack.Particles
        Reference particle extracted from ``line.particle_ref``.
    twiss : xtrack.TwissTable or None
        Twiss table (populated by :meth:`initialise_touschek` if not
        provided at construction).
    gemitt_x, gemitt_y : float
        Geometric emittances [m·rad] derived from the input normalised or
        geometric emittances.
    local_momentum_acceptance : xtrack.Table
        The (scaled) LMA table.
    sigma_z, sigma_delta : float
        Bunch length [m] and momentum spread.
    bunch_population : float
        Bunch population :math:`N_b`.
    n_simulated : int
        Number of simulated scattered candidates per element.
    nx, ny, nz : float
        Phase-space sampling truncation parameters.
    seed : int
        RNG seed.
    touschek : TouschekCalculator
        The calculator instance used internally to evaluate Piwinski rates.

    References
    ----------
    .. [1] A. Xiao and M. Borland, "Monte Carlo simulation of Touschek
       effect", Phys. Rev. ST Accel. Beams **13**, 074201 (2010).
       https://doi.org/10.1103/PhysRevSTAB.13.074201
    .. [2] M. Borland, "elegant: A Flexible SDDS-Compliant Code for
       Accelerator Simulation", APS LS-287 (2000).
    '''
    def __init__(self, line=None, twiss=None, local_momentum_acceptance=None,
                 nemitt_x=None, nemitt_y=None,
                 sigma_z=None, sigma_delta=None, bunch_population=None,
                 n_simulated=None, gemitt_x=None, gemitt_y=None,
                 local_momentum_acceptance_scale=0.85, ignored_portion=0.01,
                 seed=1997, nx=3, ny=3, nz=3, **kwargs):

        # Input validation
        if line is None:
            raise ValueError("`line` is required.")
        if not hasattr(line, "particle_ref"):
            raise ValueError("`line` must have a `particle_ref`.")
        if local_momentum_acceptance is None:
            raise ValueError("`local_momentum_acceptance` is required.")
        if sigma_z is None:
            raise ValueError("`sigma_z` is required.")
        if sigma_delta is None:
            raise ValueError("`sigma_delta` is required.")
        if bunch_population is None:
            raise ValueError("`bunch_population` is required.")
        if n_simulated is None:
            raise ValueError("`n_simulated` is required.")

        # Local momentum acceptnace validation
        required_cols = {"name", "s", "deltan", "deltap"}
        if not isinstance(local_momentum_acceptance, xt.Table):
            raise TypeError("`local_momentum_acceptance` must be an `xt.Table` object.")
        missing = required_cols - set(local_momentum_acceptance._col_names)
        if missing:
            raise ValueError(f"`local_momentum_acceptance` missing columns: {sorted(missing)}")

        for col in ("s", "deltan", "deltap"):
            try:
                vals = np.asarray(local_momentum_acceptance[col], dtype=float)
            except Exception:
                raise TypeError(f"`{col}` column must be numeric (cannot coerce to float).")
            nan_mask = np.isnan(vals)
            if nan_mask.any():
                bad = list(np.where(nan_mask)[0][:5])
                raise ValueError(f"`{col}` contains NaN at indices {bad}.")
            inf_mask = np.isinf(vals)
            if inf_mask.any():
                bad = list(np.where(inf_mask)[0][:5])
                raise ValueError(f"`{col}` contains inf at indices {bad}.")

        self.line = line
        self.particle_ref = line.particle_ref
        self.twiss = twiss

        # Check that the line contains TouschekScatterings
        tab = line.get_table()
        try:
            has = "TouschekScattering" in set(np.unique(tab.element_type))
        except Exception:
            has = "TouschekScattering" in set(getattr(tab, "element_type", []))
        if not has:
            raise ValueError("The line does not contain any TouschekScattering. "
                             "Please add them before initializing the TouschekManager.")

        # Local momentum acceptance
        local_momentum_acceptance.deltan *= local_momentum_acceptance_scale
        local_momentum_acceptance.deltap *= local_momentum_acceptance_scale
        self.local_momentum_acceptance = local_momentum_acceptance

        self.sigma_z = sigma_z
        self.sigma_delta = sigma_delta
        self.bunch_population = bunch_population
        self.n_simulated = n_simulated
        self.ignored_portion = ignored_portion
        self.seed = seed
        self.nx = nx
        self.ny = ny
        self.nz = nz

        # Limits from ELEGANT
        self._theta_min = 0.00005*np.pi
        self._theta_max = 0.99995*np.pi

        # Emittance validation
        nemitt_given = nemitt_x is not None and nemitt_y is not None
        gemitt_given = gemitt_x is not None and gemitt_y is not None

        if nemitt_given and gemitt_given:
            raise ValueError("Provide either normalized emittances (nemitt_x, nemitt_y) "
                             "OR geometric emittances (gemitt_x, gemitt_y), not both.")
        if not (nemitt_given or gemitt_given):
            raise ValueError("You must provide either both normalized emittances (nemitt_x, nemitt_y) "
                             "OR both geometric emittances (gemitt_x, gemitt_y).")

        if nemitt_given:
            beta0 = line.particle_ref.beta0[0]
            gamma0 = line.particle_ref.gamma0[0]
            self.gemitt_x = nemitt_x / (beta0 * gamma0)
            self.gemitt_y = nemitt_y / (beta0 * gamma0)
        else:
            self.gemitt_x = gemitt_x
            self.gemitt_y = gemitt_y

        self.kwargs = kwargs

        self.touschek = TouschekCalculator(self)

    def initialise_touschek(self, element=None):
        '''
        Compute and configure all Piwinski rates in the lattice.

        For each :class:`TouschekScattering` element this method:

        1. Evaluates the local Piwinski scattering rate using the Twiss
        parameters, beam emittances, and the local momentum acceptance.
        2. Integrates the rate over the preceding lattice section using the
        trapezoidal rule to obtain the per-bunch, per-turn loss probability.
        3. Stores the integrated rate and all local optics parameters on the
        element via :meth:`TouschekScattering._configure` so that
        :meth:`TouschekScattering.scatter` can weight the Monte Carlo
        macro-particles correctly.

        Parameters
        ----------
        element : str or None, optional
            If ``None`` (default), all :class:`TouschekScattering` elements in
            the line are initialised in one pass.  If a string, only the named
            element is (re-)initialised; the Piwinski rate is integrated only
            over the lattice section between that element and the preceding
            scattering centre.
        '''
        line = self.line
        tab = line.get_table()

        local_momentum_acceptance = self.local_momentum_acceptance

        if self.twiss is None:
            twiss_method = self.kwargs.get("method", "6d")
            self.twiss = self.line.twiss(method=twiss_method)

        # Pass the twiss to the TouschekCalculator
        self.touschek.twiss = self.twiss

        import time
        t0 = time.time()
        self.touschek._compute_integrated_piwinski_rates(element)
        print(f"Computed integrated piwinski rates in {time.time() - t0:.2f} s.")

        # Helper to config all fields to a single TouschekScattering
        def _config(nn):
            try:
                s = tab.rows[nn].s[0]
            except Exception:
                s = self.line.get_s_position(nn)

            twiss = self.twiss
            alfx = twiss["alfx", nn]; betx = twiss["betx", nn]
            alfy = twiss["alfy", nn]; bety = twiss["bety", nn]
            dx   = twiss["dx",   nn]; dpx = twiss["dpx",  nn]
            dy   = twiss["dy",   nn]; dpy = twiss["dpy",  nn]
            dN = np.interp(s, local_momentum_acceptance.s, local_momentum_acceptance.deltan)
            dP = np.interp(s, local_momentum_acceptance.s, local_momentum_acceptance.deltap)

            x_co = twiss["x", nn]; px_co = twiss["px", nn]
            y_co = twiss["y", nn]; py_co = twiss["py", nn]
            zeta_co = twiss["zeta", nn]; delta_co = twiss["delta", nn]

            # Adjust the effective longitudinal sampling cutoff (nz_eff) to prevent
            # generation of initial particles that are already outside
            # the local momentum aperture (LMA) before Touschek scattering.
            #
            # Background:
            # In the Touschek Monte Carlo routine, initial particle coordinates
            # are drawn from a truncated Gaussian distribution with cutoffs
            # {nx, ny, nz} in the transverse and longitudinal planes. The longitudinal
            # cutoff nz sets the maximum |δ| ≈ nz*σδ. If nz*σδ exceeds the local
            # momentum acceptance at this location, some particles
            # are sampled outside the LMA (|δ| > LMA) even before any scattering event occurs.
            #
            # These particles create a pathological situation:
            #   • For very small scattering angles (θ* --> 0 in the CM frame),
            #     such particles are flagged as "candidates for loss" and passed to tracking,
            #     even though their state is essentially unchanged by scattering.
            #   • Because the Møller differential cross-section diverges at
            #     θ* --> 0, the corresponding particle weights become extremely large.
            #   • The pickPart routine then tends to select a handful of these
            #     high-weight pathological particles, distorting both the local
            #     scattering rate (RMC/RP diverges) and the overall Touschek
            #     lifetime estimate.
            #
            # Mitigation:
            # To eliminate these spurious contributions, we dynamically reduce
            # the longitudinal cutoff at each TouschekScattering element:
            #
            #     nz_eff = min(nz, 0.85 * min(|δN|, δP) / σδ)
            #
            # where δN, δP are the negative/positive momentum aperture limits
            # (scaled by local_momentum_acceptance_scale). This ensures that the sampled
            # longitudinal range ±nz_eff*σδ always lies strictly inside the local
            # momentum aperture, with a small safety factor (0.85). As a result,
            # only pathological large-weight events are avoided, and the Monte Carlo
            # rate remains consistent with the Piwinski formula.
            #
            # NOTE: nz_eff is determined independently at each scattering element,
            # so tighter cutoffs are applied only where the local momentum aperture
            # is restrictive, while wider cutoffs are retained elsewhere.
            min_dNdP = min(abs(dN), dP)
            nz_eff = min(self.nz, 0.85 * min_dNdP / self.sigma_delta)

            if nz_eff < self.nz:
                print(f"""
            ***********************************************************************************************
            [TouschekManager] Warning: longitudinal cutoff reduced at element '{nn}' (s={s:.2f} m).

            Using nz_eff={nz_eff:.2f} instead of nz={self.nz:.2f}.
            This ensures that particles are sampled strictly within the local momentum aperture.
            ***********************************************************************************************
            """)

            piwinski_rate = self.touschek._compute_piwinski_scattering_rate(nn)

            elem = line[nn] # xf.TouschekScattering
            element_index = line.element_names.index(nn)

            elem._configure(
                s=s,
                particle_ref=self.particle_ref,
                element_index=element_index,
                bunch_population=self.bunch_population,
                gemitt_x=self.gemitt_x,
                gemitt_y=self.gemitt_y,
                alfx=alfx, betx=betx,
                alfy=alfy, bety=bety,
                dx=dx, dpx=dpx,
                dy=dy, dpy=dpy,
                x_co=x_co, px_co=px_co,
                y_co=y_co, py_co=py_co,
                zeta_co=zeta_co, delta_co=delta_co,
                deltaN=dN, deltaP=dP,
                sigma_z=self.sigma_z,
                sigma_delta=self.sigma_delta,
                n_simulated=self.n_simulated,
                nx=self.nx, ny=self.ny, nz=nz_eff,
                theta_min=self._theta_min, theta_max=self._theta_max,
                ignored_portion=self.ignored_portion,
                piwinski_rate=piwinski_rate,
                seed=self.seed, inhibit_permute=0
            )

        if element is None:
            for nn in tab.name[:-1]: # Avoid the last tab.name which is _end_point
                if isinstance(line[nn], xf.TouschekScattering):
                    print(f'Initialising TouschekScattering for {nn}')
                    _config(nn)
        else:
            if not isinstance(element, str):
                raise TypeError(f"`element` must be a string (got {type(element).__name__}).")
            if element not in set(tab.name):
                raise ValueError(
                    f"`element='{element}'` is not present in the line provided to the TouschekManager."
                )
            if not isinstance(line[element], xf.TouschekScattering):
                raise TypeError(
                    f"`line['{element}']` is not a TouschekScattering (got {type(line[element]).__name__})."
                )
            print(f'Initialising TouschekScattering for {element}')
            _config(element)