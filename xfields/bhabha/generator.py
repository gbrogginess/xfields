# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import numpy as np

import xtrack as xt

from ..beam_elements.radiative_bhabha import RadiativeBhabhaMarker
from ..touschek.generator import (
    _free_names, _geometric_emittances, _is_free, _is_sequence, _per_marker,
    _resolve_seed, _stream_seed)
from ..touschek.analysis import (
    _lost, _replica_variance, _sample_sums, _tracked)
from .analysis import RadiativeBhabhaResult


def _check_configured(name, marker, particle_ref):
    """
    Check that a marker is configured for the reference particle of the line.
    """
    if not marker.p0c > 0:
        raise ValueError(
            f"RadiativeBhabhaMarker '{name}' is not configured: call "
            "`line.xfields.bhabha.configure(...)` first.")
    p0c = particle_ref.p0c[0]
    if abs(marker.p0c - p0c) > 1e-10 * p0c:
        raise ValueError(
            f"RadiativeBhabhaMarker '{name}' is configured for p0c = "
            f"{marker.p0c:.10g} eV, but the reference particle of the line "
            f"has p0c = {p0c:.10g} eV: configure the markers again.")


def _check_positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(f"`{name}` must be a positive integer.")


class RadiativeBhabhaGenerator:
    """
    Radiative Bhabha simulation of a line, based on the
    :class:`xfields.RadiativeBhabhaMarker` elements installed at its
    interaction points.

    An instance bound to a line is available as ``line.xfields.bhabha``. The
    generator holds no state other than the line: the configuration is
    stored in the markers, so that a line saved to JSON after it has been
    configured generates the same particles once loaded again.

    At each marker, the particles of one bunch cross the opposing bunch with
    the radiative Bhabha generator of :class:`xfields.BeamBeamBiGaussian3D`.
    The particles that emitted a photon, and are beyond the configured
    momentum cut, are kept with the rate they represent as weight: tracked
    from the marker, the weight of the lost ones gives the loss rate, the
    lifetime and the loss locations.

    Parameters
    ----------
    line : xtrack.Line
        Line containing, or receiving, the radiative Bhabha markers.

    Examples
    --------
    .. code-block:: python

        bhabha = line.xfields.bhabha
        bhabha.install_markers(at=['ip.1', 'ip.2'])
        bhabha.configure(nemitt_x=7e-5, nemitt_y=1.4e-7, sigma_z=15e-3,
                         sigma_delta=1e-3, bunch_intensity=2e11, phi=15e-3,
                         compt_scale=1e6, n_generated=200_000,
                         delta_cut=-0.006, seed=1997)

        particles_list = []
        for name in bhabha.marker_names:
            part = bhabha.generate(at=name)
            line.track(part, ele_start=name, ele_stop=name, num_turns=500)
            particles_list.append(part)

        result = bhabha.analyse(xt.Particles.merge(particles_list))
        print(result.lifetime_tracking)
    """

    def __init__(self, line):
        if not isinstance(line, xt.Line):
            raise TypeError(
                f"`line` must be an xtrack.Line (got {type(line).__name__}).")
        self._line = line

    @property
    def line(self):
        """
        Line of the radiative Bhabha simulation.

        Returns
        -------
        line : xtrack.Line
            Line bound to this generator.
        """
        return self._line

    @property
    def marker_names(self):
        """
        Names of the radiative Bhabha markers of the line.

        Returns
        -------
        names : list of str
            Names of the :class:`xfields.RadiativeBhabhaMarker` elements, in
            line order.
        """
        element_dict = self.line.element_dict
        names = [nn for nn in self.line.element_names
                 if isinstance(element_dict[nn], RadiativeBhabhaMarker)]
        if len(set(names)) != len(names):
            repeated = sorted({nn for nn in names if names.count(nn) > 1})
            raise ValueError(
                f"Radiative Bhabha markers {repeated} appear more than once "
                "in the line: each marker must be a distinct element.")
        return names

    def install_markers(self, at, *, from_=None, anchor=None,
                        from_anchor=None, names=None, prefix='bhabha'):
        """
        Install radiative Bhabha markers in the line, at its interaction
        points.

        Each marker is a new :class:`xfields.RadiativeBhabhaMarker` element,
        placed with ``env.place`` and inserted with ``line.insert``: ``at``,
        ``from_``, ``anchor`` and ``from_anchor`` have the same meaning as
        there. Installing the markers by hand in the same way is equivalent.

        Parameters
        ----------
        at : float, str, xdeps reference, or sequence of them
            Position of each marker, evaluated at installation: an ``s``
            position [m], an expression of the environment variables, the
            name of an element (e.g. the marker of the interaction point), or
            a reference ``'name@anchor'`` with ``anchor`` one of ``'start'``,
            ``'center'`` and ``'end'``. With ``from_``, the offset from the
            reference element.
        from_ : str or sequence of str, optional
            Element from which ``at`` is measured.
        anchor : {'start', 'center', 'end'} or sequence of them, optional
            Anchor of the marker placed at ``at``.
        from_anchor : {'start', 'center', 'end'} or sequence of them, optional
            Anchor of ``from_`` from which ``at`` is measured.
        names : str or sequence of str, optional
            Names of the new markers. By default ``f'{prefix}.{i}'``, with the
            lowest indices whose names are free in the environment.
        prefix : str, optional
            Prefix of the default names.

        Returns
        -------
        names : list of str
            Names of the installed markers.
        """
        line = self.line
        env = line.env

        at = list(at) if _is_sequence(at) else [at]
        n_markers = len(at)
        if n_markers == 0:
            raise ValueError("`at` must give at least one position.")
        from_ = _per_marker(from_, n_markers, 'from_')
        anchor = _per_marker(anchor, n_markers, 'anchor')
        from_anchor = _per_marker(from_anchor, n_markers, 'from_anchor')

        if names is None:
            names = _free_names(env, prefix, n_markers)
        else:
            names = _per_marker(
                [names] if isinstance(names, str) else names,
                n_markers, 'names')
            if len(set(names)) != len(names):
                raise ValueError("`names` contains repeated names.")
            for nn in names:
                if not _is_free(env, nn):
                    raise ValueError(
                        f"The name '{nn}' is already used in the environment.")

        places = []
        for nn, aa, ff, an, fa in zip(names, at, from_, anchor, from_anchor):
            env.elements[nn] = RadiativeBhabhaMarker()
            places.append(env.place(
                nn, at=aa, from_=ff, anchor=an, from_anchor=fa))
        line.insert(places)

        return list(names)

    def configure(self, *, bunch_intensity=None, sigma_z=None,
                  sigma_delta=None, phi=None, compt_scale=None,
                  n_generated=None, nemitt_x=None, nemitt_y=None,
                  gemitt_x=None, gemitt_y=None, alpha=0.0, other_beam_q0=None,
                  delta_cut=0.0, n_slices=100, compt_x_min=1e-4,
                  flag_beamsize_effect=1, seed=None, twiss=None, method='6d',
                  at=None, **kwargs):
        """
        Configure the radiative Bhabha markers of the line.

        Stores on each marker the local optics and closed orbit, the beam
        parameters, the crossing geometry and the settings of the generator.
        The two beams are taken as symmetric: the opposing bunch has the same
        intensity, emittances, bunch length and optics at the interaction
        point.

        The whole configuration is stored on the markers: a line saved to
        JSON after this call generates the same particles once loaded again.

        Parameters
        ----------
        bunch_intensity : float
            Number of particles in one bunch.
        sigma_z : float
            RMS bunch length [m], including the lengthening by beamstrahlung
            if relevant.
        sigma_delta : float
            RMS relative momentum spread.
        phi : float
            Half crossing angle [rad].
        compt_scale : float
            Factor scaling up the emission probability of every particle, so
            that enough photons are emitted by ``n_generated`` particles. It
            does not bias the rates, as long as the mean number of photons
            per particle stays well below 1 (a warning is issued at the
            generation above 0.1).
        n_generated : int
            Number of bunch particles sent through the interaction point at
            each generation, at each marker.
        nemitt_x, nemitt_y : float, optional
            Normalized emittances. Mutually exclusive with ``gemitt_x`` and
            ``gemitt_y``.
        gemitt_x, gemitt_y : float, optional
            Geometric emittances. Mutually exclusive with ``nemitt_x`` and
            ``nemitt_y``.
        alpha : float, optional
            Angle of the crossing plane [rad]. Default ``0`` (horizontal).
        other_beam_q0 : float or None, optional
            Charge of the particles of the opposing beam [e]. If ``None``
            (default), opposite to the charge of the reference particle.
        delta_cut : float, optional
            Only the particles with ``delta - delta_co < delta_cut`` after
            the emission are kept. A cut well inside the momentum acceptance
            (e.g. half of it) only removes particles that cannot be lost, and
            saves their tracking. Default ``0``: all the particles that
            emitted a photon below the closed-orbit momentum.
        n_slices : int, optional
            Number of longitudinal slices of the opposing bunch.
        compt_x_min : float, optional
            Low energy cut of the virtual photon spectrum, in units of
            ``gamma^-2``.
        flag_beamsize_effect : int, optional
            Beam-size effect on the cross section: 1 (default) on, 0 off.
        seed : int or None, optional
            Seed of the random-number streams of the markers. If ``None``, a
            seed is drawn now and stored, so that the generation stays
            reproducible.
        twiss : xtrack.TwissTable or None, optional
            Twiss table of the line. If ``None``, it is computed with
            ``line.twiss(method=method, **kwargs)``.
        method : {'6d', '4d'}, optional
            Twiss method, used if ``twiss`` is ``None``.
        at : str, sequence of str, or None, optional
            Markers to configure. If ``None`` (default), all the markers.
        **kwargs
            Keyword arguments forwarded to ``line.twiss`` if ``twiss`` is
            ``None``.

        Returns
        -------
        None
        """
        line = self.line
        if getattr(line, "particle_ref", None) is None:
            raise ValueError("`line` must have a `particle_ref`.")
        for name, value in (
                ("bunch_intensity", bunch_intensity),
                ("sigma_z", sigma_z),
                ("sigma_delta", sigma_delta),
                ("phi", phi),
                ("compt_scale", compt_scale),
                ("n_generated", n_generated)):
            if value is None:
                raise ValueError(f"`{name}` is required.")
        if bunch_intensity < 0:
            raise ValueError("`bunch_intensity` must be non-negative.")
        if not compt_scale > 0:
            raise ValueError("`compt_scale` must be positive.")
        _check_positive_integer(n_generated, 'n_generated')
        _check_positive_integer(n_slices, 'n_slices')

        particle_ref = line.particle_ref
        gemitt_x, gemitt_y = _geometric_emittances(
            particle_ref, nemitt_x=nemitt_x, nemitt_y=nemitt_y,
            gemitt_x=gemitt_x, gemitt_y=gemitt_y)
        if other_beam_q0 is None:
            other_beam_q0 = -particle_ref.q0
        seed = _resolve_seed(seed)

        # Check the markers before the (long) twiss
        names = self._markers(at)

        if twiss is None:
            twiss = line.twiss(method=method, **kwargs)

        for nn in names:
            line.element_dict[nn]._configure(
                p0c=particle_ref.p0c[0],
                bunch_intensity=bunch_intensity,
                gemitt_x=gemitt_x, gemitt_y=gemitt_y,
                sigma_z=sigma_z, sigma_delta=sigma_delta,
                alfx=twiss['alfx', nn], betx=twiss['betx', nn],
                alfy=twiss['alfy', nn], bety=twiss['bety', nn],
                dx=twiss['dx', nn], dpx=twiss['dpx', nn],
                dy=twiss['dy', nn], dpy=twiss['dpy', nn],
                x_co=twiss['x', nn], px_co=twiss['px', nn],
                y_co=twiss['y', nn], py_co=twiss['py', nn],
                zeta_co=twiss['zeta', nn], delta_co=twiss['delta', nn],
                t_rev0=twiss.t_rev0,
                phi=phi, alpha=alpha, other_beam_q0=other_beam_q0,
                n_slices=n_slices, compt_x_min=compt_x_min,
                compt_scale=compt_scale,
                flag_beamsize_effect=flag_beamsize_effect,
                n_generated=n_generated, delta_cut=delta_cut, seed=seed)

    def generate(self, at=None, *, replica=0, n_replicas=1, seed=None):
        """
        Generate radiative Bhabha scattered particles at the markers.

        At each marker, ``n_generated`` particles of a Gaussian bunch cross
        the opposing bunch with the radiative Bhabha generator. The particles
        that emitted a photon and have ``delta - delta_co < delta_cut`` are
        kept. Each carries as weight the rate it represents [1/s per bunch],
        divided by ``n_replicas``: the weights of a marker sum to the rate of
        the emissions beyond the cut at its interaction point.

        The particles of a marker are tracked from the marker itself, e.g.
        ``line.track(part, ele_start=name, ele_stop=name, num_turns=...)``,
        and the samples of different markers or replicas, generated in the
        same or in different jobs, can then be merged with
        :meth:`xtrack.Particles.merge` and given to :meth:`analyse`.

        Parameters
        ----------
        at : str, sequence of str, or None, optional
            Markers at which particles are generated. If ``None`` (default),
            all the markers of the line.
        replica : int, optional
            Index of the replica: replicas of a marker are independent
            samples, e.g. generated in different jobs. Default ``0``.
        n_replicas : int, optional
            Number of replicas of each marker that are merged. The weights are
            divided by it. Default ``1``.
        seed : int or None, optional
            Seed of the random-number streams. If ``None`` (default), the seed
            stored on each marker by :meth:`configure`. The stream of each
            marker and replica is derived from the seed, the name of the
            marker and the replica, so it does not depend on the other
            markers that are generated.

        Returns
        -------
        particles : xtrack.Particles or dict
            If ``at`` is a string, the particles generated at that marker.
            Otherwise, a mapping ``{name: xtrack.Particles}`` in line order.
        """
        line = self.line
        names = self._markers(at)

        _check_positive_integer(n_replicas, 'n_replicas')
        if (isinstance(replica, (bool, np.bool_))
                or not isinstance(replica, (int, np.integer))
                or not 0 <= replica < n_replicas):
            raise ValueError(
                f"`replica` must be an integer in [0, n_replicas) "
                f"(got {replica} with n_replicas={n_replicas}).")
        if seed is not None:
            seed = _resolve_seed(seed)

        particle_ref = line.particle_ref
        markers = {nn: line.element_dict[nn] for nn in names}
        for nn, marker in markers.items():
            _check_configured(nn, marker, particle_ref)

        tab = line.get_table()
        element_index = {nn: ii for ii, nn in enumerate(line.element_names)}

        particles = {}
        for nn, marker in markers.items():
            try:
                particles[nn] = marker._generate(
                    particle_ref=particle_ref,
                    at_element=element_index[nn],
                    s=tab['s', nn],
                    seed=_stream_seed(marker.seed if seed is None else seed,
                                      nn, replica),
                    weight_scale=1.0 / n_replicas,
                )
            except ValueError as error:
                raise ValueError(
                    f"RadiativeBhabhaMarker '{nn}': {error}") from error

        if isinstance(at, str):
            return particles[at]
        return particles

    def analyse(self, particles=None, *, at=None):
        """
        Analyse the tracked radiative Bhabha particles.

        The rates are the total weights: of the generated particles (the rate
        of the emissions beyond the cut) and of the lost ones (any
        ``state <= 0``, e.g. on an aperture or in a collimator). Secondaries
        produced while tracking carry the weight of their parent and are not
        counted.

        ``particles`` can be the sample of one marker, the particles of several
        markers merged with :meth:`xtrack.Particles.merge`, possibly coming
        from separate jobs, or the samples of each marker separately. The
        per-marker rates are only available for the samples given separately,
        or for the sample of a single marker selected with ``at``. The Monte
        Carlo error of the loss rate is estimated from the spread between the
        replicas of the markers given separately.

        Parameters
        ----------
        particles : xtrack.Particles or dict
            An ``xtrack.Particles`` object, or a mapping ``{name: particles}``
            from marker names to the ``xtrack.Particles`` of the marker or a
            list of its replicas.
        at : str, sequence of str, or None, optional
            Markers whose particles are given, when ``particles`` is not a
            mapping. If ``None`` (default), all the markers of the line.

        Returns
        -------
        result : RadiativeBhabhaResult
            Rates and, with tracked particles, the loss rate and lifetime.
        """
        line = self.line

        if particles is None:
            raise ValueError("`particles` is required.")
        samples = None
        if isinstance(particles, dict):
            if at is not None:
                raise ValueError(
                    "`at` cannot be given with a mapping of particles: its "
                    "keys select the markers.")
            names = self._markers(list(particles))
            samples = {}
            for nn in names:
                replicas = particles[nn]
                if not isinstance(replicas, (list, tuple)):
                    replicas = [replicas]
                if len(replicas) == 0:
                    raise ValueError(f"No particles given for marker '{nn}'.")
                samples[nn] = list(replicas)
            all_particles = [pp for vv in samples.values() for pp in vv]
        else:
            names = self._markers(at)
            all_particles = [particles]
        for pp in all_particles:
            if not isinstance(pp, xt.Particles):
                raise TypeError(
                    "The particles must be `xtrack.Particles` objects "
                    f"(got {type(pp).__name__}).")

        markers = {nn: line.element_dict[nn] for nn in names}
        for nn, marker in markers.items():
            _check_configured(nn, marker, line.particle_ref)
        bunch_intensities = {float(mm.bunch_intensity)
                             for mm in markers.values()}
        if len(bunch_intensities) > 1:
            raise ValueError(
                "The markers are configured with different bunch "
                f"intensities: {sorted(bunch_intensities)}.")
        bunch_intensity = bunch_intensities.pop()

        tab = line.get_table()
        data = {
            'name': np.array(names),
            's': np.array([float(tab['s', nn]) for nn in names]),
            'delta_cut': np.array([mm.delta_cut for mm in markers.values()]),
        }

        tracked = any(_tracked(pp) for pp in all_particles)
        rate_tracking = None
        rate_tracking_error = None

        if samples is not None:
            sums = {nn: [_sample_sums(pp) for pp in samples[nn]]
                    for nn in names}

            def per_marker(key):
                return np.array([sum(ss[key] for ss in sums[nn])
                                 for nn in names])

            data['n_replicas'] = np.array([len(samples[nn]) for nn in names])
            data['num_particles'] = per_marker('num_particles')
            data['sum_weight'] = per_marker('sum_weight')
            rate_generated = float(np.sum(data['sum_weight']))
            if tracked:
                data['num_lost_particles'] = per_marker('num_lost_particles')
                data['sum_lost_weight'] = per_marker('sum_lost_weight')
                variance = np.array([
                    _replica_variance(
                        [ss['sum_lost_weight'] for ss in sums[nn]])
                    for nn in names])
                data['sum_lost_weight_error'] = np.sqrt(variance)
                rate_tracking = float(np.sum(data['sum_lost_weight']))
                if np.any(np.isfinite(variance)):
                    rate_tracking_error = float(np.sqrt(np.nansum(variance)))
        else:
            sums = _sample_sums(particles)
            rate_generated = sums['sum_weight']
            if tracked:
                rate_tracking = sums['sum_lost_weight']
            if len(names) == 1:
                data['num_particles'] = np.array([sums['num_particles']])
                data['sum_weight'] = np.array([sums['sum_weight']])
                if tracked:
                    data['num_lost_particles'] = np.array(
                        [sums['num_lost_particles']])
                    data['sum_lost_weight'] = np.array(
                        [sums['sum_lost_weight']])

        lifetime_tracking = None
        lifetime_tracking_error = None
        if rate_tracking is not None:
            lifetime_tracking = (np.inf if rate_tracking == 0
                                 else float(bunch_intensity / rate_tracking))
            if rate_tracking_error is not None and rate_tracking > 0:
                lifetime_tracking_error = float(
                    lifetime_tracking * rate_tracking_error / rate_tracking)

        return RadiativeBhabhaResult(
            element_names=names,
            local_rates=xt.Table(data),
            rate_generated=rate_generated,
            rate_tracking=rate_tracking,
            lifetime_tracking=lifetime_tracking,
            tracked=tracked,
            rate_tracking_error=rate_tracking_error,
            lifetime_tracking_error=lifetime_tracking_error,
        )

    def run(self, n_turns, *, at=None, n_replicas=1, keep_particles=False,
            with_progress=False):
        """
        Generate, track and analyse the radiative Bhabha particles of the
        markers.

        Convenience helper doing in one call the steps of the simulation: the
        particles of each marker (and replica) are generated with
        :meth:`generate`, tracked from the marker back to it with
        ``line.track(part, ele_start=name, ele_stop=name, num_turns=n_turns)``
        and analysed with :meth:`analyse`. Doing these steps explicitly gives
        the same result, and allows splitting them over separate jobs or
        refining the loss locations before the analysis.

        Parameters
        ----------
        n_turns : int
            Number of turns to track.
        at : str, sequence of str, or None, optional
            Markers at which particles are generated. If ``None`` (default),
            all the markers of the line.
        n_replicas : int, optional
            Number of replicas of each marker. With more than one, the result
            includes the Monte Carlo error of the loss rate.
        keep_particles : bool, optional
            If ``True``, the result keeps the particles of each marker (its
            replicas merged), all of them merged, and the lost ones (if any).
        with_progress : bool, optional
            Forwarded to ``line.track``.

        Returns
        -------
        result : RadiativeBhabhaResult
            Result of :meth:`analyse` for the particles of each marker.
        """
        _check_positive_integer(n_turns, 'n_turns')
        _check_positive_integer(n_replicas, 'n_replicas')

        line = self.line
        names = self._markers(at)
        samples = {nn: [] for nn in names}
        for replica in range(n_replicas):
            generated = self.generate(
                at=names, replica=replica, n_replicas=n_replicas)
            for nn, part in generated.items():
                line.track(part, ele_start=nn, ele_stop=nn,
                           num_turns=n_turns, with_progress=with_progress)
                samples[nn].append(part)

        result = self.analyse(samples)
        if keep_particles:
            result.particles_by_element = {
                nn: vv[0] if len(vv) == 1 else xt.Particles.merge(vv)
                for nn, vv in samples.items()}
            result.particles = xt.Particles.merge(
                list(result.particles_by_element.values()))
            lost = _lost(result.particles)
            if np.any(lost):
                result.lost_particles = result.particles.filter(lost)
        return result

    def _markers(self, at=None):
        """
        Validated names of the selected markers, in line order.

        Parameters
        ----------
        at : str, sequence of str, or None
            Selected markers. ``None`` selects all the markers of the line.

        Returns
        -------
        names : list of str
            Names of the selected markers, in line order.
        """
        names = self.marker_names
        if at is None:
            if len(names) == 0:
                raise ValueError(
                    "The line does not contain any RadiativeBhabhaMarker. "
                    "Install them with "
                    "`line.xfields.bhabha.install_markers(...)`.")
            return names

        if isinstance(at, str):
            selected = [at]
        elif hasattr(at, '__iter__'):
            selected = list(at)
        else:
            raise TypeError(
                "Radiative Bhabha markers are selected by name "
                f"(got {type(at).__name__}).")
        if len(selected) == 0:
            raise ValueError("No radiative Bhabha marker selected.")
        position = {nn: ii for ii, nn in enumerate(names)}
        for nn in selected:
            if not isinstance(nn, str):
                raise TypeError(
                    "Radiative Bhabha markers are selected by name "
                    f"(got {type(nn).__name__}).")
            if nn in position:
                continue
            if nn in set(self.line.element_names):
                raise TypeError(
                    f"Element '{nn}' is not a RadiativeBhabhaMarker "
                    f"(got {type(self.line.element_dict[nn]).__name__}).")
            raise ValueError(f"Element '{nn}' is not present in the line.")

        return sorted(set(selected), key=position.__getitem__)
