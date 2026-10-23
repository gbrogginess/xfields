# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import warnings

import numpy as np

import xobjects as xo
import xtrack as xt

from .beambeam3d import BeamBeamBiGaussian3D
from .temp_slicer import TempSlicer


# Number of bunch particles sent through the generator at once
_CHUNK_SIZE = 100_000

# Mean number of photons per bunch particle above which the pile-up of
# photons on the same particle biases the energy loss
_PILEUP_WARNING_THRESHOLD = 0.1


class RadiativeBhabhaMarker(xt.BeamElement):
    """
    Radiative Bhabha marker: an interaction point at which the particles
    scattered by radiative Bhabha (e+ e- -> e+ e- gamma) are generated.

    The marker is passive and transparent for tracking. It stores the
    configuration of the generation at its interaction point: the local
    optics and closed orbit, the beam parameters, the crossing geometry and
    the settings of the generator. The marker is installed with
    ``line.xfields.bhabha.install_markers(...)`` or by hand, configured with
    ``line.xfields.bhabha.configure(...)``, and the scattered particles are
    generated with ``line.xfields.bhabha.generate(...)``: see
    :class:`xfields.RadiativeBhabhaGenerator`. The whole configuration is
    stored in the fields of the marker, and is therefore saved with the line.

    The generation uses the radiative Bhabha generator of
    :class:`xfields.BeamBeamBiGaussian3D` (equivalent photons and Compton
    scattering, as in GUINEA-PIG): ``n_generated`` particles of a Gaussian
    bunch, matched to the optics at the marker, cross the opposing bunch in a
    weak-strong collision, beam-beam kick included. The opposing bunch has
    the same intensity, emittances, bunch length and optics at the
    interaction point. The particles that emitted a photon and whose
    momentum deviation from the closed orbit is below ``delta_cut`` are
    kept, each with the rate it represents as weight.

    Parameters
    ----------
    p0c : float, optional
        Reference momentum [eV].
    bunch_intensity : float, optional
        Number of particles in one bunch, of both beams.
    gemitt_x, gemitt_y : float, optional
        Geometric emittances [m].
    sigma_z : float, optional
        RMS bunch length [m].
    sigma_delta : float, optional
        RMS relative momentum spread.
    alfx, betx, alfy, bety : float, optional
        Twiss parameters at the marker.
    dx, dpx, dy, dpy : float, optional
        Dispersion and its derivative at the marker.
    x_co, px_co, y_co, py_co, zeta_co, delta_co : float, optional
        Closed orbit at the marker.
    t_rev0 : float, optional
        Revolution period [s].
    phi : float, optional
        Half crossing angle [rad].
    alpha : float, optional
        Angle of the crossing plane [rad] (0: horizontal crossing).
    other_beam_q0 : float, optional
        Charge of the particles of the opposing beam [e].
    n_slices : int, optional
        Number of longitudinal slices of the opposing bunch.
    compt_x_min : float, optional
        Low energy cut of the virtual photon spectrum, in units of
        ``gamma^-2`` (see :class:`xfields.BeamBeamBiGaussian3D`).
    compt_scale : float, optional
        Factor scaling up the emission probability of every particle. The
        mean number of photons per particle must stay well below 1.
    flag_beamsize_effect : int, optional
        Beam-size effect on the cross section: 1 on, 0 off.
    n_generated : int, optional
        Number of bunch particles sent through the interaction point at each
        generation.
    delta_cut : float, optional
        Only the particles with ``delta - delta_co < delta_cut`` after the
        emission are kept.
    seed : int, optional
        Seed from which the random-number streams of the marker are derived.

    Attributes
    ----------
    luminosity : float
        Luminosity per bunch crossing [m^-2] measured by the generator in the
        last generation.
    mean_photons_per_particle : float
        Mean number of photons emitted per bunch particle in the last
        generation.
    """

    _xofields = {
        'p0c': xo.Float64,
        'bunch_intensity': xo.Float64,
        'gemitt_x': xo.Float64,
        'gemitt_y': xo.Float64,
        'sigma_z': xo.Float64,
        'sigma_delta': xo.Float64,
        'alfx': xo.Float64,
        'betx': xo.Float64,
        'alfy': xo.Float64,
        'bety': xo.Float64,
        'dx': xo.Float64,
        'dpx': xo.Float64,
        'dy': xo.Float64,
        'dpy': xo.Float64,
        'x_co': xo.Float64,
        'px_co': xo.Float64,
        'y_co': xo.Float64,
        'py_co': xo.Float64,
        'zeta_co': xo.Float64,
        'delta_co': xo.Float64,
        't_rev0': xo.Float64,
        'phi': xo.Float64,
        'alpha': xo.Float64,
        'other_beam_q0': xo.Float64,
        'n_slices': xo.Int64,
        'compt_x_min': xo.Float64,
        'compt_scale': xo.Float64,
        'flag_beamsize_effect': xo.Int64,
        'n_generated': xo.Int64,
        'delta_cut': xo.Float64,
        'seed': xo.Int64,
    }

    behaves_like_drift = True
    allow_loss_refinement = True
    has_backtrack = True
    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include "xfields/beam_elements/radiative_bhabha_src/radiative_bhabha.h"'
    ]

    def __init__(self, p0c=0.0, bunch_intensity=0.0,
                 gemitt_x=0.0, gemitt_y=0.0, sigma_z=0.0, sigma_delta=0.0,
                 alfx=0.0, betx=0.0, alfy=0.0, bety=0.0,
                 dx=0.0, dpx=0.0, dy=0.0, dpy=0.0,
                 x_co=0.0, px_co=0.0, y_co=0.0, py_co=0.0,
                 zeta_co=0.0, delta_co=0.0,
                 t_rev0=0.0, phi=0.0, alpha=0.0, other_beam_q0=0.0,
                 n_slices=0, compt_x_min=0.0, compt_scale=0.0,
                 flag_beamsize_effect=1, n_generated=0, delta_cut=0.0,
                 seed=0, **kwargs):
        if '_xobject' in kwargs.keys():
            self.xoinitialize(**kwargs)
            return

        super().__init__(**kwargs)
        self._configure(
            p0c=p0c, bunch_intensity=bunch_intensity,
            gemitt_x=gemitt_x, gemitt_y=gemitt_y,
            sigma_z=sigma_z, sigma_delta=sigma_delta,
            alfx=alfx, betx=betx, alfy=alfy, bety=bety,
            dx=dx, dpx=dpx, dy=dy, dpy=dpy,
            x_co=x_co, px_co=px_co, y_co=y_co, py_co=py_co,
            zeta_co=zeta_co, delta_co=delta_co,
            t_rev0=t_rev0, phi=phi, alpha=alpha, other_beam_q0=other_beam_q0,
            n_slices=n_slices, compt_x_min=compt_x_min,
            compt_scale=compt_scale,
            flag_beamsize_effect=flag_beamsize_effect,
            n_generated=n_generated, delta_cut=delta_cut, seed=seed)

    def _configure(self, **kwargs):
        unknown = set(kwargs) - set(self._xofields)
        if unknown:
            raise KeyError(
                f"Unsupported configure() keys: {', '.join(sorted(unknown))}")
        for kk in ('bunch_intensity', 'n_generated', 'n_slices'):
            if kk in kwargs and kwargs[kk] < 0:
                raise ValueError(f"`{kk}` must be non-negative.")
        if 'compt_scale' in kwargs and kwargs['compt_scale'] < 0:
            raise ValueError("`compt_scale` must be non-negative.")
        for kk, vv in kwargs.items():
            setattr(self, kk, vv)

    def _beambeam_line(self, particle_ref):
        """
        Line with the weak-strong collision with the opposing bunch, with the
        radiative Bhabha generator on.
        """
        context = self._context

        # Opposing bunch: same emittances and optics at the interaction point
        sigma = {}
        for plane, gemitt, alf, bet, d, dp in (
                ('x', self.gemitt_x, self.alfx, self.betx, self.dx, self.dpx),
                ('y', self.gemitt_y, self.alfy, self.bety, self.dy, self.dpy)):
            gam = (1 + alf**2)/bet
            sigma[plane] = (
                gemitt*bet + (d*self.sigma_delta)**2,
                -gemitt*alf + d*dp*self.sigma_delta**2,
                gemitt*gam + (dp*self.sigma_delta)**2)
        n_slices = int(self.n_slices)
        slicer = TempSlicer(_context=context, n_slices=n_slices,
                            sigma_z=self.sigma_z, mode='shatilov')

        beambeam = BeamBeamBiGaussian3D(
            _context=context,
            config_for_update=None,
            other_beam_q0=self.other_beam_q0,
            phi=self.phi,
            alpha=self.alpha,
            min_sigma_diff=1e-28,
            slices_other_beam_num_particles=(
                slicer.bin_weights * self.bunch_intensity),
            slices_other_beam_zeta_center=slicer.bin_centers,
            slices_other_beam_Sigma_11=n_slices*[sigma['x'][0]],
            slices_other_beam_Sigma_12=n_slices*[sigma['x'][1]],
            slices_other_beam_Sigma_22=n_slices*[sigma['x'][2]],
            slices_other_beam_Sigma_33=n_slices*[sigma['y'][0]],
            slices_other_beam_Sigma_34=n_slices*[sigma['y'][1]],
            slices_other_beam_Sigma_44=n_slices*[sigma['y'][2]],
            compt_x_min=self.compt_x_min,
            compt_scale=self.compt_scale,
            flag_beamsize_effect=int(self.flag_beamsize_effect),
            flag_luminosity=1,
        )
        line = xt.Line(elements=[beambeam], element_names=['beambeam'])
        line.particle_ref = particle_ref.copy()
        line.build_tracker(_context=context)
        line.configure_radiation(model_bhabha='quantum')
        return line

    def _weak_bunch(self, particle_ref, num_particles, rng):
        """
        Gaussian bunch matched to the optics at the marker, on the closed
        orbit, with unit weights.
        """
        u = rng.standard_normal((6, num_particles))
        delta = self.sigma_delta * u[4]
        coords = {'zeta': self.zeta_co + self.sigma_z * u[5],
                  'delta': self.delta_co + delta}
        for plane, gemitt, alf, bet, d, dp in (
                ('x', self.gemitt_x, self.alfx, self.betx, self.dx, self.dpx),
                ('y', self.gemitt_y, self.alfy, self.bety, self.dy, self.dpy)):
            i0 = 0 if plane == 'x' else 2
            co = getattr(self, f'{plane}_co')
            pco = getattr(self, f'p{plane}_co')
            coords[plane] = co + np.sqrt(gemitt*bet) * u[i0] + d*delta
            coords[f'p{plane}'] = (pco + np.sqrt(gemitt/bet)
                                   * (u[i0 + 1] - alf*u[i0]) + dp*delta)

        bunch = xt.Particles(
            _context=self._context,
            p0c=self.p0c,
            mass0=particle_ref.mass0,
            q0=particle_ref.q0,
            pdg_id=particle_ref.pdg_id,
            **coords)
        bunch._init_random_number_generator(
            seeds=rng.integers(1, 2**31 - 1, size=num_particles))
        return bunch

    def _generate(self, *, particle_ref, at_element, s, seed,
                  weight_scale=1.0):
        """
        Generate the radiative Bhabha scattered particles at the marker.

        Parameters
        ----------
        particle_ref : xtrack.Particles
            Reference particle, defining the species of the particles.
        at_element : int
            Index of the marker in the line.
        s : float
            Position of the marker in the line [m].
        seed : int
            Seed of the random-number stream of this generation.
        weight_scale : float, optional
            Factor applied to the weights of the particles.

        Returns
        -------
        particles : xtrack.Particles
            Scattered particles, with a capacity of twice their number to
            leave room for secondaries. Their weights are the rates [1/s per
            bunch] they represent, times ``weight_scale``.
        """
        if self.n_generated < 1:
            raise ValueError(
                "`n_generated` must be at least 1 to generate particles "
                f"(got {self.n_generated}).")
        if not self.compt_scale > 0:
            raise ValueError("`compt_scale` must be positive.")

        ctx2np = self._context.nparray_from_context_array
        beambeam_line = self._beambeam_line(particle_ref)
        rng = np.random.default_rng(seed)

        coords = ('x', 'px', 'y', 'py', 'zeta', 'delta', 'state')
        kept = {cc: [] for cc in coords}
        n_done = 0
        n_photons = 0
        luminosity = 0.0
        while n_done < self.n_generated:
            n_chunk = min(_CHUNK_SIZE, self.n_generated - n_done)
            bunch = self._weak_bunch(particle_ref, n_chunk, rng)

            capacity = 2 * n_chunk
            record = beambeam_line.start_internal_logging_for_elements_of_type(
                BeamBeamBiGaussian3D,
                capacity={'beamstrahlungtable': 0, 'bhabhatable': capacity,
                          'lumitable': 1})
            beambeam_line.track(bunch)
            beambeam_line.stop_internal_logging_for_elements_of_type(
                BeamBeamBiGaussian3D)
            record.move(_context=xo.context_default)

            table = record.bhabhatable
            n_recorded = int(table._index.num_recorded)
            if n_recorded >= capacity:
                raise ValueError(
                    f"More than {capacity // n_chunk} photons per particle "
                    "on average: `compt_scale` is far too large.")
            n_photons += n_recorded
            luminosity += float(record.lumitable.luminosity[0])

            # Unit weights: every recorded photon was emitted by its particle
            emitted = np.zeros(n_chunk, dtype=bool)
            emitted[table.particle_id[:n_recorded]] = True
            particle_id = ctx2np(bunch.particle_id)
            state = ctx2np(bunch.state)
            delta = ctx2np(bunch.delta)
            # Particles that lost all their energy are kept as lost
            selected = emitted[particle_id] & (
                (state <= 0) | (delta - self.delta_co < self.delta_cut))
            for cc in coords:
                kept[cc].append(ctx2np(getattr(bunch, cc))[selected])
            n_done += n_chunk

        kept = {cc: np.concatenate(vv) for cc, vv in kept.items()}
        n_kept = len(kept['x'])

        self.luminosity = luminosity / self.n_generated * self.bunch_intensity
        self.mean_photons_per_particle = n_photons / self.n_generated
        if self.mean_photons_per_particle > _PILEUP_WARNING_THRESHOLD:
            warnings.warn(
                f"{self.mean_photons_per_particle:.3g} photons per particle "
                "on average: several photons pile up on the same particles, "
                "which biases the energy loss. Reduce `compt_scale`.",
                stacklevel=3)
        if n_kept == 0:
            raise ValueError(
                "No particle kept: increase `n_generated` or `compt_scale`, "
                "or raise `delta_cut`.")

        # Rate of the crossings of the bunch particles, per bunch, over the
        # emission probability scaled up by compt_scale
        weight = (self.bunch_intensity / self.t_rev0
                  / (self.compt_scale * self.n_generated) * weight_scale)

        particles = xt.Particles(
            _context=self._context,
            _capacity=2 * n_kept,
            p0c=self.p0c,
            mass0=particle_ref.mass0,
            q0=particle_ref.q0,
            pdg_id=particle_ref.pdg_id,
            weight=np.full(n_kept, weight),
            s=s,
            **kept)
        particles.at_element = at_element
        return particles

    def track(self, particles):
        """
        Track particles through the element without applying a kick.

        The scattered particles are generated explicitly, by
        ``line.xfields.bhabha.generate(...)``: during tracking this element is
        a passive marker.

        Parameters
        ----------
        particles : xtrack.Particles
            Particles tracked through the passive marker.

        Returns
        -------
        None
        """
        super().track(particles)
