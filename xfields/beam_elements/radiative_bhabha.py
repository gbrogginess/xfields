# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2026.                   #
# ########################################### #

import xobjects as xo
import xtrack as xt


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
