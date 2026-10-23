// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2026.                   //
// ########################################### //

#ifndef XFIELDS_RADIATIVE_BHABHA_MARKER_H
#define XFIELDS_RADIATIVE_BHABHA_MARKER_H

#include "xtrack/headers/track.h"

// Passive in tracking: the scattered particles are generated from Python
GPUFUN
void RadiativeBhabhaMarker_track_local_particle(
        RadiativeBhabhaMarkerData el, LocalParticle* part0) {
    (void)el; (void)part0;
}

#endif /* XFIELDS_RADIATIVE_BHABHA_MARKER_H */
